# Train the BEST configuration found by the LinearRemix campaign (E1_D2_wide):
#   per-head rank-64 linear remix of post-softmax attention values (zero-init)
#   + soft MoE of k=16 rank-32 linear remix experts (zero-init router) as the
#   FFN replacement. Beats the same-depth transformer: 5.381 vs 5.409 tokCE
#   at 92% of the params and faster wall-clock.
#
# Simple single-model training (no comparison arms, no early stop):
#   - configurable number of epochs (--epochs, fractional allowed)
#   - validate + generate a sample every --predict-every steps (default 1000)
#   - best-val checkpoint saved to ckpt.pt
#
# CHAT MODE: if a checkpoint file already exists at launch, the script enters
# an interactive chat with the trained model instead of training (unless
# --force-retrain is given).
#
# Usage:
#   python train_best.py --epochs 3                    # train 3 epochs
#   python train_best.py                               # resume / chat
#   python train_best.py --chat                        # force chat mode
#   python train_best.py --force-retrain               # discard ckpt, retrain
#   python train_best.py --smoke                       # tiny sanity run

import os, sys, math, time, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")
CKPT_SMOKE = os.path.join(HERE, "ckpt_smoke.pt")
SEED = 1337
D_MODEL = 256
LAYERS = 6
HEADS = 4
HEAD_REMIX_RANK = 64   # E1 setting
N_EXPERTS = 16         # E1 setting
EXPERT_RANK = 32       # E1 setting
SEQ = 128
BATCH = 16
LR = 3e-4
WD = 0.01
CLIP = 1.0
WARMUP = 100
EVAL_BATCHES = 24
GEN_TOKENS = 400
GEN_TEMP = 0.8
GEN_TOPK = 40
PROMPT_CHARS = 96


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- model (E1 champion) ---------------------------

class CausalAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(0.1)
        self.register_buffer("mask", torch.triu(torch.ones(1024, 1024, dtype=torch.bool), 1))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, D))


class HeadRemixAttention(nn.Module):
    """Attention whose per-head post-softmax values pass through a low-rank
    linear remix before the output projection. Zero-init => identity at start.
    Post-softmax placement means the remix cannot fold into V's projection."""
    def __init__(self, d, heads, rank):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(0.1)
        self.register_buffer("mask", torch.triu(torch.ones(1024, 1024, dtype=torch.bool), 1))
        self.h_up = nn.Parameter(torch.zeros(heads, self.dk, rank))
        self.h_down = nn.Parameter(torch.zeros(heads, rank, self.dk))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                   # [B, h, T, dk]
        y = y + (y @ self.h_up) @ self.h_down         # per-head remix, identity at init
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MoELinearRemix(nn.Module):
    """Soft mixture of k low-rank linear remixes (E1: k=16, rank=32)."""
    def __init__(self, d, rank, k):
        super().__init__()
        self.k = k
        self.ln = nn.LayerNorm(d)
        self.ups = nn.ModuleList([nn.Linear(d, rank, bias=False) for _ in range(k)])
        self.downs = nn.ModuleList([nn.Linear(rank, d, bias=False) for _ in range(k)])
        self.router = nn.Linear(d, k, bias=False)
        nn.init.zeros_(self.router.weight)

    def forward(self, x):
        h = self.ln(x)
        gates = F.softmax(self.router(h), dim=-1)
        outs = [down(up(h)) for up, down in zip(self.ups, self.downs)]
        mixed = sum(g.unsqueeze(-1) * o for g, o in zip(gates.unbind(-1), outs))
        return x + mixed


class E1Block(nn.Module):
    """The champion: head-remix attention + soft MoE of linear remixes (no FFN)."""
    def __init__(self, d, heads, head_rank, k, expert_rank):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = HeadRemixAttention(d, heads, head_rank)
        self.remix = MoELinearRemix(d, expert_rank, k)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return self.remix(x)


class LM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Embedding(1024, D_MODEL)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([
            E1Block(D_MODEL, HEADS, HEAD_REMIX_RANK, N_EXPERTS, EXPERT_RANK)
            for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# ----------------------------- data ------------------------------------------

def build_data(input_file):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    text = open(input_file, encoding="utf-8").read()
    ids = enc.encode(text)
    split = int(len(ids) * 0.90)
    return (enc,
            torch.tensor(ids[:split], dtype=torch.long),
            torch.tensor(ids[split:], dtype=torch.long))


def get_batches(tokens, batch_size, seq_len):
    max_idx = len(tokens) - seq_len - 1
    all_idx = torch.arange(0, max_idx + 1, seq_len)
    n = len(all_idx) // batch_size
    all_idx = all_idx[: n * batch_size]
    offs = torch.arange(seq_len)
    for i in range(0, len(all_idx), batch_size):
        starts = all_idx[i : i + batch_size].unsqueeze(1)
        x = starts + offs
        y = starts + offs + 1
        yield tokens[x], tokens[y]


@torch.no_grad()
def evaluate(model, val_tokens, device):
    model.eval()
    tot, n = 0.0, 0
    for bi, (x, y) in enumerate(get_batches(val_tokens, BATCH, SEQ)):
        if bi >= EVAL_BATCHES:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                             y.reshape(-1), reduction="sum")
        tot += ce.item()
        n += y.numel()
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def generate(model, enc, prompt_ids, n_tokens, device):
    model.eval()
    ids = list(prompt_ids)
    for _ in range(n_tokens):
        ctx = torch.tensor([ids[-SEQ:]], dtype=torch.long, device=device)
        with torch.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            logits = model(ctx)
        logits = logits[0, -1].float() / GEN_TEMP
        topv, topi = torch.topk(logits, GEN_TOPK)
        probs = F.softmax(topv, dim=-1)
        ids.append(topi[torch.multinomial(probs, 1)].item())
    model.train()
    return enc.decode(ids[len(prompt_ids):])


def save_ckpt(model, step, val, path=None):
    torch.save({"model": model.state_dict(), "step": step, "val": val,
                "config": dict(d=D_MODEL, layers=LAYERS, heads=HEADS,
                               head_rank=HEAD_REMIX_RANK, k=N_EXPERTS,
                               expert_rank=EXPERT_RANK, seq=SEQ)},
               path or CKPT)


def load_ckpt(model, device, path=None):
    ck = torch.load(path or CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return ck


# ----------------------------- training --------------------------------------

def train(args):
    set_seed(SEED)
    device = get_device()
    print(f"device={device}")
    enc, tr, va = build_data(args.input)
    print(f"corpus: {len(tr):,} train / {len(va):,} val tokens")

    model = LM(enc.n_vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"E1 champion config: d={D_MODEL} layers={LAYERS} heads={HEADS} "
          f"head_rank={HEAD_REMIX_RANK} k={N_EXPERTS} expert_rank={EXPERT_RANK} "
          f"-> {n_params:,} params")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ONE EPOCH = ONE FULL PASS over the training data (the 90% split of
    # args.input). Computed from the file every launch, so swapping in a bigger
    # input.txt automatically rescales epochs:
    #   windows in the non-overlapping grid = floor((N - SEQ - 1) / SEQ) + 1
    #   steps per epoch = windows // BATCH   (exactly what get_batches yields)
    max_idx = len(tr) - SEQ - 1
    n_windows = max_idx // SEQ + 1
    steps_per_epoch = n_windows // BATCH         # steps in one full data pass
    total_steps = int(round(steps_per_epoch * args.epochs))
    print(f"one epoch = full pass over train split: {n_windows:,} windows / "
          f"batch {BATCH} = {steps_per_epoch} steps; "
          f"total: {total_steps:,} steps ({args.epochs} epochs)")

    start_step = 0
    best_val = float("inf")
    ckpt_path = CKPT_SMOKE if args.smoke else CKPT
    if os.path.exists(ckpt_path) and not args.force_retrain:
        ck = load_ckpt(model, device, ckpt_path)
        start_step = ck["step"]
        best_val = ck.get("val", float("inf"))
        print(f"resuming from {ckpt_path} (step {start_step}, val {best_val:.4f})")

    prompt_ids = va[:PROMPT_CHARS].tolist()
    prompt = enc.decode(prompt_ids)
    step = start_step
    t0 = time.time()
    done = False
    while not done:
        for x, y in get_batches(tr, BATCH, SEQ):
            step += 1
            if step > total_steps:
                done = True
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            lr = LR * min(1.0, step / WARMUP)
            for g in opt.param_groups:
                g["lr"] = lr
            scaler.step(opt)
            scaler.update()

            if step % args.predict_every == 0 or step == total_steps:
                v = evaluate(model, va, device)
                star = ""
                if v < best_val:
                    best_val = v
                    save_ckpt(model, step, v, ckpt_path)
                    star = " *saved"
                sample = generate(model, enc, prompt_ids, args.sample_tokens, device)
                print(f"step {step:6d}  val {v:.4f}{star}  lr {lr:.2e}  "
                      f"({time.time()-t0:.0f}s)\n  sample: {sample[:220]}...", flush=True)
    print(f"\nDONE. best val {best_val:.4f}, ckpt -> {ckpt_path}")


# ----------------------------- chat mode --------------------------------------

def chat():
    set_seed(SEED)
    device = get_device()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model = LM(enc.n_vocab).to(device)
    ck = load_ckpt(model, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"E1 champion loaded from {CKPT}")
    print(f"  step {ck['step']}  val {ck['val']:.4f}  {n_params:,} params")
    print("chat mode -- type a prompt (empty line quits), temperature/top-k fixed "
          f"at {GEN_TEMP}/{GEN_TOPK}. The model is a story-completion model, not "
          "an assistant: it continues your text in the style of its corpus.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        ids = enc.encode(user)
        out = generate(model, enc, ids, GEN_TOKENS, device)
        print(f"model> {user}{out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--predict-every", type=int, default=1000,
                    help="validate + sample every N steps")
    ap.add_argument("--sample-tokens", type=int, default=200)
    ap.add_argument("--chat", action="store_true", help="force chat mode")
    ap.add_argument("--force-retrain", action="store_true",
                    help="ignore existing checkpoint, start fresh")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--input", default=os.path.join(HERE, "input.txt"),
                    help="path to the training corpus text file")
    args = ap.parse_args()

    if args.smoke:
        global SEQ, BATCH, GEN_TOKENS
        SEQ, BATCH = 32, 2
        args.predict_every = 4
        args.sample_tokens = 20
        args.epochs = 0.001

    if args.chat:
        if not os.path.exists(CKPT):
            sys.exit(f"no checkpoint at {CKPT} -- train first (no --chat)")
        chat()
        return

    # chat-on-launch ONLY for a bare `python train_best.py` (no training args).
    # Any explicit training request (--epochs/--smoke/--force-retrain) trains.
    training_requested = (args.force_retrain or args.smoke
                          or args.epochs != 1.0
                          or args.predict_every != 1000)
    if os.path.exists(CKPT) and not training_requested:
        print(f"checkpoint found at {CKPT} -- entering CHAT MODE "
              f"(use --force-retrain to train again)")
        chat()
        return

    train(args)


if __name__ == "__main__":
    main()
