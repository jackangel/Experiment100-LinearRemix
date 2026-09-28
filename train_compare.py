# Linear Remix vs Regular Transformer
# Compares validation loss trajectories and generation coherence of:
#   (1) a standard transformer block:  LN -> attention -> + ; LN -> FFN(4x) -> +
#   (2) the FFN-less linear-remix block: LN -> attention -> + ; LN -> Linear(d,r) -> Linear(r,d) -> +
# Training protocol (identical for both):
#   - evaluate every EVAL_EVERY steps on the same validation windows
#   - early stop when 3 CONSECUTIVE evaluations show an increase in val loss
#   - keep the best-val checkpoint, generate samples from it for coherence comparison
# Results: results.json + samples_<arm>.txt in this directory.
#
# Usage:
#   python train_compare.py            # full run
#   python train_compare.py --smoke    # tiny sanity run, writes *_smoke files

import os, sys, json, math, time, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1337
D_MODEL = 256
LAYERS = 6
HEADS = 4
REMIX_RANK = 128
SEQ = 128
BATCH = 16
LR = 3e-4
WD = 0.01
CLIP = 1.0
WARMUP = 100
EVAL_EVERY = 100
EVAL_BATCHES = 24          # cap validation windows (constant eval cost)
PATIENCE = 3               # stop after 3 consecutive eval losses increase
MIN_STEPS = 400            # no early stop before this many steps
MAX_STEPS = 6000           # hard cap
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


# ----------------------------- data -----------------------------------------

def build_data(input_file):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    text = open(input_file, encoding="utf-8").read()
    ids = enc.encode(text)
    split = int(len(ids) * 0.90)
    tr = torch.tensor(ids[:split], dtype=torch.long)
    va = torch.tensor(ids[split:], dtype=torch.long)
    return enc, tr, va


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


# ----------------------------- models ---------------------------------------

class CausalAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.register_buffer(
            "mask", torch.triu(torch.ones(1024, 1024, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.size(1)
        a, _ = self.attn(x, x, x, attn_mask=self.mask[:T, :T], need_weights=False)
        return a


class FeedForward(nn.Module):
    def __init__(self, d, mult=4.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, int(d * mult)), nn.GELU(),
                                 nn.Linear(int(d * mult), d), nn.Dropout(0.1))

    def forward(self, x):
        return self.net(x)


class LinearRemix(nn.Module):
    """The round-4/6 finding: LN -> rank-r linear map -> + residual. No nonlinearity."""
    def __init__(self, d, rank):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.up = nn.Linear(d, rank, bias=False)
        self.down = nn.Linear(rank, d, bias=False)

    def forward(self, x):
        h = self.down(self.up(self.ln(x)))
        return x + h


class MoELinearRemix(nn.Module):
    """Mixture of k low-rank linear remixes with a soft router:
        M(x) = sum_i g_i(x) * W_down,i @ W_up,i @ LN(x),  g = softmax(x @ W_r)
    Param math: k experts at rank r have k*2*d*r expert params (vs 2*d*r for one
    remix of rank r) -- so (k=4, r=32) matches one rank-128 remix exactly.
    Soft-mixed sum of k rank-r maps is a rank-<=k*r map with INPUT-DEPENDENT
    weights; if gates go uniform it collapses to a static rank-k*r remix.
    Probes logged: mean gate per expert + gate entropy (usage / collapse check).
    """
    def __init__(self, d, rank, k, router_mlp=False):
        super().__init__()
        self.k = k
        self.ln = nn.LayerNorm(d)
        self.ups = nn.ModuleList([nn.Linear(d, rank, bias=False) for _ in range(k)])
        self.downs = nn.ModuleList([nn.Linear(rank, d, bias=False) for _ in range(k)])
        if router_mlp:
            # 2-layer routing policy (E4): d -> d/4 -> k with GELU
            self.router = nn.Sequential(nn.Linear(d, d // 4), nn.GELU(),
                                        nn.Linear(d // 4, k))
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        else:
            self.router = nn.Linear(d, k, bias=False)
            nn.init.zeros_(self.router.weight)  # uniform gates at init (collapse point)

    def forward(self, x):
        h = self.ln(x)
        gates = F.softmax(self.router(h), dim=-1)  # [B, T, k]
        outs = [down(up(h)) for up, down in zip(self.ups, self.downs)]
        mixed = sum(g.unsqueeze(-1) * o for g, o in zip(gates.unbind(-1), outs))
        # probes (last batch only, cheap)
        with torch.no_grad():
            gm = gates.mean(dim=(0, 1))
            self.last_gates = [round(v, 4) for v in gm.tolist()]
            p = gates.clamp_min(1e-9)
            self.last_entropy = (-(p * p.log()).sum(-1)).mean().item()
            self.last_entropy_max = math.log(self.k)
        return x + mixed


class TopKMoELinearRemix(nn.Module):
    """Hard top-k routing over k low-rank linear remix experts.
    Per token: pick the k_active experts with the highest router logit, run only
    those, renormalize their softmax gates.  Load-balance aux loss keeps experts
    alive:  L_aux = k * sum_i (f_i * P_i)   (Switch-Transformer style), where
    f_i = fraction of tokens whose top-1 is expert i (stop-grad through f),
    P_i = mean router prob of expert i.  Added to the main loss with weight
    aux_weight (0.01, Switch default).
    """
    def __init__(self, d, rank, k, k_active=2, aux_weight=0.01):
        super().__init__()
        self.k = k
        self.k_active = k_active
        self.aux_weight = aux_weight
        self.ln = nn.LayerNorm(d)
        self.ups = nn.ModuleList([nn.Linear(d, rank, bias=False) for _ in range(k)])
        self.downs = nn.ModuleList([nn.Linear(rank, d, bias=False) for _ in range(k)])
        self.router = nn.Linear(d, k, bias=False)
        nn.init.zeros_(self.router.weight)
        self.last_aux = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln(x)
        logits = self.router(h)                       # [B, T, k]
        gates_full = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(logits, self.k_active, dim=-1)   # [B, T, ka]
        topg = F.softmax(topv, dim=-1)                # renormalized among the chosen
        # run only the chosen experts: gather per-token expert outputs
        outs = torch.zeros(B, T, D, device=x.device, dtype=h.dtype)
        flat_g = topg.reshape(-1, self.k_active)      # [N, ka]
        flat_i = topi.reshape(-1, self.k_active)      # [N, ka]
        hf = h.reshape(-1, D)
        of = outs.reshape(-1, D)
        for e in range(self.k):
            mask = (flat_i == e)                      # [N, ka]
            if not mask.any():
                continue
            w = (flat_g * mask).sum(-1, keepdim=True) # gate weight for expert e
            sel = w.squeeze(-1) > 0
            if not sel.any():
                continue
            y = self.downs[e](self.ups[e](hf[sel]))
            of[sel] += y * w[sel]
        # load-balance aux (Switch-style): encourage uniform expert usage
        with torch.no_grad():
            f = torch.zeros(self.k, device=x.device)
            f.scatter_add_(0, flat_i[:, 0], torch.ones(flat_i.size(0), device=x.device))
            f = f / f.sum()
        P = gates_full.mean(dim=(0, 1))
        aux = self.k * (f * P).sum()
        self.last_aux = aux.detach()
        with torch.no_grad():
            gm = P
            self.last_gates = [round(v, 4) for v in gm.tolist()]
            p = gates_full.clamp_min(1e-9)
            self.last_entropy = (-(p * p.log()).sum(-1)).mean().item()
            self.last_entropy_max = math.log(self.k)
            # fraction of expert-pairs actually exercised
            used = torch.zeros(self.k, device=x.device)
            for e in range(self.k):
                used[e] = (flat_i == e).any(dim=-1).float().mean()
            self.last_usage = [round(v, 4) for v in used.tolist()]
        return x + outs, aux


class TransformerBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalAttention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffwd = FeedForward(d, 4.0)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


# ---- Experiment batch C: remix inside attention + nonlinear hybrid ----

class HeadRemixAttention(nn.Module):
    """Causal attention whose per-head outputs pass through a low-rank linear
    remix BEFORE the output projection.  Sits right after attention's softmax
    (A @ V) -- the one place a linear map cannot fold into adjacent projections,
    because the softmax separates it from V's projection.  Zero-init up
    projection => exact identity at init.
    """
    def __init__(self, d, heads, rank):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(0.1)
        self.register_buffer(
            "mask", torch.triu(torch.ones(1024, 1024, dtype=torch.bool), 1))
        # per-head remix: rank-r, zero-init up => identity at start
        self.h_up = nn.Parameter(torch.zeros(heads, self.dk, rank))
        self.h_down = nn.Parameter(torch.zeros(heads, rank, self.dk))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # [B, h, T, dk]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                    # [B, h, T, dk]  (post-softmax!)
        # per-head low-rank remix of the retrieved values (zero-init => identity)
        y = y + (y @ self.h_up) @ self.h_down
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class HeadRemixBlock(nn.Module):
    """Transformer block: FFN removed, channel mixing moved INTO the heads."""
    def __init__(self, d, heads, rank):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = HeadRemixAttention(d, heads, rank)
        self.ln2 = nn.LayerNorm(d)
        self.ffwd = FeedForward(d, 4.0)   # keep FFN: this arm tests the ATTENTION remix, not deletion

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class HybridRemix(nn.Module):
    """C2 hybrid: linear remix (rank 128) + tiny NONLINEAR GLU sidecar (rank 16).
    Hypothesis: the linear remix saturates because it cannot construct new
    features; a small nonlinear branch should recover the transformer's
    asymptote.  Target: transformer's 5.409 (vs pure remix 5.602)."""
    def __init__(self, d, rank, glu_rank):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.up = nn.Linear(d, rank, bias=False)
        self.down = nn.Linear(rank, d, bias=False)
        # nonlinear sidecar: gate || value split (T GLU at width glu_rank)
        self.g_up = nn.Linear(d, glu_rank, bias=False)
        self.v_up = nn.Linear(d, glu_rank, bias=False)
        self.side_down = nn.Linear(glu_rank, d, bias=False)

    def forward(self, x):
        h = self.ln(x)
        linear = self.down(self.up(h))
        side = self.side_down(F.silu(self.g_up(h)) * self.v_up(h))
        return x + linear + side


class HybridRemixBlock(nn.Module):
    def __init__(self, d, heads, rank, glu_rank):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalAttention(d, heads)
        self.remix = HybridRemix(d, rank, glu_rank)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return self.remix(x)


class HeadRemixHybridBlock(nn.Module):
    """THE STACK: head-remix attention (C1 winner) + hybrid remix block (C2/C3 winner)."""
    def __init__(self, d, heads, rank, glu_rank):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = HeadRemixAttention(d, heads, rank)
        self.remix = HybridRemix(d, rank, glu_rank)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return self.remix(x)


class HeadRemixMoEBlock(nn.Module):
    """Stack variant: head-remix attention + soft MoE-of-remixes (moe_8x32 winner)
    instead of the single hybrid remix."""
    def __init__(self, d, heads, rank, k, expert_rank, router_mlp=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = HeadRemixAttention(d, heads, rank)
        self.remix = MoELinearRemix(d, expert_rank, k, router_mlp=router_mlp)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return self.remix(x)


class MoEHybridRemix(nn.Module):
    """E3: soft MoE of linear remix experts + a small always-on GLU sidecar.
    All three winners in one slot: dynamic selection (moe), plus a pinch of
    nonlinearity (C2/C3 lesson) to attack the saturation asymptote."""
    def __init__(self, d, expert_rank, k, glu_rank, router_mlp=False):
        super().__init__()
        self.k = k
        self.ln = nn.LayerNorm(d)
        self.ups = nn.ModuleList([nn.Linear(d, expert_rank, bias=False) for _ in range(k)])
        self.downs = nn.ModuleList([nn.Linear(expert_rank, d, bias=False) for _ in range(k)])
        if router_mlp:
            self.router = nn.Sequential(nn.Linear(d, d // 4), nn.GELU(),
                                        nn.Linear(d // 4, k))
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        else:
            self.router = nn.Linear(d, k, bias=False)
            nn.init.zeros_(self.router.weight)
        self.g_up = nn.Linear(d, glu_rank, bias=False)
        self.v_up = nn.Linear(d, glu_rank, bias=False)
        self.side_down = nn.Linear(glu_rank, d, bias=False)

    def forward(self, x):
        h = self.ln(x)
        gates = F.softmax(self.router(h), dim=-1)
        outs = [down(up(h)) for up, down in zip(self.ups, self.downs)]
        mixed = sum(g.unsqueeze(-1) * o for g, o in zip(gates.unbind(-1), outs))
        side = self.side_down(F.silu(self.g_up(h)) * self.v_up(h))
        with torch.no_grad():
            gm = gates.mean(dim=(0, 1))
            self.last_gates = [round(v, 4) for v in gm.tolist()]
            p = gates.clamp_min(1e-9)
            self.last_entropy = (-(p * p.log()).sum(-1)).mean().item()
            self.last_entropy_max = math.log(self.k)
        return x + mixed + side


class HeadRemixMoEHybridBlock(nn.Module):
    def __init__(self, d, heads, rank, k, expert_rank, glu_rank):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = HeadRemixAttention(d, heads, rank)
        self.remix = MoEHybridRemix(d, expert_rank, k, glu_rank)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return self.remix(x)


class RemixBlock(nn.Module):
    def __init__(self, d, heads, rank):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalAttention(d, heads)
        self.remix = LinearRemix(d, rank)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return self.remix(x)


class MoERemixBlock(nn.Module):
    def __init__(self, d, heads, rank, k):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalAttention(d, heads)
        self.remix = MoELinearRemix(d, rank, k)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return self.remix(x)


class TopKMoERemixBlock(nn.Module):
    """Returns (hidden, aux_loss) — block-level aux pass-through."""
    def __init__(self, d, heads, rank, k, k_active=2, aux_weight=0.01):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalAttention(d, heads)
        self.remix = TopKMoELinearRemix(d, rank, k, k_active, aux_weight)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        out, aux = self.remix(x)
        return out, aux


class LM(nn.Module):
    def __init__(self, vocab, block_cls, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Embedding(1024, D_MODEL)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([block_cls(D_MODEL, HEADS, **kw) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        aux_total = None
        for b in self.blocks:
            out = b(x)
            if isinstance(out, tuple):
                x, aux = out
                aux_total = aux if aux_total is None else aux_total + aux
            else:
                x = out
        logits = self.head(self.ln_f(x))
        self.last_aux = aux_total
        return logits

    def moe_probes(self):
        out = []
        for i, b in enumerate(self.blocks):
            r = getattr(b, "remix", None)
            if r is not None and hasattr(r, "last_gates"):
                p = dict(block=i, gates=r.last_gates,
                         entropy=round(r.last_entropy, 3),
                         entropy_max=round(r.last_entropy_max, 3))
                if hasattr(r, "last_usage"):
                    p["usage"] = r.last_usage
                if hasattr(r, "k_active"):
                    p["k_active"] = r.k_active
                out.append(p)
        return out


# ----------------------------- eval / sample --------------------------------

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
        logits = logits[0, -1].float()
        logits = logits / GEN_TEMP
        topv, topi = torch.topk(logits, GEN_TOPK)
        probs = F.softmax(topv, dim=-1)
        nxt = topi[torch.multinomial(probs, 1)].item()
        ids.append(nxt)
    model.train()
    return enc.decode(ids[len(prompt_ids):])


# ----------------------------- training -------------------------------------

def train_arm(name, model, tr, va, device, out_dir, enc, prompt_ids, smoke=False,
              emb_lr_mult=1.0, lr_schedule="constant"):
    max_steps = 8 if smoke else MAX_STEPS
    eval_every = 4 if smoke else EVAL_EVERY
    min_steps = 0 if smoke else MIN_STEPS
    seq = 32 if smoke else SEQ
    batch = 2 if smoke else BATCH
    horizon = 4000  # cosine horizon (fixed regardless of early stop)

    if emb_lr_mult != 1.0:
        # NOTE: emb and head are TIED (same tensor) -- one group, not two
        emb_ids = {id(model.emb.weight)}
        decay = [p for p in model.parameters() if p.dim() >= 2 and id(p) not in emb_ids]
        nodecay = [p for p in model.parameters() if p.dim() < 2 and id(p) not in emb_ids]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": WD},
             {"params": nodecay, "weight_decay": 0.0},
             {"params": [model.emb.weight], "weight_decay": 0.0, "lr": LR * emb_lr_mult}],
            lr=LR, betas=(0.9, 0.95))
    else:
        decay = [p for p in model.parameters() if p.dim() >= 2]
        nodecay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": WD},
             {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def lr_at(step):
        base = LR * min(1.0, step / WARMUP)
        if lr_schedule == "cosine":
            if step >= WARMUP:
                p = min(1.0, (step - WARMUP) / max(1, horizon - WARMUP))
                base = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))
        return base

    model.train()
    step = 0
    val_hist = []          # (step, val_loss)
    best_val, best_step = float("inf"), -1
    best_state = None
    stop_reason = "max_steps"
    t0 = time.time()
    while step < max_steps:
        for x, y in get_batches(tr, batch, seq):
            step += 1
            if step > max_steps:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1))
                # MoE load-balance aux (summed over blocks), scaled into fp32-safe range
                aux = getattr(model, "last_aux", None)
                if aux is not None:
                    loss = loss + aux.to(loss.dtype) * 0.01
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            # schedule: linear warmup, then constant or cosine (same rule both arms)
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            scaler.step(opt)
            scaler.update()

            if step % eval_every == 0:
                v = evaluate(model, va, device)
                val_hist.append({"step": step, "val": v})
                star = ""
                if v < best_val:
                    best_val, best_step = v, step
                    best_state = {k: t.detach().cpu().clone()
                                  for k, t in model.state_dict().items()}
                    star = " *"
                print(f"  [{name}] step {step:5d}  val {v:.4f}{star}", flush=True)
                # early stopping: 3 consecutive increases
                if step >= min_steps and len(val_hist) >= 4:
                    last4 = [h["val"] for h in val_hist[-4:]]
                    if last4[1] > last4[0] and last4[2] > last4[1] and last4[3] > last4[2]:
                        stop_reason = f"3 consecutive val increases (from step {val_hist[-4]['step']})"
                        break
        if stop_reason != "max_steps":
            break

    dt = time.time() - t0
    # regenerate from BEST checkpoint for coherence comparison
    if best_state is not None:
        model.load_state_dict(best_state)
    samples = [generate(model, enc, prompt_ids, 20 if smoke else GEN_TOKENS, device)
               for _ in range(3)]
    with open(os.path.join(out_dir, f"samples_{name}.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {name} — best val {best_val:.4f} @ step {best_step}\n")
        f.write(f"# prompt: {PROMPT!r}\n\n")
        for i, s in enumerate(samples):
            f.write(f"--- sample {i+1} ---\n{PROMPT}{s}\n\n")

    n_params = sum(p.numel() for p in model.parameters())
    probes = model.moe_probes() if hasattr(model, "moe_probes") else []
    if probes:
        print(f"  [{name}] MoE gate probes (mean gates per block, entropy/max):")
        for p in probes:
            print(f"    block {p['block']}: gates={p['gates']}  H={p['entropy']}/{p['entropy_max']}")
    return dict(
        run=name, params=n_params, best_val=round(best_val, 4),
        best_step=best_step, stop_step=step, stop_reason=stop_reason,
        emb_lr_mult=emb_lr_mult, lr_schedule=lr_schedule,
        moe_probes=probes if probes else None,
        val_history=val_hist, sec_per_step=round(dt / max(step, 1), 4),
        train_seconds=round(dt, 1),
    )


PROMPT = None  # filled in main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--input", default=os.path.join(HERE, "input.txt"),
                    help="path to the training corpus text file")
    args = ap.parse_args()
    global PROMPT

    tag = "_smoke" if args.smoke else ""
    out_dir = HERE
    results_path = os.path.join(out_dir, f"results{tag}.json")
    set_seed(SEED)
    device = get_device()
    print(f"device={device}  input={args.input}")
    enc, tr, va = build_data(args.input)
    # prompt = start of the validation split (both models see the same context)
    PROMPT = enc.decode(va[:PROMPT_CHARS].tolist())
    print(f"corpus: {len(tr):,} train tokens / {len(va):,} val tokens")
    print(f"prompt for generation:\n---\n{PROMPT}\n---")
    prompt_ids = va[:PROMPT_CHARS].tolist()

    arms = [
        # ---- batch E: push D2 to new heights ----
        # D2 reference: 5.3955 @4700 (15.61M, 0.130 s/step). Transformer: 5.4094.
        ("transformer", lambda: LM(enc.n_vocab, TransformerBlock),
         dict(emb_lr_mult=1.0, lr_schedule="constant")),
        # E1: WIDEN the MoE block: k=16 experts (2x D2) + head remix rank 64 (2x).
        # Question: does D2's edge GROW with capacity, or did we already harvest it?
        ("E1_D2_wide", lambda: LM(enc.n_vocab, HeadRemixMoEBlock,
                                  rank=64, k=16, expert_rank=32),
         dict(emb_lr_mult=1.0, lr_schedule="constant")),
        # E2: spend the same budget on rank instead of count: expert rank 64, k=8
        # (E1 vs E2 reads the k-vs-r axis at matched params)
        ("E2_D2_rank64", lambda: LM(enc.n_vocab, HeadRemixMoEBlock,
                                    rank=64, k=8, expert_rank=64),
         dict(emb_lr_mult=1.0, lr_schedule="constant")),
        # E3: D2 + hybrid sidecar (GLU-16) inside the MoE block: stack all three
        # winners (head remix + dynamic selection + a pinch of nonlinearity).
        ("E3_D2_plus_glu", lambda: LM(enc.n_vocab, HeadRemixMoEHybridBlock,
                                      rank=32, k=8, expert_rank=32, glu_rank=16),
         dict(emb_lr_mult=1.0, lr_schedule="constant")),
        # E4: tune the ONE hyperparameter never swept: router temperature/depth --
        # 2-layer router with GELU (more expressive routing policy)
        ("E4_D2_router2", lambda: LM(enc.n_vocab, HeadRemixMoEBlock,
                                     rank=32, k=8, expert_rank=32, router_mlp=True),
         dict(emb_lr_mult=1.0, lr_schedule="constant")),
    ]

    # merge with existing results (references) if present
    results = []
    results_path = os.path.join(out_dir, f"results{tag}.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        done = {r["run"] for r in results}
        arms = [a for a in arms if a[0] not in done]
        print("already done, skipping:", done - {a[0] for a in arms} if arms else done)
    for name, build, tkw in arms:
        print(f"=== TRAIN {name} ===", flush=True)
        set_seed(SEED)
        model = build().to(device)
        res = train_arm(name, model, tr, va, device, out_dir, enc, prompt_ids,
                        smoke=args.smoke, **tkw)
        results.append(res)
        print(f"=== {name}: best {res['best_val']} @ step {res['best_step']}, "
              f"stopped: {res['stop_reason']} ===\n", flush=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=1)

    # summary
    print("\n================ SUMMARY ================")
    print(f"{'arm':<16}{'params':>10}{'best val':>10}{'@step':>7}{'stop@':>7}{'s/step':>8}  recipe")
    for r in results:
        recipe = f"elrx{r['emb_lr_mult']:.0f} {r['lr_schedule']}"
        print(f"{r['run']:<16}{r['params']:>10,}{r['best_val']:>10.4f}"
              f"{r['best_step']:>7}{r['stop_step']:>7}{r['sec_per_step']:>8.3f}  {recipe}")
    if len(results) >= 2:
        base = next(r for r in results if r["run"] == "transformer")
        print("\ndeltas vs transformer (negative = remix better):")
        for r in results:
            if r["run"] != "transformer":
                print(f"  {r['run']:<16} {r['best_val'] - base['best_val']:+.4f}")
        print("\nsamples saved to samples_*.txt — read them for the coherence comparison")
    print(f"results -> {results_path}")


if __name__ == "__main__":
    main()
