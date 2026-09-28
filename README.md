# Linear Remix: Dynamic Mixtures of Low-Rank Linear Maps Replace the FFN

A research campaign on a minimal transformer-adjacent architecture, run as a
series of controlled ablation experiments on a single consumer GPU (RTX 3060
6GB). Starting from a failed attention-replacement experiment, the work
converged on a simple finding:

> **A soft mixture of 16 low-rank linear maps — with a per-token learned
> router — plus per-head linear remixes of attention values, beats a
> same-depth transformer (5.381 vs 5.409 tokCE) at 92% of its parameters,
> with no FFN anywhere in the network.**

## The idea

A standard transformer block mixes channels through an FFN: two *large*
nonlinear transformations. A **linear remix** replaces this with *small*
linear ones:

```
M(x) = W_down · W_up · LN(x)        # rank-r, purely linear, ~15% of FFN params
```

One such map is just a compression trick. The interesting question is what
happens with **many, mixed dynamically**:

```
M(x) = Σᵢ gᵢ(x) · W_down,i · W_up,i · LN(x)     g(x) = softmax(router(x))
```

This is a soft mixture-of-experts over low-rank linear maps. If the gates go
uniform, it collapses back to a single static rank-(k·r) map — so any win over
that control is attributable to *dynamic composition*, not capacity.

The second placement is inside attention: each head's **post-softmax** values
pass through their own low-rank remix before the output projection:

```
y = A·V ;  y = y + (y @ W_up_h) @ W_down_h        # per head, zero-init
```

Both remixes are **zero-initialized**, so every model starts as the exact
reference architecture and earns its deviations during training.

## Where the remix is allowed to live

The campaign's central mechanistic rule, established by controlled nulls:

> A linear map only adds value when it sits **adjacent to a nonlinearity**
> (LayerNorm, attention softmax). Linear ∘ linear folds into a single linear
> map — bolted next to other linear maps or next to an existing FFN, a remix
> does nothing.

- Bolt-on remix beside a healthy FFN: **no effect** (5.757 vs 5.774 baseline)
- Same remix replacing the FFN: most of the FFN's function at ~15% of params
- Remix *after attention's softmax*: the one fold-free site in attention — works
- GLU sidecar inside the dynamic mixture: **hurts** (nonlinearity doesn't stack
  with it); GLU sidecar inside the *static* remix: helps

## Results

All runs: d=256, 6 layers, 4 heads, GPT-2 BPE, seq 128, batch 16, constant LR
3e-4 after 100-step warmup, early stop on 3 consecutive validation increases.
Validation tokCE (lower is better); references share a seed and data pipeline.

| architecture | params | best val | s/step | Δ vs transformer |
|---|---|---|---|---|
| transformer (ref) | 17.87M | 5.409 | 0.067 | — |
| static linear remix r=128 | 15.11M | 5.602 | 0.093 | +0.193 |
| MoE 4×r32 (param-matched to static) | 15.11M | 5.458 | 0.102 | +0.049 |
| MoE 8×r32 (soft, no FFN) | 15.51M | 5.421 | 0.165 | +0.012 |
| MoE top-2 of 8 (hard routing) | 15.51M | 5.527 | 0.267 | +0.118 |
| head-remix attention + FFN | 17.96M | 5.407 | 0.058 | −0.003 |
| **stack: head-remix + MoE 8×r32** | 15.61M | 5.396 | 0.130 | −0.014 |
| **stack, k=16 (final champion)** | **16.50M** | **5.381** | 0.107 | **−0.029** |

Key controls and what they killed:

- **Param-matched MoE 4×32 vs static r=128** (both 65,536 expert params/block):
  MoE wins by 0.14 with specialized, non-collapsed gates (entropy 1.0–1.6 of
  max 2.08, distinct expert profile per block) → the active ingredient is
  *dynamic selection*, not capacity.
- **Hard top-2 routing** at identical capacity loses 0.106 to soft routing and
  is *slower* per step on this GPU (scatter vs dense batched GEMMs) →
  discretized selection is dominated at this scale.
- **Count vs rank scaling** at matched budget (+0.9M): k=8→16 buys 0.021;
  r=32→64 buys 0.006 → the architecture consumes *diversity*, not map size.
- **MLP router vs linear router**: no significant difference — routing policy
  expressiveness was never the bottleneck.

Qualitative: the champion's samples are coherent corpus-style prose
(Lovecraft pastiche); the static remix's text visibly fragments. A 3-epoch
chat checkpoint already produces readable multi-clause continuations.

## Files

- **`train_compare.py`** — the full experiment harness: all arms (transformer
  reference, static remix, soft/hard MoE, head-remix attention, hybrid,
  stacks, scaling variants), identical training protocol per arm, gate-entropy
  probes, per-arm samples, results merged into `results.json`. Add an arm by
  appending to the `arms` list; completed arms are skipped on re-run.
- **`train_best.py`** — single-model trainer for the final champion
  configuration. `--epochs N` = N full passes over the training split
  (auto-sized from the input file at launch), validates and prints a live
  sample every `--predict-every` steps, keeps the best-validation checkpoint
  in `ckpt.pt`. Launching it bare with a checkpoint on disk enters an
  interactive **chat mode** (story-completion, temperature 0.8 / top-k 40).

Data: the scripts read the training corpus via the `--input` CLI flag, which
defaults to the repo-local `input.txt` (H. P. Lovecraft's works); pass `--input <path>` to train on a different
file. 90/10 train/val split, GPT-2 tokenizer.

## Honest caveats

- Single seed, single corpus, single scale (d=256/6L, ~3.9 MB of text). The
  −0.03 final margin needs multi-seed error bars before it's a paper claim.
- Gate "specialization" is measured as entropy spread, not interpreted —
  nobody knows what individual experts encode.
- The champion's win is demonstrated at one scale; d=512+ crossover is
  untested.
- Wall-clock parity with the transformer is not achieved: the best
  quality-speed point is the head-remix attention model (−0.003 at 0.058
  s/step), not the full champion.

## Reproducing

```bash
pip install -r requirements.txt
python train_compare.py                 # full ablation grid (~6 GPU-hours on an RTX 3060)
python train_compare.py --smoke         # 2-min sanity check
python train_best.py --epochs 30        # train the champion (~20 min)
python train_best.py                    # chat with it
python train_best.py --input path/to/other_corpus.txt --epochs 3   # train on a different corpus
```
