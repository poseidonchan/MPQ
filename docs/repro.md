# Repro & gates

This repo enforces two quality gates:

## Gate A (GPTQModel, 4-bit Qwen 0.6B)
- Target: MPGQ exact-full achieves ~22.1 PPL on Wikitext2 (±0.3).
- Recipe: C4 token blocks 128×2048, seed=0, iters=10, tau=0.2, damping=0.5, polish=3, rotation=hadamard.

## Gate B (QUIP#, 2-bit Llama-3.2-1B)
- Target: `quip+mpgq` (MAP x1 + gram polish + random init) beats LDLQ baseline on Wikitext2 for seeds {0,1,2}.
- Additionally: per-layer/per-block surrogate error must satisfy `mpgq <= ldlq`.
- Latency: wall-time ≤ 5× LDLQ.

