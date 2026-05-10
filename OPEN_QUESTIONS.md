# Gemma4 — Open Questions

Priority order: P0 = blocking production confidence / reproducibility, P1 = performance headroom, P2 = future capability.

---

## P0 — Dense-31B 64K decode collapse (1.78 tok/s, AL=1.94) — RESOLVED

**Resolved post-rebase.** See EVIDENCE.md §1.6 — 2.54 tok/s sustained at 64K on Dense 31B TQ3/TQ3 (`dense-tq3-frontier/D_64k_tq3_none.log`). Root cause was the TQ3 FWHT mismatch (Bug 2); `daef232a6` + `4b0c158` fixed it.

<details>
<summary>Original investigation notes (archived)</summary>

**Observed**: `dense_code_65536.log` — 50003-token code prompt, 31B dense + dflash + Q8/Q8, ctx=65536, dm=8. Decode 1.78 tok/s, avg_accept=1.94/8 (24%). VRAM 24.00/24.00 GB (saturated).

**Confound**: at 128K and 256K, VRAM is also 24.00/24.00 GB (fully saturated) and decode is ~24 tok/s — but output is a repetition loop (token 715 repeated 8/8 at every step from step 5 onward). AL=7.11 at those sizes is trivial drafting of a degenerate sequence, not real acceptance. The 64K run produces genuinely diverse token IDs and real code output.

**Hypotheses to test**:
1. KV cache layout for dense 31B at 65536 ctx: `full_attn=65536, swa=2048` (per log: `[cache] created max_ctx=65536 (full_attn=65536, swa=2048), kv_layers=60, saved 80.7%`). At 131072: `full_attn=131072, swa=2048`, saving 82.0%. At 262144: 82.7%. The SWA ring stays constant at 2048; the full-attn allocation doubles each step. Does 65536 full-attn × 10 layers × Q8 + drafter 2096 slots push total allocations into a different fragmentation regime?
2. CUDA VMM page table behaviour at exactly 65536 slots per layer: 65536 × 10 full-attn layers × 2 (K+V) × 8 heads × 256 head_dim × 1 byte = 2.68 GB — this lands near the boundary of a CUDA VMM granule or allocation arena.
3. First decode step takes 1138.59 ms (vs ~328 ms at 128K). The 3.5× TTFT penalty at 64K is a strong signal that the first target-forward at decode time is the bottleneck, not subsequent steps. Could be cache-miss on the freshly-allocated 65536-slot full-attn KV (cold cache pages).
4. Rule out output-mode artefact: run the same config with a different prompt at 64K to check whether 1.78 tok/s is prompt-specific or context-size-specific.

**Next steps**: instrument CUDA timers on the first few decode steps at 64K vs 128K; check if TTFT disparity narrows after step 5; profile cache-line misses with `nsys`.
</details>

---

## P1 — Drafter context window cap

The 5-layer dflash drafter has a 2096-slot KV cache. On a 50k-token prompt, it skips the first 47907 tokens and materialises only the last 2096 positions (per all sweep logs: `[draft] KV prefill done: 2096 positions materialized (skipped 47907 early tokens, cap=2096)`). Larger drafter caches (5k, 10k slots) might recover meaningful AL at long context, especially on prompts where the relevant speculative context is in the middle of a large file. No public ablation exists.

**Candidate experiment**: sweep drafter KV cap at {2096, 4096, 8192} on the 50k code prompt + 64K ctx (where AL is currently ~1.94 on dense and ~1.79 on MoE), measuring AL and peak VRAM.

---

## P1 — Decode-time KV sparsity

Bandwidth model (journey blog §"What still hurts"): RTX 3090 at 936 GB/s, reading 50k-token Q8 KV per step ≈ 6.1 GB → theoretical ceiling 152 tok/s. Actual: 36.57 tok/s ≈ 24% of ceiling. The remaining gap is structural: weights + MoE FFN + drafter forward + speculative verification. None of H2O / StreamingLLM / Quest / Landmark Attention / QuantSpec is wired into any production speculative-decoding stack. This is the next research-to-production opportunity. Estimate: 2–4× decode improvement possible with ~40% KV eviction on long contexts (H2O figures from the paper).

---

## P1 — TQ3_0 SWA-wrap branch re-introducing FWHT mismatch

When the SWA ring wraps (sustained generation past ~1024 tokens past the SWA window), the wrap branch in `gemma4_mtp_graph.cpp` (post-`4b0c158`: `ggml_turbo_wht` calls at lines 442 and 518) concat-forces F32, stripping the FWHT rotation. Same class as Bug 2 (MMA intercept stripping FWHT); same fix pattern (force chunked, or split FA + combine softmax). Not hit in any current bench session because generations stayed within the unwrapped window (max 256 output tokens; SWA window = 2048). Will manifest on long chat sessions. Tracked in the journey blog.

---

## P1 — MoE MTP drafter does not exist

MoE 26B-A4B has no MTP assistant. Training one (4-layer assistant against MoE 26B target activations on HumanEval-class tasks) would unlock higher accept rates on the smaller model at short context. Until then, MoE relies on dflash only. MTP on dense 31B at 4K hits 0.87 accept; dflash on dense hits 0.64. If MoE MTP could match, expected tok/s at 4K: 60–80 (vs ~112 with dflash at dm=4).

---

## P2 — head_dim=512 + Q8 + non-aligned KV without gqa-opt dependency

Current Bug-3 fix (commit f1f811e) routes around the issue by always providing a mask when `head_dim_fa >= 512`. A cleaner kernel that handles the non-aligned head_dim=512 + Q8 case directly (without requiring gqa-opt or an explicit mask) would remove the mask-allocation overhead on every decode step for all 10 full-attn layers. Low urgency — overhead is small — but the current workaround is technically debt.

---

## P2 — MTP accept rate at 64K+

MTP at 64K shows avg_accept = 0.02 (5/256). All correctness bugs are fixed (Bugs 1–3). The most likely cause is that MTP's `h_prev` capture at the last committed token provides insufficient predictive signal when the committed-token context is 50k tokens deep into an attention-truncated SWA window. MTP was designed for autoregressive generation where h_prev is information-dense; at long context after chunked prefill the signal may be diluted. Approaches: (a) provide h_prev from a later capture layer, (b) train a longer-range MTP head, (c) accept that MTP is short-context only and document accordingly.

---

## P2 — Q8/Q8 vs TQ3/TQ3 decision boundary by context length — RESOLVED

**Resolved.** TQ3 frontier sweep 4K→1M complete (`tq3-frontier/F_*.log`). TQ3/TQ3 is the unambiguous choice for ≥64K contexts; Q8/Q8 wins under 64K. At 4K: Q8/Q8 62.83 tok/s vs TQ3/TQ3 6.95 tok/s (~9× gap on decode); at 64K+ TQ3 unlocks contexts Q8 cannot reach on 24 GB at all. The crossover is purely VRAM-driven: Q8 saturates at ~256K, TQ3 reaches 1M at 22.26 GB peak.

---

## P2 — Prompt distribution documentation for dflash drafter

The dflash drafter was trained on HumanEval-class code tasks. AL on code: ~5.12/8 (64%). AL on creative writing: ~2.13/8 (27%). A 2.4× gap means tok/s claims are meaningless without specifying the prompt distribution. Need a one-page distribution doc alongside any published benchmark number. Also: what is the drafter's training data cutoff, and does it generalise to non-Python languages?

---

## Resolved Questions

| Question | Resolution | Date |
|---|---|---|
| P0 Dense-31B 64K decode collapse (1.78 tok/s) | Fixed by `daef232a6` + `4b0c158` (TQ3 FWHT mismatch). Now 2.54 tok/s at 64K. See EVIDENCE.md §1.6. | 2026-05-10 |
| P2 Q8/Q8 vs TQ3/TQ3 decision boundary | TQ3/TQ3 wins at ≥64K (VRAM-gated). Q8/Q8 wins below 64K (decode speed). Full sweep `tq3-frontier/F_*.log`. | 2026-05-10 |
