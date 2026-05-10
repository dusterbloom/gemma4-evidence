# Gemma4 — Engineering Evidence

> Speculative-decoding (DFlash + MTP) on Gemma-4 31B Dense and 26B-A4B MoE for llama.cpp on a 24 GB RTX 3090. Energy-instrumented bench, 13 documented engineering fixes, and the path from a 0.22 contaminated baseline to a 132 tok/s ship config.

**Live site:** [https://dusterbloom.github.io/gemma4-evidence/](https://dusterbloom.github.io/gemma4-evidence/)

## What's here

| File | What it is |
|---|---|
| `index.html` | Interactive evidence site — summary, four SVG charts, 24-cell scientific sweep, 13 discoveries, timeline, open questions |
| `EVIDENCE.md` | Same content as the site, in plain markdown |
| `OPEN_QUESTIONS.md` | P0/P1/P2 follow-ups with hypotheses and proposed experiments |
| `journey.md` | Long-form debugging-journey blog: the three-bug sequence, byte-fallback contamination, dm-sweep, ship config |
| `scientific/results.csv` | 24-cell instrumented bench: 2 archs × 2 distributions × 6 dm budgets. Wall time, prefill/decode tok/s, AL, VRAM peak, average power, total/prefill/decode energy (J), decode J/tok |
| `scientific/SUMMARY.md` | Same data formatted as a markdown table |
| `scientific/power/*.csv` | Per-cell GPU power traces (nvidia-smi @ ~5 Hz) used to integrate energy |
| `scientific/timestamps.csv` | Per-cell phase boundary timestamps for energy apportionment |

## TL;DR — what worked

- **MoE 26B-A4B + DFlash + dm=16 + Q8/Q8 + pflash** → **132.1 tok/s** decode on code (4 K ctx), AL 5.22, 11.82 J/tok, 18.9 GB VRAM
- **MoE 26B-A4B + DFlash + dm=2 + Q8/Q8 + pflash** → **6.64 J/tok** at 82.1 tok/s on creative — most-efficient real-spec config
- **MoE 26B-A4B + DFlash + dm=4 + Q8/Q8** at long context → **29–37 tok/s decode sustained 16K → 262K**
- **Dense 31B + DFlash + dm=16** competes on creative → **82 tok/s, AL 5.12** because the dense drafter generalises better OOD than the MoE drafter

## TL;DR — what didn't work

- **MTP at 64K+** → ≤ 2 % accept after all 5 mechanical fixes; correctness restored but not yield
- **`fattn.cu:652` MTP+Q8 V-cache** → three crashes at different steps with different accept rates; latent kernel issue
- **Dense 31B at exactly 64K** → 1.78 tok/s, AL 1.94 (vs ~24 tok/s + AL 7.11 at 128K/256K). VRAM-allocator paging at 24/24 GB cap with 50 K prompt filling ~78 % of cache. Open puzzle.

## TL;DR — what we discovered (non-public)

13 distinct fixes/discoveries not present in upstream llama.cpp/ggml. Highlights:

1. **SWA ring geometry** — mask must map *abs* query → *ring slot* `(qpos − abs_win_start) % ring_size`; without this, chunks 2+ silently read stale KV
2. **TQ3_0 KV alignment** — `FATTN_KQ_STRIDE = 256` is hardcoded in CUDA but exposed via no public ggml API; KV must be 256-aligned
3. **TQ3 K dequant intercept** — MMA path silently strips FWHT rotation; force chunked-path dispatch for TQ3 K (submodule commit `d758ed9bf`)
4. **FA mask mandatory for `head_dim ≥ 512`** — CUDA MMA dispatcher requires `gqa_opt_applies` ⇒ both `K->ne[1]%256 == 0` AND `mask != nullptr`
5. **Narrow asymmetric KV** — pFlash block-sparse path doesn't support TQ3_0; force Q8_0 on captured full-attn layers, TQ3_0 stays on SWA
6. **GQA block-broadcast for MTP** — explicit reshape required when `n_head_norm` ≠ `head_dim_kv`
7. **MTP h_prev capture + KQ scale = 1.0** — not the usual `1/√d`; MTP uses assistant's own `rope_freqs.weight`
8. **Dense DFlash drafter generalises OOD; MoE drafter is code-distribution-bound** — empirical, from the 24-cell sweep

Full table with file:line refs in `EVIDENCE.md` §3 or on the site.

## Reproducing

The bench harness is `run_scientific.sh` in the parent project. Profile cells with:

```bash
# pseudocode — see live llama.cpp DFlash branch for actual flags
./test_gemma4_dflash <args> --ignore-eos --temp 0 --seed 0 \
    -ctk q8_0 -ctv q8_0 --pflash --draft-max <dm> \
    --prompt-file <prompt> --n-predict 256
# while logging: nvidia-smi --query-gpu=power.draw --format=csv -lms 200
```

Hardware: RTX 3090 24 GB, CUDA 13.1, 399 W TDP. Active-window peak ~395 W (~99 % TDP) on dense+code; MoE peaks lower (~210 W).

## Related work

- Branch where the engineering lives: [`dusterbloom/lucebox-hub#feature/gemma4-support`](https://github.com/dusterbloom/lucebox-hub/tree/feature/gemma4-support)
- Underlying llama.cpp fork: [`dusterbloom/llama.cpp#feature/tq3-kv-cache`](https://github.com/dusterbloom/llama.cpp)
- Open PR: [Luce-Org/lucebox-hub#131](https://github.com/Luce-Org/lucebox-hub/pull/131)

## License

Apache-2.0 — see `LICENSE`. The data and write-up are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contributing

Issues + PRs welcome. Especially:

- Reproductions on different hardware (5090, A6000, MI300X)
- Cross-context-length sweeps with the same energy harness
- A trained MoE MTP drafter (none exists today)
- Diagnosis of the dense-31B 64K decode collapse (P0 in `OPEN_QUESTIONS.md`)
