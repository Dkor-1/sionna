# Literature-reproduction validation

**Date:** 2026-06-26 · **Code:** `validate_literature.py` · **Data:** `outputs/validate_literature.{json,png}`

## Principle
The bar is **trend (shape), sign, and rough magnitude — NOT absolute dB.** Numerology
and geometry differ from each survey paper, so absolute values are not expected to
match; reproducing the *shape* of each reported effect is what shows the engine is
faithful. Where the simulation **cannot** reproduce an effect, we say so and explain
why (the project's "paper-faithfulness" principle).

**Method:** self-contained module reusing the core primitives via a backward-compatible
`mask=` injection into `synth_ofdm`/`run_mode` (one line in the core; `pilot_mask` and
the existing experiments are untouched). Config matches the S2 benchmark (fc 3.5 GHz,
B 100 MHz, N 512, M 8192) so the reference-density trend sits above the CFAR floor.
Each experiment is run at 3 SNRs (−20/−26/−32 dB drone-echo) so trends are shown to be
operating-point-robust.

---

## Experiment 1 — LTE: CRS-only vs all-symbol (LTE-23 Table I) → **reproduced (trend)**
Anchor: sym0 = 17.5 dB, all-symbol = 24.2 dB (**+6.7 dB**), but all-symbol gives **~3×
false plots**; symbols without CRS miss the target. → "CRS-only is the operating point."

Simulated (comb-6 fixed, symbol set varied; @ −26 dB):

| variant | known REs | SCR | PSLR | Pd |
|---|--:|--:|--:|--:|
| sym0 {0} | 1.2 % | 5.8 | −4.6 | 0.00 |
| nocrs {2,3} | 2.4 % | 6.8 | −3.8 | 0.00 |
| crs {0,4,7,11} | 4.8 % | 9.4 | −1.5 | 0.33 |
| allsym {0–13} | 16.7 % | 14.7 | +3.9 | 1.00 |

- ✅ **Reproduced:** SCR rises monotonically with the CRS symbols used (sym0 < crs <
  allsym at all 3 SNRs); **Δ(all − sym0) = 8.9 dB** (lit +6.7 — same sign and ballpark).
- ❌ **NOT reproduced (honest):** all-symbol's **"3× false plots"** (raw FAR). The ideal
  pilots-known engine makes **no data self-noise/clutter**, so it cannot generate the
  extra false plots. We *attempted* to map it to PSLR (ambiguity floor) but that map
  **does not hold for LTE**: the all-symbol comb-6 is a *regular* dense reference, so it
  has the **cleanest** PSLR (+3.9), not the worst. (PSLR only degrades for a
  *time-bursty* reference — the Wi-Fi preamble's Doppler grating lobes, which S2 does
  catch.) So the energy gain is real and reproduced; the false-plot **cost** of
  all-symbol is a genuine limit of the ideal model — it needs a data-driven clutter/ECA
  model (future work).
- ❌ **NOT reproduced (modeling):** "no-CRS symbols {2,3} miss." In our model the mask
  *defines* the pilots, so a comb on {2,3} is a **valid** reference and detects. The
  real-LTE miss is because {2,3} carry no CRS (an *empty* reference) — our controlled
  grid does not impose that.

## Experiment 2 — 5G: occupancy → SCR (5G-22 Fig.10) → **reproduced**
Anchor: ~10 % occupancy ≈ invisible; ~70 % ≈ +24 dB.

Simulated (graded occupancy, density f swept):

| occupancy | SCR @ −20 / −26 / −32 dB | Pd @ −26 / −32 |
|---|--:|--:|
| 5 % | 15.6 / 9.7 / 5.2 | 0.25 / 0.00 |
| 10 % | 18.4 / 12.3 / 6.8 | 0.92 / 0.08 |
| 30 % | 22.9 / 16.6 / 9.8 | 1.00 / 0.50 |
| 50 % | 25.4 / 19.2 / 13.0 | 1.00 / 0.92 |
| 70 % | 27.0 / 21.0 / 15.1 | 1.00 / 1.00 |

- ✅ **Reproduced:** SCR rises **monotonically** with occupancy at all 3 SNRs; the
  detection transition matches the paper's shape — at −26 dB, 5 % is near-invisible
  (Pd 0.25) while 70 % is strong (Pd 1.0); SCR span 11.3 dB across the range.
- Note: the absolute "+24 dB" is geometry/CPI dependent; we reproduce the **shape**
  (monotone SCR; weak→strong over the occupancy range), not the exact dB.

## Experiment 3 — Wi-Fi: preamble-only vs full reference (Wi-Fi-24) → **reproduced**
Anchor: preamble-only ≈ **−1…−11 dB** SNR vs full (but data-independent).

- ✅ **Reproduced:** SCR(full) − SCR(preamble-only) is **positive at every SNR**, median
  **7.2 dB** (inside the 1–11 dB band) — the cost of not knowing the data, right sign and
  rough size.

---

## Summary
| Effect | Status |
|---|---|
| LTE: SCR ↑ with CRS symbols used (energy) | ✅ reproduced (Δ 8.9 dB vs lit +6.7) |
| LTE: all-symbol "3× false plots" (FAR) | ❌ not reproduced (needs data-clutter model; PSLR map invalid for regular comb) |
| LTE: no-CRS symbols miss | ❌ not reproduced (degenerate — mask defines pilots) |
| 5G: occupancy → SCR (invisible → strong) | ✅ reproduced (monotone; 5%→Pd0.25, 70%→Pd1.0 @ −26) |
| Wi-Fi: preamble-only cost vs full | ✅ reproduced (+7.2 dB median, lit 1–11) |

**Bottom line:** the engine reproduces the *reference-structure* trends (more/denser
deterministic reference → higher SCR/Pd) across LTE, 5G, and Wi-Fi anchors. The effects
it cannot reproduce are exactly those requiring a **data-driven clutter/false-alarm
model** (all-symbol false plots) or an **empty-reference** condition (no-CRS miss) that
the ideal pilots-known upper-bound does not impose — both already on the roadmap
(data-driven ECA-S).
