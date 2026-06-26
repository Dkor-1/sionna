I'll synthesize the three band specs into a concrete, code-ready spec for your 4-mode benchmark. This is a pure synthesis task — all the source material is in the prompt. Let me write the complete specification directly.

# Code-Ready Spec: 4-Mode FAIR Sionna-RT Passive-Radar Drone-Detection Benchmark

## 0. Core design decision (read first)

The benchmark holds the **OFDM resource grid identical** across all 4 modes and varies **only the binary pilot mask** `M[k, l] ∈ {0,1}` over the grid (subcarrier index `k`, OFDM-symbol index `l`). The reference the radar correlates against is **reconstructed from KNOWN REs only**: REs where `M=1` carry deterministic, replicable symbols; REs where `M=0` carry unknown random data the radar cannot reproduce.

This isolates the project hypothesis: **known-RE fraction (reference density) → CAF processing gain → SCR → Pd**. Every band's literature converges on the same finding — sparse-but-deterministic references give *cleaner* CAF (lower FAR) but *lower* integration gain, and the optimum is "use all the deterministic structure you have." See LTE-23/25b (CRS-only 17.5 dB / 14 plots vs all-symbol 24.2 dB / 62 plots), 5G-26 (use ALL non-empty REs: 2.6 → 20.2 m/s), 5G-22 (10% occupancy undetectable, 70% → 24 dB), WiFi-23b (constant-modulus preamble beats dense fluctuating data).

---

## A. The 4 Pilot-Mask Patterns

### Fixed grid numerology (shared by all modes — see Section B)

Choose **one** numerology for the whole benchmark. Recommended (5G-NR FR1, μ=1):

| Param | Symbol | Value | Source |
|---|---|---|---|
| Total bandwidth | `B` | 20 MHz (or 38.16 MHz) | LTE-22/23, 5G-22/23a |
| Subcarrier spacing | `Δf` | 30 kHz (μ=1) | 5G-26 |
| FFT size | `N_FFT` | 1024 (B=20 MHz: 666 used SCs ≈ 56 RB) | derived |
| Used subcarriers | `N_sc` | 600 (50 RB) — keep round for combs | LTE 10 MHz analog |
| OFDM symbols / slot | — | 14 | 5G μ=1 |
| Slots / 10 ms frame | — | 20 | 5G μ=1 |
| Sample rate | `fs` | = `B` (critical: fs ≡ B for all modes) | — |
| CP fraction | `η` | normal CP (5G) or 0.25 (WiFi analog) | WiFi numerology |

All 4 masks live on the **same** `N_sc × N_sym` grid where `N_sym = CPI / T_ofdm`. With CPI = 100 ms and ~35.7 µs/symbol (incl. CP), `N_sym ≈ 2800` symbols. Use the **same N_sym for all modes.**

> **The mask is the ONLY thing that differs between modes.** Implement each as a function returning a boolean array `M[N_sc, N_sym]`.

---

### (M1) `wifi_preamble` — preamble-rich, dense-in-freq / sparse-in-time

**Standards basis:** 802.11 legacy preamble = L-STF + L-LTF = 16 µs = 320 samples @20 MHz, spanning **all 64 subcarriers** but only a **small time fraction** of each packet; WiFi duty factor `Fu ≈ 18%` (bursty illumination). Whole-packet references dominate, but preamble-only (320 samples) costs ~5–11 dB integration gain yet is multipath-robust (WiFi-24, WiFi-23b).

**Mask rule:** dense in frequency (all used SCs known), sparse and periodic in time (preamble "bursts").

```
# Map 802.11 to the common grid:
#   - A "packet" = one burst period of length T_pkt symbols.
#   - The preamble occupies the FIRST n_pre symbols of each packet, on ALL subcarriers.
#   - n_pre / T_pkt ≈ WiFi duty/preamble fraction.
PKT_PERIOD = 22        # OFDM symbols per packet (≈ preamble(4)+SIGNAL(1)+payload), tunable
N_PRE      = 4         # symbols of known preamble (L-STF + L-LTF + SIGNAL ≈ 4–5)
M1[k, l] = 1  if  (l mod PKT_PERIOD) < N_PRE   for ALL k in used SCs
         = 0  otherwise
```

**Occupancy:** `N_PRE / PKT_PERIOD = 4/22 ≈ 18.2%` known-RE fraction (matches WiFi `Fu≈18%` and the "4/20 OFDM symbols" packet figure). Dense-in-freq (100% of SCs during preamble), sparse-in-time.

**Symbol content on known REs:** constant-modulus (BPSK/QPSK preamble bodies) — per WiFi-23b this drives the background toward the `2·DNR` floor rather than `DNR²`. Use unit-modulus known symbols.

---

### (M2) `lte_crs` — sparse comb pilots (CRS-like)

**Standards basis (most precisely specified band):** CRS on OFDM **symbols 0 and 4** of each slot (2 of 7 per slot, normal CP), every **6th subcarrier** per antenna port (freq step = 3·Δf), 5 data SCs between pilots. Single-port density ≈ `2/(7·6) ≈ 4.8%`; 2-port interleaved (3-SC shift) ≈ `~9.5%`. QPSK, repeats every 10 ms frame (LTE-23, LTE-25b, LTE-20).

**Mask rule (use single-port 4.8% as the canonical sparse comb; offer 2-port as variant):**

```
# Grid uses 14-symbol slots (5G numerology). Map LTE "symbols 0 & 4 of a 7-sym slot"
# onto our 14-sym slot as TWO pilot-symbols per 7-symbol half-slot → symbols {0,4,7,11}.
PILOT_SYMS_PER_SLOT = {0, 4, 7, 11}   # 4 of 14  (LTE 2/7 scaled to 14)
COMB           = 6                      # every 6th subcarrier
COMB_OFFSET    = 0                      # port-0 offset; port-1 would be +3
M2[k, l] = 1  if  (l mod 14) in PILOT_SYMS_PER_SLOT  AND  (k mod COMB) == COMB_OFFSET
         = 0  otherwise
```

**Occupancy:** `(4/14) × (1/6) ≈ 4.76%` — the canonical LTE single-port CRS density.
**Variant `lte_crs_2port`:** add a second comb at `COMB_OFFSET=3` on the same symbols → `~9.5%`.

**Symbol content:** QPSK pseudo-random but **deterministic** (seeded by PCI). Optionally power-boost known REs +6 dB (LTE-20) — but if you do, you must compensate total TX power (see E, pitfall #3).

---

### (M3) `5g_ssb_sparse` — localized block, very sparse occupancy

**Standards basis:** SSB = PSS(127 SC) + SSS(127 SC) + PBCH+DMRS(240 SC) = **4 OFDM symbols × 240 SC = 960 RE**, a single localized block (240 SC = 20 RB, ≤7.2 MHz of the band), repeating once per SS-burst period (5–160 ms; ~20 ms typical → ~50 Hz). Sparsity is dominated by the **long repetition period** and **localized frequency footprint** (5G-25a, 5G-25b, 5G-26). 5G-22: SSB-only is "too sparse / low-PRF" → the explicit lower bound of the density axis.

**Mask rule:** a contiguous 4-symbol × 240-SC block, repeated with a long period; zero everywhere else.

```
SSB_NSC      = 240                       # contiguous subcarriers (centered or offset)
SSB_NSYM     = 4                         # contiguous OFDM symbols
SSB_K0       = (N_sc - SSB_NSC)//2        # frequency start (localized, e.g. centered)
SSB_PERIOD   = 560                        # OFDM symbols ≈ 20 ms at our T_ofdm (tunable)
SSB_SYM0     = 2                          # symbol offset within each period
M3[k, l] = 1 if  (SSB_K0 <= k < SSB_K0+SSB_NSC)
              AND ((l mod SSB_PERIOD) - SSB_SYM0) in [0, SSB_NSYM)
         = 0 otherwise
```

**Occupancy:** per period `= (4 × 240) / (N_sym_period × N_sc) = 960/(560×600) ≈ 0.286%`. Over the full CPI this is the **sparsest** mode (the intended floor). If too sparse to detect at all (as 5G-22 found), that is the *correct, hypothesis-confirming* result — but pick `SSB_PERIOD` so at least a few blocks fall in the CPI for a meaningful CAF.

**Symbol content:** PSS/SSS are constant-modulus, fully deterministic (Zadoff-Chu / m-sequences) — model as unit-modulus known symbols. Per 5G-25b, intra-block sparsity (even nullified SSB symbols) has minimal effect; the limiter is the block's *spatial/temporal* sparsity, which is exactly what this mask encodes.

---

### (M4) `5g_dmrs_prs_rich` — reference-rich, dense pilots

**Standards basis:** DMRS + CSI-RS + PRS-inspired dense staggered pilots — the "use ALL deterministic structure" upper bound. PRS = dense diagonal/staggered RE pattern (high overhead); 5G-26's central result is that exploiting *all* occupied REs densifies effective sampling (unambiguous velocity 2.6 → 20.2 m/s). This mode is the deterministic-rich analog: a dense staggered comb across many symbols.

**Mask rule:** staggered (diagonal-shifting) comb so the *union over symbols* covers most SCs, with several pilot-bearing symbols per slot (PRS-like density).

```
DMRS_COMB      = 2          # comb-2 (PRS/DMRS typical): every 2nd SC per pilot symbol
PILOT_SYMS     = {2,3,5,6,8,9,11,12}   # dense set: 8 of 14 symbols carry pilots (PRS-rich)
# staggered offset so consecutive pilot symbols cover the complementary comb:
def offset(l): return (l) % DMRS_COMB     # 0,1,0,1,... → full-band coverage over 2 symbols
M4[k, l] = 1 if  (l mod 14) in PILOT_SYMS  AND  (k mod DMRS_COMB) == offset(l)
         = 0 otherwise
```

**Occupancy:** `(8/14) × (1/2) ≈ 28.6%` known REs — the **densest** mode. Tune `PILOT_SYMS`/`DMRS_COMB` to hit a target (e.g. comb-1 on 8 symbols → ~57%; the 5G-26 "~70% → 24 dB SNR" regime).

**Ordering of densities (the hypothesis axis):**
`M3 (~0.29%) < M2 (~4.8%) < M1 (~18%) < M4 (~29%)`
Sparser mask → lower CAF processing gain + higher data-self-interference → lower SCR/Pd. This monotone ordering is the experiment.

---

### Summary table

| Mode | Pattern | SCs | Symbols | Period | %occupancy | Citation anchor |
|---|---|---|---|---|---|---|
| M1 wifi_preamble | all-SC burst, sparse-in-time | all used | first 4 of 22 | 22 sym/pkt | **~18%** | WiFi-24, WiFi-23b |
| M2 lte_crs | comb-6, symbols {0,4,7,11} | every 6th | 4 of 14 | 14 (10 ms frame) | **~4.8%** | LTE-23, LTE-25b, LTE-20 |
| M3 5g_ssb_sparse | 240-SC × 4-sym block | 240 contig | 4 of 560 | ~20 ms | **~0.29%** | 5G-25a/b, 5G-26, 5G-22 |
| M4 5g_dmrs_prs_rich | staggered comb-2, 8 sym | every 2nd (staggered) | 8 of 14 | 14 | **~29%** | 5G-26, 5G-25a |

---

## B. Control Protocol — Fairness Fixes

### HELD IDENTICAL across all 4 modes (must assert in code)

| Quantity | Constraint | Why |
|---|---|---|
| Total bandwidth `B` | identical (e.g. 20 MHz) | range resolution `c/2B` must not vary |
| Sample rate `fs` | `fs ≡ B` for all modes | CAF delay-bin scaling fixed |
| FFT size `N_FFT` | identical | grid geometry fixed |
| Subcarrier spacing `Δf` | identical | Doppler-bin scaling fixed |
| Used subcarrier count `N_sc` | identical | occupied bandwidth fixed |
| CP fraction / symbol duration `T_ofdm` | identical | `N_sym` per CPI fixed |
| **CPI / integration time** | identical (e.g. 100 ms) | integration gain ceiling fixed |
| `N_sym` (symbols per CPI) | identical | — |
| **Total TX power** `P_tx` | identical per CPI (see E#3) | SNR at Rx must be mode-independent |
| Geometry (Tx, Rx, target positions, bistatic angle) | identical | bistatic range/Doppler fixed |
| Trajectory (drone path, velocity) | identical | Doppler signature fixed |
| Channel (Sionna-RT scene, multipath, clutter) | identical realization per Monte-Carlo trial | clutter fixed |
| Noise PSD `N0` | identical | noise floor fixed |
| RCS model | identical | echo strength fixed |
| Clutter/DPI canceller | identical algorithm + params | cancellation residual fixed |
| CFAR detector + guard/training cells + `Pfa` | identical | detection threshold fixed |

### THE SINGLE VARIABLE THAT CHANGES

> **Only the pilot mask `M[k,l]`** (which REs are KNOWN deterministic reference vs unknown random data). Everything downstream (reference reconstruction → CAF → cancellation → CFAR) is byte-identical code; only `M` is swapped.

**Implementation contract:**
```python
def run_mode(mask: np.ndarray, scene, seed) -> Metrics:
    grid = make_grid(N_sc, N_sym, seed)        # SAME random data for all modes given seed
    grid_known = grid * mask                    # reference = KNOWN REs only
    surv = sionna_rt_propagate(grid, scene)     # full TX signal illuminates scene
    ref  = ofdm_modulate(grid_known)            # radar knows ONLY masked REs
    surv = clutter_cancel(surv, ref)            # identical canceller
    rd   = caf(surv, ref)                        # range-Doppler map
    return cfar_metrics(rd)
```
Note: `grid` (the *transmitted* data) is identical across modes for a given seed; the **full grid is propagated and carries the same total power**; the reference uses only `grid*mask`. The unknown-data REs become **self-interference** in the CAF — this is the mechanism the experiment measures.

---

## C. Metric Definitions (Monte-Carlo over noise + data realizations)

Let the range-Doppler (RD) / CAF surface be `χ[r, ν]` (range bin `r`, Doppler bin `ν`). Let `(r_t, ν_t)` be the true target bin (from known geometry/trajectory). Run `N_MC` trials varying noise and unknown-data realizations (and optionally clutter).

### C.1 Signal-to-Clutter(+noise) Ratio — SCR

```
SCR_dB = 10·log10( |χ[r_t, ν_t]|² / P_bg )
```
where `P_bg` = mean power of the background (noise + residual clutter + data-self-interference) over a region **excluding** the target cell and its guard cells:
```
P_bg = mean_{(r,ν) ∈ Ω} |χ[r,ν]|²,   Ω = all bins \ (target ∪ guard)
```
Report mean and std of `SCR_dB` over `N_MC`. This is the primary hypothesis metric (LTE-23: 17.5 vs 24.2 dB; 5G-22: 24 dB @70%).

### C.2 Detection rate — Pd

Per trial, declare detection if a CFAR detection (Section D / below) falls within ±1 bin of `(r_t, ν_t)`:
```
Pd = (# trials with a CFAR hit at target) / N_MC
```
Report `Pd` vs known-RE fraction (the headline curve), at fixed `Pfa` (use `1e-4` and `1e-6`, per 5G-22 / LTE-23 / WiFi-24).

### C.3 False-alarm rate — Pfa (empirical) and FAR

Use **CA-CFAR** (cell-averaging) with guard + training cells. Set the threshold multiplier `α` from the desired `Pfa`:
```
α = N_train · (Pfa^(-1/N_train) - 1)          # CA-CFAR, exponential (square-law) statistics
threshold[r,ν] = α · (1/N_train) · Σ_{training cells} |χ|²
detection if |χ[r,ν]|² > threshold[r,ν]
```
Empirical false-alarm rate (validate the design `Pfa`):
```
FAR_emp = (# CFAR detections in Ω, excluding target region, over all trials)
          / (N_MC · |Ω|)
```
Report `FAR_emp` to confirm it tracks the design `Pfa` (fairness check: same `α`, same `Pfa` for all modes). LTE-25a reports detection 71.9% / FAR 18.2% as a real-world reference point.

### C.4 RD-peak stability

Quantifies jitter of the target peak across trials (sparser reference → noisier peak):
```
# (a) Position stability (bins):
σ_r  = std_MC( argmax_r χ )           # range-bin jitter
σ_ν  = std_MC( argmax_ν χ )           # Doppler-bin jitter
RD_pos_stability = 1 / sqrt(σ_r² + σ_ν²)

# (b) Amplitude stability:
peak_dB[i]   = 10·log10(|χ[r̂,ν̂]|²) for trial i
RD_amp_stability = mean_MC(peak_dB) / std_MC(peak_dB)     # higher = more stable

# (c) Peak-to-sidelobe ratio (per trial, then averaged):
PSLR_dB = 10·log10( |χ[r_t,ν_t]|² / max_{Ω} |χ[r,ν]|² )
```
PSLR is especially diagnostic of mask sidelobe structure (WiFi-22: RpF 42 dB vs MF 25 dB; combs create predictable Doppler/range ambiguities — see E#5). Report `PSLR_dB` mean over trials per mode.

---

## D. Clutter / DPI Cancellation

**Use a two-stage protocol so the benchmark separates the *ideal* upper bound from the *realistic* result:**

### Stage 1 (ideal, sanity ceiling): known-clutter cancellation
Subtract the **exactly-known** direct-path + static clutter contribution (available in simulation because Sionna-RT gives you ground-truth channel taps). This removes DPI/clutter perfectly and isolates the pure reference-density effect on the CAF. Use only to establish the ceiling and to debug.

### Stage 2 (realistic, the reported pipeline): data-driven ECA
Use **ECA / ECA-S (Extensive Cancellation Algorithm, sliding)** — the dominant, well-characterized PCL canceller across all three bands (WiFi: ECA-S whole-packet, WiFi-21b/23a; LTE: ECA / LS-CC, LTE-25a/25b; 5G: adaptive/lattice, 5G-22/25a). ECA projects the surveillance signal onto the orthogonal complement of the subspace spanned by delay-(and Doppler-)shifted copies of the **reconstructed reference**:

```
# Reference reconstructed from KNOWN REs only (mode-dependent!):
ref = ofdm_modulate(grid * mask)
# Build clutter subspace from delay-shifted (and optionally Doppler-shifted) ref copies:
X = [ shift(ref, τ) for τ in 0..L ]        # L = clutter range extent (e.g. 250 m worth)
surv_clean = surv - X @ pinv(X) @ surv     # orthogonal projection (ECA-B / ECA-S sliding)
```

**Critical fairness note:** Stage-2 ECA must build its cancellation subspace from the **same mask-reconstructed reference** used in the CAF — *not* from a full clean reference. Otherwise a denser mask gets a (hidden) cancellation advantage on top of its CAF advantage, conflating two effects. Document which stage each reported number comes from.

Parameters to fix identically (from literature): clutter range extent `L` ≈ 250 m (WiFi-22), estimation/update interval 0.05–0.2 s, `P` Doppler taps (0 for static-only, or a few for slow clutter).

---

## E. Pitfalls That Would Make the Comparison UNFAIR (and fixes)

1. **Letting bandwidth / fs / N_sc drift between modes.** Different occupied bandwidth → different range resolution → not a reference-density experiment. *Fix:* assert `B, fs, N_FFT, Δf, N_sc, T_ofdm, CPI, N_sym` are byte-identical; only `M` changes.

2. **Different total TX power across modes.** If sparser masks transmit fewer/weaker REs, SNR at Rx drops for reasons unrelated to reference density. *Fix:* normalize the **full transmitted grid** (data + known REs) to identical total power `P_tx` per CPI in every mode. The mask affects only what the *radar knows*, never what is *transmitted*.

3. **Pilot power-boosting (LTE +6 dB) without compensation.** Boosting known REs inflates effective reference power. *Fix:* either disable boosting, or renormalize so total grid power is constant — and apply the same policy to all modes (or none).

4. **Reusing/forgetting to re-randomize unknown data.** If the unknown-data REs are the same across Monte-Carlo trials, you under-estimate data-self-interference variance. *Fix:* fresh random data per trial (per seed); but use the **same seed/data realization across the 4 modes within a trial** so the only difference is which REs are revealed by the mask. Re-randomize across trials.

5. **Mask-induced range/Doppler ambiguities mistaken for detector quality.** Comb-`p` pilots create grating lobes (LTE comb-6 → Doppler/range aliases; WiFi 0.8 µs L-STF sub-symbol → 240 m/380 m ambiguity peaks, WiFi-22). A mode may look worse purely from aliasing, not lower gain. *Fix:* report PSLR and explicitly map predicted ambiguity locations per mask; exclude known grating-lobe bins from the background region `Ω` consistently across modes, or interpolate-reference deep notches (I-RpF style) identically.

6. **Cancellation built from a privileged (full/clean) reference.** See D — gives dense masks a hidden DPI-rejection bonus. *Fix:* ECA subspace from the same mask-reconstructed reference.

7. **CFAR threshold computed per-mode.** If `α` or training-cell geometry adapts to each mode's background, you mask the very SCR differences you want to measure. *Fix:* identical `α` (fixed `Pfa`), identical guard/training-cell layout for all modes; verify `FAR_emp` per mode as a consistency check.

8. **Target on the zero-Doppler / bistatic-contour blind zone.** All three bands lose targets riding low Doppler after DPI rejection (LTE-23, LTE-25a, 5G). *Fix:* choose a trajectory with non-zero radial velocity throughout the CPI so the comparison isn't dominated by the canceller's notch (which is mode-independent and would wash out the reference-density signal).

9. **Different unambiguous-velocity windows from mask periodicity.** Sparse-in-time masks (M3 SSB ~50 Hz repetition, M1 bursts) set the effective pilot-PRF and thus the unambiguous Doppler span (5G-26: 2.6 → 20.2 m/s). This is a *legitimate* part of the hypothesis, but ensure the target Doppler is **unambiguous in the densest mode** so M4 isn't penalized by aliasing it cannot avoid — or, conversely, report it as an intended consequence of sparsity. *Fix:* document the effective pilot-PRF per mode and keep the true target Doppler inside the *narrowest* unambiguous window, so all modes can in principle resolve it.

10. **Comparing energy instead of detectability.** The literature's central caution (LTE-23/25b: all-symbol has *higher* SNR=24.2 dB but 3× more false plots; "energy is not the right figure of merit"). *Fix:* headline metric is **Pd at fixed Pfa** and **SCR**, not raw peak energy.

---

### Recommended defaults for the first run
- Numerology: B=20 MHz, Δf=30 kHz, N_FFT=1024, N_sc=600, 14 sym/slot.
- CPI = 100 ms (`N_sym ≈ 2800`); Monte-Carlo `N_MC ≥ 500` (≥ 10/Pfa for the 1e-4 point).
- Pfa ∈ {1e-4, 1e-6}; CA-CFAR, guard 2×2, training 8×8 (fixed).
- Stage-2 ECA: `L` = 250 m clutter extent, `P` = 0 (static), update every 50 ms.
- Trajectory: constant non-zero radial velocity (avoid Doppler blind zone), target Doppler inside M3's unambiguous window.
- Headline plot: **Pd (and SCR) vs known-RE fraction**, ordering M3 < M2 < M1 < M4.

This directly tests the hypothesis: sparser mask → lower CAF gain + higher data-self-interference → lower SCR/Pd, with M3 (SSB) as the floor and M4 (DMRS/PRS-rich) as the ceiling.