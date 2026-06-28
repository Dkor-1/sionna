# 5G-22 faithfulness — same as the paper / different (honest)

Reproduction of **5G-22** (Maksymiuk et al., *Rényi Entropy-Based Adaptive
Integration Method for 5G-Based Passive Radar Drone Detection*, Remote Sens. 2022,
14, 6146). Project principle = "paper facts only", so this file states, per item,
what is **faithful** and what **differs** (and why). Bar for "reproduced" = the
**trend/shape**, not absolute dB/cm (geometry, STFT grid, CPI differ) — same
standard as `../../docs/VALIDATION.md` and `../../lasen/docs/FAITHFULNESS.md`.

## The one thing that frames everything
The paper's **Section 6 results are themselves a synthetic simulation** (MATLAB 5G
Waveform Generator + AWGN), and the radar chain is **bistatic CAF + CFAR** — exactly
the parent benchmark's paradigm. So Phases A–C are a *direct* re-run of the paper's
own simulation in NumPy (no RT, no hardware), and the **novelty we must port
faithfully is the Rényi-entropy resource measure** (Sec 5). Only Phase D (Sec 7 real
flight) swaps in a Sionna-RT bistatic channel.

## Faithful (same as paper)
- **Bistatic PCL model:** R_b = R1+R2−L (Eq 1), V_b = −λ f_d (Eq 2), cross-ambiguity
  χ(R_b,V_b) (Eq 3). Processing chain = Fig 2 (filter/clutter → CAF → CFAR → extract).
- **5G-NR numerology (Sec 6):** SCS 30 kHz, fs 61.44 MHz, 2048-FFT, 1272 active SC =
  38.16 MHz occupied, fc 3.44 GHz; content fill 0..100 % at RB×slot granularity
  ("random positions in the time-frequency allocation grid", Figs 5-7).
- **Rényi-entropy method (Sec 5):** STFT (Eq 6) → TF = |STFT| → Rényi entropy (Eq 8)
  with **γ = 3**; calibrate to the full-allocation max, then keep frames above a
  threshold (paper: 25.5 vs max 25.67 ≈ 99 %; we use the same fraction). `renyi.py`.
- **Baselines that fail (Sec 4):** power measurement (Sec 4.2, Fig 10) and RMS
  effective bandwidth B = ∫A(f)df / A_max (Eq 4, Sec 4.3, Fig 11) — reproduced as the
  foils the entropy beats. `content_metrics.py`.
- **Range equation (Eq 9-10, Table 2):** EIRP 73 dBm, Gr 10 dBi, D0 11 dB, L0 10 dB,
  T0 493 K, RCS {1,10,50,100} m² → Fig 14. **Verified**: RCS 1 → 25.8 km @0.5 s,
  RCS 100 → 97.2 km @1 s, matching the paper's figure. `range_eq.py`.
- **CFAR / operating points:** CA-CFAR with Pfa {1e-4, 1e-6, 1e-8} (Figs 15-17);
  P_d-vs-fill families; SNR threshold 15 dB for the flight (Sec 7.2).
- **Scope: bulk Doppler only** (the paper is bulk-only; micro-Doppler is its Sec 8
  future work) — matches the parent project's scope.

## Different (stated honestly)
1. **Two equations are implemented as the paper's numbers imply, not as printed:**
   - **Eq 8 normalisation.** As printed it divides by `∫∫TF` (L1), but the reported
     magnitudes (~25.8 at full = log₂ of the TF-grid cell count) are the *normalised*
     Rényi entropy of the cited Baraniuk 2001 (ref [38]); we normalise TF to a
     distribution first. We therefore reproduce the **shape** (monotonic in fill,
     SNR-robust); the absolute value scales with the STFT grid size (our local grid
     → H ≈ 13–17, not 20–26). `renyi.py` header documents this.
   - **Eq 9 power law.** The printed `sqrt(·)` does not give Fig 14's magnitudes; the
     standard monostatic-equivalent **4th-power** law `R_e = (·)^{1/4}` does (verified
     above). Read as a typo; `range_eq.py` uses the 4th root and says so.
2. **5G waveform library.** Paper uses the MATLAB 5G toolbox; this container has only
   `sionna-rt` (no TF PHY), so the grid is generated in **NumPy** with the same
   numerology (same as `../lasen/`). Equivalent structure, different library.
3. **Synthetic echo (Phases A-C).** Surveillance = analytic delay + Doppler + AWGN
   (the paper's own Sec 6 method). Consequence: with a *clean* echo, only echo
   **energy** (∝ content) drives the CAF SCR, so Phase A shows the energy/integration
   story. The **ambiguity-quality** degradation of a sparse reference (why equal-power
   low-content still fails, Fig 10b/d) needs realistic multipath/clutter → it appears
   naturally in **Phase D** (Sionna RT). The power-vs-content *metric* decoupling is
   still shown cleanly in Phase B.
4. **Occupancy = synthetic traffic model.** RB×slot fill (Figs 5-7), not captured
   live traffic — so "modelled occupancy", as in `../lasen/`.
5. **Ground truth.** Paper Sec 7 uses GPS logs; Phase D uses **Sionna's exact GT** —
   strictly better, GT axis satisfied by construction.
6. **No SDR hardware chain.** USRP X310, amps/filters (Fig 19), GPS freq-sync — N/A
   in simulation.
7. **Target / RCS.** Paper's real target = DJI **M600 PRO** (RCS unknown at 3.44 GHz);
   Table 2's range-eq RCS sweep is generic {1..100} m². Phase D drone echo uses the
   parent `drones` literature dBsm anchors (labelled estimate, `../NOTES.md` fix #4).
8. **No tracking/Kalman.** The paper did **not** implement tracking (single receiver,
   Sec 2 end); Phase D overlays raw detections on GT (Fig 23) — faithful to that.

## Status (phase-gated)
| Phase | What | Gate | Status |
|---|---|---|---|
| **A** | content-dependency: CAF vs fill | low-content buried, high-content clear, SCR↑ (Fig 8) | ✅ PASS — SCR 7.5→19.2 dB over fill; detect ≥30 % only |
| **B** | entropy vs power & B_eff | H monotonic & SNR-robust; power/B_eff can't separate content (Fig 9/11/13) | ✅ PASS — H 12.7→16.8 monotonic; Δpower 0.02 dB vs Δentropy 0.75 at equal power |
| **C** | adaptive integration → P_d | P_d↑ with fill, Pfa {1e-4,1e-6,1e-8}; range vs T_int (Fig 14/15-17) | ✅ PASS — P_d S-curve 0→1; Fig 14 verified |
| **D** | real-flight bistatic (Sionna RT) | detections follow Sionna GT; 20→100 ms sharpens V (Fig 21-23) | ⏳ server |

Each phase emits a figure mapped to a paper figure + a JSON gate verdict; no phase
proceeds before its gate passes (visual + numeric proof).
