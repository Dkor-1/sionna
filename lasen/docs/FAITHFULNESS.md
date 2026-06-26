# LaSen faithfulness — same as the paper / different (honest)

Reproduction of **LaSen** (5G-26, SenSys 2026: 5G-NR monostatic ISAC, low-altitude
drone tracking, RTK GT). The project principle is "paper facts only" — so this file
states, per stage, what is **faithful** and what **differs** (and why). The bar for
"reproduced" is the **trend/shape**, not absolute dB/cm (geometry & CPI differ), the
same standard as `../docs/VALIDATION.md`.

## The one decision that frames everything
**LaSen is MONOSTATIC ISAC; the parent benchmark is BISTATIC passive.** The gNB is the
transmitter AND the receiver — it knows the transmitted signal X, forms the sensing
channel **H = Y / X** (Eq 1), and recovers range-Doppler from H. This is a *different
processing paradigm* (sparse recovery + tracking, not CAF/CFAR detection), so LaSen
lives in its own sub-project `lasen/`. We reuse only the parent's physical-channel
machinery (scene/PathSolver/CFR, drone RCS-dBsm scaling), not the bistatic detection
chain.

## Faithful (same as paper)
- **Numerology** (§2.2, Tab.1): SCS 30 kHz, 3072-FFT, 2604 active SC, fs 92.16 MHz,
  occupied BW 78.12 MHz, fc 5.8 GHz — `nr_waveform.NRNumerology`.
- **Monostatic geometry & channel:** tx≈rx co-located gNB; CFR H=Y/X from Sionna RT;
  monostatic Doppler **f_d = 2 v / λ** and monostatic target range **d = c·τ/2**
  (verified in Phase A: measured fD −430 Hz vs analytic −431, range 48.0 m vs 48.2 m).
- **Static-background suppression** (§4.1.1): slow-time mean subtraction removes the
  0-Hz clutter / self-leakage ridge (Phase A: −124 dB collapse → Fig 4 reproduced).
- **Pipeline stages** (planned, per Eq): non-uniform occupancy mask (Phase B), 2D-OMP
  sparse recovery Eq 4-6 (Phase B), incoherence-density ID score Eq 7-9 (Phase C),
  hierarchical global/local estimation + Kalman Eq 10-13 (Phase C), metrics
  RMSE/CE/DR (Phase D), baselines 2D-OMP-plain & Lerp (Phase D).

## Different (stated honestly)
1. **5G-NR waveform library.** Paper/real system uses a 5G toolbox (MATLAB) /
   `sionna.nr`; this container has only `sionna-rt` (no TensorFlow PHY), so the grid is
   generated in **NumPy following the same standard numerology** (DMRS comb-2 / PDSCH).
   Equivalent grid structure, different library — not a physics difference.
2. **Occupancy = synthetic traffic model.** The paper's hard problem is *real*
   non-uniform 5G traffic. We model it (`occupancy.occupancy_mask`) from the paper's
   measured density bins (sparse 0.6–6.8 % … dense 13.1–19.3 %, dense ≈ 5 % of time,
   Tab.1/Fig.12). So we report "modelled occupancy", **not "captured traffic"**.
3. **Self-leakage.** A real SDR has TX→RX self-interference (paper handles it). Sionna
   has none, so to exercise the atom-isolation step we **intentionally inject** strong
   static / near-0-Doppler leakage (clearly flagged where used).
4. **Ground truth.** Paper uses RTK (cm-level). We use **Sionna's exact, free GT** —
   strictly better, and the GT axis is satisfied by construction.
5. **Scope: bulk Doppler only.** No propeller micro-Doppler (LaSen lists it as §8 future
   work too) — matches the parent project's scope.
6. **Phase B single-target limit.** For a SINGLE target the 2D-OMP first atom equals the
   2D-FFT peak (they agree on position) — so Phase B demonstrates the **dynamic-range /
   leakage-floor** advantage (OMP clean sparse RD, PSLR 156 dB, vs the FFT's sub-Nyquist
   floor 43–49 dB that worsens with sparsity), which is what buries weak targets for a
   plain FFT. A literal weak-2nd-target demo is confounded by the RT drone's multipath
   spread (its residual floor > −40 dB hides a faint injected target for OMP too), so it
   is left to future work (a cleaner point-target or 2 RT drones). Phase B also
   sub-samples slow-time (256 symbols) so the 2D-OMP is tractable; Phase A already
   verified the full real-symbol pipeline.

## Status (phase-gated)
| Phase | What | Gate | Status |
|---|---|---|---|
| **A** | monostatic CFR → full-band 2D-FFT RD | RD peak on analytic GT + 0-Hz clutter collapses (Fig 4); **R1** slow-time = real OFDM symbols; **R2** Doppler sweep follows 2v/λ | ✅ PASS — N=2803 real symbols (PRF 28 kHz, not hardcoded), R 0.2 m / fD 1.1 Hz err, clutter −97 dB, sweep max-err 14 Hz over 4–20 m/s |
| B | non-uniform occupancy + 2D-OMP (Eq 4-6) | omp2d round-trip + sparse occupancy: 2D-FFT leakage floor worsens, 2D-OMP clean (Tab.1) | ✅ PASS — round-trip OK; OMP recovers target at sparse(3.7%)+dense(16%), OMP PSLR 156 dB vs FFT 43/49 dB (Δ113 dB), FFT leakier when sparser |
| C | ID score (Eq 7-9) + hierarchical global/local + Kalman (Eq 10-13) | track follows GT, survives low-score via local+Kalman (Fig 6) | ⏳ |
| D | metrics RMSE/CE/DR + baselines (2D-OMP, Lerp) | RMSE↓ with density↑/range↓/vel↓; LaSen < baselines (Fig 9, 12-14) | ⏳ |

Each phase emits a figure mapped to a LaSen figure; no phase proceeds before its gate
passes (visual + numeric proof).
