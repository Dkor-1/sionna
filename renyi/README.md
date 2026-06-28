# renyi — faithful reproduction (5G-22, Remote Sens. 2022)

Bistatic **5G passive-radar** drone detection with **Rényi-entropy adaptive
integration** (Maksymiuk, Abratkiewicz, Samczyński, Płotka, *"Rényi Entropy-Based
Adaptive Integration Method for 5G-Based Passive Radar Drone Detection"*,
Remote Sens. 2022, 14, 6146). Paper PDF: `IRS/논문/드론/5G/22_*.pdf`.

A **separate sub-project** from the parent benchmark, in the same mould as
`../lasen/`: **faithful to the paper's eqs/params**, **phase-gated** (each phase
proves itself with a figure before the next), **viz-first** (every module emits a
figure mapped to a paper figure). Honest paper-diff: `docs/FAITHFULNESS.md`.

## Why this paper (vs the others)
The paper's own **Section 6 is a synthetic simulation** (MATLAB 5G-NR + AWGN), so
reproducing it in Sionna/NumPy is a *direct* match, not an approximation of a
hardware rig. **Phases A–C need no Sionna RT — they run locally in seconds.** Only
Phase D (the Sec 7 real flight) uses the RT bistatic channel on the server. The
method (CAF + CFAR) reuses the parent benchmark's DSP; the **novelty = the Rényi
entropy frame selector**, implemented fully in `renyi.py`.

## 👉 먼저 볼 것
`report.ipynb` (all phases, figures embedded — no kernel needed). Or the PNGs:
`outputs/phaseA_content.png` · `phaseB_metrics.png` · `phaseC_pd.png` ·
**Phase D**: `phaseD_geometry.png` · `phaseD_trajectory.png` · `phaseD_tint.png` ·
`phaseD_entropy.png` (+ `phaseD_channel.png` · `phaseD_snapshot_rd.png`) +
`phase{A,B,C,D}.json` (gate verdicts). Faithfulness: `docs/FAITHFULNESS.md`.

## Phases (gate must pass before the next)
| | What | Gate | Status |
|---|---|---|---|
| **A** | content-dependency: CAF vs content fill | low-content target buried, high-content clear; SCR↑ with fill (Fig 8) | ✅ PASS (local) |
| **B** | content metrics: entropy vs power & B_eff | H monotonic in fill & SNR-robust; power & B_eff can't separate content (Fig 9/11/13) | ✅ PASS (local) |
| **C** | adaptive integration → P_d | P_d rises with fill for Pfa {1e-4,1e-6,1e-8}; range vs T_int (Fig 14/15-17) | ✅ PASS (local) |
| **D** | real-flight bistatic on Sionna RT | CFAR detections follow Sionna GT; T_int 20→100 ms sharpens V (Fig 21-23) | ✅ PASS (server) — 9/9 wp hit, R err 1.0 m / V err 0.7 m/s; ΔV 4.4→0.9 m/s; dense detects/sparse buried |

## Run
```bash
# Phases A-C: local, pure synthetic (numpy + scipy + matplotlib), ~seconds each
python3 run_renyi.py --phase A
python3 run_renyi.py --phase B
python3 run_renyi.py --phase C
# Phase D: server only (sionna-rt + OptiX on the RTX-4090)
PY=/home/yunjung/workspace/jeong/miniforge3/envs/sionna/bin/python
CUDA_VISIBLE_DEVICES=0 $PY run_renyi.py --phase D
```
Module self-tests also run standalone, e.g. `python3 renyi.py`, `python3 range_eq.py`.

Numerology (paper Sec 6 / Table 2): SCS 30 kHz, fs 61.44 MHz, 2048-FFT, 106 RB →
1272 SC = **38.16 MHz** occupied, fc **3.44 GHz**. Range eq Table 2: EIRP 73 dBm,
Gr 10 dBi, D0 11 dB, L0 10 dB, T0 493 K, RCS {1,10,50,100} m².

## Files
`nr_grid.py` (synthetic 5G-NR grid + RB-level occupancy + OFDM waveform) ·
`renyi.py` (**the novelty**: STFT Eq 6 → Rényi entropy Eq 8 γ=3 → frame selection) ·
`content_metrics.py` (power + RMS B_eff Eq 4 baselines that the paper shows fail) ·
`radar.py` (bistatic geometry Eq 1-2, synthetic echo, CAF Eq 3, CA-CFAR, SCR) ·
`range_eq.py` (range eq Eq 9-10 + Table 2 → Fig 14, verified vs the paper) ·
`bistatic_scene.py` (Phase D RT stub; reuses parent `build_scene`/`trace_channel`) ·
`run_renyi.py` (phase-gated pipeline + figures + JSON gates) ·
`build_report.py` (assemble `report.ipynb`) · `outputs/`. Reuses parent
`passive_radar_stage1` (RT scene) + `drones` (RCS dBsm) for Phase D.
