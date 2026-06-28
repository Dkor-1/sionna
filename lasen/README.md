# LaSen — faithful reproduction (5G-26, SenSys)

Monostatic **5G-NR ISAC** drone tracking (LaSen), reproduced on the Sionna physical
channel. **Separate sub-project** from the parent bistatic-passive benchmark — different
paradigm (CFR sparse-recovery + tracking, not CAF/CFAR). Principle: **faithful to the
paper's eqs/params**, **phase-gated** (each phase proves itself with a figure before the
next), **viz-first** (every module emits a figure mapped to a LaSen figure).

## 👉 먼저 볼 것
**`report.ipynb`** (Phase A·B 그림 내장). Faithfulness/honest paper-diff: `docs/FAITHFULNESS.md`.

## Phases (gate must pass before the next)
| | What | Gate | Status |
|---|---|---|---|
| **A** | monostatic CFR → full-band 2D-FFT RD | peak on analytic GT + 0-Hz clutter collapses (Fig 4) | ✅ PASS |
| B | non-uniform occupancy + 2D-OMP (Eq 4-6) | round-trip + **binary**: sparse FFT misses weak / OMP finds it (Fig.17) | ✅ PASS (full 2803) |
| C | ID score (Eq 7-9) + global/local + Kalman (Eq 10-13) | track follows GT (Fig 6) | ⏳ |
| D | RMSE/CE/DR + baselines (2D-OMP, Lerp) | LaSen < baselines, RMSE trends (Fig 9,12-14) | ⏳ |

## Run
```bash
PY=/home/yunjung/workspace/jeong/miniforge3/envs/sionna/bin/python
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY lasen/run_lasen.py --phase A   # CFR→RD sanity
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY lasen/run_lasen.py --phase B   # occupancy+2D-OMP
$PY lasen/build_report.py                                                  # report.ipynb 재생성
```
Numerology: SCS 30 kHz, 3072-FFT, 2604 active SC, BW 78.12 MHz, fc 5.8 GHz.

## Files
`nr_waveform.py`(5G-NR numerology+grid) · `monostatic_scene.py`(tx≈rx ISAC, CFR=Y/X) ·
`viz.py`(plot helpers) · `run_lasen.py`(pipeline+gates) · `omp2d.py`/`idscore.py`/
`tracker.py`/`baselines.py`(Phase B–D, 예정) · `build_report.py`(report) ·
`docs/FAITHFULNESS.md`. Reuses parent `passive_radar_stage1.build_scene` + `drones`.
