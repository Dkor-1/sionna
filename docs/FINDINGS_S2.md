# Stage-2 findings — fair multi-mode passive-radar benchmark

**Date:** 2026-06-25 · **Code:** `passive_radar_s2.py` · **Scope:** bulk Doppler only.

> **상태(2026-06-26):** S2(이 reference-밀도 연구)는 유효하지만 이제 **Phase-1(`phase1.py`)이
> 헤드라인**이고 S2는 그 한 축(reference-only)의 정밀 버전입니다. 외부 코드리뷰 반영으로 공유
> DSP가 바뀌어 **재실행**됨: Pd 판정이 **±1 range bin → ±30 m(미터 고정)**, CA-CFAR가 클러터
> notch를 **학습에서 제외**(사후 마스킹 아님), Hann은 `hanning(N+2)[1:-1]`. 아래 표 숫자는
> 재실행본(`outputs/s2_results.json`)으로 갱신됨 — 절대값보다 **순서/추세**가 결론.

## Objective
Test the project's core hypothesis on the *reference-structure sparsity* axis
(PROJECT_CONTEXT §3): do Wi-Fi / LTE / 5G illuminators differ in drone
detectability *purely because of how much deterministic reference structure the
passive radar can exploit*, holding everything else fixed?

## Method (literature-grounded)
A single OFDM resource grid (subcarrier × OFDM-symbol) at `fs = B`. Every RE is
transmitted with unit-power QPSK ⇒ **identical TX power across modes (fair
illuminator)**. A per-mode binary **pilot mask** marks which REs are *known* to
the radar; the CAF reference is rebuilt from KNOWN REs only, while surveillance
is the full grid through the Sionna-RT channel. Unknown-data REs become
self-interference in the CAF. This is the realistic way passive radar exploits
CRS / SSB / DMRS / preamble, and was independently recommended by a 21-paper
survey (see `outputs/_paper_survey_unified.md`).

Pilot masks (standards-inspired, density = the single varied factor):

| Mode | Pattern | Known REs | Citation anchor |
|---|---|---|---|
| `5g_ssb_sparse` | 240-SC × 4-sym block, periodic | ~0.6 % | 5G-25a/b, 5G-22, 5G-26 |
| `lte_crs` | comb-6, symbols {0,4,7,11}/14 | ~4.8 % | LTE-23, LTE-25b, LTE-20 |
| `wifi_preamble` | all-SC burst, first 4 of 22 symbols | ~18 % | WiFi-24, WiFi-23b |
| `5g_dmrs_prs_rich` | staggered comb-2, 8 of 14 symbols | ~29 % | 5G-26, 5G-25a |

**Control protocol — held identical:** geometry, trajectory, channel realization,
B, fs, FFT size, Δf, CPI, N, M, total TX power, noise PSD, RCS, clutter
canceller, CFAR α/guard/training. **Single variable:** the pilot mask. The
*same random data realization* is used across the 4 modes within each trial, so
the only difference is which REs the mask reveals. Target has non-zero radial
Doppler throughout (avoids the DPI-notch blind zone, spec pitfall E#8).

**Clutter cancellation:** ideal known-static-channel removal (spec "Stage-1"
ceiling). Data-driven ECA (spec "Stage-2", realistic) is the next step.

**Metrics (Monte-Carlo over noise + data):** SCR, Pd (CFAR hit within ±bins of
the true cell at fixed Pfa), empirical FAR, PSLR, RD-peak position stability.

## Results
Benchmark: fc 3.5 GHz, B 100 MHz, CPI 41.9 ms (N=512, M=8192), 24 Monte-Carlo
trials, drone-echo SNR −23 dB/sample, Pfa 1e-5. Ordered by reference density:

| Mode | Known REs | SCR (dB) | PSLR (dB) | Pd | empirical FAR |
|---|---:|---:|---:|---:|---:|
| `5g_ssb_sparse` | 0.21 % | **5.4 ± 2.3** | −3.8 | **0.00** | ~0 |
| `lte_crs` | 4.77 % | 16.6 ± 1.0 | 4.7 | 1.00 | 1.3e-5 |
| `wifi_preamble` | 18.42 % | 22.3 ± 0.4 | **1.0** | 1.00 | 9.7e-6 |
| `5g_dmrs_prs_rich` | 28.51 % | **24.4 ± 0.3** | 12.5 | 1.00 | 8.4e-6 |

_(post DSP-review fixes: mean-background SCR per spec C.1 — ~1.6 dB below the
earlier median estimate; empirical FAR now tracks the design Pfa=1e-5 after
excluding the target's range-Doppler sidelobe cross; SSB period→120 for the
intended ~0.2 % density floor. Per-drone & CPI/reference experiments added.)_

Figures: `outputs/s2_rd_grid.png` (per-mode RD maps), `outputs/s2_summary.png`
(SCR & Pd vs reference density), `outputs/s2_sweep_pd_snr.png` (Pd-vs-SNR).

## Key findings
1. **Monotone SCR vs reference density** — SCR increases monotonically with the
   known-RE fraction (5G-SSB < LTE-CRS < Wi-Fi < 5G-rich), directly confirming
   the hypothesis. Matches the literature trend (LTE-23/25b: CRS-only ≈17.5 dB
   vs all-symbol ≈24.2 dB; 5G-22: ~10 % occupancy ≈ undetectable, ~70 % ≈ 24 dB).
2. **5G SSB-only is the detectability floor** — at the benchmark SNR it largely
   *fails* (low Pd, unstable peak). Its reference is both ultra-sparse and
   narrowband (poor range resolution → range-smeared echo). This is exactly why
   the 5G literature adds Rényi-entropy adaptive integration / multi-block
   accumulation (5G-22/26) — bulk single-CPI SSB is not enough.
3. **Energy ≠ detectability** — consistent with LTE-23/25b's caution, the
   benchmark's figure of merit is Pd-at-fixed-Pfa and SCR, not raw peak energy.
4. **Pd-vs-SNR separation** — the sweep shows the modes' detection thresholds
   spread by reference density: the rich mode detects several dB lower in SNR
   than the sparse modes (the "fair cost of a thin reference").
5. **SCR ≠ sidelobe quality (PSLR caught it)** — Wi-Fi-preamble has high SCR
   (24 dB) but the **lowest PSLR (1.0 dB)**: its bursty/periodic preamble
   (period-22 symbols) creates Doppler grating lobes, so strong sidelobes sit
   near the target despite a strong main peak — exactly the mask-induced
   ambiguity the spec warns about (E#5; WiFi-22 reports L-STF ambiguity peaks).
   5G-rich's dense staggered comb is cleanest (PSLR 12.5 dB). So "rich reference"
   helps SCR *and* sidelobe structure, but a *dense-in-time* reference (Wi-Fi)
   buys energy without buying clean Doppler sidelobes — a real, non-obvious
   trade the benchmark surfaces.

## Caveats / fairness notes
- Ideal clutter cancellation (perfect static-scene knowledge) — an upper bound;
  realistic data-driven ECA (built from the *same* mask-reconstructed reference,
  spec D + pitfall E#6) is pending and will lower all curves.
- Numerology here: fc 3.5 GHz, B 100 MHz, Δf 24.4 kHz (the survey's default is
  B 20 MHz / Δf 30 kHz — internally consistent either way; only relative
  ordering is claimed).
- Monte-Carlo N=24 trials (good for Pd≈{0,1} & SCR; the FAR/1e-4 point wants
  N≳500 for a tight estimate — easy to scale on this GPU box).
- Comb masks create predictable range/Doppler grating lobes; the chosen range
  window (0–600 m) sits inside the first alias for every mode, and PSLR is
  reported as the ambiguity diagnostic (spec pitfall E#5).

## Next (S3/S4)
- Data-driven ECA-S; sweep bandwidth / occupancy / CPI / geometry; 2-port CRS &
  variable SSB period; larger N_MC; ROC curves; then the full control-protocol
  comparison write-up. Optional micro-Doppler layer (rotating point-scatterer).
