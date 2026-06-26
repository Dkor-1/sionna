# NOTES — running log (keep updated every step)

> Living progress + decision log for the Sionna passive-radar work.
> Newest changelog entry at the bottom. Human-readable summary: `report.ipynb`.

## ⚠️ 가장 주의 — 경로 / 저장 규칙 (항상 먼저 확인)
**허용 디렉토리는 이 3곳뿐. 그 외 어떤 디렉토리도 절대 건드리지 말 것.**
- `/workspace`  (= 실제 `/home/yunjung/workspace`)
- `/data/public/jeong`
- `/data/ckpoint/jeong`

**저장 규칙 (PROJECT_CONTEXT §11):**
| 무엇 | 어디 | 비고 |
|---|---|---|
| 코드 + 자주 읽는 작은 파일 | `/workspace/jeong` (SSD) | 작게 유지 |
| 생성 데이터/결과물 (RD-map, CIR, 그림, 데이터셋) | `/data/public/jeong` (HDD) | **여기로** |
| 모델 웨이트 | `/data/ckpoint/jeong` (HDD) | |
- **workspace(SSD)에는 큰 결과물 절대 쓰지 말 것** — SSD 여유 적음(243G). 대용량은 HDD.
- 대용량(>~1GB) 또는 보관용 데이터셋 생성 **전에 멈추고** 사용자에게 알림 → `/data/...`로 이동.

**현재 현실 (권한 블로커 — 중요):** 이 세션은 비-root `yunjung`이라 위 지정 경로 3곳에
**쓰기 권한이 없음**(`/workspace/jeong`=root:root, `/data/public/jeong`=group member,
`/data/ckpoint/jeong`=root:root; passwordless sudo 없음). → **현재 코드·결과물은 쓰기 가능한
`/home/yunjung/workspace/jeong_sionna/`에 임시 스테이징.** 권한 풀리면 위 규칙대로 이전.
관리자 조치·이전 명령: `docs/ENV_NOTES.md`. 또한 `conda activate sionna` 깨짐 → env python
직접 호출(`/home/yunjung/workspace/jeong/miniforge3/envs/sionna/bin/python`, `run.sh`).

> 코드의 출력 경로는 이미 자동 분기: `/data/public/jeong/...`가 쓰기 가능하면 거기로,
> 아니면 `./outputs`(스테이징). 권한 풀린 뒤엔 `--outdir /data/public/jeong/...` 또는
> `PR_OUTDIR` 환경변수로 지정.

### 실질 운영 (현재 — 권한 문제 존재) — 이대로 작업
- **현재는 `/home/yunjung/workspace/jeong_sionna/`에서 작업.** 여기는 yunjung 소유라
  쓰기 가능 → **코드 + 작은 결과물(그림·CSV·JSON, 개당 ~수십 MB)은 여기 둬도 OK**(SSD,
  코드 두는 곳으로 적절).
- **읽기는 됨**: `/workspace/jeong`(PROJECT_CONTEXT, conda env 실행), `/data/public/jeong/papers`
  (논문) — 읽기/실행 OK, **쓰기만 불가**.
- **용량 문제가 생기거나 대용량 생성물이 나올 때 → 직접 `/data/...`로 못 옮김(권한).
  그러니 사용자에게 요청해야 함.** 절차:
  1. 생성 전/누적이 커지면 **멈추고** 사용자에게 알림 (예상 용량 + 파일 경로).
  2. **복사/이동 명령을 만들어 드리면**, 권한 있는 쪽(member 그룹 또는 root = 보통 사용자
     본인 로그인 세션)에서 **사용자가 실행**해 `/data/public/jeong`(데이터) ·
     `/data/ckpoint/jeong`(웨이트)로 이동.
  3. 이동 후 SSD 스테이징본은 정리.
- **모니터링 기준(자가)**: 단일 작업 >~1GB, 또는 raw IQ/CIR 같은 보관용 덤프, 또는
  `jeong_sionna` 누적 >~5GB / SSD 여유 급감 시 → 위 절차로 사용자에게 요청.
- **정식 해결**: 관리자가 `yunjung`을 `member` 그룹에 추가+`chmod g+w`, 또는 jeong
  디렉토리들 `chown yunjung`, 또는 세션을 root로 기동 (`docs/ENV_NOTES.md`).

## Current status — 2026-06-26 (after external code-review round)
- **Phase-1 controlled benchmark (`phase1.py`)** is the headline. An external
  review found the DSP engine sound but flagged a **confounded headline** and stale
  numbers. This round fixes the SCIENCE/claims, not the engine. **All headline
  numbers now live in `outputs/phase1_*.json` + `report.ipynb` (auto-rendered) —
  do NOT hardcode them here (they go stale; review fix b).**
- **NEW — decoupled single-factor sweeps (`phase1.py --mode decouple`)**: the main
  matrix has std=reference=carrier **collinear**, so 'carrier dominates' is not
  identifiable. The decouple mode varies **carrier / bandwidth / reference one at a
  time** (others fixed) → `outputs/phase1_decouple.{json,png}`. Headline is now the
  honest, identifiable decomposition (see report §1; ΔSCR per factor in the JSON
  `spans_db`).
- **NEW — experiment-geometry visualization (`visualize_scene.py`)**: bistatic /
  open-field setup, real Sionna-traced rays, iso-range ellipse, geometry→Doppler →
  `outputs/viz_*.png`, embedded as report §0.
- S1 (single RD map) and S2 (reference-density study + experiments) remain valid
  (re-run after the shared DSP fixes: meter Pd window, notch-excluded CFAR, Hann).

### Phase-1 headline (qualitative — numbers in JSON/report)
- **Scenario axis:** only **radial** detects; **hover & tangential blind**. Cause is
  DUAL and both honestly attributed: **bulk-Doppler scope + the ideal zero-Doppler
  canceller** remove the 0-Doppler echo. Matches LIPASE/ONERA measured blind zones →
  correct result. Those cells are also **drone(RCS)-axis degenerate** (SCR=NaN).
- **Spec axis (decoupled, ΔSCR):** **reference structure ≈11 dB (dominant)** >
  carrier propagation λ² ≈7 dB (monotonic decline; net ≈5 dB after the C-band RCS
  rise, non-monotonic) > **bandwidth ≈1 dB (negligible, CONDITIONAL)**. So the honest
  claim is *system-level* "low-band Wi-Fi beats high-band 5G, **mostly via reference
  structure**, carrier secondary, bandwidth ~irrelevant", NOT "carrier dominates".
  **Bandwidth caveat (review-3):** 'bandwidth ~irrelevant' holds only for BAND-FILLING
  references (CRS/preamble, density scale-invariant). **SSB is a fixed 240-SC block →
  its density DILUTES as B grows** (20MHz→1.56%, 100MHz→0.20%, 8x), so the matrix's
  5g 20→100MHz SCR drop (14.6→6.8) is **SSB dilution (= reference sparsity), not pure
  bandwidth** — measured by the `bw_ssb` decouple sweep (`spans_db.bandwidth_ssb`).
  This STRENGTHENS the reference-dominates thesis. After the carrier-RCS fix the matrix
  high-band cells mostly detect (wifi 80@5.0, 5g 20@3.5); only 5g 100@3.5 fails. Exact: `spans_db`.
- **Drone axis:** monotonic in dBsm by construction (fix #4, deterministic scaling);
  carrier-dependent RCS (`rcs_dbsm_at`). Numbers: `phase1_matrix.json`.
- Operating point (snr_ref, serialized in the JSON `config` block) is a calibration;
  relative ordering is the science.

## Phase-1 design (per EXPERIMENT SPEC)
Three study axes, everything else fixed (the control). **SCR is measured, not set.**
- **A) signal spec** — realistic BW × carrier × SCS per standard (Wi-Fi/LTE/5G),
  each with its reference structure (preamble / CRS / SSB).
- **B) drone** — real DJI models; RCS enters as a deterministic scaling (see fix #4).
- **C) scenario** — motion → bistatic Doppler: hover / radial / tangential /
  **doppler_switch** (renamed from "waypoint" — it is a mid-CPI Doppler transition at
  a FIXED position, not a trajectory; review fix #6).
- **Fixed budget** (now SERIALIZED into the JSON `config` block, review fix #8):
  geometry, unit-power illumination (= fixed EIRP), absolute noise density N0
  (anchored once at LTE-10MHz-radial-refmesh, snr_ref recorded in config; averaged
  over 5 RT seeds, review fix #7), CPI=0.1 s, CFAR Pfa=1e-5, **Pd hit window = ±30 m
  (metric, not bins; review fix #2)**, #trials. Chain: Sionna RT → ref(pilots)+
  surveillance → CAF → CFAR. **SCR is the primary metric; Pd is secondary.**
- Scope: **bulk Doppler only** → hover & tangential are expected blind cases.

## Key decisions / fixes (Phase-1)
1. **Batch length M ≫ reference period.** Fixed-CPI `M=CPI·B/N` made M smaller than
   the LTE CRS period → batches with no pilots → broken integration. Fix: `M=16384`
   (≫ pilot period), `N` derived from CPI (Doppler res = 1/CPI). 
2. **Analytic ground truth.** Dense (20 M) RT sampling yields many diffuse paths;
   `argmax|doppler|` picked a wrong weak path (R=48.7 m vs true 26.7 m). Fix:
   compute GT range/Doppler analytically from geometry+velocity (exact, robust).
3. **Doppler sign convention.** Analytic Doppler had the opposite sign to the
   CAF/Sionna axis → metric read an empty cell (fake "4 dB cap", Pd=0). Fix: negate
   analytic fD to match (`paths.doppler` was −165 vs analytic +165 in S2).
4. **Per-drone RCS scaling.** Diffuse scatter off a cube isn't monotonic in size +
   has sampling noise → Mini sometimes > Mavic (inverted). Fix: trace at a FIXED
   reference mesh (reliable, drone-independent) and apply per-drone RCS as a
   deterministic power scaling from **literature dBsm anchors** (`rcs_scale`, now
   carrier-dependent — review fix #4) on the moving echo. Monotonic + faster.
   **Caveat:** RCS is an ESTIMATE (DJI doesn't publish it); the cube is a
   *calibration placeholder*, not a faithful RCS model.

## Review-round fixes (2026-06-26, external code review)
1. **Headline de-confounded (review #1).** Dropped the un-identifiable "carrier
   dominates"; added `--mode decouple` (carrier / bandwidth / reference each varied
   alone). Honest headline = reference sparsity dominates (ΔSCR≈11 dB); carrier
   propagation λ²≈7 dB (net≈5 after C-band RCS rise); bandwidth≈1 dB (negligible).
2. **Pd window in metres (review #2).** `rd_metrics` hit test is now |R−R_gt|≤30 m
   (not ±1 range bin, which was c/B-wide and biased Pd ~20× toward narrow band).
   SCR promoted to primary metric.
3. **compare_drones unified to fix #4 (review #3).** Was sizing meshes per drone
   (non-monotonic, Mini>Air3S); now fixed mesh + deterministic dBsm → monotonic.
4. **RCS = calibration placeholder + carrier-dependent (review #4).** Cube labelled
   placeholder; `rcs_dbsm_at(drone, fc)` adds the literature S→C-band rise (~+8 dB),
   which partly offsets carrier path-loss (further weakening "carrier dominates").
5. **hover/tangential blind = dual cause (review #5).** Attributed to bulk-Doppler
   scope **and** the ideal canceller (both remove the 0-Doppler echo). Degenerate
   cells now report SCR=NaN, marked `degenerate` in the JSON.
6. **waypoint → doppler_switch (review #6).** Renamed; it is a Doppler transition at
   a fixed position (no range-walk), not a trajectory.
7. **Multi-seed N0 anchor (review #7).** Averaged over 5 RT seeds; across-seed spread
   reported in the config block (single-draw would offset every SCR).
8. **Config serialized (review #8).** `phase1_matrix.json` / `phase1_decouple.json`
   carry a `config` block (snr_ref, CPI, trials, Pfa, hit_tol, anchor, …).
9. **N-depth noted (review #9).** Per-cell slow-time depth N (16…610) reported per row.
10. **Hygiene:** s2 `--snr` now reaches the report (review a); Hann uses
    `hanning(N+2)[1:-1]` (review j); CA-CFAR excludes the notch from TRAINING, not
    just post-masking (review j); SSB density comment fixed to ~0.2% (review h);
    shared doppler-switch GT helper (review g); phase1 outdir auto-branches to
    `/data/public/jeong` (review f); report section renumber + drone narrative
    corrected from data (reviews c, d).

## Verified (small runs, pre-matrix)
- Scenario→Doppler exact: hover/tangential ≈ 0 Hz, radial/waypoint = max.
- At anchor (snr_ref=−12): radial Pd=1.0 (SCR ~15 dB); hover/tangential blind (Pd=0).
- Per-drone monotonic: Mavic (RCS 0.81) > Mini (0.32) by ~4 dB.
- Measured tradeoff: LTE 5 MHz@1.8 GHz beats LTE 20 MHz@2.6 GHz — but this compares
  carrier AND bandwidth at once (confounded); see the decoupled sweep for the
  identifiable per-factor contributions (review #1).

## Open issues / caveats
- RCS estimates (Axis B) need measured/CAD values; the carrier-dependence model
  (`rcs_dbsm_at`) is a 2-anchor literature ESTIMATE, clamped outside 2.5–3.5 GHz.
- Ideal static-clutter cancellation (upper bound); data-driven ECA-S is future.
- 5G SSB reference is very sparse-in-time → weak; expected (literature agrees).
- Monte-Carlo N finite; scale up for a tight FAR/CI estimate.
- `doppler_switch` is a mid-CPI Doppler transition at a FIXED position (no range
  migration); a true position-varying trajectory is a refinement.
- Per-cell RT trace is a single draw (anchor is multi-seed); RT spread is small
  (anchor spread reported) but per-cell multi-seed would tighten CIs further.

## Next steps
1. Sanity-check the decoupled-sweep ΔSCR spans + matrix degenerate cells.
2. Re-run report (`build_report.py`) — §0 viz + §1 decoupled headline already wired.
3. Adversarial re-verification that each review item is resolved + numbers consistent.
4. (Later) grow the grid, more trials, data-driven ECA-S, real range-migration
   trajectory, optional rotor micro-Doppler layer.

## File map (see README.md for full)
`phase1.py` (Phase-1) · `passive_radar_s2.py` (S2 primitives + reference-density study)
· `passive_radar_stage1.py` (S1) · `compare_drones.py`+`drones.py` · `experiments.py`
· `build_report.py` → `report.ipynb` · `docs/` · `outputs/` · `assets/`

---
## Changelog
- **2026-06-26 (review-fix round)** — External code review reflected. Engine was
  sound; fixed experiment-design confound + claims. **Headline 'carrier dominates'
  OVERTURNED** via decoupled single-factor sweeps (`--mode decouple`): reference
  structure dominates (ΔSCR≈11 dB) > carrier propagation λ²≈7 dB (net≈5 after C-band
  RCS rise) > bandwidth≈1 dB. Added `visualize_scene.py` (report §0: bistatic/
  open-field geometry, real RT rays). Pd window→metres, compare_drones→fix#4 +
  eff_snr (moving-echo), carrier-dependent RCS, multi-seed N0, config serialization,
  waypoint→doppler_switch, degenerate cells→NaN, +hygiene a–j. Re-ran matrix +
  decouple + compare_drones + s2 (2×4090). See "Review-round fixes" above.
- **2026-06-26** — **Deep full-read of all 21 papers** (7 parallel subagents, exact
  numbers) → `docs/LITERATURE_REVIEW.md`. Key groundings applied: drone **RCS now
  in dBsm** (−16…−24, lit-anchored, `drones.py`); CPI **50→100 ms** (literature
  0.1–0.5 s); rcs_scale uses dBsm anchors. Re-ran matrix (CPI 0.1 s, snr_ref −9) —
  headline at the time read 'carrier dominates' **[later OVERTURNED — see review-fix
  round above; it was a collinear confound]**; 5G-SSB worst; Mavic>Mini; hover/
  tangential blind. Literature **confirms the zero-Doppler/tangential blind zone is real**
  (LIPASE misses on the bistatic contour; ONERA near-zero-Doppler). No paper measured
  drone RCS or drone micro-Doppler → both are genuine gaps the project fills.
- **2026-06-26** — Phase-1 reduced matrix completed (48 cells) + integrated into
  `report.ipynb` (Phase-1 headline, refpar refreshed). Fixed waypoint N-split bug
  (segments must sum to cfg.N). Set CPI=0.05 s + foreground run so it finishes within
  one turn (background runs were dying with the session). Recorded results above.
- **2026-06-26** — Added "실질 운영" practical procedure (work in jeong_sionna; on
  capacity/large-data, stop & ask user to move files to /data — permission blocker).
- **2026-06-26** — Added the ⚠️ path/storage-rules caution at the top (allowed dirs,
  storage policy, permission blocker, staging location) — highest-priority note.
- **2026-06-26** — Built Phase-1 framework (3 axes, measured SCR). Fixed batch-size,
  analytic-GT, Doppler-sign, and per-drone-RCS issues. Launched reduced matrix.
  Created this NOTES.md.
- **2026-06-25** — S1 RD map verified; S2 4-mode benchmark + 21-paper survey +
  adversarial DSP review (7 fixes) + CPI/reference experiments + per-drone compare;
  generated `report.ipynb`; reorganized folder (docs/, removed stale files).
