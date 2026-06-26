# Sionna 패시브 레이더 드론 탐지

Wi-Fi / LTE / 5G 신호를 조명원으로 쓰는 **bistatic(open-field)** 패시브 레이더로 드론을
탐지하고, **신호 스펙 × 드론 RCS × 모션 시나리오**를 공정하게 비교하는 Sionna 벤치마크.

## 👉 먼저 볼 것
**[`report.ipynb`](report.ipynb)** — 그림이 내장(base64)되어 커널 없이 바로 보입니다.
§0 = 실험 구성 시각화(bistatic geometry), §1 = Phase-1 헤드라인(단일변수 분리 sweep).

## 핵심 결과 (요약)
- **시나리오:** radial만 탐지, hover/tangential blind(0-Doppler).
- **신호 스펙(분리 sweep, ΔSCR):** reference 구조가 최대(~11 dB) > 반송파 λ²(~7) > 대역폭(~1, 무관).
  → 옛 "carrier dominates"는 collinear 교란이라 폐기. 5G-SSB가 reference 바닥.
- **기체:** RCS dBsm에 단조(추정치).

## 실행
`conda activate sionna`는 깨져 있어 env python 직접 호출(`docs/ENV_NOTES.md`).
```bash
PY=/home/yunjung/workspace/jeong/miniforge3/envs/sionna/bin/python
CUDA_VISIBLE_DEVICES=0 $PY phase1.py --mode matrix   --trials 24 --snr_ref -9
CUDA_VISIBLE_DEVICES=0 $PY phase1.py --mode decouple --trials 24 --snr_ref -9
CUDA_VISIBLE_DEVICES=0 $PY visualize_scene.py
$PY build_report.py            # report.ipynb 재생성
```
환경: sionna-rt 2.0.1 / mitsuba 3.8 / python 3.11, RTX 4090.

## 구조 / 더 보기
`phase1.py`(헤드라인) · `visualize_scene.py`(§0) · `passive_radar_{stage1,s2}.py` ·
`compare_drones.py`+`drones.py` · `build_report.py` · `outputs/`(결과) · `assets/`(메시).
상세 진행·결정·한계: **`NOTES.md`**. 문헌·S2: `docs/`. 환경/권한: `docs/ENV_NOTES.md`.

> 권한 문제로 코드·결과가 지정 경로(`/workspace/jeong`·`/data/public/jeong`) 대신 여기
> (`/home/yunjung/workspace/jeong_sionna`)에 임시 스테이징됨 — `docs/ENV_NOTES.md`.
