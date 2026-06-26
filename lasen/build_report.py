#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build lasen/report.ipynb from the phase outputs (figures embedded base64 -> opens
without a kernel). Grows as phases land; re-run after each phase.
    python lasen/build_report.py
"""
import os, json, base64

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
NB = os.path.join(HERE, "report.ipynb")


def img_md(png, title=""):
    p = os.path.join(OUT, png)
    if not os.path.exists(p):
        return f"_({png} — 아직 생성 안 됨)_"
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f"![{title}](data:image/png;base64,{b64})"


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


cells = [md(
    "# LaSen 충실 재현 — 리포트 (5G-26, monostatic ISAC)",
    "",
    "*자동 생성: `lasen/build_report.py`. 그림은 base64 내장 — 커널 없이 바로 보입니다.*",
    "",
    "**원칙:** 충실(논문 eq/param) · 단계 게이트(각 phase가 figure로 자기증명 후 다음) · "
    "viz-first. **monostatic ISAC**(gNB tx≈rx, CFR H=Y/X) — 부모 bistatic 벤치마크와 별개. "
    "충실/비충실 정직 구분: `docs/FAITHFULNESS.md`.",
)]

# ---- Phase A (R3: per-panel + commentary) ----
a = load("lasen_phaseA.json")
if a:
    g = a["gate"]; c = a["config"]
    verdict = "✅ PASS" if g["gate_pass"] else "❌ FAIL"
    cells.append(md(
        f"## Phase A — monostatic CFR → full-band 2D-FFT RD (sanity)   ·   GATE {verdict}",
        "",
        f"gNB가 알려진 5G-NR X를 송신·수신 → **CFR H=Y/X**(Eq 1). Sionna로 tx≈rx 씬 + 이동 드론 → CFR. "
        f"slow-time 평균빼기로 **0-Hz 클러터/self-leakage 제거**(§4.1.1) → 2D-FFT RD맵.",
        "",
        f"설정: fc={c['fc_ghz']:.1f} GHz · B={c['bw_mhz']:.2f} MHz · SCS={c['scs_khz']:.0f} kHz · FFT={c['n_fft']} · "
        f"active={c['n_active']} · **N(slow)={c['n_slow']} 실제 OFDM 심볼(R1)** · window={c['window_ms']:.0f} ms · "
        f"PRF={c['prf_hz']:.0f} Hz · dopp_res={c['doppler_res_hz']:.0f} Hz.",
    ))
    cells.append(md(
        "### ① monostatic 기하 + ② NR 자원격자",
        img_md("A_geometry.png", "geometry"), img_md("A_grid.png", "NR grid"),
        "*왜:* gNB(tx≈rx)가 송·수신, 드론이 산란 → monostatic. **range=c·τ/2**(왕복지연→타깃거리), "
        "**Doppler=2v/λ**. Phase A는 **full occupancy**(모든 active RE 송신) — 비균일 점유 마스크는 Phase B.",
    ))
    cells.append(md(
        "### ③ CFR |H[t,f]| = Y/X  (Fig 3)",
        img_md("A_cfr.png", "CFR"),
        f"*왜:* 채널 주파수응답이 곧 H=Y/X(gNB는 X를 앎). 주파수축 줄무늬 = 지연(거리) 구조, "
        f"slow-time축 = **실제 OFDM 심볼 {c['n_slow']}개**(R1 — 점유 마스크·2D-OMP 사전이 이 위에 놓임).",
    ))
    cells.append(md(
        "### ④ RD: 정적억제 전 → 후  (Fig 4)",
        img_md("A_rd_raw.png", "RD raw"), img_md("A_rd_clean.png", "RD clean"),
        f"*왜:* 억제 전엔 **0-Hz 클러터/self-leakage가 지배**(드론 묻힘). slow-time 평균빼기 후 "
        f"**0-Hz {g['clutter_drop_db']:.0f} dB 붕괴**({'collapses' if g['clutter_collapses'] else 'no'}) → "
        f"드론이 **GT 셀에 단일 피크**(range err {g['range_err_m']:.1f} m, fD err {g['doppler_err_hz']:.1f} Hz, "
        f"on_GT={g['peak_on_gt']}). LaSen Fig 4 재현.",
    ))
    cells.append(md(
        "### ⑤ Doppler↔velocity sweep  (R2 — 점 1개가 아니라 sweep)",
        img_md("A_doppler_sweep.png", "doppler sweep"),
        f"*왜:* 속도 4~20 m/s에서 측정 RD-peak Doppler가 해석적 **2v/λ** 선을 따라감 "
        f"(max err **{g['sweep_max_err_hz']:.0f} Hz**, pass={g['sweep_pass']}). monostatic Doppler 법칙의 "
        f"충실성을 1점이 아닌 추세로 증명.",
        "",
        f"→ **GATE {verdict}** (R1 실심볼 + R2 sweep + peak-on-GT + 클러터붕괴 모두 통과). 다음 = "
        f"**Phase B**(비균일 점유 + 2D-OMP).",
    ))

# ---- Phase B ----
b = load("lasen_phaseB.json")
if b:
    g = b["gate"]; c = b["config"]; rs = b["results"]
    verdict = "✅ PASS" if g["gate_pass"] else "❌ FAIL"
    sp, dn = rs["sparse"], rs["dense"]
    cells.append(md(
        f"## Phase B — 비균일 점유 + 2D-OMP (핵심 novelty)   ·   GATE {verdict}",
        "",
        "실제 5G 트래픽은 **비균일 점유**(전송된 RE만 관측) → masked CFR **ĥ = W∘H = Φz**(Eq 4-5). "
        "plain 2D-FFT는 미전송 RE를 0으로 두고 변환 → **sub-Nyquist 누설 바닥**(점유가 sparse할수록 악화); "
        "**2D-OMP**는 masked 관측을 sparse 모델로 적합(Eq 6) → **깨끗한 RD**(큰 동적범위). "
        f"점유밀도는 논문 측정 bin(sparse {sp['realised']*100:.1f}% / dense {dn['realised']*100:.1f}%).",
        "",
        f"- **omp2d round-trip sub-게이트**: {'PASS' if g['roundtrip_ok'] else 'FAIL'} "
        "(rd_transform(atom(di,ri))이 정확히 (di,ri) 셀에 피크 — 부호·규약 일치).",
        f"- **게이트:** OMP가 양 밀도에서 표적 복원({'예' if g['omp_recovers'] else '아니오'}); "
        f"OMP 동적범위 ≫ FFT(sparse ΔPSLR={sp['omp_pslr_db']-sp['fft_pslr_db']:.0f} dB); "
        f"FFT 누설 바닥이 sparse에서 악화({sp['fft_pslr_db']:.0f} < dense {dn['fft_pslr_db']:.0f} dB).",
        "",
        "### 점유 마스크 W (sparse vs dense) + 밀도 timeline",
        img_md("B_occupancy_sparse.png", "occ sparse"), img_md("B_occupancy_dense.png", "occ dense"),
        img_md("B_density_timeline.png", "density timeline"),
        "*왜:* DMRS comb은 항상, PDSCH 데이터는 트래픽 따라 — sparse일수록 관측 RE 급감(Fig 3a/11).",
        "",
        "### RD 비교: 2D-FFT(누설) vs 2D-OMP(깨끗)  — money figure (Tab.1)",
        img_md("B_rd_compare.png", "RD compare"),
        f"*왜:* 단일 표적은 OMP 1st atom = FFT 피크라 **피크 위치는 같다**(둘 다 on_GT). 차이는 **동적범위** — "
        f"FFT는 sparse에서 누설바닥 PSLR {sp['fft_pslr_db']:.0f} dB(약한 표적이 묻힘), OMP는 {sp['omp_pslr_db']:.0f} dB"
        "(노이즈 바닥까지 깨끗). 이게 LaSen sparse recovery가 FFT를 이기는 이유.",
        "",
        "### 2D-OMP 수렴 (잔차 norm vs iter)",
        img_md("B_omp_convergence.png", "omp convergence"),
        f"*왜:* OMP가 강한 표적을 먼저 잡고 잔차를 줄임(atoms sparse {sp['n_atoms']} / dense {dn['n_atoms']}). "
        "RT 드론은 완전 단일 atom이 아니라(다중경로 spread) 몇 개 atom으로 적합됨.",
        "",
        f"→ **GATE {verdict}.** 다음 = Phase C(ID score + global/local + Kalman 추적).",
        "",
        "_정직 노트: 단일 표적이라 'OMP만 표적을 본다'는 못 보임(둘 다 본다) — 차이는 **누설바닥/동적범위**. "
        "약한-2nd-표적 동적범위 데모는 RT 드론 spread에 가려 후속 과제. slow-time는 OMP 가용성 위해 "
        f"{c['n_slow']} 심볼로 subsample(Phase A는 full {c['n_symbols_full']}). 상세: `docs/FAITHFULNESS.md`._",
    ))

# ---- (Phase C/D append here as they land) ----

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(NB, "w"), ensure_ascii=False, indent=1)
print(f"wrote {NB} ({os.path.getsize(NB)/1e6:.2f} MB, {len(cells)} cells)")
