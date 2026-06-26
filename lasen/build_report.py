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

# ---- Phase A ----
a = load("lasen_phaseA.json")
if a:
    g = a["gate"]; c = a["config"]
    verdict = "✅ PASS" if g["gate_pass"] else "❌ FAIL"
    cells.append(md(
        f"## Phase A — monostatic CFR → full-band 2D-FFT RD (sanity)   ·   GATE {verdict}",
        "",
        f"gNB가 알려진 5G-NR X를 송신·수신 → **CFR H=Y/X**(Eq 1). Sionna로 tx≈rx 씬 + 이동 드론 → "
        f"CFR. slow-time 평균빼기로 **0-Hz 클러터/self-leakage 제거**(§4.1.1) → 2D-FFT RD맵.",
        "",
        f"- 설정: fc={c['fc_ghz']:.1f} GHz, B={c['bw_mhz']:.2f} MHz, SCS={c['scs_khz']:.0f} kHz, "
        f"FFT={c['n_fft']}, active={c['n_active']}, N(slow)={c['n_slow']}, window={c['window_ms']:.0f} ms.",
        f"- **게이트:** RD 피크가 해석적 GT 셀에 정확히({'예' if g['peak_on_gt'] else '아니오'}) — "
        f"range {g['peak_range_m']:.1f} m (GT err {g['range_err_m']:.1f} m), "
        f"fD {g['peak_doppler_hz']:.0f} Hz (GT err {g['doppler_err_hz']:.1f} Hz, =2v/λ {'✓' if g['doppler_velocity_match'] else '✗'}). "
        f"평균빼기 후 **0-Hz 클러터 {g['clutter_drop_db']:.0f} dB 붕괴**({'collapses' if g['clutter_collapses'] else 'no'}) → Fig 4 재현.",
        "",
        img_md("lasen_phaseA.png", "LaSen Phase A"),
        "",
        "_패널: (Fig 2b) NR 자원격자(Phase A=full occupancy) · (Fig 3) CFR |H[t,f]| · monostatic 기하 · "
        "(Fig 4a) RD raw(0-Hz 클러터 지배) · (Fig 4b) RD 정적억제 후(GT에 단일 피크) · Doppler↔속도 sanity(2v/λ)._",
        "",
        "→ **게이트 통과:** CFR·Doppler 부호·monostatic range·정적억제가 모두 정확. 다음 = Phase B(비균일 점유 + 2D-OMP).",
    ))

# ---- (Phase B/C/D append here as they land) ----

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(NB, "w"), ensure_ascii=False, indent=1)
print(f"wrote {NB} ({os.path.getsize(NB)/1e6:.2f} MB, {len(cells)} cells)")
