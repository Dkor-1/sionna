#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build renyi/report.ipynb from the phase outputs (figures embedded base64 -> opens
without a kernel). Grows as phases land; re-run after each phase.
    python3 renyi/build_report.py
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


def verdict(j):
    if not j:
        return "_(아직 실행 안 됨)_"
    g = j.get("gate_pass")
    tag = {True: "✅ PASS", False: "❌ FAIL", None: "⏳ 예정"}.get(g, str(g))
    return f"**Gate: {tag}** — {j.get('note', j.get('status', ''))}"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


cells = [md(
    "# 5G-22 충실 재현 — 리포트 (Rényi-entropy adaptive integration)",
    "",
    "*자동 생성: `renyi/build_report.py`. 그림은 base64 내장 — 커널 없이 바로 보입니다.*",
    "",
    "**원칙:** 충실(논문 eq/param) · 단계 게이트(각 phase가 figure로 자기증명 후 다음) · "
    "viz-first. 논문 Sec 6이 그 자체로 합성 시뮬이라 **Phase A–C는 RT 없이 로컬 재현**, "
    "Phase D(Sec 7 실비행)만 Sionna RT. 충실/비충실 정직 구분: `docs/FAITHFULNESS.md`.",
)]

A, B, C, D = load("phaseA.json"), load("phaseB.json"), load("phaseC.json"), load("phaseD.json")

cells += [
    md("## Phase A — content-dependency (논문 Fig 8)", "", verdict(A), "",
       "콘텐츠(자원 점유)가 늘수록 CAF 표적 SCR↑ → 저콘텐츠 묻힘, 고콘텐츠 검출. "
       "전력 vs 콘텐츠 분리는 Phase B에서 정량화.", "", img_md("phaseA_content.png", "Phase A")),
    md("## Phase B — entropy가 power·B_eff를 이김 (논문 Fig 9/11/13)", "", verdict(B), "",
       "Rényi entropy는 fill에 단조·SNR robust; power(Fig 10)와 B_eff(Fig 11)는 "
       "콘텐츠를 구분 못함(동일 전력이라도 entropy는 콘텐츠로 분리).", "",
       img_md("phaseB_metrics.png", "Phase B")),
    md("## Phase C — adaptive integration → P_d (논문 Fig 14/15-17)", "", verdict(C), "",
       "fill이 클수록 P_d↑ (Pfa {1e-4,1e-6,1e-8}). 고-entropy(밀집) 프레임 선택 = "
       "유효 T_int↑ → 검출 범위↑ (range eq Fig 14).", "",
       img_md("phaseC_pd.png", "P_d vs filling"), "", img_md("phaseC_fig14_range.png", "Fig 14")),
    md("## Phase D — 실비행 bistatic (Sionna RT, 서버) (논문 Fig 21-23)", "", verdict(D), "",
       "서버에서 parent build_scene/trace_channel로 실제 드론 echo → 같은 CAF+CFAR+entropy "
       "선택 → Sionna GT에 검출 오버레이; T_int 20→100 ms 속도해상도 개선."),
]

nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(NB, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {NB}  ({len(cells)} cells)")
