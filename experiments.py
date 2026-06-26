#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance experiments for the passive-radar benchmark — "why does it look
weak, and what moves the needle?"

E1  Integration-time (CPI) sweep:  Pd / SCR vs CPI length (number of slow-time
    batches N).  Longer coherent integration = more processing gain; this is the
    primary real-world lever and is exactly what the 5G literature uses
    (adaptive / multi-block integration) to rescue sparse references.

E2  Reference paradigm:  pilots-only reconstruction (realistic, reference-
    structure-limited) vs a clean reference-antenna capture (full transmitted
    signal known -> upper bound, reference structure irrelevant).  Quantifies
    how much of the "weak" performance is the *cost of not knowing the data*.

    python experiments.py --exp cpi    --trials 16 --snr -30
    python experiments.py --exp refpar --trials 16 --snr -30
    python experiments.py --exp both   --trials 16
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from passive_radar_stage1 import Config
from passive_radar_s2 import OFDM, MODES, SHORT, trace_once, run_mode


def exp_cpi(cfg: Config, ofdm: OFDM, Ns, snr_db, trials):
    """Pd / SCR vs CPI (vary N batches; M fixed)."""
    rows = {}
    for N in Ns:
        cfg.N = N
        h, gt, tap = trace_once(cfg)
        cpi_ms = N * cfg.M / cfg.fs * 1e3
        for mode in MODES:
            res, _ = run_mode(cfg, ofdm, mode, h, gt, tap, snr_db, trials,
                              base_seed=cfg.seed)
            rows[(mode, N)] = dict(pd=res["pd"], scr=res["scr_db_mean"], cpi_ms=cpi_ms)
            print(f"[cpi] N={N:4d} CPI={cpi_ms:6.1f}ms {SHORT[mode]:8s} "
                  f"Pd={res['pd']:.2f} SCR={res['scr_db_mean']:5.1f} dB")
    _plot_cpi(Ns, rows, snr_db, os.path.join(cfg.outdir, "exp_cpi.png"))
    json.dump({f"{m}|{N}": rows[(m, N)] for (m, N) in rows},
              open(os.path.join(cfg.outdir, "exp_cpi.json"), "w"), indent=2)


def _plot_cpi(Ns, rows, snr_db, path):
    cpis = [rows[(MODES[0], N)]["cpi_ms"] for N in Ns]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for mode in MODES:
        ax[0].plot(cpis, [rows[(mode, N)]["pd"] for N in Ns], marker="o", label=SHORT[mode])
        ax[1].plot(cpis, [rows[(mode, N)]["scr"] for N in Ns], marker="o", label=SHORT[mode])
    ax[0].set_xlabel("CPI [ms]"); ax[0].set_ylabel("Pd"); ax[0].set_ylim(-.03, 1.03)
    ax[0].set_title("Detection vs integration time"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("CPI [ms]"); ax[1].set_ylabel("SCR [dB]")
    ax[1].set_title("SCR vs integration time"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle(f"E1 Integration-time sweep (per-sample SNR {snr_db:.0f} dB) — "
                 f"longer CPI recovers sparse-reference modes", fontsize=12)
    fig.savefig(path, dpi=140); plt.close(fig)


def exp_refpar(cfg: Config, ofdm: OFDM, snr_db, trials):
    """pilots-only vs clean reference-antenna, per mode."""
    h, gt, tap = trace_once(cfg)
    out = {}
    for mode in MODES:
        for full in (False, True):
            res, _ = run_mode(cfg, ofdm, mode, h, gt, tap, snr_db, trials,
                              base_seed=cfg.seed, full_ref=full)
            out[(mode, full)] = dict(pd=res["pd"], scr=res["scr_db_mean"])
            tag = "full-ref" if full else "pilots"
            print(f"[refpar] {SHORT[mode]:8s} {tag:9s} Pd={res['pd']:.2f} "
                  f"SCR={res['scr_db_mean']:5.1f} dB")
    _plot_refpar(out, snr_db, os.path.join(cfg.outdir, "exp_refpar.png"))
    json.dump({f"{m}|{'full' if f else 'pilots'}": out[(m, f)] for (m, f) in out},
              open(os.path.join(cfg.outdir, "exp_refpar.json"), "w"), indent=2)


def _plot_refpar(out, snr_db, path):
    x = np.arange(len(MODES)); w = 0.38
    pil_scr = [out[(m, False)]["scr"] for m in MODES]
    full_scr = [out[(m, True)]["scr"] for m in MODES]
    pil_pd = [out[(m, False)]["pd"] for m in MODES]
    full_pd = [out[(m, True)]["pd"] for m in MODES]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax[0].bar(x - w/2, pil_scr, w, label="pilots-only (realistic)", color="indianred")
    ax[0].bar(x + w/2, full_scr, w, label="full reference (upper bound)", color="seagreen")
    ax[0].set_xticks(x); ax[0].set_xticklabels([SHORT[m] for m in MODES])
    ax[0].set_ylabel("SCR [dB]"); ax[0].set_title("SCR: cost of pilot-only reconstruction")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(x - w/2, pil_pd, w, label="pilots-only", color="indianred")
    ax[1].bar(x + w/2, full_pd, w, label="full reference", color="seagreen")
    ax[1].set_xticks(x); ax[1].set_xticklabels([SHORT[m] for m in MODES])
    ax[1].set_ylabel("Pd"); ax[1].set_ylim(0, 1.05); ax[1].set_title("Detection rate")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"E2 Reference paradigm (per-sample SNR {snr_db:.0f} dB) — a clean "
                 f"reference closes the band gap; the structure axis lives in pilots-only",
                 fontsize=11)
    fig.savefig(path, dpi=140); plt.close(fig)


def parse_args():
    cfg = Config()
    default_out = os.environ.get(
        "PR_OUTDIR",
        "/data/public/jeong/sionna/exp"
        if os.access("/data/public/jeong", os.W_OK)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="both", choices=["cpi", "refpar", "both"])
    p.add_argument("--outdir", default=default_out)
    p.add_argument("--assets", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets"))
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--snr", type=float, default=-30.0)
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--nfft", type=int, default=4096)
    p.add_argument("--cp", type=int, default=512)
    a = p.parse_args()
    cfg.outdir, cfg.assets_dir, cfg.M = a.outdir, a.assets, a.M
    return cfg, OFDM(n_fft=a.nfft, cp=a.cp), a.exp, a.trials, a.snr


if __name__ == "__main__":
    cfg, ofdm, exp, trials, snr = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)
    if exp in ("cpi", "both"):
        exp_cpi(cfg, ofdm, [128, 256, 512, 1024, 2048], snr, trials)
    if exp in ("refpar", "both"):
        exp_refpar(cfg, ofdm, snr, trials)
