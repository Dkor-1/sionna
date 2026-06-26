#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-drone detectability comparison for the candidate procurement airframes.

For each drone (DJI Mini 5 Pro / Air 3S / Mavic 4 Pro / Matrice 4E) the per-drone
RCS (deterministic dBsm anchors in `drones.py`) and max speed (Doppler, datasheet)
drive detectability. A FIXED absolute noise floor (set by a reference drone at a
chosen SNR) is shared across all drones, so a larger-RCS drone gives a higher
effective SNR and is easier to detect. We run all 4 signal modes -> (drone x mode)
Pd / SCR table.

RCS METHOD (phase1 fix #4 — review fix #3): the OLD version sized a metal cube per
drone and read RCS off the noisy RT scattering, which was NON-monotonic (a smaller
mesh sometimes scattered MORE -> Mini > Air3S). We now trace a SINGLE fixed
reference mesh and apply the per-drone RCS as a DETERMINISTIC, carrier-dependent
power scaling (`phase1.rcs_scale`), so eff-SNR is monotonic in dBsm by construction.
All RCS numbers are ESTIMATES (DJI doesn't publish RCS — see drones.py).

    python compare_drones.py --snr_ref -23 --ref mavic4pro --trials 16
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from passive_radar_stage1 import Config
from passive_radar_s2 import OFDM, MODES, SHORT, trace_once, run_mode
from drones import DRONES, Drone
from phase1 import REF_SIZE, rcs_scale          # fixed mesh + deterministic RCS (fix #4)


def _set_drone_motion(cfg: Config, d: Drone):
    cfg.drone_size = REF_SIZE                       # FIXED reference mesh (fix #4)
    cfg.drone_vel = (0.0, d.max_speed_ms, 0.0)      # per-drone Doppler (datasheet speed)


def _scale_echo(h: np.ndarray, scale: float) -> np.ndarray:
    """Apply RCS as a deterministic power scaling on the MOVING echo only (the
    static clutter is drone-independent) — identical convention to phase1.run_cell."""
    if scale == 1.0:
        return h
    hs = h.mean(axis=0, keepdims=True)
    return (hs + np.sqrt(scale) * (h - hs)).astype(np.complex64)


def _move_pow(h: np.ndarray, tap: int) -> float:
    """Power of the MOVING (drone) echo at a tap = the slow-time-varying part.
    The static component at the tap is clutter, not the drone, so eff-SNR must be
    referenced to this (else the per-drone spread collapses)."""
    hs = h.mean(axis=0, keepdims=True)
    return float(np.mean(np.abs((h - hs)[:, tap]) ** 2))


def run(cfg: Config, ofdm: OFDM, ref_key: str, snr_ref: float, trials: int):
    os.makedirs(cfg.outdir, exist_ok=True)
    cfg.samples_per_src = 20_000_000               # dense RT (fixed mesh, reliable hit)
    order = ["mini5pro", "air3s", "mavic4pro", "matrice4e"]
    ref = DRONES[ref_key]

    # 1) reference drone fixes the absolute noise floor (fixed mesh + det. RCS)
    _set_drone_motion(cfg, ref)
    h_ref, gt_ref, tap_ref = trace_once(cfg)
    p_ref = _move_pow(h_ref, tap_ref) * rcs_scale(ref, cfg.fc)   # MOVING echo (drone)
    noise_pow = p_ref / (10 ** (snr_ref / 10.0))
    print(f"[ref] {ref.name}: p_move={p_ref:.3e} -> noise floor fixed at "
          f"SNR_ref={snr_ref} dB (RCS={ref.rcs_dbsm:.0f} dBsm @{cfg.fc/1e9:.1f}GHz)")

    pd = np.zeros((len(order), len(MODES)))
    scr = np.zeros_like(pd)
    eff_snr = {}
    for i, key in enumerate(order):
        d = DRONES[key]
        _set_drone_motion(cfg, d)
        h, gt, tap = trace_once(cfg)
        h = _scale_echo(h, rcs_scale(d, cfg.fc))            # deterministic per-drone RCS
        p_echo = _move_pow(h, tap)                          # MOVING echo -> eff-SNR ∝ RCS
        eff_snr[key] = 10 * np.log10(p_echo / noise_pow)
        print(f"[drone] {d.name:16s} RCS={d.rcs_dbsm:+.0f} dBsm speed={d.max_speed_ms} m/s "
              f"fD={gt['doppler_hz']:+.0f} Hz  effSNR={eff_snr[key]:+.1f} dB")
        for j, mode in enumerate(MODES):
            res, _ = run_mode(cfg, ofdm, mode, h, gt, tap, snr_ref, trials,
                              base_seed=cfg.seed, noise_pow=noise_pow)
            pd[i, j] = res["pd"]; scr[i, j] = res["scr_db_mean"]
            print(f"        {SHORT[mode]:8s} Pd={res['pd']:.2f}  SCR={res['scr_db_mean']:5.1f} dB")

    _plot(order, pd, scr, eff_snr, snr_ref, ref_key, cfg.outdir)
    json.dump({"order": order, "modes": MODES, "pd": pd.tolist(),
               "scr": scr.tolist(), "eff_snr_db": eff_snr,
               "snr_ref": snr_ref, "ref": ref_key,
               "drones": {k: vars(v) for k, v in DRONES.items()}},
              open(os.path.join(cfg.outdir, "drones_results.json"), "w"), indent=2)
    print(f"[out] {os.path.join(cfg.outdir, 'drones_compare.png')}")


def _plot(order, pd, scr, eff_snr, snr_ref, ref_key, outdir):
    rows = [f"{DRONES[k].name}\n(eff SNR {eff_snr[k]:+.1f} dB)" for k in order]
    cols = [SHORT[m] for m in MODES]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for a, data, ttl, cmap, fmt in (
            (ax[0], pd, "Detection rate  Pd", "RdYlGn", "{:.2f}"),
            (ax[1], scr, "SCR  [dB]", "viridis", "{:.0f}")):
        im = a.imshow(data, aspect="auto", cmap=cmap,
                      vmin=(0 if ttl.startswith("Det") else None),
                      vmax=(1 if ttl.startswith("Det") else None))
        a.set_xticks(range(len(cols))); a.set_xticklabels(cols)
        a.set_yticks(range(len(rows))); a.set_yticklabels(rows, fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                a.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                       color="black", fontsize=9)
        a.set_title(ttl); fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle(
        f"Per-drone detectability x signal mode  (noise floor fixed at "
        f"{DRONES[ref_key].name} @ {snr_ref} dB; RCS = deterministic dBsm estimate, fix #4)",
        fontsize=11)
    fig.savefig(os.path.join(outdir, "drones_compare.png"), dpi=140)
    plt.close(fig)


def parse_args():
    cfg = Config()
    default_out = os.environ.get(
        "PR_OUTDIR",
        "/data/public/jeong/sionna/drones"
        if os.access("/data/public/jeong", os.W_OK)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=default_out)
    p.add_argument("--assets", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets"))
    p.add_argument("--ref", default="mavic4pro", choices=list(DRONES))
    p.add_argument("--snr_ref", type=float, default=-23.0)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--N", type=int, default=cfg.N)
    p.add_argument("--M", type=int, default=cfg.M)
    p.add_argument("--nfft", type=int, default=4096)
    p.add_argument("--cp", type=int, default=512)
    a = p.parse_args()
    cfg.outdir, cfg.assets_dir = a.outdir, a.assets
    cfg.N, cfg.M = a.N, a.M
    return cfg, OFDM(n_fft=a.nfft, cp=a.cp), a.ref, a.snr_ref, a.trials


if __name__ == "__main__":
    cfg, ofdm, ref, snr_ref, trials = parse_args()
    run(cfg, ofdm, ref, snr_ref, trials)
