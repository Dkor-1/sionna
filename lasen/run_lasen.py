#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaSen faithful-reproduction pipeline (5G-26 SenSys) — phase-gated, viz-first.

Phase A (this file, --phase A): monostatic CFR -> full-band 2D-FFT range-Doppler
SANITY. GATE: with full occupancy, the 2D-FFT RD peak lands on the analytic GT
range/Doppler cell, and after slow-time mean subtraction (LaSen §4.1.1) the 0-Hz
clutter/self-leakage ridge collapses (LaSen Fig.4). If this fails, fix the Doppler
sign / monostatic factor / CFR before adding 2D-OMP (Phase B).

Each phase: emits figures (mapped to LaSen paper figures) + a JSON with the gate
verdict. Outputs -> lasen/outputs/.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nr_waveform import NRNumerology, occupancy_mask
from monostatic_scene import GEOM, trace_cfr
import viz


# --------------------------------------------------------------------------- #
#  Core RD processing (shared by phases)
# --------------------------------------------------------------------------- #
def rd_from_cfr(H, num: NRNumerology, prf, mti=True):
    """2-D range-Doppler from CFR H[n_slow, K]: slow-time mean subtraction (static
    /0-Hz clutter suppression, LaSen §4.1.1) then IFFT over freq (range) + FFT over
    slow-time (Doppler). Returns complex RD [doppler, range] + axes."""
    Ns, K = H.shape
    if mti:
        H = H - H.mean(axis=0, keepdims=True)              # remove 0-Doppler clutter
    wf = np.hanning(K + 2)[1:-1][None, :]
    wt = np.hanning(Ns + 2)[1:-1][:, None]
    rng = np.fft.ifft(H * wf, axis=1)                      # [Ns, K]  delay/range
    rd = np.fft.fftshift(np.fft.fft(rng * wt, axis=0), axes=0)   # [Doppler, range]
    range_axis = np.arange(K) * num.range_res_m
    dopp_axis = np.fft.fftshift(np.fft.fftfreq(Ns, d=1.0 / prf))
    return rd, range_axis, dopp_axis


def peak_cell(rd, range_axis, dopp_axis, notch_hz=20.0):
    """Global peak of |rd|^2 excluding the +-notch zero-Doppler clutter band."""
    p = np.abs(rd) ** 2
    mask = np.abs(dopp_axis) < notch_hz
    pm = p.copy(); pm[mask, :] = -np.inf
    di, ri = np.unravel_index(int(np.argmax(pm)), p.shape)
    return float(range_axis[ri]), float(dopp_axis[di]), ri, di


# --------------------------------------------------------------------------- #
#  Phase A
# --------------------------------------------------------------------------- #
def phase_a(outdir, num, n_slow=256, window_s=0.1, speed=12.0, trials_note="single window"):
    os.makedirs(outdir, exist_ok=True)
    # radial-ish motion (toward +y), drone in front of the gNB
    vel = (0.0, speed, 0.0)
    print(f"[A] monostatic CFR trace: fc={num.fc/1e9} GHz B={num.bw/1e6:.2f} MHz "
          f"K={num.n_active} N={n_slow} window={window_s*1e3:.0f} ms")
    H, freqs, gt = trace_cfr(num, GEOM, vel, window_s=window_s, n_slow=n_slow)
    print(f"[A] GT: range={gt['range_m']:.1f} m (Rtx={gt['Rtx']:.1f},Rrx={gt['Rrx']:.1f}), "
          f"fD={gt['doppler_hz']:.1f} Hz (=2v/λ), v_rad={gt['v_radial_ms']:.1f} m/s, "
          f"PRF={gt['prf_hz']:.0f} Hz, paths={gt['n_paths']}")

    rd_raw, r_ax, d_ax = rd_from_cfr(H, num, gt["prf_hz"], mti=False)
    rd_cln, _, _ = rd_from_cfr(H, num, gt["prf_hz"], mti=True)

    # ---- GATE checks ----
    pr, pd, ri, di = peak_cell(rd_cln, r_ax, d_ax)
    rg = int(np.argmin(np.abs(r_ax - gt["range_m"])))
    dg = int(np.argmin(np.abs(d_ax - gt["doppler_hz"])))
    range_err_m = abs(pr - gt["range_m"]); dopp_err_hz = abs(pd - gt["doppler_hz"])
    peak_on_gt = bool(range_err_m <= 3 * num.range_res_m and dopp_err_hz <= 3 * gt["doppler_res_hz"])
    # 0-Doppler clutter collapse after mean subtraction (energy in the notch band)
    notch = np.abs(d_ax) < 20.0
    clutter_raw = float(np.sum(np.abs(rd_raw[notch, :]) ** 2))
    clutter_cln = float(np.sum(np.abs(rd_cln[notch, :]) ** 2))
    clutter_drop_db = 10 * np.log10(clutter_cln / (clutter_raw + 1e-30))
    clutter_collapses = bool(clutter_drop_db < -10.0)
    # Doppler-velocity faithfulness: measured peak fD vs analytic 2v/λ
    dopp_match = bool(dopp_err_hz <= 3 * gt["doppler_res_hz"])
    gate_pass = peak_on_gt and clutter_collapses

    print(f"[A][gate] peak: R={pr:.1f} m (GT {gt['range_m']:.1f}, err {range_err_m:.1f} m), "
          f"fD={pd:.1f} Hz (GT {gt['doppler_hz']:.1f}, err {dopp_err_hz:.1f} Hz)")
    print(f"[A][gate] peak_on_GT={peak_on_gt}  clutter_drop={clutter_drop_db:.1f} dB "
          f"(collapses={clutter_collapses})  -> GATE {'PASS' if gate_pass else 'FAIL'}")

    # ---- figures (mapped to LaSen figures) ----
    rspan = (max(0, gt["range_m"] - 40), gt["range_m"] + 40)
    dspan = (-max(400, 3*abs(gt["doppler_hz"])), max(400, 3*abs(gt["doppler_hz"])))
    full_grid = np.ones((n_slow, num.n_active), bool)     # Phase A = full occupancy

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    ax1 = fig.add_subplot(2, 3, 1); viz.resource_grid(ax1, full_grid[:60, ::20],
        title="(Fig 2b) NR grid — Phase A: full occupancy")
    ax2 = fig.add_subplot(2, 3, 2); viz.cfr_heatmap(ax2, H, freqs, gt["prf_hz"],
        title="(Fig 3) CFR |H[t,f]| = Y/X")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d"); viz.trajectory3d(ax3, GEOM, vel)
    ax4 = fig.add_subplot(2, 3, 4)
    im = viz.rd_map(ax4, rd_raw, r_ax, d_ax, gt=gt, title="(Fig 4a) RD raw — 0-Hz clutter dominates",
                    r_zoom=rspan, d_zoom=dspan); fig.colorbar(im, ax=ax4, shrink=.8, label="dB")
    ax5 = fig.add_subplot(2, 3, 5)
    im = viz.rd_map(ax5, rd_cln, r_ax, d_ax, gt=gt,
                    title=f"(Fig 4b) RD after static suppression\nclutter −{abs(clutter_drop_db):.0f} dB, peak on GT={peak_on_gt}",
                    r_zoom=rspan, d_zoom=dspan); fig.colorbar(im, ax=ax5, shrink=.8, label="dB")
    ax5.legend(loc="upper right", fontsize=7)
    ax6 = fig.add_subplot(2, 3, 6)
    # Doppler-velocity sanity: analytic line f_d = 2 v / lambda + the measured point
    vv = np.linspace(0, 20, 50)
    ax6.plot(vv, num.doppler_hz(vv), "-", color="0.5", label="analytic 2v/λ")
    ax6.plot(gt["v_radial_ms"], -gt["doppler_hz"], "o", color="tab:red",
             label=f"measured ({-pd:.0f} Hz)")
    ax6.set_xlabel("radial velocity [m/s]"); ax6.set_ylabel("|Doppler| [Hz]")
    ax6.set_title(f"Doppler↔velocity sanity (monostatic)\nf_d match={dopp_match}", fontsize=9)
    ax6.grid(alpha=.3); ax6.legend(fontsize=8)
    fig.suptitle(f"LaSen Phase A — monostatic CFR → full-band RD sanity   |   "
                 f"GATE {'PASS ✓' if gate_pass else 'FAIL ✗'}   |   "
                 f"fc={num.fc/1e9:.1f} GHz, B={num.bw/1e6:.1f} MHz, SCS={num.scs/1e3:.0f} kHz",
                 fontsize=13)
    viz.savefig(fig, os.path.join(outdir, "lasen_phaseA.png"))

    out = dict(phase="A", config=dict(fc_ghz=num.fc/1e9, bw_mhz=num.bw/1e6,
               scs_khz=num.scs/1e3, n_fft=num.n_fft, n_active=num.n_active,
               n_slow=n_slow, window_ms=window_s*1e3, prf_hz=gt["prf_hz"],
               range_res_m=num.range_res_m, speed_ms=speed),
               gt=gt,
               gate=dict(peak_range_m=pr, peak_doppler_hz=pd, range_err_m=range_err_m,
                         doppler_err_hz=dopp_err_hz, peak_on_gt=peak_on_gt,
                         clutter_drop_db=clutter_drop_db, clutter_collapses=clutter_collapses,
                         doppler_velocity_match=dopp_match, gate_pass=gate_pass))
    json.dump(out, open(os.path.join(outdir, "lasen_phaseA.json"), "w"), indent=1)
    print(f"[A][out] {outdir}/lasen_phaseA.png + lasen_phaseA.json")
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="A", choices=["A"])
    ap.add_argument("--outdir", default=os.path.join(_HERE, "outputs"))
    ap.add_argument("--n-slow", type=int, default=256)
    ap.add_argument("--window", type=float, default=0.1)
    ap.add_argument("--speed", type=float, default=12.0)
    ap.add_argument("--fc", type=float, default=5.8e9)
    a = ap.parse_args()
    num = NRNumerology(fc=a.fc)
    if a.phase == "A":
        phase_a(a.outdir, num, n_slow=a.n_slow, window_s=a.window, speed=a.speed)
