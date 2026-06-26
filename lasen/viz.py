#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared visualization helpers for the LaSen reproduction (viz-first principle:
every module emits a figure; figures map to LaSen paper figures). Common style in
one place so all phases look consistent.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD_CMAP = "viridis"


def db(x, ref=None, floor_db=-60.0):
    p = np.abs(x) ** 2
    ref = p.max() if ref is None else ref
    return 10 * np.log10(np.maximum(p, ref * 10 ** (floor_db / 10)) / (ref + 1e-30))


def resource_grid(ax, mask, title="NR resource grid (occupancy)"):
    """Occupancy/DMRS grid: time (OFDM symbol) x subcarrier. mask True = transmitted RE."""
    ax.imshow(mask.T, aspect="auto", origin="lower", cmap="Greys", interpolation="nearest")
    ax.set_xlabel("OFDM symbol (slow time)"); ax.set_ylabel("subcarrier")
    ax.set_title(f"{title}\noccupancy = {mask.mean()*100:.2f}% of grid", fontsize=9)


def cfr_heatmap(ax, H, freqs, prf, title="CFR |H[t,f]|  (= Y/X)"):
    """|CFR| over slow-time x baseband frequency."""
    ext = [freqs[0]/1e6, freqs[-1]/1e6, 0, H.shape[0]/prf*1e3]
    ax.imshow(20*np.log10(np.abs(H)+1e-9), aspect="auto", origin="lower", extent=ext,
              cmap="magma")
    ax.set_xlabel("baseband frequency [MHz]"); ax.set_ylabel("slow time [ms]")
    ax.set_title(title, fontsize=9)


def rd_map(ax, rd, range_axis, dopp_axis, gt=None, det=None, title="Range-Doppler",
           vmin=-40, vmax=0, r_zoom=None, d_zoom=None):
    """RD power map (dB, own-max normalised) with optional GT marker + detections."""
    ext = [dopp_axis[0], dopp_axis[-1], range_axis[0], range_axis[-1]]
    im = ax.imshow(db(rd), aspect="auto", origin="lower", extent=ext,
                   cmap=RD_CMAP, vmin=vmin, vmax=vmax)
    if gt is not None:
        ax.plot(gt["doppler_hz"], gt["range_m"], "o", mfc="none", mec="red", ms=14, mew=1.6,
                label="GT (analytic)")
    if det is not None and np.any(det):
        rr, cc = np.where(det)
        ax.plot(dopp_axis[cc], range_axis[rr], "x", color="red", ms=6, mew=1.2, label="detection")
    if r_zoom: ax.set_ylim(*r_zoom)
    if d_zoom: ax.set_xlim(*d_zoom)
    ax.set_xlabel("Doppler [Hz]"); ax.set_ylabel("range [m]  (monostatic)")
    ax.set_title(title, fontsize=9)
    return im


def trajectory3d(ax, geom, vel=None, title="Monostatic geometry"):
    tx = np.array(geom["tx"]); rx = np.array(geom["rx"]); p = np.array(geom["drone"])
    g = 60
    xx, yy = np.meshgrid([-g, g], [-10, 90])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.10, color="gray")
    ax.scatter(*tx, c="tab:red", s=90, marker="^", depthshade=False)
    ax.scatter(*p, c="k", s=70, marker="o", depthshade=False)
    ax.text(*tx, "  gNB (tx≈rx, monostatic)", color="tab:red", fontsize=9)
    ax.text(*p, "  drone", color="k", fontsize=9)
    ax.plot([tx[0], p[0]], [tx[1], p[1]], [tx[2], p[2]], color="tab:green", lw=1.8)
    ax.plot([p[0], rx[0]], [p[1], rx[1]], [p[2], rx[2]], color="tab:green", lw=1.0, ls="--")
    if vel is not None:
        v = np.array(vel, float) * 1.5
        ax.plot([p[0], p[0]+v[0]], [p[1], p[1]+v[1]], [p[2], p[2]+v[2]], color="tab:blue", lw=2)
        ax.text(*(p+v), " v", color="tab:blue", fontsize=9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=22, azim=-60); ax.set_box_aspect((1, 1.2, 0.5))
    ax.set_title(title, fontsize=9)


def savefig(fig, path, dpi=140):
    fig.savefig(path, dpi=dpi); plt.close(fig)
    print("[fig]", path)
