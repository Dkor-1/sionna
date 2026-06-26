#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monostatic ISAC scene + CFR for the LaSen reproduction (5G-26).

LaSen is MONOSTATIC ISAC: the gNB transmits a known 5G-NR downlink X and receives the
echo Y at a co-located antenna; the sensing channel is H = Y / X (LaSen Eq 1). In
Sionna RT, H is the channel CFR directly (paths.cfr) — the moving drone appears as a
path with Doppler f_d = 2 v_radial / lambda, static clutter (ground/building) at ~0 Hz
(this is the self-leakage / 0-Doppler clutter LaSen suppresses in §4.1.1).

We reuse the parent project's scene builder (build_scene) with tx and rx ~co-located
(0.5 m apart) to approximate a monostatic antenna; the drone RCS uses the parent's
literature-grounded dBsm scaling. This is the FAITHFUL monostatic premise (distinct
from the bistatic passive benchmark — see docs/FAITHFULNESS.md).
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):          # lasen/ + parent (reuse primitives)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from passive_radar_stage1 import Config, C0, build_scene
from nr_waveform import NRNumerology


# monostatic geometry: gNB (tx~=rx co-located), moving drone, ground clutter
GEOM = dict(tx=(0.0, 0.0, 12.0), rx=(0.6, 0.0, 12.0), drone=(0.0, 45.0, 30.0))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def analytic_gt(num: NRNumerology, geom, velocity) -> dict:
    """Exact GT from geometry+velocity (monostatic). Bistatic delay is relative to
    the direct tx->rx path (CFR uses normalize_delays); with tx~=rx this ~= 2*range."""
    tx = np.array(geom["tx"], float); rx = np.array(geom["rx"], float)
    p = np.array(geom["drone"], float); v = np.array(velocity, float)
    Rtx, Rrx, L = (np.linalg.norm(p - tx), np.linalg.norm(p - rx), np.linalg.norm(tx - rx))
    Rb = Rtx + Rrx - L                                   # round-trip path (rel. direct), = c*tau
    # The RD range axis maps the round-trip delay tau to the MONOSTATIC TARGET DISTANCE
    # d = c*tau/2 = Rb/2 (~ the true drone distance from the co-located gNB, Rtx).
    range_m = Rb / 2.0
    # Doppler: sign matches Sionna paths / the RD Doppler axis (negate as in parent)
    fD = -float(v @ (_unit(p - tx) + _unit(p - rx))) / num.wavelength
    v_radial = float(v @ _unit(p - tx))                  # ~ radial speed (tx~=rx)
    return dict(range_m=float(range_m), round_trip_m=float(Rb), delay_s=float(Rb / C0),
                doppler_hz=fD, v_radial_ms=v_radial, Rtx=float(Rtx), Rrx=float(Rrx),
                baseline_m=float(L))


def trace_cfr(num: NRNumerology, geom, velocity, *, window_s=0.1, n_slow=256,
              assets=None, drone_size=0.3, samples_per_src=20_000_000, seed=1):
    """Trace the monostatic channel and return the CFR H[n_slow, K_active] over the
    active subcarriers, the slow-time PRF, the baseband freqs, and the analytic GT.

    H[t, f] = channel frequency response at slow-time t, baseband freq f (= Y/X)."""
    import sionna.rt as rt
    assets = assets or os.path.join(os.path.dirname(_HERE), "assets")
    cfg = Config()
    cfg.fc, cfg.B = num.fc, num.bw
    cfg.tx_pos, cfg.rx_pos, cfg.drone_pos = geom["tx"], geom["rx"], geom["drone"]
    cfg.drone_vel = tuple(float(x) for x in velocity)
    cfg.drone_size = drone_size
    cfg.assets_dir = assets
    cfg.samples_per_src = samples_per_src
    cfg.seed = seed

    scene = build_scene(cfg)
    paths = rt.PathSolver()(scene, max_depth=cfg.max_depth, los=True,
                            specular_reflection=True, diffuse_reflection=True,
                            refraction=False, samples_per_src=samples_per_src, seed=seed)
    prf = n_slow / window_s
    freqs = num.baseband_freqs().astype(np.float32)
    H = paths.cfr(frequencies=freqs, sampling_frequency=prf, num_time_steps=n_slow,
                  normalize_delays=True, normalize=False, out_type="numpy")
    H = np.asarray(H)[0, 0, 0, 0, :, :].astype(np.complex64)      # [n_slow, K]
    gt = analytic_gt(num, geom, velocity)
    gt.update(prf_hz=float(prf), window_s=float(window_s), n_slow=int(n_slow),
              doppler_res_hz=float(1.0 / window_s),
              n_paths=int(np.sum(np.isfinite(np.asarray(paths.tau).ravel()))))
    return H, freqs, gt
