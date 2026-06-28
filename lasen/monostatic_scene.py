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

# Phase-B two-target geometry: a STRONG near drone + a WEAK distant drone (different
# range AND Doppler cells). The weak echo is set deterministically `rcs_gap_db` below
# the strong (RCS + range path-loss, parent fix #4 philosophy) — not RT draw noise.
GEOM_STRONG = dict(tx=(0.0, 0.0, 12.0), rx=(0.6, 0.0, 12.0), drone=(0.0, 30.0, 22.0))
GEOM_WEAK = dict(tx=(0.0, 0.0, 12.0), rx=(0.6, 0.0, 12.0), drone=(0.0, 90.0, 45.0))


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


def trace_cfr(num: NRNumerology, geom, velocity, *, window_s=0.1, n_slow=None,
              assets=None, drone_size=0.3, samples_per_src=20_000_000, seed=1):
    """Trace the monostatic channel and return the CFR H[n_slow, K_active] over the
    active subcarriers, the slow-time PRF, the baseband freqs, and the analytic GT.

    H[t, f] = channel frequency response at slow-time t, baseband freq f (= Y/X).
    R1 (faithful): the slow-time axis is REAL OFDM symbols — PRF = num.symbol_rate(),
    and n_slow defaults to num.n_symbols(window_s) (~2803 @100 ms). A smaller n_slow
    sub-samples the symbol stream (covers n_slow/PRF s) — flag it if used."""
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
    n_full = num.n_symbols(window_s)
    subsampled = n_slow is not None and n_slow < n_full
    if n_slow is None:
        n_slow = n_full
    prf = num.symbol_rate                                  # R1: slow-time = symbol rate
    freqs = num.baseband_freqs().astype(np.float32)
    H = paths.cfr(frequencies=freqs, sampling_frequency=prf, num_time_steps=n_slow,
                  normalize_delays=True, normalize=False, out_type="numpy")
    H = np.asarray(H)[0, 0, 0, 0, :, :].astype(np.complex64)      # [n_slow, K]
    gt = analytic_gt(num, geom, velocity)
    gt.update(prf_hz=float(prf), window_s=float(n_slow / prf), n_slow=int(n_slow),
              n_symbols_full=int(n_full), subsampled=bool(subsampled),
              doppler_res_hz=float(prf / n_slow),
              n_paths=int(np.sum(np.isfinite(np.asarray(paths.tau).ravel()))))
    return H, freqs, gt


def trace_two_targets(num, vel_s, vel_w, *, rcs_gap_db=25.0, **kw):
    """Combined CFR of a STRONG near drone + a WEAK distant drone (Phase B). Each is a
    separate monostatic RT trace; their MOVING components (mean-subtracted) are summed
    with the weak `rcs_gap_db` below the strong — deterministic (RCS+range encoded),
    not RT-draw noise (parent fix #4 philosophy). The strong target's static component
    is kept (suppressed later by mean-subtraction). Returns H, freqs, and a GT dict
    with 'strong'/'weak' sub-GTs."""
    Hs, freqs, gts = trace_cfr(num, GEOM_STRONG, vel_s, **kw)
    Hw, _, gtw = trace_cfr(num, GEOM_WEAK, vel_w, **kw)
    Hs_mv = Hs - Hs.mean(axis=0, keepdims=True)          # strong mover
    Hw_mv = Hw - Hw.mean(axis=0, keepdims=True)          # weak mover
    ps = float(np.mean(np.abs(Hs_mv) ** 2)); pw = float(np.mean(np.abs(Hw_mv) ** 2))
    a_w = np.sqrt(ps / max(pw, 1e-30)) * 10 ** (-rcs_gap_db / 20.0)   # weak = strong − gap dB
    H = (Hs.mean(axis=0, keepdims=True) + Hs_mv + a_w * Hw_mv).astype(np.complex64)
    return H, freqs, dict(strong=gts, weak=gtw, rcs_gap_db=float(rcs_gap_db))
