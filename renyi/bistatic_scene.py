#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase D (SERVER ONLY): real-flight bistatic drone echo from a Sionna-RT scene, for
the 5G-22 reproduction (paper Sec 7 real experiment, Figs 21-23). Requires sionna-rt
+ OptiX on the RTX-4090 server -- this file is a scaffold that REUSES the parent's
ray-traced bistatic channel; the CAF / CFAR / Renyi-selection are the same NumPy code
as Phases A-C (radar.py / renyi.py).

What Phase D adds over the synthetic Phases A-C:
  * a REAL bistatic channel (gNB Tx, surveillance Rx, drone scatterer) ray-traced by
    Sionna RT instead of the analytic delay+Doppler echo -> realistic multipath/clutter
    so the SPARSE-reference ambiguity degradation (why low content fails even at equal
    power, paper Fig 10) appears naturally;
  * a flight TRAJECTORY (paper Fig 18c / 23) -> a sequence of CAFs;
  * Renyi-entropy frame selection on the real reference capture (calibrate_max ->
    threshold ~= 0.99 * max, paper used 25.5 vs 25.67) before each CAF;
  * detections overlaid on Sionna's EXACT ground truth (paper used GPS logs) and the
    T_int 20 ms -> 100 ms velocity-resolution improvement (Fig 21 -> Fig 22).

Implementation reuses parent passive_radar_stage1.build_scene / trace_channel (which
return discrete baseband taps h[N, L] + analytic GT). Drone RCS per model comes from
the parent `drones` dBsm anchors (see ../drones.py / ../NOTES.md fix #4).
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nr_grid import NRGrid, make_reference
from renyi import calibrate_max, select_frames, renyi_entropy
from radar import caf, ca_cfar, scr_db, detected, _power


def have_sionna() -> bool:
    try:
        import sionna.rt  # noqa: F401
        return True
    except Exception:
        return False


def surveillance_from_taps(x_ref: np.ndarray, h, fs: float) -> np.ndarray:
    """Convolve the 5G reference with the Sionna-RT time-varying channel taps
    h[N_slow, L] to synthesise the surveillance channel (drone echo + clutter).
    Mirrors parent passive_radar_s2._conv_fft; ideal static-clutter cancellation is
    the parent's h - h.mean(0) trick. TODO(server): wire to parent helper to avoid
    duplication once sionna-rt is importable."""
    raise NotImplementedError("Phase D runs on the server: import parent _conv_fft + "
                              "trace_channel; see docstring.")


def flight_caf(drone: str = "mavic", t_int: float = 100e-3, fill: float = 1.0,
               waypoints=None, fc: float = 3.44e9, seed: int = 0):
    """End-to-end Phase D (server). Outline:

      1. grid = NRGrid(fc=fc); x_ref = make_reference(grid, t_int, fill)         # 5G ref
      2. cal = calibrate_max(x_ref_full, frame_len); thr = 0.99 * cal            # Sec 5.2 step 1
      3. for each waypoint / time step:
            scene = parent.build_scene(cfg@waypoint)                             # Sionna RT
            h, gt = parent.trace_channel(cfg, scene)                            # bistatic taps + exact GT
            keep, _ = select_frames(x_ref, frame_len, thr)                       # Renyi adaptive integration
            x_sur  = surveillance_from_taps(x_ref[kept], h, grid.fs)
            rd, ra, va = caf(x_sur, x_ref[kept], grid.fs, grid.wavelength)
            det, _ = ca_cfar(_power(rd), pfa=1e-6)                               # SNR threshold 15 dB (Sec 7.2)
            record detections + gt
      4. overlay detections on Sionna GT trajectory (Fig 23); compare T_int 20<->100 ms (Fig 21/22).

    GATE: CFAR detections follow the Sionna GT trajectory for ~the whole flight; the
    100 ms integration sharpens velocity vs 20 ms (paper Fig 21->22)."""
    if not have_sionna():
        raise RuntimeError("Phase D needs sionna-rt + OptiX (run on the RTX-4090 server).")
    raise NotImplementedError("Server scaffold -- fill in steps 1-4 using parent "
                              "passive_radar_stage1.build_scene/trace_channel + ../drones.")


if __name__ == "__main__":
    print("Phase D bistatic scene (server only). sionna available:", have_sionna())
    print("See flight_caf() docstring for the Sec 7 / Fig 21-23 reproduction outline.")
