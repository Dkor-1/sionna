#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-uniform 5G traffic occupancy masks for LaSen Phase B (5G-26, Tab.1 / Fig.12).

LaSen's hard problem is REAL non-uniform downlink traffic: only a small, time-varying
fraction of REs is actually transmitted, so the gNB observes the CFR only on those REs
(a masked / sub-Nyquist observation) — which makes a plain 2D-FFT leak, and motivates
the 2D-OMP sparse recovery. We MODEL that occupancy (synthetic, not captured — see
docs/FAITHFULNESS.md) from the paper's measured density bins.
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from nr_waveform import dmrs_mask

# LaSen Tab.1 / Fig.12 measured occupancy density bins (fraction of the grid)
DENSITY_BINS = {"sparse": (0.006, 0.068), "moderate": (0.068, 0.131), "dense": (0.131, 0.193)}


def mask_at_density(n_sym, n_active, rho, rng, dmrs_syms=(2, 11)):
    """Occupancy mask [n_sym, n_active] at target grid density ~rho (structured like
    DMRS comb + PDSCH data). DMRS REs are deterministic where scheduled; in sparse
    traffic fewer RBs are scheduled (DMRS thinned), in dense traffic extra data fills
    in. Returns (W, realised_density). rho is the LaSen occupancy q/(M*N)."""
    dm = dmrs_mask(n_sym, n_active, dmrs_syms)
    base = float(dm.mean())
    if rho >= base:                                   # add PDSCH data to reach rho
        p = (rho - base) / (1.0 - base)
        W = dm | ((rng.random((n_sym, n_active)) < p) & (~dm))
    else:                                             # sparse: thin the scheduled DMRS combs
        W = dm & (rng.random((n_sym, n_active)) < (rho / max(base, 1e-9)))
    return W, float(W.mean())


def density_timeline(W, slot_syms=14):
    """Per-slow-time occupancy fraction (for the Fig.11/5b density-vs-time view)."""
    return W.mean(axis=1)
