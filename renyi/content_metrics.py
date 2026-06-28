#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline content-measurement methods that the paper evaluates and shows to be
INSUFFICIENT vs the Renyi entropy (5G-22, Sections 4.2 and 4.3). Reproducing their
failure modes is what makes the entropy result meaningful.

  * power measurement (Sec 4.2, Fig 10): thresholding received power selects both a
    low-content-high-power frame and a high-content frame, but only the latter is
    useful for radar -> power conflates content with transmit power.
  * RMS effective bandwidth (Sec 4.3, Eq 4, Fig 11): B = (1/A_max) INT A(f) df,
    A(f)=PSD. For a (near-)rectangular OFDM spectrum this is sensitive to PSD peaks
    and can even DECREASE with occupancy (paper Fig 11: 2.21 -> 1.90 -> 1.33 MHz)
    -> ambiguous; far below the true 38.16 MHz.
"""
from __future__ import annotations
import numpy as np


def measured_power_dbm(x: np.ndarray, ref_ohm: float = 1.0) -> float:
    """Average received power of a frame, in dBm (relative scale; paper Fig 10
    annotations are dBm). Conflates content and transmit power by construction."""
    p = float(np.mean(np.abs(x) ** 2)) / ref_ohm
    return 10.0 * np.log10(p + 1e-30) + 30.0


def effective_bandwidth(x: np.ndarray, fs: float, nfft: int = 4096) -> float:
    """RMS / Olsen-Baker effective bandwidth (paper Eq 4): B = INT A(f) df / A_max,
    A(f) = power spectral density. Returns B in Hz."""
    f, A = _psd(x, fs, nfft)
    df = f[1] - f[0]
    return float(A.sum() * df / (A.max() + 1e-30))


def _psd(x: np.ndarray, fs: float, nfft: int = 4096):
    """Welch-style averaged periodogram -> (freqs [Hz], PSD [linear])."""
    win = min(nfft, len(x))
    w = np.hanning(win + 2)[1:-1]
    if len(x) < win:
        x = np.concatenate([x, np.zeros(win - len(x), x.dtype)])
    n_seg = max(1, len(x) // win)
    segs = x[:n_seg * win].reshape(n_seg, win)
    P = np.mean(np.abs(np.fft.fftshift(np.fft.fft(segs * w[None, :], axis=1), axes=1)) ** 2, axis=0)
    P /= (np.sum(w ** 2) * fs)
    f = np.fft.fftshift(np.fft.fftfreq(win, d=1.0 / fs))
    return f, P


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nr_grid import NRGrid, make_reference
    from renyi import renyi_entropy
    g, rng = NRGrid(), np.random.default_rng(2)
    print("method comparison (paper Sec 4-5): only entropy ranks by CONTENT")
    print(f"{'case':16s}{'power[dBm]':>12s}{'B_eff[MHz]':>12s}{'Renyi H':>10s}")
    cases = [("10% low-pwr", 0.10, 1.0), ("10% HIGH-pwr", 0.10, 2.2),
             ("70% low-pwr", 0.70, 1.0), ("100% full", 1.00, 1.0)]
    for name, fill, amp in cases:
        x, _, _ = make_reference(g, 2e-3, fill, rng, amp=amp)
        print(f"{name:16s}{measured_power_dbm(x):12.2f}"
              f"{effective_bandwidth(x, g.fs)/1e6:12.2f}{renyi_entropy(x):10.3f}")
