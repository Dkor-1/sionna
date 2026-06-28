#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THE novelty of 5G-22: Renyi-entropy-based measurement of 5G resource utilisation,
used to ADAPTIVELY select dense time frames for passive-radar integration
(Maksymiuk et al., Remote Sens. 2022, 14, 6146, Section 5).

Pipeline (paper Eq 6-8):
  Eq 6  STFT:        F_x^h(t,w) = INT x(tau) h*(tau-t) e^{-j w (tau-t)} dtau
  Eq 7  spectrogram: S_x^h(t,w) = |F_x^h(t,w)|^2
  Eq 8  Renyi:       H_R^g(TF_x) = 1/(1-g) * log2( II TF_x^g dt dw / II TF_x dt dw )
        with TF_x = |F_x^h| (the ABSOLUTE value of the STFT, paper Sec 5.1) and
        g = 3 (fixed; "ensures stable results and is the best choice", paper + refs).

Rationale (paper Sec 5.2): the Renyi entropy rises with the number of occupied
subcarriers x their duration, i.e. with the EFFECTIVE integration time/bandwidth
in the passive-radar range equation (Eq 9). Selecting only high-entropy frames
maximises effective T_int -> detects low-RCS drones (Sec 6 result, Fig 13/15-17).

Practical use (paper Sec 5.2, 4 steps):
  1. calibrate -> max entropy with full allocation (calibrate_max)
  2. receive continuously
  3. extract useful frames: entropy >= threshold (e.g. >=90% of max)  (select_frames)
  4. classical PCL processing on the selected frames (see radar.py / run_renyi.py)

FAITHFULNESS NOTE (honest paper-diff): Eq 8 as printed divides by II TF dt dw
(L1 sum), not (II TF)^g. The reported magnitudes (~25.8 at full allocation, paper
Figs 12/13/20) equal ~log2(number of occupied TF cells), which is the *normalised*
Renyi entropy of Baraniuk et al. 2001 (the paper's ref [38]); i.e. the distribution
is normalised to sum 1 first. We implement that normalised form (it reproduces the
magnitudes and the monotonic shape); the absolute value scales with the STFT grid
size, so we reproduce the TREND, per docs/FAITHFULNESS.md.
"""
from __future__ import annotations
import numpy as np

_TINY = 1e-30


def stft(x: np.ndarray, win: int = 512, hop: int = 256, nfft: int | None = None):
    """Short-time Fourier transform (Eq 6) with a Hann window.
    Returns (S, n_frames, n_freq) where S is complex [n_frames, n_freq]."""
    nfft = nfft or win
    w = np.hanning(win + 2)[1:-1].astype(np.float64)
    if len(x) < win:
        x = np.concatenate([x, np.zeros(win - len(x), x.dtype)])
    n_frames = 1 + (len(x) - win) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n_frames]
    S = np.fft.fftshift(np.fft.fft(frames * w[None, :], nfft, axis=1), axes=1)
    return S, n_frames, nfft


def renyi_entropy(x: np.ndarray, gamma: float = 3.0,
                  win: int = 512, hop: int = 256, nfft: int | None = None,
                  stft_in: np.ndarray | None = None) -> float:
    """Renyi entropy of the signal's time-frequency content (Eq 8), normalised
    (Baraniuk form, see module note). TF = |STFT| (Eq 6 absolute value), gamma=3.

    Pass a precomputed STFT via `stft_in` to avoid recomputation."""
    S = stft_in if stft_in is not None else stft(x, win, hop, nfft)[0]
    tf = np.abs(S).astype(np.float64)
    p = tf / (tf.sum() + _TINY)                       # normalise to a distribution
    return float(1.0 / (1.0 - gamma) * np.log2((p ** gamma).sum() + _TINY))


def calibrate_max(x_full: np.ndarray, frame_len: int, gamma: float = 3.0,
                  **stft_kw) -> float:
    """Step 1 (paper Sec 5.2): record the signal with FULL resource allocation and
    take its (max) Renyi entropy as the calibration reference. `x_full` should be a
    fully-allocated reference capture; we take the max over its frames for robustness."""
    n = max(1, len(x_full) // frame_len)
    vals = [renyi_entropy(x_full[i * frame_len:(i + 1) * frame_len], gamma, **stft_kw)
            for i in range(n)]
    return float(np.max(vals)) if vals else 0.0


def frame_entropies(x: np.ndarray, frame_len: int, gamma: float = 3.0,
                    **stft_kw) -> tuple[np.ndarray, np.ndarray]:
    """Renyi entropy of every consecutive frame of length `frame_len`.
    Returns (entropies, frame_start_indices). (paper Fig 24: entropy per 20 ms frame.)"""
    n = max(1, len(x) // frame_len)
    starts = np.arange(n) * frame_len
    ent = np.array([renyi_entropy(x[s:s + frame_len], gamma, **stft_kw) for s in starts])
    return ent, starts


def select_frames(x: np.ndarray, frame_len: int, threshold: float,
                  gamma: float = 3.0, **stft_kw) -> tuple[np.ndarray, np.ndarray]:
    """Step 3 (paper Sec 5.2): keep only frames whose Renyi entropy >= `threshold`
    (e.g. threshold = 0.90 * calibrated_max, paper used 25.5 vs max 25.67).
    Returns (boolean keep-mask over frames, frame_start_indices)."""
    ent, starts = frame_entropies(x, frame_len, gamma, **stft_kw)
    return ent >= threshold, starts


if __name__ == "__main__":
    # Self-test: entropy must rise monotonically with content fill and be ~power-blind.
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nr_grid import NRGrid, make_reference
    g, rng = NRGrid(), np.random.default_rng(1)
    print("fill ->  Renyi entropy (gamma=3), power-decoupling check")
    base = None
    for fill in (0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0):
        x, _, rho = make_reference(g, 2e-3, fill, rng)
        H = renyi_entropy(x)
        if base is None:
            base = H
        print(f"  fill={fill:4.2f} density={rho:5.3f}  H={H:6.3f}  dH={H-base:+5.3f}")
    # power vs content decoupling (paper Fig 10): 10% high-power vs 70% low-power
    x_lo10, _, _ = make_reference(g, 2e-3, 0.10, rng, amp=1.0)
    x_hi10, _, _ = make_reference(g, 2e-3, 0.10, rng, amp=2.2)   # +~7 dB power
    x_lo70, _, _ = make_reference(g, 2e-3, 0.70, rng, amp=1.0)
    def pdb(x): return 10*np.log10(np.mean(np.abs(x)**2)+1e-12)
    print("\npower-vs-content decoupling (entropy should rank by CONTENT, not power):")
    for name, x in [("10% low-pwr", x_lo10), ("10% HIGH-pwr", x_hi10), ("70% low-pwr", x_lo70)]:
        print(f"  {name:14s}  power={pdb(x):+6.2f} dB   H={renyi_entropy(x):6.3f}")
