#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bistatic passive-radar (PCL) signal layer for the 5G-22 reproduction
(Maksymiuk et al., Remote Sens. 2022, 14, 6146, Sections 2 & 6).

  Eq 1  bistatic range:    R_b = R1 + R2 - L
  Eq 2  bistatic velocity: V_b = -lambda * f_d
  Eq 3  cross-ambiguity:   chi(R_b,V_b) = INT_{-Tint/2}^{Tint/2}
                             x_sur(t) x_ref*(t - R_b/c) e^{-j 2pi (V_b/lambda) t} dt

Processing chain (paper Fig 2): signal filtering + clutter removal -> CAF -> target
detection (CFAR) -> extraction -> estimation -> tracking. We implement the synthetic
single-target echo of paper Sec 6 (time delay + Doppler + AWGN), a batched CAF, ideal
DPI/clutter cancellation, and CA-CFAR (Rohling). Phase D swaps the synthetic echo for
a Sionna-RT bistatic channel (bistatic_scene.py); the CAF/CFAR below are unchanged.

The reproduced KEY EFFECT (Sec 4/6): with FIXED absolute noise, the CAF target peak
rises with content fill -- because a denser grid carries more echo energy and a
sharper ambiguity function -> higher effective integration gain. Low fill -> target
buried (Fig 8b/10b); high fill -> target clear (Fig 8f/10f, ~24 dB above noise).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

C0 = 299792458.0


# --------------------------------------------------------------------------- #
#  Geometry (Eq 1, 2)
# --------------------------------------------------------------------------- #
@dataclass
class Geometry:
    tx: np.ndarray            # transmitter (gNB) position [m]
    rx: np.ndarray            # surveillance receiver position [m]
    tgt: np.ndarray           # target position [m]
    vel: np.ndarray           # target velocity vector [m/s]

    @property
    def baseline(self) -> float:               # L
        return float(np.linalg.norm(self.tx - self.rx))
    @property
    def bistatic_range(self) -> float:         # Eq 1: R_b = R1 + R2 - L
        R1 = float(np.linalg.norm(self.tgt - self.tx))
        R2 = float(np.linalg.norm(self.rx - self.tgt))
        return R1 + R2 - self.baseline
    def bistatic_velocity(self) -> float:
        """Radial closing rate along the bistatic bisector = d(R1+R2)/dt."""
        u_tx = (self.tgt - self.tx) / (np.linalg.norm(self.tgt - self.tx) + 1e-12)
        u_rx = (self.tgt - self.rx) / (np.linalg.norm(self.tgt - self.rx) + 1e-12)
        return float(np.dot(self.vel, u_tx) + np.dot(self.vel, u_rx))
    def doppler_hz(self, lam: float) -> float:  # Eq 2: V_b = -lambda f_d -> f_d = -V_b/lam
        return -self.bistatic_velocity() / lam


# --------------------------------------------------------------------------- #
#  Synthetic echo (paper Sec 6: delay + Doppler + AWGN), ideal clutter cancel
# --------------------------------------------------------------------------- #
def synth_surveillance(x_ref: np.ndarray, R_b: float, V_b: float, lam: float,
                       fs: float, noise_pow: float, rng,
                       echo_scale: float = 1.0, dpi_scale: float = 0.0,
                       cancel_dpi: bool = True) -> np.ndarray:
    """Surveillance channel = scaled, delayed, Doppler-shifted copy of x_ref
    (single moving target) + optional direct-path interference + AWGN at an
    ABSOLUTE noise power (so detectability tracks content, not per-echo SNR)."""
    n = len(x_ref)
    delay = int(round(R_b / C0 * fs))
    f_d = -V_b / lam
    t = np.arange(n) / fs
    echo = echo_scale * np.roll(x_ref, delay) * np.exp(1j * 2 * np.pi * f_d * t)
    x = echo.astype(np.complex64)
    if dpi_scale > 0:
        x = x + (dpi_scale * x_ref).astype(np.complex64)
    sigma = np.sqrt(noise_pow / 2.0)
    x = x + (sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    if cancel_dpi and dpi_scale > 0:                      # ideal DPI/static-clutter cancel
        x = x - (dpi_scale * x_ref).astype(np.complex64)
    return x


# --------------------------------------------------------------------------- #
#  Cross-ambiguity function (Eq 3), batched range-Doppler
# --------------------------------------------------------------------------- #
def caf(x_sur: np.ndarray, x_ref: np.ndarray, fs: float, lam: float,
        n_batch: int = 128, max_range_m: float = 600.0):
    """Batched cross-ambiguity (Eq 3) -> complex RD map [doppler, range] + axes.
    Range from per-batch circular cross-correlation (lag = R_b/c); Doppler from the
    FFT across batches. Doppler resolution = batch_rate/n_batch."""
    Nt = min(len(x_sur), len(x_ref))
    M = Nt // n_batch
    sur = x_sur[:M * n_batch].reshape(n_batch, M)
    ref = x_ref[:M * n_batch].reshape(n_batch, M)
    n_lag = min(M, int(np.ceil(max_range_m / C0 * fs)) + 1)
    Rc = np.fft.ifft(np.fft.fft(sur, axis=1) * np.conj(np.fft.fft(ref, axis=1)), axis=1)
    Rc = Rc[:, :n_lag]                                    # [n_batch, n_lag] range
    rd = np.fft.fftshift(np.fft.fft(Rc, axis=0), axes=0)  # [doppler, range]
    range_axis = np.arange(n_lag) * C0 / fs               # bistatic range [m]
    batch_rate = fs / M
    f_d = np.fft.fftshift(np.fft.fftfreq(n_batch, d=1.0 / batch_rate))
    vel_axis = -lam * f_d                                 # Eq 2
    return rd, range_axis, vel_axis


# --------------------------------------------------------------------------- #
#  CA-CFAR (Rohling) + metrics
# --------------------------------------------------------------------------- #
def ca_cfar(power: np.ndarray, pfa: float = 1e-6, guard=(2, 2), train=(8, 8)):
    """2-D cell-averaging CFAR. Returns (detection mask, threshold map).
    alpha from Pfa for an N-cell CA estimator: alpha = N (Pfa^(-1/N) - 1)."""
    from scipy.ndimage import uniform_filter            # local import (optional dep)
    gd, gr = guard; td, tr = train
    win_d, win_r = 2 * (gd + td) + 1, 2 * (gr + tr) + 1
    box = uniform_filter(power, size=(win_d, win_r), mode="nearest") * (win_d * win_r)
    gbox = uniform_filter(power, size=(2 * gd + 1, 2 * gr + 1), mode="nearest") * ((2*gd+1)*(2*gr+1))
    n_train = win_d * win_r - (2 * gd + 1) * (2 * gr + 1)
    noise = (box - gbox) / max(n_train, 1)
    alpha = n_train * (pfa ** (-1.0 / max(n_train, 1)) - 1.0)
    thr = alpha * noise
    return power > thr, thr


def _power(rd):
    return np.abs(rd) ** 2


def scr_db(rd, range_axis, vel_axis, R_b, V_b, notch_ms=1.0, guard=3):
    """Signal-to-clutter ratio: target peak / mean clean-noise background.

    The background EXCLUDES the zero-velocity clutter ridge, a guard box around the
    target, AND the target's Doppler row (range sidelobes) + range column (Doppler
    sidelobes) -- otherwise a strong echo's own sidelobes inflate the background and
    flatten the SCR (same exclusion geometry as the parent passive_radar_s2.rd_metrics)."""
    p = _power(rd)                                        # [doppler, range]
    ri = int(np.argmin(np.abs(range_axis - R_b)))
    di = int(np.argmin(np.abs(vel_axis - V_b)))
    sub = p[max(0, di-guard):di+guard+1, max(0, ri-guard):ri+guard+1]
    peak = float(sub.max()) if sub.size else float(p[di, ri])
    clean = np.ones_like(p, bool)
    clean[np.abs(vel_axis) < notch_ms, :] = False        # zero-Doppler clutter ridge
    clean[max(0, di-guard):di+guard+1, :] = False        # target Doppler row (range sidelobes)
    clean[:, max(0, ri-guard):ri+guard+1] = False        # target range col (Doppler sidelobes)
    return float(10 * np.log10(peak / (p[clean].mean() + 1e-30) + 1e-30)), ri, di


def detected(det, range_axis, vel_axis, R_b, V_b, tol_m=30.0, tol_dbin=1, notch_ms=1.0):
    """True if a CFAR hit lands within tol of the true (R_b, V_b) cell (and the
    target is not inside the zero-Doppler clutter notch)."""
    if abs(V_b) < notch_ms:
        return False
    ri = int(np.argmin(np.abs(range_axis - R_b)))
    di = int(np.argmin(np.abs(vel_axis - V_b)))
    res = range_axis[1] - range_axis[0] if len(range_axis) > 1 else tol_m
    rW = max(1, int(round(tol_m / res)))
    return bool(det[max(0, di-tol_dbin):di+tol_dbin+1, max(0, ri-rW):ri+rW+1].any())


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nr_grid import NRGrid, make_reference
    g, rng = NRGrid(), np.random.default_rng(3)
    geo = Geometry(tx=np.array([0, 0, 30.]), rx=np.array([200, 0, 5.]),
                   tgt=np.array([120, 80, 60.]), vel=np.array([-8, 0, 0.]))
    R_b, V_b = geo.bistatic_range, geo.bistatic_velocity()
    print(f"geometry: L={geo.baseline:.1f} m  R_b={R_b:.1f} m  V_b={V_b:+.2f} m/s  "
          f"f_d={geo.doppler_hz(g.wavelength):+.1f} Hz")
    print("\ncontent -> CAF target SCR (fixed noise; SCR must rise with fill):")
    for fill in (0.05, 0.1, 0.3, 0.7, 1.0):
        x_ref, _, rho = make_reference(g, 4e-3, fill, rng)
        x_sur = synth_surveillance(x_ref, R_b, V_b, g.wavelength, g.fs,
                                   noise_pow=2.0, rng=rng, echo_scale=0.05)
        rd, ra, va = caf(x_sur, x_ref, g.fs, g.wavelength, n_batch=64)
        scr, ri, di = scr_db(rd, ra, va, R_b, V_b)
        print(f"  fill={fill:4.2f} density={rho:5.3f}  SCR={scr:6.2f} dB  "
              f"peak@(R={ra[ri]:.0f}m, V={va[di]:+.1f}m/s)")
