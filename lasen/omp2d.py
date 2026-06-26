#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D-OMP sparse range-Doppler recovery for LaSen Phase B (5G-26, Eq 4-6).

From the MASKED CFR  h = W ∘ H_full = Φ z  (Eq 4-5), recover the K-sparse RD scene
z = argmin ‖z‖₀  s.t. ‖h − Φz‖ ≤ ε  (Eq 6) by Orthogonal Matching Pursuit. The key
to faithful efficiency: the matching step (max residual correlation over all atoms) is
a 2D-DFT (`rd_transform`), O(MN log MN) per iter, NOT an explicit M·N×M·N dictionary.

Sub-gate (`roundtrip_ok`): rd_transform(atom(di,ri)) must peak exactly at (di,ri) —
the sign/convention check. If this breaks, OMP is meaningless.
"""
from __future__ import annotations
import numpy as np


def rd_transform(X):
    """Imaging + OMP matching, shared 2D-DFT: range = freq IFFT, doppler = slow-time
    FFT (shifted). X:[N_slow, K_freq] -> RD:[doppler, range]. (Same convention as
    run_lasen.rd_from_cfr so Phase B FFT-RD and OMP-RD share axes.)"""
    return np.fft.fftshift(np.fft.fft(np.fft.ifft(X, axis=1), axis=0), axes=0)


def atom(N, K, di, ri):
    """RD-cell basis vector: the CFR whose rd_transform peaks exactly at
    (doppler bin di, range bin ri). Freq sign chosen so IFFT-over-freq peaks at ri;
    slow-time term undoes the doppler fftshift (d = di - N//2)."""
    n = np.arange(N)[:, None]; m = np.arange(K)[None, :]
    d = di - N // 2
    return (np.exp(2j * np.pi * d * n / N) * np.exp(-2j * np.pi * m * ri / K)).astype(np.complex64)


def roundtrip_ok(N=64, K=128, cells=((20, 37), (5, 100), (50, 3), (33, 64))):
    """SUB-GATE: rd_transform(atom(di,ri)) peaks at (di,ri) for several cells."""
    rows = []
    for di, ri in cells:
        C = rd_transform(atom(N, K, di, ri))
        pk = tuple(int(x) for x in np.unravel_index(int(np.argmax(np.abs(C))), C.shape))
        rows.append(dict(target=[di, ri], peak=list(pk), ok=bool(pk == (di, ri))))
    return all(r["ok"] for r in rows), rows


def omp2d(H, W, n_iter=15, resid_frac=0.05, rel_tol=1e-3):
    """2D-OMP on the masked CFR. H[N,K] full CFR, W bool occupancy mask.
    Stop when the residual falls to `resid_frac` of the observed-signal norm (captured
    the targets, the rest is noise/mismatch) or the relative change stalls, capped at
    n_iter atoms. Returns Z[doppler, range] (sparse), support, residual-norm history."""
    N, K = H.shape
    Hm = np.where(W, H, 0).astype(np.complex64)        # Eq 5: masked CFR
    occ = W.ravel(); h_occ = Hm.ravel()[occ]           # observed entries h ∈ C^q
    h_norm = float(np.linalg.norm(h_occ))
    R = Hm.copy(); support = []; cols = []
    hist = [h_norm]; z = np.zeros(0, np.complex64)
    for _ in range(n_iter):
        C = rd_transform(R)                            # matching = residual 2D-DFT
        di, ri = (int(x) for x in np.unravel_index(int(np.argmax(np.abs(C))), C.shape))
        if (di, ri) in support:
            break
        support.append((di, ri))
        cols.append(atom(N, K, di, ri).ravel()[occ])   # masked atom column (Eq 5)
        Phi = np.stack(cols, 1)
        z, *_ = np.linalg.lstsq(Phi, h_occ, rcond=None)    # LS amplitudes (Eq 6)
        Hrec = np.zeros((N, K), np.complex64)
        for (d, r), zk in zip(support, z):
            Hrec += zk * atom(N, K, d, r)
        R = Hm - np.where(W, Hrec, 0)
        hist.append(float(np.linalg.norm(R)))
        if hist[-1] <= resid_frac * h_norm or \
           abs(hist[-2] - hist[-1]) / (hist[-2] + 1e-30) < rel_tol:
            break
    Z = np.zeros((N, K), np.complex64)
    for (d, r), zk in zip(support, z):
        Z[d, r] = zk
    return Z, support, hist
