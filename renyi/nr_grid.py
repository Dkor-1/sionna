#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic 5G-NR downlink resource grid + OFDM waveform for the Renyi-entropy
reproduction (5G-22: Maksymiuk et al., "Renyi Entropy-Based Adaptive Integration
Method for 5G-Based Passive Radar Drone Detection", Remote Sens. 2022, 14, 6146).

Why synthetic is FAITHFUL here: the paper's own Section 6 simulations generate
synthetic 5G-NR signals with the MATLAB 5G Waveform Generator -- channel BW 40 MHz,
sampling 61.44 MHz, SCS 30 kHz, content filling 0..100% with RANDOM RE positions in
the time-frequency grid (Sec 6 + Table 2). We reproduce that synthetic signal in
NumPy with the SAME numerology, so Phases A-C are a direct match to the paper's
Sec 6 (no Sionna RT, no TF PHY needed). Only Phase D (real-flight bistatic echo)
uses Sionna RT. See docs/FAITHFULNESS.md.

Content-dependent transmission (the paper's core problem, Sec 4): the amount of
occupied resources varies; SSB is "always-on", PDSCH data REs follow traffic. The
key consequence reproduced here: signal POWER and signal CONTENT are decoupled
(a low-fill grid at high BS power can match a high-fill grid at low power) -- which
is exactly why the power method fails and the Renyi-entropy method wins.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

C0 = 299792458.0


@dataclass
class NRGrid:
    """5G-NR numerology matching the paper's simulation (Sec 6, Table 2)."""
    scs: float = 30e3            # subcarrier spacing [Hz] (numerology mu=1)
    fs: float = 61.44e6          # sampling rate [Hz]  (paper Sec 6)
    n_fft: int = 2048            # FFT size = fs/scs
    n_rb: int = 106              # resource blocks -> 106*12 = 1272 active SC
    cp_len: int = 144            # normal-CP length [samples] (constant-CP simplification)
    fc: float = 3.44e9           # carrier [Hz]  (paper Table 2 / Lodz testbed)
    ssb_rb: int = 20             # "always-on" SSB-like block width [RB]
    ssb_syms: tuple = (0, 1, 2, 3)   # SSB-like occupied symbols within each slot

    @property
    def n_active(self) -> int:                 # active subcarriers
        return self.n_rb * 12                  # 1272 -> 38.16 MHz occupied
    @property
    def bw(self) -> float:                     # occupied bandwidth [Hz]
        return self.n_active * self.scs        # 38.16 MHz
    @property
    def wavelength(self) -> float:
        return C0 / self.fc                    # ~8.7 cm
    @property
    def t_sym(self) -> float:                  # OFDM symbol duration (with CP) [s]
        return (self.n_fft + self.cp_len) / self.fs   # ~35.7 us
    @property
    def symbol_rate(self) -> float:
        return 1.0 / self.t_sym
    @property
    def range_res_m(self) -> float:            # bistatic range resolution ~ c/(2B)
        return C0 / (2.0 * self.bw)            # ~3.9 m (paper quotes 7.8 m bistatic)
    def n_symbols(self, window_s: float) -> int:
        return max(1, int(round(window_s / self.t_sym)))
    def active_indices(self) -> np.ndarray:
        g = (self.n_fft - self.n_active) // 2
        return np.arange(g, g + self.n_active)


def _qpsk(rng, shape) -> np.ndarray:
    b = rng.integers(0, 2, size=(2, *shape))
    return ((1 - 2 * b[0]) + 1j * (1 - 2 * b[1])).astype(np.complex64) / np.sqrt(2)


def occupancy_mask(grid: NRGrid, n_sym: int, fill: float, rng) -> np.ndarray:
    """Boolean [n_sym, n_active]: True where an RE is transmitted.

    Allocation is at RESOURCE-BLOCK granularity (12 SC), matching the paper's
    "filling 0..100% with random positions in the time-frequency allocation grid"
    (Sec 6, Figs 5-7) -- so at low fill only a few RBs (narrow band) are on, which is
    what makes the entropy (occupied bandwidth x duration) rise gradually with fill.
    The SSB-like block is ALWAYS on (a few centre RBs on configured symbols). `fill`
    is the nominal RB occupancy; the realised grid density is mask.mean()."""
    n_rb, n_active = grid.n_rb, grid.n_active
    rb_alloc = rng.random((n_sym, n_rb)) < float(fill)        # [n_sym, n_rb] PDSCH RBs
    m = np.repeat(rb_alloc, 12, axis=1)                       # -> [n_sym, n_active]
    # SSB-like always-on block (centre RBs, configured symbols of each slot)
    g0 = (n_active - grid.ssb_rb * 12) // 2
    ssb_rows = np.isin(np.arange(n_sym) % 14, grid.ssb_syms)
    m[np.ix_(ssb_rows, np.arange(g0, g0 + grid.ssb_rb * 12))] = True
    return m


def make_reference(grid: NRGrid, window_s: float, fill: float, rng,
                   amp: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Time-domain reference (direct-path) waveform x_ref for a given content
    `fill` and base-station amplitude `amp` (amp models the BS transmit-power
    scaling that DECOUPLES power from content -- paper Fig 10).

    NOTE: the waveform is NOT power-normalised: its total power scales with both
    `fill` (occupied REs) and `amp` (power). This is physical and is what makes the
    power method ambiguous (Sec 4.2) while the Renyi entropy stays content-true.

    Returns (x_ref, mask, realised_density)."""
    n_sym = grid.n_symbols(window_s)
    n_active = grid.n_active
    idx = grid.active_indices()
    mask = occupancy_mask(grid, n_sym, fill, rng)
    grid_re = np.zeros((n_sym, grid.n_fft), np.complex64)
    sym_data = _qpsk(rng, (n_sym, n_active))
    full = np.zeros((n_sym, n_active), np.complex64)
    full[mask] = sym_data[mask]
    grid_re[:, idx] = full
    # OFDM modulate: IFFT per symbol (DC-centred) + cyclic prefix, concatenate
    td = np.fft.ifft(np.fft.ifftshift(grid_re, axes=1), axis=1)        # [n_sym, n_fft]
    td = np.concatenate([td[:, -grid.cp_len:], td], axis=1)            # add CP
    x = (amp * td.reshape(-1)).astype(np.complex64)
    return x, mask, float(mask.mean())


if __name__ == "__main__":
    g = NRGrid()
    rng = np.random.default_rng(0)
    print(f"numerology: fs={g.fs/1e6:.2f} MHz  nfft={g.n_fft}  active={g.n_active}"
          f"  BW={g.bw/1e6:.2f} MHz  range_res={g.range_res_m:.2f} m  t_sym={g.t_sym*1e6:.1f} us")
    for fill in (0.0, 0.1, 0.7, 1.0):
        x, m, rho = make_reference(g, 2e-3, fill, rng)
        print(f"  fill={fill:.2f}  realised_density={rho:.3f}  "
              f"power={10*np.log10(np.mean(np.abs(x)**2)+1e-12):+.2f} dB  n={len(x)}")
