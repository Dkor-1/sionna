#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5G-NR resource grid + numerology for the LaSen faithful reproduction (5G-26, SenSys).

LaSen uses a standard 5G-NR downlink (SCS 30 kHz, 3072-FFT, 78.12 MHz active band).
The real system generates the grid with a 5G toolbox (MATLAB) / sionna.nr; this
container has only `sionna-rt` (no `sionna.nr` / TensorFlow PHY), so the grid is
generated here in NumPy following the SAME standard numerology — documented as a
faithfulness note (docs/FAITHFULNESS.md): equivalent grid structure, not the same
library.

Numerology (LaSen §2.2, Tab.1):
  SCS = 30 kHz (numerology mu=1), n_fft = 3072, active SC = 2604 (guard 468),
  fs = n_fft * SCS = 92.16 MHz, occupied BW = 2604 * 30 kHz = 78.12 MHz, fc = 5.8 GHz
  (or N41 2.5 GHz). DMRS on PDSCH; data REs follow traffic (-> occupancy, Phase B).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

C0 = 299792458.0


@dataclass
class NRNumerology:
    scs: float = 30e3                 # subcarrier spacing [Hz]
    n_fft: int = 3072                 # FFT size
    n_active: int = 2604              # active subcarriers (guard 468)
    fc: float = 5.8e9                 # carrier [Hz] (LaSen SDR; N41 = 2.5 GHz alt)
    cp_frac: float = 9 / 128          # normal CP fraction (~7%)

    @property
    def fs(self) -> float:            # sample rate = full grid bandwidth
        return self.n_fft * self.scs                       # 92.16 MHz

    @property
    def bw(self) -> float:            # occupied bandwidth
        return self.n_active * self.scs                    # 78.12 MHz

    @property
    def wavelength(self) -> float:
        return C0 / self.fc

    @property
    def range_res_m(self) -> float:   # MONOSTATIC range resolution = c / (2 B)
        return C0 / (2.0 * self.bw)                        # ~1.92 m

    @property
    def max_range_m(self) -> float:   # unambiguous (delay 1/SCS), monostatic
        return C0 / (2.0 * self.scs)                       # ~5 km

    def doppler_hz(self, v_radial: float) -> float:        # monostatic f_d = 2 v / lambda
        return 2.0 * v_radial / self.wavelength

    def active_indices(self) -> np.ndarray:
        """Active subcarrier indices within the n_fft grid (DC-centred, guard split)."""
        g = (self.n_fft - self.n_active) // 2
        return np.arange(g, g + self.n_active)

    def baseband_freqs(self) -> np.ndarray:
        """Baseband frequency [Hz] of each active subcarrier (offset from fc)."""
        return (self.active_indices() - self.n_fft / 2) * self.scs


# --------------------------------------------------------------------------- #
#  Resource grid (DMRS + PDSCH) and traffic occupancy (Phase B uses the mask)
# --------------------------------------------------------------------------- #
def dmrs_mask(n_sym: int, n_active: int, dmrs_syms=(2, 11), comb: int = 2,
              shift: int = 0) -> np.ndarray:
    """Boolean [n_sym, n_active]: True where a DMRS RE sits (always transmitted).
    PDSCH DM-RS type-1: comb-2 in frequency on configured OFDM symbols."""
    m = np.zeros((n_sym, n_active), bool)
    rows = np.isin(np.arange(n_sym) % 14, dmrs_syms)
    m[np.ix_(rows, (np.arange(n_active) % comb) == shift)] = True
    return m


def occupancy_mask(n_sym: int, n_active: int, load: float, rng,
                   dmrs_syms=(2, 11)) -> np.ndarray:
    """Traffic occupancy mask [n_sym, n_active] (review/plan Phase B): DMRS REs are
    ALWAYS present; PDSCH data REs are transmitted with probability ~`load` (the
    non-uniform traffic). `load` is the average data-RE fraction; the realised
    grid density (returned by .mean()) is what LaSen calls occupancy."""
    dm = dmrs_mask(n_sym, n_active, dmrs_syms)
    data = (rng.random((n_sym, n_active)) < load) & (~dm)
    return dm | data
