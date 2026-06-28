#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passive-radar range equation (5G-22, Eq 9-10, Table 2) -> Figure 14
(detection range vs integration time, per RCS). This is the analytic motivation for
adaptive integration: range grows with effective T_int, so selecting dense (high-
entropy) frames -- which raises the EFFECTIVE T_int -- extends the detection range.

  Eq 9   R_e < sqrt( P_T G_T G_R S_0 lambda^2 T_int / ( (4pi)^3 L_0 D_0 k T_0 ) )
  Eq 10  T_0 = T_ref ( 10^(N_f/10) - 1 ),   T_ref = 290 K

FAITHFULNESS NOTE: Eq 9 as printed (sqrt) does not reproduce Fig 14's magnitudes;
the standard monostatic-equivalent 4th-power range law (R_e = (.)^(1/4)) does -- it
gives RCS=1 m^2 -> ~26 km at T_int=0.5 s and RCS=100 m^2 -> ~97 km at 1 s, matching
Fig 14 exactly. So the printed sqrt is read as a typo; we use the 4th root and say so
(docs/FAITHFULNESS.md). Verified against the paper's own figure below.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

C0 = 299792458.0
K_BOLTZ = 1.380649e-23


@dataclass
class RangeParams:
    """Paper Table 2 (simulated network parameters)."""
    fc: float = 3.44e9          # centre frequency [Hz]
    eirp_dbm: float = 73.0      # EIRP = P_T G_T [dBm]
    gr_dbi: float = 10.0        # receiver antenna gain [dBi]
    d0_db: float = 11.0         # detection threshold (detection factor) [dB]
    l0_db: float = 10.0         # total losses [dB]
    t0_k: float = 493.0         # effective receiver noise temperature [K] (N_f ~ 4.3 dB)

    @property
    def lam(self) -> float:
        return C0 / self.fc


def noise_temp_from_nf(nf_db: float, t_ref: float = 290.0) -> float:
    """Eq 10: T_0 = T_ref (10^(N_f/10) - 1)."""
    return t_ref * (10 ** (nf_db / 10.0) - 1.0)


def detection_range(t_int: float, rcs_m2: float, p: RangeParams = RangeParams()) -> float:
    """Equivalent (monostatic) detection range R_e [m] from the 5G-22 range equation
    (Eq 9, 4th-power form -- see module note). R_e = sqrt(R1 R2)."""
    eirp = 10 ** ((p.eirp_dbm - 30) / 10.0)             # W
    gr = 10 ** (p.gr_dbi / 10.0)
    d0 = 10 ** (p.d0_db / 10.0)
    l0 = 10 ** (p.l0_db / 10.0)
    num = eirp * gr * rcs_m2 * p.lam ** 2 * t_int
    den = (4 * np.pi) ** 3 * l0 * d0 * K_BOLTZ * p.t0_k
    return float((num / den) ** 0.25)


def fig14(out_png: str | None = None, p: RangeParams = RangeParams(),
          rcs_list=(1, 10, 50, 100), t_int=np.linspace(0.01, 1.0, 100)):
    """Reproduce paper Figure 14: detection range [km] vs T_int [s] per RCS [m^2]."""
    curves = {s: np.array([detection_range(t, s, p) / 1e3 for t in t_int]) for s in rcs_list}
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for s, y in curves.items():
            plt.plot(t_int, y, label=f"{s}")
        plt.xlabel("Integration time [s]"); plt.ylabel("Detection range [km]")
        plt.legend(title="RCS [m$^2$]"); plt.grid(alpha=.3)
        plt.title("Range vs integration time (5G-22 Fig 14 reproduction)")
        plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    return t_int, curves


if __name__ == "__main__":
    p = RangeParams()
    print(f"Table 2: fc={p.fc/1e9} GHz  EIRP={p.eirp_dbm} dBm  Gr={p.gr_dbi} dBi  "
          f"D0={p.d0_db} dB  L0={p.l0_db} dB  T0={p.t0_k} K  (lambda={p.lam*100:.1f} cm)")
    print("\nVerification against paper Fig 14 (expect ~26 km @ RCS1/0.5s, ~97 km @ RCS100/1s):")
    for rcs in (1, 10, 50, 100):
        for t in (0.02, 0.1, 0.5, 1.0):
            print(f"  RCS={rcs:3d} m^2  T_int={t:4.2f} s  ->  R_e = {detection_range(t, rcs)/1e3:6.2f} km")
        print()
