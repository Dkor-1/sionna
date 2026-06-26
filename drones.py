# -*- coding: utf-8 -*-
"""
Per-drone parameters for the candidate procurement airframes (PROJECT_CONTEXT
sec.7).  Datasheet values (weight, unfolded size, max speed) are real; the
**effective radar size is an ESTIMATE**, not a datasheet number.

Why RCS is not from the spec sheet
----------------------------------
DJI does not publish radar cross section.  Real per-drone RCS at GHz bands needs
either a chamber measurement or a CAD + EM-solver model.

The metal CUBE used in the Sionna scene is a **calibration placeholder**, NOT a
faithful RCS model (review fix #4): a flat metal cube is specular-glint dominated
with RCS ~ size^4 / lambda^2 (a 0.3 m cube is ~+8..+14 dBsm at 2.4-5 GHz),
i.e. 10-30 dB above the literature drone anchor of ~-20 dBsm and 6 dB/octave
carrier-dependent.  We therefore DO NOT read absolute RCS off the cube.  Instead:
  * the cube fixes the GEOMETRY (exact bistatic delay/Doppler — verified correct);
  * per-drone echo POWER is set deterministically from literature dBsm anchors
    (`rcs_dbsm`), applied as a power scaling on the moving echo (phase1 fix #4);
  * RCS is **carrier-dependent** — `rcs_dbsm_at(drone, fc)` adds the literature
    S->C-band rise (~-20 dBsm @2.5 GHz -> ~-12 dBsm @3.5 GHz, WiFi-21a / 5G-25a/b),
    so a real drone's echo gains ~+8 dB at C-band, partly offsetting the carrier
    path-loss (this is exactly why "carrier dominates" must NOT be claimed naively).
`radar_size_m` now only sizes the calibration mesh; replace `rcs_dbsm` with
measured/CAD values when available.  All RCS numbers are ESTIMATES.

Sources (accessed 2026-06-25): DJI / DrDrone / B&H spec pages; docs/LITERATURE_REVIEW.md sec.3.
"""
from dataclasses import dataclass


@dataclass
class Drone:
    name: str
    takeoff_g: float          # takeoff weight [g]                (datasheet)
    unfolded_mm: tuple        # L x W x H unfolded [mm]           (datasheet)
    max_speed_ms: float       # max horizontal speed, Sport [m/s] (datasheet)
    radar_size_m: float       # effective conductive extent [m] for the mesh (est.)
    rcs_dbsm: float           # nominal RCS [dBsm] at ~2.5-3.5 GHz (LITERATURE-GROUNDED ESTIMATE)
    rtk: bool = False         # onboard RTK (precise GT)          (datasheet)
    note: str = ""


# RCS (dBsm) grounded in docs/LITERATURE_REVIEW.md sec.3 (no paper measured DJI
# RCS; these are size-scaled around the WiFi-21a FDTD anchor of ~-20 dBsm for a
# small quad at 2.48 GHz, and -10..-13 dBsm at 3.5 GHz C-band from 5G-25a/b).
# Still an ESTIMATE; frequency-dependent (rolls off below ~600 MHz; +10 dB H-pol
# for carbon blades at UHF, LTE-24). radar_size_m only sizes the RT mesh.
REF_RCS_DBSM = -18.0          # reference (Mavic-class) -> rcs_scale = 1 in phase1

DRONES = {
    "mini5pro": Drone("DJI Mini 5 Pro", 249.9, (304, 380, 91), 19.0, 0.17,
                      rcs_dbsm=-24.0, rtk=False, note="sub-250 g, smallest/hardest"),
    "air3s":    Drone("DJI Air 3S",     724.0, (266, 325, 106), 21.0, 0.22,
                      rcs_dbsm=-20.0, rtk=False, note="mid consumer (~WiFi-21a anchor)"),
    "mavic4pro":Drone("DJI Mavic 4 Pro",1063.0,(329, 391, 135), 25.0, 0.27,
                      rcs_dbsm=-18.0, rtk=False, note="largest consumer, fastest"),
    "matrice4e":Drone("DJI Matrice 4E", 1219.0,(307, 388, 150), 21.0, 0.29,
                      rcs_dbsm=-16.0, rtk=True,  note="enterprise + onboard RTK"),
}


# --------------------------------------------------------------------------- #
#  Carrier-dependent RCS (review fix #4)
# --------------------------------------------------------------------------- #
# `rcs_dbsm` is the nominal RCS at S-band (~2.5 GHz). Real small-quad RCS RISES
# toward C-band (body ~ lambda -> Rayleigh-to-resonance transition): the WiFi-21a
# FDTD anchor (~-20 dBsm @2.48 GHz) and the 5G-25a/b C-band link budgets
# (-10..-13 dBsm @3.5 GHz) bracket a ~+8 dB rise over 2.5->3.5 GHz. We model this
# with a 2-anchor curve, clamped flat outside [2.5, 3.5] GHz (conservative — we do
# not extrapolate the resonance). This is an ESTIMATE; frequency dependence is
# also polarization/aspect dependent (e.g. +10 dB H-pol carbon blades at UHF).
RCS_FC_LO, RCS_FC_HI = 2.5e9, 3.5e9
RCS_FC_RISE_DB = 8.0


def carrier_rcs_delta_db(fc: float) -> float:
    """dB rise of small-quad RCS at carrier `fc` relative to S-band (2.5 GHz),
    from the 2-anchor literature curve, clamped flat outside [2.5, 3.5] GHz."""
    f = min(max(float(fc), RCS_FC_LO), RCS_FC_HI)
    return RCS_FC_RISE_DB * (f - RCS_FC_LO) / (RCS_FC_HI - RCS_FC_LO)


def rcs_dbsm_at(drone: "Drone", fc: float) -> float:
    """Carrier-dependent RCS [dBsm] = nominal (S-band) + literature C-band rise."""
    return drone.rcs_dbsm + carrier_rcs_delta_db(fc)
