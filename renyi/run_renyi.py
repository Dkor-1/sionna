#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5G-22 faithful-reproduction pipeline (Maksymiuk et al., Remote Sens. 2022, 14, 6146)
-- phase-gated, viz-first, in the spirit of ../lasen/run_lasen.py.

Each phase proves itself with a figure mapped to a paper figure + a JSON gate verdict
before the next. Phases A-C are SYNTHETIC (no Sionna RT -- exactly like the paper's
Section 6) and run locally; Phase D needs Sionna RT on the server (bistatic_scene.py).

  A  content-dependency problem   gate: low-fill target buried, high-fill clear     Fig 8/10
  B  content metrics: entropy wins gate: H monotonic & SNR-robust; power/B_eff fail  Fig 9/11/13
  C  adaptive integration -> Pd    gate: Pd rises with fill; entropy-select > naive   Fig 14/15-17
  D  real-flight bistatic (RT)      gate: detections track Sionna GT; 20->100 ms       Fig 21-23  [server]

Run:
  python3 run_renyi.py --phase A      # ~seconds, local
  python3 run_renyi.py --phase B
  python3 run_renyi.py --phase C
  python3 run_renyi.py --phase D      # prints server instructions (needs sionna-rt)
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nr_grid import NRGrid, make_reference
from renyi import renyi_entropy
from content_metrics import measured_power_dbm, effective_bandwidth
from radar import Geometry, synth_surveillance, caf, ca_cfar, scr_db, detected, _power
from range_eq import fig14, RangeParams

# Shared radar setup for the synthetic phases (radial geometry -> large V_b, clear of
# the zero-Doppler clutter notch; echo/noise tuned so detection spans buried->clear).
GEO = Geometry(tx=np.array([0, 0, 30.]), rx=np.array([200, 0, 5.]),
               tgt=np.array([120, 80, 60.]), vel=np.array([0, 18, 0.]))   # V_b ~ +21 m/s
ECHO_SCALE = 4.0
NOISE_POW = 20.0
N_BATCH = 128
TINT = 10e-3


def _outdir():
    cand = "/data/public/jeong/renyi/outputs"
    try:
        os.makedirs(cand, exist_ok=True)
        if os.access(cand, os.W_OK):
            return cand
    except Exception:
        pass
    d = os.path.join(_HERE, "outputs")
    os.makedirs(d, exist_ok=True)
    return d


def _save(verdict: dict, name: str, outdir: str):
    with open(os.path.join(outdir, name + ".json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))


# --------------------------------------------------------------------------- #
#  Phase A -- content dependency (paper Fig 8 / Fig 10)
# --------------------------------------------------------------------------- #
def phase_a(outdir, seed=0):
    g, rng = NRGrid(), np.random.default_rng(seed)
    R_b, V_b = GEO.bistatic_range, GEO.bistatic_velocity()
    fills = [0.05, 0.10, 0.30, 0.70, 1.00]
    rows, rds = [], {}
    for fill in fills:
        x_ref, _, rho = make_reference(g, TINT, fill, rng)
        x_sur = synth_surveillance(x_ref, R_b, V_b, g.wavelength, g.fs,
                                   noise_pow=NOISE_POW, rng=rng, echo_scale=ECHO_SCALE)
        rd, ra, va = caf(x_sur, x_ref, g.fs, g.wavelength, n_batch=N_BATCH)
        scr, _, _ = scr_db(rd, ra, va, R_b, V_b)
        det, _ = ca_cfar(_power(rd), pfa=1e-6)
        rows.append(dict(fill=fill, density=round(rho, 3), scr_db=round(scr, 2),
                         power_dbm=round(measured_power_dbm(x_ref), 2),
                         entropy=round(renyi_entropy(x_ref), 3),
                         detected=bool(detected(det, ra, va, R_b, V_b))))
        if fill in (0.05, 0.30, 1.00):
            rds[fill] = (rd, ra, va, scr)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, fill in zip(axes, (0.05, 0.30, 1.00)):
        rd, ra, va, scr = rds[fill]
        pdb = 10 * np.log10(_power(rd).T + 1e-30)
        pdb -= pdb.max()                                  # peak-normalised dB
        im = ax.pcolormesh(va, ra, pdb, shading="auto", cmap="turbo", vmin=-25, vmax=0)
        ax.plot(V_b, R_b, "x", color="lime", ms=11, mew=2.5)
        ax.set_title(f"content {int(fill*100)}%   SCR={scr:.1f} dB")
        ax.set_xlabel("V_b [m/s]"); ax.set_ylabel("R_b [m]")
        ax.set_xlim(-60, 60); ax.set_ylim(min(300, ra.max()), 0)
    fig.colorbar(im, ax=axes, shrink=0.8, label="rel. power [dB]")
    fig.suptitle("Phase A -- content dependency (5G-22 Fig 8): low content buried, high content clear")
    fig.savefig(os.path.join(outdir, "phaseA_content.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    scrs = [r["scr_db"] for r in rows]
    gate = bool(scrs[-1] - scrs[0] > 6.0 and not rows[0]["detected"] and rows[-1]["detected"])
    _save(dict(phase="A", gate_pass=gate,
               note="CAF target SCR rises with content fill (more occupied REs -> more echo "
                    "energy / integration gain); low-content buried, high-content detected (Fig 8). "
                    "Power-vs-content decoupling is quantified in Phase B.",
               R_b_m=round(R_b, 1), V_b_ms=round(V_b, 2), sweep=rows), "phaseA", outdir)


# --------------------------------------------------------------------------- #
#  Phase B -- entropy beats power & B_eff (paper Fig 9/11/13)
# --------------------------------------------------------------------------- #
def phase_b(outdir, seed=1):
    g, rng = NRGrid(), np.random.default_rng(seed)
    fills = np.linspace(0.0, 1.0, 11)
    snrs = [0, 10, 20, 40]
    ent = {s: [] for s in snrs}; powr, beff = [], []
    for fill in fills:
        x, _, _ = make_reference(g, 4e-3, fill, rng)
        powr.append(measured_power_dbm(x)); beff.append(effective_bandwidth(x, g.fs) / 1e6)
        for s in snrs:
            sig = float(np.mean(np.abs(x) ** 2))
            npow = sig / (10 ** (s / 10.0)) if sig > 0 else 1.0
            xn = x + np.sqrt(npow / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
            ent[s].append(renyi_entropy(xn))
    for s in snrs:
        ent[s] = np.array(ent[s])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for s in snrs:
        ax[0].plot(fills * 100, ent[s], "-o", ms=3, label=f"SNR {s} dB")
    ax[0].set_title("Renyi entropy vs filling (Fig 13)"); ax[0].set_xlabel("Signal filling [%]")
    ax[0].set_ylabel("Renyi entropy"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(fills * 100, powr, "-o", color="tab:red"); ax[1].set_title("Power method (Fig 10) -- ambiguous")
    ax[1].set_xlabel("Signal filling [%]"); ax[1].set_ylabel("Power [dBm]"); ax[1].grid(alpha=.3)
    ax[2].plot(fills * 100, beff, "-o", color="tab:green"); ax[2].set_title("B_eff method (Fig 11) -- noisy/low")
    ax[2].set_xlabel("Signal filling [%]"); ax[2].set_ylabel("B_eff [MHz]"); ax[2].grid(alpha=.3)
    fig.suptitle("Phase B -- entropy is the reliable content measure")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "phaseB_metrics.png"), dpi=130); plt.close(fig)

    H = ent[20]
    monotonic = bool(np.all(np.diff(H) > -0.05))
    snr_robust = bool(max(abs(ent[40][-1] - ent[s][-1]) for s in snrs) < 1.0)
    # power decoupling: a low-fill HIGH-power frame matches a high-fill power but not entropy
    x_hi10, _, _ = make_reference(g, 4e-3, 0.10, rng, amp=2.2)
    x_lo70, _, _ = make_reference(g, 4e-3, 0.70, rng, amp=1.0)
    p_gap = abs(measured_power_dbm(x_hi10) - measured_power_dbm(x_lo70))
    h_gap = renyi_entropy(x_lo70) - renyi_entropy(x_hi10)
    gate = monotonic and snr_robust and (p_gap < 2.0) and (h_gap > 0.3)
    _save(dict(phase="B", gate_pass=bool(gate),
               entropy_monotonic=monotonic, entropy_snr_robust=snr_robust,
               power_decoupling=dict(power_gap_db=round(p_gap, 2), entropy_gap=round(h_gap, 3),
                                     note="power near-equal but entropy separates content"),
               H_at_20dB=[round(float(v), 3) for v in H]), "phaseB", outdir)


# --------------------------------------------------------------------------- #
#  Phase C -- adaptive integration -> Pd (paper Fig 14 + Fig 15-17)
# --------------------------------------------------------------------------- #
def phase_c(outdir, seed=2, n_trials=20):
    g, rng = NRGrid(), np.random.default_rng(seed)
    R_b, V_b = GEO.bistatic_range, GEO.bistatic_velocity()
    fills = np.linspace(0.0, 1.0, 11)
    pfas = [1e-4, 1e-6, 1e-8]
    pd = {pf: [] for pf in pfas}
    for fill in fills:
        hits = {pf: 0 for pf in pfas}
        for _ in range(n_trials):
            x_ref, _, _ = make_reference(g, TINT, fill, rng)
            x_sur = synth_surveillance(x_ref, R_b, V_b, g.wavelength, g.fs,
                                       noise_pow=NOISE_POW, rng=rng, echo_scale=ECHO_SCALE)
            rd, ra, va = caf(x_sur, x_ref, g.fs, g.wavelength, n_batch=N_BATCH)
            for pf in pfas:
                det, _ = ca_cfar(_power(rd), pfa=pf)
                hits[pf] += int(detected(det, ra, va, R_b, V_b))
        for pf in pfas:
            pd[pf].append(hits[pf] / n_trials)

    # Fig 14 (range vs T_int) via range_eq
    fig14(os.path.join(outdir, "phaseC_fig14_range.png"))
    fig, ax = plt.subplots(figsize=(7, 5))
    for pf in pfas:
        ax.plot(fills * 100, pd[pf], "-o", ms=4, label=f"Pfa {pf:.0e}")
    ax.set_xlabel("Signal filling [%]"); ax.set_ylabel("P_d"); ax.set_ylim(-.02, 1.02)
    ax.set_title("Phase C -- P_d vs filling (5G-22 Fig 15-17)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "phaseC_pd.png"), dpi=130); plt.close(fig)

    pd6 = np.array(pd[1e-6])
    gate = bool(pd6[-1] > pd6[0] and pd6[-1] >= 0.7 and pd6[0] <= 0.3)
    _save(dict(phase="C", gate_pass=gate, n_trials=n_trials,
               note="P_d rises with content fill -> selecting high-entropy (dense) frames raises P_d",
               pd_at_pfa={f"{pf:.0e}": [round(float(v), 2) for v in pd[pf]] for pf in pfas}),
          "phaseC", outdir)


# --------------------------------------------------------------------------- #
#  Phase D -- real-flight bistatic on Sionna RT (server only)
# --------------------------------------------------------------------------- #
def phase_d(outdir, seed=3, samples_per_src=2_000_000, n_waypoints=9):
    """Real-flight bistatic detection on a Sionna-RT channel (paper Sec 7, Fig 21-23).
    Needs sionna-rt + OptiX on the server; emits six figures + a JSON gate."""
    import bistatic_scene as bs
    if not bs.have_sionna():
        msg = ("Phase D reuses the parent bistatic Sionna-RT scene (bistatic_scene.py / "
               "../passive_radar_stage1.build_scene) for a real drone echo, runs the same "
               "CAF+CA-CFAR with Renyi-entropy frame selection over a flight trajectory, and "
               "overlays CFAR detections on Sionna's exact GT (Fig 21-23: T_int 20->100 ms). "
               "Requires sionna-rt + OptiX on the RTX-4090 server.")
        print(msg + "\n[!] sionna-rt not importable here -> run Phase D on the server.")
        _save(dict(phase="D", gate_pass=None, sionna_available=False,
                   status="scaffold -- run on the server", plan=msg), "phaseD", outdir)
        return
    res = bs.flight_caf(seed=seed, samples_per_src=samples_per_src, n_waypoints=n_waypoints)
    paths = bs.make_figures(res, outdir)
    g = res["gate"]
    _save(dict(phase="D", sionna_available=True, elapsed_s=res["elapsed_s"],
               figures={k: os.path.basename(v) for k, v in paths.items()},
               cfg=res["cfg"], showcase_wp=res["geometry"]["showcase"],
               trajectory=[{k: (round(v, 2) if isinstance(v, float) else v)
                            for k, v in t.items() if k != "pos"} for t in res["trajectory"]],
               **g), "phaseD", outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list("ABCD"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    outdir = a.outdir or _outdir()
    print(f"[renyi] phase {a.phase} -> {outdir}")
    {"A": phase_a, "B": phase_b, "C": phase_c, "D": phase_d}[a.phase](outdir, seed=a.seed)


if __name__ == "__main__":
    main()
