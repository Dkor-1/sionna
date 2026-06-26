#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-1 controlled Sionna passive-radar benchmark (per EXPERIMENT SPEC).

Three STUDY axes, everything else held fixed (the control):
  A) signal spec   — realistic BW x carrier x SCS per standard (Wi-Fi/LTE/5G)
  B) drone         — real DJI models (RCS via size), from drones.py
  C) scenario      — drone-motion pattern: hover / radial / tangential / doppler_switch

Fixed budget (control): geometry, illumination (unit-power waveform = fixed
EIRP), **absolute noise density N0** (anchored once), integration time (CPI),
CFAR operating point, #trials. => **SCR is MEASURED, not swept**: it falls out
of (spec x drone x motion) under the fixed budget.

Chain per cell: Sionna RT (CIR, bulk Doppler from the scenario's velocity)
  -> reference(pilots) + surveillance -> CAF (RD map) -> CFAR -> read SCR, Pd, FAR.
Each cell = N Monte-Carlo trials (random noise/data) -> mean +/- CI.

Scope: bulk translational Doppler only -> hover & pure-tangential are expected
HARD/blind cases (motivates a later rotor micro-Doppler layer). No blade mesh.

Reuses the validated S1/S2 primitives. This Phase-1 realistic-spec sweep
REPLACES the earlier pilot-density 4-mode comparison as the headline benchmark.
"""
from __future__ import annotations
import os, json, copy, argparse
from dataclasses import dataclass
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from passive_radar_stage1 import Config, C0, build_scene
from passive_radar_s2 import (OFDM, synth_ofdm, surveillance, caf_range_doppler,
                              ca_cfar_2d, rd_metrics, pilot_mask)
from drones import DRONES, Drone, REF_RCS_DBSM, rcs_dbsm_at

# --------------------------------------------------------------------------- #
#  FIXED control budget + geometry
# --------------------------------------------------------------------------- #
GEOM = dict(tx=(-50.0, 0.0, 20.0), rx=(50.0, 0.0, 10.0), drone=(0.0, 120.0, 60.0))
FC_ANCHOR = 1.8e9             # carrier of the N0-anchor cell (LTE 10MHz radial Mavic)
REF_DRONE_KEY = "mavic4pro"   # reference drone -> rcs_scale = 1 at FC_ANCHOR
APPLY_CARRIER_RCS = True      # add the literature S->C-band RCS rise to the echo
                              # (review fix #4); set False to isolate the carrier
                              # PROPAGATION effect in the decoupling sweep.
CPI_S = 0.1                   # fixed integration time [s] (10 Hz Doppler res; literature 0.1-0.5 s)
# Batch length must exceed the reference's time-period (LTE CRS ~14 OFDM symbols)
# so every slow-time batch carries pilot energy; otherwise coherent integration
# breaks for sparse-in-time references. N batches derived from CPI (Dopp res=1/CPI).
M_BATCH = 16384
N_MIN = 16
SCEN_SPEED = 12.0             # fixed scenario speed [m/s] (isolates motion DIRECTION)
DOPP_NOTCH_HZ = 40.0
PFA = 1e-5
SAMPLES_PER_SRC = 20_000_000  # dense RT sampling so the target is reliably hit
REF_SIZE = 0.30               # fixed reference mesh -> reliable, drone-independent echo


def rcs_scale(drone: Drone, fc: float, apply_carrier: bool = APPLY_CARRIER_RCS) -> float:
    """Deterministic per-drone (and carrier) echo POWER scaling vs the calibration
    mesh, from the LITERATURE-GROUNDED dBsm anchors in drones.py (review fix #4).
    Anchored to the reference drone at FC_ANCHOR -> scale = 1 there (so the N0
    anchor cell is unchanged). Combines:
      * the per-drone term (Mavic > Air3S > Mini, carrier-independent ratio), and
      * the carrier-dependent RCS rise (~+8 dB at C-band) — a real drone's echo
        gains energy at high fc, PARTLY OFFSETTING the carrier path-loss. This is
        why a naive 'carrier dominates' is not identifiable.
    apply_carrier=False freezes the carrier term at FC_ANCHOR -> isolates the
    per-drone term (used by the carrier-decoupling sweep to show propagation only)."""
    rd = rcs_dbsm_at(drone, fc if apply_carrier else FC_ANCHOR)
    r0 = rcs_dbsm_at(DRONES[REF_DRONE_KEY], FC_ANCHOR)
    return 10 ** ((rd - r0) / 10.0)


# --------------------------------------------------------------------------- #
#  Axis A — realistic signal specs
# --------------------------------------------------------------------------- #
@dataclass
class Spec:
    std: str            # 'wifi' | 'lte' | '5g'
    ref: str            # reference structure (pilot-mask name in passive_radar_s2)
    bw: float           # bandwidth [Hz]
    fc: float           # carrier [Hz]
    scs: float          # subcarrier spacing [Hz]
    @property
    def name(self): return f"{self.std}|{self.bw/1e6:.0f}MHz|{self.fc/1e9:.2f}GHz"

SPECS = [
    Spec("wifi", "wifi_preamble", 20e6, 2.4e9, 312.5e3),
    Spec("wifi", "wifi_preamble", 40e6, 5.0e9, 312.5e3),
    Spec("wifi", "wifi_preamble", 80e6, 5.0e9, 312.5e3),
    Spec("lte", "lte_crs", 5e6, 1.8e9, 15e3),
    Spec("lte", "lte_crs", 10e6, 1.8e9, 15e3),
    Spec("lte", "lte_crs", 20e6, 2.6e9, 15e3),
    Spec("5g", "5g_ssb_sparse", 20e6, 3.5e9, 30e3),
    Spec("5g", "5g_ssb_sparse", 50e6, 3.5e9, 30e3),
    Spec("5g", "5g_ssb_sparse", 100e6, 3.5e9, 30e3),
]
SCENARIOS = ["hover", "radial", "tangential", "doppler_switch"]


# --------------------------------------------------------------------------- #
#  Axis C — scenario motion -> velocity vector(s)
# --------------------------------------------------------------------------- #
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


def scenario_velocity(scenario: str, speed: float = SCEN_SPEED):
    """Velocity at the drone anchor. Single vector for hover/radial/tangential;
    list of (frac, vel) segments for doppler_switch (Doppler changes along the CPI)."""
    tx = np.array(GEOM["tx"]); rx = np.array(GEOM["rx"]); p = np.array(GEOM["drone"])
    grad = _unit(p - tx) + _unit(p - rx)         # bistatic range gradient
    radial = _unit(grad)                          # max-Doppler direction
    tang = _unit(np.cross(grad, [0, 0, 1.0]))     # horizontal, ~zero-Doppler
    if scenario == "hover":
        return np.zeros(3)
    if scenario == "radial":
        return speed * radial
    if scenario == "tangential":
        return speed * tang
    if scenario == "doppler_switch":
        return [(0.5, speed * radial), (0.5, speed * tang)]
    raise ValueError(scenario)


def _split_n(total: int, fracs) -> list:
    """Slow-time batch counts per doppler_switch segment, summing EXACTLY to total."""
    ns = []
    for i, f in enumerate(fracs):
        ns.append(total - sum(ns) if i == len(fracs) - 1 else int(round(total * f)))
    return ns


# --------------------------------------------------------------------------- #
#  Per-cell Config (fixed CPI -> M scales with B; N fixed)
# --------------------------------------------------------------------------- #
def cfg_for(spec: Spec, assets="assets") -> tuple[Config, OFDM]:
    cfg = Config()
    cfg.fc, cfg.B = spec.fc, spec.bw
    cfg.M = M_BATCH                                            # batch >> pilot period
    # CPI (elapsed time) is fixed, but the slow-time DEPTH N = B*CPI/M scales with B
    # (~16..610 across the matrix). Coherent gain cancels noise analytically, but the
    # CFAR small-N statistics differ per cell -> reported per row as 'N' (review #9).
    cfg.N = max(N_MIN, int(round(spec.bw * CPI_S / M_BATCH)))  # batches from CPI
    cfg.l_max = max(24, int(round(250.0 * spec.bw / C0)))     # range taps (>=24)
    cfg.tx_pos, cfg.rx_pos, cfg.drone_pos = GEOM["tx"], GEOM["rx"], GEOM["drone"]
    cfg.drone_size = REF_SIZE                                  # fixed reference mesh
    cfg.samples_per_src = SAMPLES_PER_SRC
    cfg.doppler_notch_hz = DOPP_NOTCH_HZ
    cfg.pfa = PFA
    cfg.assets_dir = assets
    # nfft is the controlled quantity (power of 2 near B/SCS); the EFFECTIVE SCS =
    # B/nfft is then within ~4x of the standard SCS, so SCS is NOT exactly held (do
    # not advertise it as a controlled axis — review note e). nfft & effective SCS
    # are recoverable from (B, nfft).
    nfft = 1 << max(8, int(round(np.log2(max(64, spec.bw / spec.scs)))))  # ~B/SCS
    ofdm = OFDM(n_fft=nfft, cp=nfft // 8)
    return cfg, ofdm


# --------------------------------------------------------------------------- #
#  Sionna trace -> physical channel taps h[N, L] + ground-truth
# --------------------------------------------------------------------------- #
def analytic_gt(cfg: Config, velocity) -> dict:
    """Exact ground truth from geometry (robust for all scenarios incl. hover).
    Bistatic delay is relative to the direct path (taps use normalize_delays)."""
    tx = np.array(cfg.tx_pos, float); rx = np.array(cfg.rx_pos, float)
    p = np.array(cfg.drone_pos, float); v = np.array(velocity, float)
    Rtx, Rrx, Rb = (np.linalg.norm(p - tx), np.linalg.norm(p - rx),
                    np.linalg.norm(tx - rx))
    bd = (Rtx + Rrx - Rb) / C0
    # bistatic Doppler; sign matches Sionna paths.doppler / the CAF axis convention
    fD = -float(v @ (_unit(p - tx) + _unit(p - rx))) / (C0 / cfg.fc)
    return dict(bistatic_delay=float(bd), bistatic_range_m=float(bd * C0),
                doppler_hz=fD)


def _solve(cfg: Config, velocity):
    """Sionna PathSolver for (fc, drone size, velocity). Reusable across B."""
    import sionna.rt as rt
    cfg.drone_vel = tuple(float(x) for x in velocity)
    scene = build_scene(cfg)
    return rt.PathSolver()(scene, max_depth=cfg.max_depth, los=True,
                           specular_reflection=True, diffuse_reflection=True,
                           refraction=False, samples_per_src=cfg.samples_per_src,
                           seed=cfg.seed)


def _taps(paths, cfg: Config) -> np.ndarray:
    taps = paths.taps(bandwidth=cfg.B, l_min=0, l_max=cfg.l_max,
                      sampling_frequency=cfg.prf, num_time_steps=cfg.N,
                      normalize_delays=True, out_type="numpy")
    return np.asarray(taps)[0, 0, 0, 0, :, :].astype(np.complex64)


def _solve_and_taps(cfg: Config, velocity) -> tuple[np.ndarray, dict]:
    paths = _solve(cfg, velocity)
    h = _taps(paths, cfg)
    tau = np.asarray(paths.tau).squeeze().ravel()
    gt = analytic_gt(cfg, velocity)
    gt["n_paths"] = int(np.sum(np.isfinite(tau) & (tau > 0)))
    return h, gt


def trace_cell(cfg: Config, scenario: str):
    """Channel for a cell. Single trace for hover/radial/tangential; for doppler_switch
    concatenate slow-time segments with different velocities (Doppler varies)."""
    vel = scenario_velocity(scenario)
    if scenario != "doppler_switch":
        return _solve_and_taps(cfg, vel)
    parts, gts = [], []
    ns = _split_n(cfg.N, [f for f, _ in vel])
    for (frac, v), ni in zip(vel, ns):
        c2 = copy.copy(cfg); c2.N = ni
        h, gt = _solve_and_taps(c2, v)
        parts.append(h); gts.append(gt)
    h = np.concatenate(parts, axis=0)               # sums to cfg.N exactly
    gt = _doppler_switch_gt(cfg, vel)               # SHARED rule with run_matrix (review g)
    gt["n_paths"] = sum(g["n_paths"] for g in gts)  # total RT paths over segments (diag)
    return h, gt


# --------------------------------------------------------------------------- #
#  Measured-SCR engine: fixed absolute noise (anchored), SCR/Pd fall out
# --------------------------------------------------------------------------- #
def run_cell(cfg: Config, ofdm: OFDM, spec: Spec, h, gt, n0_density, n_trials,
             seed=1, rcs_scale=1.0):
    L = h.shape[1]
    drone_tap = max(0, min(L - 1, int(round(gt["bistatic_delay"] * cfg.fs))))
    if rcs_scale != 1.0:                           # scale the moving (drone) echo only
        h_static = h.mean(axis=0, keepdims=True)
        h = (h_static + np.sqrt(rcs_scale) * (h - h_static)).astype(np.complex64)
    noise_pow = n0_density * cfg.B                 # P_n = N0 * B (fixed density)
    scrs, hits, fars, ntests, pkR, pkD = [], [], [], [], [], []
    rd_show = None
    for t in range(n_trials):
        rng = np.random.default_rng(seed + 7919 * t)
        s_full, s_ref, _ = synth_ofdm(cfg, ofdm, spec.ref, rng)
        X = surveillance(cfg, s_full, h, drone_tap, 0.0, rng, noise_pow=noise_pow)
        rd, range_axis, dopp_axis = caf_range_doppler(cfg, X, s_ref.reshape(cfg.N, cfg.M), mti=False)
        power = np.abs(rd) ** 2
        det, *_ = ca_cfar_2d(power, cfg, dopp_axis)
        mt = rd_metrics(power, range_axis, dopp_axis, gt, cfg, det)
        scrs.append(mt["scr_db"]); hits.append(mt["hit"]); fars.append(mt["n_fa"])
        ntests.append(mt["n_test"]); pkR.append(mt["pk_r"]); pkD.append(mt["pk_d"])
        if t == 0:
            rd_show = (rd, range_axis, dopp_axis, det)
    scrs = np.array(scrs, float)
    # Cells whose target Doppler is inside the clutter notch (hover / pure
    # tangential) are STRUCTURALLY DEGENERATE: the ideal canceller + Doppler notch
    # remove the zero-Doppler echo by construction, so SCR is NaN and the drone
    # (RCS) axis carries no information there (review fixes #3, #5, i). Report it.
    degenerate = bool(abs(gt["doppler_hz"]) < cfg.doppler_notch_hz)
    valid = scrs[np.isfinite(scrs)]
    scr_mean = float(valid.mean()) if valid.size else float("nan")
    ci = (1.96 * float(valid.std(ddof=1)) / np.sqrt(valid.size)
          if valid.size > 1 else float("nan"))            # sample std (ddof=1)
    return dict(scr_db=scr_mean, scr_ci=ci, degenerate=degenerate,
                pd=float(np.mean(hits)), far=float(np.sum(fars) / max(1, np.sum(ntests))),
                pkR_std=float(np.std(pkR)), pkD_std=float(np.std(pkD)),
                gt=gt, range_res_m=float(cfg.range_res_m), N=int(cfg.N),
                dopp_res_hz=float(cfg.doppler_res_hz), drone_tap=int(drone_tap)), rd_show


def anchor_noise_density(ref_spec, ref_scen, snr_ref_db, assets="assets", n_seed=5):
    """Fix N0 so the reference-mesh cell sits at snr_ref_db per-sample SNR; used
    for ALL cells (fixed noise budget). Then SCR/Pd fall out of spec/RCS/motion.

    The anchor echo power is AVERAGED over n_seed independent RT realizations
    (review fix #7): a single diffuse Monte-Carlo draw would shift EVERY cell's
    SCR by the same offset, and its variance would be missing from the CIs. We
    report the across-seed spread so that systematic RT noise is visible."""
    cfg, _ = cfg_for(ref_spec, assets)
    pe, gt = [], None
    for s in range(n_seed):
        c2 = copy.copy(cfg); c2.seed = 1 + s
        h, gt = trace_cell(c2, ref_scen)
        tap = max(0, min(h.shape[1] - 1, int(round(gt["bistatic_delay"] * c2.fs))))
        pe.append(float(np.mean(np.abs(h[:, tap]) ** 2)))
    pe = np.array(pe, float)
    p_echo = float(pe.mean())
    n0 = p_echo / (cfg.B * 10 ** (snr_ref_db / 10.0))
    spread = float(10 * np.log10(pe.max() / pe.min())) if pe.min() > 0 else 0.0
    info = dict(ref_cell=ref_spec.name, ref_scen=ref_scen, snr_ref_db=snr_ref_db,
                n_seed=n_seed, p_echo_mean=p_echo, p_echo_std=float(pe.std(ddof=1)),
                across_seed_spread_db=spread, n0_density=n0)
    print(f"[anchor] ref-mesh {ref_spec.name} {ref_scen}: p_echo={p_echo:.3e} "
          f"+/-{pe.std(ddof=1):.1e} ({n_seed} seeds, spread={spread:.2f}dB) "
          f"N0={n0:.3e} (snr_ref={snr_ref_db}dB, fD={gt['doppler_hz']:.0f}Hz)")
    return n0, info


# --------------------------------------------------------------------------- #
#  Matrix (Axis A x B x C) with carrier-cached tracing
# --------------------------------------------------------------------------- #
SPECS_RUN = [SPECS[0], SPECS[2], SPECS[3], SPECS[5], SPECS[6], SPECS[8]]   # span BW/fc
DRONES_RUN = ["mavic4pro", "mini5pro"]


def _doppler_switch_gt(cfg, vels):
    """GT for the doppler-switch scenario = the segment with the largest |fD|.
    Shared by trace_cell and run_matrix so the two cannot diverge (review fix g)."""
    return max((analytic_gt(cfg, v) for _, v in vels),
               key=lambda g: abs(g["doppler_hz"]))


def run_matrix(specs, drone_keys, scenarios, n0, trials, assets="assets", seed=1):
    """Trace once per (scenario, carrier) — geometry is drone-independent (fixed
    reference mesh); the drone (Axis B) enters only as an RCS scaling. Cheaper and
    monotonic in RCS."""
    rows = []
    for sc in scenarios:
        vels = scenario_velocity(sc)
        by_fc = {}
        for sp in specs:
            by_fc.setdefault(sp.fc, []).append(sp)
        for fc, sgroup in by_fc.items():
            geom, _ = cfg_for(sgroup[0], assets)
            paths = ([_solve(geom, vels)] if sc != "doppler_switch"
                     else [_solve(copy.copy(geom), v) for _, v in vels])
            for sp in sgroup:
                cfg, ofdm = cfg_for(sp, assets)
                if sc != "doppler_switch":
                    h = _taps(paths[0], cfg); gt = analytic_gt(cfg, vels)
                else:
                    parts = []
                    ns = _split_n(cfg.N, [f for f, _ in vels])
                    for (frac, v), pp, ni in zip(vels, paths, ns):
                        c2 = copy.copy(cfg); c2.N = ni
                        parts.append(_taps(pp, c2))
                    h = np.concatenate(parts, 0)              # sums to cfg.N exactly
                    gt = _doppler_switch_gt(cfg, vels)        # max-|fD| segment (shared rule)
                for dk in drone_keys:
                    res, _ = run_cell(cfg, ofdm, sp, h, gt, n0, trials, seed,
                                      rcs_scale=rcs_scale(DRONES[dk], sp.fc))
                    rows.append(dict(std=sp.std, bw_mhz=sp.bw / 1e6, fc_ghz=sp.fc / 1e9,
                                     drone=DRONES[dk].name, scenario=sc,
                                     scr_db=res["scr_db"], scr_ci=res["scr_ci"],
                                     degenerate=res["degenerate"], N=res["N"],
                                     pd=res["pd"], far=res["far"],
                                     range_res_m=res["range_res_m"],
                                     fD_hz=gt["doppler_hz"], R_m=gt["bistatic_range_m"]))
                    sd = "SCR=  nan" if not np.isfinite(res["scr_db"]) else f"SCR={res['scr_db']:5.1f}"
                    print(f"[cell] {DRONES[dk].name[:12]:12s} {sc:14s} {sp.name:22s} "
                          f"fD={gt['doppler_hz']:+6.0f}Hz {sd} Pd={res['pd']:.2f}"
                          f"{'  [degenerate]' if res['degenerate'] else ''}")
    return rows


def _save_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


def plot_matrix(rows, outdir, ref_drone="DJI Mavic 4 Pro"):
    import collections
    std_col = {"wifi": "tab:blue", "lte": "tab:orange", "5g": "tab:green"}
    scen = SCENARIOS
    ref = [r for r in rows if r["drone"] == ref_drone]

    # (1) Pd vs BW, faceted by scenario, lines per standard
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True, sharey=True)
    for ax, sc in zip(axes, scen):
        for std in ("wifi", "lte", "5g"):
            pts = sorted([(r["bw_mhz"], r["pd"]) for r in ref
                          if r["scenario"] == sc and r["std"] == std])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=std_col[std], label=std)
        ax.set_title(f"{sc}"); ax.set_xlabel("Bandwidth [MHz]"); ax.set_ylim(-.03, 1.03)
        ax.grid(alpha=.3); ax.set_xscale("log")
    axes[0].set_ylabel("Pd"); axes[0].legend(fontsize=8)
    fig.suptitle(f"Pd vs bandwidth, faceted by scenario ({ref_drone}) — radial detects, "
                 f"hover/tangential blind (bulk Doppler)", fontsize=11)
    fig.savefig(os.path.join(outdir, "phase1_pd_vs_bw.png"), dpi=140); plt.close(fig)

    # (2) scenario-difficulty + (3) SCR vs BW (radial)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    by_sc = collections.defaultdict(list)
    for r in ref:
        by_sc[r["scenario"]].append(r["pd"])
    ax[0].bar(range(len(scen)), [np.mean(by_sc[s]) for s in scen], color="steelblue")
    ax[0].set_xticks(range(len(scen))); ax[0].set_xticklabels(scen)
    ax[0].set_ylabel("mean Pd (over specs)"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Scenario difficulty (bulk Doppler)")
    for std in ("wifi", "lte", "5g"):
        pts = sorted([(r["bw_mhz"], r["scr_db"]) for r in ref
                      if r["scenario"] == "radial" and r["std"] == std])
        if pts:
            xs, ys = zip(*pts); ax[1].plot(xs, ys, "o-", color=std_col[std], label=std)
    ax[1].set_xlabel("Bandwidth [MHz]"); ax[1].set_ylabel("SCR [dB]")
    ax[1].set_title("SCR vs bandwidth (radial)"); ax[1].set_xscale("log")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.savefig(os.path.join(outdir, "phase1_scenario_scr.png"), dpi=140); plt.close(fig)

    # (4) per-drone Pd by scenario (radial spec-averaged) heatmap-ish
    drones = sorted({r["drone"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    w = 0.8 / len(drones)
    for i, dn in enumerate(drones):
        vals = [np.mean([r["pd"] for r in rows if r["drone"] == dn and r["scenario"] == s]) for s in scen]
        ax.bar(np.arange(len(scen)) + i * w, vals, w, label=dn[:14])
    ax.set_xticks(np.arange(len(scen)) + w * (len(drones) - 1) / 2); ax.set_xticklabels(scen)
    ax.set_ylabel("mean Pd"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("Per-drone detectability by scenario")
    fig.savefig(os.path.join(outdir, "phase1_per_drone.png"), dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Decoupled single-factor sweeps (review fix #1): vary ONE of {carrier, bandwidth,
#  reference structure} with the other two FIXED, so each factor's effect on SCR is
#  IDENTIFIABLE. The main matrix confounds std=reference=carrier (collinear), so a
#  single-factor 'carrier dominates' claim is not supportable from it.
# --------------------------------------------------------------------------- #
DEC_REF = "lte_crs"                              # fixed reference for fc/bw sweeps
DEC_SCS = 15e3
DEC_FCS = [1.8e9, 2.4e9, 3.5e9, 5.0e9]          # carrier-only (B, ref fixed)
DEC_BWS = [20e6, 50e6, 100e6]                   # bandwidth-only (fc, ref fixed)
DEC_REFS = ["5g_ssb_sparse", "lte_crs", "wifi_preamble", "5g_dmrs_prs_rich"]  # ref-only


def _known_fraction(spec) -> float:
    """Pilot-mask density of a spec's reference structure (ref-only x-axis).
    n_sym=240 spans >=2 SSB periods so the sparsest mask is represented."""
    _, ofdm = cfg_for(spec)
    return float(pilot_mask(spec.ref, 240, ofdm.n_fft).mean())


def run_decouple(n0, trials, assets="assets", seed=1, drone=REF_DRONE_KEY, scen="radial"):
    """Three orthogonal single-factor sweeps under the SAME fixed N0/geometry/CPI."""
    dn = DRONES[drone]

    def cell(sp, apply_carrier=True):
        cfg, ofdm = cfg_for(sp, assets)
        h, gt = trace_cell(cfg, scen)
        res, _ = run_cell(cfg, ofdm, sp, h, gt, n0, trials, seed,
                          rcs_scale=rcs_scale(dn, sp.fc, apply_carrier=apply_carrier))
        return dict(scr_db=res["scr_db"], scr_ci=res["scr_ci"], pd=res["pd"],
                    far=res["far"], fc_ghz=sp.fc / 1e9, bw_mhz=sp.bw / 1e6, ref=sp.ref,
                    fD_hz=gt["doppler_hz"], R_m=gt["bistatic_range_m"], N=res["N"])

    out = {"carrier": [], "carrier_propagation_only": [], "bw": [], "bw_ssb": [], "reference": []}
    # A) carrier-only, RADAR-EQUATION normalized (review #4): the calibration cube's
    # RCS(fc) is non-physical (the raw per-fc trace is non-monotonic), so we keep RT
    # only for geometry/clutter STRUCTURE and set the moving-echo POWER to the radar
    # equation Pr ∝ λ²·RCS(fc) relative to the S-band anchor fc0. run_cell then adds
    # the literature RCS(fc) (net) or freezes it (propagation-only = pure λ²).
    FC0 = DEC_FCS[0]
    sp0 = Spec("carrier", DEC_REF, 20e6, FC0, DEC_SCS)
    cfg0, _ = cfg_for(sp0, assets)
    h0, gt0 = trace_cell(cfg0, scen)
    tap0 = max(0, min(h0.shape[1] - 1, int(round(gt0["bistatic_delay"] * cfg0.fs))))
    p0 = float(np.mean(np.abs((h0 - h0.mean(0, keepdims=True))[:, tap0]) ** 2))

    def carrier_cell(fc, apply_carrier):
        sp = Spec("carrier", DEC_REF, 20e6, fc, DEC_SCS)
        cfg, ofdm = cfg_for(sp, assets)
        h, gt = trace_cell(cfg, scen)
        tap = max(0, min(h.shape[1] - 1, int(round(gt["bistatic_delay"] * cfg.fs))))
        hs = h.mean(0, keepdims=True)
        p_cur = float(np.mean(np.abs((h - hs)[:, tap]) ** 2))
        amp = np.sqrt(p0 * (FC0 / fc) ** 2 / max(p_cur, 1e-30))   # remove cube freq resp; impose λ²
        h_norm = (hs + amp * (h - hs)).astype(np.complex64)
        res, _ = run_cell(cfg, ofdm, sp, h_norm, gt, n0, trials, seed,
                          rcs_scale=rcs_scale(dn, fc, apply_carrier=apply_carrier))
        return dict(scr_db=res["scr_db"], scr_ci=res["scr_ci"], pd=res["pd"],
                    far=res["far"], fc_ghz=fc / 1e9, bw_mhz=20.0, ref=DEC_REF,
                    fD_hz=gt["doppler_hz"], R_m=gt["bistatic_range_m"], N=res["N"])

    for fc in DEC_FCS:
        net = carrier_cell(fc, apply_carrier=True)
        prop = carrier_cell(fc, apply_carrier=False)
        out["carrier"].append(net); out["carrier_propagation_only"].append(prop)
        print(f"[dec-fc]  {fc/1e9:>4.1f} GHz  net SCR={net['scr_db']:5.1f}  "
              f"propagation-only(λ²)={prop['scr_db']:5.1f}  Pd={net['pd']:.2f}")
    # B) bandwidth-only with a BAND-FILLING reference (LTE-CRS, comb -> density is
    # scale-invariant ~4.8% regardless of B): isolates the PURE bandwidth effect.
    for bw in DEC_BWS:
        sp = Spec("bw", DEC_REF, bw, 3.5e9, DEC_SCS)
        r = cell(sp); r["known_frac"] = _known_fraction(sp)
        out["bw"].append(r)
        print(f"[dec-bw]   {bw/1e6:>4.0f} MHz  CRS known={r['known_frac']*100:5.2f}%  SCR={r['scr_db']:5.1f}  Pd={r['pd']:.2f}")
    # B2) bandwidth with a FIXED-BLOCK reference (5G SSB = 240-SC block): its density
    # DILUTES as B (nfft) grows, so 'bandwidth' here folds in reference sparsity. This
    # is why the matrix's 5G 20->100MHz SCR drop is NOT a pure bandwidth effect but
    # SSB dilution (review-3 point). Shown alongside (B) to make the interaction explicit.
    for bw in DEC_BWS:
        sp = Spec("bw_ssb", "5g_ssb_sparse", bw, 3.5e9, 30e3)
        r = cell(sp); r["known_frac"] = _known_fraction(sp)
        out["bw_ssb"].append(r)
        print(f"[dec-bwssb]{bw/1e6:>4.0f} MHz  SSB known={r['known_frac']*100:5.2f}%  SCR={r['scr_db']:5.1f}  Pd={r['pd']:.2f}")
    # C) reference-only: fc=3.5 GHz + B=20 MHz fixed, vary the pilot mask
    for rf in DEC_REFS:
        r = cell(Spec("ref", rf, 20e6, 3.5e9, 30e3))
        r["known_frac"] = _known_fraction(Spec("ref", rf, 20e6, 3.5e9, 30e3))
        out["reference"].append(r)
        print(f"[dec-ref] {rf:18s} known={r['known_frac']*100:5.2f}%  SCR={r['scr_db']:5.1f}  Pd={r['pd']:.2f}")

    def span(rs):
        v = [x["scr_db"] for x in rs if np.isfinite(x["scr_db"])]
        return float(max(v) - min(v)) if v else float("nan")
    out["spans_db"] = dict(
        carrier_net=span(out["carrier"]),
        carrier_propagation=span(out["carrier_propagation_only"]),
        bandwidth=span(out["bw"]), bandwidth_ssb=span(out["bw_ssb"]),
        reference=span(out["reference"]))
    print(f"[dec-span] ΔSCR  carrier(net)={out['spans_db']['carrier_net']:.1f}  "
          f"carrier(prop)={out['spans_db']['carrier_propagation']:.1f}  "
          f"bw(CRS)={out['spans_db']['bandwidth']:.1f}  "
          f"bw(SSB-dilution)={out['spans_db']['bandwidth_ssb']:.1f}  "
          f"reference={out['spans_db']['reference']:.1f} dB")
    return out


def plot_decouple(dec, outdir):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    fcs = [r["fc_ghz"] for r in dec["carrier"]]
    ax[0].plot(fcs, [r["scr_db"] for r in dec["carrier"]], "o-", color="tab:red",
               label=f"net (incl. C-band RCS rise)  ΔSCR={dec['spans_db']['carrier_net']:.1f} dB")
    ax[0].plot(fcs, [r["scr_db"] for r in dec["carrier_propagation_only"]], "s--",
               color="0.5",
               label=f"propagation only (λ²)  ΔSCR={dec['spans_db']['carrier_propagation']:.1f} dB")
    ax[0].set_xlabel("carrier  fc [GHz]"); ax[0].set_ylabel("SCR [dB]")
    ax[0].set_title("(A) carrier-only\n(ref=LTE-CRS, B=20 MHz fixed)")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=7.5)

    bws = [r["bw_mhz"] for r in dec["bw"]]
    ax[1].plot(bws, [r["scr_db"] for r in dec["bw"]], "o-", color="tab:orange",
               label=f"CRS ref (band-filling, ~5%): ΔSCR={dec['spans_db']['bandwidth']:.1f} dB")
    if dec.get("bw_ssb"):
        bws2 = [r["bw_mhz"] for r in dec["bw_ssb"]]
        ax[1].plot(bws2, [r["scr_db"] for r in dec["bw_ssb"]], "s--", color="tab:green",
                   label=f"SSB ref (fixed block, dilutes): ΔSCR={dec['spans_db'].get('bandwidth_ssb', float('nan')):.1f} dB")
    ax[1].set_xscale("log"); ax[1].set_xlabel("bandwidth  B [MHz]"); ax[1].set_ylabel("SCR [dB]")
    ax[1].set_title("(B) bandwidth-only (fc=3.5 GHz fixed)\npure BW ~0; SSB loss = dilution, not BW")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)

    kf = np.array([r["known_frac"] * 100 for r in dec["reference"]])
    sc = np.array([r["scr_db"] for r in dec["reference"]])
    o = np.argsort(kf)
    ax[2].plot(kf[o], sc[o], "o-", color="tab:green")
    for r in dec["reference"]:
        ax[2].annotate(r["ref"].replace("_", "\n"), (r["known_frac"] * 100, r["scr_db"]),
                       fontsize=6, xytext=(4, 2), textcoords="offset points")
    ax[2].set_xscale("log"); ax[2].set_xlabel("known reference REs [%]"); ax[2].set_ylabel("SCR [dB]")
    ax[2].set_title(f"(C) reference-only  ΔSCR={dec['spans_db']['reference']:.1f} dB\n"
                    "(fc=3.5 GHz, B=20 MHz fixed)")
    ax[2].grid(alpha=.3)
    fig.suptitle("Decoupled single-factor sweeps — each axis varied ALONE -> ΔSCR identifies each factor's "
                 "true contribution\n(the main matrix confounds carrier=bandwidth=reference)", fontsize=11)
    fig.savefig(os.path.join(outdir, "phase1_decouple.png"), dpi=140); plt.close(fig)


def run_config_dict(snr_ref, trials, anchor_info):
    """Serialized fixed budget so any result is reproducible (review fix #8)."""
    return dict(geometry=GEOM, fc_anchor_hz=FC_ANCHOR, ref_drone=REF_DRONE_KEY,
                apply_carrier_rcs=APPLY_CARRIER_RCS, cpi_s=CPI_S, m_batch=M_BATCH,
                n_min=N_MIN, scen_speed_ms=SCEN_SPEED, dopp_notch_hz=DOPP_NOTCH_HZ,
                pfa=PFA, samples_per_src=SAMPLES_PER_SRC, ref_size_m=REF_SIZE,
                hit_tol_m=Config().hit_tol_m, snr_ref_db=snr_ref, trials=trials,
                cfar_exclude_notch=Config().cfar_exclude_notch, anchor=anchor_info,
                scenarios=SCENARIOS, scs_note="effective SCS = B/nfft (power-of-2 "
                "nfft), within ~4x of the standard SCS — nfft is the controlled "
                "quantity, not SCS exactly (review note e)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="minimal",
                   choices=["minimal", "scenarios", "matrix", "decouple"])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--snr_ref", type=float, default=-12.0)   # anchor: ref cell ~Pd=1
    p.add_argument("--assets", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets"))
    p.add_argument("--outdir", default=os.environ.get(           # auto-branch like
        "PR_OUTDIR",                                             # s2/stage1 (review fix f)
        "/data/public/jeong/sionna/phase1"
        if os.access("/data/public/jeong", os.W_OK)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")))
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    def _scr(r):
        return "  nan" if not np.isfinite(r["scr_db"]) else f"{r['scr_db']:5.1f}±{r['scr_ci']:.1f}"

    if a.mode == "minimal":
        spec = Spec("lte", "lte_crs", 10e6, 1.8e9, 15e3)
        cfg, ofdm = cfg_for(spec, a.assets)
        print(f"[minimal] {spec.name} ref-mesh radial  M={cfg.M} N={cfg.N} l_max={cfg.l_max} "
              f"nfft={ofdm.n_fft} range_res={cfg.range_res_m:.1f}m dopp_res={cfg.doppler_res_hz:.1f}Hz "
              f"PRF={cfg.prf:.0f}Hz CPI={cfg.cpi_s*1e3:.0f}ms")
        h, gt = trace_cell(cfg, "radial")
        print(f"[minimal] GT bistatic R={gt['bistatic_range_m']:.1f}m fD={gt['doppler_hz']:.1f}Hz "
              f"npaths={gt['n_paths']}")
        n0, _info = anchor_noise_density(spec, "radial", a.snr_ref, a.assets)
        res, _ = run_cell(cfg, ofdm, spec, h, gt, n0, a.trials)
        print(f"[minimal] SCR={_scr(res)}dB  Pd={res['pd']:.2f}  FAR={res['far']:.1e}")

    elif a.mode == "scenarios":
        # verify the motion axis: Doppler & detectability per scenario (one spec)
        spec = Spec("lte", "lte_crs", 10e6, 1.8e9, 15e3)
        cfg, ofdm = cfg_for(spec, a.assets)
        n0, _info = anchor_noise_density(spec, "radial", a.snr_ref, a.assets)
        for sc in SCENARIOS:
            h, gt = trace_cell(cfg, sc)
            res, _ = run_cell(cfg, ofdm, spec, h, gt, n0, a.trials)
            print(f"[scen] {sc:14s} fD={gt['doppler_hz']:+7.1f}Hz  SCR={_scr(res)}dB  "
                  f"Pd={res['pd']:.2f}{'  [degenerate]' if res['degenerate'] else ''}")

    elif a.mode == "decouple":
        # review fix #1: isolate carrier / bandwidth / reference one at a time.
        n0, info = anchor_noise_density(Spec("lte", "lte_crs", 10e6, 1.8e9, 15e3),
                                        "radial", a.snr_ref, a.assets)
        dec = run_decouple(n0, a.trials, a.assets)
        dec["config"] = run_config_dict(a.snr_ref, a.trials, info)
        json.dump(dec, open(os.path.join(a.outdir, "phase1_decouple.json"), "w"), indent=1)
        plot_decouple(dec, a.outdir)
        print(f"[out] {a.outdir}/phase1_decouple.json + phase1_decouple.png")

    elif a.mode == "matrix":
        # Axis A (spec) x B (drone) x C (scenario), measured SCR under fixed budget.
        # Anchor noise once (LTE 10MHz radial Mavic ~ Pd=1), then read everything off.
        n0, info = anchor_noise_density(Spec("lte", "lte_crs", 10e6, 1.8e9, 15e3),
                                        "radial", a.snr_ref, a.assets)
        rows = run_matrix(SPECS_RUN, DRONES_RUN, SCENARIOS, n0, a.trials, a.assets)
        out = dict(config=run_config_dict(a.snr_ref, a.trials, info), rows=rows)
        _save_csv(rows, os.path.join(a.outdir, "phase1_matrix.csv"))
        json.dump(out, open(os.path.join(a.outdir, "phase1_matrix.json"), "w"), indent=1)
        plot_matrix(rows, a.outdir)
        print(f"[out] {a.outdir}/phase1_matrix.csv + phase1_matrix.json + phase1_*.png "
              f"({len(rows)} cells)")
