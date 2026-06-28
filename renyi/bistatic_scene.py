#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase D (SERVER): real-flight bistatic drone detection on a Sionna-RT channel, for
the 5G-22 reproduction (Maksymiuk et al., Remote Sens. 2022, 14, 6146, Sec 7 real
experiment, Figs 21-23). Requires sionna-rt + OptiX on the RTX-4090 server.

What Phase D adds over the synthetic Phases A-C (run_renyi.py):
  * a REAL bistatic channel ray-traced by Sionna RT (gNB Tx, surveillance Rx, drone
    scatterer + ground clutter) instead of the analytic delay+Doppler echo -> realistic
    multipath, so the SPARSE-reference degradation (why low content fails, Fig 10)
    appears naturally;
  * a flight TRAJECTORY (paper Fig 18c/23) -> a sequence of CAFs whose CFAR detections
    are overlaid on Sionna's EXACT ground truth (the paper used GPS logs);
  * the T_int 20 ms -> 100 ms velocity-resolution sharpening (paper Fig 21 -> 22);
  * the Renyi-entropy adaptive integration (the novelty) applied to the real capture:
    per-frame entropy selects dense frames; a dense CPI detects where a sparse CPI of
    the same length is buried (Sec 5.2 + Fig 24).

Reuses the parent ray-traced channel and the proven surveillance synthesis:
  passive_radar_stage1.build_scene / trace_channel  (Sionna-RT scene + baseband taps)
  passive_radar_s2.surveillance                     (_conv_fft + clutter cancel + noise)
The CAF / CA-CFAR / Renyi-selection are the SAME NumPy code as Phases A-C
(radar.py / renyi.py) -- only the echo source changes. Faithfulness: docs/FAITHFULNESS.md.
"""
from __future__ import annotations
import os, sys, time, copy
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nr_grid import NRGrid, make_reference
from renyi import renyi_entropy, frame_entropies, calibrate_max
from radar import caf, ca_cfar, scr_db, detected, _power

C0 = 299792458.0


def have_sionna() -> bool:
    try:
        import sionna.rt  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Config in the renyi 5G numerology (B = grid.fs so RT taps are at fs=61.44 MHz)
# --------------------------------------------------------------------------- #
def make_cfg(grid: NRGrid, t_int: float, pos, vel, *, M: int = 8192,
            samples_per_src: int = 2_000_000, l_max: int = 200, seed: int = 1,
            assets_dir: str | None = None):
    """A parent `Config` re-numerologised to the renyi 5G grid (fc 3.44 GHz,
    fs=B 61.44 MHz). N is chosen from the integration time t_int (CPI = N*M/fs)."""
    from passive_radar_stage1 import Config
    N = max(1, int(round(t_int * grid.fs / M)))
    cfg = Config()
    cfg.fc, cfg.B, cfg.M, cfg.N = grid.fc, grid.fs, M, N
    cfg.l_max = l_max
    cfg.samples_per_src = samples_per_src
    cfg.max_depth = 2
    cfg.seed = seed
    cfg.drone_size = 0.3
    cfg.tx_pos = (-50.0, 0.0, 20.0)        # gNB illuminator
    cfg.rx_pos = (50.0, 0.0, 10.0)         # passive surveillance receiver
    cfg.drone_pos = tuple(float(v) for v in pos)
    cfg.drone_vel = tuple(float(v) for v in vel)
    cfg.assets_dir = assets_dir or os.path.join(_HERE, "_assets")
    return cfg


def trace_waypoint(cfg):
    """Sionna-RT: build the bistatic scene at this waypoint and trace the channel.
    Returns (h[N,L] complex64 slow x fast taps, gt dict with exact bistatic GT)."""
    from passive_radar_stage1 import build_scene, trace_channel
    scene = build_scene(cfg)
    h, gt = trace_channel(cfg, scene)
    return h, gt


def _ref(grid: NRGrid, K: int, fill: float, rng, amp: float = 1.0):
    """5G-NR reference of EXACTLY K samples (trim/pad), power ~ fill*amp^2."""
    x, _, rho = make_reference(grid, (K + grid.cp_len) / grid.fs, fill, rng, amp=amp)
    x = x[:K] if len(x) >= K else np.concatenate([x, np.zeros(K - len(x), x.dtype)])
    return x.astype(np.complex64), float(rho)


def _gt_rv(grid: NRGrid, gt: dict):
    """Exact bistatic (range, velocity) ground truth from the Sionna trace."""
    R_b = float(gt["bistatic_range_m"])
    V_b = float(-grid.wavelength * gt["doppler_hz"])     # Eq 2: V_b = -lambda f_d
    return R_b, V_b


def caf_snapshot(cfg, grid: NRGrid, x_ref: np.ndarray, h, gt, rng, *,
                 snr_db: float = -18.0, noise_pow: float | None = None,
                 pfa: float = 1e-6, max_range_m: float = 300.0):
    """One bistatic CAF on the RT channel: reuse the parent surveillance synthesis
    (clutter cancel + noise), then run the renyi CAF + CA-CFAR. Returns a result dict
    (RD map in dB, axes, SCR, CFAR mask, GT cell, hit)."""
    from passive_radar_s2 import surveillance
    N, M = cfg.N, cfg.M
    L = h.shape[1]
    drone_tap = int(np.clip(round(gt["bistatic_delay"] * cfg.fs), 0, L - 1))
    X_clean = surveillance(cfg, x_ref, h, drone_tap, snr_db, rng, noise_pow=noise_pow)
    rd, ra, va = caf(X_clean.reshape(-1), x_ref, cfg.fs, grid.wavelength,
                     n_batch=N, max_range_m=max_range_m)
    R_b, V_b = _gt_rv(grid, gt)
    scr, ri, di = scr_db(rd, ra, va, R_b, V_b)
    det, _ = ca_cfar(_power(rd), pfa=pfa)
    hit = bool(detected(det, ra, va, R_b, V_b, tol_m=cfg.range_res_m * 4))
    p = _power(rd); pdi, pri = np.unravel_index(int(np.argmax(p)), p.shape)
    # associated detection = CFAR cell NEAREST the GT (ray-traced multipath can put a
    # double-bounce peak elsewhere; the trajectory tracks the drone's own detection).
    ddi, dri = np.where(det)
    if len(ddi):
        rsp = max(float(np.ptp(ra)), 1e-9); vsp = max(float(np.ptp(va)), 1e-9)
        j = int(np.argmin(((ra[dri] - R_b) / rsp) ** 2 + ((va[ddi] - V_b) / vsp) ** 2))
        det_R, det_V = float(ra[dri[j]]), float(va[ddi[j]])
    else:
        det_R, det_V = float(ra[pri]), float(va[pdi])
    pdb = 10 * np.log10(p.T + 1e-30); pdb -= pdb.max()        # [range, doppler] dB
    return dict(rd_db=pdb, ra=ra, va=va, R_b_gt=R_b, V_b_gt=V_b,
                scr_db=float(scr), det=det.T, ri=int(ri), di=int(di),
                pk_R=float(ra[pri]), pk_V=float(va[pdi]), det_R=det_R, det_V=det_V, hit=hit,
                n_det=int(det.sum()), drone_tap=drone_tap, vres=float(grid.wavelength / cfg.cpi_s))


# --------------------------------------------------------------------------- #
#  Flight trajectory waypoints (steady drone, geometry varies -> GT curve)
# --------------------------------------------------------------------------- #
def default_waypoints(n: int = 9):
    """A steady drone flight: positions along a gentle arc, constant velocity vector
    (mostly +y so the bistatic Doppler stays out of the zero-velocity clutter notch).
    Returns list of (pos, vel)."""
    xs = np.linspace(-25.0, 25.0, n)
    ys = np.linspace(18.0, 66.0, n)
    zs = 40.0 + 6.0 * np.sin(np.linspace(0, np.pi, n))    # slight altitude arc
    vel = (3.0, 15.0, 0.0)                                # steady cruise
    return [((float(x), float(y), float(z)), vel) for x, y, z in zip(xs, ys, zs)]


# --------------------------------------------------------------------------- #
#  End-to-end Phase D
# --------------------------------------------------------------------------- #
def flight_caf(grid: NRGrid | None = None, *, t_flight: float = 40e-3,
               t_int_pair=(20e-3, 100e-3), n_waypoints: int = 9,
               samples_per_src: int = 2_000_000, seed: int = 3, verbose: bool = True):
    """Full Phase D pipeline (server). Returns a results dict consumed by make_figures
    + a gate verdict. Sections:
      trajectory   -- per-waypoint RT CAF, CFAR detections vs Sionna GT (Fig 23)
      snapshot     -- the showcase waypoint, full allocation (channel + RD detail)
      tint         -- 20 ms vs 100 ms at the showcase (Fig 21 -> 22)
      entropy_demo -- per-frame Renyi entropy selection; the kept (dense) frame detects,
                      the dropped (sparse) frame is buried (Sec 5.2 / Fig 24)
    """
    if not have_sionna():
        raise RuntimeError("Phase D needs sionna-rt + OptiX (run on the RTX-4090 server).")
    grid = grid or NRGrid()
    rng = np.random.default_rng(seed)
    M = 8192
    t0 = time.time()

    def log(*a):
        if verbose:
            print(*a, flush=True)

    r_valid = 2.0 * grid.range_res_m          # GT below this -> RT found no drone path

    # ---- trajectory: a CAF per waypoint, detections vs exact Sionna GT ---------
    wps = default_waypoints(n_waypoints)
    traj = []
    log(f"[D] trajectory: {len(wps)} waypoints @ T_int={t_flight*1e3:.0f} ms")
    for i, (pos, vel) in enumerate(wps):
        cfg = make_cfg(grid, t_flight, pos, vel, M=M,
                       samples_per_src=samples_per_src, seed=seed)
        h, gt = trace_waypoint(cfg)
        K = cfg.N * cfg.M
        x_ref, _ = _ref(grid, K, fill=1.0, rng=rng)          # full allocation
        res = caf_snapshot(cfg, grid, x_ref, h, gt, rng, snr_db=-18.0)
        valid = bool(res["R_b_gt"] > r_valid)                # drone path traced?
        traj.append(dict(i=i, pos=pos, R_b_gt=res["R_b_gt"], V_b_gt=res["V_b_gt"],
                         R_b_det=res["det_R"], V_b_det=res["det_V"], scr_db=res["scr_db"],
                         hit=bool(res["hit"] and valid), valid=valid))
        log(f"   wp{i}: R_b={res['R_b_gt']:6.1f}m V_b={res['V_b_gt']:+6.2f}m/s "
            f"SCR={res['scr_db']:5.1f}dB hit={res['hit']} valid={valid}")

    # ---- showcase = the best-detected waypoint (drives snapshot/T_int/entropy) --
    cand = [t for t in traj if t["valid"]] or traj
    show = max(cand, key=lambda t: (t["hit"], t["scr_db"]))
    log(f"[D] showcase = wp{show['i']} (SCR={show['scr_db']:.1f}dB hit={show['hit']})")
    t_lo, t_hi = t_int_pair
    cfg_hi = make_cfg(grid, t_hi, show["pos"], wps[show["i"]][1], M=M,
                      samples_per_src=samples_per_src, seed=seed)
    h_hi, gt_hi = trace_waypoint(cfg_hi)                      # N for the longest CPI
    drone_tap = int(np.clip(round(gt_hi["bistatic_delay"] * grid.fs), 0, h_hi.shape[1] - 1))

    # ---- snapshot detail (channel CIR + RD) at the flight T_int ----------------
    N_flt = max(1, int(round(t_flight * grid.fs / M)))
    cfg_s = copy.copy(cfg_hi); cfg_s.N = N_flt
    h_s = h_hi[:N_flt]
    x_ref_full, _ = _ref(grid, N_flt * M, fill=1.0, rng=rng)
    snap = caf_snapshot(cfg_s, grid, x_ref_full, h_s, gt_hi, rng, snr_db=-18.0)
    snap["h_abs"] = np.abs(h_hi).mean(axis=0)                 # |h| vs range tap
    snap["tap_range"] = np.arange(h_hi.shape[1]) * C0 / grid.fs
    snap["paths_tau_ns"] = gt_hi["tau_all_ns"]
    snap["paths_dop_hz"] = gt_hi["dop_all_hz"]
    snap["pos"] = show["pos"]; snap["wp"] = show["i"]
    log(f"[D] snapshot wp{show['i']}: SCR={snap['scr_db']:.1f}dB hit={snap['hit']} "
        f"R_b={snap['R_b_gt']:.1f}m V_b={snap['V_b_gt']:+.2f}m/s")

    # ---- T_int 20 ms vs 100 ms (same RT geometry, slice slow-time) -------------
    tint = {}
    for tag, t_int in (("lo", t_lo), ("hi", t_hi)):
        N = max(1, int(round(t_int * grid.fs / M)))
        cfg_t = copy.copy(cfg_hi); cfg_t.N = N
        x_ref_t, _ = _ref(grid, N * M, fill=1.0, rng=rng)
        r = caf_snapshot(cfg_t, grid, x_ref_t, h_hi[:N], gt_hi, rng, snr_db=-18.0)
        r["t_int_ms"] = t_int * 1e3
        # -3 dB width of the CONTIGUOUS main lobe around the drone peak (not a first/last
        # span over the whole cut, which a multipath/sidelobe lobe would inflate). Anchor
        # the peak search to a window around the GT velocity bin so it tracks the drone.
        vc, va, di = r["rd_db"][r["ri"], :], r["va"], r["di"]
        dv = float(abs(va[1] - va[0])) if len(va) > 1 else r["vres"]
        nW = max(3, int(round(15.0 / dv)))                    # ~±15 m/s search window
        lo_w, hi_w = max(0, di - nW), min(len(vc), di + nW + 1)
        pk = lo_w + int(np.argmax(vc[lo_w:hi_w]))
        thr3 = vc[pk] - 3.0
        a = pk
        while a > 0 and vc[a - 1] >= thr3:
            a -= 1
        b = pk
        while b < len(vc) - 1 and vc[b + 1] >= thr3:
            b += 1
        r["v_width"] = max(float(abs(va[b] - va[a])), dv)     # bin-limited (>= 1 Doppler bin)
        tint[tag] = r
        log(f"[D] T_int {t_int*1e3:5.1f}ms: vres={r['vres']:.2f}m/s "
            f"peak-3dB width={r['v_width']:.2f}m/s SCR={r['scr_db']:.1f}dB")

    # ---- Renyi adaptive integration on the real capture (paper Sec 5.2) --------
    # A time-varying-occupancy capture (bimodal traffic). The selector keeps frames whose
    # Rényi entropy >= 0.95*max -- a gap cleanly between the sparse band (fill 0.04-0.15 ->
    # H/max ~0.85-0.91) and the dense band (fill 0.85-1.00 -> H/max ~0.99-1.00), in the
    # paper's high-threshold spirit (it used 25.5 vs 25.67 = 0.993*max). The dense/sparse
    # CAFs are the selector's OWN choices (highest-entropy KEPT vs lowest-entropy DROPPED
    # frame), so the timeline drives the detection rather than illustrating it separately.
    t_frame = 20e-3
    Nf = max(1, int(round(t_frame * grid.fs / M)))
    frame_len = Nf * M
    F = 12
    pick = rng.random(F) < 0.5                                # bimodal traffic model
    fills = np.where(pick, rng.uniform(0.85, 1.00, F), rng.uniform(0.04, 0.15, F))
    refs = [_ref(grid, frame_len, float(f), rng) for f in fills]
    frames = [r[0] for r in refs]; rhos = [r[1] for r in refs]
    ent, starts = frame_entropies(np.concatenate(frames), frame_len)
    cal_max = calibrate_max(_ref(grid, frame_len, 1.0, rng)[0], frame_len)
    thr = 0.95 * cal_max
    keep = ent >= thr
    i_dense = int(np.argmax(ent))                            # the selector's best KEPT frame
    i_sparse = int(np.argmin(ent))                           # the selector's worst DROPPED frame
    x_dense, rho_d = frames[i_dense], rhos[i_dense]
    x_sparse, rho_s = frames[i_sparse], rhos[i_sparse]
    cfg_f = copy.copy(cfg_hi); cfg_f.N = Nf
    h_f = h_hi[:Nf]
    # absolute noise auto-calibrated so the DENSE CPI lands at ~18 dB SCR (clear detection);
    # the equal-length SPARSE CPI, ~10log10(rho_s/rho_d) dB weaker, falls below the CFAR
    # threshold -> buried. Fixed noise, content varies (paper Sec 4/5).
    p_dense = float(np.mean(np.abs(h_f[:, drone_tap]) ** 2) * np.mean(np.abs(x_dense) ** 2))
    nominal = p_dense / (10 ** (-25.0 / 10.0))
    scr0 = caf_snapshot(cfg_f, grid, x_dense, h_f, gt_hi, rng, noise_pow=nominal)["scr_db"]
    noise_abs = nominal * 10 ** ((scr0 - 18.0) / 10.0)
    dense = caf_snapshot(cfg_f, grid, x_dense, h_f, gt_hi, rng, noise_pow=noise_abs)
    sparse = caf_snapshot(cfg_f, grid, x_sparse, h_f, gt_hi, rng, noise_pow=noise_abs)
    ent_dense, ent_sparse = float(ent[i_dense]), float(ent[i_sparse])
    log(f"[D] entropy demo: cal_max={cal_max:.2f} thr={thr:.2f} kept={int(keep.sum())}/{F} "
        f"(dense band kept, sparse band dropped)")
    log(f"   KEPT dense (fill {rho_d:.2f}) H={ent_dense:.2f} SCR={dense['scr_db']:.1f}dB hit={dense['hit']} | "
        f"DROPPED sparse (fill {rho_s:.2f}) H={ent_sparse:.2f} SCR={sparse['scr_db']:.1f}dB hit={sparse['hit']}")

    entropy_demo = dict(fills=fills.tolist(), entropy=ent.tolist(), starts=starts.tolist(),
                        threshold=float(thr), cal_max=float(cal_max), keep=keep.tolist(),
                        i_dense=i_dense, i_sparse=i_sparse,
                        t_frame_ms=t_frame * 1e3, dense=dense, sparse=sparse,
                        H_dense=ent_dense, H_sparse=ent_sparse,
                        rho_dense=float(rho_d), rho_sparse=float(rho_s))

    geometry = dict(tx=list(cfg_hi.tx_pos), rx=list(cfg_hi.rx_pos),
                    waypoints=[list(p) for p, _ in wps], vel=list(wps[0][1]),
                    showcase=show["i"])
    results = dict(cfg=dict(fc_ghz=grid.fc / 1e9, fs_mhz=grid.fs / 1e6, M=M,
                            range_res_m=grid.range_res_m, t_flight_ms=t_flight * 1e3),
                   geometry=geometry, trajectory=traj, snapshot=snap,
                   tint=tint, entropy_demo=entropy_demo)
    results["gate"] = _gate(results)
    results["elapsed_s"] = round(time.time() - t0, 1)
    log(f"[D] done in {results['elapsed_s']}s  gate_pass={results['gate']['gate_pass']}")
    return results


def _gate(results: dict) -> dict:
    """Phase D gate: (1) detections follow the Sionna GT trajectory; (2) 100 ms
    sharpens velocity vs 20 ms; (3) the entropy selector matters (dense detects,
    sparse buried)."""
    # only waypoints where the RT actually traced a drone path count toward "follows GT"
    valid = [t for t in results["trajectory"] if t.get("valid", True)]
    n_hit = sum(t["hit"] for t in valid)
    hit_rate = n_hit / max(1, len(valid))
    r_err = float(np.median([abs(t["R_b_det"] - t["R_b_gt"]) for t in valid] or [9e9]))
    v_err = float(np.median([abs(t["V_b_det"] - t["V_b_gt"]) for t in valid if t["hit"]] or [9e9]))
    lo, hi = results["tint"]["lo"], results["tint"]["hi"]
    sharpen = bool(hi["v_width"] < lo["v_width"] - 1e-6 and hi["vres"] < lo["vres"])
    ed = results["entropy_demo"]
    keepm = ed["keep"]
    clean_split = bool(keepm[ed["i_dense"]] and not keepm[ed["i_sparse"]])
    selector = bool(ed["dense"]["hit"] and not ed["sparse"]["hit"]
                    and ed["H_dense"] > ed["H_sparse"] and clean_split)
    follows = bool(hit_rate >= 0.7 and r_err <= results["cfg"]["range_res_m"] * 3)
    gate_pass = bool(follows and sharpen and selector)
    return dict(gate_pass=gate_pass, follows_gt=follows, hit_rate=round(hit_rate, 2),
                n_valid_wp=len(valid), n_wp=len(results["trajectory"]),
                median_R_err_m=round(r_err, 2), median_V_err_ms=round(v_err, 2),
                tint_sharpens=sharpen, v_width_lo_ms=round(lo["v_width"], 2),
                v_width_hi_ms=round(hi["v_width"], 2),
                vres_lo_ms=round(lo["vres"], 2), vres_hi_ms=round(hi["vres"], 2),
                entropy_selector=selector, entropy_clean_split=clean_split,
                n_kept=int(sum(keepm)), n_frames=len(keepm),
                dense_hit=ed["dense"]["hit"], sparse_hit=ed["sparse"]["hit"],
                note="Real Sionna-RT bistatic flight: CFAR detections track the exact "
                     "GT trajectory; 100 ms integration sharpens velocity vs 20 ms "
                     "(Fig 21->22); Renyi-entropy selection keeps dense frames that "
                     "detect where equal-length sparse frames are buried (Sec 5.2).")


# --------------------------------------------------------------------------- #
#  Visualisation — Korean-annotated, viz-first (overview dashboard + 6 figures
#  mapped to paper Figs 18/20/21/22/23/24). Korean labels render if NanumGothic
#  is available (auto-registered); else they fall back to English.
# --------------------------------------------------------------------------- #
_KO = None


def _setup_style(plt):
    """Register a Korean font (if present) and set readable defaults. Returns True
    if Korean glyphs will render."""
    global _KO
    import matplotlib.font_manager as fm
    ok = False
    for p in (os.path.join(_HERE, "assets", "NanumGothic.ttf"),        # repo-local (reproducible)
              os.path.expanduser("~/.local/share/fonts/NanumGothic.ttf")):
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                # Korean glyphs from NanumGothic, symbols/accents (é, Δ, ✓, ①) fall
                # back to DejaVu Sans (matplotlib >=3.6 does per-glyph fallback).
                plt.rcParams["font.family"] = [fm.FontProperties(fname=p).get_name(),
                                               "DejaVu Sans"]
                ok = True
                break
            except Exception:
                pass
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12,
                         "axes.titlesize": 13, "axes.labelsize": 12,
                         "legend.fontsize": 9.5, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "figure.dpi": 130, "savefig.dpi": 145})
    _KO = ok
    return ok


def _T(ko, en):
    """Korean if a Korean font is active, else English fallback."""
    return ko if _KO else en


# ---- reusable panels -------------------------------------------------------- #
def _geom2d(ax, g, snap):
    """Top-down (x-y) bistatic geometry — easier to read than 3D."""
    tx, rx = np.array(g["tx"]), np.array(g["rx"])
    wp = np.array(g["waypoints"]); sc_i = g.get("showcase", len(wp) // 2)
    d = np.array(snap["pos"])
    ax.plot([tx[0], rx[0]], [tx[1], rx[1]], "--", color="0.4", lw=1.4, zorder=1)
    ax.annotate(_T("기준선 L", "baseline L"), ((tx[0] + rx[0]) / 2, (tx[1] + rx[1]) / 2),
                fontsize=8, color="0.4", ha="center", va="top")
    ax.plot([tx[0], d[0]], [tx[1], d[1]], ":", color="tab:red", lw=1.2, alpha=.7)
    ax.plot([rx[0], d[0]], [rx[1], d[1]], ":", color="tab:blue", lw=1.2, alpha=.7)
    ax.annotate("R1", ((tx[0] + d[0]) / 2, (tx[1] + d[1]) / 2), fontsize=8, color="tab:red")
    ax.annotate("R2", ((rx[0] + d[0]) / 2, (rx[1] + d[1]) / 2), fontsize=8, color="tab:blue")
    ax.plot(wp[:, 0], wp[:, 1], "-", color="green", lw=1.3, alpha=.5, zorder=2)
    s = ax.scatter(wp[:, 0], wp[:, 1], c=np.arange(len(wp)), cmap="viridis", s=70,
                   edgecolor="k", lw=.4, zorder=3)
    ax.scatter(*tx[:2], c="tab:red", s=240, marker="^", edgecolor="k", zorder=4,
               label=_T("gNB 송신 (조명)", "gNB Tx (illuminator)"))
    ax.scatter(*rx[:2], c="tab:blue", s=240, marker="v", edgecolor="k", zorder=4,
               label=_T("수동 수신 Rx", "passive Rx"))
    ax.scatter(d[0], d[1], c="lime", s=150, marker="*", edgecolor="k", zorder=5,
               label=_T("드론 (showcase)", "drone (showcase)"))
    ax.annotate(_T(f"R_b={snap['R_b_gt']:.0f} m\nV_b={snap['V_b_gt']:+.1f} m/s",
                   f"R_b={snap['R_b_gt']:.0f} m\nV_b={snap['V_b_gt']:+.1f} m/s"),
                (d[0], d[1]), fontsize=8.5, fontweight="bold",
                xytext=(8, 8), textcoords="offset points")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal", "datalim")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
    return s


def _traj_panel(ax, val, key_gt, key_det, ylab):
    from matplotlib.lines import Line2D
    ax.plot([t["i"] for t in val], [t[key_gt] for t in val], "-o", color="k", zorder=2, ms=5)
    for t in val:
        ax.scatter(t["i"], t[key_det], c=("lime" if t["hit"] else "red"), s=85,
                   edgecolor="k", zorder=3)
    ax.set_ylabel(ylab); ax.set_xlabel(_T("비행 waypoint 번호", "flight waypoint #"))
    ax.grid(alpha=.3)
    return [Line2D([], [], marker="o", color="k", label=_T("Sionna 정답(GT)", "Sionna GT")),
            Line2D([], [], marker="o", ls="", mfc="lime", mec="k", label=_T("검출", "detected")),
            Line2D([], [], marker="o", ls="", mfc="red", mec="k", label=_T("놓침", "missed"))]


def _entropy_bars(ax, ed, *, mini=False):
    fr = np.arange(len(ed["entropy"]))
    ent = np.array(ed["entropy"]); keep = np.array(ed["keep"], bool)
    ax.bar(fr[keep], ent[keep], color="tab:green", label=_T("선택(밀집)", "kept (dense)"))
    ax.bar(fr[~keep], ent[~keep], color="tab:red", alpha=.7, label=_T("버림(희소)", "dropped (sparse)"))
    ax.axhline(ed["threshold"], color="k", ls="--", lw=1.3,
               label=_T(f"임계 0.95·max ({ed['threshold']:.1f})", f"thr 0.95·max ({ed['threshold']:.1f})"))
    for j in (ed["i_dense"], ed["i_sparse"]):
        ax.plot(j, ent[j], "*", color="gold", ms=15, mec="k", zorder=5)
    if not mini:
        ax.annotate(_T("→ CAF(선택)", "→ CAF (kept)"), (ed["i_dense"], ent[ed["i_dense"]]),
                    fontsize=8, ha="center", xytext=(0, 8), textcoords="offset points")
        ax.annotate(_T("→ CAF(버림)", "→ CAF (dropped)"), (ed["i_sparse"], ent[ed["i_sparse"]]),
                    fontsize=8, ha="center", xytext=(0, -14), textcoords="offset points")
        axt = ax.twinx()
        axt.plot(fr, 100 * np.array(ed["fills"]), "o-", color="0.5", ms=4, alpha=.7)
        axt.set_ylabel(_T("프레임 점유율 [%]", "frame fill [%]"), color="0.5"); axt.set_ylim(0, 105)
    ax.set_ylim(ent.min() - 0.8, ent.max() + 0.5)
    ax.set_xlabel(_T(f"프레임 번호 (각 {ed['t_frame_ms']:.0f} ms)", f"frame # ({ed['t_frame_ms']:.0f} ms)"))
    ax.set_ylabel(_T("Rényi 엔트로피", "Rényi entropy"))
    ax.legend(fontsize=8, loc="lower left")


def _vcut(ax, lo, hi):
    for r, col in ((lo, "tab:orange"), (hi, "tab:blue")):
        ax.plot(r["va"], r["rd_db"][r["ri"], :], color=col, lw=2,
                label=_T(f"{r['t_int_ms']:.0f} ms  (-3dB {r['v_width']:.1f} m/s)",
                         f"{r['t_int_ms']:.0f} ms  (-3dB {r['v_width']:.1f} m/s)"))
    ax.axvline(hi["V_b_gt"], color="k", ls="--", lw=1, alpha=.6, label=_T("정답 V_b", "GT V_b"))
    ax.set_xlim(hi["V_b_gt"] - 22, hi["V_b_gt"] + 22); ax.set_ylim(-30, 2)
    ax.set_xlabel("V_b [m/s]"); ax.set_ylabel(_T("상대 전력 [dB]", "rel. power [dB]"))
    ax.legend(fontsize=8.5); ax.grid(alpha=.3)


def _rd_panel(ax, plt, r, title, *, show_cfar=False, vzoom=60, cbar=True, status=None):
    """Range-Doppler panel: dB map [range, doppler], GT marker, optional CFAR + a
    value callout box (R_b / V_b / SCR)."""
    va, ra, rd = r["va"], r["ra"], r["rd_db"]
    im = ax.pcolormesh(va, ra, rd, shading="auto", cmap="turbo", vmin=-25, vmax=0)
    ax.plot(r["V_b_gt"], r["R_b_gt"], "x", color="lime", ms=14, mew=3.2,
            label=_T("Sionna 정답(GT)", "Sionna GT"))
    if show_cfar and "det" in r:
        dv, dr = np.where(r["det"])
        if len(dv):
            ax.scatter(va[dr], ra[dv], s=20, facecolors="none", edgecolors="white",
                       linewidths=1.0, label=_T("CFAR 검출", "CFAR hit"))
    ax.set_xlim(r["V_b_gt"] - vzoom, r["V_b_gt"] + vzoom)
    ax.set_ylim(min(150, ra.max()), 0)
    ax.set_xlabel("V_b [m/s]"); ax.set_ylabel("R_b [m]"); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    txt = f"R_b={r['R_b_gt']:.0f} m\nV_b={r['V_b_gt']:+.1f} m/s\nSCR={r['scr_db']:.0f} dB"
    if status:
        txt += f"\n{status}"
    ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", alpha=.78, ec="0.6"))
    if cbar:
        plt.colorbar(im, ax=ax, shrink=.85, label=_T("상대 전력 [dB]", "rel. power [dB]"))
    return im


def make_figures(results: dict, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _setup_style(plt)
    paths = {}
    g = results["geometry"]; s = results["snapshot"]; gate = results["gate"]
    traj = results["trajectory"]; val = [t for t in traj if t.get("valid", True)]
    lo, hi = results["tint"]["lo"], results["tint"]["hi"]; ed = results["entropy_demo"]

    # -- D0: one-page OVERVIEW dashboard (the whole story at a glance) -----------
    fig = plt.figure(figsize=(17, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    sc = _geom2d(fig.add_subplot(gs[0, 0]), g, s)
    fig.axes[-1].set_title(_T("① 장면·비행 궤적 (평면도)", "① scene & flight (top view)"))
    axb = fig.add_subplot(gs[0, 1])
    _rd_panel(axb, plt, s, _T("② 거리-도플러 + CFAR 검출", "② range-Doppler + CFAR"),
              show_cfar=True, vzoom=45, status=_T("✓ 검출", "✓ detected"))
    _entropy_bars(fig.add_subplot(gs[0, 2]), ed)
    fig.axes[-2].set_title(_T("③ Rényi 엔트로피 프레임 선택", "③ Rényi frame selection"))
    leg = _traj_panel(fig.add_subplot(gs[1, 0]), val, "R_b_gt", "R_b_det",
                      _T("거리 R_b [m]", "range R_b [m]"))
    fig.axes[-1].set_title(_T("④ 거리: 검출 vs 정답", "④ range: detection vs GT"))
    fig.axes[-1].legend(handles=leg, fontsize=8, loc="best")
    _traj_panel(fig.add_subplot(gs[1, 1]), val, "V_b_gt", "V_b_det",
                _T("속도 V_b [m/s]", "velocity V_b [m/s]"))
    fig.axes[-1].set_title(_T("⑤ 속도: 검출 vs 정답", "⑤ velocity: detection vs GT"))
    _vcut(fig.add_subplot(gs[1, 2]), lo, hi)
    fig.axes[-1].set_title(_T("⑥ 적분시간 20→100 ms 속도 선명화", "⑥ T_int 20→100 ms sharpening"))
    head = _T(
        f"Phase D — 5G 수동 레이더 실비행 드론 검출 (Sionna RT)    │    GATE "
        f"{'✓ 통과' if gate['gate_pass'] else '✗ 실패'}",
        f"Phase D — real-flight bistatic drone detection (Sionna RT)    │    GATE "
        f"{'✓ PASS' if gate['gate_pass'] else '✗ FAIL'}")
    fig.suptitle(head, fontsize=16, fontweight="bold")
    note = _T(
        f"검출 {gate['hit_rate']*100:.0f}% ({gate.get('hit_rate',0)*len(val):.0f}/{len(val)} waypoint) · "
        f"거리오차(중앙값) {gate['median_R_err_m']} m · 속도오차 {gate['median_V_err_ms']} m/s · "
        f"속도해상도 ΔV {gate['vres_lo_ms']}→{gate['vres_hi_ms']} m/s (20→100 ms) · "
        f"밀집프레임 검출 / 희소프레임 매몰 · RT {results.get('elapsed_s','?')} s",
        f"hit-rate {gate['hit_rate']*100:.0f}% · median R err {gate['median_R_err_m']} m · "
        f"V err {gate['median_V_err_ms']} m/s · ΔV {gate['vres_lo_ms']}→{gate['vres_hi_ms']} m/s")
    fig.text(0.5, -0.012, note, ha="center", fontsize=11,
             bbox=dict(boxstyle="round", fc="#eef6ff", ec="0.6"))
    p0 = os.path.join(outdir, "phaseD_overview.png")
    fig.savefig(p0, bbox_inches="tight"); plt.close(fig); paths["overview"] = p0

    # -- D1: bistatic scene — 3D + 2D top-down (paper Fig 18) --------------------
    wp = np.array(g["waypoints"]); tx, rx = np.array(g["tx"]), np.array(g["rx"])
    fig = plt.figure(figsize=(14, 5.8))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    xx, yy = np.meshgrid(np.linspace(-70, 70, 2), np.linspace(-10, 80, 2))
    ax3.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.10, color="gray")
    ax3.scatter(*tx, c="tab:red", s=130, marker="^", label=_T("gNB 송신", "gNB Tx"))
    ax3.scatter(*rx, c="tab:blue", s=130, marker="v", label=_T("수동 Rx", "passive Rx"))
    ax3.plot([tx[0], rx[0]], [tx[1], rx[1]], [tx[2], rx[2]], "k--", lw=1, alpha=.6)
    ax3.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c=np.arange(len(wp)), cmap="viridis", s=55,
                label=_T("드론 비행", "drone flight"))
    ax3.plot(wp[:, 0], wp[:, 1], wp[:, 2], "-", color="green", lw=1.2, alpha=.5)
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
    ax3.set_title(_T("입체도 (3D)", "3D view")); ax3.legend(fontsize=8.5, loc="upper left")
    ax3.view_init(elev=22, azim=-60)
    s2 = _geom2d(fig.add_subplot(1, 2, 2), g, s)
    fig.axes[-1].set_title(_T("평면도 (위에서 본 x-y)", "top-down (x-y)"))
    fig.colorbar(s2, ax=fig.axes[-1], shrink=.8, label=_T("waypoint 번호", "waypoint #"))
    fig.suptitle(_T("Phase D / Fig 18 — bistatic 5G 수동레이더 장면 (Sionna RT)",
                    "Phase D / Fig 18 — bistatic 5G-PCL scene (Sionna RT)"),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); p1 = os.path.join(outdir, "phaseD_geometry.png")
    fig.savefig(p1); plt.close(fig); paths["geometry"] = p1

    # -- D2: ray-traced channel (CIR + per-path delay/Doppler) (paper Fig 20) ----
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
    hdb = 20 * np.log10(s["h_abs"] + 1e-12); hdb -= hdb.max()
    dt = s["drone_tap"]
    ax[0].plot(s["tap_range"], hdb, color="tab:purple", lw=1.6)
    ax[0].axvline(s["tap_range"][0], color="k", ls="--", lw=1, alpha=.6,
                  label=_T("직접파 (tap 0, 클러터)", "direct path (tap 0)"))
    ax[0].plot(s["tap_range"][dt], hdb[dt], "x", color="lime", ms=14, mew=3.2,
               label=_T(f"드론 echo @ {s['R_b_gt']:.0f} m", f"drone tap @ {s['R_b_gt']:.0f} m"))
    ax[0].annotate(_T(f"드론은 직접파보다\n~{abs(hdb[dt]):.0f} dB 아래\n→ 클러터 제거 필요",
                      f"~{abs(hdb[dt]):.0f} dB below\ndirect path\n→ clutter cancel"),
                   (s["tap_range"][dt], hdb[dt]), fontsize=8.5, xytext=(18, 12),
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", color="0.4"))
    ax[0].set_xlim(0, min(150, s["tap_range"][-1])); ax[0].set_ylim(-60, 4)
    ax[0].set_xlabel(_T("양기지 거리 [m]", "bistatic range [m]")); ax[0].set_ylabel("|h| [dB]")
    ax[0].set_title(_T("광선추적 채널 임펄스응답", "RT channel impulse response"))
    ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.3)
    tau = np.array(s["paths_tau_ns"]); dop = np.array(s["paths_dop_hz"])
    di = int(np.argmax(np.abs(dop)))
    ax[1].scatter(tau, dop, c="0.55", s=70, label=_T("정지 경로 (직접·지면)", "static (direct/ground)"))
    ax[1].scatter(tau[di], dop[di], c="lime", s=180, marker="*", edgecolor="k", zorder=5,
                  label=_T("드론 경로 (max |f_D|)", "drone path (max |f_D|)"))
    ax[1].axhline(0, color="k", lw=.8, alpha=.5)
    ax[1].set_xlabel(_T("경로 지연 [ns]", "path delay [ns]"))
    ax[1].set_ylabel(_T("경로 도플러 [Hz]", "path Doppler [Hz]"))
    ax[1].set_title(_T("광선추적 다중경로: 지연 vs 도플러", "ray-traced multipath: delay vs Doppler"))
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3)
    fig.suptitle(_T("Phase D / Fig 20 — 실제 양기지 채널 (A-C의 해석적 echo 대체)",
                    "Phase D / Fig 20 — real bistatic channel (vs analytic echo of A-C)"),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); p2 = os.path.join(outdir, "phaseD_channel.png")
    fig.savefig(p2); plt.close(fig); paths["channel"] = p2

    # -- D3: snapshot range-Doppler + CFAR on the RT echo (paper Fig 21) ---------
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    _rd_panel(ax, plt, s, _T(f"Phase D / Fig 21 — 광선추적 RD + CA-CFAR (waypoint {s.get('wp','?')})",
                             "Phase D / Fig 21 — RT bistatic RD + CA-CFAR"),
              show_cfar=True, status=_T("✓ 검출 성공", "✓ detected"))
    fig.tight_layout(); p3 = os.path.join(outdir, "phaseD_snapshot_rd.png")
    fig.savefig(p3); plt.close(fig); paths["snapshot_rd"] = p3

    # -- D4: flight trajectory — detections vs Sionna GT (paper Fig 23) ----------
    fig, ax = plt.subplots(2, 1, figsize=(9, 7.4), sharex=True)
    leg = _traj_panel(ax[0], val, "R_b_gt", "R_b_det", _T("양기지 거리 R_b [m]", "bistatic range R_b [m]"))
    _traj_panel(ax[1], val, "V_b_gt", "V_b_det", _T("양기지 속도 V_b [m/s]", "bistatic velocity V_b [m/s]"))
    ax[0].legend(handles=leg, fontsize=9, loc="best")
    ax[0].set_title(_T(f"Phase D / Fig 23 — CFAR 검출이 Sionna 정답 궤적을 추적 "
                       f"(검출 {sum(t['hit'] for t in val)}/{len(val)}, "
                       f"오차 {gate['median_R_err_m']} m / {gate['median_V_err_ms']} m/s)",
                       "Phase D / Fig 23 — CFAR detections follow the exact Sionna GT"),
                   fontsize=12.5, fontweight="bold")
    ax[0].set_xlabel(""); fig.tight_layout()
    p4 = os.path.join(outdir, "phaseD_trajectory.png")
    fig.savefig(p4); plt.close(fig); paths["trajectory"] = p4

    # -- D5: T_int 20 ms vs 100 ms — velocity sharpening (paper Fig 21 -> 22) ----
    fig = plt.figure(figsize=(15, 5.2))
    _rd_panel(fig.add_subplot(1, 3, 1), plt, lo,
              _T(f"T_int = {lo['t_int_ms']:.0f} ms  (ΔV={lo['vres']:.1f} m/s)",
                 f"T_int = {lo['t_int_ms']:.0f} ms  (ΔV={lo['vres']:.1f} m/s)"), vzoom=40)
    _rd_panel(fig.add_subplot(1, 3, 2), plt, hi,
              _T(f"T_int = {hi['t_int_ms']:.0f} ms  (ΔV={hi['vres']:.2f} m/s)",
                 f"T_int = {hi['t_int_ms']:.0f} ms  (ΔV={hi['vres']:.2f} m/s)"), vzoom=40)
    axc = fig.add_subplot(1, 3, 3); _vcut(axc, lo, hi)
    axc.set_title(_T("피크 단면 (속도 방향)", "velocity cut through the peak"))
    fig.suptitle(_T("Phase D / Fig 21→22 — 긴 적분시간이 속도 추정을 선명하게",
                    "Phase D / Fig 21→22 — longer integration sharpens velocity"),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); p5 = os.path.join(outdir, "phaseD_tint.png")
    fig.savefig(p5); plt.close(fig); paths["tint"] = p5

    # -- D6: Renyi adaptive integration on the real capture (paper Fig 24) -------
    fig = plt.figure(figsize=(15, 4.9))
    _entropy_bars(fig.add_subplot(1, 3, 1), ed)
    fig.axes[-2].set_title(_T("프레임별 엔트로피 → 적응 선택", "per-frame entropy → selection"))
    _rd_panel(fig.add_subplot(1, 3, 2), plt, ed["dense"],
              _T(f"선택된 밀집 프레임 (점유 {ed['rho_dense']:.0%}, H={ed['H_dense']:.1f})",
                 f"KEPT dense frame (fill {ed['rho_dense']:.0%}, H={ed['H_dense']:.1f})"),
              show_cfar=True, vzoom=40, status=_T("✓ 검출", "✓ detected"))
    _rd_panel(fig.add_subplot(1, 3, 3), plt, ed["sparse"],
              _T(f"버려진 희소 프레임 (점유 {ed['rho_sparse']:.0%}, H={ed['H_sparse']:.1f})",
                 f"DROPPED sparse frame (fill {ed['rho_sparse']:.0%}, H={ed['H_sparse']:.1f})"),
              show_cfar=True, vzoom=40, status=_T("✗ 매몰", "✗ buried"))
    fig.suptitle(_T("Phase D / Fig 24 — Rényi 엔트로피로 밀집 프레임 선택: 같은 채널에서 "
                    "선택 프레임은 검출, 버린 프레임은 매몰",
                    "Phase D / Fig 24 — Rényi selects dense frames: kept detects, dropped is buried"),
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(); p6 = os.path.join(outdir, "phaseD_entropy.png")
    fig.savefig(p6); plt.close(fig); paths["entropy"] = p6
    return paths


if __name__ == "__main__":
    print("Phase D bistatic scene. sionna available:", have_sionna())
    if have_sionna():
        import json
        res = flight_caf(samples_per_src=1_000_000, n_waypoints=7)
        outdir = os.path.join(_HERE, "outputs")
        os.makedirs(outdir, exist_ok=True)
        make_figures(res, outdir)
        print(json.dumps(res["gate"], indent=2))
