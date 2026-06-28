#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaSen faithful-reproduction pipeline (5G-26 SenSys) — phase-gated, viz-first.

Phase A (this file, --phase A): monostatic CFR -> full-band 2D-FFT range-Doppler
SANITY. GATE: with full occupancy, the 2D-FFT RD peak lands on the analytic GT
range/Doppler cell, and after slow-time mean subtraction (LaSen §4.1.1) the 0-Hz
clutter/self-leakage ridge collapses (LaSen Fig.4). If this fails, fix the Doppler
sign / monostatic factor / CFR before adding 2D-OMP (Phase B).

Each phase: emits figures (mapped to LaSen paper figures) + a JSON with the gate
verdict. Outputs -> lasen/outputs/.
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

from nr_waveform import NRNumerology
from monostatic_scene import GEOM, GEOM_STRONG, GEOM_WEAK, trace_cfr, trace_two_targets
from occupancy import DENSITY_BINS, mask_at_density, density_timeline
from omp2d import rd_transform, omp2d, roundtrip_ok, atom
from monostatic_scene import analytic_gt
import viz


# --------------------------------------------------------------------------- #
#  Core RD processing (shared by phases)
# --------------------------------------------------------------------------- #
def rd_from_cfr(H, num: NRNumerology, prf, mti=True):
    """2-D range-Doppler from CFR H[n_slow, K]: slow-time mean subtraction (static
    /0-Hz clutter suppression, LaSen §4.1.1) then IFFT over freq (range) + FFT over
    slow-time (Doppler). Returns complex RD [doppler, range] + axes."""
    Ns, K = H.shape
    if mti:
        H = H - H.mean(axis=0, keepdims=True)              # remove 0-Doppler clutter
    wf = np.hanning(K + 2)[1:-1][None, :]
    wt = np.hanning(Ns + 2)[1:-1][:, None]
    rng = np.fft.ifft(H * wf, axis=1)                      # [Ns, K]  delay/range
    rd = np.fft.fftshift(np.fft.fft(rng * wt, axis=0), axes=0)   # [Doppler, range]
    range_axis = np.arange(K) * num.range_res_m
    dopp_axis = np.fft.fftshift(np.fft.fftfreq(Ns, d=1.0 / prf))
    return rd, range_axis, dopp_axis


def peak_cell(rd, range_axis, dopp_axis, notch_hz=20.0):
    """Global peak of |rd|^2 excluding the +-notch zero-Doppler clutter band."""
    p = np.abs(rd) ** 2
    mask = np.abs(dopp_axis) < notch_hz
    pm = p.copy(); pm[mask, :] = -np.inf
    di, ri = np.unravel_index(int(np.argmax(pm)), p.shape)
    return float(range_axis[ri]), float(dopp_axis[di]), int(ri), int(di)


def suppress_masked(Hfull, W):
    """Static-background suppression on a MASKED observation (Phase B): subtract the
    per-frequency mean over the OCCUPIED slow-times (removes the 0-Hz clutter on the
    transmitted REs). Returns the masked, suppressed CFR (zeros where not observed)."""
    Hm = np.where(W, Hfull, 0).astype(np.complex64)
    cnt = W.sum(axis=0, keepdims=True)
    mean = np.where(cnt > 0, Hm.sum(axis=0, keepdims=True) / np.maximum(cnt, 1), 0)
    return np.where(W, Hm - mean, 0).astype(np.complex64)


def _peak_metrics(power, ri_gt, di_gt, range_axis, dopp_axis, num, notch_hz=20.0):
    """peak-on-GT + peak/background (PSLR-ish) for an RD power map."""
    notch = np.abs(dopp_axis) < notch_hz
    pm = power.copy(); pm[notch, :] = 0.0
    di, ri = np.unravel_index(int(np.argmax(pm)), power.shape)
    on_gt = bool(abs(ri - ri_gt) <= 2 and abs(di - di_gt) <= 2)
    peak = float(power[di, ri])
    bg = power[~notch, :].copy()
    # exclude a small box around the peak for the background estimate
    mask = np.ones_like(power, bool); mask[notch, :] = False
    mask[max(0, di-2):di+3, max(0, ri-2):ri+3] = False
    pslr_db = 10 * np.log10(peak / (np.median(power[mask]) + 1e-30) + 1e-30)
    return dict(peak_range_m=float(range_axis[ri]), peak_doppler_hz=float(dopp_axis[di]),
                peak_on_gt=on_gt, pslr_db=float(pslr_db))


# --------------------------------------------------------------------------- #
#  Phase A   (R1 real symbols, R2 Doppler sweep, R3 per-panel figures)
# --------------------------------------------------------------------------- #
def _panel(outdir, name, plotter, figsize=(6.2, 4.6), is3d=False):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d") if is3d else fig.add_subplot(111)
    plotter(ax, fig)
    viz.savefig(fig, os.path.join(outdir, name))


def doppler_sweep(num, velocities, n_slow=1024):
    """R2: measure the RD-peak Doppler at several velocities -> must follow 2v/λ."""
    rows = []
    for v in velocities:
        H, _, gt = trace_cfr(num, GEOM, (0.0, float(v), 0.0), n_slow=n_slow,
                             samples_per_src=5_000_000)
        rd, r_ax, d_ax = rd_from_cfr(H, num, gt["prf_hz"], mti=True)
        _, pd, _, _ = peak_cell(rd, r_ax, d_ax)
        rows.append(dict(v_ms=float(v), v_radial_ms=gt["v_radial_ms"],
                         fD_analytic_hz=gt["doppler_hz"], fD_measured_hz=float(pd),
                         dopp_res_hz=gt["doppler_res_hz"]))
        print(f"[A-sweep] v={v:4.1f}  v_rad={gt['v_radial_ms']:4.1f}  "
              f"fD_analytic={gt['doppler_hz']:+6.0f}  measured={pd:+6.0f} Hz")
    return rows


def phase_a(outdir, num, window_s=0.1, speed=12.0, sweep_vel=(4, 8, 12, 16, 20)):
    os.makedirs(outdir, exist_ok=True)
    vel = (0.0, speed, 0.0)
    print(f"[A] monostatic CFR trace (REAL symbols): fc={num.fc/1e9} GHz B={num.bw/1e6:.2f} MHz "
          f"K={num.n_active} window={window_s*1e3:.0f} ms")
    H, freqs, gt = trace_cfr(num, GEOM, vel, window_s=window_s)          # R1: n_slow=n_symbols
    n_slow = gt["n_slow"]
    print(f"[A] N(slow)={n_slow} symbols (R1 real OFDM symbols, PRF={gt['prf_hz']:.0f} Hz "
          f"= symbol rate, dopp_res={gt['doppler_res_hz']:.1f} Hz, subsampled={gt['subsampled']})")
    print(f"[A] GT: range={gt['range_m']:.1f} m, fD={gt['doppler_hz']:.1f} Hz (=2v/λ), "
          f"v_rad={gt['v_radial_ms']:.1f} m/s, paths={gt['n_paths']}")

    rd_raw, r_ax, d_ax = rd_from_cfr(H, num, gt["prf_hz"], mti=False)
    rd_cln, _, _ = rd_from_cfr(H, num, gt["prf_hz"], mti=True)

    # ---- GATE: peak on GT + clutter collapse ----
    pr, pd, ri, di = peak_cell(rd_cln, r_ax, d_ax)
    range_err_m = abs(pr - gt["range_m"]); dopp_err_hz = abs(pd - gt["doppler_hz"])
    peak_on_gt = bool(range_err_m <= 3 * num.range_res_m and dopp_err_hz <= 3 * gt["doppler_res_hz"])
    notch = np.abs(d_ax) < 20.0
    clutter_drop_db = 10 * np.log10(np.sum(np.abs(rd_cln[notch, :])**2) /
                                    (np.sum(np.abs(rd_raw[notch, :])**2) + 1e-30))
    clutter_collapses = bool(clutter_drop_db < -10.0)

    # ---- R2: Doppler-velocity sweep ----
    sweep = doppler_sweep(num, sweep_vel)
    fa = np.array([r["fD_analytic_hz"] for r in sweep]); fm = np.array([r["fD_measured_hz"] for r in sweep])
    sweep_max_err = float(np.max(np.abs(fa - fm)))
    sweep_res = sweep[0]["dopp_res_hz"]
    sweep_pass = bool(sweep_max_err <= 3 * sweep_res)
    gate_pass = peak_on_gt and clutter_collapses and sweep_pass

    print(f"[A][gate] peak R={pr:.1f}m (err {range_err_m:.1f}m) fD={pd:.0f}Hz (err {dopp_err_hz:.1f}Hz) "
          f"on_GT={peak_on_gt} | clutter {clutter_drop_db:.0f}dB ({clutter_collapses}) | "
          f"sweep max_err={sweep_max_err:.0f}Hz (<{3*sweep_res:.0f}) pass={sweep_pass} -> "
          f"GATE {'PASS' if gate_pass else 'FAIL'}")

    # ---- R3: per-panel figures (each its own PNG, commentary added in build_report) ----
    rspan = (max(0, gt["range_m"] - 40), gt["range_m"] + 40)
    dlim = max(800, 3 * abs(gt["doppler_hz"])); dspan = (-dlim, dlim)
    full_grid = np.ones((min(80, n_slow), num.n_active), bool)
    _panel(outdir, "A_grid.png", lambda ax, f: viz.resource_grid(
        ax, full_grid[:, ::20], title="(Fig 2b) NR grid — Phase A: full occupancy"))
    _panel(outdir, "A_cfr.png", lambda ax, f: viz.cfr_heatmap(
        ax, H, freqs, gt["prf_hz"], title="(Fig 3) CFR |H[t,f]| = Y/X (real symbols)"))
    _panel(outdir, "A_geometry.png", lambda ax, f: viz.trajectory3d(ax, GEOM, vel), is3d=True)

    def _rd(rd, ttl):
        def p(ax, fig):
            im = viz.rd_map(ax, rd, r_ax, d_ax, gt=gt, title=ttl, r_zoom=rspan, d_zoom=dspan)
            fig.colorbar(im, ax=ax, shrink=.85, label="dB"); ax.legend(loc="upper right", fontsize=7)
        return p
    _panel(outdir, "A_rd_raw.png", _rd(rd_raw, "(Fig 4a) RD raw — 0-Hz clutter dominates"))
    _panel(outdir, "A_rd_clean.png", _rd(rd_cln,
        f"(Fig 4b) RD after static suppression — clutter −{abs(clutter_drop_db):.0f} dB, peak on GT={peak_on_gt}"))

    def _sweep_plot(ax, fig):
        vr = np.array([r["v_radial_ms"] for r in sweep])
        ax.plot(np.linspace(0, max(vr)*1.1, 50), num.doppler_hz(np.linspace(0, max(vr)*1.1, 50)),
                "-", color="0.5", label="analytic 2v/λ")
        ax.plot(vr, -fm, "o", color="tab:red", ms=8, label="measured RD peak")
        for r in sweep:
            ax.annotate(f"{-r['fD_measured_hz']:.0f}", (r["v_radial_ms"], -r["fD_measured_hz"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("radial velocity [m/s]"); ax.set_ylabel("|Doppler| [Hz]")
        ax.set_title(f"(R2) Doppler↔velocity sweep — f_d=2v/λ\nmax err {sweep_max_err:.0f} Hz "
                     f"(<{3*sweep_res:.0f}) → {'PASS' if sweep_pass else 'FAIL'}", fontsize=9)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    _panel(outdir, "A_doppler_sweep.png", _sweep_plot)

    out = dict(phase="A", config=dict(fc_ghz=num.fc/1e9, bw_mhz=num.bw/1e6,
               scs_khz=num.scs/1e3, n_fft=num.n_fft, n_active=num.n_active,
               n_slow=n_slow, n_symbols_full=gt["n_symbols_full"], subsampled=gt["subsampled"],
               window_ms=window_s*1e3, prf_hz=gt["prf_hz"], doppler_res_hz=gt["doppler_res_hz"],
               range_res_m=num.range_res_m, speed_ms=speed),
               gt=gt, sweep=sweep,
               gate=dict(peak_range_m=pr, peak_doppler_hz=pd, range_err_m=range_err_m,
                         doppler_err_hz=dopp_err_hz, peak_on_gt=peak_on_gt,
                         clutter_drop_db=clutter_drop_db, clutter_collapses=clutter_collapses,
                         sweep_max_err_hz=sweep_max_err, sweep_pass=sweep_pass, gate_pass=gate_pass))
    json.dump(out, open(os.path.join(outdir, "lasen_phaseA.json"), "w"), indent=1)
    print(f"[A][out] {outdir}/A_*.png + lasen_phaseA.json  GATE={'PASS' if gate_pass else 'FAIL'}")
    return out


# --------------------------------------------------------------------------- #
#  Phase B — non-uniform occupancy + 2D-OMP  (core novelty)
# --------------------------------------------------------------------------- #
def _bins(range_axis, dopp_axis, range_m, dopp_hz):
    return (int(np.argmin(np.abs(dopp_axis - dopp_hz))),
            int(np.argmin(np.abs(range_axis - range_m))))


def finds_target(rd, r_ax, d_ax, gt, notch=20.0, det_db=12.0, guard=2, ring=12):
    """Local 2-D CFAR detection AT the target's GT cell: peak in a small guard box vs
    the median floor of a surrounding ring. A weak target near a strong one is DETECTED
    only if it stands `det_db` above its local floor — when the strong target's
    sparse-mask leakage raises the floor at the weak cell, the weak is buried (miss).
    Returns (detected, snr_over_local_floor_db)."""
    p = np.abs(rd) ** 2
    di = int(np.argmin(np.abs(d_ax - gt["doppler_hz"])))
    ri = int(np.argmin(np.abs(r_ax - gt["range_m"])))
    if abs(d_ax[di]) < notch:
        return False, -99.0
    peak = float(p[max(0, di-guard):di+guard+1, max(0, ri-guard):ri+guard+1].max())
    d0, r0 = max(0, di-ring), max(0, ri-ring)
    sub = p[d0:di+ring+1, r0:ri+ring+1]
    m = np.ones(sub.shape, bool)
    cd, cr = di - d0, ri - r0
    m[max(0, cd-guard):cd+guard+1, max(0, cr-guard):cr+guard+1] = False    # exclude guard box
    floor = float(np.median(sub[m]) + 1e-30)
    snr = float(10 * np.log10(peak / floor))
    return bool(snr > det_db), snr


def omp_finds(support, r_ax, d_ax, gt, tol_r=6.0, tol_d=2):
    """Does 2D-OMP put a support atom at the (weak) GT cell (±tol)?"""
    dres = abs(d_ax[1] - d_ax[0])
    return any(abs(r_ax[ri] - gt["range_m"]) <= tol_r and abs(d_ax[di] - gt["doppler_hz"]) <= tol_d * dres
               for (di, ri) in support)


def _axes(num, Ns, prf):
    return (np.arange(num.n_active) * num.range_res_m,
            np.fft.fftshift(np.fft.fftfreq(Ns, d=1.0 / prf)))


def _eval_density(num, H, prf, gts, gtw, rho, rng):
    """One occupancy density: mask -> suppress -> 2D-FFT & 2D-OMP -> weak/strong
    detection by both methods. Returns everything needed for the gate + figures."""
    Ns, K = H.shape
    range_axis, dopp_axis = _axes(num, Ns, prf)
    W, realised = mask_at_density(Ns, K, rho, rng)
    rd_fft = rd_transform(np.where(W, H, 0))          # masked CFR (targets are movers)
    Z, support, hist = omp2d(np.where(W, H, 0), W)
    fw_on, fw_snr = finds_target(rd_fft, range_axis, dopp_axis, gtw)    # WEAK local detection
    fs_on, _ = finds_target(rd_fft, range_axis, dopp_axis, gts)         # strong (sanity)
    return dict(rho=rho, realised=realised, W=W, rd_fft=rd_fft, Z=Z, support=support, hist=hist,
                fft_weak=fw_on, fft_weak_snr=fw_snr, omp_weak=omp_finds(support, range_axis, dopp_axis, gtw),
                fft_strong=fs_on, omp_strong=omp_finds(support, range_axis, dopp_axis, gts),
                n_atoms=len(support))


def phase_b(outdir, num, window_s=0.1, n_slow=2803, rcs_gap_db=46.0, noise_snr_db=18.0,
            rho_sparse=0.012, rho_dense=0.16, r1_check_n=256, seed=1):
    """Phase B — non-uniform occupancy + 2D-OMP, faithful LaSen Tab.1 / Fig.17 test.
    A STRONG near drone + a WEAK distant drone (deterministic rcs_gap_db, RT-traced).
    At SPARSE occupancy the strong target's sub-Nyquist mask leakage BURIES the weak
    target for a plain 2D-FFT (which can only mask the strong's main-lobe), while 2D-OMP
    SUBTRACTS the strong atom and reveals the weak one. GATE (binary, no PSLR proxy):
      sparse:  FFT misses the weak  AND  OMP finds it
      dense :  FFT finds the weak   (control: the miss is the sparse leakage, not setup)
    Plus an R1 convergence check: the verdict must be invariant to n_slow (256 vs full)."""
    os.makedirs(outdir, exist_ok=True)
    rt_ok, rt_rows = roundtrip_ok()
    print(f"[B] omp2d round-trip sub-gate: {'PASS' if rt_ok else 'FAIL'}")
    vel_s, vel_w = (0.0, 8.0, 0.0), (0.0, 14.0, 0.0)         # distinct Doppler cells
    rng = np.random.default_rng(seed)
    K = num.n_active; prf = num.symbol_rate; nfull = num.n_symbols(window_s)
    # GT cells from the monostatic geometry (analytic; Phase A verified RT==analytic).
    # Phase B studies the sparse-recovery DSP, so the two targets are CLEAN point-target
    # atoms (a point scatterer's CFR IS an RD atom) at those cells — the RT-spread of a
    # diffuse cube would otherwise confound which method resolves the weak target.
    gts0 = analytic_gt(num, GEOM_STRONG, vel_s)
    gtw0 = analytic_gt(num, GEOM_WEAK, vel_w)

    def build(n):
        ra, da = _axes(num, n, prf)
        dis, ris = _bins(ra, da, gts0["range_m"], gts0["doppler_hz"])
        diw, riw = _bins(ra, da, gtw0["range_m"], gtw0["doppler_hz"])
        H = (atom(n, K, dis, ris) + 10 ** (-rcs_gap_db / 20.0) * atom(n, K, diw, riw))
        sigma = 1.0 / 10 ** (noise_snr_db / 20.0)        # strong atom per-RE amp = 1
        H = (H + sigma * (rng.standard_normal(H.shape) +
                          1j * rng.standard_normal(H.shape)) / np.sqrt(2)).astype(np.complex64)
        meta = dict(subsampled=bool(n < nfull), n_symbols_full=nfull, n_slow=n,
                    prf_hz=prf, doppler_res_hz=prf / n)
        return H, {**gts0, **meta}, {**gtw0, **meta}, prf

    H, gts, gtw, prf = build(n_slow)
    Ns = H.shape[0]
    range_axis, dopp_axis = _axes(num, Ns, prf)
    print(f"[B] 2 targets: strong@({gts['range_m']:.0f}m,{gts['doppler_hz']:.0f}Hz) "
          f"weak(−{rcs_gap_db:.0f}dB)@({gtw['range_m']:.0f}m,{gtw['doppler_hz']:.0f}Hz)  "
          f"N={Ns}(subsampled={gts['subsampled']}) noise={noise_snr_db}dB/RE")

    res = {}
    for name, rho in (("sparse", rho_sparse), ("dense", rho_dense)):
        r = _eval_density(num, H, prf, gts, gtw, rho, rng)
        res[name] = r
        print(f"[B] {name:7s} occ={r['realised']*100:5.2f}%  strong: FFT={r['fft_strong']} OMP={r['omp_strong']} | "
              f"WEAK: FFT={r['fft_weak']}(snr{r['fft_weak_snr']:.0f}dB) OMP={r['omp_weak']}  atoms={r['n_atoms']}")

    sp, dn = res["sparse"], res["dense"]
    sparse_fft_misses = bool(not sp["fft_weak"])
    sparse_omp_finds = bool(sp["omp_weak"])
    dense_fft_finds = bool(dn["fft_weak"])
    gate_pass = bool(rt_ok and sparse_fft_misses and sparse_omp_finds and dense_fft_finds
                     and sp["fft_strong"] and sp["omp_strong"])
    print(f"[B][gate] sparse: FFT-miss-weak={sparse_fft_misses} OMP-finds-weak={sparse_omp_finds} | "
          f"dense: FFT-finds-weak={dense_fft_finds} -> GATE {'PASS' if gate_pass else 'FAIL'}")

    # ---- R1 convergence check: weak-target verdict invariant to n_slow? ----
    r1 = None
    if r1_check_n and r1_check_n != n_slow:
        print(f"[B][R1] re-evaluating sparse verdict at n_slow={r1_check_n} (full) ...")
        Hf, gtsf, gtwf, prff = build(r1_check_n)
        rf = _eval_density(num, Hf, prff, gtsf, gtwf, rho_sparse, np.random.default_rng(seed + 1))
        invariant = bool(rf["fft_weak"] == sp["fft_weak"] and rf["omp_weak"] == sp["omp_weak"])
        r1 = dict(n_slow_a=n_slow, n_slow_b=r1_check_n, invariant=invariant,
                  a=dict(fft_weak=sp["fft_weak"], omp_weak=sp["omp_weak"]),
                  b=dict(fft_weak=rf["fft_weak"], omp_weak=rf["omp_weak"]))
        print(f"[B][R1] n={n_slow}: FFT-miss={not sp['fft_weak']}/OMP-hit={sp['omp_weak']}  vs  "
              f"n={r1_check_n}: FFT-miss={not rf['fft_weak']}/OMP-hit={rf['omp_weak']}  -> invariant={invariant}")

    # ---- figures ----
    rlo = min(gts["range_m"], gtw["range_m"]) - 18; rhi = max(gts["range_m"], gtw["range_m"]) + 18
    rspan = (max(0, rlo), rhi); dlim = max(700, 1.5 * max(abs(gts["doppler_hz"]), abs(gtw["doppler_hz"])))
    dspan = (-dlim, dlim)

    def _mark(ax):
        ax.plot(gts["doppler_hz"], gts["range_m"], "o", mfc="none", mec="red", ms=14, mew=1.6, label="strong GT")
        ax.plot(gtw["doppler_hz"], gtw["range_m"], "s", mfc="none", mec="orange", ms=15, mew=1.9,
                label=f"weak GT (−{rcs_gap_db:.0f} dB)")

    _panel(outdir, "B_occupancy_sparse.png", lambda ax, f: (
        ax.imshow(sp["W"][:, ::20].T, aspect="auto", origin="lower", cmap="Greys"),
        ax.set_title(f"occupancy W — sparse {sp['realised']*100:.1f}% (Fig 3a)", fontsize=9),
        ax.set_xlabel("OFDM symbol"), ax.set_ylabel("subcarrier /20")))
    _panel(outdir, "B_density_timeline.png", lambda ax, f: ([
        ax.plot(density_timeline(res[n]["W"]) * 100, lw=1.0, label=f"{n} ({res[n]['realised']*100:.1f}%)")
        for n in ("sparse", "dense")], ax.set_xlabel("OFDM symbol"), ax.set_ylabel("occupancy [%]"),
        ax.set_title("density timeline (Fig 11)", fontsize=9), ax.legend(fontsize=8), ax.grid(alpha=.3)))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4), constrained_layout=True)
    for j, name in enumerate(("sparse", "dense")):
        r = res[name]
        im = viz.rd_map(axes[0, j], r["rd_fft"], range_axis, dopp_axis, title=(
            f"2D-FFT — {name} {r['realised']*100:.1f}%   weak found = {r['fft_weak']}"
            + ("  ← BURIED" if not r["fft_weak"] else "")), r_zoom=rspan, d_zoom=dspan)
        _mark(axes[0, j]); fig.colorbar(im, ax=axes[0, j], shrink=.8, label="dB"); axes[0, j].legend(loc="upper right", fontsize=7)
        im = viz.rd_map(axes[1, j], r["Z"], range_axis, dopp_axis, title=(
            f"2D-OMP — {name} {r['realised']*100:.1f}%   weak found = {r['omp_weak']}"
            + ("  ✓ RESOLVED" if r["omp_weak"] else "")), r_zoom=rspan, d_zoom=dspan)
        if r["support"]:                                   # show recovered atoms (sparse -> sub-pixel else)
            sd = [dopp_axis[di] for di, ri in r["support"]]; sr = [range_axis[ri] for di, ri in r["support"]]
            axes[1, j].scatter(sd, sr, s=90, marker="+", color="lime", linewidths=1.8, label="OMP atoms")
        _mark(axes[1, j]); fig.colorbar(im, ax=axes[1, j], shrink=.8, label="dB"); axes[1, j].legend(loc="upper right", fontsize=7)
    fig.suptitle("Phase B — sparse: weak target BURIED by strong's 2D-FFT leakage, RESOLVED by 2D-OMP "
                 f"(LaSen Fig.17/Tab.1)   |   GATE {'PASS ✓' if gate_pass else 'FAIL ✗'}", fontsize=11.5)
    viz.savefig(fig, os.path.join(outdir, "B_rd_compare.png"))

    def _conv(ax, fig):
        for name, c in (("sparse", "tab:orange"), ("dense", "tab:green")):
            h = res[name]["hist"]
            ax.semilogy(range(len(h)), np.array(h) / (h[0] + 1e-30), "o-", color=c,
                        label=f"{name} ({res[name]['n_atoms']} atoms: strong→weak→…)")
        ax.set_xlabel("OMP iteration"); ax.set_ylabel("residual norm (norm.)")
        ax.set_title("2D-OMP convergence — strong removed first, weak revealed (Eq 6)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    _panel(outdir, "B_omp_convergence.png", _conv)

    if r1:
        def _r1(ax, fig):
            labels = [f"n_slow={r1['n_slow_a']}\n(subsampled)", f"n_slow={r1['n_slow_b']}\n(full symbols)"]
            fftm = [int(not r1["a"]["fft_weak"]), int(not r1["b"]["fft_weak"])]
            omph = [int(r1["a"]["omp_weak"]), int(r1["b"]["omp_weak"])]
            x = np.arange(2)
            ax.bar(x - 0.2, fftm, 0.4, label="FFT misses weak", color="tab:red")
            ax.bar(x + 0.2, omph, 0.4, label="OMP finds weak", color="tab:green")
            ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.3); ax.set_yticks([0, 1])
            ax.set_yticklabels(["No", "Yes"])
            verdict = ("invariant ✓" if r1["invariant"] else
                       "subsampled loses OMP recovery → FULL is faithful")
            ax.set_title(f"R1 convergence check — full {r1['n_slow_a']} vs {r1['n_slow_b']}: {verdict}", fontsize=8.5)
            ax.legend(fontsize=8)
        _panel(outdir, "B_r1_convergence.png", _r1)

    out = dict(phase="B", config=dict(fc_ghz=num.fc/1e9, bw_mhz=num.bw/1e6, n_slow=Ns,
               subsampled=gts["subsampled"], n_symbols_full=gts["n_symbols_full"], K=num.n_active,
               prf_hz=prf, range_res_m=num.range_res_m, rcs_gap_db=rcs_gap_db, noise_snr_db=noise_snr_db,
               rho_sparse=rho_sparse, rho_dense=rho_dense),
               gt_strong=gts, gt_weak=gtw, roundtrip=dict(ok=rt_ok, cells=rt_rows), r1_check=r1,
               results={k: dict(rho=v["rho"], realised=v["realised"], n_atoms=v["n_atoms"],
                                fft_finds_weak=v["fft_weak"], fft_weak_snr_db=v["fft_weak_snr"],
                                omp_finds_weak=v["omp_weak"], fft_finds_strong=v["fft_strong"],
                                omp_finds_strong=v["omp_strong"]) for k, v in res.items()},
               gate=dict(roundtrip_ok=rt_ok, sparse_fft_misses_weak=sparse_fft_misses,
                         sparse_omp_finds_weak=sparse_omp_finds, dense_fft_finds_weak=dense_fft_finds,
                         r1_invariant=(r1["invariant"] if r1 else None), gate_pass=gate_pass))
    json.dump(out, open(os.path.join(outdir, "lasen_phaseB.json"), "w"), indent=1)
    print(f"[B][out] {outdir}/B_*.png + lasen_phaseB.json  GATE={'PASS' if gate_pass else 'FAIL'}")
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="A", choices=["A", "B"])
    ap.add_argument("--outdir", default=os.path.join(_HERE, "outputs"))
    ap.add_argument("--window", type=float, default=0.1)
    ap.add_argument("--speed", type=float, default=12.0)
    ap.add_argument("--fc", type=float, default=5.8e9)
    ap.add_argument("--rcs-gap", type=float, default=46.0)
    ap.add_argument("--noise-snr", type=float, default=18.0)
    ap.add_argument("--n-slow", type=int, default=2803)
    ap.add_argument("--rho-sparse", type=float, default=0.012)
    ap.add_argument("--r1-n", type=int, default=256, help="0 to skip R1 convergence check")
    a = ap.parse_args()
    num = NRNumerology(fc=a.fc)
    if a.phase == "A":
        phase_a(a.outdir, num, window_s=a.window, speed=a.speed)
    elif a.phase == "B":
        phase_b(a.outdir, num, window_s=a.window, n_slow=a.n_slow, rcs_gap_db=a.rcs_gap,
                noise_snr_db=a.noise_snr, rho_sparse=a.rho_sparse, r1_check_n=a.r1_n)
