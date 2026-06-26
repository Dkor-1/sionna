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
from monostatic_scene import GEOM, trace_cfr
from occupancy import DENSITY_BINS, mask_at_density, density_timeline
from omp2d import rd_transform, omp2d, roundtrip_ok
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


def phase_b(outdir, num, window_s=0.1, speed=12.0, n_slow=256,
            densities=("sparse", "dense"), noise_snr_db=0.0, seed=1):
    """Phase B — non-uniform occupancy + 2D-OMP. From the SAME masked CFR we compare a
    plain 2D-FFT (range = freq IFFT, doppler = slow-time FFT, with zeros at the
    untransmitted REs) against 2D-OMP. Both recover the target's PEAK (for a single
    target OMP's 1st atom == the FFT peak), but the 2D-FFT has a sub-Nyquist LEAKAGE
    FLOOR that rises as occupancy gets sparser (limited dynamic range), while 2D-OMP
    fits the masked observation with a sparse model -> a CLEAN RD (huge dynamic range).
    That dynamic-range deficit is exactly what LaSen's sparse recovery overcomes
    (Tab.1). GATE: OMP recovers the target at sparse & dense; OMP RD is far cleaner
    than the FFT (PSLR gap), and the FFT floor measurably worsens from dense->sparse."""
    os.makedirs(outdir, exist_ok=True)
    rt_ok, rt_rows = roundtrip_ok()
    print(f"[B] omp2d round-trip sub-gate: {'PASS' if rt_ok else 'FAIL'}")
    vel = (0.0, speed, 0.0); rng = np.random.default_rng(seed)
    H, freqs, gt = trace_cfr(num, GEOM, vel, n_slow=n_slow)
    Ns, K = H.shape; prf = gt["prf_hz"]
    # light receiver noise so the FFT leakage floor is a realistic floor (not -inf)
    mov = H - H.mean(axis=0, keepdims=True)
    echo_rms = float(np.sqrt(np.mean(np.abs(mov) ** 2)))
    sigma = echo_rms / 10 ** (noise_snr_db / 20.0)
    H = (H + sigma * (rng.standard_normal(H.shape) +
                      1j * rng.standard_normal(H.shape)) / np.sqrt(2)).astype(np.complex64)
    range_axis = np.arange(K) * num.range_res_m
    dopp_axis = np.fft.fftshift(np.fft.fftfreq(Ns, d=1.0 / prf))
    ri_gt = int(np.argmin(np.abs(range_axis - gt["range_m"])))
    di_gt = int(np.argmin(np.abs(dopp_axis - gt["doppler_hz"])))
    print(f"[B] CFR N={Ns}(subsampled={gt['subsampled']}) K={K} prf={prf:.0f}Hz "
          f"GT@({gt['range_m']:.0f}m,{gt['doppler_hz']:.0f}Hz) noise={noise_snr_db}dB/RE")

    res = {}
    for name in densities:
        lo, hi = DENSITY_BINS[name]; rho = (lo + hi) / 2.0
        W, realised = mask_at_density(Ns, K, rho, rng)
        Hs = suppress_masked(H, W)
        rd_fft = rd_transform(Hs)                          # (a) plain 2D-FFT (leaky)
        Z, support, hist = omp2d(Hs, W)                    # (b) 2D-OMP (Eq 5-6)
        m_fft = _peak_metrics(np.abs(rd_fft) ** 2, ri_gt, di_gt, range_axis, dopp_axis, num)
        m_omp = _peak_metrics(np.abs(Z) ** 2, ri_gt, di_gt, range_axis, dopp_axis, num)
        res[name] = dict(rho=rho, realised=realised, W=W, rd_fft=rd_fft, Z=Z, support=support,
                         hist=hist, fft=m_fft, omp=m_omp, n_atoms=len(support))
        print(f"[B] {name:7s} occ={realised*100:5.2f}%  FFT on_GT={m_fft['peak_on_gt']} "
              f"floor-PSLR={m_fft['pslr_db']:5.1f}dB | OMP on_GT={m_omp['peak_on_gt']} "
              f"PSLR={m_omp['pslr_db']:5.1f}dB atoms={len(support)}")

    # ---- GATE: OMP recovers target both densities; OMP far cleaner than FFT (dynamic
    # range); FFT floor worsens dense->sparse (leakier when sparser).
    sp = res["sparse"]; dn = res["dense"]
    omp_recovers = bool(sp["omp"]["peak_on_gt"] and dn["omp"]["peak_on_gt"])
    dyn_range_gap = bool((sp["omp"]["pslr_db"] - sp["fft"]["pslr_db"]) > 20.0)
    fft_leakier_sparse = bool(sp["fft"]["pslr_db"] < dn["fft"]["pslr_db"] - 2.0)
    gate_pass = bool(rt_ok and omp_recovers and dyn_range_gap and fft_leakier_sparse)
    print(f"[B][gate] roundtrip={rt_ok} omp_recovers={omp_recovers} "
          f"dyn_range_gap={dyn_range_gap}(ΔPSLR={sp['omp']['pslr_db']-sp['fft']['pslr_db']:.0f}dB) "
          f"FFT_leakier_sparse={fft_leakier_sparse}({sp['fft']['pslr_db']:.0f}<{dn['fft']['pslr_db']:.0f}) "
          f"-> GATE {'PASS' if gate_pass else 'FAIL'}")

    # ---- figures (R3 per-panel) ----
    rspan = (max(0, gt["range_m"] - 35), gt["range_m"] + 35)
    dlim = max(700, 1.6 * abs(gt["doppler_hz"])); dspan = (-dlim, dlim)

    def _occ(ax, fig):
        ax.imshow(sp["W"][:, ::20].T, aspect="auto", origin="lower", cmap="Greys")
        ax.set_title(f"occupancy mask W — sparse {sp['realised']*100:.1f}% (Fig 3a)", fontsize=9)
        ax.set_xlabel("OFDM symbol"); ax.set_ylabel("subcarrier /20")
    _panel(outdir, "B_occupancy_sparse.png", _occ)

    def _occd(ax, fig):
        ax.imshow(dn["W"][:, ::20].T, aspect="auto", origin="lower", cmap="Greys")
        ax.set_title(f"occupancy mask W — dense {dn['realised']*100:.1f}%", fontsize=9)
        ax.set_xlabel("OFDM symbol"); ax.set_ylabel("subcarrier /20")
    _panel(outdir, "B_occupancy_dense.png", _occd)

    def _timeline(ax, fig):
        for name, c in (("sparse", "tab:orange"), ("dense", "tab:green")):
            ax.plot(density_timeline(res[name]["W"]) * 100, color=c, lw=1.0,
                    label=f"{name} (mean {res[name]['realised']*100:.1f}%)")
        ax.set_xlabel("OFDM symbol (slow time)"); ax.set_ylabel("occupancy [%]")
        ax.set_title("occupancy density timeline (Fig 11/5b)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    _panel(outdir, "B_density_timeline.png", _timeline)

    # money figure: 2x2 RD {FFT,OMP} x {sparse,dense}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), constrained_layout=True)
    for j, name in enumerate(("sparse", "dense")):
        r = res[name]
        im = viz.rd_map(axes[0, j], r["rd_fft"], range_axis, dopp_axis, gt=gt, title=(
            f"2D-FFT — {name} {r['realised']*100:.1f}%   leakage-floor PSLR={r['fft']['pslr_db']:.0f} dB "
            f"(on_GT={r['fft']['peak_on_gt']})"), r_zoom=rspan, d_zoom=dspan)
        fig.colorbar(im, ax=axes[0, j], shrink=.8, label="dB"); axes[0, j].legend(loc="upper right", fontsize=7)
        im = viz.rd_map(axes[1, j], r["Z"], range_axis, dopp_axis, gt=gt, title=(
            f"2D-OMP — {name} {r['realised']*100:.1f}%   clean PSLR={r['omp']['pslr_db']:.0f} dB "
            f"(on_GT={r['omp']['peak_on_gt']}, {r['n_atoms']} atoms)"), r_zoom=rspan, d_zoom=dspan)
        fig.colorbar(im, ax=axes[1, j], shrink=.8, label="dB"); axes[1, j].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Phase B — 2D-FFT leakage floor (worse when sparser) vs 2D-OMP clean sparse RD (LaSen Tab.1)"
                 f"   |   GATE {'PASS ✓' if gate_pass else 'FAIL ✗'}", fontsize=12)
    viz.savefig(fig, os.path.join(outdir, "B_rd_compare.png"))

    def _conv(ax, fig):
        for name, c in (("sparse", "tab:orange"), ("dense", "tab:green")):
            h = res[name]["hist"]
            ax.semilogy(range(len(h)), np.array(h) / (h[0] + 1e-30), "o-", color=c,
                        label=f"{name} ({len(h)-1} iters, {len(res[name]['support'])} atoms)")
        ax.set_xlabel("OMP iteration"); ax.set_ylabel("residual norm (norm.)")
        ax.set_title("2D-OMP convergence (Eq 6)", fontsize=9); ax.legend(fontsize=8); ax.grid(alpha=.3)
    _panel(outdir, "B_omp_convergence.png", _conv)

    out = dict(phase="B", config=dict(fc_ghz=num.fc/1e9, bw_mhz=num.bw/1e6, n_slow=Ns,
               subsampled=gt["subsampled"], n_symbols_full=gt["n_symbols_full"],
               K=K, prf_hz=prf, range_res_m=num.range_res_m, speed_ms=speed,
               noise_snr_db=noise_snr_db), gt=gt, roundtrip=dict(ok=rt_ok, cells=rt_rows),
               results={k: dict(rho=v["rho"], realised=v["realised"], n_atoms=v["n_atoms"],
                                fft_on_gt=v["fft"]["peak_on_gt"], fft_pslr_db=v["fft"]["pslr_db"],
                                omp_on_gt=v["omp"]["peak_on_gt"], omp_pslr_db=v["omp"]["pslr_db"])
                        for k, v in res.items()},
               gate=dict(roundtrip_ok=rt_ok, omp_recovers=omp_recovers,
                         dyn_range_gap=dyn_range_gap, fft_leakier_sparse=fft_leakier_sparse,
                         gate_pass=gate_pass))
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
    a = ap.parse_args()
    num = NRNumerology(fc=a.fc)
    if a.phase == "A":
        phase_a(a.outdir, num, window_s=a.window, speed=a.speed)
    elif a.phase == "B":
        phase_b(a.outdir, num, window_s=a.window, speed=a.speed)
