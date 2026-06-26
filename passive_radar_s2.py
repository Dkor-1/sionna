#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 fair multi-mode passive-radar benchmark (Sionna RT 2.0.1)
=================================================================

Compares the 4 project signal modes along the *reference-structure sparsity*
axis (PROJECT_CONTEXT sec.3) under a strict control protocol:

    M1 wifi_preamble    : Wi-Fi-like, preamble-rich (full-band known symbols)
    M2 lte_crs          : LTE-like, CRS comb pilots (sym 0&4, every 6th SC)
    M3 5g_ssb_sparse    : 5G-like sparse, SSB-like localized block
    M4 5g_dmrs_prs_rich : 5G-like reference-rich, DMRS/CSI-RS/PRS dense comb

Model
-----
A single OFDM resource grid (subcarrier x OFDM-symbol) at fs = B.  Every RE is
transmitted with unit-power QPSK -> identical TX power across modes (fair
illuminator).  A per-mode binary PILOT MASK marks which REs are *known* to the
passive radar.  The radar reference is rebuilt from ONLY the known REs (data REs
zeroed); the surveillance signal is the full grid through the Sionna RT channel.
Sparser mask -> less exploitable reference energy -> lower CAF gain / SCR / Pd.
This is the realistic way passive radar exploits CRS/SSB/DMRS/preamble and is
the project's core hypothesis test.

Control protocol (held identical across modes): geometry, trajectory, channel,
bandwidth B, FFT size, subcarrier spacing, CPI (N,M), total TX power, noise.
The single varied factor is the pilot mask (reference structure).

Stage-1 scope still applies: bulk translational Doppler only (no blade mesh).
Clutter/DPI cancellation = ideal known-static-channel removal (proxy for ECA;
data-driven ECA is a later step).

Reuses the validated Stage-1 scene/trace/CAF/CFAR code.
"""
from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from passive_radar_stage1 import (
    Config, C0, build_scene, trace_channel, _conv_batches,
    caf_range_doppler, ca_cfar_2d,
)

MODES = ["wifi_preamble", "lte_crs", "5g_ssb_sparse", "5g_dmrs_prs_rich"]
MODE_LABEL = {
    "wifi_preamble": "Wi-Fi-like\n(preamble-rich)",
    "lte_crs": "LTE-like\n(CRS comb)",
    "5g_ssb_sparse": "5G-like sparse\n(SSB block)",
    "5g_dmrs_prs_rich": "5G-like rich\n(DMRS/PRS)",
}
SHORT = {"wifi_preamble": "Wi-Fi", "lte_crs": "LTE",
         "5g_ssb_sparse": "5G-SSB", "5g_dmrs_prs_rich": "5G-rich"}

# Fixed seed/values for the *known* pilot resources (deterministic == radar can
# regenerate them without demodulation, like real CRS/SSB/DMRS/preamble).
PILOT_SEED = 0xC0FFEE


# --------------------------------------------------------------------------- #
#  OFDM resource grid + per-mode pilot masks
# --------------------------------------------------------------------------- #
@dataclass
class OFDM:
    n_fft: int = 4096
    cp: int = 512

    @property
    def sym_len(self) -> int:
        return self.n_fft + self.cp


def _qpsk(rng: np.random.Generator, shape) -> np.ndarray:
    b = rng.integers(0, 4, size=shape)
    return np.exp(1j * (np.pi / 4 + b * np.pi / 2)).astype(np.complex64)


def pilot_mask(mode: str, n_sym: int, n_fft: int) -> np.ndarray:
    """Boolean [n_sym, n_fft] grid: True where the RE is a KNOWN reference RE.
    Patterns are standards-inspired; refine from the paper-survey later."""
    m = np.zeros((n_sym, n_fft), bool)
    sc = np.arange(n_fft)
    if mode == "wifi_preamble":
        # 802.11 preamble-rich: first N_PRE symbols of each packet, ALL SCs
        # known (dense in freq, sparse/periodic in time). ~18% (WiFi duty ~Fu).
        pkt_period, n_pre = 22, 4
        m[(np.arange(n_sym) % pkt_period) < n_pre, :] = True          # ~18.2%
    elif mode == "lte_crs":
        # CRS-like: symbols {0,4,7,11} of a 14-symbol slot, comb-6 in frequency
        slot = np.arange(n_sym) % 14
        crs_sym = np.isin(slot, [0, 4, 7, 11])
        comb = (sc % 6) == 0
        m[np.ix_(crs_sym, comb)] = True                              # ~4.8%
    elif mode == "5g_ssb_sparse":
        # SSB-like: localized 4-symbol x 240-SC block, periodic (sparsest mode)
        ssb_period, ssb_nsym, ssb_nsc = 120, 4, 240   # ~0.2% (spec floor ~0.29%)
        sc0 = (n_fft - ssb_nsc) // 2
        for s0 in range(0, n_sym, ssb_period):
            m[s0:s0 + ssb_nsym, sc0:sc0 + ssb_nsc] = True   # ~0.2% (4/120 * 240/n_fft)
    elif mode == "5g_dmrs_prs_rich":
        # PRS/DMRS-rich: staggered comb-2 over 8 of every 14 symbols (densest)
        pilot_syms = {2, 3, 5, 6, 8, 9, 11, 12}
        for s in range(n_sym):
            if (s % 14) in pilot_syms:
                m[s, (sc % 2) == (s % 2)] = True       # staggered comb-2  ~28.6%
    else:
        raise ValueError(mode)
    return m


def synth_ofdm(cfg: Config, ofdm: OFDM, mode: str, rng: np.random.Generator,
               full_ref: bool = False):
    """Return (s_full, s_ref, known_fraction).
    s_full : transmitted illuminator (all REs, unit power)  -> surveillance input
    s_ref  : reference for the CAF.
             full_ref=False -> rebuilt from KNOWN pilot REs only (realistic,
             reference-structure-limited; the project's hypothesis regime).
             full_ref=True  -> the full transmitted signal (clean reference-
             antenna capture; upper bound, reference structure irrelevant)."""
    K = cfg.N * cfg.M
    sym_len = ofdm.sym_len
    n_sym = int(np.ceil((K + sym_len) / sym_len))

    mask = pilot_mask(mode, n_sym, ofdm.n_fft)
    pilot_vals = _qpsk(np.random.default_rng(PILOT_SEED), (n_sym, ofdm.n_fft))
    data_vals = _qpsk(rng, (n_sym, ofdm.n_fft))           # unknown to radar
    grid_full = np.where(mask, pilot_vals, data_vals)
    grid_ref = np.where(mask, pilot_vals, 0.0).astype(np.complex64)

    def modulate(grid):
        t = np.fft.ifft(np.fft.ifftshift(grid, axes=1), axis=1)   # [n_sym, n_fft]
        t *= np.sqrt(ofdm.n_fft)                                   # unit-power REs
        with_cp = np.concatenate([t[:, -ofdm.cp:], t], axis=1)    # add CP
        return with_cp.reshape(-1)[:K].astype(np.complex64)

    s_full = modulate(grid_full)
    s_ref = s_full if full_ref else modulate(grid_ref)
    scale = 1.0 / np.sqrt(np.mean(np.abs(s_full) ** 2))   # fair unit TX power
    kf = 1.0 if full_ref else float(mask.mean())
    return s_full * scale, s_ref * scale, kf


# --------------------------------------------------------------------------- #
#  Surveillance synthesis + clutter cancellation (reuses Stage-1)
# --------------------------------------------------------------------------- #
def _conv_fft(s: np.ndarray, h: np.ndarray, N: int, M: int) -> np.ndarray:
    """Vectorised time-varying batch convolution (FFT) — identical result to
    Stage-1 `_conv_batches` but ~all batches at once. h may be [N,L] (per-batch)
    or [1,L] (static, broadcast)."""
    L = h.shape[1]
    sp = np.concatenate([np.zeros(L - 1, np.complex64), s])
    seg = np.lib.stride_tricks.sliding_window_view(sp, M + L - 1)[::M][:N]  # [N,M+L-1]
    nfft = 1 << int(np.ceil(np.log2(M + 2 * L - 2)))
    full = np.fft.ifft(np.fft.fft(seg, nfft, axis=1) *
                       np.fft.fft(h, nfft, axis=1), axis=1)
    return full[:, L - 1:L - 1 + M].astype(np.complex64)                    # 'valid'


def surveillance(cfg: Config, s_full: np.ndarray, h: np.ndarray,
                 drone_tap: int, snr_db: float, rng: np.random.Generator,
                 noise_pow: float | None = None):
    """noise_pow=None -> noise referenced to THIS drone's echo (fixed per-sample
    SNR; used for the fair signal-MODE comparison). Pass an absolute noise_pow to
    compare DRONES: then a larger-RCS drone yields a higher effective SNR."""
    N, M = cfg.N, cfg.M
    X = _conv_fft(s_full, h, N, M)
    p_drone = float(np.mean(np.abs(h[:, drone_tap]) ** 2) *
                    np.mean(np.abs(s_full) ** 2))
    if noise_pow is None:
        noise_pow = p_drone / (10 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_pow / 2.0)
    X = X + (sigma * (rng.standard_normal((N, M))
                      + 1j * rng.standard_normal((N, M)))).astype(np.complex64)
    h_static = h.mean(axis=0, keepdims=True).astype(np.complex64)   # [1,L] broadcast
    X_clean = X - _conv_fft(s_full, h_static, N, M)        # ideal DPI/clutter cancel
    return X_clean


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def rd_metrics(power, range_axis, dopp_axis, gt, cfg, det=None):
    """Metrics for one RD map, with mode-symmetric exclusions (DSP-review M1/M2).

    Background (mean-SCR) and the empirical-FAR region exclude the clutter notch
    AND the target's range-Doppler *cross* — the range-sidelobe column at the
    target Doppler and the Doppler-sidelobe/grating-lobe row at the target range.
    Same geometry for all modes, so FAR measures noise, not each mask's ambiguity
    structure. PSLR deliberately INCLUDES those ambiguities (it is the ambiguity
    diagnostic — e.g. Wi-Fi preamble grating lobes). SCR uses the mean background
    (spec C.1).

    Detection scoring (review fixes #2, i):
      * Pd hit window is METRES in range (|R-R_gt| <= cfg.hit_tol_m) and +-1 bin in
        Doppler — a fixed +-1 RANGE bin would be c/B wide and bias Pd ~20x toward
        narrow band. Doppler res 1/CPI is fixed across the matrix, so +-1 Doppler
        bin is already band-fair.
      * If the target's true Doppler is inside the clutter notch (hover / pure
        tangential), the cell is structurally removed by the ideal canceller +
        notch -> SCR/Pd there are meaningless; we return scr_db=NaN and hit=False
        (do NOT average a notch value into the SCR)."""
    nr, nd = power.shape
    notch = np.abs(dopp_axis) < cfg.doppler_notch_hz
    ri = int(np.argmin(np.abs(range_axis - gt["bistatic_range_m"])))
    di = int(np.argmin(np.abs(dopp_axis - gt["doppler_hz"])))
    rW, dW = 2, 2
    # range half-window for the Pd hit, in BINS, derived from a fixed metric tol
    range_res = float(range_axis[1] - range_axis[0]) if nr > 1 else cfg.range_res_m
    hit_rW = max(1, int(round(cfg.hit_tol_m / max(range_res, 1e-9))))
    target_in_notch = bool(np.abs(gt["doppler_hz"]) < cfg.doppler_notch_hz)

    # local target peak (handles straddle) and its position
    r0, r1 = max(0, ri-1), min(nr, ri+2)
    d0, d1 = max(0, di-1), min(nd, di+2)
    sub = power[r0:r1, d0:d1]
    pkl = np.unravel_index(int(np.argmax(sub)), sub.shape)
    p_t = float(sub.max())
    pk_r, pk_d = float(range_axis[r0 + pkl[0]]), float(dopp_axis[d0 + pkl[1]])

    # clean noise region = exclude notch + target range row + target Doppler col
    clean = np.ones_like(power, bool)
    clean[:, notch] = False
    clean[max(0, ri-rW):ri+rW+1, :] = False     # target range row (Doppler/grating lobes)
    clean[:, max(0, di-dW):di+dW+1] = False      # target Doppler col (range sidelobes)
    bg = power[clean]
    scr = (np.nan if target_in_notch
           else 10 * np.log10(p_t / (bg.mean() + 1e-30) + 1e-30))

    # PSLR includes ambiguities: max over all minus notch minus target main-lobe
    psl_reg = np.ones_like(power, bool); psl_reg[:, notch] = False
    psl_reg[max(0, ri-2):ri+3, max(0, di-2):di+3] = False
    psl = 10 * np.log10(p_t / (power[psl_reg].max() + 1e-30) + 1e-30)

    out = dict(scr_db=float(scr), psl_db=float(psl), pk_r=pk_r, pk_d=pk_d,
               ri=ri, di=di)
    if det is not None:
        det = det.copy(); det[:, notch] = False
        # range window in METRES (hit_rW bins), Doppler window +-1 bin; no hit if
        # the target is structurally inside the clutter notch (review fixes #2, i)
        hit = (False if target_in_notch else
               det[max(0, ri-hit_rW):ri+hit_rW+1, max(0, di-1):di+2].any())
        fa = det & clean                          # FAs only in clean noise region
        out.update(hit=bool(hit), n_fa=int(fa.sum()), n_test=int(clean.sum()))
    return out


def run_mode(cfg: Config, ofdm: OFDM, mode: str, h, gt, drone_tap,
             snr_db, n_trials, base_seed, noise_pow=None, full_ref=False):
    """Monte-Carlo over data+noise realizations for one signal mode.
    noise_pow=None -> per-drone-normalised SNR (mode comparison); set absolute
    noise_pow to compare drones (RCS then drives effective SNR).
    full_ref=True -> clean reference-antenna upper bound."""
    scrs, psls, hits, fas, n_tests, peaksR, peaksD = [], [], [], [], [], [], []
    rd_show = known_frac = None
    for t in range(n_trials):
        rng = np.random.default_rng(base_seed + 7919 * t)   # SAME seq across modes
        s_full, s_ref, known_frac = synth_ofdm(cfg, ofdm, mode, rng, full_ref)
        X_clean = surveillance(cfg, s_full, h, drone_tap, snr_db, rng, noise_pow)
        Rmat = s_ref.reshape(cfg.N, cfg.M)
        rd, range_axis, dopp_axis = caf_range_doppler(cfg, X_clean, Rmat, mti=False)
        power = np.abs(rd) ** 2
        det, *_ = ca_cfar_2d(power, cfg, dopp_axis)
        mt = rd_metrics(power, range_axis, dopp_axis, gt, cfg, det)
        scrs.append(mt["scr_db"]); psls.append(mt["psl_db"]); hits.append(mt["hit"])
        fas.append(mt["n_fa"]); n_tests.append(mt["n_test"])
        peaksR.append(mt["pk_r"]); peaksD.append(mt["pk_d"])     # target-local peak
        if t == 0:
            rd_show = (rd, range_axis, dopp_axis, det)
    scrs = np.array(scrs)
    res = dict(
        mode=mode, known_fraction=known_frac,
        scr_db_mean=float(scrs.mean()), scr_db_std=float(scrs.std()),
        psl_db_mean=float(np.mean(psls)),
        pd=float(np.mean(hits)),
        far=float(np.sum(fas) / max(1, np.sum(n_tests))),
        peakR_std=float(np.std(peaksR)), peakD_std=float(np.std(peaksD)),
        n_trials=n_trials, snr_db=snr_db,
    )
    return res, rd_show


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #
def _db(x):
    p = np.abs(x) ** 2
    return 10 * np.log10(np.maximum(p, p.max() * 1e-9) / p.max())


def plot_rd_grid(cfg, gt, shows, results, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    gR, gD = gt["bistatic_range_m"], gt["doppler_hz"]
    d_lim = float(min(6000.0, max(600.0, 6 * abs(gD))))
    r_hi = float(min(600.0, max(60.0, 3 * gR)))
    for ax, mode in zip(axes.ravel(), MODES):
        rd, range_axis, dopp_axis, det = shows[mode]
        extent = [dopp_axis[0], dopp_axis[-1], range_axis[0], range_axis[-1]]
        im = ax.imshow(_db(rd), aspect="auto", origin="lower", extent=extent,
                       cmap="viridis", vmin=-40, vmax=0)
        ax.set_xlim(-d_lim, d_lim); ax.set_ylim(0, r_hi)
        ax.plot(gD, gR, "o", mfc="none", mec="red", ms=14, mew=1.6)
        notch = np.abs(dopp_axis) < cfg.doppler_notch_hz
        dd = det.copy(); dd[:, notch] = False
        rr, cc = np.where(dd)
        if rr.size:
            ax.plot(dopp_axis[cc], range_axis[rr], "x", color="red", ms=6, mew=1.2)
        r = results[mode]
        ax.set_title(f"{MODE_LABEL[mode]}  |  known REs={r['known_fraction']*100:.2f}%\n"
                     f"SCR={r['scr_db_mean']:.1f}±{r['scr_db_std']:.1f} dB, "
                     f"Pd={r['pd']:.2f}, FAR={r['far']:.1e}", fontsize=9)
        ax.set_xlabel("Bistatic Doppler [Hz]"); ax.set_ylabel("Bistatic range [m]")
        fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    fig.suptitle(
        f"Fair multi-mode passive-radar benchmark (control protocol)  |  "
        f"fc={cfg.fc/1e9:.2f} GHz, B={cfg.B/1e6:.0f} MHz, CPI={cfg.cpi_s*1e3:.0f} ms  |  "
        f"GT drone R={gR:.0f} m, fD={gD:.0f} Hz, SNR/samp={results[MODES[0]]['snr_db']:.0f} dB",
        fontsize=11)
    fig.savefig(path, dpi=140); plt.close(fig)


def plot_summary(results, path):
    modes = MODES
    kf = [results[m]["known_fraction"] * 100 for m in modes]
    scr = [results[m]["scr_db_mean"] for m in modes]
    scr_e = [results[m]["scr_db_std"] for m in modes]
    pd = [results[m]["pd"] for m in modes]
    labels = [SHORT[m] for m in modes]
    order = np.argsort(kf)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax[0].errorbar(np.array(kf)[order], np.array(scr)[order],
                   yerr=np.array(scr_e)[order], marker="o", capsize=4)
    for i in order:
        ax[0].annotate(labels[i], (kf[i], scr[i]),
                       fontsize=9, xytext=(5, 4), textcoords="offset points")
    ax[0].set_xlabel("Known reference REs  [% of resource grid]")
    ax[0].set_ylabel("SCR  [dB]")
    ax[0].set_title("SCR vs reference density"); ax[0].grid(alpha=.3)
    ax[1].bar(range(len(modes)), pd, color="steelblue")
    ax[1].set_xticks(range(len(modes)))
    ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set_ylabel("Detection rate  Pd"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Pd by signal mode")
    fig.suptitle("Reference-structure sparsity vs detection performance", fontsize=12)
    fig.savefig(path, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
def trace_once(cfg: Config):
    """Trace the fixed-geometry channel a single time (shared across modes/SNRs)."""
    print("[rt] tracing channel once (fixed geometry/channel) ...")
    scene = build_scene(cfg)
    h, gt = trace_channel(cfg, scene)
    drone_tap = int(round(gt["bistatic_delay"] * cfg.fs))
    drone_tap = max(0, min(h.shape[1] - 1, drone_tap))
    print(f"[rt] GT drone R={gt['bistatic_range_m']:.2f} m fD={gt['doppler_hz']:.2f} Hz "
          f"tap={drone_tap}")
    return h, gt, drone_tap


def plot_sweep(snrs, sweep, path):
    """Pd-vs-SNR and SCR-vs-SNR curves per mode."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for mode in MODES:
        pd = [sweep[(mode, s)]["pd"] for s in snrs]
        scr = [sweep[(mode, s)]["scr_db_mean"] for s in snrs]
        lab = f"{SHORT[mode]} ({sweep[(mode, snrs[0])]['known_fraction']*100:.1f}%)"
        ax[0].plot(snrs, pd, marker="o", label=lab)
        ax[1].plot(snrs, scr, marker="o", label=lab)
    ax[0].set_xlabel("Per-sample SNR of drone echo  [dB]"); ax[0].set_ylabel("Pd")
    ax[0].set_ylim(-0.03, 1.03); ax[0].set_title("Detection rate vs SNR"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("Per-sample SNR of drone echo  [dB]"); ax[1].set_ylabel("SCR [dB]")
    ax[1].set_title("SCR vs SNR"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle("Fair benchmark — detection sensitivity by reference structure", fontsize=12)
    fig.savefig(path, dpi=140); plt.close(fig)


def run_sweep(cfg: Config, ofdm: OFDM, snrs, n_trials: int):
    os.makedirs(cfg.outdir, exist_ok=True)
    h, gt, drone_tap = trace_once(cfg)
    sweep = {}
    for snr_db in snrs:
        for mode in MODES:
            res, _ = run_mode(cfg, ofdm, mode, h, gt, drone_tap, snr_db,
                              n_trials, base_seed=cfg.seed)
            sweep[(mode, snr_db)] = res
            print(f"[sweep] snr={snr_db:+5.0f} {mode:18s} "
                  f"SCR={res['scr_db_mean']:5.1f} dB  Pd={res['pd']:.2f}  FAR={res['far']:.1e}")
    png = os.path.join(cfg.outdir, "s2_sweep_pd_snr.png")
    js = os.path.join(cfg.outdir, "s2_sweep_results.json")
    plot_sweep(snrs, sweep, png)
    json.dump({f"{m}|{s}": sweep[(m, s)] for (m, s) in sweep},
              open(js, "w"), indent=2)
    print(f"[out] {png}\n[out] {js}")
    return sweep


def run(cfg: Config, ofdm: OFDM, n_trials: int, snr_db: float):
    os.makedirs(cfg.outdir, exist_ok=True)
    print(f"[cfg] fc={cfg.fc/1e9} GHz B={cfg.B/1e6} MHz CPI={cfg.cpi_s*1e3:.1f} ms "
          f"N={cfg.N} M={cfg.M}  | OFDM nfft={ofdm.n_fft} cp={ofdm.cp} "
          f"df={cfg.B/ofdm.n_fft/1e3:.1f} kHz | trials={n_trials} snr={snr_db} dB")
    h, gt, drone_tap = trace_once(cfg)
    results, shows = {}, {}
    for mode in MODES:
        res, show = run_mode(cfg, ofdm, mode, h, gt, drone_tap, snr_db,
                             n_trials, base_seed=cfg.seed)
        results[mode] = res
        shows[mode] = show
        print(f"[mode] {mode:18s} knownREs={res['known_fraction']*100:6.2f}%  "
              f"SCR={res['scr_db_mean']:5.1f}±{res['scr_db_std']:.1f} dB  "
              f"PSLR={res['psl_db_mean']:5.1f} dB  "
              f"Pd={res['pd']:.2f}  FAR={res['far']:.1e}  "
              f"peakStd(R,D)=({res['peakR_std']:.1f}m,{res['peakD_std']:.0f}Hz)")

    grid_png = os.path.join(cfg.outdir, "s2_rd_grid.png")
    summ_png = os.path.join(cfg.outdir, "s2_summary.png")
    js = os.path.join(cfg.outdir, "s2_results.json")
    plot_rd_grid(cfg, gt, shows, results, grid_png)
    plot_summary(results, summ_png)
    json.dump(dict(config=asdict(cfg), ofdm=asdict(ofdm), gt=gt,
                   results=results), open(js, "w"), indent=2)
    print(f"[out] {grid_png}\n[out] {summ_png}\n[out] {js}")
    return results


def parse_args():
    cfg = Config()
    default_out = os.environ.get(
        "PR_OUTDIR",
        "/data/public/jeong/sionna/stage2"
        if os.access("/data/public/jeong", os.W_OK)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=default_out)
    p.add_argument("--assets", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets"))
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--snr", type=float, default=-23.0)
    p.add_argument("--N", type=int, default=cfg.N)
    p.add_argument("--M", type=int, default=cfg.M)
    p.add_argument("--nfft", type=int, default=4096)
    p.add_argument("--cp", type=int, default=512)
    p.add_argument("--sweep", default="", help="comma SNRs, e.g. -32,-28,-24,-20,-16")
    a = p.parse_args()
    cfg.outdir, cfg.assets_dir = a.outdir, a.assets
    cfg.N, cfg.M = a.N, a.M
    cfg.input_snr_db = a.snr            # was dropped -> report showed the default
                                       # SNR, not the one actually run (review fix a)
    snrs = [float(x) for x in a.sweep.split(",")] if a.sweep else None
    return cfg, OFDM(n_fft=a.nfft, cp=a.cp), a.trials, a.snr, snrs


if __name__ == "__main__":
    cfg, ofdm, n_trials, snr_db, snrs = parse_args()
    if snrs:
        run_sweep(cfg, ofdm, snrs, n_trials)
    else:
        run(cfg, ofdm, n_trials, snr_db)
