#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 passive-radar drone detection pipeline (Sionna RT 2.0.1)
================================================================

Single-CPI demonstration of the project's S1/S2 pipeline:

    TX (illuminator)  +  surveillance RX  +  moving drone (scatterer)
        -> Sionna RT  ->  time-varying CIR (bulk Doppler from drone velocity)
        -> synth illuminator waveform -> reference / surveillance signals
        -> CAF (cross-ambiguity, batches algorithm) -> Range-Doppler map
        -> zero-Doppler clutter cancellation (MTI) -> 2D CA-CFAR detection

Scope (per PROJECT_CONTEXT sec.6): **bulk translational Doppler only**.
Propeller micro-Doppler (rotating point-scatterer / HERM layer) is a later,
toggleable option layer and is NOT modelled here. No blade mesh.

Geometry / signal mode here is a generic 5G-like illuminator (fc=3.5 GHz,
B=100 MHz). S2-S4 will swap the waveform for Wi-Fi / LTE / 5G reference
structures and sweep the control protocol; this file is the common pipeline.

Run:
    PY=/home/yunjung/workspace/jeong/miniforge3/envs/sionna/bin/python
    CUDA_VISIBLE_DEVICES=0 $PY passive_radar_stage1.py

Storage policy (PROJECT_CONTEXT sec.11): code under /workspace/jeong,
generated results under /data/public/jeong.  NOTE: as of 2026-06-25 the
session user `yunjung` has no write access to those dirs, so outputs default
to a writable staging dir; override with --outdir once permissions are fixed.
"""
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass, field, asdict

import numpy as np

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C0 = 299792458.0  # speed of light [m/s]


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- illuminator / waveform ---
    fc: float = 3.5e9          # carrier [Hz]  (5G n78-like)
    B: float = 100e6           # bandwidth = complex sample rate fs [Hz]
    M: int = 8192              # fast-time samples per slow-time batch
    N: int = 512               # number of slow-time batches (CPI = N*M/fs)
    l_max: int = 200           # max correlation lag / range tap (bistatic)
    input_snr_db: float = -20.0  # per-sample SNR of the *drone* echo vs noise
    waveform: str = "qpsk"     # 'qpsk' (constant modulus) or 'gauss'

    # --- geometry (metres, scene frame) ---
    tx_pos: tuple = (-50.0, 0.0, 20.0)
    rx_pos: tuple = (50.0, 0.0, 10.0)
    drone_pos: tuple = (0.0, 30.0, 40.0)
    drone_vel: tuple = (0.0, 15.0, 0.0)   # constant velocity -> bulk Doppler
    drone_size: float = 0.3               # cube edge [m]

    # --- ray tracing ---
    max_depth: int = 2
    samples_per_src: int = 2_000_000
    seed: int = 1

    # --- CFAR ---
    pfa: float = 1e-5
    guard: tuple = (2, 2)      # (range, doppler) guard cells
    train: tuple = (6, 8)      # (range, doppler) training cells (half-widths)
    doppler_notch_hz: float = 30.0  # |f_D| below this excluded from CFAR (clutter)
    cfar_exclude_notch: bool = True  # exclude the zero-Doppler notch from CFAR TRAINING
                                     # (not just post-mask) so the clutter ridge does
                                     # not inflate the noise estimate (review fix j)

    # --- detection scoring (review fix #2) ---
    hit_tol_m: float = 30.0    # Pd hit window: |R - R_gt| <= hit_tol_m (METRES, NOT
                               # range bins). Bin width c/B varies ~20x across B, so a
                               # +-1-bin window biased Pd toward narrow band; fix to a
                               # fixed physical window. Doppler window stays +-bins
                               # (Doppler res 1/CPI is fixed across the matrix).

    # --- bookkeeping ---
    outdir: str = ""
    assets_dir: str = ""
    tag: str = "stage1_5g_like"

    @property
    def fs(self) -> float:
        return self.B

    @property
    def prf(self) -> float:
        return self.B / self.M

    @property
    def cpi_s(self) -> float:
        return self.N * self.M / self.fs

    @property
    def wavelength(self) -> float:
        return C0 / self.fc

    @property
    def range_res_m(self) -> float:        # bistatic range resolution
        return C0 / self.B

    @property
    def doppler_res_hz(self) -> float:
        return 1.0 / self.cpi_s


# --------------------------------------------------------------------------- #
#  Scene construction
# --------------------------------------------------------------------------- #
_CUBE_OBJ = """# unit cube centred at origin
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 2 3
f 1 3 4
f 5 7 6
f 5 8 7
f 1 6 2
f 1 5 6
f 2 7 3
f 2 6 7
f 3 8 4
f 3 7 8
f 4 5 1
f 4 8 5
"""

_GROUND_OBJ = """# 120x120 m ground quad at z=0
v -60 -60 0
v  60 -60 0
v  60  60 0
v -60  60 0
f 1 2 3
f 1 3 4
"""


def _write_assets(assets_dir: str):
    os.makedirs(assets_dir, exist_ok=True)
    cube = os.path.join(assets_dir, "cube.obj")
    ground = os.path.join(assets_dir, "ground.obj")
    if not os.path.exists(cube):
        open(cube, "w").write(_CUBE_OBJ)
    if not os.path.exists(ground):
        open(ground, "w").write(_GROUND_OBJ)
    return cube, ground


def build_scene(cfg: Config):
    """Empty scene + ground plane (clutter) + moving drone + TX/RX."""
    import sionna.rt as rt
    import mitsuba as mi

    cube_obj, ground_obj = _write_assets(cfg.assets_dir)

    scene = rt.load_scene()
    scene.frequency = cfg.fc

    mat_g = rt.ITURadioMaterial(name="ground_mat", itu_type="concrete",
                                thickness=0.3)
    mat_d = rt.ITURadioMaterial(name="drone_mat", itu_type="metal",
                                thickness=0.01, scattering_coefficient=0.6)

    ground = rt.SceneObject(fname=ground_obj, name="ground", radio_material=mat_g)
    drone = rt.SceneObject(fname=cube_obj, name="drone", radio_material=mat_d)
    scene.edit(add=[ground, drone])

    # geometry must be set AFTER the object is part of the scene
    drone.scaling = cfg.drone_size
    drone.position = mi.Point3f(*cfg.drone_pos)
    drone.velocity = mi.Vector3f(*cfg.drone_vel)

    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                    polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                    polarization="V")
    scene.add(rt.Transmitter("tx", position=mi.Point3f(*cfg.tx_pos)))
    scene.add(rt.Receiver("rx", position=mi.Point3f(*cfg.rx_pos)))
    return scene


def trace_channel(cfg: Config, scene):
    """Run PathSolver once; return discrete baseband taps h[N, L] (slow x fast)
    and ground-truth (bistatic delay, Doppler) of the drone path."""
    import sionna.rt as rt

    solver = rt.PathSolver()
    paths = solver(scene, max_depth=cfg.max_depth, los=True,
                   specular_reflection=True, diffuse_reflection=True,
                   refraction=False, samples_per_src=cfg.samples_per_src,
                   seed=cfg.seed)

    tau = np.asarray(paths.tau).squeeze().ravel().astype(np.float64)
    dop = np.asarray(paths.doppler).squeeze().ravel().astype(np.float64)
    valid = np.isfinite(tau) & (tau > 0)
    tau_v, dop_v = tau[valid], dop[valid]
    tau_direct = float(tau_v.min())
    # drone path == the path with the largest |Doppler| (bulk-moving scatterer)
    d_idx = int(np.argmax(np.abs(dop_v)))
    gt = dict(
        tau_direct=tau_direct,
        tau_drone=float(tau_v[d_idx]),
        bistatic_delay=float(tau_v[d_idx] - tau_direct),
        bistatic_range_m=float((tau_v[d_idx] - tau_direct) * C0),
        doppler_hz=float(dop_v[d_idx]),
        n_paths=int(valid.sum()),
        tau_all_ns=(tau_v * 1e9).round(2).tolist(),
        dop_all_hz=dop_v.round(2).tolist(),
    )

    # Discrete baseband taps, normalised so the LOS/direct path sits at tap 0.
    taps = paths.taps(bandwidth=cfg.B, l_min=0, l_max=cfg.l_max,
                      sampling_frequency=cfg.prf, num_time_steps=cfg.N,
                      normalize_delays=True, out_type="numpy")
    taps = np.asarray(taps)                      # (rx,rxa,tx,txa,N,L)
    h = taps[0, 0, 0, 0, :, :].astype(np.complex64)   # (N, L)
    return h, gt


# --------------------------------------------------------------------------- #
#  Passive-radar signal processing
# --------------------------------------------------------------------------- #
def synth_waveform(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Unit-power complex baseband illuminator, length K = N*M."""
    K = cfg.N * cfg.M
    if cfg.waveform == "gauss":
        s = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
    else:  # constant-modulus random QPSK -> ~thumbtack ambiguity
        bits = rng.integers(0, 4, size=K)
        s = np.exp(1j * (np.pi / 4 + bits * np.pi / 2))
    return s.astype(np.complex64)


def _conv_batches(s: np.ndarray, h: np.ndarray, N: int, M: int) -> np.ndarray:
    """Batch-by-batch causal convolution of waveform s with time-varying
    channel h[n,:] (history carried across batch boundaries). -> [N, M]."""
    L = h.shape[1]
    sp = np.concatenate([np.zeros(L - 1, np.complex64), s])
    out = np.empty((N, M), np.complex64)
    for n in range(N):
        a = sp[n * M: n * M + M + (L - 1)]            # length M + L-1
        out[n] = np.convolve(a, h[n], mode="valid")   # length M
    return out


def apply_channel(cfg: Config, s: np.ndarray, h: np.ndarray,
                  gt: dict, rng: np.random.Generator):
    """Build surveillance signals and reference.

    Returns
    -------
    X        : [N,M] surveillance = waveform * full channel + receiver noise
    X_clean  : [N,M] after ideal static-clutter cancellation (direct + ground
               response removed in the *signal* domain -> also kills the DPI
               range-sidelobe pedestal).  Proxy for ECA/CLEAN; the data-driven
               version is an S2+ task.
    Rmat     : [N,M] reference = clean illuminator copy (perfect ref channel).
    """
    N, M, L = cfg.N, cfg.M, h.shape[1]
    Xmat = _conv_batches(s, h, N, M)                 # full channel response
    Rmat = s.reshape(N, M).copy()

    # receiver thermal noise referenced to the drone echo's per-sample power
    drone_tap = int(round(gt["bistatic_delay"] * cfg.fs))
    drone_tap = max(0, min(L - 1, drone_tap))
    p_drone = float(np.mean(np.abs(h[:, drone_tap]) ** 2))      # E|s|^2 = 1
    noise_pow = p_drone / (10 ** (cfg.input_snr_db / 10.0))
    sigma = np.sqrt(noise_pow / 2.0)
    Xmat = Xmat + (sigma * (rng.standard_normal((N, M))
                            + 1j * rng.standard_normal((N, M)))).astype(np.complex64)

    # ideal clutter cancellation: subtract the static (zero-Doppler) channel's
    # signal contribution.  h_static = slow-time mean channel ~ direct+ground.
    h_static = h.mean(axis=0, keepdims=True).repeat(N, axis=0).astype(np.complex64)
    X_static = _conv_batches(s, h_static, N, M)      # noiseless clutter signal
    X_clean = Xmat - X_static                        # drone + noise (+residual)
    return Xmat, X_clean, Rmat, drone_tap, noise_pow


def caf_range_doppler(cfg: Config, Xmat: np.ndarray, Rmat: np.ndarray,
                      mti: bool = True):
    """Cross-ambiguity by batches:
        per-batch fast-time cross-correlation (range) -> slow-time FFT (Doppler).
    Returns complex RD map [range(l=0..l_max), doppler] and the Doppler axis."""
    N, M = cfg.N, cfg.M
    Lmax = cfg.l_max
    nfft = int(2 ** np.ceil(np.log2(M + Lmax)))

    Xf = np.fft.fft(Xmat, n=nfft, axis=1)
    Rf = np.fft.fft(Rmat, n=nfft, axis=1)
    cc = np.fft.ifft(Xf * np.conj(Rf), axis=1)        # circular -> linear (pad)
    c = cc[:, : Lmax + 1]                              # [N, range], lags 0..Lmax

    if mti:                                            # zero-Doppler canceller
        c = c - c.mean(axis=0, keepdims=True)

    # Hann taper; use the (N+2)[1:-1] form so the endpoints are NOT exactly zero
    # (np.hanning(N) zeros sample 0 and N-1 -> throws away 2 of N slow-time samples,
    # which hurts the small-N cells of the fixed-CPI matrix; review fix j).
    win = np.hanning(N + 2)[1:-1][:, None]
    rd = np.fft.fftshift(np.fft.fft(c * win, axis=0), axes=0)   # [doppler, range]
    rd = rd.T                                          # [range, doppler]

    dopp_axis = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / cfg.prf))
    range_axis = np.arange(Lmax + 1) * C0 / cfg.fs     # bistatic extra range [m]
    return rd, range_axis, dopp_axis


def ca_cfar_2d(power: np.ndarray, cfg: Config, dopp_axis: np.ndarray | None = None):
    """2-D cell-averaging CFAR on a power map [range, doppler].

    If cfg.cfar_exclude_notch and dopp_axis is given, the zero-Doppler clutter
    notch is removed from the TRAINING statistic (its huge ridge is replaced, per
    range row, by the median of the non-notch cells) before the box filter, so it
    does not inflate the noise estimate of neighbouring Doppler cells (review fix
    j). Detections are still reported on the full map (callers post-mask the
    notch). With no dopp_axis the behaviour is the original whole-map CFAR."""
    from scipy.ndimage import uniform_filter

    gr, gd = cfg.guard
    tr, td = cfg.train
    win_r, win_d = 2 * (tr + gr) + 1, 2 * (td + gd) + 1
    grd_r, grd_d = 2 * gr + 1, 2 * gd + 1

    train_pow = power
    if cfg.cfar_exclude_notch and dopp_axis is not None:
        notch = np.abs(dopp_axis) < cfg.doppler_notch_hz
        if notch.any() and (~notch).any():
            train_pow = power.copy()
            row_med = np.median(power[:, ~notch], axis=1, keepdims=True)
            train_pow[:, notch] = row_med            # neutralise the clutter ridge

    box_sum = uniform_filter(train_pow, size=(win_r, win_d), mode="reflect") * (win_r * win_d)
    grd_sum = uniform_filter(train_pow, size=(grd_r, grd_d), mode="reflect") * (grd_r * grd_d)
    n_train = win_r * win_d - grd_r * grd_d
    noise = (box_sum - grd_sum) / n_train

    alpha = n_train * (cfg.pfa ** (-1.0 / n_train) - 1.0)   # CA-CFAR scaling
    thr = alpha * noise
    det = power > thr                                # tested on the TRUE power map
    return det, thr, alpha, n_train


# --------------------------------------------------------------------------- #
#  Plot + save
# --------------------------------------------------------------------------- #
def _db(x, ref=None):
    x = np.abs(x) ** 2
    ref = x.max() if ref is None else ref
    return 10 * np.log10(np.maximum(x, ref * 1e-12) / ref)


def make_figure(cfg, rd_raw, rd_clean, range_axis, dopp_axis, gt, det, drone_tap):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    extent = [dopp_axis[0], dopp_axis[-1], range_axis[0], range_axis[-1]]

    # common reference = direct-path (global raw) peak -> honest relative scale
    ref = (np.abs(rd_raw) ** 2).max()
    gR, gD = gt["bistatic_range_m"], gt["doppler_hz"]

    # left: raw CAF, full extent, direct-path-referenced
    im0 = axes[0].imshow(_db(rd_raw, ref), aspect="auto", origin="lower",
                         extent=extent, cmap="viridis", vmin=-70, vmax=0)
    axes[0].set_title("Raw CAF  (zero-Doppler direct path / clutter dominates)")
    axes[0].set_xlabel("Bistatic Doppler [Hz]"); axes[0].set_ylabel("Bistatic range [m]")
    axes[0].plot(gD, gR, "o", mfc="none", mec="red", ms=13, mew=1.6)
    fig.colorbar(im0, ax=axes[0], label="dB re direct-path peak")

    # right: clutter-cancelled CAF, zoomed to ROI, own-max normalised
    im1 = axes[1].imshow(_db(rd_clean), aspect="auto", origin="lower",
                         extent=extent, cmap="viridis", vmin=-40, vmax=0)
    axes[1].set_title("Clutter-cancelled CAF + CA-CFAR  (zoom on target)")
    axes[1].set_xlabel("Bistatic Doppler [Hz]"); axes[1].set_ylabel("Bistatic range [m]")
    r_hi = float(min(range_axis[-1], max(60.0, 3 * gR)))
    d_lim = float(min(dopp_axis[-1], max(600.0, 6 * abs(gD))))
    axes[1].set_xlim(-d_lim, d_lim); axes[1].set_ylim(0, r_hi)
    rr, dd = np.where(det)
    if rr.size:
        axes[1].plot(dopp_axis[dd], range_axis[rr], "x", color="red", ms=7,
                     mew=1.4, label="CFAR det")
    axes[1].plot(gD, gR, "o", mfc="none", mec="red", ms=15, mew=1.8, label="GT drone")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(im1, ax=axes[1], label="dB (norm.)")

    fig.suptitle(
        f"Sionna passive-radar Stage-1 (bulk Doppler)  |  fc={cfg.fc/1e9:.2f} GHz, "
        f"B={cfg.B/1e6:.0f} MHz, CPI={cfg.cpi_s*1e3:.1f} ms (N={cfg.N}, M={cfg.M})  |  "
        f"GT drone: R={gR:.1f} m, fD={gD:.1f} Hz, input SNR/samp={cfg.input_snr_db:.0f} dB",
        fontsize=11)
    return fig


def run(cfg: Config):
    os.makedirs(cfg.outdir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    print(f"[cfg] fc={cfg.fc/1e9} GHz  B={cfg.B/1e6} MHz  fs={cfg.fs/1e6} MHz")
    print(f"[cfg] M={cfg.M} N={cfg.N}  PRF={cfg.prf:.1f} Hz  CPI={cfg.cpi_s*1e3:.2f} ms")
    print(f"[cfg] range_res={cfg.range_res_m:.2f} m  doppler_res={cfg.doppler_res_hz:.2f} Hz"
          f"  unambig_dopp=+/-{cfg.prf/2:.0f} Hz")

    print("[rt] building scene + tracing channel ...")
    scene = build_scene(cfg)
    h, gt = trace_channel(cfg, scene)
    print(f"[rt] paths={gt['n_paths']}  tau(ns)={gt['tau_all_ns']}  dop(Hz)={gt['dop_all_hz']}")
    print(f"[rt] GT drone: bistatic R={gt['bistatic_range_m']:.2f} m "
          f"(tau={gt['bistatic_delay']*1e9:.1f} ns, tap~{gt['bistatic_delay']*cfg.fs:.2f}), "
          f"fD={gt['doppler_hz']:.2f} Hz")

    print("[dsp] synth waveform + channel + clutter-cancel + CAF ...")
    s = synth_waveform(cfg, rng)
    Xmat, X_clean, Rmat, drone_tap, noise_pow = apply_channel(cfg, s, h, gt, rng)
    rd_raw, range_axis, dopp_axis = caf_range_doppler(cfg, Xmat, Rmat, mti=False)
    rd_clean, _, _ = caf_range_doppler(cfg, X_clean, Rmat, mti=False)

    # --- CFAR on the clutter-cancelled power map, excluding zero-Doppler notch ---
    power = np.abs(rd_clean) ** 2
    notch = np.abs(dopp_axis) < cfg.doppler_notch_hz
    det, thr, alpha, n_train = ca_cfar_2d(power, cfg, dopp_axis)
    det[:, notch] = False                      # ignore residual clutter ridge

    # --- verification: does the global peak / a detection land on the GT cell? ---
    ri = int(np.argmin(np.abs(range_axis - gt["bistatic_range_m"])))
    di = int(np.argmin(np.abs(dopp_axis - gt["doppler_hz"])))
    pmask = np.ones_like(power, bool); pmask[:, notch] = False
    pk = np.unravel_index(np.argmax(np.where(pmask, power, -np.inf)), power.shape)
    peak_R, peak_fD = range_axis[pk[0]], dopp_axis[pk[1]]
    # SCR/SNR of the GT cell vs the background median (outside target+clutter)
    bg = power[:, ~notch].copy()
    snr_db = 10 * np.log10(power[ri, di] / np.median(bg))
    peak_ok = (abs(peak_R - gt["bistatic_range_m"]) < 2 * cfg.range_res_m
               and abs(peak_fD - gt["doppler_hz"]) < 2 * cfg.doppler_res_hz)
    det_near_gt = det[max(0, ri-2):ri+3, max(0, di-2):di+3].any()

    print(f"[ver] global peak : R={peak_R:.2f} m, fD={peak_fD:.2f} Hz")
    print(f"[ver] GT cell     : R={gt['bistatic_range_m']:.2f} m, fD={gt['doppler_hz']:.2f} Hz "
          f"(bin r={ri}, d={di})  SNR~{snr_db:.1f} dB")
    print(f"[ver] global peak == GT target : {peak_ok}")
    print(f"[ver] CFAR detections={int(det.sum())}  detection on GT target: {bool(det_near_gt)}")

    # --- save figure + arrays ---
    fig = make_figure(cfg, rd_raw, rd_clean, range_axis, dopp_axis, gt, det, drone_tap)
    png = os.path.join(cfg.outdir, f"rd_map_{cfg.tag}.png")
    npz = os.path.join(cfg.outdir, f"rd_map_{cfg.tag}.npz")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    np.savez_compressed(
        npz, rd_raw=rd_raw, rd_clean=rd_clean, det=det,
        range_axis=range_axis, dopp_axis=dopp_axis,
        config=np.array(str(asdict(cfg))), gt=np.array(str(gt)),
        peak=np.array([peak_R, peak_fD, snr_db]))
    print(f"[out] {png}")
    print(f"[out] {npz}")
    return dict(png=png, npz=npz, gt=gt, peak=(peak_R, peak_fD, snr_db),
                cfar=int(det.sum()), det_on_target=bool(det_near_gt),
                peak_ok=bool(peak_ok))


# --------------------------------------------------------------------------- #
def parse_args() -> Config:
    cfg = Config()
    default_out = os.environ.get(
        "PR_OUTDIR",
        "/data/public/jeong/sionna/stage1"
        if os.access("/data/public/jeong", os.W_OK)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    default_assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

    p = argparse.ArgumentParser(description="Sionna passive-radar Stage-1 RD map")
    p.add_argument("--outdir", default=default_out)
    p.add_argument("--assets", default=default_assets)
    p.add_argument("--fc", type=float, default=cfg.fc)
    p.add_argument("--B", type=float, default=cfg.B)
    p.add_argument("--M", type=int, default=cfg.M)
    p.add_argument("--N", type=int, default=cfg.N)
    p.add_argument("--snr", type=float, default=cfg.input_snr_db)
    p.add_argument("--waveform", default=cfg.waveform, choices=["qpsk", "gauss"])
    p.add_argument("--tag", default=cfg.tag)
    a = p.parse_args()

    cfg.outdir, cfg.assets_dir = a.outdir, a.assets
    cfg.fc, cfg.B, cfg.M, cfg.N = a.fc, a.B, a.M, a.N
    cfg.input_snr_db, cfg.waveform, cfg.tag = a.snr, a.waveform, a.tag
    return cfg


if __name__ == "__main__":
    run(parse_args())
