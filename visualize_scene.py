#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment-geometry visualizer for the Sionna passive-radar benchmark.
======================================================================

Answers the question "what does the bistatic / chamber-like setup actually
LOOK like, and how does geometry turn into the range-Doppler cell?"

Two complementary views, both headless (-> PNG, embedded in report.ipynb):

  (A) Sionna RT scene RENDER  — the real ray-traced picture: TX (illuminator),
      RX (surveillance receiver), the drone target, the ground plane, and the
      actual paths (direct TX->RX = the reference/DPI, and TX->drone->RX = the
      bistatic echo). This is "the experiment as Sionna sees it".

  (B) Bistatic-geometry SCHEMATIC (matplotlib) — the textbook picture that the
      render is an instance of: baseline L, the two legs R_tx / R_rx, the
      bistatic angle beta, the iso-range ellipse (constant bistatic range =
      confocal ellipse with TX & RX as foci), and how each motion scenario's
      velocity projects onto the bistatic Doppler direction (why radial detects
      and hover / tangential are blind).

NOTE on "chamber-like": the current scene is NOT a walled anechoic chamber.
It is an OPEN-FIELD bistatic geometry — empty scene + a 120x120 m concrete
ground patch (the only clutter) + a metal-cube drone proxy + two isotropic,
vertically-polarized antennas. The render makes that explicit.

The drone marker is OPTIONALLY enlarged for the render only (the real target is
a 0.3 m cube, invisible at a 120 m scale); the bistatic delay/Doppler ground
truth is unaffected (it is geometry, computed analytically). Clearly labelled.

Run:
    PY=/home/yunjung/workspace/jeong/miniforge3/envs/sionna/bin/python
    CUDA_VISIBLE_DEVICES=0 $PY visualize_scene.py --outdir outputs
"""
from __future__ import annotations
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from passive_radar_stage1 import Config, C0
from phase1 import GEOM, SCEN_SPEED, scenario_velocity, _unit

# carriers to annotate the Doppler with (one representative per standard)
CARRIERS = [("Wi-Fi 2.4 GHz", 2.4e9), ("LTE 1.8 GHz", 1.8e9), ("5G 3.5 GHz", 3.5e9)]


# --------------------------------------------------------------------------- #
#  geometry helpers (single source of truth = phase1.GEOM)
# --------------------------------------------------------------------------- #
def geom_arrays():
    tx = np.array(GEOM["tx"], float)
    rx = np.array(GEOM["rx"], float)
    p = np.array(GEOM["drone"], float)
    return tx, rx, p


def bistatic_quantities():
    tx, rx, p = geom_arrays()
    Rtx = np.linalg.norm(p - tx)
    Rrx = np.linalg.norm(p - rx)
    L = np.linalg.norm(tx - rx)
    Rb = Rtx + Rrx - L                                   # bistatic (extra) range
    # bistatic angle beta = angle TX-drone-RX
    u_dt = _unit(tx - p); u_dr = _unit(rx - p)
    beta = np.degrees(np.arccos(np.clip(u_dt @ u_dr, -1, 1)))
    grad = _unit(p - tx) + _unit(p - rx)                 # bistatic range gradient
    return dict(tx=tx, rx=rx, p=p, Rtx=Rtx, Rrx=Rrx, L=L, Rb=Rb,
                beta=beta, grad=grad, gnorm=float(np.linalg.norm(grad)))


def doppler_for(velocity, fc):
    """Bistatic Doppler [Hz] for a velocity vector at the drone (phase1 sign)."""
    bq = bistatic_quantities()
    lam = C0 / fc
    return -float(np.asarray(velocity, float) @ bq["grad"]) / lam


# --------------------------------------------------------------------------- #
#  (A) Sionna RT scene render
# --------------------------------------------------------------------------- #
def build_viz_scene(cfg: Config, drone_marker_m: float, ground_half: float):
    """Faithful geometry, but with an enlarged drone marker + larger ground so
    the render is legible. Geometry/positions match phase1.GEOM exactly."""
    import sionna.rt as rt
    import mitsuba as mi
    from passive_radar_stage1 import _CUBE_OBJ

    # write a bigger ground for the render (does not affect any physics run)
    os.makedirs(cfg.assets_dir, exist_ok=True)
    g = os.path.join(cfg.assets_dir, f"ground_viz_{int(ground_half)}.obj")
    if not os.path.exists(g):
        h = ground_half
        open(g, "w").write(
            f"# {2*h:.0f}x{2*h:.0f} m ground quad at z=0\n"
            f"v {-h} {-h} 0\nv {h} {-h} 0\nv {h} {h} 0\nv {-h} {h} 0\n"
            "f 1 2 3\nf 1 3 4\n")
    cube = os.path.join(cfg.assets_dir, "cube.obj")
    if not os.path.exists(cube):
        open(cube, "w").write(_CUBE_OBJ)

    scene = rt.load_scene()
    scene.frequency = cfg.fc
    mat_g = rt.ITURadioMaterial(name="ground_mat", itu_type="concrete", thickness=0.3)
    mat_d = rt.ITURadioMaterial(name="drone_mat", itu_type="metal", thickness=0.01,
                                scattering_coefficient=0.6)
    ground = rt.SceneObject(fname=g, name="ground", radio_material=mat_g)
    drone = rt.SceneObject(fname=cube, name="drone", radio_material=mat_d)
    scene.edit(add=[ground, drone])
    drone.scaling = drone_marker_m
    drone.position = mi.Point3f(*cfg.drone_pos)
    drone.velocity = mi.Vector3f(*cfg.drone_vel)
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx", position=mi.Point3f(*cfg.tx_pos)))
    scene.add(rt.Receiver("rx", position=mi.Point3f(*cfg.rx_pos)))
    return scene


def trace_polylines(fc, scenario="radial"):
    """Trace the scene and return the REAL Sionna ray paths as labelled polylines
    (TX -> interaction vertices -> RX), classified into direct / ground / drone."""
    import sionna.rt as rt
    cfg = Config(); cfg.fc = float(fc)
    cfg.assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    cfg.tx_pos, cfg.rx_pos, cfg.drone_pos = GEOM["tx"], GEOM["rx"], GEOM["drone"]
    cfg.drone_vel = tuple(float(x) for x in np.atleast_2d(scenario_velocity(scenario))[0])
    cfg.drone_size = 0.3
    from passive_radar_stage1 import build_scene
    scene = build_scene(cfg)
    paths = rt.PathSolver()(scene, max_depth=2, los=True, specular_reflection=True,
                            diffuse_reflection=True, refraction=False,
                            samples_per_src=2_000_000, seed=1)
    V = np.asarray(paths.vertices)            # (depth,rx,tx,npath,3)
    inter = np.asarray(paths.interactions)    # (depth,rx,tx,npath)
    tau = np.asarray(paths.tau).reshape(-1)
    dop = np.asarray(paths.doppler).reshape(-1)
    valid = np.asarray(paths.valid).reshape(-1)
    tx = np.array(GEOM["tx"]); rx = np.array(GEOM["rx"]); dr = np.array(GEOM["drone"])
    t0 = float(np.nanmin(tau[valid]))
    out = []
    for k in range(V.shape[3]):
        if not valid[k]:
            continue
        verts = [V[d, 0, 0, k] for d in range(V.shape[0]) if int(inter[d, 0, 0, k]) != 0]
        pts = np.array([tx] + verts + [rx])
        Rb = (tau[k] - t0) * C0
        if not verts:
            kind, col, lbl = "direct", "0.45", "direct path TX→RX (reference / DPI, clutter)"
        elif np.linalg.norm(np.asarray(verts[0]) - dr) < 5.0:
            kind, col = "drone", "tab:green"
            lbl = f"drone echo TX→drone→RX  (R_b={Rb:.0f} m, fD={dop[k]:+.0f} Hz)"
        else:
            kind, col, lbl = "ground", "tab:brown", f"ground bounce (clutter, R_b={Rb:.0f} m)"
        out.append(dict(kind=kind, color=col, label=lbl, pts=pts,
                        Rb=float(Rb), dop=float(dop[k])))
    return tx, rx, dr, out


def make_rt_paths_figure(outdir, fc=3.5e9):
    """Labelled 3D plot of the REAL Sionna-traced rays — the experiment as the RT
    engine sees it, but legible (devices + ground + colour-coded paths)."""
    tx, rx, dr, polylines = trace_polylines(fc)
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    g = 130
    xx, yy = np.meshgrid(np.linspace(-g, g, 2), np.linspace(-20, 150, 2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.10, color="gray")
    # devices + drone
    ax.scatter(*tx, c="tab:red", s=130, marker="^", depthshade=False, zorder=5)
    ax.scatter(*rx, c="tab:blue", s=130, marker="v", depthshade=False, zorder=5)
    ax.scatter(*dr, c="k", s=90, marker="o", depthshade=False, zorder=5)
    ax.text(*tx, "  TX (illuminator: gNB/AP)", color="tab:red", fontsize=10, weight="bold")
    ax.text(*rx, "  RX (surveillance receiver)", color="tab:blue", fontsize=10, weight="bold")
    ax.text(*dr, "  drone (moving scatterer)", color="k", fontsize=10, weight="bold")
    seen = set()
    for pl in polylines:
        lab = pl["label"] if pl["kind"] not in seen else None
        seen.add(pl["kind"])
        ls = "--" if pl["kind"] == "direct" else "-"
        ax.plot(pl["pts"][:, 0], pl["pts"][:, 1], pl["pts"][:, 2],
                color=pl["color"], lw=2.4, ls=ls, label=lab, zorder=4)
    # velocity arrow (radial) at the drone
    v = np.array(scenario_velocity("radial"), float) * 2.4
    ax.plot([dr[0], dr[0]+v[0]], [dr[1], dr[1]+v[1]], [dr[2], dr[2]+v[2]],
            color="tab:green", lw=2.0)
    ax.text(*(dr + v), " v (radial)", color="tab:green", fontsize=9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_zlim(0, 90); ax.view_init(elev=22, azim=-68)
    ax.set_box_aspect((1, 1.1, 0.45))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("Sionna RT — actual traced paths of the bistatic passive-radar experiment\n"
                 f"(fc = {fc/1e9:.1f} GHz; open-field: empty scene + concrete ground + metal-cube drone)",
                 fontsize=11)
    fn = os.path.join(outdir, "viz_rt_paths.png")
    fig.savefig(fn, dpi=140); plt.close(fig)
    print("[rt-paths]", fn)
    return fn


def render_views(outdir, drone_marker_m=4.0, ground_half=140.0, num_samples=256):
    import sionna.rt as rt
    cfg = Config()
    cfg.fc = 3.5e9
    cfg.assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    cfg.tx_pos, cfg.rx_pos, cfg.drone_pos = GEOM["tx"], GEOM["rx"], GEOM["drone"]
    cfg.drone_vel = tuple(float(x) for x in scenario_velocity("radial"))
    scene = build_viz_scene(cfg, drone_marker_m, ground_half)

    paths = rt.PathSolver()(scene, max_depth=2, los=True, specular_reflection=True,
                            diffuse_reflection=True, refraction=False,
                            samples_per_src=2_000_000, seed=1)

    mx = float((GEOM["tx"][0] + GEOM["rx"][0]) / 2)
    cams = {
        "viz_scene_wide": rt.Camera(position=[mx - 70.0, -230.0, 150.0],
                                    look_at=[mx, 70.0, 35.0]),
        "viz_scene_top":  rt.Camera(position=[mx + 1.0, 55.0, 360.0],
                                    look_at=[mx, 55.0, 0.0]),
        "viz_scene_side": rt.Camera(position=[280.0, 60.0, 70.0],
                                    look_at=[0.0, 60.0, 40.0]),
    }
    saved = []
    for name, cam in cams.items():
        fn = os.path.join(outdir, name + ".png")
        scene.render_to_file(camera=cam, filename=fn, paths=paths,
                             resolution=(1280, 960), num_samples=num_samples,
                             show_devices=True)
        saved.append(fn); print("[render]", fn)
    return saved


# --------------------------------------------------------------------------- #
#  (B) bistatic-geometry schematic
# --------------------------------------------------------------------------- #
def _arrow3d(ax, a, b, **kw):
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], **kw)


def schematic_3d(ax, bq):
    tx, rx, p = bq["tx"], bq["rx"], bq["p"]
    # ground patch
    g = 130
    xx, yy = np.meshgrid([-g, g], [-20, 150])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.12, color="gray")
    # devices
    ax.scatter(*tx, c="tab:red", s=70, marker="^", depthshade=False)
    ax.scatter(*rx, c="tab:blue", s=70, marker="v", depthshade=False)
    ax.scatter(*p, c="k", s=60, marker="o", depthshade=False)
    ax.text(*tx, "  TX (illuminator)", color="tab:red", fontsize=9)
    ax.text(*rx, "  RX (surveillance)", color="tab:blue", fontsize=9)
    ax.text(*p, "  drone", color="k", fontsize=9)
    # bistatic triangle: direct baseline + two legs
    _arrow3d(ax, tx, rx, color="0.4", ls="--", lw=1.6)      # baseline L (direct path)
    _arrow3d(ax, tx, p, color="tab:red", lw=1.8)            # R_tx
    _arrow3d(ax, p, rx, color="tab:blue", lw=1.8)           # R_rx
    ax.text(*((tx + rx) / 2 + [0, -8, 3]), f"L={bq['L']:.0f} m\n(direct path)",
            color="0.3", fontsize=8, ha="center")
    ax.text(*((tx + p) / 2), f" R_tx={bq['Rtx']:.0f}", color="tab:red", fontsize=8)
    ax.text(*((rx + p) / 2), f" R_rx={bq['Rrx']:.0f}", color="tab:blue", fontsize=8)
    # velocity arrows (radial vs tangential), scaled for visibility
    s = 3.0
    for sc, col in (("radial", "tab:green"), ("tangential", "tab:purple")):
        v = np.array(scenario_velocity(sc), float)
        _arrow3d(ax, p, p + s * v, color=col, lw=2.2)
        ax.text(*(p + s * v), f" {sc}", color=col, fontsize=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"Bistatic geometry (open-field)\nbistatic range R_b = R_tx+R_rx-L "
                 f"= {bq['Rb']:.1f} m,  bistatic angle β = {bq['beta']:.0f}°", fontsize=9)
    ax.view_init(elev=24, azim=-62)
    ax.set_box_aspect((1, 1, 0.5))


def schematic_ellipse(ax, bq):
    """Iso-range ellipse in the TX-RX-drone plane: foci TX,RX; the drone sits on
    the constant-bistatic-range contour R_tx+R_rx = R_b+L (a 'range cell' is the
    gap between two such confocal ellipses)."""
    tx, rx, p = bq["tx"], bq["rx"], bq["p"]
    # build an orthonormal 2D frame in the plane of (tx, rx, p)
    e1 = _unit(rx - tx)
    n = np.cross(rx - tx, p - tx); n = _unit(n)
    e2 = _unit(np.cross(n, e1))
    o = (tx + rx) / 2

    def to2d(q):
        d = np.asarray(q) - o
        return np.array([d @ e1, d @ e2])
    tx2, rx2, p2 = to2d(tx), to2d(rx), to2d(p)
    c = bq["L"] / 2                                   # focal half-distance
    for k, (rb, style) in enumerate([(bq["Rb"], "-"), (bq["Rb"] - 30, ":"),
                                     (bq["Rb"] + 30, ":")]):
        a = (rb + bq["L"]) / 2                        # semi-major (R_tx+R_rx=2a)
        b = np.sqrt(max(a * a - c * c, 1e-6))         # semi-minor
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(a * np.cos(th), b * np.sin(th), style, color="tab:green",
                lw=1.8 if k == 0 else 1.0,
                label=("iso-bistatic-range\n(R_b=%.0f m)" % bq["Rb"]) if k == 0
                else ("±30 m range cells" if k == 1 else None))
    ax.plot(*tx2, "^", color="tab:red", ms=11); ax.text(*tx2, " TX", color="tab:red")
    ax.plot(*rx2, "v", color="tab:blue", ms=11); ax.text(*rx2, " RX", color="tab:blue")
    ax.plot(*p2, "o", color="k", ms=8); ax.text(*p2, "  drone")
    ax.plot([tx2[0], p2[0]], [tx2[1], p2[1]], color="tab:red", lw=1.3)
    ax.plot([rx2[0], p2[0]], [rx2[1], p2[1]], color="tab:blue", lw=1.3)
    ax.plot([tx2[0], rx2[0]], [tx2[1], rx2[1]], "--", color="0.4", lw=1.3)
    # Doppler-sensitive direction (grad) and tangential at the drone
    g2 = np.array([bq["grad"] @ e1, bq["grad"] @ e2]); g2 = _unit(g2) * 38
    t2 = np.array([-g2[1], g2[0]])
    ax.annotate("", p2 + g2, p2, arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2))
    ax.annotate("", p2 + t2, p2, arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=2))
    ax.text(*(p2 + g2), " radial\n(perp. to ellipse\n -> max Doppler)", color="tab:green", fontsize=8)
    ax.text(*(p2 + t2), " tangential\n(along ellipse\n -> ~0 Doppler)", color="tab:purple", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower left")
    ax.set_xlabel("in-plane x [m]"); ax.set_ylabel("in-plane y [m]")
    ax.set_title("Iso-range ellipse & Doppler direction\n"
                 "(motion across ellipse -> detectable; along ellipse -> blind)", fontsize=9)


def schematic_doppler_bars(ax):
    """Bistatic Doppler per scenario x carrier — why hover/tangential are blind."""
    scens = ["hover", "radial", "tangential", "waypoint"]
    width = 0.25
    x = np.arange(len(scens))
    for i, (lab, fc) in enumerate(CARRIERS):
        vals = []
        for sc in scens:
            v = scenario_velocity(sc)
            if isinstance(v, list):                  # waypoint -> max-|fD| segment
                vals.append(max((doppler_for(seg, fc) for _, seg in v), key=abs))
            else:
                vals.append(doppler_for(v, fc))
        ax.bar(x + (i - 1) * width, np.abs(vals), width, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(scens)
    ax.set_ylabel("|bistatic Doppler|  [Hz]")
    ax.set_title(f"Geometry → Doppler  (speed = {SCEN_SPEED:.0f} m/s)\n"
                 "hover & tangential ≈ 0 Hz → fall in the clutter notch (blind)",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")


def make_schematic(outdir):
    bq = bistatic_quantities()
    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    schematic_3d(ax1, bq)
    schematic_ellipse(ax2, bq)
    schematic_doppler_bars(ax3)
    fig.suptitle("Passive-radar bistatic experiment geometry — how the scene becomes a range-Doppler cell",
                 fontsize=12)
    fn = os.path.join(outdir, "viz_geometry.png")
    fig.savefig(fn, dpi=140); plt.close(fig)
    print("[schematic]", fn)
    return fn


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"))
    ap.add_argument("--no-render", action="store_true", help="skip the Sionna RT render (schematic only)")
    ap.add_argument("--marker", type=float, default=4.0, help="drone marker size for the render [m]")
    ap.add_argument("--samples", type=int, default=256, help="render samples/pixel")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    bq = bistatic_quantities()
    print(f"[geom] R_tx={bq['Rtx']:.1f} R_rx={bq['Rrx']:.1f} L={bq['L']:.1f} "
          f"R_b={bq['Rb']:.1f} m  beta={bq['beta']:.1f}deg")
    for lab, fc in CARRIERS:
        print(f"[dopp] {lab:14s} radial fD={doppler_for(scenario_velocity('radial'), fc):+7.1f} Hz "
              f"tangential={doppler_for(scenario_velocity('tangential'), fc):+5.1f} Hz")

    make_schematic(a.outdir)
    if not a.no_render:
        make_rt_paths_figure(a.outdir)
        render_views(a.outdir, drone_marker_m=a.marker, num_samples=a.samples)
