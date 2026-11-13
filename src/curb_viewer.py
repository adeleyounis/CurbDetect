"""
Interactive viewer for one frame's curb slices (the .npz written by results_io.save_frame_results).

    pip install viser plotly
    python curb_viewer.py ../outputs/<seq>_cam0_frame000250_slices.npz
    -> open http://localhost:8080

Controls
  ← / →          previous / next slice (click the 3D view first so the page has focus)
  click a ball   select that slice
  shift + click  pick any LiDAR point in 3D -> marked in 3D and in the photo, with its height above road
  mouse / WASD / ↑↓ / QE   move the 3D camera (viser defaults)

3D view
  * points are coloured by height above the local road surface (or by road/sidewalk label),
  * each slice has a "curb ribbon": its fitted profile extruded a short way along the curb,
    with the bottom and top edges of the curb drawn as lines,
  * the same ribbons and edges are drawn on the photo, so you can match 3D and image.

Coordinates: the 3D scene is world XYZ shifted by a local origin (the slices' mean), in metres.
"Vertical exaggeration" only scales the 3D view; numbers in the panels are always true values.
"""
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
import numpy as np
import plotly.graph_objects as go
import viser
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

import analyze_boundary
import results_io

# ----------------------------------------------------------------------------- colours
ROAD_RGB = np.array([70, 110, 200])
SIDE_RGB = np.array([215, 90, 80])
SIDE_COLOR = {"left": (0, 160, 255), "right": (255, 120, 0), "unknown": (160, 160, 160)}
SELECTED = (235, 0, 210)      # magenta
PICK = (60, 255, 60)          # picked point
OUTLINE = (0, 0, 0)

HEIGHT_RANGE = (-0.03, 0.22)  # m, colour scale for "height above road"
HEIGHT_CMAP = matplotlib.colormaps["viridis"]

RIBBON_HALF_LENGTH = 0.5      # m along the curb, each side of the slice
RIBBON_MARGIN = 0.4           # m of road / sidewalk shown each side of the ramp
LIFT = np.array([0, 0, 0.02])  # 3D only: draw ribbons/lines 2 cm above the points so they aren't hidden


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def label_colors(lab, road_id, scale=1.0):
    c = np.where((lab == road_id)[:, None], ROAD_RGB, SIDE_RGB) * scale
    return np.clip(c, 0, 255).astype(np.uint8)


def height_colors(h):
    t = (np.clip(h, *HEIGHT_RANGE) - HEIGHT_RANGE[0]) / (HEIGHT_RANGE[1] - HEIGHT_RANGE[0])
    return (HEIGHT_CMAP(t)[:, :3] * 255).astype(np.uint8)


# ----------------------------------------------------------------------------- data
class Frame:
    """Everything the viewer needs, pre-computed once."""

    def __init__(self, npz_path):
        self.path = Path(npz_path)
        self.meta, self.slices = results_io.load_frame_results(self.path)
        m = self.meta
        self.road_id = int(m["road_id"])
        self.c2w = np.asarray(m["cam_to_world"], float)
        self.P = np.asarray(m["proj_matrix"], float)
        self.K = len(self.slices)
        if self.K == 0:
            raise SystemExit("No slices in this file.")

        img_path = Path(str(m["image_path"]))
        if img_path.exists():
            self.img = np.asarray(Image.open(img_path).convert("RGB"))
        else:
            print(f"WARNING: image not found at {img_path}; using a blank image")
            self.img = np.full((376, 1408, 3), 40, np.uint8)
        self.H, self.W = self.img.shape[:2]

        anchors = np.array([s["anchor_3d"] for s in self.slices])
        self.origin = np.r_[anchors[:, :2].mean(0), np.median(anchors[:, 2])]

        self.ctx_xyz = np.asarray(m["context_xyz"], float)
        self.ctx_lab = np.asarray(m["context_lab"])
        self._build_road_level()
        self.ctx_h = self.height_above_road(self.ctx_xyz)

        for s in self.slices:
            s["h"] = self.height_above_road(s["xyz"])
            s["uv_pts"] = self.to_uv(s["xyz"])
            s["uv_curve"] = self.to_uv(s["curve_xyz"])
            s["uv_anchor"] = self.to_uv(s["anchor_3d"][None])[0]
            self._build_ribbon(s)

    # --- geometry helpers
    def local(self, xyz):
        return np.asarray(xyz, float) - self.origin

    def to_uv(self, xyz):
        return results_io._to_image(np.asarray(xyz, float), self.c2w, self.P)

    def _build_road_level(self, cell=1.0, k=6):
        """Road surface height: median z of road points per 1 m cell, IDW-interpolated."""
        road = self.ctx_xyz[self.ctx_lab == self.road_id]
        if len(road) < 10:  # fall back to all context points
            road = self.ctx_xyz
        ij = np.floor(road[:, :2] / cell).astype(np.int64)
        cells, inv = np.unique(ij, axis=0, return_inverse=True)
        inv = inv.ravel()
        order = np.argsort(inv)
        splits = np.split(road[order, 2], np.cumsum(np.bincount(inv))[:-1])
        keep = np.array([len(z) >= 3 for z in splits])
        self._cell_xy = (cells[keep] + 0.5) * cell
        self._cell_z = np.array([np.median(z) for z, ok in zip(splits, keep) if ok])
        self._cell_tree = cKDTree(self._cell_xy)
        self._k = min(k, len(self._cell_xy))

    def road_level(self, xy):
        d, i = self._cell_tree.query(np.atleast_2d(xy), k=self._k)
        d, i = np.atleast_2d(d).reshape(len(xy), -1), np.atleast_2d(i).reshape(len(xy), -1)
        w = 1.0 / np.maximum(d, 0.05) ** 2
        return (w * self._cell_z[i]).sum(1) / w.sum(1)

    def height_above_road(self, xyz):
        xyz = np.atleast_2d(xyz)
        return xyz[:, 2] - self.road_level(xyz[:, :2])

    def _build_ribbon(self, s, n=40):
        fit = s["fit"]
        c, w, beta = fit["c"], fit["width"], fit["beta"]
        lo = max(c - w / 2 - RIBBON_MARGIN, s["s"].min())
        hi = min(c + w / 2 + RIBBON_MARGIN, s["s"].max())
        ss = np.linspace(lo, hi, n)
        zz = analyze_boundary.design_matrix(ss, c, w) @ beta
        nrm = np.asarray(s["normal"], float)
        tan = np.array([-nrm[1], nrm[0]])
        base = np.asarray(s["mean"], float) + ss[:, None] * nrm
        A = np.column_stack((base - RIBBON_HALF_LENGTH * tan, zz))
        B = np.column_stack((base + RIBBON_HALF_LENGTH * tan, zz))
        s["ribbon_verts"] = np.vstack((A, B))
        f = [(i, i + 1, n + i) for i in range(n - 1)] + [(i + 1, n + i + 1, n + i) for i in range(n - 1)]
        s["ribbon_faces"] = np.array(f, np.uint32)
        # curb edges (bottom and top of the ramp), as 3D segments along the curb
        edges = []
        for se in (c - w / 2, c + w / 2):
            z = analyze_boundary.design_matrix(np.array([se]), c, w) @ beta
            p = np.asarray(s["mean"]) + se * nrm
            edges.append(np.array([np.r_[p - RIBBON_HALF_LENGTH * tan, z], np.r_[p + RIBBON_HALF_LENGTH * tan, z]]))
        s["edges_xyz"] = np.array(edges)                             # (2, 2, 3)
        s["uv_ribbon"] = self.to_uv(np.vstack((A, B[::-1])))         # outline polygon
        s["uv_edges"] = [self.to_uv(e) for e in edges]
        s["curb_zone"] = np.abs(s["s"] - c) <= w / 2 + 0.15

    def describe(self, k):
        s = self.slices[k]
        return f"{k}  ·  {s['side'][0].upper()}  ·  {s['step_cm']:5.1f} cm  ·  {s['depth']:4.1f} m"


# ----------------------------------------------------------------------------- photo overlay
def _ok(uv):
    return uv[np.isfinite(uv).all(1)]


def _line(d, uv, color, width, outline=True):
    pts = [tuple(p) for p in _ok(uv)]
    if len(pts) < 2:
        return
    if outline:
        d.line(pts, fill=OUTLINE, width=width + 3)
    d.line(pts, fill=color, width=width)


def render_overlay(fr, sel, visible, pick=None):
    base = Image.fromarray((fr.img * 0.85).astype(np.uint8)).convert("RGBA")
    fill = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fd = ImageDraw.Draw(fill)

    # translucent ribbons
    for k in visible:
        s = fr.slices[k]
        poly = [tuple(p) for p in _ok(s["uv_ribbon"])]
        if len(poly) >= 3:
            col = SELECTED if k == sel else SIDE_COLOR[s["side"]]
            fd.polygon(poly, fill=col + ((130,) if k == sel else (70,)))
    img = Image.alpha_composite(base, fill).convert("RGB")
    d = ImageDraw.Draw(img)

    # selected slice: its LiDAR points, coloured by label
    s = fr.slices[sel]
    cols = label_colors(s["lab"], fr.road_id)
    uv = s["uv_pts"]
    okm = np.isfinite(uv).all(1)
    for (x, y), c in zip(uv[okm], cols[okm]):
        d.point((x, y), fill=tuple(int(v) for v in c))

    # profiles + curb edges (others first, selected on top)
    for k in [k for k in visible if k != sel] + [sel]:
        sk = fr.slices[k]
        col = SELECTED if k == sel else SIDE_COLOR[sk["side"]]
        wide = 4 if k == sel else 3
        _line(d, sk["uv_curve"], col, wide - 1)
        for e in sk["uv_edges"]:
            _line(d, e, col, wide + 1)
        a = sk["uv_anchor"]
        if np.isfinite(a).all():
            r = 7 if k == sel else 5
            d.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], fill=col, outline=OUTLINE, width=2)

    if pick is not None:
        p = fr.to_uv(pick["xyz"][None])[0]
        if np.isfinite(p).all():
            x, y = p
            for w_, c_ in ((5, OUTLINE), (2, PICK)):
                d.line([(x - 14, y), (x - 4, y)], fill=c_, width=w_)
                d.line([(x + 4, y), (x + 14, y)], fill=c_, width=w_)
                d.line([(x, y - 14), (x, y - 4)], fill=c_, width=w_)
                d.line([(x, y + 4), (x, y + 14)], fill=c_, width=w_)
    full = np.asarray(img)

    # zoom on the selected slice (points + ribbon)
    pts = np.vstack([_ok(uv), _ok(s["uv_ribbon"])])
    if len(pts):
        lo = np.maximum(pts.min(0) - 40, 0).astype(int)
        hi = np.minimum(pts.max(0) + 40, [fr.W, fr.H]).astype(int)
        w, h = hi - lo
        if w > 3 * h:  # keep at most 3:1
            cy, h = (lo[1] + hi[1]) // 2, w // 3
            lo[1], hi[1] = max(cy - h // 2, 0), min(cy + h // 2, fr.H)
        crop = Image.fromarray(full[lo[1]:hi[1], lo[0]:hi[0]])
        crop = crop.resize((700, max(int(crop.height * 700 / max(crop.width, 1)), 1)), Image.LANCZOS)
        zoom = np.asarray(crop)
    else:
        zoom = np.zeros((10, 10, 3), np.uint8)
    return full, zoom


# ----------------------------------------------------------------------------- plots / text
def cross_section_figure(fr, k):
    s = fr.slices[k]
    fit = s["fit"]
    road = s["lab"] == fr.road_id
    fig = go.Figure()
    for mask, name, rgb in ((road, "road", ROAD_RGB), (~road, "sidewalk", SIDE_RGB)):
        fig.add_trace(go.Scattergl(x=s["s"][mask] * 100, y=s["z"][mask] * 100, mode="markers", name=name,
                                   marker=dict(size=3, color=hexc(rgb), opacity=0.35)))
    fig.add_trace(go.Scatter(x=s["bins"][:, 0] * 100, y=s["bins"][:, 1] * 100, mode="lines",
                             name="bin median", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=s["fit_sz"][:, 0] * 100, y=s["fit_sz"][:, 1] * 100, mode="lines",
                             name="fit", line=dict(color=hexc(SELECTED), width=3)))
    c, w = fit["c"] * 100, fit["width"] * 100
    fig.add_vrect(x0=c - w / 2, x1=c + w / 2, fillcolor=hexc(SELECTED), opacity=0.12, line_width=0)
    fig.update_layout(
        title=dict(text=f"slice {k} · {s['side']} curb · step {s['step_cm']:.1f} cm", font=dict(size=13)),
        xaxis_title="across curb s (cm), road → sidewalk", yaxis_title="world z (cm)",
        margin=dict(l=50, r=10, t=35, b=40), legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        plot_bgcolor="white", height=330,
    )
    # explicit range (the browser keeps a previous "reversed" state otherwise);
    # left curbs are flipped so the plot reads like the image
    lo, hi = s["s"].min() * 100 - 5, s["s"].max() * 100 + 5
    fig.update_xaxes(range=[hi, lo] if s["side"] == "left" else [lo, hi], autorange=False, gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def heights_figure(fr, sel, visible):
    fig = go.Figure()
    for side in ("left", "right", "unknown"):
        ks = [k for k in visible if fr.slices[k]["side"] == side]
        if not ks:
            continue
        fig.add_trace(go.Scatter(
            x=[fr.slices[k]["depth"] for k in ks], y=[fr.slices[k]["step_cm"] for k in ks],
            mode="markers+lines+text", name=side, text=[str(k) for k in ks], textposition="top center",
            marker=dict(size=9, color=hexc(SIDE_COLOR[side])), line=dict(width=1, dash="dot")))
    s = fr.slices[sel]
    fig.add_trace(go.Scatter(x=[s["depth"]], y=[s["step_cm"]], mode="markers", name="selected",
                             marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=hexc(SELECTED), width=3))))
    fig.update_layout(xaxis_title="camera depth (m)", yaxis_title="step height (cm)", height=260,
                      margin=dict(l=50, r=10, t=10, b=40), plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.35, font=dict(size=10)))
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def readout(fr, k):
    s = fr.slices[k]
    b = s["fit"]["beta"]  # [b, m_r, h, m_s, q]
    n_road = int((s["lab"] == fr.road_id).sum())
    return (
        f"**Slice {k}** · {s['side']} curb · camera depth {s['depth']:.1f} m\n\n"
        f"| | |\n|---|---|\n"
        f"| step height | **{s['step_cm']:.1f} cm** |\n"
        f"| ramp width | {s['fit']['width'] * 100:.0f} cm |\n"
        f"| road slope / curvature | {b[1] * 100:.1f} % / {b[4]:.3f} m⁻¹ |\n"
        f"| sidewalk slope | {b[3] * 100:.1f} % |\n"
        f"| fit RMSE (bins) | {np.sqrt(s['fit']['mse']) * 100:.2f} cm |\n"
        f"| points (road / sidewalk) | {n_road} / {len(s['lab']) - n_road} |\n"
    )


# ----------------------------------------------------------------------------- arrow keys
# viser has no keyboard API and uses the arrow keys for its camera. This snippet (run via an
# <img onerror> in a GUI html element) grabs ←/→ before viser sees them and pings a tiny
# local HTTP endpoint, which steps the selection on the Python side.
KEY_JS = (
    "if(!window.__curbKeys){{window.__curbKeys=1;"
    "const u='http://'+location.hostname+':{port}/key?d=';"
    "const h=e=>{{if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight')return;"
    "const t=e.target;if(t&&(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.tagName==='SELECT'||t.isContentEditable))return;"
    "e.stopImmediatePropagation();e.preventDefault();"
    "if(e.type==='keydown'&&!e.repeat)fetch(u+(e.key==='ArrowLeft'?-1:1),{{mode:'no-cors'}}).catch(()=>{{}});}};"
    "window.addEventListener('keydown',h,true);window.addEventListener('keyup',h,true);}}"
)


def start_key_server(port, on_step):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            q = parse_qs(urlparse(self.path).query)
            try:
                on_step(int(q.get("d", ["0"])[0]))
            except Exception as e:  # never let a bad request kill the server
                print("key handler error:", e)
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ----------------------------------------------------------------------------- viewer
class CurbViewer:
    def __init__(self, fr: Frame, port=8080, key_port=None):
        self.fr = fr
        self.lock = threading.RLock()
        self.server = viser.ViserServer(port=port, label=f"curb viewer · {fr.path.name}")
        self.server.gui.configure_theme(control_width="large", show_share_button=False)
        self.server.scene.set_up_direction("+z")
        self.key_port = key_port or port + 1
        self.key_server = start_key_server(self.key_port, self.step)
        self.sel = 0
        self.exag = 1.0
        self.pick = None
        self.handles = {}
        self._build_gui()
        self._build_scene()
        self.select(self.visible()[0])

    def visible(self):
        f = self.side_filter.value.lower()
        ks = [k for k, s in enumerate(self.fr.slices) if f == "both" or s["side"] == f]
        return ks or list(range(self.fr.K))

    # ---------------- GUI
    def _build_gui(self):
        g, fr = self.server.gui, self.fr
        g.add_markdown(f"**{fr.meta['seq']}** · frame **{int(fr.meta['frame'])}** · {fr.K} slices "
                       f"({int(np.isnan(fr.meta['all_heights_cm']).sum())} failed bands)")

        with g.add_folder("Slice"):
            self.side_filter = g.add_dropdown("Curb side", ["Both", "Left", "Right"])
            self.slice_dd = g.add_dropdown("Slice", [fr.describe(k) for k in range(fr.K)],
                                           hint="slice · side · height · depth")
            g.add_html(
                '<div style="font-size:12px;opacity:.7;padding:2px 0 4px">'
                '<b>← / →</b> previous / next slice (click the 3D view first) · '
                '<b>shift+click</b> a point in 3D to locate it in the photo</div>'
                f'<img src="/__curb_keys_{self.key_port}.png" style="display:none" '
                f'onerror="{KEY_JS.format(port=self.key_port)}">'
            )
            self.info = g.add_markdown("")
            self.pick_info = g.add_markdown("")

        with g.add_folder("Photo"):
            blank = np.zeros((10, 10, 3), np.uint8)
            self.img_full = g.add_image(blank, label="camera (selected = magenta)", format="jpeg", jpeg_quality=88)
            self.img_zoom = g.add_image(blank, label="zoom on selected slice", format="jpeg", jpeg_quality=92)

        with g.add_folder("Cross-section"):
            self.xsec = g.add_plotly(go.Figure(), aspect=1.6)

        with g.add_folder("Heights by depth", expand_by_default=False):
            self.hplot = g.add_plotly(go.Figure(), aspect=2.0)

        with g.add_folder("3D display"):
            self.color_mode = g.add_dropdown("Colour points by", ["Height above road", "Road / sidewalk"])
            g.add_markdown(f"<small>height colours: dark purple ≤ {HEIGHT_RANGE[0] * 100:.0f} cm → "
                           f"green → yellow ≥ {HEIGHT_RANGE[1] * 100:.0f} cm</small>")
            self.exag_slider = g.add_slider("Vertical exaggeration", 1.0, 10.0, 0.5, 3.0,
                                            hint="Scales heights in the 3D view only")
            self.pt_size = g.add_slider("Point size (cm)", 0.5, 8.0, 0.5, 3.0)
            self.show_ctx = g.add_checkbox("LiDAR context", True)
            self.show_all_pts = g.add_checkbox("Points of all slices", False)
            self.show_ribbons = g.add_checkbox("Curb ribbons", True)
            self.show_labels = g.add_checkbox("Height labels", True)
            self.show_cam = g.add_checkbox("Camera position", True)
            self.btn_focus = g.add_button("Focus on selected slice")
            self.btn_cam = g.add_button("View from camera", hint="Looks through the camera (resets exaggeration to 1)")
            self.btn_clear = g.add_button("Clear picked point")

        self.side_filter.on_update(lambda _: self.select(self.sel if self.sel in self.visible() else self.visible()[0]))
        self.slice_dd.on_update(lambda _: (k := self.slice_dd.options.index(self.slice_dd.value)) != self.sel
                                and self.select(k))
        self.exag_slider.on_update(lambda _: self._set_exag(self.exag_slider.value))
        self.color_mode.on_update(lambda _: self._recolor())
        self.pt_size.on_update(lambda _: self._refresh_scene())
        for cb in (self.show_ctx, self.show_all_pts, self.show_ribbons, self.show_labels, self.show_cam):
            cb.on_update(lambda _: self._refresh_scene())
        self.btn_focus.on_click(lambda ev: self._focus(ev.client))
        self.btn_cam.on_click(lambda ev: self._view_from_camera(ev.client))
        self.btn_clear.on_click(lambda _: self._set_pick(None))

    # ---------------- 3D scene
    def _colors(self, which):
        fr = self.fr
        by_h = self.color_mode.value.startswith("Height")
        if which == "ctx":
            return height_colors(fr.ctx_h) if by_h else label_colors(fr.ctx_lab, fr.road_id, 0.8)
        s = fr.slices[which]
        return height_colors(s["h"]) if by_h else label_colors(s["lab"], fr.road_id, 1.1)

    def _build_scene(self):
        sc, fr = self.server.scene, self.fr
        self.world = sc.add_frame("/world", show_axes=False)
        ps = self.pt_size.value / 100
        pc_kw = dict(point_shape="circle", point_shading="gradient", precision="float32")

        self.handles["ctx"] = sc.add_point_cloud("/world/context", fr.local(fr.ctx_xyz), self._colors("ctx"),
                                                 point_size=ps * 0.8, **pc_kw)
        for k, s in enumerate(fr.slices):
            col = SIDE_COLOR[s["side"]]
            pts = sc.add_point_cloud(f"/world/slice_{k}/points", fr.local(s["xyz"]), self._colors(k),
                                     point_size=ps, visible=False, **pc_kw)
            curve = fr.local(s["curve_xyz"]) + LIFT
            prof = sc.add_line_segments(f"/world/slice_{k}/profile", np.stack([curve[:-1], curve[1:]], 1),
                                        col, thickness=5, thickness_units="screen")
            ribbon = sc.add_mesh_simple(f"/world/slice_{k}/ribbon", fr.local(s["ribbon_verts"]) + LIFT, s["ribbon_faces"],
                                        color=col, opacity=0.85, side="double", flat_shading=False)
            ribbon.on_click(lambda _, k=k: self.select(k))
            edges = sc.add_line_segments(f"/world/slice_{k}/edges", fr.local(s["edges_xyz"]) + LIFT, col,
                                         thickness=8, thickness_units="screen")
            anchor = sc.add_icosphere(f"/world/slice_{k}/anchor", radius=0.07, color=col,
                                      position=fr.local(s["anchor_3d"]))
            anchor.on_click(lambda _, k=k: self.select(k))
            lab = sc.add_label(f"/world/slice_{k}/label", f"{k}: {s['step_cm']:.1f} cm",
                               position=fr.local(s["anchor_3d"]) + [0, 0, 0.3], anchor="bottom-center")
            self.handles[k] = dict(points=pts, profile=prof, ribbon=ribbon, edges=edges, anchor=anchor,
                                   label=lab, n_seg=len(curve) - 1)

        self.handles["pick"] = sc.add_icosphere("/world/pick", radius=0.06, color=PICK, visible=False)

        # camera position only (wireframe frustum, no image)
        self.cam_fov = 2 * np.arctan2(fr.H / 2, fr.P[1, 1])
        self.cam_wxyz = vtf.SO3.from_matrix(fr.c2w[:3, :3]).wxyz
        self.cam_pos = fr.local(fr.c2w[:3, 3])
        self.handles["cam"] = sc.add_camera_frustum(
            "/camera", fov=self.cam_fov, aspect=fr.W / fr.H, scale=0.4, wxyz=self.cam_wxyz,
            position=self.cam_pos, color=(30, 30, 30), thickness=3, thickness_units="screen")

        @sc.on_click(modifier="shift")
        def _(ev):
            self._pick_from_ray(np.asarray(ev.ray_origin), np.asarray(ev.ray_direction))

        @self.server.on_client_connect
        def _(client):
            back = fr.c2w[:3, 2].copy()
            back[2] = 0
            back /= np.linalg.norm(back) + 1e-9
            with client.atomic():
                client.camera.up_direction = (0.0, 0.0, 1.0)
                client.camera.position = self.cam_pos * [1, 1, self.exag] - 6 * back + np.array([0, 0, 6.0])
                client.camera.look_at = (0.0, 0.0, 0.0)

        self._set_exag(self.exag_slider.value)

    def _set_exag(self, e):
        with self.lock:
            self.exag = float(e)
            self.world.scale = (1.0, 1.0, self.exag)
            self.handles["cam"].position = self.cam_pos * [1, 1, self.exag]

    def _recolor(self):
        with self.lock:
            self.handles["ctx"].colors = self._colors("ctx")
            for k in range(self.fr.K):
                self.handles[k]["points"].colors = self._colors(k)

    def _refresh_scene(self):
        with self.lock:
            ps = self.pt_size.value / 100
            vis = set(self.visible())
            self.handles["ctx"].visible = self.show_ctx.value
            self.handles["ctx"].point_size = ps * 0.8
            self.handles["cam"].visible = self.show_cam.value
            for k, s in enumerate(self.fr.slices):
                h = self.handles[k]
                on, selected = k in vis, k == self.sel
                color = np.array(SELECTED if selected else SIDE_COLOR[s["side"]], np.uint8)
                h["points"].visible = on and (selected or self.show_all_pts.value)
                h["points"].point_size = ps
                h["profile"].visible = on
                h["profile"].colors = color  # single colour: switching to per-vertex arrays turns lines grey in viser
                h["profile"].thickness = 9 if selected else 5
                h["ribbon"].visible = on and self.show_ribbons.value
                h["ribbon"].color = tuple(int(c) for c in color)
                h["edges"].visible = on and self.show_ribbons.value
                h["edges"].colors = color
                h["edges"].thickness = 12 if selected else 8
                h["anchor"].visible = on
                h["anchor"].color = tuple(int(c) for c in color)
                h["label"].visible = on and (self.show_labels.value or selected)

    # ---------------- picking
    def _pick_from_ray(self, o, d):
        fr = self.fr
        pools = [fr.ctx_xyz] + [s["xyz"] for s in fr.slices]
        xyz = np.vstack(pools)
        scene = fr.local(xyz) * [1, 1, self.exag]
        d = d / np.linalg.norm(d)
        t = (scene - o) @ d
        perp = np.linalg.norm(scene - o - t[:, None] * d, axis=1)
        ok = (t > 0) & (perp < np.maximum(0.08, 0.008 * t))
        if not ok.any():
            return
        cand = np.flatnonzero(ok)
        best = cand[np.argmin(t[cand] + 5 * perp[cand])]  # front-most, closest to the ray
        p = xyz[best]
        lab = np.concatenate([fr.ctx_lab] + [s["lab"] for s in fr.slices])[best]
        self._set_pick(dict(xyz=p, lab=int(lab), h=float(fr.height_above_road(p[None])[0])))

    def _set_pick(self, pick):
        with self.lock:
            self.pick = pick
            hp = self.handles["pick"]
            if pick is None:
                hp.visible = False
                self.pick_info.content = ""
            else:
                hp.position = self.fr.local(pick["xyz"])
                hp.visible = True
                name = "road" if pick["lab"] == self.fr.road_id else "sidewalk"
                self.pick_info.content = (f"**Picked point** ({name}): height above road "
                                          f"**{pick['h'] * 100:.1f} cm** · green cross in the photo")
            self._render_photo()

    # ---------------- selection
    def step(self, d):
        ks = self.visible()
        i = ks.index(self.sel) if self.sel in ks else 0
        self.select(ks[(i + d) % len(ks)])

    def select(self, k):
        with self.lock:
            k = int(k)
            self.sel = k
            fr, vis = self.fr, self.visible()
            if self.slice_dd.value != fr.describe(k):
                self.slice_dd.value = fr.describe(k)  # on_update sees k == self.sel and does nothing
            self.info.content = readout(fr, k)
            self._render_photo()
            self.xsec.figure = cross_section_figure(fr, k)
            self.hplot.figure = heights_figure(fr, k, vis)
            self._refresh_scene()

    def _render_photo(self):
        full, zoom = render_overlay(self.fr, self.sel, self.visible(), self.pick)
        self.img_full.image = full
        self.img_zoom.image = zoom

    def _focus(self, client):
        if client is None:
            return
        s = self.fr.slices[self.sel]
        a = self.fr.local(s["anchor_3d"]) * [1, 1, self.exag]
        n = np.r_[s["normal"], 0]
        along = np.cross([0, 0, 1], n)
        with client.atomic():
            client.camera.position = a - 2.0 * n + 1.5 * along + np.array([0, 0, 2.5])
            client.camera.look_at = a

    def _view_from_camera(self, client):
        if client is None:
            return
        self.exag_slider.value = 1.0
        self._set_exag(1.0)
        with client.atomic():
            client.camera.fov = float(self.cam_fov)
            client.camera.position = self.cam_pos
            client.camera.wxyz = self.cam_wxyz

    def run(self):
        print(f"\n  curb viewer running -> http://localhost:{self.server.get_port()}   (Ctrl+C to stop)"
              f"\n  (arrow keys use port {self.key_port})\n")
        self.server.sleep_forever()


def main():
    ap = argparse.ArgumentParser(description="Interactive viewer for curb slice results (.npz).")
    ap.add_argument("npz", type=Path)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--key-port", type=int, default=None, help="port for arrow-key events (default: port+1)")
    args = ap.parse_args()
    CurbViewer(Frame(args.npz), port=args.port, key_port=args.key_port).run()


if __name__ == "__main__":
    main()