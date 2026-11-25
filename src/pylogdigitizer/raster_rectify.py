# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""raster_rectify: crooked raster log calibration, 2-D flattening, and ML dataset capture.

Models:
- DepthTieLine: a depth reference line across the sheet with tilt/angle or multi-point curvature
  (for warped or folded scans), optional gap flag (skipping intervals with missing/discontinuous
  sections), and synthetic flag (temporary header/footer nodes used during straightening).
- TrackEdgeGuide: left and right boundary polylines for each track (gamma, resistivity, porosity, or a
  center depth column), with curve-bridging vertical line tracking between user-picked nodes.
- RasterCalibration: combines depth tie-lines and track boundaries into a full 2-D
  grid_rectify.Lattice that maps angled/curved depth lines to 90 degrees (horizontal) against
  straightened vertical track boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

from pylogdigitizer.grid_rectify import Lattice, rectify_image


@dataclass
class DepthTieLine:
    """A depth reference line across the log sheet.

    ``nodes`` is a list of ``(x, y)`` sheet points sorted by ``x``. Two points define a tilted
    straight line; >= 3 points define a piecewise curve across a warped or folded scan.
    """
    depth: float
    nodes: list[tuple[float, float]]
    is_gap: bool = False          # True = interval below this line up to next line is a skipped gap
    synthetic: bool = False       # True = temporary header/footer node used only for straightening

    def __post_init__(self):
        pts = sorted((float(x), float(y)) for x, y in self.nodes)
        if len(pts) < 2:
            raise ValueError("DepthTieLine needs at least 2 (x, y) points")
        self.depth = float(self.depth)
        self.nodes = pts

    @classmethod
    def from_angle(cls, depth: float, x_ref: float, y_ref: float, angle_deg: float,
                   sheet_width: int, is_gap: bool = False, synthetic: bool = False) -> "DepthTieLine":
        """Create a tilted straight tie-line through ``(x_ref, y_ref)`` at ``angle_deg`` from horizontal."""
        slope = float(np.tan(np.deg2rad(angle_deg)))
        y0 = y_ref + slope * (0.0 - x_ref)
        y1 = y_ref + slope * (float(sheet_width - 1) - x_ref)
        return cls(depth=depth, nodes=[(0.0, y0), (float(sheet_width - 1), y1)],
                   is_gap=is_gap, synthetic=synthetic)

    @property
    def angle_deg(self) -> float:
        (x0, y0), (x1, y1) = self.nodes[0], self.nodes[-1]
        return float(np.rad2deg(np.arctan2(y1 - y0, max(1e-6, x1 - x0))))

    def y_at(self, x) -> np.ndarray:
        """Evaluate sheet row ``y`` at column(s) ``x`` with linear end extrapolation."""
        xx = np.asarray(x, float)
        nx = np.array([p[0] for p in self.nodes], float)
        ny = np.array([p[1] for p in self.nodes], float)
        yy = np.interp(xx, nx, ny)
        lo, hi = xx < nx[0], xx > nx[-1]
        if lo.any():
            s0 = (ny[1] - ny[0]) / max(1e-6, nx[1] - nx[0])
            yy = np.where(lo, ny[0] + (xx - nx[0]) * s0, yy)
        if hi.any():
            s1 = (ny[-1] - ny[-2]) / max(1e-6, nx[-1] - nx[-2])
            yy = np.where(hi, ny[-1] + (xx - nx[-1]) * s1, yy)
        return yy

    def shift(self, dy: float = 0.0, d_angle_deg: float = 0.0, x_pivot: float | None = None) -> "DepthTieLine":
        """Shift vertically by ``dy`` px and/or rotate around ``x_pivot`` by ``d_angle_deg``."""
        nx = np.array([p[0] for p in self.nodes], float)
        ny = np.array([p[1] for p in self.nodes], float) + float(dy)
        if abs(d_angle_deg) > 1e-9:
            xp = float(np.mean(nx)) if x_pivot is None else float(x_pivot)
            ny = ny + np.tan(np.deg2rad(d_angle_deg)) * (nx - xp)
        return DepthTieLine(self.depth, list(zip(nx, ny)), is_gap=self.is_gap, synthetic=self.synthetic)

    def to_dict(self) -> dict:
        return dict(depth=self.depth, nodes=[list(p) for p in self.nodes],
                    is_gap=self.is_gap, synthetic=self.synthetic)

    @classmethod
    def from_dict(cls, d: dict) -> "DepthTieLine":
        return cls(depth=d["depth"], nodes=[tuple(p) for p in d["nodes"]],
                   is_gap=bool(d.get("is_gap", False)), synthetic=bool(d.get("synthetic", False)))


@dataclass
class TrackEdgeGuide:
    """Left and right boundary polylines for one log track (or center depth column)."""
    name: str
    kind: str = "track"   # "track" or "depth"
    left_nodes: list[tuple[float, float]] = field(default_factory=list)   # (x, y) sorted by y
    right_nodes: list[tuple[float, float]] = field(default_factory=list)  # (x, y) sorted by y

    def __post_init__(self):
        self.left_nodes = sorted(((float(x), float(y)) for x, y in self.left_nodes), key=lambda p: p[1])
        self.right_nodes = sorted(((float(x), float(y)) for x, y in self.right_nodes), key=lambda p: p[1])

    @staticmethod
    def _eval_poly(nodes: list[tuple[float, float]], y) -> np.ndarray:
        if len(nodes) < 2:
            raise ValueError("Track edge needs at least 2 (x, y) nodes")
        yy = np.asarray(y, float)
        ny = np.array([p[1] for p in nodes], float)
        nx = np.array([p[0] for p in nodes], float)
        xx = np.interp(yy, ny, nx)
        lo, hi = yy < ny[0], yy > ny[-1]
        if lo.any():
            s0 = (nx[1] - nx[0]) / max(1e-6, ny[1] - ny[0])
            xx = np.where(lo, nx[0] + (yy - ny[0]) * s0, xx)
        if hi.any():
            s1 = (nx[-1] - nx[-2]) / max(1e-6, ny[-1] - ny[-2])
            xx = np.where(hi, nx[-1] + (yy - ny[-1]) * s1, xx)
        return xx

    def left_at(self, y) -> np.ndarray:
        return self._eval_poly(self.left_nodes, y)

    def right_at(self, y) -> np.ndarray:
        return self._eval_poly(self.right_nodes, y)

    def to_dict(self) -> dict:
        return dict(name=self.name, kind=self.kind,
                    left_nodes=[list(p) for p in self.left_nodes],
                    right_nodes=[list(p) for p in self.right_nodes])

    @classmethod
    def from_dict(cls, d: dict) -> "TrackEdgeGuide":
        return cls(name=d["name"], kind=d.get("kind", "track"),
                   left_nodes=[tuple(p) for p in d["left_nodes"]],
                   right_nodes=[tuple(p) for p in d["right_nodes"]])


def follow_vertical_boundary(ink: np.ndarray, p_start: tuple[float, float], p_end: tuple[float, float],
                             step_px: int = 40, search_px: int = 12, line_len: int = 45) -> list[tuple[float, float]]:
    """Trace a crooked vertical track boundary between ``p_start=(x0, y0)`` and ``p_end=(x1, y1)``
    while bridging across intersecting log curves using morphological vertical filtering + slope prediction.
    """
    import cv2
    x0, y0 = float(p_start[0]), float(p_start[1])
    x1, y1 = float(p_end[0]), float(p_end[1])
    if y1 < y0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    if y1 - y0 < step_px * 1.5:
        return [(x0, y0), (x1, y1)]

    a = np.asarray(ink, np.uint8)
    a = cv2.dilate(a, np.ones((1, 5), np.uint8))
    vert = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((line_len, 1), np.uint8)).astype(bool) & np.asarray(ink, bool)

    ys = np.arange(y0, y1 + 0.5 * step_px, step_px)
    ys[-1] = y1
    pts = [(x0, y0)]
    for k in range(1, len(ys) - 1):
        yk = float(ys[k])
        t = (yk - pts[-1][1]) / max(1e-6, y1 - pts[-1][1])
        x_pred = pts[-1][0] + t * (x1 - pts[-1][0])
        r0 = max(0, int(round(yk - step_px / 2)))
        r1 = min(vert.shape[0], int(round(yk + step_px / 2)))
        c0 = max(0, int(round(x_pred - search_px)))
        c1 = min(vert.shape[1], int(round(x_pred + search_px + 1)))
        prof = vert[r0:r1, c0:c1].mean(axis=0)
        if prof.size and prof.max() >= 0.35:
            w = np.where(prof >= 0.6 * prof.max(), prof, 0.0)
            cols = np.arange(c0, c1, dtype=float)
            x_best = float((cols * w).sum() / w.sum())
            pts.append((x_best, yk))
        else:
            pts.append((float(x_pred), yk))
    pts.append((x1, y1))
    return pts


def interpolate_depths(tie_lines: list[DepthTieLine], step_ft: float) -> list[DepthTieLine]:
    """Insert interpolated angled/curved tie-lines every ``step_ft`` between calibrated depths
    (skips intervals marked ``is_gap=True``)."""
    lines = sorted(tie_lines, key=lambda l: l.depth)
    if len(lines) < 2 or step_ft <= 0:
        return lines
    out: list[DepthTieLine] = []
    for a, b in zip(lines[:-1], lines[1:]):
        out.append(a)
        if a.is_gap:
            continue
        span = b.depth - a.depth
        n_steps = int(np.floor((span - 1e-6) / step_ft))
        if n_steps < 1:
            continue
        xs = np.unique(np.r_[[p[0] for p in a.nodes], [p[0] for p in b.nodes]])
        ya, yb = a.y_at(xs), b.y_at(xs)
        for k in range(1, n_steps + 1):
            d = a.depth + k * step_ft
            if d >= b.depth - 1e-6:
                break
            t = (d - a.depth) / span
            yk = (1.0 - t) * ya + t * yb
            out.append(DepthTieLine(depth=float(d), nodes=list(zip(xs, yk))))
    out.append(lines[-1])
    return out


def decimate_depths(tie_lines: list[DepthTieLine], keep_every: int = 2) -> list[DepthTieLine]:
    """Thin interior tie-lines while keeping first, last, gap, and synthetic anchors."""
    lines = sorted(tie_lines, key=lambda l: l.depth)
    if len(lines) <= 2 or keep_every <= 1:
        return lines
    out = [lines[0]]
    for idx in range(1, len(lines) - 1):
        if lines[idx].is_gap or lines[idx].synthetic or (idx % keep_every == 0):
            out.append(lines[idx])
    out.append(lines[-1])
    return out


def rectify_any(img: np.ndarray, lat: Lattice, out_shape: tuple[int, int] | None = None,
                order: int = 1) -> np.ndarray:
    """Warp 2-D (bool/uint8) or 3-channel (H, W, C) images using ``grid_rectify.rectify_image``."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return rectify_image(arr, lat, out_shape=out_shape, order=order)
    if arr.ndim == 3:
        H, W = out_shape or (lat.shape if lat.shape[0] else arr.shape[:2])
        chans = [rectify_image(arr[..., c], lat, out_shape=(H, W), order=order) for c in range(arr.shape[2])]
        return np.stack(chans, axis=-1)
    raise ValueError(f"unsupported image shape {arr.shape}")


@dataclass
class RasterCalibration:
    """Complete manual/semi-auto calibration for crooked multi-track raster logs."""
    sheet_shape: tuple[int, int]                          # (H, W) of raw sheet
    tie_lines: list[DepthTieLine] = field(default_factory=list)
    tracks: list[TrackEdgeGuide] = field(default_factory=list)
    px_per_ft: float | None = None                        # optional target vertical scale (auto if None)

    def sorted_ties(self) -> list[DepthTieLine]:
        return sorted(self.tie_lines, key=lambda l: l.depth)

    def sorted_tracks(self) -> list[TrackEdgeGuide]:
        return sorted(self.tracks, key=lambda t: np.median([p[0] for p in t.left_nodes]))

    def build_lattice(self, n_subdiv_y: int = 4) -> Lattice:
        """Construct a 2-D ``grid_rectify.Lattice`` that flattens angled/curved depth tie-lines to
        horizontal rows (90 deg to vertical track edges) and straightens crooked track edges.

        ``n_subdiv_y`` inserts intermediate rows between tie-lines so curved track boundaries between
        sparse depth picks are followed smoothly.
        """
        ties = self.sorted_ties()
        trks = self.sorted_tracks()
        if len(ties) < 2 or not trks:
            raise ValueError("Need >= 2 DepthTieLines and >= 1 TrackEdgeGuide to build a Lattice")

        dense_ties: list[DepthTieLine] = []
        for a, b in zip(ties[:-1], ties[1:]):
            dense_ties.append(a)
            if not a.is_gap and n_subdiv_y > 1:
                xs = np.unique(np.r_[[p[0] for p in a.nodes], [p[0] for p in b.nodes]])
                ya, yb = a.y_at(xs), b.y_at(xs)
                for s in range(1, n_subdiv_y):
                    t = s / n_subdiv_y
                    dense_ties.append(DepthTieLine(
                        depth=(1.0 - t) * a.depth + t * b.depth,
                        nodes=list(zip(xs, (1.0 - t) * ya + t * yb)),
                        synthetic=True,
                    ))
        dense_ties.append(ties[-1])

        edge_fns = []
        ideal_u = []
        cur_u = 0.0
        for idx, tr in enumerate(trks):
            ly = np.array([p[1] for p in tr.left_nodes], float)
            w_med = float(np.median(tr.right_at(ly) - tr.left_at(ly)))
            w_med = max(20.0, w_med)
            if idx == 0:
                cur_u = float(np.median([p[0] for p in tr.left_nodes]))
            else:
                prev_tr = trks[idx - 1]
                gap = float(np.median(tr.left_at(ly) - prev_tr.right_at(ly)))
                cur_u = ideal_u[-1] + max(0.0, gap)
            edge_fns.append(tr.left_at)
            ideal_u.append(cur_u)
            edge_fns.append(tr.right_at)
            ideal_u.append(cur_u + w_med)

        nh, nv = len(dense_ties), len(edge_fns)
        X = np.zeros((nh, nv), float)
        Y = np.zeros((nh, nv), float)

        for i, tie in enumerate(dense_ties):
            for j, efn in enumerate(edge_fns):
                y_est = float(np.median([p[1] for p in tie.nodes]))
                x_est = float(efn(y_est))
                for _ in range(6):
                    y_est = float(tie.y_at(x_est))
                    x_est = float(efn(y_est))
                X[i, j], Y[i, j] = x_est, y_est

        v0 = float(np.mean(Y[0]))
        if self.px_per_ft is not None and self.px_per_ft > 0:
            ppf = float(self.px_per_ft)
        else:
            dy_tot = float(np.mean(Y[-1]) - np.mean(Y[0]))
            dd_tot = max(1e-6, dense_ties[-1].depth - dense_ties[0].depth)
            ppf = max(1.0, dy_tot / dd_tot)
        v = np.array([v0 + (t.depth - dense_ties[0].depth) * ppf for t in dense_ties], float)
        u = np.array(ideal_u, float)

        out_h = max(self.sheet_shape[0], int(np.ceil(v[-1] + (self.sheet_shape[0] - np.mean(Y[-1])))))
        out_w = max(self.sheet_shape[1], int(np.ceil(u[-1] + (self.sheet_shape[1] - np.mean(X[:, -1])))))
        return Lattice(X=X, Y=Y, u=u, v=v, shape=(out_h, out_w))

    def rectify(self, img: np.ndarray, n_subdiv_y: int = 4, order: int = 1) -> tuple[np.ndarray, Lattice]:
        lat = self.build_lattice(n_subdiv_y=n_subdiv_y)
        rect = rectify_any(img, lat, out_shape=lat.shape, order=order)
        return rect, lat

    def to_dict(self) -> dict:
        return dict(
            sheet_shape=list(self.sheet_shape),
            px_per_ft=self.px_per_ft,
            tie_lines=[t.to_dict() for t in self.tie_lines],
            tracks=[t.to_dict() for t in self.tracks],
        )

    @classmethod
    def from_dict(cls, d: dict) -> "RasterCalibration":
        return cls(
            sheet_shape=tuple(d["sheet_shape"]),
            px_per_ft=d.get("px_per_ft"),
            tie_lines=[DepthTieLine.from_dict(x) for x in d.get("tie_lines", [])],
            tracks=[TrackEdgeGuide.from_dict(x) for x in d.get("tracks", [])],
        )


def save_rectification_sample(out_dir: str | Path, sample_id: str, raw_img: np.ndarray,
                              cal: RasterCalibration, meta: dict | None = None) -> Path:
    """Save paired raw scan, flattened scan, 2-D Lattice, and control points for AI training."""
    from PIL import Image
    d = Path(out_dir) / sample_id
    d.mkdir(parents=True, exist_ok=True)
    rect_img, lat = cal.rectify(raw_img)

    def _to_u8(im):
        return np.where(im, 0, 255).astype(np.uint8) if im.dtype == bool else im.astype(np.uint8)

    Image.fromarray(_to_u8(raw_img)).save(d / "raw.png")
    Image.fromarray(_to_u8(rect_img)).save(d / "rectified.png")
    np.savez_compressed(d / "lattice.npz", X=lat.X, Y=lat.Y, u=lat.u, v=lat.v, shape=np.array(lat.shape))
    payload = dict(sample_id=sample_id, calibration=cal.to_dict(), meta=meta or {})
    (d / "calibration.json").write_text(json.dumps(payload, indent=2))
    return d
