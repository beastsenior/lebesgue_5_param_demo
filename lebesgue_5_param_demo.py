"""
Interactive D/R3/R5 test-body demo for Lebesgue's universal covering
problem.

The app shows three diameter-one test bodies -- the disk D, the Reuleaux
triangle R3, and the Reuleaux pentagon R5 -- placed in a common convex hull.
It is intended as a geometric visualization of the finite Reuleaux test-body
method.

Run:
    streamlit run lebesgue_5_param_demo_visual.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


WIDTH = 1.0
DISK_RADIUS = WIDTH / 2.0

# Reference orientations for the visual model.  R3 is fixed; R5 is rotated by
# psi5.  If the paper fixes a different reference convention, only these
# constants need to be changed.
R3_FIXED_ANGLE_DEG = 0.0
BASE_VERTEX_ANGLE_DEG = 0.0

SAMPLES_PER_REULEAUX_ARC = 100
DISK_BOUNDARY_SAMPLES = 240

DEFAULT_Z: Dict[str, float] = {
    "x3": 0.0,
    "y3": 0.0,
    "psi5_deg": 0.0,
    "x5": 0.0,
    "y5": 0.0,
}

DEMO_UI_RANGES = {
    "x3": (-0.20, 0.20, 0.001, "%.9f"),
    "y3": (-0.20, 0.20, 0.001, "%.9f"),
    "psi5_deg": (0.0, 36.0, 0.05, "%.9f"),
    "x5": (-0.20, 0.20, 0.001, "%.9f"),
    "y5": (-0.20, 0.20, 0.001, "%.9f"),
}

ABOUT_DEMO = (
    "This interactive page illustrates a low-order finite Reuleaux test-body "
    "configuration used to explain lower-bound constructions for Lebesgue's "
    "universal covering problem. The displayed area is computed from the "
    "sampled boundaries shown in the figure."
)

Point = Tuple[float, float]


@dataclass(frozen=True)
class Pose:
    center: Tuple[float, float]
    angle_rad: float


def init_param_state() -> None:
    for name, value in DEFAULT_Z.items():
        st.session_state.setdefault(name, value)
        st.session_state.setdefault(f"{name}_slider", value)
        st.session_state.setdefault(f"{name}_input", value)


def set_param_state(values: Dict[str, float]) -> None:
    for name, value in values.items():
        value = float(value)
        st.session_state[name] = value
        st.session_state[f"{name}_slider"] = value
        st.session_state[f"{name}_input"] = value


def sync_from_slider(name: str) -> None:
    value = float(st.session_state[f"{name}_slider"])
    st.session_state[name] = value
    st.session_state[f"{name}_input"] = value


def sync_from_input(name: str) -> None:
    value = float(st.session_state[f"{name}_input"])
    st.session_state[name] = value
    st.session_state[f"{name}_slider"] = value


def slider_with_exact_input(
    name: str,
    label: str,
    ranges: Dict[str, Tuple[float, float, float, str]],
) -> float:
    min_value, max_value, step, fmt = ranges[name]
    current = float(st.session_state[name])
    current = min(max(current, min_value), max_value)
    if current != st.session_state[name]:
        set_param_state({name: current})

    col_slider, col_input = st.columns([2.2, 1.0])
    with col_slider:
        st.slider(
            label,
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(step),
            key=f"{name}_slider",
            on_change=sync_from_slider,
            args=(name,),
        )
    with col_input:
        st.number_input(
            "exact value",
            min_value=float(min_value),
            max_value=float(max_value),
            step=float(step),
            format=fmt,
            key=f"{name}_input",
            on_change=sync_from_input,
            args=(name,),
            label_visibility="hidden",
        )
    return float(st.session_state[name])


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_points(points: np.ndarray, pose: Pose) -> np.ndarray:
    rot = rotation_matrix(pose.angle_rad)
    return points @ rot.T + np.array(pose.center, dtype=float)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 1:
        return np.array(pts, dtype=float)

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-15:
            lower.pop()
        lower.append(p)

    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-15:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def sample_disk(radius: float = DISK_RADIUS, samples: int = DISK_BOUNDARY_SAMPLES) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t)))


def regular_reuleaux_vertices(n: int, width: float = WIDTH) -> np.ndarray:
    if n < 3 or n % 2 == 0:
        raise ValueError("n must be an odd integer at least 3.")

    radius = width / (2.0 * math.cos(math.pi / (2.0 * n)))
    base = math.radians(BASE_VERTEX_ANGLE_DEG)
    t = base + np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t)))


def shortest_angle_interval(a0: float, a1: float, samples: int) -> np.ndarray:
    delta = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    return a0 + np.linspace(0.0, delta, samples)


def sample_reuleaux_odd_ngon(
    n: int,
    width: float = WIDTH,
    samples_per_arc: int = SAMPLES_PER_REULEAUX_ARC,
    pose: Pose | None = None,
) -> np.ndarray:
    vertices = regular_reuleaux_vertices(n, width)
    m = (n - 1) // 2

    arc_points: List[np.ndarray] = []
    for i in range(n):
        center = vertices[i]
        p0 = vertices[(i - m) % n]
        p1 = vertices[(i + m) % n]

        a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
        a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
        angles = shortest_angle_interval(a0, a1, samples_per_arc)

        arc = np.column_stack(
            (
                center[0] + width * np.cos(angles),
                center[1] + width * np.sin(angles),
            )
        )
        arc_points.append(arc)

    pts = np.vstack(arc_points)
    if pose is not None:
        pts = transform_points(pts, pose)
    return pts


def close_polygon(poly: np.ndarray) -> np.ndarray:
    if len(poly) == 0:
        return poly
    return np.vstack([poly, poly[0]])


def add_polygon_trace(
    fig: go.Figure,
    poly: np.ndarray,
    name: str,
    fill: str = "toself",
    opacity: float = 0.35,
    line_width: float = 2.0,
    visible: bool = True,
) -> None:
    closed = close_polygon(poly)
    fig.add_trace(
        go.Scatter(
            x=closed[:, 0],
            y=closed[:, 1],
            mode="lines",
            name=name,
            fill=fill,
            opacity=opacity,
            line=dict(width=line_width),
            visible=True if visible else "legendonly",
        )
    )



def build_figure(
    disk_pts: np.ndarray,
    r3_pts: np.ndarray,
    r5_pts: np.ndarray,
    hull: np.ndarray,
    area_hull: float,
    show_component_bodies: bool,
    show_sample_points: bool,
) -> go.Figure:
    fig = go.Figure()

    add_polygon_trace(
        fig,
        hull,
        f"Convex hull, sampled area ≈ {area_hull:.9f}",
        opacity=0.18,
        line_width=3.0,
    )
    if show_component_bodies:
        add_polygon_trace(fig, convex_hull(disk_pts), "Disk D", opacity=0.35)
        add_polygon_trace(fig, convex_hull(r3_pts), "Reuleaux triangle R₃", opacity=0.35)
        add_polygon_trace(fig, convex_hull(r5_pts), "Reuleaux pentagon R₅", opacity=0.35)

    if show_sample_points:
        for pts, name in [(disk_pts, "D samples"), (r3_pts, "R₃ samples"), (r5_pts, "R₅ samples")]:
            fig.add_trace(
                go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(size=2),
                    visible=True,
                )
            )

    fig.update_layout(
        title="D, R₃ and R₅ test-body configuration",
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=10, r=10, t=80, b=10),
        height=720,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    return fig


def main() -> None:
    st.set_page_config(page_title="Lebesgue D/R3/R5 test-body demo", layout="wide")
    init_param_state()

    st.title("Interactive D/R₃/R₅ test-body demo")
    st.markdown(
        "This demo visualizes how the disk D, the Reuleaux triangle R₃, and "
        "the Reuleaux pentagon R₅ can be placed in a common convex hull. "
        "Move the bodies to see how the sampled convex-hull area changes."
    )

    st.markdown(
        "The sampled area shown below is the area of the convex hull of the "
        "three displayed test bodies."
    )

    with st.sidebar:
        st.header("Configuration")
        if st.button("Reset", use_container_width=True):
            set_param_state(DEFAULT_Z)
            st.rerun()

        st.header("Parameters")
        ranges = DEMO_UI_RANGES

        st.subheader("Reuleaux triangle R₃")
        x3 = slider_with_exact_input("x3", "x₃: horizontal shift", ranges)
        y3 = slider_with_exact_input("y3", "y₃: vertical shift", ranges)

        st.subheader("Reuleaux pentagon R₅")
        psi5_deg = slider_with_exact_input("psi5_deg", "ψ₅: rotation, degrees", ranges)
        x5 = slider_with_exact_input("x5", "x₅: horizontal shift", ranges)
        y5 = slider_with_exact_input("y5", "y₅: vertical shift", ranges)

        st.header("Display")
        show_component_bodies = st.checkbox("Show component bodies", value=True)
        show_sample_points = st.checkbox("Show sampled boundary points", value=False)

    psi5 = math.radians(psi5_deg)
    theta3 = math.radians(R3_FIXED_ANGLE_DEG)

    disk_pts = sample_disk()
    r3_pts = sample_reuleaux_odd_ngon(
        3,
        width=WIDTH,
        pose=Pose(center=(x3, y3), angle_rad=theta3),
    )
    r5_pts = sample_reuleaux_odd_ngon(
        5,
        width=WIDTH,
        pose=Pose(center=(x5, y5), angle_rad=psi5),
    )

    all_pts = np.vstack([disk_pts, r3_pts, r5_pts])
    hull = convex_hull(all_pts)
    area_hull = polygon_area(hull)

    left, right = st.columns([3, 1])

    with left:
        fig = build_figure(
            disk_pts,
            r3_pts,
            r5_pts,
            hull,
            area_hull,
            show_component_bodies=show_component_bodies,
            show_sample_points=show_sample_points,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Current placement")
        st.code(
            "\n".join(
                [
                    f"R3 shift x = {x3:.9f}",
                    f"R3 shift y = {y3:.9f}",
                    f"R5 rotation = {psi5_deg:.9f} deg",
                    f"R5 shift x = {x5:.9f}",
                    f"R5 shift y = {y5:.9f}",
                ]
            ),
            language="text",
        )

        st.subheader("Sampled area")
        st.metric("area of convex hull", f"{area_hull:.9f}")
        st.caption("This value is computed from the displayed sampled boundary points.")

    st.divider()
    st.subheader("About this demo")
    st.markdown(ABOUT_DEMO)


if __name__ == "__main__":
    main()
