"""
Interactive visualization of the D/R3/R5 test-body method for deriving a
lower bound in Lebesgue's universal covering problem.

The app shows how three diameter-one test bodies—the disk D, the Reuleaux
triangle R3, and the Reuleaux pentagon R5—are placed in a common convex hull.
The displayed area is a floating-point sampled value used for visualization;
the rigorous lower bound is established separately by interval verification.

Run:
    streamlit run lebesgue_5_param_demo.py
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

R3_ANGLE_OFFSET_DEG = 0.0
R5_ANGLE_OFFSET_DEG = 0.0
BASE_VERTEX_ANGLE_DEG = 0.0

SAMPLES_PER_REULEAUX_ARC = 100
DISK_BOUNDARY_SAMPLES = 240

DEFAULT_Z: Dict[str, float] = {
    "psi3_deg": 0.0,
    "d3": 0.0,
    "alpha5_deg": 0.0,
    "x5": 0.0,
    "y5": 0.0,
}

# Representative parameter value for the 0.834 lower-bound example.
# It is the center of a certified leaf box used to illustrate the method,
# not a point whose sampled area is exactly 0.834.
REPRESENTATIVE_0834_Z: Dict[str, float] = {
    "psi3_deg": 105.625,
    "d3": 0.0078125,
    "alpha5_deg": 52.875,
    "x5": 0.01671875,
    "y5": 0.0075,
}

NORMAL_UI_RANGES = {
    "psi3_deg": (-180.0, 180.0, 0.5, "%.6f"),
    "d3": (0.0, 0.8, 0.002, "%.9f"),
    "alpha5_deg": (-180.0, 180.0, 0.5, "%.6f"),
    "x5": (-0.8, 0.8, 0.002, "%.9f"),
    "y5": (-0.8, 0.8, 0.002, "%.9f"),
}

PAPER_INFO = {
    "title": "[Manuscript title to be inserted]",
    "authors": "[Author names and affiliations to be inserted]",
    "correspondence": "[Corresponding author contact to be inserted]",
    "note": "This interactive page accompanies a manuscript on a D/R3/R5 test-body lower-bound method for Lebesgue's universal covering problem.",
}

REFERENCE_PLACEHOLDERS = [
    "[1] H. Lebesgue, original formulation of the universal covering problem. Full bibliographic data to be inserted.",
    "[2] Classical and modern surveys on Lebesgue's universal covering problem. Full bibliographic data to be inserted.",
    "[3] Previous best lower-bound and upper-bound constructions for universal covers. Full bibliographic data to be inserted.",
    "[4] Interval arithmetic / computer-assisted proof references relevant to the certificate. Full bibliographic data to be inserted.",
]

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
        )
    )


def build_figure(
    disk_pts: np.ndarray,
    r3_pts: np.ndarray,
    r5_pts: np.ndarray,
    hull: np.ndarray,
    area_hull: float,
) -> go.Figure:
    fig = go.Figure()

    add_polygon_trace(
        fig,
        hull,
        f"conv(D ∪ R₃ ∪ R₅), sampled area ≈ {area_hull:.9f}",
        opacity=0.18,
        line_width=3.0,
    )
    add_polygon_trace(fig, convex_hull(disk_pts), "D", opacity=0.35)
    add_polygon_trace(fig, convex_hull(r3_pts), "R₃", opacity=0.35)
    add_polygon_trace(fig, convex_hull(r5_pts), "R₅", opacity=0.35)

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
    st.set_page_config(page_title="Lebesgue lower-bound method demo", layout="wide")
    init_param_state()

    st.title("Interactive demonstration of the lower-bound method")
    st.markdown("This demo illustrates how the D, R₃ and R₅ test bodies are used to obtain a lower bound for Lebesgue's universal covering problem.")

    st.latex(
        r"A(z)=\operatorname{area}\operatorname{conv}\bigl(D\cup R_3(z)\cup R_5(z)\bigr),"
        r"\qquad z=(\psi_3,d_3,\alpha_5,x_5,y_5)."
    )

    st.markdown(
        "Here D is the radius-1/2 disk fixed at the origin. "
        "The parameters (ψ₃, d₃) determine the normalized placement of R₃, "
        "while (α₅, x₅, y₅) determine the placement of R₅. "
        "The displayed area is a floating-point sampled value; the certified lower bound "
        "is established separately by the interval certificate."
    )

    with st.sidebar:
        st.header("Configurations")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reset: z = 0", use_container_width=True):
                set_param_state(DEFAULT_Z)
                st.rerun()
        with col_b:
            if st.button("0.834", use_container_width=True):
                set_param_state(REPRESENTATIVE_0834_Z)
                st.rerun()

        st.header("Parameters")
        ranges = NORMAL_UI_RANGES

        st.subheader("R₃")
        psi3_deg = slider_with_exact_input("psi3_deg", "ψ₃: center direction, degrees", ranges)
        d3 = slider_with_exact_input("d3", "d₃: center distance", ranges)

        st.subheader("R₅")
        alpha5_deg = slider_with_exact_input("alpha5_deg", "α₅: orientation, degrees", ranges)
        x5 = slider_with_exact_input("x5", "x₅: center x-coordinate", ranges)
        y5 = slider_with_exact_input("y5", "y₅: center y-coordinate", ranges)

    psi3 = math.radians(psi3_deg)
    alpha5 = math.radians(alpha5_deg + R5_ANGLE_OFFSET_DEG)
    theta3 = math.radians(psi3_deg + R3_ANGLE_OFFSET_DEG)

    c3 = (d3 * math.cos(psi3), d3 * math.sin(psi3))
    c5 = (x5, y5)

    disk_pts = sample_disk()
    r3_pts = sample_reuleaux_odd_ngon(
        3,
        width=WIDTH,
        pose=Pose(center=c3, angle_rad=theta3),
    )
    r5_pts = sample_reuleaux_odd_ngon(
        5,
        width=WIDTH,
        pose=Pose(center=c5, angle_rad=alpha5),
    )

    all_pts = np.vstack([disk_pts, r3_pts, r5_pts])
    hull = convex_hull(all_pts)
    area_hull = polygon_area(hull)

    left, right = st.columns([3, 1])

    with left:
        fig = build_figure(disk_pts, r3_pts, r5_pts, hull, area_hull)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Current parameters")
        st.code(
            "\n".join(
                [
                    f"psi3   = {psi3_deg:.9f} deg",
                    f"d3     = {d3:.9f}",
                    f"alpha5 = {alpha5_deg:.9f} deg",
                    f"x5     = {x5:.9f}",
                    f"y5     = {y5:.9f}",
                ]
            ),
            language="text",
        )

        st.subheader("Sampled area")
        st.metric("area conv(D ∪ R₃ ∪ R₅)", f"{area_hull:.9f}")
        st.caption(
            "This is the sampled area at the displayed parameter value. "
            "It can be larger than 0.834 because 0.834 is the certified interval lower bound "
            "for the corresponding leaf box after all safety margins are included, not the sampled area at this center point."
        )

    st.divider()
    st.subheader("Manuscript information")
    st.markdown(f"**Title.** {PAPER_INFO['title']}")
    st.markdown(f"**Authors and affiliations.** {PAPER_INFO['authors']}")
    st.markdown(f"**Correspondence.** {PAPER_INFO['correspondence']}")
    st.markdown(PAPER_INFO["note"])

    st.subheader("References")
    for reference in REFERENCE_PLACEHOLDERS:
        st.markdown(reference)


if __name__ == "__main__":
    main()
