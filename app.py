"""
app.py — Precipitation Nowcasting Demo
════════════════════════════════════════════════════════════════════════════════
Causal-Mamba U-Net · SEVIR Dataset · t+30 min & t+60 min Forecasts

Run:
    streamlit run app.py

Requires:
    • model_arch.py  (in same directory)
    • A trained checkpoint  (best_model.pt  or  model.pt)
    • H5 event files  
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import io, math, os, tempfile, warnings
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Precipitation Nowcasting · Causal-Mamba",
    page_icon="🌩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"]  { font-size: 1.5rem; }
    .section-header               { font-size: 1.1rem; font-weight: 600;
                                    color: #4A90D9; margin: 0.4rem 0; }
    .severity-none   { background:#1e3a2f; color:#4ade80;  padding:6px 14px;
                       border-radius:8px; display:inline-block; font-weight:700; }
    .severity-light  { background:#1a3a4a; color:#60c8f0;  padding:6px 14px;
                       border-radius:8px; display:inline-block; font-weight:700; }
    .severity-mod    { background:#3a3010; color:#fbbf24;  padding:6px 14px;
                       border-radius:8px; display:inline-block; font-weight:700; }
    .severity-heavy  { background:#3a1010; color:#f87171;  padding:6px 14px;
                       border-radius:8px; display:inline-block; font-weight:700; }
    .metric-card     { background:#1a1a2e; border-radius:10px; padding:12px 16px;
                       margin:4px 0; border-left:3px solid #4A90D9; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════════
T_PAST        = 13
T_OUT         = 2
IMG_SIZE      = 128
TARGET_FRAMES = [5, 11]         # future indices → t+30, t+60

# Normalisation parameters (must match training)
VIL_MU,  VIL_SIG  = 3.5, 1.5   # (log1p(raw) - VIL_MU) / VIL_SIG
IR_SCALE          = 350.0       # raw / IR_SCALE

# Classification thresholds (in normalised VIL space)
THR_LIGHT  = -0.45
THR_MOD    =  0.54
THR_HEAVY  =  0.93

# Radar colorscale  (transparent → blue → green → yellow → red)
RADAR_COLORSCALE = [
    [0.00, "rgba(0,0,0,0)"],
    [0.10, "#0000FF"],
    [0.30, "#00AAFF"],
    [0.50, "#00FF00"],
    [0.65, "#FFFF00"],
    [0.80, "#FF8800"],
    [1.00, "#FF0000"],
]
MAGMA = "magma"


# ════════════════════════════════════════════════════════════════════════════════
# Data utilities
# ════════════════════════════════════════════════════════════════════════════════

def normalise_vil(raw: np.ndarray) -> np.ndarray:
    return (np.log1p(raw.astype(np.float32)) - VIL_MU) / VIL_SIG

def denormalise_vil(norm: np.ndarray) -> np.ndarray:
    return np.expm1(norm.astype(np.float32) * VIL_SIG + VIL_MU).clip(0)

def normalise_ir(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / IR_SCALE


def _read_h5_frames(h5file: h5py.File, key: str,
                     frame_indices: list | None = None) -> np.ndarray:
    """
    Robustly read frames from a SEVIR H5 dataset.

    Handles every layout we encounter:

    Layout A  (T, H, W)  — saved by generate_sample_data.py
                           AND by the corrected fetch_real_sevir_data.py
    Layout B  (H, W, T)  — raw SEVIR bulk files (time is LAST axis)
    Layout C  (N, H, W, T) — raw SEVIR bulk with event axis (takes [0])
    Layout D  (N, T, H, W) — rare alternative bulk layout (takes [0])

    Spatial dimensions are always padded/cropped to IMG_SIZE × IMG_SIZE.
    Returns float32 array shaped (T, IMG_SIZE, IMG_SIZE) or
    (len(frame_indices), IMG_SIZE, IMG_SIZE).
    """
    ds    = h5file[key]
    shape = ds.shape
    arr   = ds[:].astype(np.float32)

    # ── Step 1: collapse to 3-D (T, H, W) ────────────────────────────────
    if arr.ndim == 4:
        # (N, H, W, T)  or  (N, T, H, W)
        arr = arr[0]                          # drop event axis → 3-D
        shape = arr.shape

    if arr.ndim == 3:
        d0, d1, d2 = arr.shape
        # Decide axis order.  Time axis is the one with value ≤ 49.
        # Spatial axes of SEVIR are 128, 192, 208, 384 — always > 49.
        if d2 <= 49 and d0 > 49 and d1 > 49:
            # (H, W, T)  →  (T, H, W)
            arr = arr.transpose(2, 0, 1)
        elif d0 <= 49 and d1 > 49 and d2 > 49:
            pass   # already (T, H, W)
        else:
            # Ambiguous — assume first axis is time (our saved format)
            pass
    else:
        raise ValueError(f"Cannot interpret dataset '{key}' with shape {shape}")

    # ── Step 2: spatial normalisation to IMG_SIZE × IMG_SIZE ─────────────
    T, H, W = arr.shape

    if H != IMG_SIZE or W != IMG_SIZE:
        if H < IMG_SIZE or W < IMG_SIZE:
            # Pad (rare, but guard against it)
            ph = max(0, IMG_SIZE - H)
            pw = max(0, IMG_SIZE - W)
            arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="edge")
            T, H, W = arr.shape

        # Center-crop
        cy  = H // 2;  cx  = W // 2
        half = IMG_SIZE // 2
        arr = arr[:, cy - half: cy + half,
                     cx - half: cx + half]

    # ── Step 3: optional frame selection ─────────────────────────────────
    if frame_indices is not None:
        n_avail = arr.shape[0]
        safe    = [min(fi, n_avail - 1) for fi in frame_indices]
        arr     = arr[safe]

    return arr.astype(np.float32)


def load_event_from_h5(vil_path: str | Path, ir_path: str | Path) -> dict | None:
    """
    Load a single event from two H5 files.
    Handles both our own generated H5 files (past/future datasets)
    and the raw-style H5 files produced by fetch_real_sevir_data.py
    which may store everything under a single key or as (H,W,T) arrays.

    Returns a dict with keys: 'vil_past', 'vil_future', 'ir_past'
    All arrays in RAW (physical) units, shape (T, H, W).
    """
    try:
        with h5py.File(vil_path, "r") as vf, h5py.File(ir_path, "r") as irf:
            # ── Dataset key discovery ──────────────────────────────────────
            def _keys(f): return list(f.keys())

            vil_keys = _keys(vf)
            ir_keys  = _keys(irf)

            # ── Case A: files have 'past' and 'future' datasets (our format)
            if "past" in vil_keys and "future" in vil_keys:
                vil_past   = _read_h5_frames(vf, "past")
                vil_future = _read_h5_frames(vf, "future",
                                              frame_indices=list(TARGET_FRAMES))
                ir_past    = _read_h5_frames(irf, "past")

            # ── Case B: single dataset key == modality name (raw SEVIR style)
            elif "vil" in vil_keys:
                all_vil    = _read_h5_frames(vf,  "vil")     # (T, H, W)
                all_ir     = _read_h5_frames(irf, "ir069")
                # split: first 13 = past, rest = future
                vil_past   = all_vil[:T_PAST]
                ir_past    = all_ir[:T_PAST]
                future_arr = all_vil[T_PAST:]
                # safe indexing: clamp TARGET_FRAMES to available future length
                n_fut = future_arr.shape[0]
                safe_idx   = [min(f, n_fut - 1) for f in TARGET_FRAMES]
                vil_future = future_arr[safe_idx]

            # ── Case C: first key is used for the data (flexible fallback)
            else:
                main_vil_key = vil_keys[0]
                main_ir_key  = ir_keys[0]
                all_vil  = _read_h5_frames(vf,  main_vil_key)
                all_ir   = _read_h5_frames(irf, main_ir_key)
                vil_past   = all_vil[:T_PAST]
                ir_past    = all_ir[:T_PAST]
                future_arr = all_vil[T_PAST:]
                n_fut      = future_arr.shape[0]
                safe_idx   = [min(f, n_fut - 1) for f in TARGET_FRAMES]
                vil_future = future_arr[safe_idx]

        # Final shape guard — vil_future must be (2, H, W)
        assert vil_future.shape == (2, IMG_SIZE, IMG_SIZE), \
            f"vil_future shape {vil_future.shape} ≠ (2,{IMG_SIZE},{IMG_SIZE})"

        return {"vil_past": vil_past, "vil_future": vil_future,
                "ir_past": ir_past,
                "_vil_keys": vil_keys, "_ir_keys": ir_keys}

    except Exception as e:
        st.error(f"Failed to read H5 files: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def load_event_from_bytes(vil_bytes: bytes, ir_bytes: bytes) -> dict | None:
    """Load event from in-memory bytes (file-upload path)."""
    try:
        # Write to named temp files so h5py can open them reliably
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as vf_tmp, \
             tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as ir_tmp:
            vf_tmp.write(vil_bytes);  vf_path = vf_tmp.name
            ir_tmp.write(ir_bytes);   ir_path = ir_tmp.name
        result = load_event_from_h5(vf_path, ir_path)
        os.unlink(vf_path); os.unlink(ir_path)
        return result
    except Exception as e:
        st.error(f"Failed to parse uploaded files: {e}")
        return None


# ────────────────── Synthetic demo event ─────────────────────────────────────
def _gaussian_blob(H, W, cy, cx, sigma, amp):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return amp * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))


def generate_synthetic_event(seed: int = 0, severity: str = "moderate") -> dict:
    """Generate a synthetic storm event without any real data."""
    rng = np.random.default_rng(seed)
    H = W = IMG_SIZE
    n_cells = rng.integers(2, 5)
    cells = []
    for _ in range(n_cells):
        peak = {"light": 50.0, "moderate": 110.0, "heavy": 200.0}.get(severity, 110.0)
        cells.append(dict(
            cy=rng.uniform(20, H-20), cx=rng.uniform(20, W-20),
            vy=rng.uniform(-0.5, 0.5), vx=rng.uniform(-0.8, 1.2),
            sigma=rng.uniform(8, 18),
            peak=rng.uniform(peak * 0.6, peak),
            grow=rng.uniform(-0.01, 0.04),
        ))

    total = T_PAST + max(TARGET_FRAMES) + 1   # 13+12 = 25
    vil_seq = np.zeros((total, H, W), np.float32)
    ir_seq  = np.full((total, H, W), 280.0, np.float32)

    for t in range(total):
        for c in cells:
            cy = c["cy"] + c["vy"] * t
            cx = c["cx"] + c["vx"] * t
            sg = max(3.0, c["sigma"] * (1 + c["grow"] * t))
            decay = max(0.0, 1 - max(0, t - 18) * 0.08)
            amp_v  = c["peak"] * decay
            amp_ir = (210.0 - 280.0) * (amp_v / 200.0) * 0.9
            vil_seq[t] += _gaussian_blob(H, W, cy, cx, sg, amp_v)
            ir_seq[t]  += _gaussian_blob(H, W, cy, cx, sg * 1.3, amp_ir)
        vil_seq[t] += rng.uniform(0, 4, (H, W)).astype(np.float32)
        ir_seq[t]  += rng.uniform(-2, 2, (H, W)).astype(np.float32)

    vil_seq = np.clip(vil_seq, 0, 255)
    ir_seq  = np.clip(ir_seq, 190, 310)

    return {
        "vil_past":   vil_seq[:T_PAST],
        "vil_future": vil_seq[[T_PAST + f for f in TARGET_FRAMES]],
        "ir_past":    ir_seq[:T_PAST],
    }


def build_model_input(event: dict) -> torch.Tensor:
    """
    Pre-process an event dict into a model-ready tensor.
    Returns: (1, T_PAST, 2, IMG_SIZE, IMG_SIZE)  float32
    """
    vp = normalise_vil(event["vil_past"])   # (T, H, W)
    ip = normalise_ir(event["ir_past"])     # (T, H, W)

    def _resize(arr):
        t = torch.from_numpy(arr).unsqueeze(1)         # T 1 H W
        if t.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            t = F.interpolate(t, (IMG_SIZE, IMG_SIZE),
                              mode="bilinear", align_corners=False)
        return t                                       # T 1 H W

    pv = _resize(vp)   # T 1 H W
    pi = _resize(ip)   # T 1 H W
    x  = torch.cat([pv, pi], dim=1)    # T 2 H W
    return x.unsqueeze(0)              # 1 T 2 H W


# ════════════════════════════════════════════════════════════════════════════════
# Model loading (cached across Streamlit re-runs)
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_cached_model(path: str, device: str):
    from model_arch import load_model
    return load_model(path, device)


# ════════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred_norm: np.ndarray,
                    gt_norm:   np.ndarray) -> dict:
    """
    pred_norm, gt_norm: (H, W) normalised VIL for one horizon.
    Returns a dict of scalar metrics.
    """
    diff = pred_norm - gt_norm
    mse  = float((diff ** 2).mean())
    mae  = float(np.abs(diff).mean())
    rmse = float(np.sqrt(mse))
    bias = float(diff.mean())

    def _csi(p, t, thr):
        pp, pt = p > thr, t > thr
        tp = int((pp & pt).sum())
        fp = int((pp & ~pt).sum())
        fn = int((~pp & pt).sum())
        csi = tp / (tp + fp + fn + 1e-9)
        pod = tp / (tp + fn + 1e-9)
        far = fp / (tp + fp + 1e-9)
        return csi, pod, far

    csi_l, pod_l, far_l = _csi(pred_norm, gt_norm, THR_LIGHT)
    csi_m, pod_m, far_m = _csi(pred_norm, gt_norm, THR_MOD)
    csi_h, pod_h, far_h = _csi(pred_norm, gt_norm, THR_HEAVY)

    return dict(mse=mse, mae=mae, rmse=rmse, bias=bias,
                csi_light=csi_l, pod_light=pod_l,
                csi_mod=csi_m,   pod_mod=pod_m,
                csi_heavy=csi_h, pod_heavy=pod_h)


def severity_label(max_phys: float) -> tuple[str, str]:
    """Return (label, css_class) based on peak physical VIL (kg/m²)."""
    if max_phys > 133:
        return "⛈️  SEVERE / HEAVY RAIN", "severity-heavy"
    elif max_phys > 74:
        return "🌧️  MODERATE RAIN",        "severity-mod"
    elif max_phys > 17:
        return "🌦️  LIGHT RAIN",            "severity-light"
    else:
        return "🌤️  NO SIGNIFICANT PRECIP", "severity-none"


# ════════════════════════════════════════════════════════════════════════════════
# Plotly visualisation helpers
# ════════════════════════════════════════════════════════════════════════════════

def _heatmap_trace(data: np.ndarray, name: str = "",
                   colorscale: str = MAGMA,
                   zmin: float = -1.2, zmax: float = 2.5,
                   showscale: bool = True,
                   colorbar_x: float = 1.01):
    return go.Heatmap(
        z=data[::-1],
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        showscale=showscale,
        colorbar=dict(
            thickness=10, len=0.45, x=colorbar_x,
            tickfont=dict(size=9), title=dict(text="VIL", font=dict(size=9)),
            outlinewidth=0,
        ),
        hovertemplate="VIL: %{z:.2f}<extra></extra>",
        name=name,
    )


def plot_input_animation(vil_past_raw: np.ndarray,
                          show_last_n: int = 6) -> go.Figure:
    """
    Animated Plotly heatmap of the last `show_last_n` past VIL frames.
    Buttons are positioned above the plot to avoid overlap with the slider.
    """
    T  = vil_past_raw.shape[0]
    frames_to_show = vil_past_raw[max(0, T - show_last_n):]
    n   = frames_to_show.shape[0]
    norm = normalise_vil(frames_to_show)

    frames = []
    for i in range(n):
        frames.append(go.Frame(
            data=[_heatmap_trace(norm[i], showscale=True)],
            name=str(i),
        ))

    fig = go.Figure(
        data=[_heatmap_trace(norm[0])],
        frames=frames,
        layout=go.Layout(
            height=360,
            # top margin gives room for buttons; bottom for slider
            margin=dict(l=5, r=55, t=55, b=60),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False,
                       zeroline=False, scaleanchor="x"),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            # ── Buttons sit in the top-left, ABOVE the axes ──────────────
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=0.0, xanchor="left",
                y=1.12, yanchor="top",   # above the plot area
                bgcolor="#1e2130",
                bordercolor="#444",
                font=dict(size=11),
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 700, "redraw": True},
                                      "fromcurrent": True,
                                      "transition": {"duration": 150}}]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0},
                                        "mode": "immediate"}]),
                ],
            )],
            # ── Slider sits below the axes, no overlap ────────────────────
            sliders=[dict(
                steps=[dict(method="animate",
                             args=[[str(i)], {"mode": "immediate",
                                              "frame": {"duration": 0,
                                                        "redraw": True}}],
                             label=f"t−{(n-1-i)*5}m") for i in range(n)],
                active=0,
                currentvalue=dict(
                    prefix="Showing: ",
                    font=dict(size=11, color="white"),
                    xanchor="left",
                ),
                pad=dict(t=10, b=5, l=0, r=0),
                x=0.0, len=1.0,
                bgcolor="#1e2130",
                bordercolor="#444",
                tickcolor="white",
                font=dict(size=9, color="white"),
            )],
        )
    )
    return fig


def plot_comparison_grid(
        pred_t30: np.ndarray, pred_t60: np.ndarray,
        gt_t30:   np.ndarray, gt_t60:   np.ndarray,
) -> go.Figure:
    """
    2×3 grid: rows = horizons (t+30, t+60), cols = Pred | GT | Diff
    One VIL colorbar on the far right of col-2; one Δ colorbar for col-3.
    Colorbars are anchored with explicit x positions so they never overlap.
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Prediction (t+30)", "Ground Truth (t+30)", "Difference (t+30)",
            "Prediction (t+60)", "Ground Truth (t+60)", "Difference (t+60)",
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.14,
    )

    # VIL colorbar: shown once, anchored just right of col-2 (~x=0.66)
    # Diff colorbar: shown once, anchored right of col-3 (~x=1.0)
    for row, (pred, gt) in enumerate([(pred_t30, gt_t30), (pred_t60, gt_t60)], 1):
        diff = pred - gt

        # col-1: prediction — no colorbar (shared with col-2)
        fig.add_trace(go.Heatmap(
            z=pred[::-1], colorscale=MAGMA,
            zmin=-1.2, zmax=2.5, showscale=False,
            hovertemplate="VIL: %{z:.2f}<extra></extra>",
        ), row=row, col=1)

        # col-2: ground truth — one shared VIL colorbar, shown only row-1
        fig.add_trace(go.Heatmap(
            z=gt[::-1], colorscale=MAGMA,
            zmin=-1.2, zmax=2.5,
            showscale=(row == 1),
            colorbar=dict(
                x=0.655, y=0.78, len=0.42,   # top-right of col-2
                thickness=10, outlinewidth=0,
                title=dict(text="VIL", font=dict(size=9)),
                tickfont=dict(size=8),
            ),
            hovertemplate="VIL: %{z:.2f}<extra></extra>",
        ), row=row, col=2)

        # col-3: difference — one shared diff colorbar, shown only row-1
        fig.add_trace(go.Heatmap(
            z=diff[::-1], colorscale="RdBu_r",
            zmid=0, zmin=-1.0, zmax=1.0,
            showscale=(row == 1),
            colorbar=dict(
                x=1.01, y=0.78, len=0.42,    # far right
                thickness=10, outlinewidth=0,
                title=dict(text="Δ", font=dict(size=9)),
                tickfont=dict(size=8),
            ),
            hovertemplate="Δ: %{z:.2f}<extra></extra>",
        ), row=row, col=3)

    fig.update_layout(
        height=480,
        margin=dict(l=5, r=60, t=55, b=5),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white",
        font=dict(size=10),
    )
    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig.layout[axis].update(showticklabels=False, showgrid=False,
                                     zeroline=False)
    return fig


def plot_coverage_bars(pred_t30, pred_t60, gt_t30, gt_t60) -> go.Figure:
    """
    Grouped bar chart: % of pixels in each rain category (None/Light/Mod/Heavy)
    for Prediction vs Ground Truth at both horizons.
    Much cleaner than a histogram — directly answers "did the model get coverage right?"
    """
    categories = ["No Rain", "Light", "Moderate", "Heavy"]
    thresholds = [(-99, THR_LIGHT), (THR_LIGHT, THR_MOD),
                  (THR_MOD, THR_HEAVY), (THR_HEAVY, 99)]
    colors_pred = ["#374151", "#60a5fa", "#facc15", "#f87171"]
    colors_gt   = ["#1f2937", "#1d4ed8", "#b45309", "#b91c1c"]

    def _pct(arr, lo, hi):
        return float(((arr > lo) & (arr <= hi)).sum()) / arr.size * 100

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["t +30 min", "t +60 min"],
        shared_yaxes=True,
    )

    for col, (pred, gt, label) in enumerate(
            [(pred_t30, gt_t30, "t+30"), (pred_t60, gt_t60, "t+60")], 1):

        pred_pcts = [_pct(pred, lo, hi) for lo, hi in thresholds]
        gt_pcts   = [_pct(gt,   lo, hi) for lo, hi in thresholds]

        for i, cat in enumerate(categories):
            fig.add_trace(go.Bar(
                name=f"Pred – {cat}",
                x=[cat], y=[pred_pcts[i]],
                marker_color=colors_pred[i],
                legendgroup=f"pred_{i}",
                showlegend=(col == 1),
                text=f"{pred_pcts[i]:.1f}%",
                textposition="outside",
                textfont=dict(size=9),
            ), row=1, col=col)
            fig.add_trace(go.Bar(
                name=f"GT – {cat}",
                x=[cat], y=[gt_pcts[i]],
                marker_color=colors_gt[i],
                legendgroup=f"gt_{i}",
                showlegend=(col == 1),
                text=f"{gt_pcts[i]:.1f}%",
                textposition="outside",
                textfont=dict(size=9),
            ), row=1, col=col)

    fig.update_layout(
        height=320,
        barmode="group",
        bargap=0.25, bargroupgap=0.05,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", font=dict(size=11),
        yaxis_title="% pixels",
        legend=dict(
            orientation="v", x=1.02, y=1,
            font=dict(size=9), bgcolor="rgba(0,0,0,0)",
            tracegroupgap=2,
        ),
        showlegend=True,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a3a", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Interpretation / Decision logic
# ════════════════════════════════════════════════════════════════════════════════

def _coverage_pct(arr: np.ndarray, thr: float) -> float:
    return float((arr > thr).sum()) / arr.size * 100


def _centroid(arr: np.ndarray, thr: float):
    mask = arr > thr
    if mask.sum() == 0:
        return None
    rows, cols = np.where(mask)
    return float(rows.mean()) / arr.shape[0], float(cols.mean()) / arr.shape[1]


def _estimate_motion(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[float, float]:
    """Phase-correlation to estimate pixel shift between two frames."""
    fa = np.fft.fft2(frame_a)
    fb = np.fft.fft2(frame_b)
    cross = fa * np.conj(fb)
    cross /= (np.abs(cross) + 1e-8)
    corr  = np.fft.ifft2(cross).real
    H, W  = corr.shape
    corr  = np.roll(corr, H // 2, axis=0)
    corr  = np.roll(corr, W // 2, axis=1)
    peak  = np.unravel_index(corr.argmax(), corr.shape)
    dy    = peak[0] - H // 2
    dx    = peak[1] - W // 2
    return float(dy), float(dx)


def _compass(dy: float, dx: float) -> str:
    angle = math.degrees(math.atan2(-dy, dx)) % 360
    dirs  = ["E","NE","N","NW","W","SW","S","SE","E"]
    return dirs[round(angle / 45) % 8]


def _describe_frame(phys_arr: np.ndarray) -> dict:
    """
    Given a physical-VIL 2-D array, return a concise description dict:
        severity_label, severity_class, peak, coverage per category,
        dominant_category, centroid.
    """
    peak = float(phys_arr.max())
    total = phys_arr.size

    no_rain_pct  = float((phys_arr <  17).sum())  / total * 100
    light_pct    = float(((phys_arr >= 17)  & (phys_arr <  74)).sum()) / total * 100
    mod_pct      = float(((phys_arr >= 74)  & (phys_arr < 133)).sum()) / total * 100
    heavy_pct    = float( (phys_arr >= 133).sum()) / total * 100

    # Dominant category by covered area (ignoring no-rain background)
    cats = {"Light": light_pct, "Moderate": mod_pct, "Heavy": heavy_pct}
    dom  = max(cats, key=cats.get) if max(cats.values()) > 0.5 else "No significant rain"

    sev_lbl, sev_cls = severity_label(peak)

    return dict(
        peak=peak,
        no_rain_pct=no_rain_pct,
        light_pct=light_pct,
        mod_pct=mod_pct,
        heavy_pct=heavy_pct,
        dominant=dom,
        sev_label=sev_lbl,
        sev_class=sev_cls,
    )


def build_interpretation(event: dict,
                          pred_t30: np.ndarray,
                          pred_t60: np.ndarray,
                          gt_t30:   np.ndarray,
                          gt_t60:   np.ndarray) -> dict:
    """
    Analyse predictions, ground truth and observed frames.
    Every text field is derived from the actual computed numbers.
    """
    last_obs = normalise_vil(event["vil_past"][-1])

    # ── Storm motion: average over last 3 frame-pairs for stability ───────────
    shifts = []
    for k in range(-3, 0):
        fa = normalise_vil(event["vil_past"][k - 1])
        fb = normalise_vil(event["vil_past"][k])
        dy_k, dx_k = _estimate_motion(fa, fb)
        shifts.append((dy_k, dx_k))
    dy = float(np.median([s[0] for s in shifts]))
    dx = float(np.median([s[1] for s in shifts]))
    speed_pxf = math.sqrt(dy**2 + dx**2)
    # 1 km/pixel → speed in km per 5-min frame → km/h
    speed_kmh = speed_pxf * 1.0 * 12     # 12 × (km/frame) = km/h
    motion_reliable = speed_pxf > 0.5    # flag if motion is near-zero
    direction = _compass(dy, dx) if motion_reliable else "—"

    # ── Physical VIL values ───────────────────────────────────────────────────
    max_phys_obs = float(denormalise_vil(last_obs).max())
    max_phys_t30 = float(denormalise_vil(pred_t30).max())
    max_phys_t60 = float(denormalise_vil(pred_t60).max())

    # ── Amplification: how much does the model change peak vs observed? ───────
    amp_ratio = max_phys_t30 / (max_phys_obs + 1e-3)
    if amp_ratio > 1.3:
        amp_flag = f"⚠️ Model over-predicts peak by {(amp_ratio-1)*100:.0f}% vs observed"
    elif amp_ratio < 0.7:
        amp_flag = f"ℹ️ Model under-predicts peak by {(1-amp_ratio)*100:.0f}% vs observed"
    else:
        amp_flag = ""

    # ── Intensity trend: compare t+60 vs t+30 (predicted evolution) ──────────
    if max_phys_t60 > max_phys_t30 * 1.10:
        intensity_trend = "intensifying ↑"
    elif max_phys_t60 < max_phys_t30 * 0.90:
        intensity_trend = "weakening ↓"
    else:
        intensity_trend = "steady →"

    # ── Coverage (% pixels above moderate threshold) ──────────────────────────
    cov_obs = _coverage_pct(last_obs, THR_MOD)
    cov_t30 = _coverage_pct(pred_t30, THR_MOD)
    cov_t60 = _coverage_pct(pred_t60, THR_MOD)

    # Coverage change flag
    cov_ratio = cov_t30 / (cov_obs + 1e-3)
    if cov_ratio > 1.5:
        cov_flag = f"⚠️ Predicted coverage is {cov_ratio:.1f}× larger than observed — possible over-spread"
    elif cov_ratio < 0.5:
        cov_flag = f"ℹ️ Predicted coverage is {cov_ratio:.1f}× observed — possible under-spread"
    else:
        cov_flag = ""

    # ── Severity label (based on observed, not predicted peak) ───────────────
    sev_label, sev_class = severity_label(max_phys_obs)

    # ── Predicted centroid displacement ───────────────────────────────────────
    c_obs = _centroid(last_obs, THR_MOD)
    c_t60 = _centroid(pred_t60, THR_MOD)
    position_note = ""
    if c_obs and c_t60 and motion_reliable:
        drow = (c_t60[0] - c_obs[0]) * IMG_SIZE   # km (1 km/px)
        dcol = (c_t60[1] - c_obs[1]) * IMG_SIZE
        dist = math.sqrt(drow**2 + dcol**2)
        if dist > 2:
            position_note = (f"Predicted centroid shifts ≈ **{dist:.0f} km** "
                             f"to the **{direction}** by t+60.")

    # ── Ground-truth descriptions ─────────────────────────────────────────────
    gt_t30_desc = _describe_frame(denormalise_vil(gt_t30))
    gt_t60_desc = _describe_frame(denormalise_vil(gt_t60))

    # GT vs Pred agreement check
    def _agree(pred_d, gt_d):
        """Were prediction and GT in the same severity tier?"""
        return pred_d["sev_class"] == gt_d["sev_class"]

    gt_t30_agree = _agree(
        _describe_frame(denormalise_vil(pred_t30)), gt_t30_desc)
    gt_t60_agree = _agree(
        _describe_frame(denormalise_vil(pred_t60)), gt_t60_desc)

    return dict(
        direction        = direction,
        speed_kmh        = speed_kmh,
        motion_reliable  = motion_reliable,
        max_phys_obs     = max_phys_obs,
        max_phys_t30     = max_phys_t30,
        max_phys_t60     = max_phys_t60,
        amp_flag         = amp_flag,
        cov_obs          = cov_obs,
        cov_t30          = cov_t30,
        cov_t60          = cov_t60,
        cov_flag         = cov_flag,
        intensity_trend  = intensity_trend,
        sev_label        = sev_label,
        sev_class        = sev_class,
        position_note    = position_note,
        gt_t30           = gt_t30_desc,
        gt_t60           = gt_t60_desc,
        gt_t30_agree     = gt_t30_agree,
        gt_t60_agree     = gt_t60_agree,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar controls. Returns a config dict."""
    st.sidebar.title("⚙️  Controls")

    # ── Model ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🧠 Model")
    use_model = st.sidebar.checkbox("Load trained model", value=True,
                                     help="Uncheck to run a persistence baseline instead")
    model_path = ""
    if use_model:
        model_path = st.sidebar.text_input(
            "Checkpoint path",
            value="best_model.pt",
            help="Path to best_model.pt or model.pt",
        )

    device_opts = ["cpu"]
    if torch.cuda.is_available():
        device_opts.insert(0, "cuda")
    device = st.sidebar.selectbox("Device", device_opts)

    # ── Data source ───────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📂 Data")
    data_mode = st.sidebar.radio(
        "Source",
        ["Synthetic demo", "Local H5 directory", "Upload H5 pair"],
        index=0,
    )

    data_dir    = None
    event_id    = None
    vil_upload  = None
    ir_upload   = None
    severity    = "moderate"
    seed_val    = 0

    if data_mode == "Synthetic demo":
        severity = st.sidebar.selectbox("Storm severity",
                                          ["light", "moderate", "heavy"], index=1)
        seed_val = st.sidebar.slider("Random seed", 0, 99, 0)

    elif data_mode == "Local H5 directory":
        data_dir = st.sidebar.text_input(
            "Data directory", value="sample_data",
            help="Works with generate_sample_data.py OR fetch_real_sevir_data.py output.")
        if data_dir and Path(data_dir).is_dir():
            vil_files = sorted(Path(data_dir).glob("*_vil.h5"))
            ids = [f.stem.replace("_vil", "") for f in vil_files]
            if ids:
                event_id = st.sidebar.selectbox("Event ID", ids)
                with st.sidebar.expander("\U0001f50d Inspect H5 structure", expanded=False):
                    try:
                        vp = Path(data_dir) / f"{event_id}_vil.h5"
                        with h5py.File(vp, "r") as f:
                            for k in f.keys():
                                st.code(f"\'{k}\': shape={f[k].shape}  dtype={f[k].dtype}")
                    except Exception as e:
                        st.error(str(e))
            else:
                st.sidebar.warning("No *_vil.h5 files found.")
                st.sidebar.caption(
                    "Run `python fetch_real_sevir_data.py` or "
                    "`python generate_sample_data.py` to create files.")

    elif data_mode == "Upload H5 pair":
        st.sidebar.caption("Upload the *_vil.h5 and *_ir069.h5 for one event.")
        vil_upload = st.sidebar.file_uploader("VIL H5 file",  type=["h5","hdf5"])
        ir_upload  = st.sidebar.file_uploader("IR069 H5 file", type=["h5","hdf5"])
        if vil_upload:
            import tempfile, os
            with st.sidebar.expander("\U0001f50d VIL structure", expanded=True):
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                    tmp.write(vil_upload.read()); tpath = tmp.name
                vil_upload.seek(0)
                with h5py.File(tpath, "r") as f:
                    for k in f.keys():
                        st.code(f"\'{k}\': shape={f[k].shape}  dtype={f[k].dtype}")
                os.unlink(tpath)

    # ── Visualisation ─────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🎨 Visualisation")
    show_last_n = st.sidebar.slider("Past frames shown", 3, T_PAST, 6)

    return dict(
        use_model=use_model, model_path=model_path, device=device,
        data_mode=data_mode, data_dir=data_dir, event_id=event_id,
        vil_upload=vil_upload, ir_upload=ir_upload,
        severity=severity, seed_val=seed_val,
        show_last_n=show_last_n,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Main app
# ════════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🌩️ Precipitation Nowcasting")
    st.markdown(
        "**Causal-Mamba U-Net**  ·  SEVIR NEXRAD VIL + IR069  ·  "
        "t+30 min & t+60 min forecasts  |  128 × 128 px  (~1 km/pixel, cropped from 384×384)"
    )

    # ── Localtunnel / ngrok JS-fetch warning (Colab users) ────────────────────
    import os
    if os.environ.get("COLAB_BACKEND_URL") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        st.info(
            "**Running via tunnel (Colab/Kaggle)?**  "
            "If you see *'Failed to fetch dynamically imported module'* errors "
            "in the sidebar, your browser is blocking the tunnel URL. "
            "Fix: open the tunnel URL directly in a new tab first "
            "(click through the localtunnel password page), then reload this tab. "
            "For ngrok no extra step is needed."
        )

    st.divider()
    cfg = render_sidebar()

    # ── Load model ────────────────────────────────────────────────────────────
    model      = None
    model_info = ""
    if cfg["use_model"]:
        mp = cfg["model_path"]
        if mp and Path(mp).is_file():
            with st.spinner("Loading model …"):
                try:
                    model = _load_cached_model(mp, cfg["device"])
                    n_params = sum(p.numel() for p in model.parameters())
                    model_info = (f"✅ **Model loaded** · {n_params/1e6:.2f} M params · "
                                  f"device: `{cfg['device']}`")
                except Exception as e:
                    st.error(f"Could not load model: {e}")
                    model = None
        elif mp:
            st.sidebar.warning(f"File not found: `{mp}`")

    if model is None and cfg["use_model"]:
        st.info("ℹ️  Model not loaded — running **persistence baseline** (last observed frame).")
    elif model_info:
        st.sidebar.success(model_info)

    # ── Load event ────────────────────────────────────────────────────────────
    event = None

    if cfg["data_mode"] == "Synthetic demo":
        event = generate_synthetic_event(seed=cfg["seed_val"], severity=cfg["severity"])
        st.sidebar.success(f"🔬 Synthetic event · severity={cfg['severity']}")

    elif cfg["data_mode"] == "Local H5 directory":
        if cfg["event_id"] and cfg["data_dir"]:
            d  = Path(cfg["data_dir"])
            vp = d / f"{cfg['event_id']}_vil.h5"
            ip = d / f"{cfg['event_id']}_ir069.h5"
            if vp.is_file() and ip.is_file():
                event = load_event_from_h5(vp, ip)
            else:
                st.error(f"Missing files for event `{cfg['event_id']}`")

    elif cfg["data_mode"] == "Upload H5 pair":
        if cfg["vil_upload"] and cfg["ir_upload"]:
            # Rewind — sidebar debug expander may have consumed the buffer
            cfg["vil_upload"].seek(0)
            cfg["ir_upload"].seek(0)
            event = load_event_from_bytes(
                cfg["vil_upload"].read(), cfg["ir_upload"].read()
            )

    if event is None:
        st.info("👈  Select a data source in the sidebar to begin.")
        _render_architecture_diagram()
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    x_tensor = build_model_input(event)

    if model is not None:
        with torch.no_grad():
            x_in = x_tensor.to(cfg["device"])
            y    = model(x_in).cpu().numpy()      # (1, 2, 1, H, W)
        pred_t30 = y[0, 0, 0]
        pred_t60 = y[0, 1, 0]
        method = "Causal-Mamba"
    else:
        # Persistence: last observed VIL frame
        last_norm = normalise_vil(event["vil_past"][-1])
        pred_t30  = last_norm
        pred_t60  = last_norm
        method    = "Persistence"

    gt_t30 = normalise_vil(event["vil_future"][0])
    gt_t60 = normalise_vil(event["vil_future"][1])

    # ── Compute metrics once ─────────────────────────────────────────────────
    m30 = compute_metrics(pred_t30, gt_t30)
    m60 = compute_metrics(pred_t60, gt_t60)

    # ── Layout: two columns (input | comparison), stats below ────────────────
    col1, col2 = st.columns([1, 1.6], gap="medium")

    with col1:
        st.markdown('<p class="section-header">📡 Observed Radar (last frames)</p>',
                    unsafe_allow_html=True)
        fig_input = plot_input_animation(event["vil_past"],
                                          show_last_n=cfg["show_last_n"])
        st.plotly_chart(fig_input, use_container_width=True, key="input_anim")

    with col2:
        st.markdown(f'<p class="section-header">🔮 Forecast vs. Ground Truth ({method})</p>',
                    unsafe_allow_html=True)
        fig_comp = plot_comparison_grid(pred_t30, pred_t60, gt_t30, gt_t60)
        st.plotly_chart(fig_comp, use_container_width=True, key="comparison")

    # ── Key metrics as a clean inline row ────────────────────────────────────
    st.markdown("##### 📊 Key Scores")
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("RMSE t+30",     f"{m30['rmse']:.3f}")
    mc2.metric("RMSE t+60",     f"{m60['rmse']:.3f}")
    mc3.metric("CSI-mod t+30",  f"{m30['csi_mod']:.3f}")
    mc4.metric("CSI-mod t+60",  f"{m60['csi_mod']:.3f}")
    mc5.metric("POD-heavy t+30",f"{m30['pod_heavy']:.3f}")
    mc6.metric("POD-heavy t+60",f"{m60['pod_heavy']:.3f}")

    st.divider()

    # ── Two tabs only ─────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs([
        "🌧️ Rain Coverage",
        "🧠 Storm Analysis",
    ])

    with tab1:
        fig_cov = plot_coverage_bars(pred_t30, pred_t60, gt_t30, gt_t60)
        st.plotly_chart(fig_cov, use_container_width=True, key="coverage")
        st.caption(
            "Each bar shows what % of the 128×128 grid falls into that rain category. "
            "**Light** > 17 kg/m²  ·  **Moderate** > 74 kg/m²  ·  **Heavy** > 133 kg/m²  "
            "· Lighter shade = Prediction, darker = Ground Truth."
        )

    with tab2:
        interp = build_interpretation(event, pred_t30, pred_t60, gt_t30, gt_t60)
        _render_interpretation(interp, m30, m60, method)


# ════════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ════════════════════════════════════════════════════════════════════════════════

def _render_metric_table(m30: dict, m60: dict):
    import pandas as pd
    rows = []
    for name, key in [
        ("MAE ↓",         "mae"),
        ("RMSE ↓",        "rmse"),
        ("Bias",          "bias"),
        ("CSI  Light ↑",  "csi_light"),
        ("CSI  Mod   ↑",  "csi_mod"),
        ("CSI  Heavy ↑",  "csi_heavy"),
        ("POD  Light ↑",  "pod_light"),
        ("POD  Mod   ↑",  "pod_mod"),
        ("POD  Heavy ↑",  "pod_heavy"),
    ]:
        rows.append({"Metric": name,
                     "t+30 min": round(m30[key], 4),
                     "t+60 min": round(m60[key], 4)})
    df = pd.DataFrame(rows).set_index("Metric")
    st.dataframe(df.style.format("{:.4f}").background_gradient(
        cmap="RdYlGn", subset=["t+30 min", "t+60 min"]
    ), use_container_width=True)


def _render_interpretation(interp: dict, m30: dict, m60: dict,
                             method: str):
    """
    Data-driven storm analysis panel.
    Every sentence is derived from actual computed values — no hardcoded phrases.
    """

    # ── Severity badge ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="{interp['sev_class']}" style="font-size:1.1rem; margin-bottom:10px">
        {interp['sev_label']}
        <span style="font-size:0.85rem; font-weight:400; margin-left:12px; opacity:0.8">
        based on last observed frame · {interp['max_phys_obs']:.0f} kg/m² peak VIL
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Ground Truth Status ───────────────────────────────────────────────────
    st.markdown("#### 🌍 What Actually Happened (Ground Truth)")

    def _gt_badge(d: dict, horizon: str) -> str:
        icon = {"severity-none":"🌤️","severity-light":"🌦️",
                "severity-mod":"🌧️","severity-heavy":"⛈️"}.get(d["sev_class"],"🌀")
        return (f'<span class="{d["sev_class"]}" style="font-size:0.95rem;margin-right:8px">'
                f'{icon} {horizon}: {d["sev_label"].split("  ")[-1].strip()}'
                f'</span>')

    g30 = interp["gt_t30"];  g60 = interp["gt_t60"]
    st.markdown(
        _gt_badge(g30, "t+30 min") + _gt_badge(g60, "t+60 min"),
        unsafe_allow_html=True,
    )
    st.markdown("")   # spacer

    gt1, gt2 = st.columns(2)
    with gt1:
        st.markdown("**t+30 min  —  actual**")
        st.markdown(f"""
| Category | % of grid |
|---|---|
| No Rain  | {g30['no_rain_pct']:.1f}% |
| Light    | {g30['light_pct']:.1f}% |
| Moderate | {g30['mod_pct']:.1f}% |
| Heavy    | {g30['heavy_pct']:.1f}% |
| **Peak VIL** | **{g30['peak']:.0f} kg/m²** |
| **Dominant** | **{g30['dominant']}** |
""")
        agree30 = interp["gt_t30_agree"]
        if agree30:
            st.success("✅ Model predicted the **correct severity tier** at t+30")
        else:
            st.error("❌ Model severity tier **did not match** ground truth at t+30")

    with gt2:
        st.markdown("**t+60 min  —  actual**")
        st.markdown(f"""
| Category | % of grid |
|---|---|
| No Rain  | {g60['no_rain_pct']:.1f}% |
| Light    | {g60['light_pct']:.1f}% |
| Moderate | {g60['mod_pct']:.1f}% |
| Heavy    | {g60['heavy_pct']:.1f}% |
| **Peak VIL** | **{g60['peak']:.0f} kg/m²** |
| **Dominant** | **{g60['dominant']}** |
""")
        agree60 = interp["gt_t60_agree"]
        if agree60:
            st.success("✅ Model predicted the **correct severity tier** at t+60")
        else:
            st.error("❌ Model severity tier **did not match** ground truth at t+60")

    st.divider()

    # ── Two-column layout ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🌀 Storm Dynamics")

        # Motion row — only show direction/speed if reliable
        if interp["motion_reliable"]:
            motion_str = f"**{interp['direction']}** at ≈ {interp['speed_kmh']:.0f} km/h"
        else:
            motion_str = "Near-stationary (< 0.5 px/frame)"

        st.markdown(f"""
| Property | Observed → Predicted |
|---|---|
| Storm motion | {motion_str} |
| Peak VIL · obs / t+30 / t+60 | {interp['max_phys_obs']:.0f} → {interp['max_phys_t30']:.0f} → {interp['max_phys_t60']:.0f} kg/m² |
| Intensity trend (t+30→t+60) | {interp['intensity_trend']} |
| Coverage mod+ · obs / t+30 / t+60 | {interp['cov_obs']:.1f}% → {interp['cov_t30']:.1f}% → {interp['cov_t60']:.1f}% |
""")
        # Data-driven flags — only shown when warranted
        if interp["amp_flag"]:
            st.warning(interp["amp_flag"])
        if interp["cov_flag"]:
            st.warning(interp["cov_flag"])
        if interp["position_note"]:
            st.info(f"📍 {interp['position_note']}")

    with c2:
        st.markdown("#### 🎯 Forecast Skill")

        # Skill quality labels
        def _skill(v, good, ok):
            if v >= good: return f"{v:.3f} ✅"
            if v >= ok:   return f"{v:.3f} ⚠️"
            return           f"{v:.3f} ❌"

        st.markdown(f"""
| Metric | t+30 | t+60 |
|---|---|---|
| CSI-mod  (≥0.3 good) | {_skill(m30['csi_mod'],  0.3, 0.15)} | {_skill(m60['csi_mod'],  0.3, 0.15)} |
| CSI-heavy (≥0.2 good)| {_skill(m30['csi_heavy'],0.2, 0.10)} | {_skill(m60['csi_heavy'],0.2, 0.10)} |
| POD-heavy (≥0.5 good)| {_skill(m30['pod_heavy'],0.5, 0.25)} | {_skill(m60['pod_heavy'],0.5, 0.25)} |
| RMSE                 | {m30['rmse']:.3f} | {m60['rmse']:.3f} |
| Bias                 | {m30['bias']:+.3f} | {m60['bias']:+.3f} |
""")

    # ── Narrative — fully data-driven, no hardcoded descriptions ─────────────
    st.markdown("#### 💡 What the Model Saw")

    obs_p  = interp["max_phys_obs"]
    t30_p  = interp["max_phys_t30"]
    t60_p  = interp["max_phys_t60"]
    trend  = interp["intensity_trend"]
    csi_m  = m30["csi_mod"]
    pod_h  = m30["pod_heavy"]
    bias   = m30["bias"]

    # Observed intensity description
    if obs_p > 133:
        obs_desc = f"severe convective precipitation (peak VIL **{obs_p:.0f} kg/m²**)"
    elif obs_p > 74:
        obs_desc = f"moderate-to-heavy precipitation (peak VIL **{obs_p:.0f} kg/m²**)"
    elif obs_p > 17:
        obs_desc = f"light precipitation (peak VIL **{obs_p:.0f} kg/m²**)"
    else:
        obs_desc = f"little or no significant precipitation (peak VIL **{obs_p:.0f} kg/m²**)"

    # Motion sentence
    if interp["motion_reliable"]:
        motion_sent = (f"Storm motion is estimated at **{interp['direction']}** "
                       f"≈ {interp['speed_kmh']:.0f} km/h.")
    else:
        motion_sent = ("The system appears **near-stationary** — "
                       "phase correlation returned a sub-pixel shift.")

    # Forecast evolution sentence — uses actual numbers, not just trend label
    pred_sent = (f"The model forecasts peak VIL of **{t30_p:.0f} kg/m²** at t+30 "
                 f"and **{t60_p:.0f} kg/m²** at t+60 ({trend}).")

    # Skill caveat 
    if csi_m < 0.15:
        skill_note = (f"⚠️ **Low forecast skill** (CSI-mod = {csi_m:.3f}) — "
                      "spatial placement of rain areas is uncertain. "
                      "Use the coverage trend rather than exact locations.")
    elif csi_m < 0.3:
        skill_note = (f"ℹ️ Moderate forecast skill (CSI-mod = {csi_m:.3f}). "
                      "General area coverage is useful; fine-scale placement may differ.")
    else:
        skill_note = f"✅ Good forecast skill (CSI-mod = {csi_m:.3f})."

    # Bias note
    if abs(bias) > 0.15:
        bias_note = (f"Model has a {'positive' if bias > 0 else 'negative'} bias "
                     f"of {bias:+.2f} normalised VIL — predictions are systematically "
                     f"{'too high' if bias > 0 else 'too low'}.")
    else:
        bias_note = ""

    # Combine — no hardcoded "squall line" or weather type
    full_narrative = f"""
The **{method}** model observed {obs_desc} in the input sequence.
{motion_sent}
{pred_sent}
{skill_note}
{bias_note}
"""
    st.info(full_narrative.strip())

    # ── VIL key ───────────────────────────────────────────────────────────────
    with st.expander("🔢 VIL Intensity Reference", expanded=False):
        st.markdown("""
| Category | Physical VIL | Typical hazard |
|---|---|---|
| No rain | < 17 kg/m² | None |
| Light | 17 – 74 kg/m² | Light rain, isolated lightning |
| Moderate | 74 – 133 kg/m² | Heavy rain, frequent lightning |
| Severe | > 133 kg/m² | Flash floods, hail, damaging wind |
""")


if __name__ == "__main__":
    main()
