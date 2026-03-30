"""Ground-truth data browser for the PhysicalAI-AV dataset.

Displays camera images with projected GT trajectory and a BEV plot.
No model inference required.

Launch:
    source a1_5_venv/bin/activate
    python browse_gt.py
Then open http://localhost:7860 in your browser.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr
import physical_ai_av
from PIL import Image, ImageDraw

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.viz_utils import CAMERA_GRID_LAYOUT

CAM_IDX_TO_NAME = {
    0: "camera_cross_left_120fov",
    1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov",
    3: "camera_rear_left_70fov",
    4: "camera_rear_tele_30fov",
    5: "camera_rear_right_70fov",
    6: "camera_front_tele_30fov",
}

SLIDER_MIN, SLIDER_MAX, SLIDER_STEP = -3.0, 10.0, 0.1

# ---------------------------------------------------------------------------
# Sample list
# ---------------------------------------------------------------------------

def _load_samples():
    samples = []
    with open("notebooks/nav_demo_samples.json") as f:
        nav_samples = json.load(f)
    for s in nav_samples:
        samples.append({
            "clip_id": s["clip_id"],
            "t0_us": s["t0_relative"],
            "label": f"[nav] {s['nav_maneuver']} | {s['nav_text']} | {s['clip_id'][:8]}",
            "nav_text": s.get("nav_text", ""),
            "nav_maneuver": s.get("nav_maneuver", ""),
            "distance_m": s.get("distance_m", None),
            "cot": s.get("cot", ""),
        })
    clip_df = pd.read_parquet("notebooks/clip_ids.parquet")
    nav_clip_ids = {s["clip_id"] for s in nav_samples}
    for clip_id in clip_df["clip_id"]:
        if clip_id in nav_clip_ids:
            continue
        samples.append({
            "clip_id": clip_id,
            "t0_us": 5_100_000,
            "label": f"[clip] {clip_id}",
            "nav_text": "", "nav_maneuver": "", "distance_m": None, "cot": "",
        })
    return samples


SAMPLES = _load_samples()
SAMPLE_LABELS = [s["label"] for s in SAMPLES]
AVDI = physical_ai_av.PhysicalAIAVDatasetInterface()

# ---------------------------------------------------------------------------
# Calibration cache (per clip — same for all t0 in a clip)
# ---------------------------------------------------------------------------

_calib_cache: dict[str, tuple] = {}
_calib_lock = threading.Lock()

def _get_calib(clip_id: str):
    with _calib_lock:
        if clip_id not in _calib_cache:
            intr = AVDI.get_clip_feature(clip_id, AVDI.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True)
            extr = AVDI.get_clip_feature(clip_id, AVDI.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True)
            _calib_cache[clip_id] = (intr, extr)
        return _calib_cache[clip_id]

# ---------------------------------------------------------------------------
# Render cache: (clip_id, t0_us) -> (cam_grid_np, hist_ego, fut_ego, meta_str)
# ---------------------------------------------------------------------------

_render_cache: dict[tuple, tuple] = {}
_render_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project_onto_camera(pts_ego, pose, cam_model):
    p_cam = pose.inv().apply(pts_ego)
    front = p_cam[:, 2] > 0.3
    if not front.any():
        return np.empty((0, 2))
    rays = p_cam[front] / np.linalg.norm(p_cam[front], axis=-1, keepdims=True)
    pixels = cam_model.ray2pixel(rays)
    return pixels[~cam_model.is_out_of_bounds(pixels)]


def _draw_trajectory_on_image(img_hwc, hist_ego, fut_ego, pose, cam_model, dot_radius=6):
    img = Image.fromarray(img_hwc)
    draw = ImageDraw.Draw(img)

    def _draw_pts(pts, color):
        if len(pts) == 0:
            return
        for u, v in _project_onto_camera(pts, pose, cam_model):
            u, v = int(round(u)), int(round(v))
            draw.ellipse([u - dot_radius, v - dot_radius, u + dot_radius, v + dot_radius],
                         fill=color, outline="white")

    _draw_pts(hist_ego, (160, 160, 160))
    _draw_pts(fut_ego,  (50, 220, 50))
    return np.array(img)


def _make_annotated_camera_grid(image_frames, camera_indices, hist_ego, fut_ego, intr, extr):
    last_frames = image_frames[:, -1]
    frames_hwc = last_frames.permute(0, 2, 3, 1).numpy()
    h, w = frames_hwc.shape[1], frames_hwc.shape[2]
    grid = np.zeros((2, 3, h, w, 3), dtype=np.uint8)

    for i, cam_idx in enumerate(camera_indices.tolist()):
        cam_name = CAM_IDX_TO_NAME.get(cam_idx)
        img = frames_hwc[i]
        if cam_name and cam_name in extr.sensor_poses and cam_name in intr.camera_models:
            img = _draw_trajectory_on_image(img, hist_ego, fut_ego,
                                            extr.sensor_poses[cam_name],
                                            intr.camera_models[cam_name])
        if cam_idx in CAMERA_GRID_LAYOUT:
            r, c = CAMERA_GRID_LAYOUT[cam_idx]
            grid[r, c] = img

    return np.concatenate([np.concatenate(grid[r], axis=1) for r in range(2)], axis=0)

# ---------------------------------------------------------------------------
# Core data loading (cached)
# ---------------------------------------------------------------------------

def _load_and_cache(clip_id: str, t0_us: int) -> tuple | None:
    """Load, project, and cache one frame. Returns None on error."""
    key = (clip_id, t0_us)
    with _render_lock:
        if key in _render_cache:
            return _render_cache[key]

    try:
        data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=AVDI)
        intr, extr = _get_calib(clip_id)
    except Exception as e:
        print(f"[cache] error loading {clip_id} t0={t0_us}: {e}")
        return None

    hist_ego = data["ego_history_xyz"][0, 0].numpy()
    fut_ego  = data["ego_future_xyz"][0, 0].numpy()
    cam_grid = _make_annotated_camera_grid(
        data["image_frames"], data["camera_indices"], hist_ego, fut_ego, intr, extr)
    meta = (clip_id, t0_us, data.get("clip_id", clip_id))

    result = (cam_grid, hist_ego, fut_ego, meta)
    with _render_lock:
        _render_cache[key] = result
    return result

# ---------------------------------------------------------------------------
# Preloader
# ---------------------------------------------------------------------------

_preload_executor = ThreadPoolExecutor(max_workers=3)
_preload_generation = 0       # incremented on each new sample to cancel old work
_preload_gen_lock = threading.Lock()


def _submit_preload(clip_id: str, t0_us: int, generation: int):
    """Worker: skip if generation is stale or key already cached."""
    with _preload_gen_lock:
        if generation != _preload_generation:
            return
    key = (clip_id, t0_us)
    with _render_lock:
        if key in _render_cache:
            return
    _load_and_cache(clip_id, t0_us)


def _schedule_preload(sample: dict, current_t0_offset_s: float):
    """Enqueue all frames in the slider range, closest-first."""
    global _preload_generation
    with _preload_gen_lock:
        _preload_generation += 1
        gen = _preload_generation

    clip_id = sample["clip_id"]
    base_t0 = sample["t0_us"]

    # Build list of all offsets in slider range, sorted by distance from current
    all_offsets = np.arange(SLIDER_MIN, SLIDER_MAX + SLIDER_STEP / 2, SLIDER_STEP)
    dist = np.abs(all_offsets - current_t0_offset_s)
    ordered = all_offsets[np.argsort(dist)]

    for off in ordered:
        t0_us = base_t0 + int(round(off, 1) * 1_000_000)
        _preload_executor.submit(_submit_preload, clip_id, t0_us, gen)


# ---------------------------------------------------------------------------
# Rendering (uses cache if warm)
# ---------------------------------------------------------------------------

def _make_bev_img(hist_ego, fut_ego, t0_us: int) -> np.ndarray:
    """Render BEV plot to a numpy uint8 array so gr.Image controls its size."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(hist_ego[:, 0], hist_ego[:, 1], "o--", color="gray",  lw=1.5, ms=3, label="History GT")
    ax.plot(fut_ego[:, 0],  fut_ego[:, 1],  "o-",  color="green", lw=2,   ms=3, label="Future GT")
    ax.plot(0, 0, marker="^", color="black", markersize=9, label="Ego (t0)", zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"t0 = {t0_us / 1e6:.1f}s", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return buf[:, :, :3]


def render_sample(sample: dict, t0_offset_s: float):
    """Return (cam_grid, bev_fig, meta_str). Uses cache when available."""
    clip_id = sample["clip_id"]
    t0_us = sample["t0_us"] + int(round(t0_offset_s, 1) * 1_000_000)

    result = _load_and_cache(clip_id, t0_us)
    if result is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, f"Load error\n{clip_id[:12]}\nt0={t0_us/1e6:.1f}s",
                ha="center", va="center", transform=ax.transAxes, color="red")
        ax.axis("off")
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return None, buf[:, :, :3], f"Error loading {clip_id}"

    cam_grid, hist_ego, fut_ego, _ = result
    bev_fig = _make_bev_img(hist_ego, fut_ego, t0_us)

    lines = [f"clip_id : {clip_id}", f"t0      : {t0_us/1e6:.3f}s"]
    if sample["nav_text"]:
        lines += [f"nav     : {sample['nav_text']}",
                  f"maneuver: {sample['nav_maneuver']}  ({sample['distance_m']:.1f}m)"]
    if sample["cot"]:
        lines.append(f"cot     : {sample['cot']}")
    return cam_grid, bev_fig, "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_select(label: str, t0_offset_s: float):
    idx = SAMPLE_LABELS.index(label)
    sample = SAMPLES[idx]
    cam, bev, meta = render_sample(sample, t0_offset_s)
    _schedule_preload(sample, t0_offset_s)
    return cam, bev, meta, idx


def on_prev(current_idx: int, t0_offset_s: float):
    idx = max(0, int(current_idx) - 1)
    cam, bev, meta = render_sample(SAMPLES[idx], t0_offset_s)
    _schedule_preload(SAMPLES[idx], t0_offset_s)
    return cam, bev, meta, idx, SAMPLE_LABELS[idx]


def on_next(current_idx: int, t0_offset_s: float):
    idx = min(len(SAMPLES) - 1, int(current_idx) + 1)
    cam, bev, meta = render_sample(SAMPLES[idx], t0_offset_s)
    _schedule_preload(SAMPLES[idx], t0_offset_s)
    return cam, bev, meta, idx, SAMPLE_LABELS[idx]


def on_t0_change(current_idx: int, t0_offset_s: float):
    cam, bev, meta = render_sample(SAMPLES[int(current_idx)], t0_offset_s)
    return cam, bev, meta


def on_play_pause(is_playing: bool):
    new_playing = not is_playing
    label = "⏸ Pause" if new_playing else "▶ Play"
    return new_playing, gr.update(active=new_playing), label


def on_tick(current_idx: int, t0_offset_s: float, is_playing: bool):
    """Timer tick: advance one step and render from cache."""
    if not is_playing:
        return gr.update(), gr.update(), gr.update(), gr.update()

    next_t0 = round(t0_offset_s + SLIDER_STEP, 1)
    if next_t0 > SLIDER_MAX:
        # Stop at end
        return (gr.update(value=t0_offset_s),
                gr.update(), gr.update(),
                gr.update())

    sample = SAMPLES[int(current_idx)]
    cam, bev, meta = render_sample(sample, next_t0)
    return gr.update(value=next_t0), cam, bev, meta


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_css = """
#cam-col { flex: 7 1 0% !important; min-width: 0 !important; }
#bev-col { flex: 3 1 0% !important; min-width: 0 !important; }
"""

with gr.Blocks(title="PhysicalAI-AV GT Browser") as demo:
    gr.Markdown(
        "## PhysicalAI-AV Ground-Truth Browser\n"
        "Gray dots = history GT · Green dots = future GT · No model inference."
    )

    current_idx = gr.State(value=0)
    is_playing  = gr.State(value=False)

    with gr.Row():
        sample_dd = gr.Dropdown(
            choices=SAMPLE_LABELS, value=SAMPLE_LABELS[0],
            label=f"Sample ({len(SAMPLES)} total)", scale=6,
        )

    with gr.Row():
        btn_prev  = gr.Button("← Prev",  scale=1)
        btn_next  = gr.Button("Next →",  scale=1)
        btn_play  = gr.Button("▶ Play",  scale=1)
        t0_slider = gr.Slider(
            minimum=SLIDER_MIN, maximum=SLIDER_MAX, value=0.0, step=SLIDER_STEP,
            label="t0 offset (s)", scale=5,
        )

    with gr.Row():
        cam_out = gr.Image(label="Camera grid (GT projected)", type="numpy", elem_id="cam-col")
        with gr.Column(elem_id="bev-col"):
            bev_out  = gr.Image(label="BEV ground truth", type="numpy")
            meta_out = gr.Textbox(label="Metadata", lines=5, interactive=False)

    timer = gr.Timer(value=0.5, active=False)

    # Events
    sample_dd.change(on_select, [sample_dd, t0_slider],
                     [cam_out, bev_out, meta_out, current_idx])
    btn_prev.click(on_prev, [current_idx, t0_slider],
                   [cam_out, bev_out, meta_out, current_idx, sample_dd])
    btn_next.click(on_next, [current_idx, t0_slider],
                   [cam_out, bev_out, meta_out, current_idx, sample_dd])
    t0_slider.release(on_t0_change, [current_idx, t0_slider],
                      [cam_out, bev_out, meta_out])
    btn_play.click(on_play_pause, [is_playing],
                   [is_playing, timer, btn_play])
    timer.tick(on_tick, [current_idx, t0_slider, is_playing],
               [t0_slider, cam_out, bev_out, meta_out])

    demo.load(
        lambda: on_select(SAMPLE_LABELS[0], 0.0),
        outputs=[cam_out, bev_out, meta_out, current_idx],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=_css)
