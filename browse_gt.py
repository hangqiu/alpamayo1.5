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
import scipy.spatial.transform as spt
import gradio as gr
import physical_ai_av
from PIL import Image, ImageDraw

from alpamayo1_5.viz_utils import CAMERA_GRID_LAYOUT

# Display at half resolution to reduce memory and projection time
DISPLAY_SCALE = 0.5

CAM_IDX_TO_FEATURE = {
    0: "camera_cross_left_120fov",
    1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov",
    6: "camera_front_tele_30fov",
}

SLIDER_MIN, SLIDER_MAX, SLIDER_STEP = -3.0, 10.0, 0.1
NUM_HISTORY, NUM_FUTURE, TIME_STEP, NUM_FRAMES = 16, 64, 0.1, 4

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
            "clip_id": clip_id, "t0_us": 5_100_000,
            "label": f"[clip] {clip_id}",
            "nav_text": "", "nav_maneuver": "", "distance_m": None, "cot": "",
        })
    return samples


SAMPLES = _load_samples()
SAMPLE_LABELS = [s["label"] for s in SAMPLES]
AVDI = physical_ai_av.PhysicalAIAVDatasetInterface()

# ---------------------------------------------------------------------------
# ClipData: load once per clip, cheap to query at any t0
# ---------------------------------------------------------------------------

class _ClipData:
    """Holds all clip-level data loaded from the dataset.

    All network I/O happens once in __init__. After that, sample_at_t0()
    only does in-memory operations (egomotion interpolation + video seeks
    against the already-buffered BytesIO).
    """

    def __init__(self, clip_id: str):
        self.clip_id = clip_id
        self.egomotion = AVDI.get_clip_feature(
            clip_id, AVDI.features.LABELS.EGOMOTION, maybe_stream=True)
        self.cameras: dict[int, object] = {}   # cam_idx -> SeekVideoReader
        for cam_idx, cam_name in CAM_IDX_TO_FEATURE.items():
            feat_name = getattr(AVDI.features.CAMERA, cam_name.upper())
            self.cameras[cam_idx] = AVDI.get_clip_feature(
                clip_id, feat_name, maybe_stream=True)
        intr = AVDI.get_clip_feature(
            clip_id, AVDI.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True)
        extr = AVDI.get_clip_feature(
            clip_id, AVDI.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True)
        self.cam_models  = intr.camera_models   # cam_name -> FThetaCameraModel
        self.cam_poses   = extr.sensor_poses    # cam_name -> RigidTransform

        # Frame cache: {cam_idx: {requested_ts_us: frame_hwc}}
        self._frame_cache: dict[int, dict[int, np.ndarray]] = {
            idx: {} for idx in self.cameras}
        self._frame_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Egomotion helpers (pure in-memory)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_timestamps(t0_us: int) -> np.ndarray:
        return np.array(
            [t0_us - (NUM_FRAMES - 1 - i) * int(TIME_STEP * 1_000_000)
             for i in range(NUM_FRAMES)], dtype=np.int64)

    def ego_at_t0(self, t0_us: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (hist_xyz, fut_xyz) in ego frame at t0. Shape [T, 3] each."""
        dt = int(TIME_STEP * 1_000_000)
        hist_ts = np.arange(
            t0_us - (NUM_HISTORY - 1) * dt, t0_us + dt // 2, dt, dtype=np.int64)
        fut_ts  = np.arange(
            t0_us + dt, t0_us + (NUM_FUTURE + 1) * dt, dt, dtype=np.int64)

        ego_hist = self.egomotion(hist_ts)
        ego_fut  = self.egomotion(fut_ts)

        t0_xyz  = ego_hist.pose.translation[-1].copy()
        t0_rot  = spt.Rotation.from_quat(ego_hist.pose.rotation.as_quat()[-1])
        R_inv   = t0_rot.inv()

        hist_xyz = R_inv.apply(ego_hist.pose.translation - t0_xyz)
        fut_xyz  = R_inv.apply(ego_fut.pose.translation  - t0_xyz)
        return hist_xyz, fut_xyz

    # ------------------------------------------------------------------
    # Frame access (backed by per-clip cache)
    # ------------------------------------------------------------------

    def _decode_and_cache(self, cam_idx: int, requested_ts: np.ndarray):
        """Decode frames for requested timestamps and store in cache."""
        reader = self.cameras[cam_idx]
        imgs, _ = reader.decode_images_from_timestamps(requested_ts)
        with self._frame_lock:
            for ts, img in zip(requested_ts, imgs):
                self._frame_cache[cam_idx][int(ts)] = img

    def get_frames(self, t0_us: int) -> dict[int, np.ndarray]:
        """Return {cam_idx: frame_hwc} for the frame at t0 for each camera.

        Uses per-clip frame cache; only decodes frames not yet cached.
        """
        # We only need the single frame at t0 for the camera grid display
        ts_single = np.array([t0_us], dtype=np.int64)
        result = {}
        for cam_idx in self.cameras:
            with self._frame_lock:
                cached = self._frame_cache[cam_idx].get(t0_us)
            if cached is None:
                self._decode_and_cache(cam_idx, ts_single)
                with self._frame_lock:
                    cached = self._frame_cache[cam_idx][t0_us]
            result[cam_idx] = cached
        return result

    def preload_range(self, base_t0_us: int, offsets: np.ndarray):
        """Batch-decode all unique frames for a range of t0 offsets.

        Calls decode_images_from_timestamps once per camera with all unique
        timestamps, so the video is read in a single sequential pass.
        """
        all_ts: list[int] = []
        for off in offsets:
            t0_us = base_t0_us + int(round(float(off), 1) * 1_000_000)
            all_ts.append(t0_us)

        ts_arr = np.array(sorted(set(all_ts)), dtype=np.int64)

        for cam_idx in self.cameras:
            with self._frame_lock:
                missing = ts_arr[
                    np.array([t not in self._frame_cache[cam_idx] for t in ts_arr])]
            if len(missing):
                self._decode_and_cache(cam_idx, missing)


# Clip-level cache: clip_id -> _ClipData
_clip_cache: dict[str, _ClipData] = {}
_clip_cache_lock = threading.Lock()
# Per-clip event: set once the clip finishes loading, so waiters don't double-load
_clip_loading_events: dict[str, threading.Event] = {}

# ---------------------------------------------------------------------------
# Preload status (thread-safe, polled by UI timer)
# ---------------------------------------------------------------------------

_status: dict = {"phase": "idle", "clip_id": "", "rendered": 0, "total": 0}
_status_lock = threading.Lock()


def _set_status(**kwargs):
    with _status_lock:
        _status.update(kwargs)


def get_status_html() -> str:
    with _status_lock:
        s = dict(_status)
    phase = s["phase"]
    clip  = s["clip_id"][:8] if s["clip_id"] else ""
    done, total = s["rendered"], s["total"]

    if phase == "idle":
        return '<span style="color:#888">—</span>'
    if phase == "loading_clip":
        return f'<span style="color:#f59e0b">⏳ Loading clip {clip}…</span>'
    if phase == "decoding":
        return f'<span style="color:#3b82f6">📦 Batch-decoding video frames for {clip}…</span>'
    if phase == "rendering":
        pct  = int(100 * done / total) if total else 0
        fill = "█" * (pct // 5)
        empty = "░" * (20 - pct // 5)
        color = "#22c55e" if done >= total else "#3b82f6"
        label = "✅ Ready" if done >= total else f"🎨 Preloading"
        return (f'<span style="color:{color}">'
                f'{label} [{fill}{empty}] {done}/{total} frames ({pct}%)'
                f'</span>')
    return ""


def _get_clip_data(clip_id: str, blocking: bool = True) -> "_ClipData | None":
    """Return ClipData for clip_id.

    If blocking=True (default): loads the clip if not cached, waits if another
    thread is already loading it. Safe to call from background workers.

    If blocking=False: returns None immediately if the clip is not yet ready.
    Use this from UI callbacks (timers) to avoid thread-pool exhaustion.
    """
    # Fast path — already cached
    with _clip_cache_lock:
        if clip_id in _clip_cache:
            return _clip_cache[clip_id]
        if not blocking:
            return None
        # Slow path — decide whether this thread should load or wait
        if clip_id in _clip_loading_events:
            event = _clip_loading_events[clip_id]
            should_load = False
        else:
            event = threading.Event()
            _clip_loading_events[clip_id] = event
            should_load = True

    if should_load:
        try:
            _set_status(phase="loading_clip", clip_id=clip_id, rendered=0, total=0)
            new_data = _ClipData(clip_id)
            with _clip_cache_lock:
                _clip_cache[clip_id] = new_data
                _clip_loading_events.pop(clip_id, None)
        except Exception:
            with _clip_cache_lock:
                _clip_loading_events.pop(clip_id, None)
            event.set()
            raise
        event.set()
        return new_data
    else:
        # Another thread is loading — wait for it (with timeout)
        event.wait(timeout=60)
        with _clip_cache_lock:
            return _clip_cache.get(clip_id)


# ---------------------------------------------------------------------------
# Projection & rendering helpers
# ---------------------------------------------------------------------------

def _project_onto_camera(pts_ego, pose, cam_model):
    p_cam = pose.inv().apply(pts_ego)
    front = p_cam[:, 2] > 0.3
    if not front.any():
        return np.empty((0, 2))
    rays = p_cam[front] / np.linalg.norm(p_cam[front], axis=-1, keepdims=True)
    pixels = cam_model.ray2pixel(rays)
    return pixels[~cam_model.is_out_of_bounds(pixels)]


def _build_camera_grid(
        frames: dict[int, np.ndarray],
        hist_ego: np.ndarray, fut_ego: np.ndarray,
        cam_models: dict, cam_poses: dict) -> np.ndarray:
    """Annotate frames and assemble into 2×3 grid."""
    # Use first available frame to determine (scaled) size
    sample_frame = next(iter(frames.values()))
    orig_h, orig_w = sample_frame.shape[:2]
    h = int(orig_h * DISPLAY_SCALE)
    w = int(orig_w * DISPLAY_SCALE)

    grid = np.zeros((2, 3, h, w, 3), dtype=np.uint8)

    for cam_idx, frame in frames.items():
        cam_name = CAM_IDX_TO_FEATURE[cam_idx]
        # Scale down for display
        img = np.array(Image.fromarray(frame).resize((w, h), Image.BILINEAR))

        if cam_name in cam_poses and cam_name in cam_models:
            # Scale intrinsics: principal_point and r2th already in original pixels;
            # create a scaled proxy by scaling the projected pixels after projection
            img = _draw_trajectory_on_image_scaled(
                img, hist_ego, fut_ego,
                cam_poses[cam_name], cam_models[cam_name],
                DISPLAY_SCALE)

        if cam_idx in CAMERA_GRID_LAYOUT:
            r, c = CAMERA_GRID_LAYOUT[cam_idx]
            grid[r, c] = img

    return np.concatenate(
        [np.concatenate(grid[r], axis=1) for r in range(2)], axis=0)


def _draw_trajectory_on_image_scaled(
        img_hwc, hist_ego, fut_ego, pose, cam_model, scale, dot_radius=4):
    """Project using full-res camera model, then scale pixel coords for display."""
    img = Image.fromarray(img_hwc)
    draw = ImageDraw.Draw(img)

    def _pts(pts, color):
        for u, v in _project_onto_camera(pts, pose, cam_model):
            u, v = int(round(u * scale)), int(round(v * scale))
            r = dot_radius
            draw.ellipse([u-r, v-r, u+r, v+r], fill=color, outline="white")

    _pts(hist_ego, (160, 160, 160))
    _pts(fut_ego,  (50, 220, 50))
    return np.array(img)


# ---------------------------------------------------------------------------
# Render cache: (clip_id, t0_us) -> (cam_grid, hist_ego, fut_ego)
# ---------------------------------------------------------------------------

_render_cache: dict[tuple, tuple] = {}
_render_lock  = threading.Lock()


def _render_at_t0(clip_data: _ClipData, t0_us: int) -> tuple | None:
    key = (clip_data.clip_id, t0_us)
    with _render_lock:
        if key in _render_cache:
            return _render_cache[key]
    try:
        hist_ego, fut_ego = clip_data.ego_at_t0(t0_us)
        frames = clip_data.get_frames(t0_us)
        cam_grid = _build_camera_grid(
            frames, hist_ego, fut_ego, clip_data.cam_models, clip_data.cam_poses)
    except Exception as e:
        print(f"[render] error {clip_data.clip_id} t0={t0_us}: {e}")
        return None
    result = (cam_grid, hist_ego, fut_ego)
    with _render_lock:
        _render_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Preloader
# ---------------------------------------------------------------------------

_preload_executor  = ThreadPoolExecutor(max_workers=2)
_preload_generation = 0
_preload_gen_lock  = threading.Lock()


def _preload_worker(clip_id: str, base_t0_us: int, offsets: np.ndarray, generation: int):
    with _preload_gen_lock:
        if generation != _preload_generation:
            return
    try:
        clip_data = _get_clip_data(clip_id)
    except Exception as e:
        print(f"[preload] error loading clip {clip_id}: {e}")
        _set_status(phase="idle")
        return

    total = len(offsets)

    # Phase 1: batch-decode all raw video frames (one sequential pass per camera)
    _set_status(phase="decoding", clip_id=clip_id, rendered=0, total=total)
    clip_data.preload_range(base_t0_us, offsets)

    # Phase 2: render annotated grids for each t0
    _set_status(phase="rendering", rendered=0, total=total)
    for i, off in enumerate(offsets):
        with _preload_gen_lock:
            if generation != _preload_generation:
                _set_status(phase="idle")
                return
        t0_us = base_t0_us + int(round(float(off), 1) * 1_000_000)
        _render_at_t0(clip_data, t0_us)
        _set_status(phase="rendering", rendered=i + 1, total=total)

    _set_status(phase="rendering", rendered=total, total=total)


def _schedule_preload(sample: dict, current_t0_offset_s: float):
    global _preload_generation
    with _preload_gen_lock:
        _preload_generation += 1
        gen = _preload_generation

    offsets = np.arange(SLIDER_MIN, SLIDER_MAX + SLIDER_STEP / 2, SLIDER_STEP)
    dist    = np.abs(offsets - current_t0_offset_s)
    ordered = offsets[np.argsort(dist)]

    _preload_executor.submit(
        _preload_worker, sample["clip_id"], sample["t0_us"], ordered, gen)


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def _make_bev_img(hist_ego, fut_ego, t0_us: int) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(hist_ego[:, 0], hist_ego[:, 1], "o--", color="gray",  lw=1.5, ms=3, label="History")
    ax.plot(fut_ego[:, 0],  fut_ego[:, 1],  "o-",  color="green", lw=2,   ms=3, label="Future")
    ax.plot(0, 0, marker="^", color="black", ms=9, label="Ego (t0)", zorder=5)
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


def render_sample(sample: dict, t0_offset_s: float, blocking: bool = True):
    clip_id = sample["clip_id"]
    t0_us   = sample["t0_us"] + int(round(t0_offset_s, 1) * 1_000_000)

    try:
        clip_data = _get_clip_data(clip_id, blocking=blocking)
    except Exception as e:
        blank = np.zeros((100, 400, 3), dtype=np.uint8)
        return blank, blank, f"Error loading clip: {e}"

    if clip_data is None:
        # Clip still loading — return a placeholder immediately
        blank = np.zeros((100, 400, 3), dtype=np.uint8)
        return blank, blank, "Loading clip…"

    result = _render_at_t0(clip_data, t0_us)
    if result is None:
        blank = np.zeros((100, 400, 3), dtype=np.uint8)
        return blank, blank, f"Error rendering {clip_id} t0={t0_us/1e6:.1f}s"

    cam_grid, hist_ego, fut_ego = result
    bev = _make_bev_img(hist_ego, fut_ego, t0_us)

    lines = [f"clip_id : {clip_id}", f"t0      : {t0_us/1e6:.3f}s"]
    if sample["nav_text"]:
        lines += [f"nav     : {sample['nav_text']}",
                  f"maneuver: {sample['nav_maneuver']}  ({sample['distance_m']:.1f}m)"]
    if sample["cot"]:
        lines.append(f"cot     : {sample['cot']}")
    return cam_grid, bev, "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_select(label: str, t0_offset_s: float):
    idx = SAMPLE_LABELS.index(label)
    cam, bev, meta = render_sample(SAMPLES[idx], t0_offset_s)
    _schedule_preload(SAMPLES[idx], t0_offset_s)
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
    return new_playing, gr.update(active=new_playing), "⏸ Pause" if new_playing else "▶ Play"


def on_tick(current_idx: int, t0_offset_s: float, is_playing: bool):
    if not is_playing:
        return gr.update(), gr.update(), gr.update(), gr.update()
    # Non-blocking: if clip not ready yet, skip this tick silently
    sample = SAMPLES[int(current_idx)]
    if _get_clip_data(sample["clip_id"], blocking=False) is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    next_t0 = round(t0_offset_s + SLIDER_STEP, 1)
    if next_t0 > SLIDER_MAX:
        return gr.update(value=t0_offset_s), gr.update(), gr.update(), gr.update()
    cam, bev, meta = render_sample(sample, next_t0, blocking=False)
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
        "Gray = history GT · Green = future GT · No model inference."
    )

    current_idx = gr.State(value=0)
    is_playing  = gr.State(value=False)

    with gr.Row():
        sample_dd = gr.Dropdown(
            choices=SAMPLE_LABELS, value=SAMPLE_LABELS[0],
            label=f"Sample ({len(SAMPLES)} total)", scale=6)

    with gr.Row():
        btn_prev  = gr.Button("← Prev", scale=1)
        btn_next  = gr.Button("Next →", scale=1)
        btn_play  = gr.Button("▶ Play", scale=1)
        t0_slider = gr.Slider(
            minimum=SLIDER_MIN, maximum=SLIDER_MAX, value=0.0, step=SLIDER_STEP,
            label="t0 offset (s)", scale=5)

    status_html = gr.HTML(value=get_status_html())
    progress_timer = gr.Timer(value=0.4, active=True)

    with gr.Row():
        cam_out = gr.Image(
            label="Camera grid (GT projected)", type="numpy", elem_id="cam-col")
        with gr.Column(elem_id="bev-col"):
            bev_out  = gr.Image(label="BEV ground truth", type="numpy")
            meta_out = gr.Textbox(label="Metadata", lines=5, interactive=False)

    play_timer = gr.Timer(value=0.5, active=False)

    sample_dd.change(on_select,       [sample_dd, t0_slider], [cam_out, bev_out, meta_out, current_idx])
    btn_prev.click(on_prev,           [current_idx, t0_slider], [cam_out, bev_out, meta_out, current_idx, sample_dd])
    btn_next.click(on_next,           [current_idx, t0_slider], [cam_out, bev_out, meta_out, current_idx, sample_dd])
    t0_slider.release(on_t0_change,   [current_idx, t0_slider], [cam_out, bev_out, meta_out])
    btn_play.click(on_play_pause,     [is_playing], [is_playing, play_timer, btn_play])
    play_timer.tick(on_tick,          [current_idx, t0_slider, is_playing], [t0_slider, cam_out, bev_out, meta_out])
    progress_timer.tick(get_status_html, [], [status_html])

    demo.load(
        lambda: on_select(SAMPLE_LABELS[0], 0.0),
        outputs=[cam_out, bev_out, meta_out, current_idx])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=_css)
