"""Ground-truth data browser for the PhysicalAI-AV dataset.

Displays camera images with projected GT trajectory and a BEV plot.
Optionally overlays Alpamayo 1.5 inference results (3 nav conditions).

Launch:
    source a1_5_venv/bin/activate
    python browse_gt.py
Then open http://localhost:7860 in your browser.
"""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Resolve everything relative to this file's directory so subprocess
# launches work regardless of the working directory.
_HERE = Path(__file__).parent.resolve()

# browse_gt never runs any GPU ops itself, but importing torch (via viz_utils)
# would claim ~600 MiB of VRAM for a CUDA context, leaving the inference
# worker too little headroom on a 24 GB GPU.  Hide the GPU from this process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import gradio as gr
import physical_ai_av
from PIL import Image, ImageDraw

from alpamayo1_5.viz_utils import CAMERA_GRID_LAYOUT, plot_condition

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

INFERENCE_DIR = _HERE / "inference_results"

# Condition definitions: (result_key, color_str, PIL_color_rgb, legend_label)
_COND_DEFS = [
    ("with_nav",       "tab:blue",  (31, 119, 180),  "with nav"),
    ("no_nav",         "tab:red",   (214, 39, 40),   "no nav"),
    ("counterfactual", "tab:orange", (255, 127, 14),  "counterfactual"),
]

# ---------------------------------------------------------------------------
# Sample list
# ---------------------------------------------------------------------------

def _load_samples():
    samples = []
    with open(_HERE / "notebooks/nav_demo_samples.json") as f:
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
    clip_df = pd.read_parquet(_HERE / "notebooks/clip_ids.parquet")
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

    All network I/O happens once in __init__. After that, ego_at_t0()
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
        # Per-camera decode lock: SeekVideoReader is NOT thread-safe.
        self._decode_locks: dict[int, threading.Lock] = {
            idx: threading.Lock() for idx in self.cameras}

    # ------------------------------------------------------------------
    # Egomotion helpers (pure in-memory)
    # ------------------------------------------------------------------

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
        with self._decode_locks[cam_idx]:
            with self._frame_lock:
                still_missing = requested_ts[
                    np.array([t not in self._frame_cache[cam_idx] for t in requested_ts])]
            if not len(still_missing):
                return
            imgs, _ = self.cameras[cam_idx].decode_images_from_timestamps(still_missing)
            with self._frame_lock:
                for ts, img in zip(still_missing, imgs):
                    self._frame_cache[cam_idx][int(ts)] = img

    def get_frames(self, t0_us: int) -> dict[int, np.ndarray]:
        """Return {cam_idx: frame_hwc} for the frame at t0 for each camera."""
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
        """Batch-decode all unique frames for a range of t0 offsets."""
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
_clip_loading_events: dict[str, threading.Event] = {}

# ---------------------------------------------------------------------------
# Status (two independent bars: preload and inference)
# ---------------------------------------------------------------------------

_preload_status: dict = {"phase": "idle", "clip_id": "", "rendered": 0, "total": 0}
_preload_status_lock = threading.Lock()

_inference_status: dict = {"phase": "idle", "clip_id": "", "rendered": 0, "total": 0}
_inference_status_lock = threading.Lock()


def _set_preload_status(**kwargs):
    with _preload_status_lock:
        _preload_status.update(kwargs)


def _set_inference_status(**kwargs):
    with _inference_status_lock:
        _inference_status.update(kwargs)


def get_preload_status_html() -> str:
    with _preload_status_lock:
        s = dict(_preload_status)
    phase = s["phase"]
    clip  = s["clip_id"][:8] if s["clip_id"] else ""
    done, total = s["rendered"], s["total"]

    if phase == "idle":
        return '<span style="color:#888">Preload: —</span>'
    if phase == "loading_clip":
        return f'<span style="color:#f59e0b">⏳ Loading clip {clip}…</span>'
    if phase == "decoding":
        return f'<span style="color:#3b82f6">📦 Decoding {clip}…</span>'
    if phase == "rendering":
        pct  = int(100 * done / total) if total else 0
        fill = "█" * (pct // 5)
        empty = "░" * (20 - pct // 5)
        color = "#22c55e" if done >= total else "#3b82f6"
        label = "✅ Ready" if done >= total else "🎨 Preloading"
        return (f'<span style="color:{color}">'
                f'{label} [{fill}{empty}] {done}/{total} ({pct}%)'
                f'</span>')
    return ""


def get_inference_status_html() -> str:
    with _inference_status_lock:
        s = dict(_inference_status)
    phase = s["phase"]
    clip  = s["clip_id"][:8] if s["clip_id"] else ""
    done, total = s["rendered"], s["total"]

    if phase == "idle":
        return '<span style="color:#888">Inference: —</span>'
    if phase == "inference_running":
        pct = int(100 * done / total) if total else 0
        fill = "█" * (pct // 5)
        empty = "░" * (20 - pct // 5)
        return (f'<span style="color:#a855f7">'
                f'⚙️ Inference {clip} [{fill}{empty}] {done}/{total} ({pct}%)'
                f'</span>')
    if phase == "inference_done":
        return f'<span style="color:#22c55e">✅ Inference done {clip} — toggle conditions to view</span>'
    if phase == "inference_error":
        return f'<span style="color:#ef4444">❌ Inference failed {clip} — check console</span>'
    return ""


def _get_clip_data(clip_id: str, blocking: bool = True) -> "_ClipData | None":
    with _clip_cache_lock:
        if clip_id in _clip_cache:
            return _clip_cache[clip_id]
        if not blocking:
            return None
        if clip_id in _clip_loading_events:
            event = _clip_loading_events[clip_id]
            should_load = False
        else:
            event = threading.Event()
            _clip_loading_events[clip_id] = event
            should_load = True

    if should_load:
        try:
            _set_preload_status(phase="loading_clip", clip_id=clip_id, rendered=0, total=0)
            new_data = _ClipData(clip_id)
            with _clip_cache_lock:
                _clip_cache[clip_id] = new_data
                _clip_loading_events.pop(clip_id, None)
        except Exception:
            with _clip_cache_lock:
                _clip_loading_events.pop(clip_id, None)
            event.set()
            _set_preload_status(phase="idle")
            raise
        event.set()
        return new_data
    else:
        event.wait(timeout=60)
        with _clip_cache_lock:
            return _clip_cache.get(clip_id)


# ---------------------------------------------------------------------------
# InferenceResultStore
# ---------------------------------------------------------------------------

class InferenceResultStore:
    """Read/write interface for offline inference results stored on disk."""

    @staticmethod
    def result_paths(clip_id: str, t0_us: int) -> dict[str, Path]:
        d = INFERENCE_DIR / clip_id
        return {
            "with_nav":       d / f"{t0_us}_with_nav.npz",
            "no_nav":         d / f"{t0_us}_no_nav.npz",
            "counterfactual": d / f"{t0_us}_counterfactual.npz",
            "meta":           d / f"{t0_us}_meta.json",
        }

    @staticmethod
    def has_result(clip_id: str, t0_us: int) -> bool:
        return all(p.exists()
                   for p in InferenceResultStore.result_paths(clip_id, t0_us).values())

    @staticmethod
    def has_any(clip_id: str) -> bool:
        d = INFERENCE_DIR / clip_id
        return d.is_dir() and any(d.glob("*_meta.json"))

    @staticmethod
    def load_result(clip_id: str, t0_us: int) -> dict | None:
        """Load inference result. Returns None if not cached."""
        paths = InferenceResultStore.result_paths(clip_id, t0_us)
        if not all(p.exists() for p in paths.values()):
            return None
        try:
            with_nav = np.load(paths["with_nav"])["pred"]        # [K, T, 3]
            no_nav   = np.load(paths["no_nav"])["pred"]
            counter  = np.load(paths["counterfactual"])["pred"]
            meta     = json.loads(paths["meta"].read_text())
            return {
                "with_nav":       with_nav,
                "no_nav":         no_nav,
                "counterfactual": counter,
                "nav_text":       meta.get("nav_text", ""),
                "nav_text_swapped": meta.get("nav_text_swapped", ""),
                "cot_with_nav":   meta.get("cot_with_nav", ""),
                "cot_no_nav":     meta.get("cot_no_nav", ""),
            }
        except Exception as e:
            print(f"[InferenceResultStore] load error {clip_id} t0={t0_us}: {e}")
            return None


# ---------------------------------------------------------------------------
# Inference subprocess launcher
# ---------------------------------------------------------------------------

_inference_proc: subprocess.Popen | None = None
_inference_monitor_thread: threading.Thread | None = None
_inference_completed_clip: str | None = None  # set when worker exits OK; cleared on auto-show


def _inference_running() -> bool:
    return _inference_proc is not None and _inference_proc.poll() is None


def _kill_inference_if_running():
    """Terminate any in-progress inference subprocess and reset its status."""
    global _inference_proc
    if _inference_running():
        _inference_proc.terminate()
        try:
            _inference_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _inference_proc.kill()
        _set_inference_status(phase="idle")


def launch_inference_for_clip(sample: dict, num_samples: int, max_frames: int = 0) -> None:
    """Launch inference_worker.py as a subprocess for all t0 offsets of a clip."""
    global _inference_proc, _inference_monitor_thread

    if _inference_running():
        _inference_proc.terminate()
        try:
            _inference_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _inference_proc.kill()

    offsets = list(np.arange(SLIDER_MIN, SLIDER_MAX + SLIDER_STEP / 2, SLIDER_STEP))
    offsets = [round(float(o), 1) for o in offsets]

    cmd = [
        sys.executable, str(_HERE / "inference_worker.py"),
        "--clip_id",      sample["clip_id"],
        "--base_t0_us",   str(sample["t0_us"]),
        "--offsets_json", json.dumps(offsets),
        "--nav_text",     sample["nav_text"],
        "--out_dir",      str(INFERENCE_DIR),
        "--num_samples",  str(num_samples),
        "--max_frames",   str(max_frames),
    ]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.pop("CUDA_VISIBLE_DEVICES", None)  # restore GPU visibility for the worker

    _inference_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(_HERE), env=env,
    )

    def _monitor():
        for line in _inference_proc.stdout:
            line = line.strip()
            if line.startswith("PROGRESS "):
                parts = line.split()
                if len(parts) >= 2 and "/" in parts[1]:
                    done_str, total_str = parts[1].split("/")
                    _set_inference_status(
                        phase="inference_running",
                        clip_id=sample["clip_id"],
                        rendered=int(done_str),
                        total=int(total_str),
                    )
        _inference_proc.wait()
        if _inference_proc.returncode == 0:
            _set_inference_status(phase="inference_done", clip_id=sample["clip_id"])
            global _inference_completed_clip
            _inference_completed_clip = sample["clip_id"]
        else:
            rc = _inference_proc.returncode
            if rc == -15 or rc == -9:  # SIGTERM / SIGKILL — user cancelled
                _set_inference_status(phase="idle")
            else:
                _set_inference_status(phase="inference_error", clip_id=sample["clip_id"])

    _inference_monitor_thread = threading.Thread(target=_monitor, daemon=True)
    _inference_monitor_thread.start()
    _set_inference_status(phase="inference_running", clip_id=sample["clip_id"],
                          rendered=0, total=len(offsets))


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


def _draw_trajectory_on_image_scaled(
        img_hwc, hist_ego, fut_ego, pose, cam_model, scale, dot_radius=4):
    """Project GT trajectory using full-res model, scale pixel coords for display."""
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


def _draw_pred_on_image_scaled(
        img_hwc, traj_ego, color_rgb, pose, cam_model, scale, dot_radius=3):
    """Project a single predicted trajectory (median) onto a camera image."""
    img = Image.fromarray(img_hwc)
    draw = ImageDraw.Draw(img)
    for u, v in _project_onto_camera(traj_ego, pose, cam_model):
        u, v = int(round(u * scale)), int(round(v * scale))
        r = dot_radius
        draw.ellipse([u-r, v-r, u+r, v+r], fill=color_rgb, outline="white")
    return np.array(img)


def _build_camera_grid(
        frames: dict[int, np.ndarray],
        hist_ego: np.ndarray, fut_ego: np.ndarray,
        cam_models: dict, cam_poses: dict,
        show_gt: bool = True,
        pred_medians_ego: list | None = None) -> np.ndarray:
    """Annotate frames and assemble into 2×3 grid.

    Args:
        pred_medians_ego: List of (median_traj [T,3], color_rgb tuple) pairs,
            one per active prediction condition. Median is used here because
            projecting K=6-16 full trajectory distributions onto fisheye camera
            views would be unreadably cluttered.
    """
    sample_frame = next(iter(frames.values()))
    orig_h, orig_w = sample_frame.shape[:2]
    h = int(orig_h * DISPLAY_SCALE)
    w = int(orig_w * DISPLAY_SCALE)

    grid = np.zeros((2, 3, h, w, 3), dtype=np.uint8)

    for cam_idx, frame in frames.items():
        cam_name = CAM_IDX_TO_FEATURE[cam_idx]
        img = np.array(Image.fromarray(frame).resize((w, h), Image.BILINEAR))

        if cam_name in cam_poses and cam_name in cam_models:
            if show_gt:
                img = _draw_trajectory_on_image_scaled(
                    img, hist_ego, fut_ego,
                    cam_poses[cam_name], cam_models[cam_name],
                    DISPLAY_SCALE)
            if pred_medians_ego:
                for median_traj, color_rgb in pred_medians_ego:
                    img = _draw_pred_on_image_scaled(
                        img, median_traj, color_rgb,
                        cam_poses[cam_name], cam_models[cam_name],
                        DISPLAY_SCALE)

        if cam_idx in CAMERA_GRID_LAYOUT:
            r, c = CAMERA_GRID_LAYOUT[cam_idx]
            grid[r, c] = img

    return np.concatenate(
        [np.concatenate(grid[r], axis=1) for r in range(2)], axis=0)


# ---------------------------------------------------------------------------
# Render cache: (clip_id, t0_us) -> (frames, hist_ego, fut_ego)
# Stores raw data; annotated outputs are built on demand (PIL drawing is fast).
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
    except Exception as e:
        import traceback
        print(f"[render] error {clip_data.clip_id} t0={t0_us}: {e}")
        traceback.print_exc()
        return None
    result = (frames, hist_ego, fut_ego)
    with _render_lock:
        _render_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Preloader
# ---------------------------------------------------------------------------

_preload_executor   = ThreadPoolExecutor(max_workers=2)
_preload_generation = 0
_preload_gen_lock   = threading.Lock()


def _preload_worker(clip_id: str, base_t0_us: int, offsets: np.ndarray, generation: int):
    with _preload_gen_lock:
        if generation != _preload_generation:
            return
    try:
        clip_data = _get_clip_data(clip_id)
    except Exception as e:
        print(f"[preload] error loading clip {clip_id}: {e}")
        _set_preload_status(phase="idle")
        return

    total = len(offsets)

    _set_preload_status(phase="decoding", clip_id=clip_id, rendered=0, total=total)
    clip_data.preload_range(base_t0_us, offsets)

    _set_preload_status(phase="rendering", clip_id=clip_id, rendered=0, total=total)
    for i, off in enumerate(offsets):
        with _preload_gen_lock:
            if generation != _preload_generation:
                _set_preload_status(phase="idle")
                return
        t0_us = base_t0_us + int(round(float(off), 1) * 1_000_000)
        _render_at_t0(clip_data, t0_us)
        _set_preload_status(phase="rendering", clip_id=clip_id, rendered=i + 1, total=total)

    _set_preload_status(phase="rendering", clip_id=clip_id, rendered=total, total=total)


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
# Public render functions
# ---------------------------------------------------------------------------

def _make_bev_img(
        hist_ego, fut_ego, t0_us: int,
        show_gt: bool = True,
        pred_with_nav: np.ndarray | None = None,
        pred_no_nav: np.ndarray | None = None,
        pred_counterfactual: np.ndarray | None = None) -> np.ndarray:
    """Render BEV image with GT and/or prediction overlays.

    Predictions are rendered using plot_condition() (faint individual samples +
    bold median + KDE on endpoints), showing the full trajectory distribution.
    GT is rendered on top so it stays visible.
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

    # Prediction conditions (drawn first, GT goes on top)
    preds = [
        (pred_with_nav,       "tab:blue",  "with nav"),
        (pred_no_nav,         "tab:red",   "no nav"),
        (pred_counterfactual, "tab:orange", "counterfactual"),
    ]
    for pred, color, label in preds:
        if pred is not None:
            # plot_condition expects [K, T, 2]
            plot_condition(ax, pred[:, :, :2], color, label)

    # GT on top
    if show_gt:
        ax.plot(hist_ego[:, 0], hist_ego[:, 1], "o--", color="gray",  lw=1.5, ms=3, label="History")
        ax.plot(fut_ego[:, 0],  fut_ego[:, 1],  "o-",  color="green", lw=2,   ms=3, label="Future")

    ax.plot(0, 0, marker=">", color="black", ms=9, label="Ego (t0)", zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_xlim(-10, 50); ax.set_ylim(-25, 25)
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"t0 = {t0_us / 1e6:.1f}s", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return buf[:, :, :3]


def render_sample(
        sample: dict, t0_offset_s: float,
        show_gt: bool = True,
        show_with_nav: bool = False,
        show_no_nav: bool = False,
        show_counterfactual: bool = False,
        blocking: bool = True):
    clip_id = sample["clip_id"]
    t0_us   = sample["t0_us"] + int(round(t0_offset_s, 1) * 1_000_000)

    try:
        clip_data = _get_clip_data(clip_id, blocking=blocking)
    except Exception as e:
        blank = np.zeros((100, 400, 3), dtype=np.uint8)
        return blank, blank, f"Error loading clip: {e}"

    if clip_data is None:
        blank = np.zeros((100, 400, 3), dtype=np.uint8)
        return blank, blank, "Loading clip…"

    result = _render_at_t0(clip_data, t0_us)
    if result is None:
        blank = np.zeros((100, 400, 3), dtype=np.uint8)
        return blank, blank, f"Error rendering {clip_id} t0={t0_us/1e6:.1f}s"

    frames, hist_ego, fut_ego = result

    # Load inference results if any condition is toggled on
    inf_result = None
    if show_with_nav or show_no_nav or show_counterfactual:
        inf_result = InferenceResultStore.load_result(clip_id, t0_us)

    # Build camera grid: GT overlay + median per active pred condition
    # (median is used for camera projection; full distribution is shown in BEV)
    pred_medians_ego: list[tuple[np.ndarray, tuple]] = []
    if inf_result:
        for key, _color_str, color_rgb, _label in _COND_DEFS:
            active = (key == "with_nav" and show_with_nav) or \
                     (key == "no_nav" and show_no_nav) or \
                     (key == "counterfactual" and show_counterfactual)
            if active:
                median_traj = np.median(inf_result[key], axis=0)  # [T, 3]
                pred_medians_ego.append((median_traj, color_rgb))

    cam_grid = _build_camera_grid(
        frames, hist_ego, fut_ego,
        clip_data.cam_models, clip_data.cam_poses,
        show_gt=show_gt,
        pred_medians_ego=pred_medians_ego if pred_medians_ego else None)

    bev = _make_bev_img(
        hist_ego, fut_ego, t0_us,
        show_gt=show_gt,
        pred_with_nav=inf_result["with_nav"]       if (inf_result and show_with_nav)       else None,
        pred_no_nav=inf_result["no_nav"]           if (inf_result and show_no_nav)         else None,
        pred_counterfactual=inf_result["counterfactual"] if (inf_result and show_counterfactual) else None,
    )

    lines = [f"clip_id : {clip_id}", f"t0      : {t0_us/1e6:.3f}s"]
    if sample["nav_text"]:
        lines += [f"nav     : {sample['nav_text']}",
                  f"maneuver: {sample['nav_maneuver']}  ({sample['distance_m']:.1f}m)"]
    if sample["cot"]:
        lines.append(f"cot     : {sample['cot']}")
    if inf_result and (show_with_nav or show_no_nav or show_counterfactual):
        if inf_result.get("cot_with_nav") and show_with_nav:
            lines.append(f"cot(nav): {inf_result['cot_with_nav']}")
        if inf_result.get("cot_no_nav") and show_no_nav:
            lines.append(f"cot(no-nav): {inf_result['cot_no_nav']}")
    return cam_grid, bev, "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def _sample_ui_updates(sample: dict):
    """Compute UI update values that depend on which sample is selected."""
    has_results = InferenceResultStore.has_any(sample["clip_id"])
    has_nav     = bool(sample.get("nav_text"))
    nav_display = sample.get("nav_text") or "(no nav text for this clip)"
    return has_results, has_nav, nav_display


def on_select_step1(label: str):
    idx = SAMPLE_LABELS.index(label)
    return idx, False, "▶ Play"

def on_select_step2(t0_offset_s: float, idx: int):
    _kill_inference_if_running()
    sample = SAMPLES[int(idx)]
    _set_preload_status(phase="decoding", clip_id=sample["clip_id"], rendered=0, total=0)
    cam, bev, meta = render_sample(sample, t0_offset_s, show_gt=True)
    _schedule_preload(sample, t0_offset_s)
    has_results, has_nav, nav_display = _sample_ui_updates(sample)
    return (cam, bev, meta,
            gr.update(value=False, interactive=has_results),
            gr.update(value=False, interactive=has_results),
            gr.update(value=False, interactive=has_results),
            nav_display,
            gr.update(interactive=has_nav and not _inference_running()),
            get_preload_status_html(),
            get_inference_status_html())


def on_prev_step1(current_idx: int):
    idx = max(0, int(current_idx) - 1)
    return idx, SAMPLE_LABELS[idx], False, "▶ Play"

def on_prev_step2(t0_offset_s: float, idx: int):
    _kill_inference_if_running()
    sample = SAMPLES[int(idx)]
    _set_preload_status(phase="decoding", clip_id=sample["clip_id"], rendered=0, total=0)
    cam, bev, meta = render_sample(sample, t0_offset_s, show_gt=True)
    _schedule_preload(sample, t0_offset_s)
    has_results, has_nav, nav_display = _sample_ui_updates(sample)
    return (cam, bev, meta,
            gr.update(value=False, interactive=has_results),
            gr.update(value=False, interactive=has_results),
            gr.update(value=False, interactive=has_results),
            nav_display,
            gr.update(interactive=has_nav and not _inference_running()),
            get_preload_status_html(),
            get_inference_status_html())


def on_next_step1(current_idx: int):
    idx = min(len(SAMPLES) - 1, int(current_idx) + 1)
    return idx, SAMPLE_LABELS[idx], False, "▶ Play"

def on_next_step2(t0_offset_s: float, idx: int):
    _kill_inference_if_running()
    sample = SAMPLES[int(idx)]
    _set_preload_status(phase="decoding", clip_id=sample["clip_id"], rendered=0, total=0)
    cam, bev, meta = render_sample(sample, t0_offset_s, show_gt=True)
    _schedule_preload(sample, t0_offset_s)
    has_results, has_nav, nav_display = _sample_ui_updates(sample)
    return (cam, bev, meta,
            gr.update(value=False, interactive=has_results),
            gr.update(value=False, interactive=has_results),
            gr.update(value=False, interactive=has_results),
            nav_display,
            gr.update(interactive=has_nav and not _inference_running()),
            get_preload_status_html(),
            get_inference_status_html())


def on_t0_change(current_idx: int, t0_offset_s: float,
                 show_gt: bool, show_with_nav: bool, show_no_nav: bool, show_counterfactual: bool):
    cam, bev, meta = render_sample(
        SAMPLES[int(current_idx)], t0_offset_s,
        show_gt=show_gt, show_with_nav=show_with_nav,
        show_no_nav=show_no_nav, show_counterfactual=show_counterfactual)
    return cam, bev, meta


def on_toggle(current_idx: int, t0_offset_s: float,
              show_gt: bool, show_with_nav: bool, show_no_nav: bool, show_counterfactual: bool):
    cam, bev, _ = render_sample(
        SAMPLES[int(current_idx)], t0_offset_s,
        show_gt=show_gt, show_with_nav=show_with_nav,
        show_no_nav=show_no_nav, show_counterfactual=show_counterfactual)
    return cam, bev


def on_play_pause(is_playing: bool):
    new_playing = not is_playing
    return new_playing, "⏸ Pause" if new_playing else "▶ Play"


def on_tick(current_idx: int, t0_offset_s: float, is_playing: bool,
            show_gt: bool, show_with_nav: bool, show_no_nav: bool, show_counterfactual: bool):
    if not is_playing:
        return gr.update(), gr.update(), gr.update(), gr.update()
    sample = SAMPLES[int(current_idx)]
    if _get_clip_data(sample["clip_id"], blocking=False) is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    next_t0 = round(t0_offset_s + SLIDER_STEP, 1)
    if next_t0 > SLIDER_MAX:
        return gr.update(value=t0_offset_s), gr.update(), gr.update(), gr.update()
    cam, bev, meta = render_sample(
        sample, next_t0, blocking=False,
        show_gt=show_gt, show_with_nav=show_with_nav,
        show_no_nav=show_no_nav, show_counterfactual=show_counterfactual)
    return gr.update(value=next_t0), cam, bev, meta


def on_progress_tick(current_idx: int, t0_offset_s: float):
    """Called by progress_timer; returns both status HTMLs + dynamic UI state.

    When inference just completed for the currently-displayed clip, also
    auto-enables all 3 prediction toggles, re-renders, and clears the flag.
    """
    global _inference_completed_clip
    preload_html   = get_preload_status_html()
    inference_html = get_inference_status_html()
    sample = SAMPLES[int(current_idx)]
    has_results = InferenceResultStore.has_any(sample["clip_id"])
    has_nav     = bool(sample.get("nav_text"))
    running     = _inference_running()

    # Auto-show: inference just finished for the current clip
    if _inference_completed_clip and _inference_completed_clip == sample["clip_id"]:
        _inference_completed_clip = None
        cam, bev, meta = render_sample(
            sample, t0_offset_s,
            show_gt=True, show_with_nav=True, show_no_nav=True, show_counterfactual=True)
        return (preload_html, inference_html,
                gr.update(value=True, interactive=True),
                gr.update(value=True, interactive=True),
                gr.update(value=True, interactive=True),
                gr.update(interactive=has_nav and not running),
                cam, bev, meta)

    return (preload_html, inference_html,
            gr.update(interactive=has_results),
            gr.update(interactive=has_results),
            gr.update(interactive=has_results),
            gr.update(interactive=has_nav and not running),
            gr.update(), gr.update(), gr.update())


def on_run_inference(current_idx: int, num_samples_val: int, max_frames_val: int):
    sample = SAMPLES[int(current_idx)]
    if not sample.get("nav_text"):
        return gr.update()
    launch_inference_for_clip(sample, int(num_samples_val), int(max_frames_val))
    return gr.update(interactive=False)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_css = """
#cam-col { flex: 7 1 0% !important; min-width: 0 !important; }
#bev-col { flex: 3 1 0% !important; min-width: 0 !important; }
"""

_TOGGLE_INPUTS = None  # filled after UI components are defined

with gr.Blocks(title="PhysicalAI-AV GT Browser") as demo:
    gr.Markdown(
        "## PhysicalAI-AV Ground-Truth Browser\n"
        "Gray = history GT · Green = future GT"
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

    # Display toggles
    with gr.Row():
        chk_gt           = gr.Checkbox(value=True,  label="GT",                  scale=1)
        chk_with_nav     = gr.Checkbox(value=False, label="Pred: with nav",       scale=1, interactive=False)
        chk_no_nav       = gr.Checkbox(value=False, label="Pred: no nav",         scale=1, interactive=False)
        chk_counterfactual = gr.Checkbox(value=False, label="Pred: counterfactual", scale=1, interactive=False)

    with gr.Row():
        preload_status_html   = gr.HTML(value=get_preload_status_html(),   scale=1)
        inference_status_html = gr.HTML(value=get_inference_status_html(), scale=1)
    progress_timer = gr.Timer(value=0.4, active=True)

    # Inference accordion
    first_sample = SAMPLES[0]
    first_has_nav = bool(first_sample.get("nav_text"))
    with gr.Accordion("Run Inference", open=False):
        nav_text_display = gr.Textbox(
            value=first_sample.get("nav_text") or "(no nav text for this clip)",
            label="Nav text (from dataset, not editable)",
            interactive=False)
        num_samples_slider = gr.Slider(
            minimum=1, maximum=16, value=6, step=1,
            label="Number of trajectory samples per condition")
        max_frames_slider = gr.Slider(
            minimum=0, maximum=131, value=0, step=1,
            label="Max frames (0 = all 131; set small for quick testing)")
        run_infer_btn = gr.Button(
            "Run Inference for this clip",
            interactive=first_has_nav)

    with gr.Row():
        cam_out = gr.Image(
            label="Camera grid", type="numpy", elem_id="cam-col")
        with gr.Column(elem_id="bev-col"):
            bev_out  = gr.Image(label="BEV", type="numpy")
            meta_out = gr.Textbox(label="Metadata", lines=5, interactive=False)

    play_timer = gr.Timer(value=0.5, active=True)

    # Shared toggle inputs list
    _toggle_inputs = [current_idx, t0_slider, chk_gt, chk_with_nav, chk_no_nav, chk_counterfactual]
    _nav_extra_outputs = [chk_with_nav, chk_no_nav, chk_counterfactual, nav_text_display, run_infer_btn,
                          preload_status_html, inference_status_html]

    # Sample navigation
    sample_dd.change(on_select_step1, [sample_dd], [current_idx, is_playing, btn_play])\
             .then(on_select_step2,   [t0_slider, current_idx],
                   [cam_out, bev_out, meta_out] + _nav_extra_outputs)
    btn_prev.click(on_prev_step1,    [current_idx],
                   [current_idx, sample_dd, is_playing, btn_play])\
            .then(on_prev_step2,     [t0_slider, current_idx],
                  [cam_out, bev_out, meta_out] + _nav_extra_outputs)
    btn_next.click(on_next_step1,    [current_idx],
                   [current_idx, sample_dd, is_playing, btn_play])\
            .then(on_next_step2,     [t0_slider, current_idx],
                  [cam_out, bev_out, meta_out] + _nav_extra_outputs)

    # t0 slider and toggles
    t0_slider.release(on_t0_change,  _toggle_inputs,          [cam_out, bev_out, meta_out])
    for chk in [chk_gt, chk_with_nav, chk_no_nav, chk_counterfactual]:
        chk.change(on_toggle, _toggle_inputs, [cam_out, bev_out])

    # Play controls
    btn_play.click(on_play_pause,    [is_playing],             [is_playing, btn_play])
    play_timer.tick(on_tick,
                    [current_idx, t0_slider, is_playing, chk_gt, chk_with_nav, chk_no_nav, chk_counterfactual],
                    [t0_slider, cam_out, bev_out, meta_out])

    # Progress timer — updates both status bars + toggle/button interactivity
    # Also outputs cam/bev/meta for the auto-show when inference completes
    progress_timer.tick(on_progress_tick,
                        [current_idx, t0_slider],
                        [preload_status_html, inference_status_html,
                         chk_with_nav, chk_no_nav, chk_counterfactual, run_infer_btn,
                         cam_out, bev_out, meta_out])

    # Inference launch
    run_infer_btn.click(on_run_inference,
                        [current_idx, num_samples_slider, max_frames_slider],
                        [run_infer_btn])

    # Initial load
    demo.load(
        lambda: on_select_step2(0.0, 0),
        outputs=[cam_out, bev_out, meta_out] + _nav_extra_outputs)
    # Note: _nav_extra_outputs now includes preload_status_html and inference_status_html


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=_css)
