"""Tests for InferenceResultStore and related browse_gt.py utilities.

These tests are self-contained: they do not require GPU, network access,
or any dataset files. All I/O is redirected into a pytest tmp_path.
"""

import importlib
import json
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to import browse_gt without triggering top-level side effects
# (AVDI init, SAMPLES load, Gradio UI construction).
# ---------------------------------------------------------------------------

def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _stub_heavy_imports():
    """Stub out all imports that touch the network or GPU before browse_gt loads."""
    # physical_ai_av
    pav = _make_stub_module("physical_ai_av")
    pav.PhysicalAIAVDatasetInterface = MagicMock()

    # gradio — stub just enough for the module-level `with gr.Blocks` to not crash
    gr_mod = _make_stub_module("gradio")
    for cls in ("Blocks", "Row", "Column", "Accordion", "Dropdown", "Slider",
                "Checkbox", "Button", "Image", "Textbox", "HTML", "Timer",
                "State", "Markdown", "update"):
        setattr(gr_mod, cls, MagicMock())
    gr_mod.Blocks.return_value.__enter__ = lambda s, *a, **kw: s
    gr_mod.Blocks.return_value.__exit__ = MagicMock(return_value=False)

    # pandas (used in _load_samples)
    pd_mod = _make_stub_module("pandas")
    fake_df = MagicMock()
    fake_df.__getitem__ = MagicMock(return_value=[])
    pd_mod.read_parquet = MagicMock(return_value=fake_df)

    # alpamayo1_5.viz_utils
    viz = _make_stub_module("alpamayo1_5")
    viz_utils = _make_stub_module("alpamayo1_5.viz_utils")
    viz_utils.CAMERA_GRID_LAYOUT = {0: (1, 0), 1: (1, 1), 2: (1, 2), 6: (0, 1)}
    viz_utils.plot_condition = MagicMock()
    sys.modules["alpamayo1_5.viz_utils"] = viz_utils

    # scipy, PIL, matplotlib — let the real ones load (they are installed)


@pytest.fixture(scope="module")
def browse_gt(tmp_path_factory):
    """Import browse_gt with heavy deps stubbed out.

    Returns the module object so tests can access InferenceResultStore,
    get_status_html, _set_status, _make_bev_img, etc.
    """
    _stub_heavy_imports()

    tmp = tmp_path_factory.mktemp("browse_root")

    # browse_gt reads notebooks/nav_demo_samples.json and clip_ids.parquet
    # at import time — provide minimal stubs on disk.
    nb_dir = tmp / "notebooks"
    nb_dir.mkdir()
    (nb_dir / "nav_demo_samples.json").write_text(json.dumps([
        {
            "clip_id": "aaaabbbb-0000-0000-0000-000000000001",
            "t0_relative": 4_000_000,
            "nav_text": "Turn right in 30m",
            "nav_maneuver": "TURN_RIGHT",
            "distance_m": 30.0,
            "cot": "Keep distance to lead vehicle.",
        }
    ]))
    # clip_ids.parquet is loaded via pandas (already stubbed), so no real file needed.

    # Patch open() for nav_demo_samples.json to use our tmp file, and
    # patch Path("inference_results") to use tmp_path.
    inf_dir = tmp / "inference_results"
    inf_dir.mkdir()

    import os
    old_cwd = os.getcwd()
    os.chdir(tmp)

    try:
        import browse_gt as bg
    finally:
        os.chdir(old_cwd)

    # Point INFERENCE_DIR at our temp dir so store operations go there.
    bg.INFERENCE_DIR = inf_dir
    return bg, inf_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLIP_ID = "aaaabbbb-0000-0000-0000-000000000001"
T0_US   = 4_000_000
K, T    = 8, 64   # trajectory samples, timesteps


def _make_pred(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((K, T, 3)).astype(np.float32)


def _write_result(inf_dir: Path, clip_id: str, t0_us: int,
                  with_nav=None, no_nav=None, counterfactual=None,
                  meta: dict | None = None) -> Path:
    """Write a complete inference result set into inf_dir."""
    d = inf_dir / clip_id
    d.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(d / f"{t0_us}_with_nav.npz",       pred=with_nav       if with_nav       is not None else _make_pred(0))
    np.savez_compressed(d / f"{t0_us}_no_nav.npz",         pred=no_nav         if no_nav         is not None else _make_pred(1))
    np.savez_compressed(d / f"{t0_us}_counterfactual.npz", pred=counterfactual if counterfactual is not None else _make_pred(2))

    meta = meta or {
        "nav_text": "Turn right in 30m",
        "nav_text_swapped": "Turn left in 30m",
        "cot_with_nav": "Keep distance.",
        "cot_no_nav": "Proceed normally.",
        "t0_us": t0_us,
        "num_samples": K,
    }
    (d / f"{t0_us}_meta.json").write_text(json.dumps(meta))
    return d


# ---------------------------------------------------------------------------
# InferenceResultStore — result_paths
# ---------------------------------------------------------------------------

class TestResultPaths:
    def test_keys(self, browse_gt):
        bg, inf_dir = browse_gt
        paths = bg.InferenceResultStore.result_paths(CLIP_ID, T0_US)
        assert set(paths) == {"with_nav", "no_nav", "counterfactual", "meta"}

    def test_paths_under_clip_dir(self, browse_gt):
        bg, inf_dir = browse_gt
        paths = bg.InferenceResultStore.result_paths(CLIP_ID, T0_US)
        for key, path in paths.items():
            assert path.parent == bg.INFERENCE_DIR / CLIP_ID
            assert str(T0_US) in path.name

    def test_npz_extension(self, browse_gt):
        bg, inf_dir = browse_gt
        paths = bg.InferenceResultStore.result_paths(CLIP_ID, T0_US)
        for key in ("with_nav", "no_nav", "counterfactual"):
            assert paths[key].suffix == ".npz"
        assert paths["meta"].suffix == ".json"


# ---------------------------------------------------------------------------
# InferenceResultStore — has_result / has_any
# ---------------------------------------------------------------------------

class TestHasResult:
    def test_missing_all_files(self, browse_gt, tmp_path):
        bg, inf_dir = browse_gt
        # Use a fresh clip ID that has no files
        assert not bg.InferenceResultStore.has_result("no-such-clip", T0_US)

    def test_partial_files_returns_false(self, browse_gt, tmp_path):
        bg, inf_dir = browse_gt
        clip = "partial-clip-0000-0000-0000-000000000099"
        d = inf_dir / clip
        d.mkdir(parents=True, exist_ok=True)
        # Write only 2 of 4 files
        np.savez_compressed(d / f"{T0_US}_with_nav.npz", pred=_make_pred())
        np.savez_compressed(d / f"{T0_US}_no_nav.npz",   pred=_make_pred())
        assert not bg.InferenceResultStore.has_result(clip, T0_US)

    def test_all_files_present_returns_true(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "full-clip-0000-0000-0000-000000000001"
        _write_result(inf_dir, clip, T0_US)
        assert bg.InferenceResultStore.has_result(clip, T0_US)

    def test_has_any_no_dir(self, browse_gt):
        bg, inf_dir = browse_gt
        assert not bg.InferenceResultStore.has_any("no-such-clip-xyz")

    def test_has_any_empty_dir(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "empty-dir-clip-000000000001"
        (inf_dir / clip).mkdir(parents=True, exist_ok=True)
        assert not bg.InferenceResultStore.has_any(clip)

    def test_has_any_with_results(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "has-any-clip-000-0000-0000-000000000001"
        _write_result(inf_dir, clip, T0_US)
        assert bg.InferenceResultStore.has_any(clip)

    def test_has_any_multiple_t0s(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "multi-t0-clip-000-0000-0000-000000000001"
        _write_result(inf_dir, clip, T0_US)
        _write_result(inf_dir, clip, T0_US + 100_000)
        assert bg.InferenceResultStore.has_any(clip)


# ---------------------------------------------------------------------------
# InferenceResultStore — load_result
# ---------------------------------------------------------------------------

class TestLoadResult:
    def test_returns_none_when_missing(self, browse_gt):
        bg, inf_dir = browse_gt
        result = bg.InferenceResultStore.load_result("no-such-clip", T0_US)
        assert result is None

    def test_returns_none_when_partial(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "partial-load-clip-000-0000-0000-000000000001"
        d = inf_dir / clip
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{T0_US}_with_nav.npz", pred=_make_pred())
        # no_nav and counterfactual missing
        result = bg.InferenceResultStore.load_result(clip, T0_US)
        assert result is None

    def test_shapes(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "shape-test-clip-000-0000-0000-000000000001"
        with_nav = _make_pred(0)
        no_nav   = _make_pred(1)
        counter  = _make_pred(2)
        _write_result(inf_dir, clip, T0_US,
                      with_nav=with_nav, no_nav=no_nav, counterfactual=counter)
        result = bg.InferenceResultStore.load_result(clip, T0_US)
        assert result is not None
        assert result["with_nav"].shape      == (K, T, 3)
        assert result["no_nav"].shape        == (K, T, 3)
        assert result["counterfactual"].shape == (K, T, 3)

    def test_values_roundtrip(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "value-roundtrip-clip-0000-0000-000000000001"
        with_nav = _make_pred(42)
        _write_result(inf_dir, clip, T0_US, with_nav=with_nav)
        result = bg.InferenceResultStore.load_result(clip, T0_US)
        np.testing.assert_allclose(result["with_nav"], with_nav, rtol=1e-5)

    def test_meta_fields(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "meta-fields-clip-000-0000-0000-000000000001"
        meta = {
            "nav_text": "Turn right in 42m",
            "nav_text_swapped": "Turn left in 42m",
            "cot_with_nav": "Some reasoning.",
            "cot_no_nav": "Other reasoning.",
            "t0_us": T0_US,
            "num_samples": K,
        }
        _write_result(inf_dir, clip, T0_US, meta=meta)
        result = bg.InferenceResultStore.load_result(clip, T0_US)
        assert result["nav_text"]         == "Turn right in 42m"
        assert result["nav_text_swapped"] == "Turn left in 42m"
        assert result["cot_with_nav"]     == "Some reasoning."
        assert result["cot_no_nav"]       == "Other reasoning."

    def test_missing_meta_fields_default_empty(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "sparse-meta-clip-000-0000-0000-000000000001"
        _write_result(inf_dir, clip, T0_US, meta={"t0_us": T0_US})
        result = bg.InferenceResultStore.load_result(clip, T0_US)
        assert result["nav_text"] == ""
        assert result["cot_with_nav"] == ""

    def test_dtype_is_float32(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "dtype-clip-000-0000-0000-000000000001"
        _write_result(inf_dir, clip, T0_US)
        result = bg.InferenceResultStore.load_result(clip, T0_US)
        for key in ("with_nav", "no_nav", "counterfactual"):
            assert result[key].dtype == np.float32, f"{key} dtype wrong"

    def test_different_t0s_independent(self, browse_gt):
        bg, inf_dir = browse_gt
        clip = "two-t0s-clip-000-0000-0000-000000000001"
        t0a, t0b = 3_000_000, 5_000_000
        pred_a = _make_pred(10)
        pred_b = _make_pred(20)
        _write_result(inf_dir, clip, t0a, with_nav=pred_a)
        _write_result(inf_dir, clip, t0b, with_nav=pred_b)

        r_a = bg.InferenceResultStore.load_result(clip, t0a)
        r_b = bg.InferenceResultStore.load_result(clip, t0b)
        np.testing.assert_allclose(r_a["with_nav"], pred_a, rtol=1e-5)
        np.testing.assert_allclose(r_b["with_nav"], pred_b, rtol=1e-5)


# ---------------------------------------------------------------------------
# get_status_html / _set_status
# ---------------------------------------------------------------------------

class TestStatusHtml:
    def _reset(self, bg):
        bg._set_status(phase="idle", clip_id="", rendered=0, total=0)

    def test_idle(self, browse_gt):
        bg, _ = browse_gt
        self._reset(bg)
        html = bg.get_status_html()
        assert "—" in html

    def test_loading_clip(self, browse_gt):
        bg, _ = browse_gt
        bg._set_status(phase="loading_clip", clip_id="abcd1234-xxxx", rendered=0, total=0)
        html = bg.get_status_html()
        assert "Loading clip" in html
        assert "abcd1234" in html

    def test_decoding(self, browse_gt):
        bg, _ = browse_gt
        bg._set_status(phase="decoding", clip_id="abcd1234-xxxx", rendered=0, total=10)
        html = bg.get_status_html()
        assert "Batch-decoding" in html

    def test_rendering_progress(self, browse_gt):
        bg, _ = browse_gt
        bg._set_status(phase="rendering", clip_id="abcd1234-xxxx", rendered=5, total=10)
        html = bg.get_status_html()
        assert "5/10" in html
        assert "50%" in html

    def test_rendering_complete(self, browse_gt):
        bg, _ = browse_gt
        bg._set_status(phase="rendering", clip_id="abcd1234-xxxx", rendered=10, total=10)
        html = bg.get_status_html()
        assert "Ready" in html or "100%" in html

    def test_inference_running(self, browse_gt):
        bg, _ = browse_gt
        bg._set_status(phase="inference_running", clip_id="abcd1234-xxxx",
                       rendered=3, total=20)
        html = bg.get_status_html()
        assert "inference" in html.lower() or "⚙️" in html
        assert "3/20" in html

    def test_inference_done(self, browse_gt):
        bg, _ = browse_gt
        bg._set_status(phase="inference_done", clip_id="abcd1234-xxxx", rendered=0, total=0)
        html = bg.get_status_html()
        assert "complete" in html.lower() or "✅" in html

    def test_inference_error(self, browse_gt):
        bg, _ = browse_gt
        bg._set_status(phase="inference_error", clip_id="abcd1234-xxxx", rendered=0, total=0)
        html = bg.get_status_html()
        assert "failed" in html.lower() or "❌" in html

    def test_thread_safety(self, browse_gt):
        """Concurrent _set_status calls must not corrupt state."""
        bg, _ = browse_gt
        errors = []

        def writer(phase, n):
            for i in range(50):
                try:
                    bg._set_status(phase=phase, clip_id="x", rendered=i, total=100)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(p, i))
                   for i, p in enumerate(["idle", "rendering", "inference_running"])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # After all writes, get_status_html must not raise
        bg.get_status_html()


# ---------------------------------------------------------------------------
# _make_bev_img — output shape and content
# ---------------------------------------------------------------------------

class TestMakeBevImg:
    def _ego(self):
        t = np.linspace(0, 1, NUM_HISTORY := 16)
        hist = np.stack([t * 5, np.zeros(16), np.zeros(16)], axis=1)
        fut  = np.stack([np.linspace(1, 30, 64),
                         np.linspace(0, 5, 64),
                         np.zeros(64)], axis=1)
        return hist, fut

    def test_output_is_uint8_rgb(self, browse_gt):
        bg, _ = browse_gt
        hist, fut = self._ego()
        img = bg._make_bev_img(hist, fut, T0_US)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_output_nonzero(self, browse_gt):
        bg, _ = browse_gt
        hist, fut = self._ego()
        img = bg._make_bev_img(hist, fut, T0_US)
        assert img.max() > 0

    def test_with_pred_overlay_calls_plot_condition(self, browse_gt):
        """_make_bev_img with pred_with_nav should call plot_condition once."""
        bg, _ = browse_gt
        import alpamayo1_5.viz_utils as viz_utils
        viz_utils.plot_condition.reset_mock()

        hist, fut = self._ego()
        pred = _make_pred(0)
        bg._make_bev_img(hist, fut, T0_US, show_gt=True, pred_with_nav=pred)

        viz_utils.plot_condition.assert_called_once()
        call_args = viz_utils.plot_condition.call_args
        # Second positional arg is the trajectory array [K, T, 2]
        trajs_arg = call_args[0][1]
        assert trajs_arg.shape == (K, T, 2)

    def test_pred_overlay_uses_xy_only(self, browse_gt):
        """plot_condition must receive XY (first 2 dims), not full XYZ."""
        bg, _ = browse_gt
        import alpamayo1_5.viz_utils as viz_utils
        viz_utils.plot_condition.reset_mock()

        hist, fut = self._ego()
        pred = _make_pred(3)
        bg._make_bev_img(hist, fut, T0_US, pred_no_nav=pred)

        call_args = viz_utils.plot_condition.call_args
        trajs_arg = call_args[0][1]
        assert trajs_arg.shape[-1] == 2, "plot_condition should receive [K, T, 2], not [K, T, 3]"

    def test_gt_off_removes_gt_lines(self, browse_gt):
        """show_gt=False should produce a different image than show_gt=True."""
        bg, _ = browse_gt
        hist, fut = self._ego()
        img_with_gt    = bg._make_bev_img(hist, fut, T0_US, show_gt=True)
        img_without_gt = bg._make_bev_img(hist, fut, T0_US, show_gt=False)
        assert not np.array_equal(img_with_gt, img_without_gt)

    def test_no_preds_no_crash(self, browse_gt):
        bg, _ = browse_gt
        hist, fut = self._ego()
        # All pred args None, show_gt False — should still produce a valid image
        img = bg._make_bev_img(hist, fut, T0_US,
                               show_gt=False,
                               pred_with_nav=None,
                               pred_no_nav=None,
                               pred_counterfactual=None)
        assert img.shape[2] == 3


# ---------------------------------------------------------------------------
# inference_worker.py — PROGRESS line parsing (unit test of the parsing logic
# used by the monitor thread in browse_gt.py)
# ---------------------------------------------------------------------------

class TestProgressParsing:
    """Test the PROGRESS line format produced by inference_worker and consumed
    by the monitor thread."""

    @staticmethod
    def _parse(line: str):
        """Mirror of the parsing logic in launch_inference_for_clip's monitor thread."""
        if not line.startswith("PROGRESS "):
            return None
        parts = line.split()
        if len(parts) >= 2 and "/" in parts[1]:
            done_str, total_str = parts[1].split("/")
            return int(done_str), int(total_str)
        return None

    def test_parses_zero(self):
        assert self._parse("PROGRESS 0/131") == (0, 131)

    def test_parses_mid(self):
        assert self._parse("PROGRESS 65/131") == (65, 131)

    def test_parses_done(self):
        assert self._parse("PROGRESS 131/131") == (131, 131)

    def test_ignores_other_lines(self):
        assert self._parse("Loading model...") is None
        assert self._parse("ERROR t0=1234: something") is None
        assert self._parse("DONE") is None
        assert self._parse("") is None

    def test_ignores_malformed(self):
        assert self._parse("PROGRESS noSlash") is None
        assert self._parse("PROGRESS") is None
