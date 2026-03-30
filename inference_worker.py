"""Offline inference worker for Alpamayo 1.5 nav-comparison results.

Loads the model once, runs 3-condition inference (with-nav, no-nav, counterfactual)
for every t0 offset in a clip, and saves results to disk.

Progress is reported on stdout as "PROGRESS n/total" lines.
browse_gt.py launches this as a subprocess and monitors stdout to update
the browser status bar.

Usage:
    python inference_worker.py \
        --clip_id <uuid> \
        --base_t0_us <int> \
        --offsets_json '[-3.0, -2.9, ..., 10.0]' \
        --nav_text "Turn right in 30m" \
        --out_dir inference_results \
        --num_samples 6
"""

import argparse
import gc
import json
import os
import traceback
from pathlib import Path

# Must be set before torch initializes CUDA to avoid fragmentation.
# 8-bit quantization leaves 2+ GiB reserved-but-fragmented; this lets
# PyTorch reuse those fragments for large activation allocations.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from transformers import BitsAndBytesConfig

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper, nav_utils
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset


def _patch_image_num_counting(vlm):
    """Monkeypatch Qwen3VL to count images via input_ids, not inputs_embeds.

    When inputs_embeds is provided with visual features pre-injected, the
    default implementation tries to identify image positions by comparing
    embedding vectors — this fails because those positions now hold actual
    visual features, not the raw image_token_id embedding.  Since we always
    pass input_ids (which still contains image_token_id values), using
    input_ids is both correct and simpler.
    """
    from types import MethodType

    def _count_from_input_ids(self, input_ids, inputs_embeds=None):
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_start_mask = input_ids == vision_start_token_id
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)
        return image_nums, video_nums

    vlm._get_image_nums_and_video_nums = MethodType(_count_from_input_ids, vlm)


def _tokenize(processor, data, nav_text, use_nav_prompt=False):
    """Tokenize one condition; returns processor output (CPU tensors)."""
    frames = data["image_frames"].flatten(0, 1)
    camera_indices = data.get("camera_indices")
    messages = helper.create_message(
        frames, camera_indices=camera_indices,
        nav_text=nav_text, use_nav_prompt=use_nav_prompt,
    )
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )


def _inject_image_embeds(model, tok, image_embeds_flat, image_grid_thw):
    """Build inputs_embeds with image patches pre-filled from the shared encoder.

    Keeps input_ids (needed by fuse_traj_tokens inside the model), sets
    inputs_embeds so Qwen3VL uses these instead of re-encoding pixel_values,
    and omits pixel_values so the vision encoder is skipped.
    """
    input_ids = tok["input_ids"].cuda()
    attn_mask = tok["attention_mask"].cuda()

    # Build inputs_embeds: text tokens + image patches merged in
    with torch.no_grad():
        inputs_embeds = model.vlm.get_input_embeddings()(input_ids)  # [1, L, D]
        split_sizes = (image_grid_thw.prod(-1) // model.vlm.visual.spatial_merge_size ** 2).tolist()
        image_embeds_cat = torch.cat(
            torch.split(image_embeds_flat, split_sizes), dim=0
        ).to(inputs_embeds.device, inputs_embeds.dtype)
        image_token_id = model.vlm.config.image_token_id
        image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat).detach()

    return {
        "input_ids":      input_ids,       # model pops this for fuse_traj_tokens
        "inputs_embeds":  inputs_embeds,   # Qwen3VL uses this, skipping vision encoder
        "attention_mask": attn_mask,
        "image_grid_thw": image_grid_thw.cuda(),
        # pixel_values omitted — vision encoder will be skipped
    }


def _run_all_conditions(model, processor, data, nav_text, nav_text_swapped, num_samples):
    """Run 3 nav conditions, encoding camera images only once.

    The vision encoder runs a single time on the first condition's pixel_values.
    The resulting image embeddings are reused for the other two conditions,
    which never load pixel_values onto the GPU.

    Returns (pred_with_nav, pred_no_nav, pred_counterfactual, cot_with_nav)
    each shape [K, T, 3] float32 numpy.
    """
    # Tokenize first condition and encode images once.
    # no_grad is critical: without it the vision encoder stores gradient tensors
    # that consume ~10 GiB of activations on top of the model weights.
    tok_nav = _tokenize(processor, data, nav_text)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pv  = tok_nav.pop("pixel_values").cuda()
        thw = tok_nav["image_grid_thw"].cuda()
        image_embeds, _ = model.vlm.get_image_features(pv, thw)
        image_embeds_flat = torch.cat(image_embeds, dim=0).detach()  # [N_patches, D]
        del pv

    def _make_model_inputs(tok):
        """Build inputs with pre-computed image embeds (no pixel_values on GPU)."""
        tok.pop("pixel_values", None)  # already encoded; don't send to GPU
        injected = _inject_image_embeds(model, tok, image_embeds_flat, thw)
        return helper.to_device({
            "tokenized_data":  injected,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }, "cuda")

    def _run(model_inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_samples,
                max_generation_length=256,
                return_extra=True,
            )
        pred_xyz = outputs[0][0, 0].float().cpu().numpy()
        extra = outputs[2] if len(outputs) > 2 else None
        cot = ""
        if extra and "cot" in extra:
            cot_val = extra["cot"]
            if cot_val is not None and len(cot_val) > 0:
                cot = str(cot_val[0])
        return pred_xyz, cot

    inp = _make_model_inputs(tok_nav)
    pred_with_nav, cot_with_nav = _run(inp)
    del inp

    tok_none = _tokenize(processor, data, None, use_nav_prompt=True)
    inp = _make_model_inputs(tok_none)
    pred_no_nav, _ = _run(inp)
    del inp

    tok_swap = _tokenize(processor, data, nav_text_swapped)
    inp = _make_model_inputs(tok_swap)
    pred_counterfactual, _ = _run(inp)
    del inp

    return pred_with_nav, pred_no_nav, pred_counterfactual, cot_with_nav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--base_t0_us", type=int, required=True)
    ap.add_argument("--offsets_json", required=True,
                    help="JSON list of float offsets in seconds from base_t0_us")
    ap.add_argument("--nav_text", required=True)
    ap.add_argument("--out_dir", default="inference_results")
    ap.add_argument("--num_samples", type=int, default=6)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Cap number of frames to process (0 = all)")
    args = ap.parse_args()

    offsets = json.loads(args.offsets_json)
    if args.max_frames > 0:
        offsets = offsets[:args.max_frames]

    out_dir = Path(args.out_dir) / args.clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(offsets)

    print("Loading model...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder="/tmp/alpamayo_offload",
        offload_state_dict=True,
    )
    model.tie_weights()
    _patch_image_num_counting(model.vlm)
    torch.cuda.empty_cache()
    processor = helper.get_processor(model.tokenizer)

    import physical_ai_av
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    done = 0
    print(f"PROGRESS {done}/{total}", flush=True)

    for off in offsets:
        t0_us = args.base_t0_us + int(round(float(off), 1) * 1_000_000)

        paths = {
            "with_nav":      out_dir / f"{t0_us}_with_nav.npz",
            "no_nav":        out_dir / f"{t0_us}_no_nav.npz",
            "counterfactual": out_dir / f"{t0_us}_counterfactual.npz",
            "meta":          out_dir / f"{t0_us}_meta.json",
        }

        if all(p.exists() for p in paths.values()):
            done += 1
            print(f"PROGRESS {done}/{total}", flush=True)
            continue

        try:
            data = load_physical_aiavdataset(
                args.clip_id, t0_us=t0_us, avdi=avdi
            )
            nav_text_swapped = nav_utils.swap_direction(args.nav_text)

            pred_with_nav, pred_no_nav, pred_counterfactual, cot_with_nav = \
                _run_all_conditions(
                    model, processor, data,
                    args.nav_text, nav_text_swapped, args.num_samples,
                )

            np.savez_compressed(paths["with_nav"],      pred=pred_with_nav)
            np.savez_compressed(paths["no_nav"],        pred=pred_no_nav)
            np.savez_compressed(paths["counterfactual"], pred=pred_counterfactual)

            meta = {
                "nav_text": args.nav_text,
                "nav_text_swapped": nav_text_swapped,
                "cot_with_nav": cot_with_nav,
                "cot_no_nav": "",
                "t0_us": t0_us,
                "num_samples": args.num_samples,
            }
            paths["meta"].write_text(json.dumps(meta, indent=2))

        except Exception as e:
            print(f"ERROR t0={t0_us}: {e}", flush=True)
            traceback.print_exc()

        done += 1
        print(f"PROGRESS {done}/{total}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
