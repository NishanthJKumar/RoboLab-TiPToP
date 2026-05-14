# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Episode runner for the TipTop + VLM execution-monitor hybrid.

TipTop is a planner: each plan step carries a label like "Pick(banana, grasp1, q1)"
or "Place(banana, place1)". Consecutive steps sharing the same action (Pick/Place)
form one logical subtask. This runner watches `client.current_subtask_label` for
transitions and, when a Pick or Place subtask ends, schedules a VLM check
`vlm_check_delay_steps` later asking Gemini whether it actually succeeded — using
the planner's own action name and target object to ground the prompt.

Results are *passively logged* — the episode is never terminated or replanned
based on the VLM's judgements. They're written to `vlm_checks_{episode}.json`
alongside the harness log and per-event before/transition/check PNGs (under
`vlm_debug_{episode}/`) for offline analysis.

Single-env only (TipTop limitation).
"""

import json
import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from episode import TimingStats

from robolab.constants import VISUALIZE, get_output_dir
from robolab.core.logging.results import get_all_env_subtask_infos
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter
from robolab.core.world.world_state import get_world
from robolab.inference.tiptop import TiptopWebsocketClient
from robolab.inference.vlm_done_checker import ProgressMonitor

logger = logging.getLogger(__name__)

# Only these action kinds (parsed from plan-step labels) trigger VLM checks for now.
_SUPPORTED_ACTIONS = {"pick", "place"}


def _parse_subtask_label(label: str | None) -> tuple[str | None, str | None]:
    """Parse a TipTop plan-step label.

    Examples:
        "Pick(banana, grasp1, q1)" -> ("pick", "banana")
        "Place(banana, place1)"    -> ("place", "banana")
        "Approach(banana)"          -> (None, None)   # not a supported action
        None                        -> (None, None)

    Returns (kind, target) where `kind` is in `_SUPPORTED_ACTIONS` or None,
    and `target` is the first positional argument inside the parentheses
    (typically the object name) or None.
    """
    if not label:
        return None, None
    m = re.match(r"\s*(\w+)\s*\(\s*([^,)]*)", label)
    if not m:
        return None, None
    action = m.group(1).strip().lower()
    if action not in _SUPPORTED_ACTIONS:
        return None, None
    target = m.group(2).strip() or None
    return action, target


def _build_vlm_view(obs, env_id: int = 0, scale: float = 0.5):
    """Compose the VLM input frame: viewport (third-person scene) | wrist (gripper close-up).

    For TipTop envs, `image_obs` only contains `wrist_cam` (external_cam is dropped to
    save VRAM at plan time), so the third-person view lives in the separate `viewport_cam`
    obs group. We resize both to a shared height and concatenate horizontally so the VLM
    sees scene context and grasp/release detail in one image.
    """
    image_obs = unpack_image_obs(obs, scale=scale, env_id=env_id)
    viewport = unpack_viewport_cams(obs, scale=scale, env_id=env_id)

    wrist = image_obs.get("wrist_cam")
    external = viewport.get("combined_image") if isinstance(viewport, dict) else None

    panels = [p for p in (external, wrist) if p is not None]
    if not panels:
        # Last-resort fallback: whatever combined_image was in image_obs.
        return image_obs.get("combined_image")
    if len(panels) == 1:
        return panels[0]

    target_h = min(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != target_h:
            new_w = int(p.shape[1] * target_h / p.shape[0])
            p = cv2.resize(p, (new_w, target_h))
        resized.append(p)
    return np.concatenate(resized, axis=1)


_VIEW_LAYOUT_NOTE = (
    "You are given TWO composite images: the FIRST is the BEFORE frame "
    "(the scene at the previous event boundary, or episode start for the first "
    "check) and the SECOND is the AFTER frame (the current scene). Each composite "
    "is itself laid out side-by-side: the LEFT panel is a third-person view of "
    "the full scene, and the RIGHT panel is a close-up from the gripper-mounted "
    "wrist camera. Use both panels in BOTH images — the wrist view shows fine "
    "grasp/release detail, the third-person view shows scene context. Compare "
    "AFTER to BEFORE to judge whether the action was successful."
)


def _build_pick_subtask(label: str, target: str | None, instruction: str) -> str:
    obj = f" '{target}'" if target else " the target object"
    return (
        f"The TipTop planner just finished executing the subtask labeled "
        f"'{label}': the robot attempted to PICK{obj} as part of the overall "
        f"task \"{instruction}\". Verify the pick was successful: in the "
        f"AFTER frame{obj} should be visibly grasped in the closed gripper "
        "(held / lifted off whatever surface it was on), whereas in the BEFORE "
        f"frame the gripper should not yet be holding{obj}. "
        + _VIEW_LAYOUT_NOTE
    )


def _build_place_subtask(label: str, target: str | None, instruction: str) -> str:
    obj = f" '{target}'" if target else " the held object"
    return (
        f"The TipTop planner just finished executing the subtask labeled "
        f"'{label}': the robot attempted to PLACE{obj} as part of the overall "
        f"task \"{instruction}\". Verify the place was successful: in the "
        f"AFTER frame{obj} should be at its goal destination and no longer "
        "held by the gripper, whereas in the BEFORE frame the gripper should "
        f"still be holding{obj}. "
        + _VIEW_LAYOUT_NOTE
    )


def run_episode_tiptop_vlm(
    env,
    env_cfg,
    episode,
    headless: bool = False,
    save_videos: bool = True,
    video_mode: str = "all",
    remote_host: str = "localhost",
    remote_port: int = 8765,
    vlm_check_delay_steps: int = 35,
    vlm_model: str = "gemini-robotics-er-1.6-preview",
    vlm_verbose: bool = False,
    save_vlm_debug_images: bool = False,
):
    """Run a TipTop-driven episode with passive VLM auditing of planner subtasks.

    Watches `client.current_subtask_label` for transitions and, whenever the
    planner finishes a Pick or Place subtask (i.e. the label kind changes away
    from pick/place, or the plan ends), schedules a VLM check
    `vlm_check_delay_steps` later. The planner's own label and target object
    ground the VLM prompt. Episode is not terminated or replanned based on
    VLM output.
    """
    if env.num_envs != 1:
        raise ValueError(
            f"TipTop+VLM requires --num-envs 1 (got {env.num_envs}). "
            "TipTop is single-env only."
        )

    timer = TimingStats()
    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    instruction = env_cfg.instruction
    action_dim = 8  # 7 joints + 1 gripper

    subtask_status = []

    client = TiptopWebsocketClient(remote_host=remote_host, remote_port=remote_port)

    if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        env.recorder_manager.set_episode_index(0, env_ids=[0])

    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")
    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        suffix = f"_{episode}"
        if save_sensor:
            video_path = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4")
            video_writers_obs.append(VideoWriter(video_path, video_fps))
        if save_viewport:
            video_path_vp = os.path.join(
                get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4"
            )
            video_writers_viewport.append(VideoWriter(video_path_vp, video_fps))

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    # --- Harness setup (env 0 only — TipTop is single-env). ---
    harness_dir = Path(get_output_dir())
    log_file = harness_dir / f"harness_{episode}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [Harness] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    def hlog(msg, color="\033[96m"):
        print(f"{color}[Harness] {msg}\033[0m")
        logger.info(msg)

    monitor = ProgressMonitor(model_id=vlm_model)

    debug_dir: Path | None = None
    if save_vlm_debug_images:
        debug_dir = harness_dir / f"vlm_debug_{episode}"
        debug_dir.mkdir(parents=True, exist_ok=True)

    def _save_debug_image(frame, filename: str) -> str | None:
        """Save an RGB frame as PNG into debug_dir. Returns the absolute path or None."""
        if debug_dir is None or frame is None:
            return None
        path = debug_dir / filename
        try:
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception:
            logger.exception("Failed to save debug image %s", path)
            return None
        return str(path)

    # "Before" reference for the VLM. Initialised from the post-reset scene and updated
    # to each check's "after" frame after that check fires, so every check is "did the
    # scene change in the expected way since the previous event-boundary?".
    before_frame_vlm = _build_vlm_view(obs, env_id=0)
    before_frame_path = _save_debug_image(before_frame_vlm, "initial_step0000.png")

    # Pending VLM checks: list of dicts with {fire_step, kind, label, target,
    # event_step, event_index, transition_image (optional path)}.
    pending_checks: list[dict] = []
    check_results: list[dict] = []  # All fired and unfired checks, written to JSON at end.
    event_counter = 0
    # Track the planner subtask we were in last iteration so we can detect "left a
    # Pick/Place segment" boundaries via `client.current_subtask_label` transitions.
    prev_kind: str | None = None
    prev_label: str | None = None
    prev_target: str | None = None
    plan_done_logged = False

    hlog(f"Instruction: \"{instruction}\"")
    hlog(f"VLM check delay after gripper transition: {vlm_check_delay_steps} steps")
    if debug_dir is not None:
        hlog(f"Saving VLM debug images to {debug_dir}")

    actual_steps = 0
    try:
        for step in tqdm(range(max_steps)):

            while not timeline.is_playing():
                kit_app.update()

            timer.start("policy_inference")
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            for env_id in env.active_env_ids:
                ret = client.infer(obs, instruction, env_id=env_id)
                actions[env_id] = torch.tensor(ret["action"], device=env.device)
                if env_id == 0 or last_viz is None:
                    last_viz = ret.get("viz")
            timer.stop("policy_inference")

            # --- Planner subtask-label transition detection (env 0 only). ---
            # We fire the VLM check ONLY when we have just *left* a Pick or Place
            # subtask, i.e. `prev` was a Pick/Place label and the planner has now
            # moved on to a different step. By the time `_executing_label` changes,
            # `_step_plan` has already returned every action of the previous subtask
            # (approach + gripper actuation + lift/retract), so the Pick/Place is
            # fully complete. The check is then scheduled for `step + delay`, which
            # is strictly *after* the subtask finished — never during or before.
            #
            # We compare full labels (not just `kind`) so that a hypothetical back-
            # to-back Pick(A) → Pick(B) boundary would still be caught.
            if 0 in env.active_env_ids:
                curr_label = client.current_subtask_label
                curr_kind, curr_target = _parse_subtask_label(curr_label)

                left_supported_subtask = (
                    prev_kind in _SUPPORTED_ACTIONS and prev_label != curr_label
                )
                if left_supported_subtask:
                    event_counter += 1
                    fire_step = step + vlm_check_delay_steps
                    transition_path = None
                    if debug_dir is not None:
                        transition_frame = _build_vlm_view(obs, env_id=0)
                        transition_path = _save_debug_image(
                            transition_frame,
                            f"event_{event_counter:03d}_{prev_kind}_transition_step{step:04d}.png",
                        )
                    pending_checks.append({
                        "fire_step": fire_step,
                        "kind": prev_kind,
                        "label": prev_label,
                        "target": prev_target,
                        "event_step": step,
                        "event_index": event_counter,
                        "transition_image": transition_path,
                    })
                    hlog(
                        f"Subtask boundary at step {step}: left '{prev_label}' "
                        f"(now {curr_label!r}) → {prev_kind} check #{event_counter} "
                        f"scheduled for step {fire_step}"
                    )

                prev_kind, prev_label, prev_target = curr_kind, curr_label, curr_target

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if VISUALIZE:
                get_world(env).visualize()

            timer.start("env_step")
            obs, reward, term, trunc, info = env.step(actions)
            timer.stop("env_step")

            per_env_infos = get_all_env_subtask_infos(env)
            subtask_status.append(per_env_infos)

            if save_videos:
                timer.start("video_write")
                for env_id in range(env.num_envs):
                    if env._frozen_envs[env_id]:
                        continue
                    if save_sensor:
                        frame_obs = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                        video_writers_obs[env_id].write(frame_obs)
                    if save_viewport:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        video_writers_viewport[env_id].write(frame_vp)
                timer.stop("video_write")

            actual_steps += 1

            # --- Fire any due VLM checks against the post-step external-cam frame. ---
            still_pending: list[dict] = []
            for check in pending_checks:
                if step < check["fire_step"]:
                    still_pending.append(check)
                    continue

                frame = _build_vlm_view(obs, env_id=0)
                monitor.set_frame(frame)

                check_image_path = _save_debug_image(
                    frame,
                    f"event_{check['event_index']:03d}_{check['kind']}_check_step{step:04d}.png",
                )
                # Snapshot the before pointers before we slide them forward below — these
                # are what got sent to the VLM for this specific check.
                before_image_path_for_this_check = before_frame_path

                if check["kind"] == "pick":
                    check_subtask = _build_pick_subtask(
                        check["label"] or "Pick(...)", check.get("target"), instruction
                    )
                else:
                    check_subtask = _build_place_subtask(
                        check["label"] or "Place(...)", check.get("target"), instruction
                    )

                timer.start("vlm_check")
                try:
                    result = monitor.check_completion(
                        check_subtask, memory="", before_frame=before_frame_vlm
                    )
                except Exception as e:
                    timer.stop("vlm_check")
                    hlog(
                        f"VLM check FAILED for event #{check['event_index']} "
                        f"({check['kind']}, {check['label']}): {type(e).__name__}: {e}",
                        color="\033[91m",
                    )
                    check_results.append({
                        "event_index": check["event_index"],
                        "event_kind": check["kind"],
                        "subtask_label": check.get("label"),
                        "target": check.get("target"),
                        "event_step": check["event_step"],
                        "checked_step": step,
                        "completed": None,
                        "reason": f"VLM error: {type(e).__name__}: {e}",
                        "transition_image": check.get("transition_image"),
                        "before_image": before_image_path_for_this_check,
                        "check_image": check_image_path,
                    })
                    # Slide forward even on error — next event should compare against the
                    # most recent observed state, not a stale one.
                    before_frame_vlm = frame
                    before_frame_path = check_image_path
                    continue
                timer.stop("vlm_check")

                completed = bool(result["completed"])
                color = "\033[92m" if completed else "\033[93m"
                hlog(
                    f"VLM event #{check['event_index']} ({check['kind']}, "
                    f"{check['label']}) @ boundary={check['event_step']}, "
                    f"checked={step}: completed={completed} | {result.get('reason', '')}",
                    color=color,
                )

                record = {
                    "event_index": check["event_index"],
                    "event_kind": check["kind"],
                    "subtask_label": check.get("label"),
                    "target": check.get("target"),
                    "event_step": check["event_step"],
                    "checked_step": step,
                    "completed": completed,
                    "reason": result.get("reason", ""),
                    "transition_image": check.get("transition_image"),
                    "before_image": before_image_path_for_this_check,
                    "check_image": check_image_path,
                }
                if vlm_verbose:
                    record["raw_vlm_response"] = result.get("raw_vlm_response", "")
                check_results.append(record)

                # Slide the before pointers forward: the current check's "after" frame
                # becomes the next check's "before".
                before_frame_vlm = frame
                before_frame_path = check_image_path

            pending_checks = still_pending

            # --- Plan-done handling. ---
            # TipTop doesn't drive sim termination, so we mirror episode.py and freeze
            # env 0 manually when its plan is done. But: if there are pending VLM checks
            # we haven't drained yet, keep stepping (TipTop returns hold actions) so the
            # final post-place check can still fire.
            plan_done = bool(getattr(client, "plan_done", False))
            if plan_done and not plan_done_logged:
                if pending_checks:
                    hlog(
                        f"Plan completed at step {step}. Draining "
                        f"{len(pending_checks)} pending VLM check(s) before freezing."
                    )
                plan_done_logged = True

            if (
                plan_done
                and not pending_checks
                and 0 in env.active_env_ids
            ):
                env._frozen_envs[0] = True
                env._env_results[0] = bool(env.termination_manager.terminated[0])
                env._env_term_step[0] = int(env.episode_length_buf[0].item())
                if env.recorder_manager is not None:
                    try:
                        env.recorder_manager.export_episodes(env_ids=[0])
                    except Exception:
                        logger.exception("Failed to export recorder for env 0")

            if env.all_terminated:
                break
    finally:
        # Record any pending checks that didn't fire (episode ended too early).
        if pending_checks:
            hlog(
                f"Episode ended with {len(pending_checks)} VLM check(s) unfired.",
                color="\033[93m",
            )
            for check in pending_checks:
                check_results.append({
                    "event_index": check["event_index"],
                    "event_kind": check["kind"],
                    "subtask_label": check.get("label"),
                    "target": check.get("target"),
                    "event_step": check["event_step"],
                    "checked_step": None,
                    "completed": None,
                    "reason": "Episode ended before check fired",
                    "transition_image": check.get("transition_image"),
                    "before_image": before_frame_path,
                    "check_image": None,
                })

        try:
            with open(harness_dir / f"vlm_checks_{episode}.json", "w") as f:
                json.dump({
                    "instruction": instruction,
                    "vlm_check_delay_steps": vlm_check_delay_steps,
                    "vlm_model": vlm_model,
                    "num_events": event_counter,
                    "checks": check_results,
                }, f, indent=2)
        except Exception:
            logger.exception("Failed to write vlm_checks_%d.json", episode)

        logger.removeHandler(file_handler)
        file_handler.close()

        if save_videos:
            for vw in video_writers_obs + video_writers_viewport:
                try:
                    vw.release()
                except Exception:
                    logger.exception("Failed to release video writer")

    client.reset()

    timing = timer.to_dict(actual_steps)
    return env.get_env_results(), subtask_status, timing
