# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Episode runner for the TipTop + VLA + VLM hybrid method.

State machine:
  - Try TipTop first. On `PlanningError`, drop to VLA-recovery mode.
  - While in TIPTOP: every Pick/Place subtask boundary schedules a VLM check
    `vlm_check_delay_steps` later. If the VLM says the planner step failed,
    switch to VLA-recovery mode with the failed label as context.
  - While in VLA_RECOVERY: ask the VLM for ONE combined pick+place command, run
    the VLA on it, and poll completion every `check_every_n_steps`. On
    completion (or per-subtask timeout), try `client.reset()` + `client.infer()`
    to replan with TipTop from the current state. Success → TIPTOP. Failure
    (`PlanningError`) → if overall goal is done end the episode; otherwise ask
    the VLM for another recovery pick+place and stay in VLA_RECOVERY.

Saved videos overlay a color-coded banner (green=TIPTOP, orange=VLA-RECOVERY,
red=transition step) plus the active subtask string and the step counter so the
viewer can tell at a glance which agent is driving.

Single-env only — TipTop is single-env only.
"""

import json
import logging
import os
import re
import signal
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from episode import TimingStats
from episode_tiptop_vlm import (
    _SUPPORTED_ACTIONS,
    _build_pick_subtask,
    _build_place_subtask,
    _build_vlm_view,
    _parse_subtask_label,
)

from robolab.constants import VISUALIZE, get_output_dir
from robolab.core.logging.results import get_all_env_subtask_infos
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.cv2_utils import add_multiline_text_overlay_with_background
from robolab.core.utils.video_utils import VideoWriter
from robolab.core.world.world_state import get_world
from robolab.inference.tiptop import PlanningError, TiptopWebsocketClient
from robolab.inference.vlm_done_checker import ProgressMonitor, get_recovery_pick_place

logger = logging.getLogger(__name__)

MODE_TIPTOP = "TIPTOP"
MODE_VLA = "VLA-RECOVERY"
MODE_HOMING = "HOMING"

# cv2_utils' add_multiline_text_overlay_with_background applies rectangle/text
# colors to a BGR image without re-swapping, so the colors below are BGR triples
# even though the docstring claims RGB. Green is symmetric (G is middle channel)
# so it looks fine either way; red/orange/blue are written in BGR for clarity.
_BANNER_BG_TIPTOP = (0, 110, 0)        # green
_BANNER_BG_VLA = (0, 110, 220)         # orange (B=0, G=110, R=220)
_BANNER_BG_HOMING = (180, 90, 0)       # blue (B=180, G=90, R=0)
_BANNER_BG_TRANSITION = (0, 0, 220)    # red (B=0, G=0, R=220)

# Franka home configuration matching DroidCfg.init_state in robolab/robots/droid.py.
# Matches the joint values we observed at episode start in a standalone VLA run
# ([0.0, -π/5, 0.0, -4π/5, 0.0, 3π/5, 0.0]).
HOME_JOINT_POS = np.array(
    [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0],
    dtype=np.float32,
)
# Gripper command sent during HOMING. 0.0 = open. We open by default so the next
# TipTop plan starts from a clean "nothing held" state. If a recovery VLA timed
# out mid-grasp, this means the object is dropped at the home location — that's
# usually preferable to TipTop planning around a phantom held object.
HOME_GRIPPER = 0.0


def _truncate(text: str | None, max_len: int = 60) -> str:
    if not text:
        return "(idle)"
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _annotate(frame: np.ndarray, mode: str, subtask: str | None, step: int, is_transition: bool) -> np.ndarray:
    """Draw a 3-line mode banner on a video frame. RGB in, RGB out."""
    if is_transition:
        bg = _BANNER_BG_TRANSITION
        lines = [f"MODE → {mode}", _truncate(subtask), f"step {step} (switched)"]
    else:
        if mode == MODE_TIPTOP:
            bg = _BANNER_BG_TIPTOP
        elif mode == MODE_HOMING:
            bg = _BANNER_BG_HOMING
        else:
            bg = _BANNER_BG_VLA
        lines = [f"MODE: {mode}", _truncate(subtask), f"step {step}"]
    return add_multiline_text_overlay_with_background(
        frame,
        lines,
        start_position=(10, 30),
        font_scale=0.7,
        thickness=2,
        line_spacing=28,
        background_color=bg,
        background_alpha=0.65,
        padding=10,
    )


def run_episode_tiptop_vla_vlm(
    env,
    env_cfg,
    episode,
    headless: bool = False,
    save_videos: bool = True,
    video_mode: str = "all",
    tiptop_host: str = "localhost",
    tiptop_port: int = 8765,
    vla_host: str = "localhost",
    vla_port: int = 8000,
    vlm_check_delay_steps: int = 10,
    check_every_n_steps: int = 15,
    subtask_timeout_steps: int = 150,
    vlm_model: str = "gemini-robotics-er-1.6-preview",
    vlm_verbose: bool = False,
    save_vlm_debug_images: bool = False,
    tiptop_gripper_steps: int = 20,
    tiptop_waypoint_stride: int | None = None,
    home_pose_steps: int = 30,
):
    """Hybrid TipTop+VLA+VLM episode. See module docstring for the state machine."""
    if env.num_envs != 1:
        raise ValueError(
            f"TipTop+VLA+VLM requires --num-envs 1 (got {env.num_envs}). "
            "TipTop is single-env only."
        )

    timer = TimingStats()

    # --- VLA backend selection: copy the dispatch from episode_vla_vlm.py. ---
    backend = getattr(env_cfg, "policy", "pi05").lower()
    if backend in ("pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05"):
        from robolab.inference.pi0_family import Pi0DroidJointposClient as VlaPolicyClient
    elif "gr00t" in backend:
        from robolab.inference.gr00t import GR00TDroidJointposClient as VlaPolicyClient
    elif backend == "dreamzero":
        from robolab.inference.dreamzero import DreamZeroClient as VlaPolicyClient
    elif backend == "molmo":
        from robolab.inference.droid_molmo import MolmoActClient as VlaPolicyClient
    elif backend == "openvla":
        from robolab.inference.openvla import OpenVLAClient as VlaPolicyClient
    elif backend == "openvla_oft":
        from robolab.inference.openvla_oft import OpenVLAOFTClient as VlaPolicyClient
    else:
        raise ValueError(
            f"Unsupported VLA backend '{backend}' for the TipTop+VLA+VLM hybrid. "
            "Use a VLA policy (pi05, gr00t, etc.) for the VLA-recovery client."
        )

    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    instruction = env_cfg.instruction
    action_dim = 8  # 7 joints + 1 gripper

    subtask_status: list = []

    tiptop_client = TiptopWebsocketClient(
        remote_host=tiptop_host,
        remote_port=tiptop_port,
        gripper_action_steps=tiptop_gripper_steps,
        waypoint_stride=tiptop_waypoint_stride,
    )

    # Lazy + fresh VLA client: openpi websocket servers can drop idle connections
    # silently and the first request after a long TipTop session would land on a
    # half-closed socket. Re-create the client every time we enter VLA-recovery
    # mode so the websocket is always fresh.
    vla_client = None

    def _make_vla_client():
        return VlaPolicyClient(remote_host=vla_host, remote_port=vla_port)

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
        if debug_dir is None or frame is None:
            return None
        path = debug_dir / filename
        try:
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except Exception:
            logger.exception("Failed to save debug image %s", path)
            return None
        return str(path)

    before_frame_vlm = _build_vlm_view(obs, env_id=0)
    before_frame_path = _save_debug_image(before_frame_vlm, "initial_step0000.png")

    # --- State machine variables ---
    # `pending_tiptop_replan` triggers a TipTop replan probe on the next step. True at
    # episode start (initial planning) and after every HOMING completes.
    pending_tiptop_replan: bool = True
    # `pending_homing` is set when VLA recovery completes/times out or when a TipTop
    # plan finishes without reaching the goal. The next steps drive the arm to
    # HOME_JOINT_POS before TipTop is re-queried, so the planner starts from a
    # canonical joint configuration instead of a random VLA end-pose.
    pending_homing: bool = False
    homing_steps_remaining: int = 0
    mode: str = MODE_TIPTOP  # tentative — first step will resolve it for real
    prev_executed_mode: str | None = None  # for transition-frame detection on overlay

    # VLA-recovery state
    vla_subtask: str | None = None
    vla_subtask_start_step: int = 0
    failed_label_context: str | None = None

    # TipTop boundary-check state (mirrors episode_tiptop_vlm.py)
    pending_checks: list[dict] = []
    event_counter: int = 0
    prev_kind: str | None = None
    prev_label: str | None = None
    prev_target: str | None = None

    check_results: list[dict] = []
    mode_transitions: list[dict] = []
    goal_achieved: bool = False
    plan_done_logged: bool = False
    actual_steps: int = 0

    # Isaac Sim / kit installs its own SIGINT handler that swallows the default
    # Ctrl-C → KeyboardInterrupt path, so the episode runner never exits and the
    # finally block (video release, JSON dump) never fires. Install our own
    # handler that sets a flag the main loop checks, and restore the previous
    # handler in finally. Second Ctrl-C escalates to raising KeyboardInterrupt
    # so the user can still force-kill if cleanup itself hangs.
    interrupted = {"count": 0}
    prev_sigint_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):
        interrupted["count"] += 1
        if interrupted["count"] == 1:
            print(
                "\n\033[93m[Harness] Ctrl-C received — finishing current step then "
                "shutting down cleanly (videos + logs). Press Ctrl-C again to force.\033[0m",
                flush=True,
            )
        else:
            print(
                "\n\033[91m[Harness] Second Ctrl-C — raising KeyboardInterrupt (may skip cleanup).\033[0m",
                flush=True,
            )
            raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        # Non-main thread or platform restriction — signal install isn't critical,
        # the launcher's KeyboardInterrupt fallback still works.
        prev_sigint_handler = None

    hlog(f"Instruction: \"{instruction}\"")
    hlog(f"TipTop server: {tiptop_host}:{tiptop_port}  |  VLA server: {vla_host}:{vla_port}")
    hlog(
        f"VLM model: {vlm_model}  |  TipTop boundary check delay: {vlm_check_delay_steps}  |  "
        f"VLA-recovery check every: {check_every_n_steps} steps  |  "
        f"VLA-recovery timeout: {subtask_timeout_steps} steps  |  "
        f"home-pose steps before TipTop replan: {home_pose_steps}"
    )

    def _record_transition(step: int, from_mode: str | None, to_mode: str, reason: str,
                           subtask_before: str | None, subtask_after: str | None):
        mode_transitions.append({
            "step": step,
            "from_mode": from_mode,
            "to_mode": to_mode,
            "reason": reason,
            "subtask_before": subtask_before,
            "subtask_after": subtask_after,
        })
        hlog(
            f"MODE TRANSITION @ step {step}: {from_mode} → {to_mode} | {reason}",
            color="\033[95m",
        )

    def _ensure_fresh_vla_client():
        """(Re)create the VLA client so its websocket is fresh. Closes the old one
        if present — calls .close() if available, otherwise drops the reference."""
        nonlocal vla_client
        if vla_client is not None:
            ws = getattr(getattr(vla_client, "client", None), "_ws", None)
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
        vla_client = _make_vla_client()

    def _request_vla_recovery(step: int, reason: str) -> bool:
        """Pick a fresh recovery subtask via the VLM and prep the VLA client.
        Returns True if a subtask was acquired, False if the VLM declared the goal done."""
        nonlocal vla_subtask, vla_subtask_start_step, goal_achieved
        scene_frame = _build_vlm_view(obs, env_id=0)
        monitor.set_frame(scene_frame)
        _save_debug_image(scene_frame, f"recovery_request_step{step:04d}.png")
        timer.start("vlm_check")
        try:
            rec = get_recovery_pick_place(
                instruction, scene_frame, failed_subtask=failed_label_context
            )
        except Exception as e:
            timer.stop("vlm_check")
            hlog(
                f"get_recovery_pick_place failed at step {step}: {type(e).__name__}: {e}. "
                "Falling back to the raw task instruction as VLA goal.",
                color="\033[91m",
            )
            vla_subtask = instruction
            vla_subtask_start_step = step
            _ensure_fresh_vla_client()
            return True
        timer.stop("vlm_check")

        if rec.done:
            hlog(
                f"VLM declares the overall goal already achieved during recovery request "
                f"(step {step}, reason: {reason}).",
                color="\033[92m",
            )
            goal_achieved = True
            return False

        vla_subtask = rec.instruction or instruction
        vla_subtask_start_step = step
        _ensure_fresh_vla_client()
        hlog(f"VLA recovery subtask: \"{vla_subtask}\"")
        return True

    try:
        for step in tqdm(range(max_steps)):

            if interrupted["count"] > 0:
                hlog("Loop exit triggered by SIGINT.", color="\033[93m")
                break

            while not timeline.is_playing():
                kit_app.update()
                if interrupted["count"] > 0:
                    break
            if interrupted["count"] > 0:
                break

            # ---------------------------------------------------------------
            # Resolve the action for this step (and the active mode).
            # ---------------------------------------------------------------
            timer.start("policy_inference")
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            this_step_action = None

            # ---------------------------------------------------------------
            # HOMING: drive the arm to a canonical pose before re-querying
            # TipTop. Enter when `pending_homing` is set (VLA recovery done /
            # timed out, or TipTop plan finished without the goal achieved).
            # Once the home-step counter drains, hand off to the TipTop replan
            # branch below.
            # ---------------------------------------------------------------
            if pending_homing:
                pending_homing = False
                homing_steps_remaining = home_pose_steps
                if mode != MODE_HOMING:
                    _record_transition(
                        step, mode, MODE_HOMING,
                        f"moving to home pose ({home_pose_steps} steps) before TipTop replan",
                        vla_subtask, "(home pose)",
                    )
                mode = MODE_HOMING

            if homing_steps_remaining > 0 and not goal_achieved:
                this_step_action = np.concatenate(
                    [HOME_JOINT_POS, np.array([HOME_GRIPPER], dtype=np.float32)]
                )
                homing_steps_remaining -= 1
                if homing_steps_remaining == 0:
                    # Hand off to the TipTop replan branch on the NEXT iteration.
                    pending_tiptop_replan = True
                    hlog(
                        f"Homing complete at step {step}. Queuing TipTop replan for next step."
                    )

            if this_step_action is None and pending_tiptop_replan:
                # Try TipTop: reset + infer triggers a fresh _query_server. On success the
                # returned action drives this step; on PlanningError we fall through to VLA.
                pending_tiptop_replan = False
                try:
                    if tiptop_client._plan is not None:
                        tiptop_client.reset()
                    timer.stop("policy_inference")
                    timer.start("vlm_next_subtask")  # use this bucket for planning latency
                    ret = tiptop_client.infer(obs, instruction, env_id=0)
                    timer.stop("vlm_next_subtask")
                    timer.start("policy_inference")
                    this_step_action = ret["action"]
                    last_viz = ret.get("viz")
                    if mode != MODE_TIPTOP:
                        _record_transition(
                            step, mode if prev_executed_mode is not None else None,
                            MODE_TIPTOP,
                            "tiptop planner produced a plan",
                            vla_subtask, tiptop_client.current_subtask_label,
                        )
                    mode = MODE_TIPTOP
                    # Reset the boundary tracker: any stale prev_label from before the
                    # replan would falsely schedule a check the moment the new plan loads
                    # its first Pick/Place. Treat the new plan as a clean start.
                    prev_kind = prev_label = prev_target = None
                    plan_done_logged = False
                except PlanningError as e:
                    timer.stop("vlm_next_subtask")
                    timer.start("policy_inference")
                    hlog(
                        f"TipTop replan FAILED at step {step}: {e}. Dropping to VLA recovery.",
                        color="\033[93m",
                    )
                    # Check whether the overall task is actually done before recovering,
                    # otherwise we'd ask for recovery when there's nothing left to do.
                    scene_frame = _build_vlm_view(obs, env_id=0)
                    monitor.set_frame(scene_frame)
                    _save_debug_image(scene_frame, f"overall_done_after_replan_fail_step{step:04d}.png")
                    timer.stop("policy_inference")
                    timer.start("vlm_check")
                    try:
                        done_chk = monitor.check_completion(
                            instruction, memory="", before_frame=before_frame_vlm
                        )
                    except Exception as ee:
                        hlog(f"Overall-goal check failed (assuming not done): {type(ee).__name__}: {ee}",
                             color="\033[91m")
                        done_chk = {"completed": False, "reason": "vlm error"}
                    timer.stop("vlm_check")
                    timer.start("policy_inference")
                    check_results.append({
                        "type": "overall_done_after_replan_failure",
                        "step": step,
                        "completed": bool(done_chk.get("completed", False)),
                        "reason": done_chk.get("reason", ""),
                    })
                    if done_chk.get("completed"):
                        hlog(f"Overall goal already achieved at step {step}; ending episode.",
                             color="\033[92m")
                        goal_achieved = True
                    else:
                        if _request_vla_recovery(step, reason="tiptop replan failed"):
                            if mode != MODE_VLA:
                                _record_transition(
                                    step, mode if prev_executed_mode is not None else None,
                                    MODE_VLA,
                                    f"tiptop PlanningError: {e}",
                                    None, vla_subtask,
                                )
                            mode = MODE_VLA
                        # else: goal_achieved set, will exit below

            # If we didn't acquire an action via the replan branch, drive normally.
            if this_step_action is None and not goal_achieved:
                if mode == MODE_TIPTOP:
                    ret = tiptop_client.infer(obs, instruction, env_id=0)
                    this_step_action = ret["action"]
                    last_viz = ret.get("viz")
                else:  # MODE_VLA
                    if vla_subtask is None or vla_client is None:
                        # Safety net — should have been set when entering VLA mode.
                        vla_subtask = instruction
                        vla_subtask_start_step = step
                        _ensure_fresh_vla_client()
                    ret = vla_client.infer(obs, vla_subtask, env_id=0)
                    this_step_action = ret["action"]
                    last_viz = ret.get("viz")

            if not goal_achieved:
                actions[0] = torch.tensor(this_step_action, device=env.device)

            timer.stop("policy_inference")

            # ---------------------------------------------------------------
            # TipTop boundary detection (only meaningful while in TIPTOP mode).
            # ---------------------------------------------------------------
            if not goal_achieved and mode == MODE_TIPTOP and 0 in env.active_env_ids:
                curr_label = tiptop_client.current_subtask_label
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
                        f"TipTop subtask boundary at step {step}: left '{prev_label}' "
                        f"(now {curr_label!r}) → {prev_kind} check #{event_counter} "
                        f"scheduled for step {fire_step}"
                    )

                prev_kind, prev_label, prev_target = curr_kind, curr_label, curr_target

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if VISUALIZE:
                get_world(env).visualize()

            # ---------------------------------------------------------------
            # Step the env (skip if we already decided the goal is achieved
            # before producing an action this step).
            # ---------------------------------------------------------------
            if not goal_achieved:
                timer.start("env_step")
                obs, reward, term, trunc, info = env.step(actions)
                timer.stop("env_step")

                per_env_infos = get_all_env_subtask_infos(env)
                subtask_status.append(per_env_infos)
                actual_steps += 1
            else:
                # Mark a None for shape parity with the other runners' subtask_status.
                subtask_status.append(None)

            # ---------------------------------------------------------------
            # Write video frames with the mode banner overlaid.
            # ---------------------------------------------------------------
            if save_videos and not env._frozen_envs[0]:
                timer.start("video_write")
                executed_mode = mode
                is_transition_frame = (
                    prev_executed_mode is not None and executed_mode != prev_executed_mode
                )
                if mode == MODE_TIPTOP:
                    banner_subtask = tiptop_client.current_subtask_label or "(plan idle)"
                elif mode == MODE_HOMING:
                    banner_subtask = f"home pose ({homing_steps_remaining} steps left)"
                else:
                    banner_subtask = vla_subtask or "(recovery pending)"
                if save_sensor:
                    frame_obs = unpack_image_obs(obs, scale=0.5, env_id=0).get("combined_image")
                    if frame_obs is not None:
                        frame_obs = _annotate(frame_obs, executed_mode, banner_subtask, step, is_transition_frame)
                        video_writers_obs[0].write(frame_obs)
                if save_viewport:
                    frame_vp = unpack_viewport_cams(obs, env_id=0).get("combined_image")
                    if frame_vp is not None:
                        frame_vp = _annotate(frame_vp, executed_mode, banner_subtask, step, is_transition_frame)
                        video_writers_viewport[0].write(frame_vp)
                prev_executed_mode = executed_mode
                timer.stop("video_write")
            else:
                prev_executed_mode = mode

            if goal_achieved:
                # Freeze env 0 once and break.
                if 0 in env.active_env_ids:
                    env._frozen_envs[0] = True
                    env._env_results[0] = True
                    env._env_term_step[0] = int(env.episode_length_buf[0].item())
                    if env.recorder_manager is not None:
                        try:
                            env.recorder_manager.export_episodes(env_ids=[0])
                        except Exception:
                            logger.exception("Failed to export recorder for env 0")
                if env.all_terminated:
                    break
                continue

            # ---------------------------------------------------------------
            # Fire any due TipTop boundary VLM checks.
            # ---------------------------------------------------------------
            still_pending: list[dict] = []
            switched_to_vla_this_step = False
            for check in pending_checks:
                if step < check["fire_step"]:
                    still_pending.append(check)
                    continue
                # If we already switched to VLA earlier in this loop iteration,
                # defer remaining checks: their context (TipTop label) no longer
                # matches what's executing. Drop them with a "skipped" record.
                if switched_to_vla_this_step or mode != MODE_TIPTOP:
                    check_results.append({
                        "type": "tiptop_boundary",
                        "event_index": check["event_index"],
                        "event_kind": check["kind"],
                        "subtask_label": check.get("label"),
                        "target": check.get("target"),
                        "event_step": check["event_step"],
                        "checked_step": step,
                        "completed": None,
                        "reason": "skipped: switched to VLA recovery before check fired",
                        "transition_image": check.get("transition_image"),
                    })
                    continue

                frame = _build_vlm_view(obs, env_id=0)
                monitor.set_frame(frame)
                check_image_path = _save_debug_image(
                    frame,
                    f"event_{check['event_index']:03d}_{check['kind']}_check_step{step:04d}.png",
                )
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
                        f"TipTop boundary VLM check FAILED for event #{check['event_index']} "
                        f"({check['kind']}, {check['label']}): {type(e).__name__}: {e}",
                        color="\033[91m",
                    )
                    check_results.append({
                        "type": "tiptop_boundary",
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
                    before_frame_vlm = frame
                    before_frame_path = check_image_path
                    continue
                timer.stop("vlm_check")

                completed = bool(result["completed"])
                color = "\033[92m" if completed else "\033[93m"
                hlog(
                    f"TipTop boundary VLM event #{check['event_index']} ({check['kind']}, "
                    f"{check['label']}) checked={step}: completed={completed} | "
                    f"{result.get('reason', '')}",
                    color=color,
                )
                record = {
                    "type": "tiptop_boundary",
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

                before_frame_vlm = frame
                before_frame_path = check_image_path

                if not completed:
                    # Switch to VLA recovery. Drop any further pending checks for THIS
                    # iteration (they belong to the now-aborted plan); same applies to
                    # checks scheduled by later TipTop plan steps that we never executed.
                    failed_label_context = check.get("label")
                    if _request_vla_recovery(
                        step,
                        reason=f"TipTop {check['kind']} judged failed: "
                               f"{result.get('reason', '')[:120]}",
                    ):
                        _record_transition(
                            step, MODE_TIPTOP, MODE_VLA,
                            f"tiptop {check['kind']} failed (event #{check['event_index']})",
                            check.get("label"), vla_subtask,
                        )
                        mode = MODE_VLA
                    switched_to_vla_this_step = True
                    # Don't break: continue to drain the loop so we emit "skipped" records
                    # for any remaining due checks (above branch handles that).
            pending_checks = still_pending

            # ---------------------------------------------------------------
            # TipTop plan-done handling: when the whole plan has finished,
            # check the overall goal and either end or replan.
            # ---------------------------------------------------------------
            if not goal_achieved and mode == MODE_TIPTOP:
                plan_done = bool(getattr(tiptop_client, "plan_done", False))
                if plan_done and not pending_checks and not plan_done_logged:
                    plan_done_logged = True
                    hlog(f"TipTop plan finished at step {step}. Checking overall goal.")
                    scene_frame = _build_vlm_view(obs, env_id=0)
                    monitor.set_frame(scene_frame)
                    _save_debug_image(scene_frame, f"overall_done_after_plan_step{step:04d}.png")
                    timer.start("vlm_check")
                    try:
                        done_chk = monitor.check_completion(
                            instruction, memory="", before_frame=before_frame_vlm
                        )
                    except Exception as e:
                        hlog(f"Plan-done VLM check failed: {type(e).__name__}: {e}",
                             color="\033[91m")
                        done_chk = {"completed": False, "reason": f"VLM error: {e}"}
                    timer.stop("vlm_check")
                    check_results.append({
                        "type": "overall_done_after_plan",
                        "step": step,
                        "completed": bool(done_chk.get("completed", False)),
                        "reason": done_chk.get("reason", ""),
                    })
                    before_frame_vlm = scene_frame
                    if done_chk.get("completed"):
                        hlog(f"Overall goal achieved after plan: {done_chk.get('reason', '')}",
                             color="\033[92m")
                        goal_achieved = True
                    else:
                        hlog(
                            "Overall goal NOT done after plan; homing arm before "
                            "TipTop replan.",
                            color="\033[93m",
                        )
                        pending_homing = True

            # ---------------------------------------------------------------
            # VLA-recovery periodic completion check.
            # ---------------------------------------------------------------
            if (
                not goal_achieved
                and mode == MODE_VLA
                and vla_subtask is not None
                and 0 in env.active_env_ids
            ):
                elapsed = step - vla_subtask_start_step
                if (
                    elapsed > 0
                    and (
                        elapsed % check_every_n_steps == 0
                        or elapsed >= subtask_timeout_steps
                    )
                ):
                    frame = _build_vlm_view(obs, env_id=0)
                    monitor.set_frame(frame)
                    _save_debug_image(frame, f"vla_recovery_check_step{step:04d}.png")
                    timer.start("vlm_check")
                    try:
                        result = monitor.check_completion(
                            vla_subtask, memory="", before_frame=before_frame_vlm
                        )
                    except Exception as e:
                        timer.stop("vlm_check")
                        hlog(
                            f"VLA recovery VLM check failed at step {step}: "
                            f"{type(e).__name__}: {e}",
                            color="\033[91m",
                        )
                        before_frame_vlm = frame
                        result = None
                    else:
                        timer.stop("vlm_check")
                    if result is not None:
                        completed = bool(result.get("completed", False))
                        color = "\033[92m" if completed else "\033[93m"
                        hlog(
                            f"VLA recovery check at step {step} (elapsed {elapsed}): "
                            f"completed={completed} | {result.get('reason', '')}",
                            color=color,
                        )
                        record = {
                            "type": "vla_recovery",
                            "step": step,
                            "vla_subtask": vla_subtask,
                            "failed_label_context": failed_label_context,
                            "elapsed_steps": elapsed,
                            "completed": completed,
                            "reason": result.get("reason", ""),
                        }
                        if vlm_verbose:
                            record["raw_vlm_response"] = result.get("raw_vlm_response", "")
                        check_results.append(record)
                        before_frame_vlm = frame

                        timed_out = elapsed >= subtask_timeout_steps
                        if completed or timed_out:
                            reason = "completed" if completed else f"timeout ({elapsed} steps)"
                            hlog(
                                f"VLA recovery subtask ended ({reason}); "
                                "homing arm before TipTop replan.",
                            )
                            pending_homing = True
                            # Clear failed_label_context — the VLM saw the recovery move;
                            # the next replan should plan from the fresh scene.
                            failed_label_context = None

            if env.all_terminated:
                break
    finally:
        # Restore the previous SIGINT handler so we don't leak our handler into
        # subsequent code (e.g. summarize_experiment_results in the launcher).
        if prev_sigint_handler is not None:
            try:
                signal.signal(signal.SIGINT, prev_sigint_handler)
            except (ValueError, OSError):
                pass

        # MOST critical first: finalize video files so a Ctrl-C still leaves
        # playable output of whatever the agent did up to the abort. cv2's
        # VideoWriter buffers frames internally; without release() the .mp4
        # header isn't written and the file is unplayable.
        if save_videos:
            for vw in video_writers_obs + video_writers_viewport:
                try:
                    vw.release()
                except Exception:
                    logger.exception("Failed to release video writer")

        # Also flush the recorder so partial trajectory data is preserved on
        # Ctrl-C. export_episodes() is safe to call even mid-episode.
        if env.recorder_manager is not None and not env._frozen_envs[0]:
            try:
                env.recorder_manager.export_episodes(env_ids=[0])
            except Exception:
                logger.exception("Failed to export recorder on early exit")

        # Drain any unfired TipTop boundary checks.
        if pending_checks:
            hlog(
                f"Episode ended with {len(pending_checks)} TipTop boundary VLM check(s) "
                "unfired.",
                color="\033[93m",
            )
            for check in pending_checks:
                check_results.append({
                    "type": "tiptop_boundary",
                    "event_index": check["event_index"],
                    "event_kind": check["kind"],
                    "subtask_label": check.get("label"),
                    "target": check.get("target"),
                    "event_step": check["event_step"],
                    "checked_step": None,
                    "completed": None,
                    "reason": "Episode ended before check fired",
                    "transition_image": check.get("transition_image"),
                })

        try:
            with open(harness_dir / f"vlm_checks_{episode}.json", "w") as f:
                json.dump({
                    "instruction": instruction,
                    "vlm_check_delay_steps": vlm_check_delay_steps,
                    "check_every_n_steps": check_every_n_steps,
                    "subtask_timeout_steps": subtask_timeout_steps,
                    "vlm_model": vlm_model,
                    "checks": check_results,
                    "goal_achieved": goal_achieved,
                }, f, indent=2)
        except Exception:
            logger.exception("Failed to write vlm_checks_%d.json", episode)

        try:
            with open(harness_dir / f"mode_transitions_{episode}.json", "w") as f:
                json.dump({
                    "instruction": instruction,
                    "transitions": mode_transitions,
                    "final_mode": mode,
                    "goal_achieved": goal_achieved,
                }, f, indent=2)
        except Exception:
            logger.exception("Failed to write mode_transitions_%d.json", episode)

        logger.removeHandler(file_handler)
        file_handler.close()

    try:
        tiptop_client.reset()
    except Exception:
        logger.exception("tiptop_client.reset() at end of episode failed")
    if vla_client is not None:
        try:
            vla_client.reset()
        except Exception:
            logger.exception("vla_client.reset() at end of episode failed")
        ws = getattr(getattr(vla_client, "client", None), "_ws", None)
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    timing = timer.to_dict(actual_steps)
    return env.get_env_results(), subtask_status, timing
