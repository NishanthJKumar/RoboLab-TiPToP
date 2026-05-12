# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Episode runner for the VLA + VLM hybrid method.

Drives the robot with a VLA policy (pi05, gr00t, ...) and runs a VLM-based
harness on env 0 every `check_every_n_steps` policy steps. Two modes:

- Plain (default): the VLM is asked whether the (fixed) task instruction is
  complete given the current frame and the previous-check frame. On YES, env
  0 is frozen as a success.
- Dynamic (`dynamic_prompting=True`): the VLM picks a subtask reactively from
  the current scene + memory. The policy is driven by the *current subtask*
  on env 0, the global instruction on the other envs. After each completion
  or timeout, memory is updated and the VLM is reprompted; the run ends when
  the VLM declares the overall goal achieved.

Other envs (env_id > 0) always receive the raw instruction and rely on sim
termination, matching the source repo (`prompt-harness-robolab`).
"""

import json
import logging
import os
import re
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from episode import TimingStats

from robolab.constants import VISUALIZE, get_output_dir
from robolab.core.logging.results import get_all_env_subtask_infos
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter
from robolab.core.world.world_state import get_world
from robolab.inference.vlm_done_checker import (
    MemoryManager,
    ProgressMonitor,
    get_next_subtask,
)

logger = logging.getLogger(__name__)


def run_episode_vla_vlm(
    env,
    env_cfg,
    episode,
    headless: bool = False,
    save_videos: bool = True,
    video_mode: str = "all",
    remote_host: str = "localhost",
    remote_port: int = 8000,
    check_every_n_steps: int = 15,
    dynamic_prompting: bool = False,
    subtask_timeout_steps: int = 150,
    vlm_model: str = "gemini-robotics-er-1.6-preview",
    vlm_verbose: bool = False,
):
    """Run a VLA-driven episode with VLM-based termination on env 0.

    See module docstring for the difference between plain and dynamic modes.
    """
    timer = TimingStats()
    backend = getattr(env_cfg, "policy", "pi05").lower()

    if backend in ("pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05"):
        from robolab.inference.pi0_family import Pi0DroidJointposClient as PolicyClient
    elif "gr00t" in backend:
        from robolab.inference.gr00t import GR00TDroidJointposClient as PolicyClient
    elif backend == "dreamzero":
        from robolab.inference.dreamzero import DreamZeroClient as PolicyClient
    elif backend == "molmo":
        from robolab.inference.droid_molmo import MolmoActClient as PolicyClient
    elif backend == "openvla":
        from robolab.inference.openvla import OpenVLAClient as PolicyClient
    elif backend == "openvla_oft":
        from robolab.inference.openvla_oft import OpenVLAOFTClient as PolicyClient
    else:
        raise ValueError(
            f"Unsupported VLA backend '{backend}' for the VLA+VLM hybrid. "
            "Use a VLA policy (pi05, gr00t, etc.) — TipTop has its own termination signal."
        )

    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    instruction = env_cfg.instruction
    action_dim = 8  # 7 joints + 1 gripper

    subtask_status = []

    client = PolicyClient(remote_host=remote_host, remote_port=remote_port)
    clients = [client] * env.num_envs

    if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")
    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_{episode}_env{env_id}" if env.num_envs > 1 else f"_{episode}"
            if save_sensor:
                video_path = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4")
                video_writers_obs.append(VideoWriter(video_path, video_fps))
            if save_viewport:
                video_path_viewport = os.path.join(
                    get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4"
                )
                video_writers_viewport.append(VideoWriter(video_path_viewport, video_fps))

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    # --- Harness setup (env 0 only) ---
    harness_dir = Path(get_output_dir())
    log_file = harness_dir / f"harness_{episode}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [Harness] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    def hlog(msg, color="\033[96m"):
        print(f"{color}[Harness] {msg}\033[0m")
        logger.info(msg)

    initial_frame = unpack_image_obs(obs, scale=0.5, env_id=0).get("combined_image")
    monitor = ProgressMonitor(check_every_n_steps=check_every_n_steps, model_id=vlm_model)

    memory: MemoryManager | None = None
    current_subtask = instruction
    subtask_index = 0
    subtask_start_step = 0
    subtask_log: list[dict] = []
    prev_frame = initial_frame
    goal_achieved = False

    hlog(f"Instruction: \"{instruction}\"")

    if dynamic_prompting:
        memory = MemoryManager(harness_dir / f"memory_{episode}.md", model_id=vlm_model)
        try:
            memory.reset(instruction, initial_frame=initial_frame)
        except Exception as e:
            hlog(f"Initial memory.reset failed (continuing without scene description): {type(e).__name__}: {e}", color="\033[91m")
        timer.start("vlm_next_subtask")
        try:
            plan_result = get_next_subtask(instruction, initial_frame, memory.get_memory())
        except Exception as e:
            hlog(f"Initial get_next_subtask failed (falling back to raw instruction): {type(e).__name__}: {e}", color="\033[91m")
            from robolab.inference.vlm_done_checker import NextSubtaskResult
            plan_result = NextSubtaskResult(subtask=instruction, done=False)
        timer.stop("vlm_next_subtask")
        if plan_result.done:
            hlog(f"VLM says goal is already achieved: \"{instruction}\"")
            goal_achieved = True
        else:
            current_subtask = plan_result.subtask or instruction
            hlog(f"Subtask {subtask_index + 1}: \"{current_subtask}\"")

    actual_steps = 0
    try:
        for step in tqdm(range(max_steps)):

            while not timeline.is_playing():
                kit_app.update()

            timer.start("policy_inference")
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            for env_id in env.active_env_ids:
                # env 0 gets the current (sub)task; others get the raw instruction.
                instr = current_subtask if (env_id == 0 and dynamic_prompting) else instruction
                ret = clients[env_id].infer(obs, instr, env_id=env_id)
                actions[env_id] = torch.tensor(ret["action"], device=env.device)
                if env_id == 0 or last_viz is None:
                    last_viz = ret.get("viz")
            timer.stop("policy_inference")

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

            # --- VLM check (env 0 only, synchronous, every N policy steps) ---
            if (
                not goal_achieved
                and 0 in env.active_env_ids
                and step % check_every_n_steps == 0
                and step > 0
            ):
                frame = unpack_image_obs(obs, scale=0.5, env_id=0).get("combined_image")
                monitor.set_frame(frame)

                timer.start("vlm_check")
                try:
                    if dynamic_prompting:
                        mem_ctx = memory.context_for(current_subtask, subtask_index)
                        result = monitor.check_completion(
                            current_subtask, memory=mem_ctx, before_frame=memory.last_frame()
                        )
                    else:
                        result = monitor.check_completion(
                            instruction, memory="", before_frame=prev_frame
                        )
                except Exception as e:
                    timer.stop("vlm_check")
                    hlog(f"VLM check failed (skipping): {type(e).__name__}: {e}", color="\033[91m")
                    prev_frame = frame
                    continue
                timer.stop("vlm_check")

                if vlm_verbose:
                    logger.info(
                        "VLM check: completed=%s reason=%s",
                        result["completed"],
                        result.get("reason", ""),
                    )

                subtask_elapsed = step - subtask_start_step

                if result["completed"]:
                    if dynamic_prompting:
                        hlog(
                            f"Subtask succeeded: {subtask_index + 1} \"{current_subtask}\" | {result['reason']}",
                            color="\033[92m",
                        )
                        subtask_log.append({
                            "index": subtask_index + 1,
                            "subtask": current_subtask,
                            "status": "succeeded",
                            "steps_taken": subtask_elapsed,
                        })
                        try:
                            memory.update(frame, current_subtask)
                        except Exception as e:
                            hlog(f"memory.update failed (continuing): {type(e).__name__}: {e}", color="\033[91m")

                        timer.start("vlm_next_subtask")
                        try:
                            plan_result = get_next_subtask(instruction, frame, memory.get_memory())
                        except Exception as e:
                            hlog(f"get_next_subtask failed (keeping current subtask): {type(e).__name__}: {e}", color="\033[91m")
                            timer.stop("vlm_next_subtask")
                            prev_frame = frame
                            continue
                        timer.stop("vlm_next_subtask")

                        if plan_result.done:
                            hlog(f"Goal achieved: \"{instruction}\"", color="\033[92m")
                            goal_achieved = True
                        else:
                            subtask_index += 1
                            current_subtask = plan_result.subtask or instruction
                            subtask_start_step = step
                            hlog(f"Subtask {subtask_index + 1}: \"{current_subtask}\"")
                    else:
                        hlog(
                            f"Goal achieved: \"{instruction}\" | {result['reason']}",
                            color="\033[92m",
                        )
                        goal_achieved = True

                elif dynamic_prompting and subtask_elapsed >= subtask_timeout_steps:
                    hlog(
                        f"Subtask timed out: {subtask_index + 1} \"{current_subtask}\" after {subtask_elapsed} steps",
                        color="\033[91m",
                    )
                    subtask_log.append({
                        "index": subtask_index + 1,
                        "subtask": current_subtask,
                        "status": "timed_out",
                        "steps_taken": subtask_elapsed,
                    })
                    try:
                        memory.update(frame, current_subtask)
                    except Exception as e:
                        hlog(f"memory.update failed (continuing): {type(e).__name__}: {e}", color="\033[91m")

                    timer.start("vlm_next_subtask")
                    try:
                        plan_result = get_next_subtask(instruction, frame, memory.get_memory())
                    except Exception as e:
                        hlog(f"get_next_subtask failed (keeping current subtask): {type(e).__name__}: {e}", color="\033[91m")
                        timer.stop("vlm_next_subtask")
                        prev_frame = frame
                        continue
                    timer.stop("vlm_next_subtask")

                    if plan_result.done:
                        hlog(f"Goal achieved after timeout reprompt: \"{instruction}\"", color="\033[92m")
                        goal_achieved = True
                    else:
                        subtask_index += 1
                        current_subtask = plan_result.subtask or instruction
                        subtask_start_step = step
                        hlog(f"Subtask {subtask_index + 1} (reprompted): \"{current_subtask}\"")
                else:
                    if dynamic_prompting:
                        hlog(
                            f"Subtask not done: {subtask_index + 1} \"{current_subtask}\" | "
                            f"{result['reason']} ({subtask_elapsed}/{subtask_timeout_steps} steps)",
                            color="\033[93m",
                        )
                    else:
                        hlog(
                            f"Goal not done: \"{instruction}\" | {result['reason']}",
                            color="\033[93m",
                        )

                prev_frame = frame

            # --- VLM-signalled termination: freeze env 0 only (harness runs there). ---
            if goal_achieved and not env._frozen_envs[0]:
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
    finally:
        if dynamic_prompting and not goal_achieved:
            subtask_log.append({
                "index": subtask_index + 1,
                "subtask": current_subtask,
                "status": "abandoned",
                "steps_taken": actual_steps - subtask_start_step,
            })

        try:
            if dynamic_prompting:
                with open(harness_dir / f"subtasks_{episode}.json", "w") as f:
                    json.dump({
                        "goal": instruction,
                        "goal_achieved": goal_achieved,
                        "subtasks": subtask_log,
                    }, f, indent=2)
            else:
                with open(harness_dir / f"subtasks_{episode}.json", "w") as f:
                    json.dump({
                        "goal": instruction,
                        "goal_achieved": goal_achieved,
                        "subtasks": {1: instruction},
                    }, f, indent=2)
        except Exception:
            logger.exception("Failed to write subtasks_%d.json", episode)

        logger.removeHandler(file_handler)
        file_handler.close()

    if save_videos:
        for vw in video_writers_obs + video_writers_viewport:
            vw.release()

    client.reset()

    timing = timer.to_dict(actual_steps)
    return env.get_env_results(), subtask_status, timing
