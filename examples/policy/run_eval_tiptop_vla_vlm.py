# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run TipTop + VLA + VLM hybrid evaluation.

Tries TipTop first; on planning failure or per-Pick/Place VLM failure, drops
to a single combined-pick+place VLA recovery picked by the VLM; after each
recovery subtask completes (or times out) retries TipTop from the current
state. Saved videos overlay a colored mode banner so the viewer can see when
control switches between TipTop and the VLA.

Single-env only (TipTop limitation). Requires:
    - GEMINI_API_KEY (or GOOGLE_API_KEY) for the Gemini VLM
    - A running TipTop websocket server (default port 8765)
    - A running VLA policy server (default port 8000)

Usage:
    uv run python examples/policy/run_eval_tiptop_vla_vlm.py \\
        --task BananaOnPlateTask --num-envs 1 \\
        --tiptop-host luma02.csail.mit.edu --tiptop-port 8765 \\
        --vla-host luma02.csail.mit.edu --vla-port 8000 \\
        --vla-policy pi05 --headless --video-mode all \\
        --tiptop-vlm-check-delay-steps 10 \\
        --vla-vlm-check-every-n-steps 15 \\
        --vla-subtask-timeout-steps 150
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import re
import traceback
import sys
from collections import Counter
from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS # noqa

parser = argparse.ArgumentParser(description="")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1,
                    help="Number of environments to spawn. Must be 1 — TipTop is single-env only.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None,
                    help="List of tasks to evaluate on")
parser.add_argument("--tag", nargs='+', default=None,
                    help="List of tags of tasks to evaluate on")
parser.add_argument("--task-dirs", nargs='+', default=DEFAULT_TASK_SUBFOLDERS,
                    help="List of task directories to evaluate on")
parser.add_argument("--vla-policy", "--vla_policy",
                    choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05",
                             "gr00t", "dreamzero", "molmo", "openvla", "openvla_oft"],
                    default="pi05",
                    help="VLA backend used as the recovery executor (default: pi05). "
                         "TipTop is always the primary planner.")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                    help="Number of sequential runs per task (default: 1).")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true",
                    help="Enable subtask progress checking (default: False)")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                    help="Enable proprio image data recording (default: False)")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                    help="Output folder name under /robolab/output. "
                         "Default is <timestamp>_tiptop_vla_vlm_<policy>.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                    help="Verbose output (default: False)")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                    help="Debug output (default: False)")
parser.add_argument("--tiptop-host", "--tiptop_host", type=str, default="localhost",
                    help="Remote host for the TipTop websocket server (default: localhost).")
parser.add_argument("--tiptop-port", "--tiptop_port", type=int, default=8765,
                    help="Remote port for the TipTop websocket server (default: 8765).")
parser.add_argument("--vla-host", "--vla_host", type=str, default="localhost",
                    help="Remote host for the VLA policy server (default: localhost).")
parser.add_argument("--vla-port", "--vla_port", type=int, default=8000,
                    help="Remote port for the VLA policy server (default: 8000).")
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                    help="Which instruction variant to use when a task defines multiple")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save (default: all)")
# --- TipTop-side VLM args ---
parser.add_argument("--tiptop-vlm-check-delay-steps", "--tiptop_vlm_check_delay_steps",
                    type=int, default=10,
                    help="(TipTop mode) Sim steps to wait after a TipTop Pick/Place subtask "
                         "boundary before running the VLM check on that subtask (default: 10). "
                         "Just needs to cover settling time.")
# --- VLA-side VLM args ---
parser.add_argument("--vla-vlm-check-every-n-steps", "--vla_vlm_check_every_n_steps",
                    type=int, default=15,
                    help="(VLA-recovery mode) How often (in VLA policy steps) to ask the VLM "
                         "whether the current recovery pick+place is done (default: 15).")
parser.add_argument("--vla-subtask-timeout-steps", "--vla_subtask_timeout_steps",
                    type=int, default=150,
                    help="(VLA-recovery mode) Max VLA steps on one recovery subtask before it "
                         "is force-completed and TipTop is retried (default: 150).")
# --- Shared VLM args (apply to both TipTop boundary checks and VLA-recovery checks) ---
parser.add_argument("--vlm-model", "--vlm_model", type=str, default="gemini-robotics-er-1.6-preview",
                    help="(Shared) Gemini model id used for both TipTop boundary checks and "
                         "VLA-recovery completion/recovery-subtask prompts.")
parser.add_argument("--vlm-verbose", "--vlm_verbose", action="store_true",
                    help="(Shared) Save raw VLM responses in vlm_checks_*.json (default: False).")
parser.add_argument("--save-vlm-debug-images", "--save_vlm_debug_images", action="store_true",
                    help="(Shared) Save frames sent to the VLM under vlm_debug_<episode>/ for "
                         "offline debugging (default: False).")
parser.add_argument("--tiptop-waypoint-stride", "--tiptop_waypoint_stride", type=int, default=None,
                    help="Override the TipTop client's trajectory waypoint stride "
                         "(default: derived from sim/CuRobo Hz, typically 3).")
parser.add_argument("--tiptop-gripper-steps", "--tiptop_gripper_steps", type=int, default=20,
                    help="Sim steps to hold the gripper command during each Pick/Place gripper "
                         "step (default: 20, ~1.3s at 15Hz).")
parser.add_argument("--home-pose-steps", "--home_pose_steps", type=int, default=30,
                    help="Sim steps to drive the arm to a canonical home pose (gripper open) "
                         "before re-querying TipTop after VLA recovery or a plan completion that "
                         "didn't achieve the overall goal (default: 30, ~2s at 15Hz). Set to 0 "
                         "to skip homing.")
parser.add_argument("--episode-length-s", "--episode_length_s", type=float, default=None,
                    help="Override env_cfg.episode_length_s (in seconds) before each episode.")

args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"

if args_cli.num_envs != 1:
    raise SystemExit(
        f"TipTop+VLA+VLM requires --num-envs 1 (got {args_cli.num_envs}). "
        "TipTop is single-env only."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import PACKAGE_DIR, set_output_dir # noqa
from episode_tiptop_vla_vlm import run_episode_tiptop_vla_vlm # noqa
from robolab.core.environments.runtime import create_env # noqa
from robolab.core.logging.recorder_manager import patch_recorder_manager # noqa
from robolab.core.environments.factory import get_envs # noqa
from robolab.core.utils.print_utils import print_experiment_summary # noqa
from robolab.core.logging.results import check_all_episodes_complete, check_run_complete, dump_results_to_file # noqa
from robolab.core.logging.results import init_experiment, update_experiment_results, summarize_experiment_results, get_final_subtask_info # noqa
from robolab.core.metrics import load_demo_data, compute_episode_metrics # noqa
from robolab.core.logging.results import extract_subtask_status_changes # noqa
from robolab.core.task.status import StatusCode, get_status_name # noqa
from robolab.core.utils.file_utils import load_file # noqa
import robolab.constants # noqa

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

patch_recorder_manager()

# TipTop needs wrist depth + intrinsics + extrinsics (enable_camera_params=True),
# AND the VLA needs the external_cam in image_obs (keep_external_cam=True). The TipTop
# path normally drops external_cam to save VRAM; we opt back in here because the hybrid
# needs both observations live in the same episode.
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs # noqa
auto_register_droid_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    enable_camera_params=True,
    keep_external_cam=True,
)

METHOD_LABEL = f"tiptop+vla+vlm_{args_cli.vla_policy}"

EVENT_STATUS_CODES = {
    StatusCode.WRONG_OBJECT_GRABBED_FAILURE,
    StatusCode.GRIPPER_HIT_TABLE,
    StatusCode.WRONG_OBJECT_DETACHED,
    StatusCode.OBJECT_BUMPED,
    StatusCode.OBJECT_MOVED,
    StatusCode.OBJECT_OUT_OF_SCENE,
    StatusCode.OBJECT_TIPPED_OVER,
    StatusCode.TARGET_OBJECT_DROPPED,
    StatusCode.GRIPPER_HIT_OBJECT,
    StatusCode.MULTIPLE_OBJECTS_GRABBED,
    StatusCode.GRIPPER_FULLY_CLOSED,
}


def _extract_events_from_log(log_file: str) -> dict:
    if not os.path.exists(log_file):
        return {}

    log_data = load_file(log_file)
    if log_data is None:
        return {}

    status_changes = extract_subtask_status_changes(log_data)
    if not status_changes:
        return {}

    event_counts: Counter = Counter()
    wrong_objects_grabbed: list[str] = []

    for change in status_changes:
        status_code = change.get("status", 0)
        if status_code not in EVENT_STATUS_CODES:
            continue

        event_name = get_status_name(status_code)
        if event_name.endswith("_FAILURE"):
            event_name = event_name[:-8]

        event_counts[event_name] += 1

        if status_code == StatusCode.WRONG_OBJECT_GRABBED_FAILURE:
            info = change.get("info", "")
            match = re.search(r"Wrong object grabbed: '([^']+)'", info)
            if match:
                wrong_objects_grabbed.append(match.group(1))

    events: dict = {}
    for event_name, count in event_counts.items():
        events[event_name] = count
    if wrong_objects_grabbed:
        events["wrong_objects_grabbed"] = wrong_objects_grabbed

    return events


def main():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError(
            "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. "
            "Export one before running the TipTop+VLA+VLM script."
        )

    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = get_timestamp() + f"_tiptop_vla_vlm_{args_cli.vla_policy}"
        if args_cli.instruction_type != "default":
            args_cli.output_folder_name += f"_{args_cli.instruction_type}"

    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.task:
        task_envs = get_envs(task=args_cli.task)
        filter_str = f"tasks: {', '.join(args_cli.task)}"
    elif args_cli.tag:
        task_envs = get_envs(tag=args_cli.tag)
        filter_str = f"tags: {', '.join(args_cli.tag)}"
    else:
        task_envs = get_envs()
        filter_str = "all"

    num_envs = args_cli.num_envs
    num_runs = args_cli.num_runs
    total_episodes = num_runs * num_envs

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=num_envs,
        num_episodes=total_episodes,
        policy=METHOD_LABEL,
        instruction_type=args_cli.instruction_type,
        output_dir=output_dir,
    )

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(episode_results=episode_results, env_name=task_env, num_episodes=total_episodes):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        # Pass the VLA backend as the `policy` arg so env_cfg.policy is set to the VLA
        # name — the episode runner reads it to select the right VLA client class. The
        # env still gets TipTop-style camera_params_obs because the auto_register call
        # above sets enable_camera_params=True (with keep_external_cam=True so the VLA
        # also gets the third-person external view).
        env, env_cfg = create_env(task_env,
            device=args_cli.device,
            num_envs=num_envs,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy=args_cli.vla_policy)

        if args_cli.episode_length_s is not None:
            env.cfg.episode_length_s = args_cli.episode_length_s
            print(f"\033[96m[RoboLab] Overriding episode_length_s -> {args_cli.episode_length_s}s "
                  f"(max_episode_length now {env.max_episode_length} steps)\033[0m")

        for run_idx in range(num_runs):
            run_episode_ids = [run_idx * num_envs + eid for eid in range(num_envs)]
            if all(check_run_complete(episode_results=episode_results, env_name=task_env, episode=ep_id) for ep_id in run_episode_ids):
                print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                continue

            if args_cli.instruction_type != "default":
                run_name = task_env + f"_{args_cli.instruction_type}_{run_idx}"
            else:
                run_name = task_env + f"_{run_idx}"
            print(f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' "
                  f"(run {run_idx}, TipTop+VLA+VLM hybrid, vla_policy={args_cli.vla_policy})\033[0m")
            print(f"\033[96m[RoboLab] TipTop server {args_cli.tiptop_host}:{args_cli.tiptop_port}  "
                  f"|  VLA server {args_cli.vla_host}:{args_cli.vla_port}  "
                  f"|  VLM {args_cli.vlm_model}\033[0m")

            env_results, msgs, timing = run_episode_tiptop_vla_vlm(
                env=env,
                env_cfg=env_cfg,
                episode=run_idx,
                save_videos=args_cli.save_videos,
                video_mode=args_cli.video_mode,
                headless=args_cli.headless,
                tiptop_host=args_cli.tiptop_host,
                tiptop_port=args_cli.tiptop_port,
                vla_host=args_cli.vla_host,
                vla_port=args_cli.vla_port,
                vlm_check_delay_steps=args_cli.tiptop_vlm_check_delay_steps,
                check_every_n_steps=args_cli.vla_vlm_check_every_n_steps,
                subtask_timeout_steps=args_cli.vla_subtask_timeout_steps,
                vlm_model=args_cli.vlm_model,
                vlm_verbose=args_cli.vlm_verbose,
                save_vlm_debug_images=args_cli.save_vlm_debug_images,
                tiptop_gripper_steps=args_cli.tiptop_gripper_steps,
                tiptop_waypoint_stride=args_cli.tiptop_waypoint_stride,
                home_pose_steps=args_cli.home_pose_steps,
            )

            final_infos = get_final_subtask_info(env, env_id=None)

            per_env_msgs: dict[int, list] = {eid: [] for eid in range(num_envs)}
            for step_infos in msgs:
                if step_infos is None:
                    for eid in range(num_envs):
                        per_env_msgs[eid].append(None)
                else:
                    for eid in range(num_envs):
                        per_env_msgs[eid].append(step_infos[eid] if eid < len(step_infos) else None)

            per_env_events: dict[int, dict] = {}
            for eid in range(num_envs):
                log_file = os.path.join(scene_output_dir, f"log_{run_idx}_env{eid}.json")
                dump_results_to_file(log_file, per_env_msgs[eid], append=False)
                per_env_events[eid] = _extract_events_from_log(log_file)

            dt = env_cfg.sim.dt * env_cfg.decimation

            for r in env_results:
                env_id = r['env_id']
                episode_id = run_idx * num_envs + env_id

                hdf5_path = os.path.join(scene_output_dir, f"run_{run_idx}.hdf5")
                demo_key = f"demo_{env_id}"
                traj_data = load_demo_data(hdf5_path, demo_key)
                traj_metrics = compute_episode_metrics(traj_data, dt=dt) if traj_data else None

                events = per_env_events.get(env_id, {})

                run_summary = {
                    "env_name": task_env,
                    "task_name": env_cfg._task_name,
                    "run_name": run_name,
                    "run": run_idx,
                    "episode": episode_id,
                    "env_id": env_id,
                    "policy": METHOD_LABEL,
                    "instruction": env_cfg.instruction,
                    "instruction_type": args_cli.instruction_type,
                    "attributes": env_cfg._task_attributes,
                    "success": r['success'],
                    "episode_step": r['step'],
                    "duration": r['step'] * dt if r['step'] else 0,
                    "dt": dt,
                    "metrics": traj_metrics if traj_metrics else {},
                    "events": events if events else {},
                    "timing": timing,
                }

                if robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING:
                    env_msgs = per_env_msgs.get(env_id, [])
                    last_msg = None
                    for m in reversed(env_msgs):
                        if m is not None:
                            last_msg = m
                            break

                    if last_msg is not None:
                        run_summary["score"] = last_msg.get("score", None)
                        run_summary["reason"] = last_msg.get("info", None)
                    else:
                        run_summary["score"] = None
                        run_summary["reason"] = None

                    final_info = final_infos[env_id] if final_infos else None
                    if not r['success'] and final_info is not None:
                        run_summary["reason"] = final_info.get("info", run_summary.get("reason"))

                episode_results = update_experiment_results(run_summary=run_summary, episode_results=episode_results, episode_results_file=episode_results_file)

            env.reset_eval_state()

        env.close()

    summarize_experiment_results(episode_results, show_timing=True)

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Caught explicitly so simulation_app.close() runs even on Ctrl-C.
        # The episode runner's finally block already finalized the videos and
        # flushed the recorder; here we just shut the sim down cleanly.
        print("\033[96m[RoboLab] Interrupted by user (Ctrl-C). Shutting down.\033[0m")
        try:
            simulation_app.close()
        except Exception:
            pass
        sys.exit(130)  # 128 + SIGINT
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
