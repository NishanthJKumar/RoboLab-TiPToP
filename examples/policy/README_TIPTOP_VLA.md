# TipTop + VLA + VLM hybrid

This folder contains three related eval scripts. The new one,
[`run_eval_tiptop_vla_vlm.py`](run_eval_tiptop_vla_vlm.py), combines the other
two into a state machine where a Gemini VLM mediates between a symbolic
planner (TipTop) and a learned VLA policy (e.g. pi05).

The intuition: **TipTop is precise but brittle** (it can fail to plan, or its
Pick/Place can slip). **The VLA is flexible but slow and less reliable**. The
VLM watches each TipTop subtask, decides when something has gone wrong, and
hands a single combined pick+place recovery instruction to the VLA. After the
VLA finishes, control flips back to TipTop. Two scripts you already have run
the planner-only and the VLA-only halves of this idea independently; the
hybrid stitches them together.

## Scripts in this folder

| Script | Role |
|---|---|
| [`run_eval_tiptop_vlm.py`](run_eval_tiptop_vlm.py) | TipTop alone. VLM **passively logs** whether each Pick/Place succeeded — no replanning, no recovery. |
| [`run_eval_vla_vlm.py`](run_eval_vla_vlm.py) | VLA alone. VLM either terminates the episode when it judges the task done, or (with `--dynamic-prompting`) reactively feeds the VLA a sequence of subtasks. |
| [`run_eval_tiptop_vla_vlm.py`](run_eval_tiptop_vla_vlm.py) | **Hybrid (this README).** Full state machine, see below. |

## State machine (hybrid)

```
                  episode start
                       │
                       ▼
              ┌────► TIPTOP ◄─────────┐
              │       │                │
              │       │ VLM at         │ plan succeeds
              │       │ Pick/Place     │
              │       │ boundary       │
              │       │  ▼             │
              │     "failed?"          │
              │       │ yes            │
              │       ▼                │
              │  VLA-RECOVERY          │
              │  (one combined         │
              │   pick + place)        │
              │       │ done or        │
              │       │ timeout        │
              │       ▼                │
              └─── HOMING ─────────────┘
                  (drive to canonical
                   joint pose, gripper
                   open, then replan)
```

- **TIPTOP**: planner drives. Each Pick/Place boundary schedules a VLM check `--tiptop-vlm-check-delay-steps` later.
- **VLA-RECOVERY**: Gemini picks ONE combined pick+place (e.g. *"pick up the green lizard and place it in the bin"*) and the VLA executes it. VLM polls every `--vla-vlm-check-every-n-steps` for completion, with `--vla-subtask-timeout-steps` as a hard ceiling.
- **HOMING**: between VLA→TipTop transitions, the arm is driven to the Franka home pose with gripper open for `--home-pose-steps` steps. This gives TipTop a clean starting joint config to replan from (otherwise CuRobo often produces wild trajectories from arbitrary VLA end-poses).
- If TipTop ever raises `PlanningError` (initial plan fails, or replan fails), we drop straight to VLA-RECOVERY.
- No replan cap — `episode_length_s` is the only ceiling.

## Output video legend

Every saved frame gets a colored banner so it's obvious who's driving:

| Color | Mode |
|---|---|
| Green | TIPTOP |
| Blue | HOMING |
| Orange | VLA-RECOVERY |
| Red (1-frame flash) | mode just switched |

## Example commands

All assume a TipTop server at `luma02:31416`, a pi05 server at `luma02:31415`,
and `GEMINI_API_KEY` exported.

### TipTop alone

```bash
uv run python examples/policy/run_eval_tiptop_vlm.py \
    --task AnimalsInBinTask --num-envs 1 \
    --remote-host luma02.csail.mit.edu --remote-port 31416 \
    --headless --vlm-check-delay-steps 50 --save-vlm-debug-images
```

### VLA alone (dynamic subtasks)

```bash
uv run python examples/policy/run_eval_vla_vlm.py \
    --policy pi05 --task AnimalsInBinTask --num-envs 1 \
    --remote-host luma02.csail.mit.edu --remote-port 31415 \
    --headless --video-mode all \
    --check-every-n-steps 15 --dynamic-prompting --subtask-timeout-steps 75
```

### Hybrid

```bash
uv run python examples/policy/run_eval_tiptop_vla_vlm.py \
    --task AnimalsInBinTask --num-envs 1 \
    --tiptop-host luma02.csail.mit.edu --tiptop-port 31416 \
    --vla-host luma02.csail.mit.edu --vla-port 31415 \
    --vla-policy pi05 --headless --video-mode all \
    --tiptop-vlm-check-delay-steps 50 \
    --vla-vlm-check-every-n-steps 15 \
    --vla-subtask-timeout-steps 75 \
    --home-pose-steps 30 --save-vlm-debug-images
```

## Per-episode artifacts

Under `output/<timestamp>_<method>/<task>/`:

- `<instruction>_<episode>.mp4` / `..._viewport.mp4` — sensor and viewport videos with mode banner.
- `harness_<episode>.log` — timestamped events: mode transitions, VLM judgements, replan attempts.
- `vlm_checks_<episode>.json` — every VLM call (boundary checks, recovery checks, overall-done checks) with judgements.
- `mode_transitions_<episode>.json` — every mode switch with step, reason, subtask before/after. Useful for video review.
- `vlm_debug_<episode>/*.png` — (with `--save-vlm-debug-images`) every frame sent to the VLM, named by call site and step.
- `run_<episode>.hdf5` — full proprio/image trajectory (flushed even on Ctrl+C).

## Where to look first

| File | Why |
|---|---|
| [`episode_tiptop_vla_vlm.py`](episode_tiptop_vla_vlm.py) | The state machine. Read top-to-bottom. |
| [`episode_tiptop_vlm.py`](episode_tiptop_vlm.py) | Hybrid reuses `_parse_subtask_label`, `_build_vlm_view`, and the Pick/Place prompt builders from here. |
| [`../../robolab/inference/tiptop.py`](../../robolab/inference/tiptop.py) | `current_subtask_label` and `plan_done` drive boundary detection; `reset()` + `infer()` is the replan mechanism. |
| [`../../robolab/inference/vlm_done_checker.py`](../../robolab/inference/vlm_done_checker.py) | All Gemini prompts. `get_recovery_pick_place` is the prompt for the VLA-recovery instruction. |
| [`../../robolab/registrations/droid_jointpos/auto_env_registrations.py`](../../robolab/registrations/droid_jointpos/auto_env_registrations.py) | `keep_external_cam=True` is what lets TipTop (wrist depth/intrinsics) and the VLA (external cam) coexist in one env. |

## Notes for the next person

- **TipTop perception sometimes fails** on small/cluttered scenes (you'll see `PlanningError` followed by an auto-switch to VLA-RECOVERY — that's the designed fallback, not a bug).
- **`msgpack_numpy.patch()` is intentionally NOT called** in `tiptop.py`. Adding it back would re-break pi05 inference in the hybrid by globally monkey-patching msgpack and shadowing openpi-client's encoder. There's a defensive override in [`pi0_family.py`](../../robolab/inference/pi0_family.py) that forces openpi's encoder regardless, but don't undo either change without thinking.
- **Ctrl+C is safe**: videos and HDF5s are flushed in `finally`, and the launcher catches `KeyboardInterrupt` so the sim shuts down cleanly.
- **TipTop replan uses the ORIGINAL task instruction**, not the recovery subtask. The recovery instruction is only ever fed to the VLA.
- **Easy way to get a fast smoke test**: add `--episode-length-s 30` to any of the commands above.
