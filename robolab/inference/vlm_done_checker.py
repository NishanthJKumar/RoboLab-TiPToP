# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""VLM harness for the VLA + VLM hybrid method.

Three components, all backed by Google Gemini via the `google-genai` SDK:

- `ProgressMonitor`: asks the VLM whether a (sub)task is complete given the
  current frame, optionally with a before-frame for comparison and a memory
  blob for scene context. Returns a structured {completed, reason} dict.
- `MemoryManager`: maintains a `memory.md` file tracking the initial scene
  and what each completed subtask changed, so repeated subtasks can be
  disambiguated (e.g. "1 cube in bowl" vs "2 cubes in bowl").
- `get_next_subtask`: reactive VLM-driven planner that returns the next
  subtask given the goal, current scene, and memory.

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment before use.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

MODEL_ID = "gemini-robotics-er-1.6-preview"


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. "
            "Export one before running the VLA+VLM script."
        )
    return key


COMPLETION_PROMPT_TEMPLATE = """You are a robot task completion checker.

{memory}

The robot's current subtask is: "{subtask}"

You are given one or two camera images:
- If two images are provided: the FIRST image shows the scene before this subtask started, and the SECOND image shows the current scene.
- If one image is provided: it shows the current scene (no prior reference available).

Use the before image (if present) to judge whether the scene has changed in the expected way — not just whether the subtask description is satisfied in isolation.

Respond with JSON only, no markdown:
{{"completed": true/false, "reason": "brief explanation"}}
"""


MEMORY_WRITER_TEMPLATE = """A robot arm just completed a subtask. You are given two camera images: the FIRST shows the scene before, and the SECOND shows the scene after.

## Completed subtask
"{subtask}"

## Instructions
Describe only what changed between the two images, focusing on objects relevant to the task. Be short and concrete.

Respond with JSON only, no markdown:
{{"changes": "brief description of what changed (e.g. red cube moved from table to inside bowl)"}}
"""


NEXT_SUBTASK_PROMPT = """Given the overall goal, the current scene image, and what has been completed so far, decide the single next subtask the robot should perform.

## Memory
{memory}

## Rules
- Use the image to identify the current scene state and refer to objects by distinguishing attributes (e.g. color, size, label).
- Choose ONE concrete next subtask that makes progress toward the goal given the current scene — not what was originally planned, but what actually makes sense right now.
- Each subtask should be a short command for one continuous action (e.g. "put the blue cube in the bowl").
- Do NOT use motor primitives like "move arm left" or "open gripper".
- The subtask must have a visually verifiable end state.
- If the goal is already fully achieved in the current scene, set "done" to true.

Respond with JSON only, no markdown:
{{"subtask": "next subtask description or null if done", "done": true/false}}
"""


INITIAL_TEMPLATE_REACTIVE = """# Task Memory

## Goal
{goal}

## Initial Scene
{initial_scene}

## Completed Subtasks
(none yet)
"""


def _image_to_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    img = Image.fromarray(frame)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text


class ProgressMonitor:
    """Calls a VLM to check if the current subtask is done."""

    def __init__(self, check_every_n_steps: int = 15, model_id: str = MODEL_ID):
        self.client = genai.Client(api_key=_get_api_key())
        self.model_id = model_id
        self.check_every_n_steps = check_every_n_steps
        self._latest_frame: np.ndarray | None = None

    def reset(self):
        self._latest_frame = None

    def set_frame(self, frame: np.ndarray):
        """Store the current camera frame (H, W, 3 uint8 RGB)."""
        self._latest_frame = frame

    def check_completion(
        self,
        subtask: str,
        memory: str = "",
        before_frame: np.ndarray | None = None,
    ) -> dict:
        prompt = COMPLETION_PROMPT_TEMPLATE.format(subtask=subtask, memory=memory)

        current_part = types.Part.from_bytes(
            data=_image_to_bytes(self._latest_frame), mime_type="image/jpeg"
        )

        if before_frame is not None:
            before_part = types.Part.from_bytes(
                data=_image_to_bytes(before_frame), mime_type="image/jpeg"
            )
            contents = [before_part, current_part, prompt]
        else:
            contents = [current_part, prompt]

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0),
        )

        raw_response = response.text or ""
        try:
            parsed = json.loads(_strip_fence(raw_response))
            result = {
                "completed": bool(parsed.get("completed", False)),
                "reason": str(parsed.get("reason", "")),
            }
        except json.JSONDecodeError:
            result = {"completed": False, "reason": f"Failed to parse VLM response: {raw_response}"}

        result["raw_vlm_response"] = raw_response
        result["prompt_sent"] = prompt
        result["frame"] = self._latest_frame.copy()
        return result


class MemoryManager:
    """Maintains a memory.md file tracking scene state across subtasks."""

    def __init__(self, memory_path: Path, model_id: str = MODEL_ID):
        self.client = genai.Client(api_key=_get_api_key())
        self.model_id = model_id
        self.memory_path = Path(memory_path)
        self._memory: str = ""
        self._completed_subtasks: list[str] = []
        self._last_frame: np.ndarray | None = None

    def reset(self, goal: str, initial_frame: np.ndarray | None = None):
        initial_scene = (
            self._describe_initial_scene(initial_frame, goal)
            if initial_frame is not None
            else "(no image available)"
        )
        self._memory = INITIAL_TEMPLATE_REACTIVE.format(goal=goal, initial_scene=initial_scene)
        self._completed_subtasks = []
        self._last_frame = initial_frame.copy() if initial_frame is not None else None
        self._write()

    def get_memory(self) -> str:
        return self._memory

    def last_frame(self) -> np.ndarray | None:
        return self._last_frame

    def context_for(self, subtask: str, subtask_index: int) -> str:
        prior_count = self._completed_subtasks.count(subtask)
        warning = ""
        if prior_count > 0:
            warning = (
                f"\n⚠ WARNING: This exact subtask has already been completed "
                f"{prior_count} time(s). Do NOT mark it complete just because "
                f"the action from a previous run of this subtask is visible — "
                f"the scene must show additional progress beyond what is described "
                f"in 'Scene Before This Subtask' above.\n"
            )
        header = f"## Current Subtask Context\nSubtask {subtask_index + 1}: \"{subtask}\"{warning}"
        return self._memory + "\n" + header

    def update(self, frame: np.ndarray, completed_subtask: str):
        changes = self._get_scene_diff(frame, completed_subtask)
        entry = f"- {completed_subtask}: {changes}"
        if "(none yet)" in self._memory:
            self._memory = self._memory.replace("(none yet)", entry)
        else:
            self._memory = self._memory + "\n" + entry
        self._completed_subtasks.append(completed_subtask)
        self._last_frame = frame.copy()
        self._write()

    def _get_scene_diff(self, after_frame: np.ndarray, subtask: str) -> str:
        prompt = MEMORY_WRITER_TEMPLATE.format(subtask=subtask)
        after_part = types.Part.from_bytes(
            data=_image_to_bytes(after_frame), mime_type="image/jpeg"
        )
        if self._last_frame is not None:
            before_part = types.Part.from_bytes(
                data=_image_to_bytes(self._last_frame), mime_type="image/jpeg"
            )
            contents = [before_part, after_part, prompt]
        else:
            contents = [after_part, prompt]

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = _strip_fence(response.text or "")
        try:
            return json.loads(text).get("changes", text)
        except json.JSONDecodeError:
            return text

    def _describe_initial_scene(self, frame: np.ndarray, goal: str) -> str:
        image_part = types.Part.from_bytes(data=_image_to_bytes(frame), mime_type="image/jpeg")
        prompt = (
            f"Describe the current scene state for objects relevant to this task: \"{goal}\". "
            "Be short and concrete (e.g. \"red cube on table to the left of bowl, bowl is empty\"). "
            "Only mention task-relevant objects."
        )
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text.strip() if response.text else "(unable to describe scene)"

    def _write(self):
        self.memory_path.write_text(self._memory)


@dataclass
class NextSubtaskResult:
    subtask: str | None
    done: bool


def get_next_subtask(goal: str, scene_frame: np.ndarray, memory: str) -> NextSubtaskResult:
    """Ask the VLM to decide the single next subtask given the current scene and memory."""
    client = genai.Client(api_key=_get_api_key())
    prompt = NEXT_SUBTASK_PROMPT.format(goal=goal, memory=memory)

    image_part = types.Part.from_bytes(
        data=_image_to_bytes(scene_frame), mime_type="image/jpeg"
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(temperature=0.0),
    )

    text = _strip_fence(response.text or "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return NextSubtaskResult(subtask=None, done=False)
    return NextSubtaskResult(
        subtask=parsed.get("subtask"),
        done=bool(parsed.get("done", False)),
    )


RECOVERY_PICK_PLACE_PROMPT = """A symbolic task planner has either failed to find a plan or just executed a Pick/Place step that did not succeed. You will now hand control to a Vision-Language-Action (VLA) policy that can execute ONE pick-and-place command. Your job is to pick the single most useful pick+place that gets the scene closer to the overall goal.

## Overall goal
{goal}
{failed_context}{memory}
## Rules
- Output EXACTLY ONE command for the robot that includes BOTH a pick AND a place — e.g. "pick up the red cube and place it on the blue plate", "pick up the banana and put it back on the table". Never output a pick alone or a place alone.
- Use distinguishing object attributes that are clearly visible in the image (color, shape, label, relative position). Avoid pronouns or vague references.
- Choose the move that, after it is executed, will leave the scene in a state where the planner is most likely to succeed when called again — typically: undo a recent failure, clear an obstruction, or make direct progress on the goal.
- Do NOT use motor primitives (e.g. "open gripper", "move arm up"). The VLA expects a high-level pick+place.
- If the overall goal is already fully satisfied in the current image, set "done": true and "instruction": null.

Respond with JSON only, no markdown:
{{"instruction": "<single pick-and-place command, or null if done>", "done": true/false}}
"""


@dataclass
class RecoveryResult:
    instruction: str | None
    done: bool


def get_recovery_pick_place(
    goal: str,
    scene_frame: np.ndarray,
    failed_subtask: str | None = None,
    memory: str = "",
) -> RecoveryResult:
    """Ask the VLM for ONE combined pick+place command for the VLA to execute as recovery.

    Used by the TipTop+VLA+VLM hybrid runner whenever the planner either can't produce
    a plan from the current state or just executed a Pick/Place that the VLM judged
    failed. The returned instruction is fed straight to the VLA as its language goal.
    """
    client = genai.Client(api_key=_get_api_key())
    failed_context = (
        f"\n## Failed planner subtask\nThe planner most recently attempted: \"{failed_subtask}\". "
        "The VLM judged this Pick/Place to have failed. Plan your recovery accordingly.\n"
        if failed_subtask
        else "\n## Failed planner subtask\n(none — the planner could not find a plan from the current state)\n"
    )
    memory_block = f"\n## Memory\n{memory}\n" if memory else ""
    prompt = RECOVERY_PICK_PLACE_PROMPT.format(
        goal=goal, failed_context=failed_context, memory=memory_block
    )

    image_part = types.Part.from_bytes(
        data=_image_to_bytes(scene_frame), mime_type="image/jpeg"
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(temperature=0.0),
    )

    text = _strip_fence(response.text or "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return RecoveryResult(instruction=None, done=False)
    return RecoveryResult(
        instruction=parsed.get("instruction"),
        done=bool(parsed.get("done", False)),
    )
