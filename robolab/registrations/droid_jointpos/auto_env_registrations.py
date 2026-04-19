# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR

"""
Scene registration:

For the same task, we can register multiple variants. For example, the following script will register something like this:

Task Name                | Environment                       | Config Class                            | Reg | Tags
---------------------------------------------------------------------------------------------------------------------------------------------------
BagelOnPlateTableTask    | BagelOnPlateTableTaskHomeOffice   | BagelOnPlateTableTaskHomeOfficeEnvCfg   | ✓   | all, pick_place
BagelOnPlateTableTask    | BagelOnPlateTableTaskBilliardHall | BagelOnPlateTableTaskBilliardHallEnvCfg | ✓   | all, pick_place
BananaInBowlTableTask    | BananaInBowlTableTaskHomeOffice   | BananaInBowlTableTaskHomeOfficeEnvCfg   | ✓   | all, pick_place
BananaInBowlTableTask    | BananaInBowlTableTaskBilliardHall | BananaInBowlTableTaskBilliardHallEnvCfg | ✓   | all, pick_place

The columns are:
- Task Name: The base task class name (groups variants together)
- Environment: The registered environment name (also the Gymnasium ID)
- Config Class: The generated configuration class name
- Reg: Registration status (✓ = registered with Gymnasium)
- Tags: Tag names this environment belongs to

"""
def auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, lighting_intensity=None, task=None, enable_camera_params=False):
    """Automatically discover and register tasks.

    Args:
        task_dirs: Subdirectories to search for tasks.
        lighting_intensity: Optional lighting intensity override.
        task: If provided, only register the specified task(s) instead of discovering
              all tasks. Accepts a single task name/filename/path (str) or a list of them.
              Significantly faster when running a subset of tasks.
        enable_camera_params: If True, enable wrist-camera depth + intrinsics + extrinsics
              observations (required by TipTop). Off by default because adding
              ``distance_to_image_plane`` to the tiled wrist camera roughly doubles its
              render-buffer VRAM cost, which pushes high ``num_envs`` runs out of memory.
    """
    from robolab.core.environments.factory import auto_discover_and_create_cfgs, create_env_cfg
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.droid_jointpos.observations import ImageObsCfg, ProprioceptionObservationCfg
    from robolab.robots.droid import (
        DroidCfg,
        DroidJointPositionActionCfg,
        ProprioceptionObservationCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg, OverShoulderLeftCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    if enable_camera_params:
        # TipTop path: it only consumes the wrist camera (RGB + depth + intrinsics/extrinsics)
        # at plan time. Drop external_cam and viewport cam to save render cost every step.
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import configclass
        import isaaclab.envs.mdp as mdp

        @configclass
        class WristOnlyImageObsCfg(ObsGroup):
            wrist_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = False

        obs_groups = {
            "image_obs": WristOnlyImageObsCfg(),
            "proprio_obs": ProprioceptionObservationCfg(),
        }

        from robolab.robots.droid_camera_params import CameraParamsObservationCfg
        # Enable depth on the wrist camera — cost: ~2x wrist render-buffer VRAM per env.
        # @configclass turns wrist_cam into a dataclass field with a default_factory,
        # so we patch the factory rather than the (absent) class attribute.
        _wrist_field = DroidCfg.__dataclass_fields__["wrist_cam"]
        _orig_factory = _wrist_field.default_factory
        def _wrist_factory_with_depth(_orig=_orig_factory):
            cam = _orig()
            if "distance_to_image_plane" not in cam.data_types:
                cam.data_types = list(cam.data_types) + ["distance_to_image_plane"]
            return cam
        _wrist_field.default_factory = _wrist_factory_with_depth
        obs_groups["camera_params_obs"] = CameraParamsObservationCfg()

        camera_cfg = []
    else:
        obs_groups = {
            "image_obs": ImageObsCfg(),
            "proprio_obs": ProprioceptionObservationCfg(),
            "viewport_cam": ViewportCameraCfg(),
        }
        camera_cfg = [OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg]

    ObservationCfg = generate_obs_cfg(obs_groups)

    shared_kwargs = dict(
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=camera_cfg,
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )

    if task is not None:
        tasks = task if isinstance(task, list) else [task]
        print(f"\033[96m[RoboLab] Registering {len(tasks)} task(s): {tasks}\033[0m")
        for t in tasks:
            create_env_cfg(
                t,
                task_dir=TASK_DIR,
                env_prefix="",
                env_postfix="",
                **shared_kwargs,
            )
    else:
        print(f"\033[96m[RoboLab] Registering all tasks in {task_dirs}\033[0m")
        for subdir in task_dirs:
            auto_discover_and_create_cfgs(
                task_dir=TASK_DIR,
                task_subdirs=[subdir],
                pattern="*.py",
                env_prefix="",
                env_postfix="",
                **shared_kwargs,
            )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
