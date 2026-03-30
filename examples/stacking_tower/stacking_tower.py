#!/usr/bin/env python3
"""Stacking tower bimanual manipulation demo.

Usage:
    python examples/stacking_tower/generate_tower.py   # generate assets first
    python examples/stacking_tower/stacking_tower.py -v # run with viewer
"""

import argparse
import hashlib
import os
import pickle
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

import genesis as gs
from genesis.typing import FArrayType, NonNegativeFloat, PositiveFloat, StrictInt, ValidFloat, Vec3FType, Vec4FType
from genesis.utils.misc import tensor_to_array


########################## tower / robot constants ##########################

BASE_HEIGHT = 0.020
RING_HEIGHT = 0.020
RINGS_ORDER = (0, 1, 2, 3, 5, 4)
BALL_HEIGHT = 0.0215
POLE_HEIGHT = 0.145
TABLE_HEIGHT = 0.755
TOWER_OFFSET_X = -0.02

LIFT_CLEARANCE = 0.02
BALL_TABLE_OFFSET_Y = -0.18

APPROACH_DIST = 0.05
GRASP_DEPTH = 0.03

# Maximum per-DOF acceleration (rad/s²)
MAX_ACCEL = 13.0  # rad/s²
# Velocity overhead factor: accounts for the acceleration/deceleration ramps that reduce the effective cruise velocity
# in trapezoidal-like velocity profiles.
VEL_OVERHEAD = 1.875

GRIPPER_OPEN = 0.045
GRIPPER_CLOSE = 0.0
GRIPPER_DURATION = 0.3  # seconds (full open/close); scaled by travel fraction for partial moves

RING_COLORS = [
    (0.95, 0.95, 0.95, 1.0),
    (0.60, 0.80, 0.70, 1.0),
    (0.78, 0.88, 0.80, 1.0),
    (0.90, 0.55, 0.60, 1.0),
    (0.85, 0.72, 0.35, 1.0),
    (0.95, 0.95, 0.95, 1.0),
]

ROBOT_INIT_DOFS_RIGHT = np.deg2rad([-90, -75, 90, -90, -75, 0, -20])
ROBOT_INIT_DOFS_LEFT = np.deg2rad([90, -75, -90, -90, 75, 0, 20])

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ROBOT_DIR = os.path.join(ASSETS_DIR, "marvin_bimanual")
TOWER_DIR = os.path.join(ASSETS_DIR, "tower")
MOTION_CACHE_PATH = os.path.join(os.path.dirname(__file__), "motion_plan.pkl")


########################## config types ##########################


class ArmConfig(BaseModel):
    tip_link_name: str
    arm_dofs_idx_local: list[StrictInt]
    gripper_dofs_idx_local: list[StrictInt]


class Waypoint(BaseModel):
    """Base class for all waypoint types."""


class EndEffectorWaypoint(Waypoint):
    """Cartesian target for a specific end-effector link.

    At resolve time, the kinematic subchain from root to ``link_name`` is computed to determine which DOFs
    participate in IK.
    """

    pos: Vec3FType
    quat: Vec4FType
    link_name: str


class JointWaypoint(Waypoint):
    """Joint-space target for specific DOFs."""

    qpos: FArrayType
    dofs_idx_local: list[StrictInt]


class Connector(BaseModel):
    """How to interpolate between two consecutive waypoints in a Chunk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    method: Literal["linear", "plan_path"] = "linear"
    avoid_collision: StrictBool = False  # only valid with "plan_path"
    max_joint_vel: PositiveFloat = 1.5
    max_cartesian_vel: float = float("inf")  # inf = no limit (not PositiveFloat, which forbids inf)
    # Pre-built dense positions for plan_path connectors (set during trajectory evaluation)
    prebuilt_positions: torch.Tensor | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _validate_method(self):
        if self.avoid_collision and self.method != "plan_path":
            raise ValueError("avoid_collision=True requires method='plan_path'")
        return self


class Chunk(BaseModel):
    """A continuous motion from first waypoint to last.

    Initialize with ``entries``: a list of Waypoints, or interleaved Waypoints and Connectors. If only Waypoints are
    given, default Connectors are inserted. ``entries`` is excluded from repr/export — use ``waypoints``/``connectors``.
    """

    entries: list[Waypoint | Connector] = Field(exclude=True, repr=False)
    duration: NonNegativeFloat = 0.0  # seconds. 0 = auto from velocity limits
    default_connector: Connector = Field(default_factory=Connector, exclude=True, repr=False)

    # Derived — populated by validator
    waypoints: list[Waypoint] = Field(default_factory=list)
    connectors: list[Connector] = Field(default_factory=list)

    def __init__(self, entries, **kwargs):
        super().__init__(entries=entries, **kwargs)

    @model_validator(mode="after")
    def _validate_interleave(self):
        default = self.default_connector
        # Normalize entries: auto-insert default Connectors between consecutive Waypoints. Accepts any mix —
        # [wp, wp, conn, wp, wp] becomes [wp, default, wp, conn, wp, default, wp].
        normalized = []
        for entry in self.entries:
            if isinstance(entry, Waypoint) and normalized and isinstance(normalized[-1], Waypoint):
                normalized.append(default)
            normalized.append(entry)

        # Split into waypoints and connectors
        wps, conns = [], []
        for entry in normalized:
            if isinstance(entry, Waypoint):
                wps.append(entry)
            elif isinstance(entry, Connector):
                conns.append(entry)
            else:
                raise ValueError(f"Entry must be a Waypoint or Connector, got {type(entry).__name__}")

        if len(wps) < 1:
            raise ValueError("Chunk must have at least 1 waypoint")
        if len(conns) != len(wps) and len(conns) != len(wps) - 1:
            raise ValueError(f"Expected {len(wps)} or {len(wps) - 1} connectors, got {len(conns)}")
        self.waypoints = wps
        self.connectors = conns
        return self


class SyncPoint(BaseModel):
    """Base: start of a TimelineEntry is determined by a reference point."""


class ChunkSync(SyncPoint):
    """Start relative to a referenced Chunk.

    Specify either ``fraction`` (0.0 = chunk start, 1.0 = chunk end) or ``offset`` (seconds after chunk start).
    If both are set, they are summed.
    """

    chunk: Chunk
    fraction: ValidFloat = 0.0
    offset: ValidFloat = 0.0  # seconds relative to chunk start


class WaypointSync(SyncPoint):
    """Start when a referenced Waypoint is reached within its chunk, plus an optional time offset.

    ``offset`` is in seconds relative to the moment the waypoint is reached (positive = after, negative = before).
    """

    waypoint: Waypoint
    offset: ValidFloat = 0.0


class TimelineSync(SyncPoint):
    """Start at a fraction of the referenced Timeline's total duration.

    If both ``fraction`` and ``offset`` are set, they are summed.
    """

    timeline: "Timeline"
    fraction: ValidFloat = 1.0
    offset: ValidFloat = 0.0  # seconds relative to the fraction point


class TimelineEntry(BaseModel):
    """One step in a Timeline: play a Chunk, optionally synced."""

    chunk: Chunk
    wait_after: NonNegativeFloat = 0.0
    sync: SyncPoint | None = None


class Timeline(BaseModel):
    """Ordered list of chunks for a set of DOFs."""

    arm: ArmConfig | None = None
    dofs_idx_local: list[StrictInt]
    entries: list[TimelineEntry]


class MotionPlan(BaseModel):
    positions: tuple[tuple[float, ...], ...]  # (N, n_dofs)
    velocities: tuple[tuple[float, ...], ...]  # (N, n_dofs)
    timestep: float  # seconds per step
    joint_names: list[str]  # joint name for each column index
    config_hash: str = ""  # hash of the timeline config that produced this plan


########################## scene construction ##########################


def add_tower(scene, args):
    wood = gs.textures.ImageTexture(image_path=os.path.join(TOWER_DIR, "wood_texture.jpg"))
    vis_mode = "collision" if args.collision else "visual"
    entities = []
    entities.append(scene.add_entity(gs.morphs.Box(size=(0.8, 1.6, 0.02), pos=(0, 0, TABLE_HEIGHT - 0.01), fixed=True)))
    entities.append(
        scene.add_entity(
            morph=gs.morphs.URDF(
                file=os.path.join(TOWER_DIR, "base_pole.urdf"),
                pos=(TOWER_OFFSET_X, 0, BASE_HEIGHT / 2 + TABLE_HEIGHT),
                file_meshes_are_zup=True,
            ),
            surface=gs.surfaces.Default(diffuse_texture=wood),
            material=gs.materials.Rigid(rho=600.0),
            vis_mode=vis_mode,
        )
    )
    height = BASE_HEIGHT + TABLE_HEIGHT
    for ring_idx in RINGS_ORDER:
        entities.append(
            scene.add_entity(
                morph=gs.morphs.URDF(
                    file=os.path.join(TOWER_DIR, f"ring_{ring_idx + 1:02d}.urdf"),
                    # Slight overlap (-1e-4) ensures rings rest in contact at init
                    pos=(TOWER_OFFSET_X, 0, height + (RING_HEIGHT - 1e-4) / 2),
                    file_meshes_are_zup=True,
                ),
                surface=gs.surfaces.Default(color=RING_COLORS[ring_idx]),
                material=gs.materials.Rigid(rho=600.0),
                vis_mode=vis_mode,
            )
        )
        height += RING_HEIGHT - 1e-4
    entities.append(
        scene.add_entity(
            morph=gs.morphs.URDF(
                file=os.path.join(TOWER_DIR, "ball.urdf"),
                pos=(TOWER_OFFSET_X, 0, height + BALL_HEIGHT),
                file_meshes_are_zup=True,
            ),
            surface=gs.surfaces.Default(diffuse_texture=wood),
            material=gs.materials.Rigid(rho=600.0),
            vis_mode=vis_mode,
        )
    )
    return height, entities


def add_robot(scene, args):
    vis_mode = "collision" if args.collision else "visual"
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=os.path.join(ROBOT_DIR, "urdf/marvin_pika.urdf"),
            pos=(-0.6, 0, 1.1),
            fixed=True,
            merge_fixed_links=False,
        ),
        vis_mode=vis_mode,
        visualize_contact=True,
    )
    arm_right, arm_left, grip_right, grip_left = [], [], [], []
    for joint in robot.joints:
        if joint.type == gs.JOINT_TYPE.REVOLUTE:
            (dof_idx,) = joint.dofs_idx_local
            (arm_right if joint.name.endswith("_R") else arm_left).append(dof_idx)
        elif joint.type == gs.JOINT_TYPE.PRISMATIC:
            (dof_idx,) = joint.dofs_idx_local
            (grip_right if joint.name.endswith("_R") else grip_left).append(dof_idx)
    return (
        robot,
        ArmConfig(tip_link_name="Gripper_Tip_L", arm_dofs_idx_local=arm_left, gripper_dofs_idx_local=grip_left),
        ArmConfig(tip_link_name="Gripper_Tip_R", arm_dofs_idx_local=arm_right, gripper_dofs_idx_local=grip_right),
    )


def build_scene(args):
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="64" if args.cpu else "32", performance_mode=False)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005, substeps=1),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            constraint_timeconst=0.005,
            max_collision_pairs=1000,
            use_gjk_collision=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, 0, 1.0 + TABLE_HEIGHT), camera_lookat=(0, 0, TABLE_HEIGHT), max_FPS=52
        ),
        vis_options=gs.options.VisOptions(show_world_frame=False),
        show_viewer=args.vis,
    )
    tower_top_z, tower_entities = add_tower(scene, args)
    robot, left_arm, right_arm = add_robot(scene, args)
    if args.vis:
        scene.viewer.add_plugin(gs.vis.viewer_plugins.MouseInteractionPlugin(use_force=True, spring_const=5))

    current_target = [torch.zeros(robot.n_qs), torch.zeros(robot.n_qs)]
    all_arm_dofs = [*left_arm.arm_dofs_idx_local, *right_arm.arm_dofs_idx_local]
    all_grip_dofs = [*left_arm.gripper_dofs_idx_local, *right_arm.gripper_dofs_idx_local]
    plot_dofs = all_arm_dofs + all_grip_dofs

    # Closure: captures robot, current_target, plot_dofs for live viewer plot
    def tracking_func():
        qpos = robot.get_dofs_position(dofs_idx_local=plot_dofs)
        target_pos = current_target[0][plot_dofs]
        error = qpos - target_pos
        target_vel = current_target[1][plot_dofs]
        return {
            "tracking error": error[: len(all_arm_dofs)].tolist(),
            "target velocity": target_vel[: len(all_arm_dofs)].tolist(),
            "gripper target": target_pos[len(all_arm_dofs) :].tolist(),
        }

    if args.vis:
        scene.start_recording(
            data_func=tracking_func,
            rec_options=gs.recorders.MPLLinePlot(
                title="DOF Tracking",
                labels={
                    "tracking error": tuple(f"d{i}" for i in all_arm_dofs),
                    "target velocity": tuple(f"d{i}" for i in all_arm_dofs),
                    "gripper target": tuple(f"d{i}" for i in all_grip_dofs),
                },
                history_length=100_000,
                show_window=True,
            ),
        )

    cam = scene.add_camera(
        res=(1920, 1080),
        pos=(1.8, 0, 1.0 + TABLE_HEIGHT),
        lookat=(0, 0, TABLE_HEIGHT),
        fov=40,
        model="thinlens",
        aperture=2.8,
        focus_dist=1.3,
    )
    scene.build()

    robot.set_dofs_kp(5000.0, dofs_idx_local=all_arm_dofs)
    robot.set_dofs_kv(500.0, dofs_idx_local=all_arm_dofs)
    robot.set_dofs_kp(200.0, dofs_idx_local=all_grip_dofs)
    robot.set_dofs_kv(80.0, dofs_idx_local=all_grip_dofs)
    robot.set_dofs_force_range(-20.0, 20.0, dofs_idx_local=all_grip_dofs)
    robot.control_dofs_position(0.0)

    geom_indices = [geom.idx for entity in tower_entities for geom in entity.geoms]
    sol_params = scene.rigid_solver.get_sol_params(geoms_idx=geom_indices)
    sol_params[..., 1] = 0.7
    scene.rigid_solver.set_sol_params(sol_params, geoms_idx=geom_indices)

    return scene, robot, left_arm, right_arm, tower_top_z, current_target, cam


########################## IK helpers ##########################


def _quat(roll, yaw):
    return gs.utils.geom.xyz_to_quat(np.array([roll, 0.0, yaw], dtype=gs.np_float), rpy=True, degrees=True)


def _direction(yaw):
    rad = np.deg2rad(yaw)
    return np.array([np.cos(rad - np.pi / 2), np.sin(rad - np.pi / 2)])


########################## trajectory materialization ##########################


def _resolve_waypoint_to_qpos(waypoint, robot, timeline, prev_qpos, scene=None):
    """Resolve a single waypoint to a qpos tensor."""
    if isinstance(waypoint, JointWaypoint):
        qpos = prev_qpos.clone()
        values = torch.tensor(waypoint.qpos, dtype=gs.tc_float, device=gs.device)
        qpos[waypoint.dofs_idx_local] = values
        return qpos
    if isinstance(waypoint, EndEffectorWaypoint):
        tip = robot.get_link(waypoint.link_name)
        # Collect revolute DOFs along the kinematic chain from root to tip
        ik_dofs = []
        current = tip
        while current.parent_idx != -1:
            for joint in current.joints:
                if joint.type == gs.JOINT_TYPE.REVOLUTE:
                    ik_dofs.extend(joint.dofs_idx_local)
            current = robot.links[current.parent_idx - robot.link_start]
        ik_dofs = sorted(ik_dofs)
        pos = np.asarray(waypoint.pos, dtype=gs.np_float)
        quat = np.asarray(waypoint.quat, dtype=gs.np_float)
        qpos, error = robot.inverse_kinematics(
            link=tip,
            pos=pos,
            quat=quat,
            init_qpos=prev_qpos,
            dofs_idx_local=ik_dofs,
            pos_tol=1e-4,
            rot_tol=1e-4,
            return_error=True,
        )
        assert (error.abs() < 2e-4).all(), f"IK failed: {error}"
        if scene is not None and scene.viewer is not None:
            robot.set_dofs_position(qpos)
            scene.viewer.update(force=True)
            assert float(robot.get_AABB()[0, 2]) >= TABLE_HEIGHT, "Robot below table!"
        return qpos
    raise TypeError(f"Unknown waypoint type: {type(waypoint)}")


def evaluate_trajectory(scene, robot, timelines, cache_path=None):
    """Materialize timelines into a MotionPlan.

    If ``cache_path`` is given, the result is cached to disk. On subsequent calls with the same timeline config, the
    cached plan is returned without recomputation.

    1. Resolve waypoints to qpos (IK for PoseWaypoint, direct for JointWaypoint).
    2. Resolve connectors to segments.
    3. Compute segment durations from velocity limits.
    4. Build per-chunk positions with acceleration-limited velocity profile.
    5. Resolve timeline timing (sequential + sync overrides: ChunkSync, WaypointSync, TimelineSync).
    6. Merge timelines.
    """
    timestep = scene.sim.dt

    # Cache check
    if cache_path is not None:
        config_str = "".join(timeline.model_dump_json(exclude={"arm"}) for timeline in timelines)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            expected_joint_names = [joint.name for joint in robot.joints for _ in joint.dofs_idx_local]
            if (
                isinstance(cached, MotionPlan)
                and cached.config_hash == config_hash
                and abs(cached.timestep - timestep) < 1e-10
                and cached.joint_names == expected_joint_names
            ):
                gs.logger.info(f"Loaded cached motion plan ({config_hash})")
                return cached
            gs.logger.info("Config changed — recomputing motion plan...")
        else:
            gs.logger.info("Computing motion plan...")
    n_dofs = robot.n_qs
    device = gs.device

    # Infer global init_qpos. Scan first waypoints across all timelines. At least one timeline must start with an
    # explicit Waypoint (not a leading Connector) to define the initial state. Timelines with leading Connectors
    # inherit the global init_qpos.

    global_init_qpos = torch.zeros(n_dofs, dtype=gs.tc_float, device=device)
    has_explicit_init = False
    init_waypoints = []  # waypoints that defined the initial state — skip during chunk building
    for timeline in timelines:
        if not timeline.entries or timeline.arm is None:
            continue
        first_chunk = timeline.entries[0].chunk
        has_leading = len(first_chunk.connectors) == len(first_chunk.waypoints)
        if not has_leading:
            first_waypoint = first_chunk.waypoints[0]
            global_init_qpos = _resolve_waypoint_to_qpos(first_waypoint, robot, timeline, global_init_qpos, scene)
            init_waypoints.append(first_waypoint)
            has_explicit_init = True

    if not has_explicit_init:
        raise ValueError(
            "At least one timeline must start with an explicit Waypoint to define the initial robot state."
        )
    robot.set_dofs_position(global_init_qpos)

    # Step 1-3: resolve chunks to qpos + compute segment data

    chunk_data = {}  # chunk_key -> {ik, init_qpos, segment_steps, connectors}
    entry_chunk_key = {}  # (tl_idx, chunk_id) -> chunk_key (handles shared JointWaypoint chunks)
    waypoint_to_ik_idx = {}  # id(waypoint) -> (chunk_key, ik_solution_index); -1 = init waypoint (fraction 0.0)

    for tl_idx, timeline in enumerate(timelines):
        torch.manual_seed(0)
        prev_qpos = global_init_qpos.clone()
        robot.set_dofs_position(prev_qpos)

        for timeline_entry in timeline.entries:
            chunk = timeline_entry.chunk
            # For JointWaypoint-only chunks, cache by chunk_id (shared across timelines). For chunks with
            # PoseWaypoints, cache per-timeline (arm-specific IK).
            all_joint = all(isinstance(wp, JointWaypoint) for wp in chunk.waypoints)
            chunk_key = (tl_idx, id(chunk)) if not all_joint else (-1, id(chunk))
            entry_chunk_key[(tl_idx, id(chunk))] = chunk_key
            if chunk_key in chunk_data:
                cd = chunk_data[chunk_key]
                prev_qpos = cd["ik"][-1] if cd["ik"] else cd["init_qpos"]
                continue

            # Resolve waypoints to qpos, expanding plan_path connectors. A leading connector
            # (len(connectors) == len(waypoints)) means the first connector connects from init_qpos to the first wp.
            has_leading_connector = len(chunk.connectors) == len(chunk.waypoints)
            ik_solutions = []
            effective_connectors = []
            for waypoint_idx, waypoint in enumerate(chunk.waypoints):
                # Skip waypoints that defined the global init state (already at that position)
                if waypoint_idx == 0 and any(waypoint is wp for wp in init_waypoints):
                    waypoint_to_ik_idx[id(waypoint)] = (chunk_key, -1)
                    continue

                if has_leading_connector:
                    connector = chunk.connectors[waypoint_idx]
                elif waypoint_idx > 0:
                    connector = chunk.connectors[waypoint_idx - 1]
                else:
                    connector = None

                if connector is not None and connector.method == "plan_path":
                    target_qpos = _resolve_waypoint_to_qpos(waypoint, robot, timeline, prev_qpos, scene)
                    robot.set_dofs_position(prev_qpos)
                    path = robot.plan_path(
                        qpos_goal=target_qpos,
                        ignore_collision=not connector.avoid_collision,
                        smooth_path=True,
                        num_waypoints=6,
                    )
                    # Treat entire plan_path as ONE segment: interpolate through all intermediate waypoints. Only the
                    # final qpos goes into ik_solutions.
                    sub_waypoints = [prev_qpos.clone()] + [path[i].clone() for i in range(1, len(path))]

                    # Build dense linear interpolation through all path waypoints
                    sub_segments = []
                    sub_prev = sub_waypoints[0]
                    for sub_wp in sub_waypoints[1:]:
                        joint_dist = float((sub_wp - sub_prev).abs().max())
                        n_sub_steps = max(int(joint_dist * VEL_OVERHEAD / (connector.max_joint_vel * timestep)), 2)
                        alpha = torch.linspace(0, 1, n_sub_steps, device=gs.device)
                        sub_segments.append(sub_prev * (1 - alpha[:, None]) + sub_wp * alpha[:, None])
                        sub_prev = sub_wp
                    # Concatenate (skip duplicate boundary points)
                    dense_parts = [sub_segments[0]]
                    for seg in sub_segments[1:]:
                        dense_parts.append(seg[1:])
                    dense_positions = torch.cat(dense_parts)

                    # Append as a single ik_solution with a marker connector that tells _build_trajectory to use
                    # these pre-built positions directly.
                    ik_solutions.append(path[-1].clone())
                    effective_connectors.append(
                        Connector(max_joint_vel=connector.max_joint_vel, prebuilt_positions=dense_positions)
                    )
                    waypoint_to_ik_idx[id(waypoint)] = (chunk_key, len(ik_solutions) - 1)
                    prev_qpos = path[-1]
                else:
                    qpos = _resolve_waypoint_to_qpos(waypoint, robot, timeline, prev_qpos, scene)
                    ik_solutions.append(qpos.clone())
                    effective_connectors.append(connector if connector is not None else chunk.default_connector)
                    waypoint_to_ik_idx[id(waypoint)] = (chunk_key, len(ik_solutions) - 1)
                    prev_qpos = qpos

            # Determine init_qpos for this chunk (the qpos BEFORE it starts)
            init_qpos_for_chunk = None
            for other_entry in timeline.entries:
                other_ck = entry_chunk_key.get((tl_idx, id(other_entry.chunk)))
                if other_ck == chunk_key:
                    break
                if other_ck is not None and other_ck in chunk_data:
                    other_cd = chunk_data[other_ck]
                    init_qpos_for_chunk = other_cd["ik"][-1] if other_cd["ik"] else other_cd["init_qpos"]
            if init_qpos_for_chunk is None:
                init_qpos_for_chunk = global_init_qpos.clone()

            # Compute per-segment step counts
            segment_steps = []
            seg_prev = init_qpos_for_chunk
            for i, qpos in enumerate(ik_solutions):
                connector = effective_connectors[i] if i < len(effective_connectors) else Connector()
                max_jv = connector.max_joint_vel
                max_cv = connector.max_cartesian_vel

                joint_dist = float((qpos - seg_prev).abs().max())

                # Cartesian distance (only when max_cv is finite)
                cart_dist = 0.0
                if max_cv < float("inf"):
                    tip = robot.get_link(timeline.arm.tip_link_name)
                    robot.set_dofs_position(seg_prev)
                    cart_from = tensor_to_array(tip.get_pos()).ravel()
                    robot.set_dofs_position(qpos)
                    cart_to = tensor_to_array(tip.get_pos()).ravel()
                    cart_dist = float(np.linalg.norm(cart_to - cart_from))

                total_duration = max(cart_dist / max_cv, joint_dist / (max_jv / VEL_OVERHEAD))
                segment_steps.append(max(int(total_duration / timestep), 2))
                seg_prev = qpos

            # Handle explicit chunk duration: uniform distribution, never violate velocity limits
            if chunk.duration > 0:
                total_target = max(int(chunk.duration / timestep), len(ik_solutions) * 2)
                per_segment = max(total_target // max(len(ik_solutions), 1), 2)
                segment_steps = [max(per_segment, s) for s in segment_steps]

            chunk_data[chunk_key] = {
                "chunk": chunk,
                "ik": ik_solutions,
                "init_qpos": init_qpos_for_chunk,
                "segment_steps": segment_steps,
                "connectors": effective_connectors,
            }

    # Step 4: build per-chunk positions

    chunk_positions = {}
    chunk_total_steps = {}
    chunk_waypoint_fractions = {}  # chunk_key -> list[float] (fraction of duration at which each ik_solution is reached)

    for chunk_key, data in chunk_data.items():
        positions, wp_fractions = _build_trajectory(
            data["ik"],
            data["init_qpos"],
            data["segment_steps"],
            data["connectors"],
            timestep,
        )
        chunk_positions[chunk_key] = [positions]
        chunk_total_steps[chunk_key] = len(positions)
        chunk_waypoint_fractions[chunk_key] = wp_fractions

    # Build waypoint -> (chunk_key, fraction_of_chunk_duration)
    waypoint_time_fraction = {}  # id(waypoint) -> (chunk_key, fraction)
    for wp_id, (ck, ik_idx) in waypoint_to_ik_idx.items():
        waypoint_time_fraction[wp_id] = (ck, 0.0 if ik_idx == -1 else chunk_waypoint_fractions[ck][ik_idx])

    # Step 5: resolve timing and build per-timeline positions

    chunk_duration_s = {ck: steps * timestep for ck, steps in chunk_total_steps.items()}

    # Map chunk id -> any chunk_key that has it (for sync_chunk lookup across timelines)
    chunk_id_to_key = {}
    for ck in chunk_data:
        chunk_id_to_key.setdefault(ck[1], ck)

    chunk_start_time = {}  # chunk_key -> start time

    # Multi-pass: resolve timing across timelines until stable (handles cross-refs)
    for _pass in range(3):
        for tl_idx, timeline in enumerate(timelines):
            current_time = 0.0
            for timeline_entry in timeline.entries:
                ck = entry_chunk_key[(tl_idx, id(timeline_entry.chunk))]
                sync = timeline_entry.sync
                if isinstance(sync, ChunkSync):
                    sync_any_key = chunk_id_to_key.get(id(sync.chunk))
                    if sync_any_key is not None and sync_any_key in chunk_start_time:
                        sync_dur = chunk_duration_s.get(sync_any_key, 0.0)
                        sync_time = chunk_start_time[sync_any_key] + sync.fraction * sync_dur + sync.offset
                        current_time = max(current_time, sync_time)
                elif isinstance(sync, WaypointSync):
                    wp_info = waypoint_time_fraction.get(id(sync.waypoint))
                    if wp_info is not None:
                        ref_ck, wp_frac = wp_info
                        if ref_ck in chunk_start_time:
                            ref_dur = chunk_duration_s.get(ref_ck, 0.0)
                            sync_time = chunk_start_time[ref_ck] + wp_frac * ref_dur + sync.offset
                            current_time = max(current_time, sync_time)
                elif isinstance(sync, TimelineSync):
                    # Total duration of the referenced timeline
                    ref_tl_idx = next((i for i, tl in enumerate(timelines) if tl is sync.timeline), None)
                    if ref_tl_idx is not None:
                        ref_entries = timelines[ref_tl_idx].entries
                        if ref_entries:
                            last_ck = entry_chunk_key.get((ref_tl_idx, id(ref_entries[-1].chunk)))
                            if last_ck is not None and last_ck in chunk_start_time:
                                tl_end = chunk_start_time[last_ck] + chunk_duration_s.get(last_ck, 0.0)
                                sync_time = tl_end * sync.fraction + sync.offset
                                current_time = max(current_time, sync_time)
                chunk_start_time[ck] = current_time
                current_time = chunk_start_time[ck] + chunk_duration_s.get(ck, 0.0) + timeline_entry.wait_after

    # Build dense position arrays per timeline
    timeline_positions = []
    for tl_idx, timeline in enumerate(timelines):
        parts = []
        current_time = 0.0
        for timeline_entry in timeline.entries:
            ck = entry_chunk_key[(tl_idx, id(timeline_entry.chunk))]
            start_time = chunk_start_time[ck]

            # Gap: hold at previous position (or first waypoint qpos before any chunk)
            gap_steps = max(0, int((start_time - current_time) / timestep))
            if gap_steps > 0:
                if parts:
                    hold_pos = parts[-1][-1]
                else:
                    hold_pos = global_init_qpos
                parts.append(hold_pos.unsqueeze(0).expand(gap_steps, -1).clone())

            # Chunk positions
            for chunk_pos in chunk_positions[ck]:
                parts.append(chunk_pos)

            # Wait after
            wait_steps = int(timeline_entry.wait_after / timestep)
            if wait_steps > 0:
                parts.append(parts[-1][-1:].expand(wait_steps, -1).clone())

            current_time = start_time + chunk_duration_s.get(ck, 0.0) + timeline_entry.wait_after

        timeline_positions.append(torch.cat(parts) if parts else torch.zeros(1, n_dofs, device=device))

    # Step 6: merge timelines

    max_len = max(len(pos) for pos in timeline_positions)
    for i in range(len(timeline_positions)):
        pos = timeline_positions[i]
        if len(pos) < max_len:
            pad_n = max_len - len(pos)
            timeline_positions[i] = torch.cat([pos, pos[-1:].expand(pad_n, -1)])

    merged = timeline_positions[0].clone()
    for i, timeline in enumerate(timelines):
        if i == 0:
            continue
        merged[:, timeline.dofs_idx_local] = timeline_positions[i][:, timeline.dofs_idx_local]

    velocities = merged.diff(dim=0) / timestep
    velocities = torch.cat([velocities, torch.zeros(1, n_dofs, device=device)])

    joint_names = [joint.name for joint in robot.joints for _ in joint.dofs_idx_local]

    gs.logger.info(f"Total: {len(merged)} steps ({len(merged) * timestep:.1f}s)")
    plan = MotionPlan(
        positions=tuple(tuple(row) for row in merged.tolist()),
        velocities=tuple(tuple(row) for row in velocities.tolist()),
        timestep=timestep,
        joint_names=joint_names,
        config_hash=config_hash if cache_path is not None else "",
    )

    # Cache save
    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(plan, f)
        gs.logger.info(f"Saved motion plan ({config_hash})")

    return plan


def _build_trajectory(ik_solutions, init_qpos, segment_steps, connectors=None, timestep=0.02):
    """Linear interpolation with acceleration-limited velocity profile.

    Each segment is linearly interpolated (constant velocity plateau). The concatenated trajectory is then resampled
    with a velocity profile that respects a global ``MAX_ACCEL`` limit via forward-backward clamping. Positions stay
    exactly on the original piecewise-linear path — only the *timing* (velocity at each point) changes.

    Returns ``(positions, waypoint_fractions)`` where ``waypoint_fractions[i]`` is the fraction [0, 1] of total
    duration at which ``ik_solutions[i]`` is reached.
    """
    device = init_qpos.device
    n_segments = len(ik_solutions)

    if n_segments == 0:
        return init_qpos.unsqueeze(0), []

    # Step 1: per-segment linear interpolation (no ramps)
    segments = []
    prev_qpos = init_qpos
    for seg_idx, (qpos, n_steps) in enumerate(zip(ik_solutions, segment_steps)):
        conn = connectors[seg_idx] if connectors and seg_idx < len(connectors) else None
        if conn is not None and conn.prebuilt_positions is not None:
            segments.append(conn.prebuilt_positions)
        else:
            alpha = torch.linspace(0, 1, n_steps, device=device)
            segments.append(prev_qpos * (1 - alpha[:, None]) + qpos * alpha[:, None])
        prev_qpos = qpos

    # Step 2: concatenate (skip duplicate boundary points)
    parts = [segments[0]]
    # Track where each segment boundary falls in the concatenated array
    boundary_indices = []
    offset = len(segments[0]) - 1
    for seg in segments[1:]:
        boundary_indices.append(offset)
        offset += len(seg) - 1
        parts.append(seg[1:])
    positions = torch.cat(parts)
    n_original = len(positions)

    # Step 3: arc-length parameterization
    step_displacements = positions[1:] - positions[:-1]  # (N-1, n_dofs)
    deltas = step_displacements.abs().max(dim=1).values  # (N-1,) max-norm per step

    s_total = deltas.sum().item()
    if s_total < 1e-10:
        n = len(positions)
        wp_fracs = [bnd / max(n - 1, 1) for bnd in boundary_indices] + [1.0]
        return positions, wp_fracs

    # Natural (unclamped) arc-length velocity at each step: ds/dt.
    # Constant within each linear segment.
    v_natural = deltas / timestep  # (N-1,)

    # Step 4: per-DOF direction-change velocity limits. At segment boundaries the path direction changes. If the
    # arc-length velocity stays at v, the per-DOF velocity on a reversing DOF jumps by |dir_new - dir_old| * v,
    # giving acceleration |dir_change| * v / dt. Cap v so that this stays within MAX_ACCEL.
    v_max = v_natural.clone()
    max_dv = MAX_ACCEL * timestep

    for bnd in boundary_indices:
        if bnd < 1 or bnd >= len(v_max):
            continue
        # Direction vectors of the two segments meeting at this boundary
        d_before = step_displacements[bnd - 1]
        d_after = step_displacements[bnd]
        ds_before = deltas[bnd - 1].clamp(min=1e-10)
        ds_after = deltas[bnd].clamp(min=1e-10)
        dir_before = d_before / ds_before
        dir_after = d_after / ds_after

        # Max per-DOF direction change
        dir_change = (dir_after - dir_before).abs().max().item()
        if dir_change > 1e-6:
            # v_arc * dir_change ≤ MAX_ACCEL * dt  ->  v_arc ≤ max_dv / dir_change
            v_limit = max_dv / dir_change
            v_max[bnd - 1] = min(v_max[bnd - 1].item(), v_limit)
            v_max[bnd] = min(v_max[bnd].item(), v_limit)

    # Step 5: forward-backward acceleration clamping. Clamp velocity so |dv/dt| <= MAX_ACCEL everywhere, v=0 at ends.
    v_limited = v_max.clone()

    # Forward pass: can't accelerate faster than MAX_ACCEL from v=0
    v_fwd = 0.0
    for i in range(len(v_limited)):
        v_fwd = min(v_fwd + max_dv, v_limited[i].item())
        v_limited[i] = v_fwd

    # Backward pass: must decelerate to v=0 at the end
    v_bwd = 0.0
    for i in range(len(v_limited) - 1, -1, -1):
        v_bwd = min(v_bwd + max_dv, v_limited[i].item())
        v_limited[i] = v_bwd

    # Step 6: resample at uniform timestep. Integrate limited velocity to get arc-length vs time. Each original step i
    # now takes dt_i = delta_s[i] / v_limited[i] (where v_limited > 0; if v_limited=0, the step is instantaneous).
    v_limited_clamped = v_limited.clamp(min=1e-10)
    step_durations = deltas / v_limited_clamped  # (N-1,) time per original step
    cumulative_time = torch.zeros(n_original, device=device)
    cumulative_time[1:] = torch.cumsum(step_durations, dim=0)
    total_time = cumulative_time[-1].item()

    n_resampled = max(round(total_time / timestep), 2)
    t_uniform = torch.linspace(0, total_time, n_resampled, device=device)

    # For each uniform time, find the arc-length then interpolate position (searchsorted gives insertion index).
    idx = torch.searchsorted(cumulative_time, t_uniform).clamp(1, n_original - 1)
    t_lo = cumulative_time[idx - 1]
    t_hi = cumulative_time[idx]
    frac = ((t_uniform - t_lo) / (t_hi - t_lo).clamp(min=1e-10)).clamp(0, 1)
    resampled = positions[idx - 1] * (1 - frac[:, None]) + positions[idx] * frac[:, None]

    # Compute per-waypoint time fractions from cumulative_time at segment boundaries
    wp_fracs = [cumulative_time[bnd].item() / total_time for bnd in boundary_indices] + [1.0]
    return resampled, wp_fracs


########################## waypoint generators ##########################
# Each returns a list of Waypoints (optionally interleaved with a Connector).


def wp_ball_grab(link_name, ball_z):
    direction = _direction(135.0)
    center = np.array([TOWER_OFFSET_X, 0.0])
    approach_xy = center - APPROACH_DIST * direction
    grasp_xy = center + GRASP_DEPTH * direction
    grab_z = ball_z
    lift_z = TABLE_HEIGHT + BASE_HEIGHT + POLE_HEIGHT + LIFT_CLEARANCE
    mid_z = (grab_z + lift_z) / 2
    orientation = _quat(115.0, 135.0)
    return [
        Connector(max_joint_vel=1.3),
        EndEffectorWaypoint(pos=[*approach_xy, grab_z], quat=orientation, link_name=link_name),
        EndEffectorWaypoint(pos=[*grasp_xy, grab_z], quat=orientation, link_name=link_name),
        Connector(max_joint_vel=1.0),
        EndEffectorWaypoint(pos=[*grasp_xy, mid_z], quat=orientation, link_name=link_name),
        Connector(max_joint_vel=0.7),
        EndEffectorWaypoint(pos=[*grasp_xy, lift_z], quat=orientation, link_name=link_name),
    ]


def wp_ball_place(link_name):
    table_pos = np.array([0.0, BALL_TABLE_OFFSET_Y, TABLE_HEIGHT + BALL_HEIGHT])
    direction = _direction(135.0)
    place_xy = table_pos[:2] + GRASP_DEPTH * direction
    lift_z = TABLE_HEIGHT + BASE_HEIGHT + POLE_HEIGHT + LIFT_CLEARANCE
    orientation = _quat(115.0, 135.0)
    return [
        Connector(max_joint_vel=1.0),
        EndEffectorWaypoint(pos=[TOWER_OFFSET_X, -0.05, lift_z], quat=orientation, link_name=link_name),
        # Remaining transitions use chunk default (1.5)
        EndEffectorWaypoint(pos=[*place_xy, table_pos[2] + 0.01], quat=orientation, link_name=link_name),
        EndEffectorWaypoint(pos=[*place_xy, table_pos[2]], quat=orientation, link_name=link_name),
        EndEffectorWaypoint(pos=[*place_xy, table_pos[2] + 0.02], quat=orientation, link_name=link_name),
    ]


def wp_grab(link_name, roll, yaw_approach, grasp_z):
    direction = _direction(yaw_approach)
    center = np.array([TOWER_OFFSET_X, 0.0])
    approach_xy = center - APPROACH_DIST * direction
    grasp_xy = center + GRASP_DEPTH * direction
    retreat_xy = center - (APPROACH_DIST + 0.03) * direction
    lift_z = TABLE_HEIGHT + BASE_HEIGHT + POLE_HEIGHT + LIFT_CLEARANCE
    mid_z = (grasp_z + lift_z) / 2
    orientation = _quat(roll, yaw_approach)
    return [
        EndEffectorWaypoint(pos=[*approach_xy, grasp_z], quat=orientation, link_name=link_name),
        Connector(max_joint_vel=1.3),
        EndEffectorWaypoint(pos=[*grasp_xy, grasp_z], quat=orientation, link_name=link_name),
        # Lift and retreat: default velocity (1.5) except slower approach to grasp/retreat
        EndEffectorWaypoint(pos=[*grasp_xy, mid_z], quat=orientation, link_name=link_name),
        EndEffectorWaypoint(pos=[*grasp_xy, lift_z], quat=orientation, link_name=link_name),
        Connector(max_joint_vel=1.3),
        EndEffectorWaypoint(pos=[*retreat_xy, lift_z], quat=orientation, link_name=link_name),
    ]


def wp_insert(link_name, roll, yaw_approach, use_shift=False):
    """Insert ring/ball onto pole.

    The align-to-lower transition has no explicit Connector — it uses the chunk's default_connector for a careful
    insertion speed (e.g. 0.45 for rings, 0.38 for ball).
    """
    direction = _direction(yaw_approach)
    insert_xy = np.array([TOWER_OFFSET_X, 0.0]) + GRASP_DEPTH * direction
    pole_z = TABLE_HEIGHT + BASE_HEIGHT + POLE_HEIGHT
    orientation = _quat(roll, yaw_approach)
    center = np.array([TOWER_OFFSET_X, 0.0])
    retreat_xy = center - (APPROACH_DIST + 0.03) * direction
    wps: list[Waypoint | Connector] = []
    if use_shift:
        # Lateral shift to avoid collision with the other arm
        side = np.sign(np.cos(np.deg2rad(yaw_approach)))
        wps += [
            Connector(max_joint_vel=2.5),
            EndEffectorWaypoint(
                pos=[TOWER_OFFSET_X, side * 0.04, pole_z + RING_HEIGHT / 2 + 0.015],
                quat=orientation,
                link_name=link_name,
            ),
            Connector(max_joint_vel=2.0),
        ]
    else:
        wps += [
            Connector(max_joint_vel=1.3),
        ]
    wps += [
        EndEffectorWaypoint(pos=[*insert_xy, pole_z + RING_HEIGHT / 2 + 0.015], quat=orientation, link_name=link_name),
        # align -> lower: gap filled by chunk default_connector (careful insertion speed)
        EndEffectorWaypoint(pos=[*insert_xy, pole_z + RING_HEIGHT / 2 - 0.008], quat=orientation, link_name=link_name),
        Connector(max_joint_vel=1.3),
        EndEffectorWaypoint(pos=[*insert_xy, pole_z + 0.045], quat=orientation, link_name=link_name),
        Connector(max_joint_vel=1.3),
        EndEffectorWaypoint(pos=[*retreat_xy, pole_z + 0.045], quat=orientation, link_name=link_name),
    ]
    return wps


def wp_ball_pickup(link_name):
    """Pick ball from table. Chunk default_connector should be 1.0 rad/s."""
    table_pos = np.array([0.0, BALL_TABLE_OFFSET_Y, TABLE_HEIGHT + BALL_HEIGHT])
    direction = _direction(135.0)
    pickup_xy = table_pos[:2] + GRASP_DEPTH * direction
    orientation = _quat(115.0, 135.0)
    return [
        Connector(max_joint_vel=2.0),  # fast approach above ball
        EndEffectorWaypoint(pos=[*pickup_xy, table_pos[2] + 0.02], quat=orientation, link_name=link_name),
        # Slow descent and lift use chunk default (1.0)
        EndEffectorWaypoint(pos=[*pickup_xy, table_pos[2] - 0.007], quat=orientation, link_name=link_name),
        EndEffectorWaypoint(pos=[*pickup_xy, table_pos[2] + 0.02], quat=orientation, link_name=link_name),
    ]


def wp_push(link_name, yaw_approach, push_distance):
    """Push ring sideways on pole. Chunk default_connector should be 2.0 rad/s."""
    yaw_rad = np.deg2rad(yaw_approach)
    side = np.sign(np.cos(yaw_rad))
    push_z = TABLE_HEIGHT + BASE_HEIGHT + RING_HEIGHT
    start_z = TABLE_HEIGHT + BASE_HEIGHT + POLE_HEIGHT + 0.03
    orientation = _quat(135.0, 35.0)
    return [
        # Fast approach and descent use chunk default (2.0)
        EndEffectorWaypoint(pos=[TOWER_OFFSET_X + 0.01, side * 0.06, start_z], quat=orientation, link_name=link_name),
        EndEffectorWaypoint(pos=[TOWER_OFFSET_X + 0.01, side * 0.06, push_z], quat=orientation, link_name=link_name),
        Connector(max_joint_vel=0.25),  # slow push
        EndEffectorWaypoint(
            pos=[TOWER_OFFSET_X + 0.01, side * 0.06 + push_distance, push_z], quat=orientation, link_name=link_name
        ),
    ]


########################## plan motion ##########################


def _grip_chunk(aperture, dofs, duration=GRIPPER_DURATION):
    """Create a single-waypoint chunk that sets all gripper DOFs to ``aperture``."""
    return Chunk(
        entries=[JointWaypoint(qpos=tuple(aperture for _ in dofs), dofs_idx_local=dofs)],
        duration=duration,
    )


def build_timelines(tower_top_z, left_arm, right_arm):
    """Build the timeline config (pure data, no IK or trajectory computation)."""

    ball_pos = np.array([TOWER_OFFSET_X, 0.0, tower_top_z + BALL_HEIGHT])

    right_ready_qpos = tuple(float(x) for x in ROBOT_INIT_DOFS_RIGHT)
    left_ready_qpos = tuple(float(x) for x in ROBOT_INIT_DOFS_LEFT)
    right_zero_qpos = tuple(0.0 for _ in right_arm.arm_dofs_idx_local)
    left_zero_qpos = tuple(0.0 for _ in left_arm.arm_dofs_idx_local)

    # Define chunks

    init_chunk_right = Chunk(
        entries=[
            # JointWaypoint(qpos=right_zero_qpos, dofs_idx_local=right_arm.arm_dofs_idx_local),
            # Connector(method="plan_path", avoid_collision=True),
            JointWaypoint(qpos=right_ready_qpos, dofs_idx_local=right_arm.arm_dofs_idx_local),
        ],
    )
    init_chunk_left = Chunk(
        entries=[
            # JointWaypoint(qpos=left_zero_qpos, dofs_idx_local=left_arm.arm_dofs_idx_local),
            # Connector(method="plan_path", avoid_collision=True),
            JointWaypoint(qpos=left_ready_qpos, dofs_idx_local=left_arm.arm_dofs_idx_local),
        ],
    )

    right_roll, right_yaw = 96.0, 135.0
    left_roll, left_yaw = 96.0, 35.0

    # Insertion connectors: careful descent speed for align->lower gap in wp_insert
    ring_insert_conn = Connector(max_joint_vel=0.45)
    ball_insert_conn = Connector(max_joint_vel=0.38)

    # Right arm chunks (default connector = 1.5 unless overridden)
    r_ball_grab = Chunk(entries=wp_ball_grab(right_arm.tip_link_name, ball_pos[2]))
    r_ball_place = Chunk(entries=wp_ball_place(right_arm.tip_link_name))
    r_ring_grab = Chunk(
        # Grasp at ring center height; +0.009 compensates for contact settling
        entries=wp_grab(right_arm.tip_link_name, right_roll, right_yaw, tower_top_z - RING_HEIGHT),
    )
    r_ring_ins = Chunk(
        entries=wp_insert(right_arm.tip_link_name, right_roll, right_yaw), default_connector=ring_insert_conn
    )
    r_ball_pick = Chunk(entries=wp_ball_pickup(right_arm.tip_link_name), default_connector=Connector(max_joint_vel=1.0))
    r_ball_ins = Chunk(
        entries=wp_insert(right_arm.tip_link_name, 110, right_yaw, use_shift=True),
        default_connector=ball_insert_conn,
    )
    r_return = Chunk(
        entries=[
            Connector(max_joint_vel=2.2),
            JointWaypoint(qpos=right_ready_qpos, dofs_idx_local=right_arm.arm_dofs_idx_local),
        ],
    )

    # Left arm chunks
    l_ring_grab = Chunk(
        # Grasp at ring center height; +0.002 compensates for contact settling
        entries=wp_grab(left_arm.tip_link_name, left_roll, left_yaw, tower_top_z),
    )
    l_ring_ins = Chunk(
        entries=wp_insert(left_arm.tip_link_name, left_roll, left_yaw), default_connector=ring_insert_conn
    )
    l_push = Chunk(
        entries=wp_push(left_arm.tip_link_name, left_yaw, push_distance=-0.06),
        default_connector=Connector(max_joint_vel=2.0),
    )
    l_return = Chunk(
        entries=[
            Connector(max_joint_vel=2.4),
            JointWaypoint(qpos=left_ready_qpos, dofs_idx_local=left_arm.arm_dofs_idx_local),
        ],
    )

    # Arm timelines

    right_timeline = Timeline(
        arm=right_arm,
        dofs_idx_local=right_arm.arm_dofs_idx_local,
        entries=[
            TimelineEntry(chunk=init_chunk_right),
            TimelineEntry(chunk=r_ball_grab),
            TimelineEntry(chunk=r_ball_place),
            TimelineEntry(chunk=r_ring_grab, sync=ChunkSync(chunk=l_ring_grab, fraction=0.725)),
            TimelineEntry(chunk=r_ring_ins, sync=ChunkSync(chunk=l_ring_ins, fraction=0.815)),
            TimelineEntry(chunk=r_ball_pick),
            TimelineEntry(chunk=r_ball_ins),
            TimelineEntry(chunk=r_return),
        ],
    )

    left_timeline = Timeline(
        arm=left_arm,
        dofs_idx_local=left_arm.arm_dofs_idx_local,
        entries=[
            TimelineEntry(chunk=init_chunk_left),
            TimelineEntry(chunk=l_ring_grab, sync=ChunkSync(chunk=r_ball_place, fraction=-0.2)),
            TimelineEntry(chunk=l_ring_ins, sync=ChunkSync(chunk=r_ring_grab, fraction=0.85)),
            TimelineEntry(chunk=l_push, sync=ChunkSync(chunk=r_ball_ins, fraction=0.3)),
            TimelineEntry(chunk=l_return),
        ],
    )

    # Gripper timelines. WaypointSync ties gripper actuation to the exact arm waypoint (grasp/release pose) with a
    # negative offset so the gripper starts closing before the arm arrives (50% of close duration as approximation).
    grip_close_offset = -0.5 * GRIPPER_DURATION
    r_grip = right_arm.gripper_dofs_idx_local
    l_grip = left_arm.gripper_dofs_idx_local

    right_gripper_timeline = Timeline(
        arm=None,
        dofs_idx_local=r_grip,
        entries=[
            TimelineEntry(chunk=_grip_chunk(GRIPPER_OPEN, r_grip)),
            TimelineEntry(  # close after ball grasp
                chunk=_grip_chunk(GRIPPER_CLOSE, r_grip),
                sync=WaypointSync(waypoint=r_ball_grab.waypoints[1], offset=grip_close_offset),
            ),
            TimelineEntry(  # open to release ball on table
                chunk=_grip_chunk(GRIPPER_OPEN, r_grip),
                sync=WaypointSync(waypoint=r_ball_place.waypoints[2]),
            ),
            TimelineEntry(  # close to grasp ring
                chunk=_grip_chunk(GRIPPER_CLOSE, r_grip),
                sync=WaypointSync(waypoint=r_ring_grab.waypoints[1], offset=grip_close_offset),
            ),
            TimelineEntry(  # open to release ring on pole
                chunk=_grip_chunk(GRIPPER_OPEN, r_grip),
                sync=WaypointSync(waypoint=r_ring_ins.waypoints[1], offset=grip_close_offset),
            ),
            TimelineEntry(  # close to pick ball from table
                chunk=_grip_chunk(GRIPPER_CLOSE, r_grip),
                sync=WaypointSync(waypoint=r_ball_pick.waypoints[1], offset=grip_close_offset),
            ),
            TimelineEntry(  # open to release ball on pole
                chunk=_grip_chunk(GRIPPER_OPEN, r_grip),
                sync=WaypointSync(waypoint=r_ball_ins.waypoints[2]),
            ),
        ],
    )

    left_gripper_timeline = Timeline(
        arm=None,
        dofs_idx_local=l_grip,
        entries=[
            TimelineEntry(chunk=_grip_chunk(GRIPPER_OPEN, l_grip)),
            TimelineEntry(  # close to grasp ring
                chunk=_grip_chunk(GRIPPER_CLOSE, l_grip),
                sync=WaypointSync(waypoint=l_ring_grab.waypoints[1], offset=grip_close_offset),
            ),
            TimelineEntry(  # open to release ring on pole
                chunk=_grip_chunk(GRIPPER_OPEN, l_grip),
                sync=WaypointSync(waypoint=l_ring_ins.waypoints[1], offset=grip_close_offset),
            ),
            TimelineEntry(  # close for push contact
                chunk=_grip_chunk(GRIPPER_CLOSE, l_grip),
                sync=WaypointSync(waypoint=l_ring_ins.waypoints[-1]),
            ),
            TimelineEntry(  # open after return
                chunk=_grip_chunk(GRIPPER_OPEN, l_grip),
                sync=ChunkSync(chunk=l_return, fraction=0.42),
            ),
        ],
    )

    return [right_timeline, left_timeline, right_gripper_timeline, left_gripper_timeline]


########################## main ##########################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis")
    parser.add_argument("-c", "--collision", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--gpu", action="store_false", dest="cpu")
    parser.add_argument("--time-dilation", type=float, default=1.0)
    args = parser.parse_args()

    scene, robot, left_arm, right_arm, tower_top_z, current_target, cam = build_scene(args)

    timelines = build_timelines(tower_top_z, left_arm, right_arm)
    plan = evaluate_trajectory(scene, robot, timelines, cache_path=MOTION_CACHE_PATH)

    cam.start_recording()
    video_path = os.path.join(os.path.dirname(__file__), "stacking_tower.mp4")

    arm_dofs = [*left_arm.arm_dofs_idx_local, *right_arm.arm_dofs_idx_local]
    # Gripper geom indices for cross-gripper collision detection
    left_gripper_geoms = {geom.idx for geom in robot.get_link(left_arm.tip_link_name).geoms}
    right_gripper_geoms = {geom.idx for geom in robot.get_link(right_arm.tip_link_name).geoms}

    # Convert plan data from tuples to tensors for the simulation loop
    positions = torch.tensor(plan.positions, dtype=gs.tc_float, device=gs.device)
    velocities = torch.tensor(plan.velocities, dtype=gs.tc_float, device=gs.device)
    n_steps = len(positions)

    time_dilation = args.time_dilation
    tracking_failure = None
    robot.set_dofs_position(positions[0])
    try:
        for step_idx in range(n_steps):
            target_pos = positions[step_idx]
            target_vel = velocities[step_idx] / time_dilation
            current_target[0] = target_pos
            current_target[1] = target_vel
            if time_dilation <= 1.0:
                robot.control_dofs_position_velocity(target_pos, target_vel)
                scene.step()
                cam.render()
            else:
                prev_pos = positions[step_idx - 1] if step_idx > 0 else target_pos
                for sub_step in range(int(time_dilation)):
                    alpha = (sub_step + 1) / int(time_dilation)
                    interp_pos = (1 - alpha) * prev_pos + alpha * target_pos
                    current_target[0] = interp_pos
                    robot.control_dofs_position_velocity(interp_pos, target_vel)
                    scene.step()
                    cam.render()
            # Check tracking error
            actual_qpos = robot.get_dofs_position(dofs_idx_local=arm_dofs)
            tracking_error = (actual_qpos - target_pos[arm_dofs]).abs()
            if (tracking_error > 0.018).any():  # ~1 degree tolerance
                worst_dof = arm_dofs[tracking_error.argmax().item()]
                side = "L" if worst_dof in left_arm.arm_dofs_idx_local else "R"
                tracking_failure = (
                    f"Tracking error {tracking_error.max().item():.4f} rad on dof_{worst_dof} ({side})"
                    f"(vel={target_vel[worst_dof].abs().item():.4f}) at step {step_idx}/{n_steps}"
                )
                raise RuntimeError(tracking_failure)
            # Check cross-gripper collision
            contacts = robot.get_contacts(with_entity=robot)
            if len(contacts["geom_a"]) > 0:
                geom_a = contacts["geom_a"]
                geom_b = contacts["geom_b"]
                for contact_idx in range(len(geom_a)):
                    ga, gb = int(geom_a[contact_idx]), int(geom_b[contact_idx])
                    left_involved = ga in left_gripper_geoms or gb in left_gripper_geoms
                    right_involved = ga in right_gripper_geoms or gb in right_gripper_geoms
                    if left_involved and right_involved:
                        raise RuntimeError(f"Gripper collision at step {step_idx}/{n_steps}")
    finally:
        cam.stop_recording(save_to_filename=video_path, fps=int(1 / scene.sim.dt))
        if tracking_failure:
            gs.logger.error(f"TRACKING FAILURE (video saved to {video_path}): {tracking_failure}")

    robot.control_dofs_position(positions[-1])
    for _ in range(100):
        scene.step()

    if args.vis:
        for _ in range(10000):
            scene.step()


if __name__ == "__main__":
    main()
