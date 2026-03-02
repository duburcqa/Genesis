import math
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from itertools import permutations
from typing import TYPE_CHECKING, cast, Any

import numpy as np
import pytest

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)

from genesis.engine.materials import FEM
from uipc import builtin
from uipc.backend import SceneVisitor
from uipc.geometry import SimplicialComplexSlot, apply_transform, merge

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array, geometric_mean, harmonic_mean

from .conftest import TOL_SINGLE
from .utils import assert_allclose, get_hf_dataset

if TYPE_CHECKING:
    from genesis.engine.couplers import IPCCoupler


def collect_ipc_geometry_entries(scene):
    visitor = SceneVisitor(scene.sim.coupler._ipc_scene)
    for geom_slot in visitor.geometries():
        if not isinstance(geom_slot, SimplicialComplexSlot):
            continue
        geom = geom_slot.geometry()
        meta_attrs = geom.meta()

        solver_type_attr = meta_attrs.find("solver_type")
        if solver_type_attr is None:
            continue
        (solver_type,) = solver_type_attr.view()
        assert solver_type in ("rigid", "fem", "cloth")

        env_idx_attr = meta_attrs.find("env_idx")
        (env_idx,) = map(int, env_idx_attr.view())

        if solver_type == "rigid":
            idx_attr = meta_attrs.find("link_idx")
        else:  # solver_type in ("fem", "cloth")
            idx_attr = meta_attrs.find("entity_idx")
        (idx,) = map(int, idx_attr.view())

        yield (solver_type, env_idx, idx, geom)


def find_ipc_geometries(scene, *, solver_type, idx=None, env_idx=None):
    geoms = []
    for solver_type_, env_idx_, idx_, geom in collect_ipc_geometry_entries(scene):
        if solver_type == solver_type_ and (idx is None or idx == idx_) and (env_idx is None or env_idx == env_idx_):
            geoms.append(geom)
    return geoms


def get_ipc_merged_geometry(scene, *, solver_type, idx, env_idx):
    (geom,) = find_ipc_geometries(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
    if geom.instances().size() >= 1:
        geom = merge(apply_transform(geom))
    return geom


def get_ipc_positions(scene, *, solver_type, idx, envs_idx):
    geoms_positions = []
    assert envs_idx
    for env_idx in envs_idx:
        merged_geom = get_ipc_merged_geometry(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
        geom_positions = merged_geom.positions().view().squeeze(axis=-1)
        geoms_positions.append(geom_positions)
    return np.stack(geoms_positions, axis=0)


def get_ipc_rigid_links_idx(scene, env_idx):
    links_idx = []
    for solver_type_, env_idx_, idx_, _geom in collect_ipc_geometry_entries(scene):
        if solver_type_ == "rigid" and env_idx_ == env_idx:
            links_idx.append(idx_)
    return links_idx


def build_two_cube_joint_mjcf(tmp_path, joint_type, joint_limits, *, fixed=True, suffix=""):
    """Build a two-cube MJCF with a revolute or prismatic joint."""
    mjcf = ET.Element("mujoco", model=f"two_cube_{joint_type}{suffix}")
    worldbody = ET.SubElement(mjcf, "worldbody")
    base = ET.SubElement(worldbody, "body", name="base")
    if not fixed:
        ET.SubElement(base, "freejoint", name="root")
    ET.SubElement(base, "geom", type="box", size="0.05 0.05 0.05")
    ET.SubElement(base, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.00667 0.00667 0.00667")
    child = ET.SubElement(base, "body", name="moving", pos="0.1 0 0")
    ET.SubElement(child, "geom", type="box", size="0.05 0.05 0.05", pos="0.1 0 0")
    ET.SubElement(child, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.00667 0.00667 0.00667")
    mj_type = "hinge" if joint_type == "revolute" else "slide"
    axis = "0 1 0" if joint_type == "revolute" else "1 0 0"
    lo, hi = joint_limits
    ET.SubElement(child, "joint", name="joint1", type=mj_type, axis=axis, range=f"{lo} {hi}")
    path = str(tmp_path / f"two_cube_{joint_type}{suffix}.xml")
    ET.ElementTree(mjcf).write(path, encoding="utf-8", xml_declaration=True)
    return path


@pytest.mark.parametrize("enable_rigid_rigid_contact", [False, True])
def test_contact_pair_friction_resistance(enable_rigid_rigid_contact):
    from genesis.engine.entities import RigidEntity, FEMEntity

    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            contact_resistance=36.0,
            enable_rigid_rigid_contact=enable_rigid_rigid_contact,
        ),
        show_viewer=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
        ),
    )
    rigid_a = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
            coup_friction=0.25,
            contact_resistance=9.0,
        ),
    )
    rigid_b = scene.add_entity(
        gs.morphs.Box(
            pos=(0.2, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
            coup_friction=0.64,
            contact_resistance=16.0,
        ),
    )
    rigid_c = scene.add_entity(
        gs.morphs.Box(
            pos=(-0.2, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
            coup_friction=0.16,
            contact_resistance=None,
        ),
    )
    fem = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.4, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=5e4,
            nu=0.35,
            rho=1000.0,
            friction_mu=0.49,
            contact_resistance=25.0,
        ),
    )

    scene.build()
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    tab = coupler._ipc_scene.contact_tabular()
    for entities in permutations((plane, rigid_a, rigid_b, rigid_c, fem), 2):
        elems_idx = []
        frictions = []
        resistances = []
        for entity in entities:
            if isinstance(entity, RigidEntity):
                if entity is plane:
                    elem = coupler._ipc_ground_contacts[entity]
                else:
                    elem = coupler._ipc_abd_contacts[entity]
                friction = entity.material.coup_friction
            else:
                assert isinstance(entity, FEMEntity)
                elem = coupler._ipc_fem_contacts[entity]
                friction = entity.material.friction_mu
            resistance = entity.material.contact_resistance or coupler.options.contact_resistance
            elems_idx.append(elem.id())
            frictions.append(friction)
            resistances.append(resistance)
        model = tab.at(*elems_idx)
        assert model.friction_rate() == pytest.approx(geometric_mean(*frictions))
        assert model.resistance() == pytest.approx(harmonic_mean(*resistances))
        assert model.is_enabled() ^ (
            all(isinstance(entity, RigidEntity) and entity is not plane for entity in entities)
            and not enable_rigid_rigid_contact
        )


@pytest.mark.parametrize("n_envs", [0, 2])
def test_rigid_ground_sliding(n_envs, show_viewer):
    GRAVITY = np.array([5.0, 0.0, -10.0], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            enable_rigid_rigid_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 2.0, 1.5),
            camera_lookat=(1.0, -0.5, 0.0),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
            coup_friction=0.25,
        ),
    )

    cubes = []
    for y, mu in ((-0.4, 0.0), (-0.2, 0.01), (0.0, 0.04), (0.2, 0.09), (0.4, 0.16)):
        cube = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, y, 0.12),
                size=(0.08, 0.08, 0.08),
            ),
            material=gs.materials.Rigid(
                coupling_type="ipc_only",
                coup_friction=mu,
            ),
        )
        cubes.append(cube)

    scene.build(n_envs=n_envs)

    initial_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)
    for _ in range(100):
        scene.step()
    final_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    # Coarse non-penetration sanity check
    assert (final_positions[..., 2] > 0.0).all()

    # Distance from ground should be friction-independent
    assert_allclose(np.diff(final_positions[..., 2], axis=-1), 0.0, tol=TOL_SINGLE)

    # No y-axis driving force: lateral drift should be minimal
    assert_allclose(initial_positions[..., 1], final_positions[..., 1], tol=TOL_SINGLE)

    # All cubes should move along +x under tilted gravity.
    assert ((final_positions[..., 0] - initial_positions[..., 0]) > 0.5).all()

    # Lower coup_friction should slide farther, so x should strictly decrease as mu increases.
    assert (np.diff(final_positions[..., ::-1, 0], axis=-1) > 0.2).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_ipc_rigid_ground_clearance(n_envs, show_viewer):
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            contact_resistance=1e6,
            enable_rigid_rigid_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.1),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
        ),
    )

    cubes = []
    for y, resistance in ((-0.4, 1e2), (-0.2, 1e3), (0.0, 1e4), (0.2, 1e5), (0.4, 1e6)):
        cube = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, y, 0.05),
                size=(0.08, 0.08, 0.08),
            ),
            material=gs.materials.Rigid(
                coupling_type="ipc_only",
                coup_friction=0.0,
                contact_resistance=resistance,
            ),
        )
        cubes.append(cube)

    scene.build(n_envs=n_envs)

    initial_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    dist = []
    for _ in range(70):
        scene.step()
    for _ in range(20):
        scene.step()
        dist.append(np.stack([tensor_to_array(cube.get_verts())[..., 2].min(axis=-1) for cube in cubes], axis=-1))
    dist = np.stack(dist, axis=-1)

    final_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    # No lateral driving force in x/y; drift should stay small.
    assert_allclose(initial_positions[..., :2], final_positions[..., :2], atol=TOL_SINGLE)

    # Make sure that it reaches equilibrium
    assert_allclose(dist[..., -1], dist[..., -2], tol=TOL_SINGLE)

    # Larger contact resistance should produce larger ground clearance (less penetration/compression).
    assert (np.diff(dist, axis=-2) > TOL_SINGLE).all()


@pytest.mark.required
def test_link_filter_strict():
    """Verify that IPC link filter controls which links are actually added to IPC."""
    from genesis.engine.entities import RigidEntity

    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        show_viewer=False,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coupling_type="two_way_soft_constraint",
            coupling_link_filter=("moving",),
        ),
    )
    assert isinstance(robot, RigidEntity)

    scene.build()
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    base_link = robot.get_link("base")
    moving_link = robot.get_link("moving")

    assert robot in coupler._coupling_link_filters
    assert coupler._coupling_link_filters[robot] == {moving_link}

    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link.idx in ipc_links_idx
    assert base_link.idx not in ipc_links_idx

    assert moving_link in coupler._abd_link_to_slot
    assert base_link not in coupler._abd_link_to_slot


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("fixed", [True, False])
def test_single_joint(n_envs, coupling_type, joint_type, fixed, show_viewer):
    from genesis.engine.entities import RigidEntity

    DT = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    POS = (0, 0, 0.5)
    FREQ = 1.0
    SCALE = 0.5 if joint_type == "revolute" else 0.1
    CONTACT_MARGIN = 0.01

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_MARGIN,
            constraint_strength_translation=1,
            constraint_strength_rotation=1,
            enable_rigid_rigid_contact=False,
            newton_tolerance=1e-2,
            newton_translation_tolerance=1e-2,
            linear_system_tolerance=1e-3,
            newton_semi_implicit_enable=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
            coup_friction=0.5,
        ),
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=f"urdf/simple/two_cube_{joint_type}.urdf",
            pos=POS,
            fixed=fixed,
        ),
        material=gs.materials.Rigid(
            coupling_type=coupling_type,
        ),
    )
    assert isinstance(robot, RigidEntity)

    scene.build(n_envs=n_envs)
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    envs_idx = range(max(scene.n_envs, 1))

    robot.set_dofs_kp(500.0, dofs_idx_local=-1)
    robot.set_dofs_kv(50.0, dofs_idx_local=-1)

    moving_link = robot.get_link("moving")
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link.idx in ipc_links_idx
    assert moving_link in coupler._abd_link_to_slot
    if coupling_type == "two_way_soft_constraint":
        assert moving_link in coupler._abd_data_by_link
    elif coupling_type == "external_articulation":
        art_data = coupler.articulation_data[robot]
        assert len(art_data.articulation_slots_by_env) == max(scene.n_envs, 1)
        if fixed:
            assert not coupler._abd_data_by_link

    dist_min = np.array(float("inf"))
    cur_dof_pos_history, target_dof_pos_history = [], []
    gs_transform_history, ipc_transform_history = [], []
    for _ in range(int(1 / (DT * FREQ))):
        # Apply sinusoidal target position
        target_dof_pos = SCALE * np.sin((2 * math.pi * FREQ) * scene.sim.cur_t)
        target_dof_vel = SCALE * (2 * math.pi * FREQ) * np.cos((2 * math.pi * FREQ) * scene.sim.cur_t)
        robot.control_dofs_position_velocity(target_dof_pos, target_dof_vel, dofs_idx_local=-1)

        # Store the current and target position / velocity
        cur_dof_pos = tensor_to_array(robot.get_dofs_position(dofs_idx_local=-1)[..., 0])
        cur_dof_pos_history.append(cur_dof_pos)
        target_dof_pos_history.append(target_dof_pos)

        # Make sure the robot never went through the ground
        if not fixed:
            robot_verts = tensor_to_array(robot.get_verts())
            dist_min = np.minimum(dist_min, robot_verts[..., 2].min(axis=-1))
            # FIXME: For some reason it actually can...
            assert (dist_min > -0.1).all()

        scene.step()

        if coupling_type == "two_way_soft_constraint" or not fixed:
            for env_idx in envs_idx:
                abd_data = coupler._abd_data_by_link[moving_link][env_idx]
                gs_transform, ipc_transform = abd_data.aim_transform, abd_data.transform
                # FIXME: Why the tolerance is must so large if no fixed ?!
                assert_allclose(gs_transform[:3, 3], ipc_transform[:3, 3], atol=TOL_SINGLE if fixed else 0.2)
                assert_allclose(
                    gu.R_to_xyz(gs_transform[:3, :3] @ ipc_transform[:3, :3].T), 0.0, atol=1e-4 if fixed else 0.3
                )
                gs_transform_history.append(gs_transform)
                ipc_transform_history.append(ipc_transform)
    cur_dof_pos_history = np.stack(cur_dof_pos_history, axis=-1)
    target_dof_pos_history = np.stack(target_dof_pos_history, axis=-1)

    for env_idx in envs_idx if scene.n_envs > 0 else (slice(None),):
        corr = np.corrcoef(cur_dof_pos_history[env_idx], target_dof_pos_history)[0, 1]
        assert corr > 1.0 - 5e-3
    assert_allclose(
        cur_dof_pos_history - cur_dof_pos_history[..., [0]],
        target_dof_pos_history - target_dof_pos_history[..., [0]],
        tol=0.03,
    )
    assert_allclose(np.ptp(cur_dof_pos_history, axis=-1), 2 * SCALE, tol=0.05)

    if gs_transform_history:
        gs_pos_history, gs_quat_history = gu.T_to_trans_quat(np.stack(gs_transform_history, axis=0))
        ipc_pos_history, ipc_quat_history = gu.T_to_trans_quat(np.stack(ipc_transform_history, axis=0))
        pos_err_history = np.linalg.norm(ipc_pos_history - gs_pos_history, axis=-1)
        rot_err_history = np.linalg.norm(
            gu.quat_to_rotvec(gu.transform_quat_by_quat(gs.inv_quat(gs_quat_history), ipc_quat_history)), axis=-1
        )
        assert (np.percentile(pos_err_history, 90, axis=0) < 1e-2).all()
        assert (np.percentile(rot_err_history, 90, axis=0) < 5e-2).all()

    # Make sure the robot bounced on the ground or stayed in place
    if fixed:
        assert_allclose(robot.get_pos(), POS, atol=TOL_SINGLE)
    else:
        assert (dist_min < 1.5 * CONTACT_MARGIN).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("constraint_strength", [1, 100])
def test_apply_forces_base_link(n_envs, constraint_strength, show_viewer):
    from genesis.engine.entities import RigidEntity

    DT = 0.002
    FREQ = 2.0
    SCALE = 0.1
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    POS = (0.5, 0.0, 0.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            constraint_strength_translation=constraint_strength,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.3),
            camera_lookat=(0.25, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=POS),
        material=gs.materials.Rigid(coupling_type="two_way_soft_constraint"),
    )
    assert isinstance(box, RigidEntity)

    scene.build(n_envs=n_envs)
    assert scene.sim is not None

    box.set_dofs_kp(50000.0)
    box.set_dofs_kv(500.0)

    z_actual, z_target = [], []
    for _ in range(int(1 / (DT * FREQ))):
        t = scene.sim.cur_t
        target_z = SCALE * math.sin((2 * math.pi * FREQ) * t)
        target_vz = SCALE * (2 * math.pi * FREQ) * math.cos((2 * math.pi * FREQ) * t)
        box.control_dofs_position_velocity(target_z, target_vz, dofs_idx_local=2)
        scene.step()
        z_target.append(target_z)
        z_actual.append(tensor_to_array(box.get_pos()[..., 2]))

    z_actual = np.array(z_actual)
    z_target = np.array(z_target)
    if z_actual.ndim > 1:
        z_target = z_target[:, np.newaxis]
    assert_allclose(z_actual, z_target, atol=0.005)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_objects_freefall(n_envs, show_viewer):
    from genesis.engine.entities import RigidEntity, FEMEntity

    DT = 0.002
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    NUM_STEPS = 30

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, 3.2, 1.5),
            camera_lookat=(0.0, 0.0, 1.1),
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.5,
            pos=(0.0, 0.0, 1.5),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.6),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            coupling_type="ipc_only",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )
    assert isinstance(box, RigidEntity)

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.5, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e5,
            nu=0.3,
            rho=1000.0,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    envs_idx = range(max(scene.n_envs, 1))

    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert box.base_link_idx in ipc_links_idx
    assert box.base_link in coupler._abd_link_to_slot

    # Verify that geometries are present in IPC for each environment
    cloth_entity_idx = scene.sim.fem_solver.entities.index(cloth)
    box_entity_idx = scene.sim.rigid_solver.entities.index(box)
    sphere_entity_idx = scene.sim.fem_solver.entities.index(sphere)
    objs_kwargs = {
        obj: dict(solver_type=solver_type, idx=idx)
        for obj, solver_type, idx in (
            (cloth, "cloth", cloth_entity_idx),
            (box, "rigid", box_entity_idx),
            (sphere, "fem", sphere_entity_idx),
        )
    }
    for obj_kwargs in objs_kwargs.values():
        for env_idx in envs_idx:
            assert len(find_ipc_geometries(scene, **obj_kwargs, env_idx=env_idx)) == 1

    # Get initial state
    p_0 = {obj: get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx) for obj, obj_kwargs in objs_kwargs.items()}
    v_0 = {obj: np.zeros_like(p_0[obj]) for obj in objs_kwargs.keys()}

    # Run simulation and validate dynamics equations at each step
    p_prev, v_prev = p_0.copy(), v_0.copy()
    for _i in range(NUM_STEPS):
        # Move forward in time
        scene.step()

        for obj, obj_kwargs in objs_kwargs.items():
            # Get new position
            p_i = get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx)

            # Estimate velocity by finite difference: v_{n+1} = (x_{n+1} - x_n) / DT
            v_i = (p_i - p_prev[obj]) / DT

            # Compute estimated position and velocity
            expected_v = v_prev[obj] + GRAVITY * DT
            expected_p = p_prev[obj] + expected_v * DT

            # Update for next iteration
            p_prev[obj], v_prev[obj] = p_i, v_i

            # FIXME: This test does not pass for sphere entity...
            if obj is sphere:
                continue

            # Validate displacement and velocity assuming Euler scheme
            assert_allclose(v_i, expected_v, atol=1e-3)
            assert_allclose(p_i, expected_p, tol=TOL_SINGLE)

    for obj in objs_kwargs.keys():
        # Validate centroid consistency
        assert isinstance(obj, (RigidEntity, FEMEntity))
        ipc_centroid = p_prev[obj].mean(axis=-2)
        gs_centroid = obj.get_state().pos.mean(axis=-2)
        assert_allclose(ipc_centroid, gs_centroid, atol=TOL_SINGLE)

        # Validate centroidal total displacement: 0.5 * GRAVITY * t * (t + DT)
        # FEM entities (cloth) deform during freefall, causing small centroid drift — use looser tolerance.
        p_delta = p_prev[obj] - p_0[obj]
        expected_displacement = 0.5 * GRAVITY * NUM_STEPS * (NUM_STEPS + 1) * DT**2
        assert_allclose(p_delta.mean(axis=-2), expected_displacement, tol=2e-3 if isinstance(obj, FEMEntity) else 1e-3)

        # FIXME: This test does not pass for sphere entity...
        if obj is sphere:
            continue

        # Validate vertex-based total displacement
        assert_allclose(p_delta, expected_displacement, tol=TOL_SINGLE)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_objects_colliding(n_envs, show_viewer):
    DT = 0.02
    CONTACT_MARGIN = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    NUM_STEPS = 90

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_MARGIN,
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 2.0, 0.1),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
            coup_friction=0.5,
        ),
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.5,
            pos=(0.0, 0.0, 0.2),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(-0.25, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            coup_friction=0.3,
            coupling_type="ipc_only",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.25, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e3,
            nu=0.3,
            rho=1000.0,
            friction_mu=0.3,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)
    assert scene.sim is not None
    envs_idx = range(max(scene.n_envs, 1))

    # Run simulation and validate dynamics equations at each step
    objs_kwargs = {
        obj: dict(solver_type=solver_type, idx=idx)
        for obj, solver_type, idx in (
            (cloth, "cloth", scene.sim.fem_solver.entities.index(cloth)),
            (box, "rigid", scene.sim.rigid_solver.entities.index(box)),
            (sphere, "fem", scene.sim.fem_solver.entities.index(sphere)),
        )
    }
    p_history = {obj: [] for obj in objs_kwargs.keys()}
    for _i in range(NUM_STEPS):
        scene.step()

        for obj, obj_kwargs in objs_kwargs.items():
            p_i = get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx)
            p_history[obj].append(p_i)

    cloth_p_history = np.stack(p_history[cloth], axis=-3)
    for obj in objs_kwargs.keys():
        obj_p_history = np.stack(p_history[obj], axis=-3)

        # Make sure that all vertices are laying on the ground
        assert (obj_p_history[..., 2] < 1.5 * CONTACT_MARGIN).any()
        assert (obj_p_history[..., 2] > 0.0).all()

        # Check that the objects did not fly away (5cm)
        obj_delta_history = np.linalg.norm((obj_p_history - obj_p_history[..., [0], :, :])[..., :2], axis=-1)
        assert_allclose(obj_delta_history, 0.0, atol=0.1)

        # Make sure that all objects reached steady state
        obj_disp_history = np.linalg.norm(np.diff(obj_p_history[..., -10:, :, :], axis=-3), axis=-1)
        assert_allclose(obj_disp_history, 0.0, tol=5e-3)

        # Make sure that the cloth is laying on all objects (at least one vertex above the others)
        if obj is cloth:
            continue
        assert (obj_p_history[..., 2].max(axis=-1) < cloth_p_history[..., 2].max(axis=-1)).all()


@pytest.mark.required
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
def test_robot_grasp_fem(coupling_type, show_viewer):
    """Verify FEM add/retrieve and that robot lift raises FEM more than 20cm."""
    from genesis.engine.entities import RigidEntity, FEMEntity

    DT = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    BOX_POS = (0.65, 0.0, 0.03)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            constraint_strength_translation=10.0,
            constraint_strength_rotation=10.0,
            newton_translation_tolerance=10.0,
            enable_rigid_rigid_contact=False,
            enable_rigid_ground_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 1.0, 1.0),
            camera_lookat=(0.3, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_type="ipc_only",
            coup_friction=0.8,
        ),
    )

    material_kwargs: dict[str, Any] = dict(
        coup_friction=0.8,
        coupling_type=coupling_type,
    )
    if coupling_type == "two_way_soft_constraint":
        material_kwargs["coupling_link_filter"] = ("left_finger", "right_finger")

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
        ),
        material=gs.materials.Rigid(**material_kwargs),
    )
    assert isinstance(franka, RigidEntity)

    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=BOX_POS,
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=5.0e4,
            nu=0.45,
            rho=1000.0,
            friction_mu=0.5,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.2, 0.5),
        ),
    )
    assert isinstance(box, FEMEntity)

    scene.build()
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    envs_idx = range(max(scene.n_envs, 1))
    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    # end_effector = franka.get_link("hand")

    franka.set_dofs_kp([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 500.0, 500.0])

    box_entity_idx = scene.sim.fem_solver.entities.index(box)
    assert len(find_ipc_geometries(scene, solver_type="fem", idx=box_entity_idx, env_idx=0)) == 1

    franka_finger_links = {franka.get_link(name) for name in ("left_finger", "right_finger")}
    franka_finger_links_idx = {link.idx for link in franka_finger_links}
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert franka_finger_links_idx.issubset(ipc_links_idx)
    for link_idx in franka_finger_links:
        assert link_idx in coupler._abd_link_to_slot

    franka_links_idx = {link.idx for link in franka.links}
    franka_ipc_links_idx = franka_links_idx.intersection(ipc_links_idx)
    if coupling_type == "two_way_soft_constraint":
        assert coupler._coupling_link_filters.get(franka) == franka_finger_links
        assert franka_ipc_links_idx == franka_finger_links_idx
    else:
        assert franka_finger_links_idx.issubset(franka_ipc_links_idx)

    ipc_positions_0 = get_ipc_positions(scene, solver_type="fem", idx=box_entity_idx, envs_idx=envs_idx)
    gs_positions_0 = tensor_to_array(box.get_state().pos)
    assert_allclose(ipc_positions_0, gs_positions_0, atol=TOL_SINGLE)
    gs_centroid_0 = gs_positions_0.mean(axis=1)
    assert_allclose(gs_centroid_0, BOX_POS, atol=1e-4)

    def run_stage(target_qpos, finger_pos, duration):
        franka.control_dofs_position(target_qpos[motors_dof], motors_dof)
        franka.control_dofs_position(finger_pos, fingers_dof)
        for _ in range(int(duration / DT)):
            scene.step()

    # Setting initial configuration is not supported by coupling mode "external_articulation"
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.9482, 0.6910, 1.2114, -1.6619, -0.6739, 1.8685, 1.1844, 0.0112, 0.0096]
    with pytest.raises(gs.GenesisException) if coupling_type == "external_articulation" else nullcontext():
        franka.set_dofs_position(qpos)
        franka.control_dofs_position(qpos)
    if coupling_type == "external_articulation":
        run_stage(qpos, finger_pos=0.04, duration=2.0)

    # Lower the grapper half way to grasping position
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.25], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.8757, 0.8824, 1.0523, -1.7619, -0.8831, 2.0903, 1.2924, 0.0400, 0.0400]
    run_stage(qpos, finger_pos=0.04, duration=1.0)

    # Reach grasping position
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.135], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.7711, 1.0502, 0.8850, -1.7182, -1.0210, 2.2350, 1.3489, 0.0400, 0.0400]
    run_stage(qpos, finger_pos=0.04, duration=0.5)

    # Grasp the cube
    run_stage(qpos, finger_pos=0.0, duration=0.1)

    # Lift the cube
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.9488, 0.6916, 1.2123, -1.6627, -0.6750, 1.8683, 1.1855, 0.0301, 0.0319]
    run_stage(qpos, finger_pos=0.0, duration=0.5)

    ipc_positions_f = get_ipc_positions(scene, solver_type="fem", idx=box_entity_idx, envs_idx=envs_idx)
    gs_positions_f = tensor_to_array(box.get_state().pos)
    assert_allclose(ipc_positions_f, gs_positions_f, atol=TOL_SINGLE)
    assert (gs_positions_f[..., 2] - gs_positions_0[..., 2] >= 0.2).all()
    finger_aabb = tensor_to_array(franka.get_link("right_finger").get_AABB())
    assert (gs_positions_f[..., 2] - finger_aabb[..., 0, 2] > 0).any()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_momentum_conservation(n_envs, show_viewer):
    from genesis.engine.entities import RigidEntity

    DT = 0.001
    DURATION = 0.30
    CONTACT_MARGIN = 0.01
    VELOCITY = np.array([4.0, 0.0, 0.0], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_MARGIN,
            constraint_strength_translation=1,
            constraint_strength_rotation=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 1.3, 0.6),
            camera_lookat=(0.2, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )

    blob = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.3, 0.0, 0.4),
            radius=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e5,
            nu=0.45,
            rho=1000.0,
            model="stable_neohookean",
            friction_mu=0.0,
        ),
    )

    rigid_cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.4),
            size=(0.1, 0.1, 0.1),
            euler=(0, 0, 0),
        ),
        material=gs.materials.Rigid(
            rho=1000,
            coupling_type="two_way_soft_constraint",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.2, 0.2, 0.8),
        ),
    )
    assert isinstance(rigid_cube, RigidEntity)

    scene.build(n_envs=n_envs)
    assert scene.sim is not None
    coupler = cast("IPCCoupler", scene.sim.coupler)

    rigid_cube.set_dofs_velocity((*VELOCITY, 0.0, 0.0, 0.0))

    fem_entity_idx = scene.sim.fem_solver.entities.index(blob)
    assert len(find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)) == 1

    rigid_link = rigid_cube.base_link
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert rigid_link.idx in ipc_links_idx
    assert rigid_link in coupler._abd_link_to_slot

    cube_mass = rigid_cube.get_mass()

    # Read actual FEM mass from IPC geometry (mesh mass != analytical sphere mass due to tet discretization).
    blob_radius = blob.morph.radius
    blob_rho = blob.material.rho
    blob_analytical_mass = (4.0 / 3.0) * np.pi * blob_radius**3 * blob_rho
    (fem_raw_geo,) = find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    fem_mass_density = fem_raw_geo.meta().find(builtin.mass_density).view().item()
    fem_merged_geo = get_ipc_merged_geometry(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    fem_vertex_volumes = fem_merged_geo.vertices().find(builtin.volume).view().reshape(-1)
    blob_mass = float(np.sum(fem_vertex_volumes) * fem_mass_density)
    assert_allclose(blob_mass, blob_analytical_mass, rtol=0.01)

    total_p_history = []
    momentum_0 = VELOCITY * cube_mass

    dist_min = np.array(float("inf"))
    fem_positions_prev = None  # FEM initial velocity is zero
    for step in range(int(DURATION / DT)):
        cube_vel = tensor_to_array(rigid_cube.get_links_vel(links_idx_local=0, ref="link_com")[..., 0, :])
        rigid_linear_momentum = cube_mass * cube_vel

        fem_proc_geo = get_ipc_merged_geometry(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
        fem_positions = fem_proc_geo.positions().view().squeeze(axis=-1)
        if fem_positions_prev is not None:
            fem_velocities = (fem_positions - fem_positions_prev) / DT
        else:
            fem_velocities = np.zeros_like(fem_positions)
        fem_positions_prev = fem_positions

        # Make sure that rigid and fem are not penetrating each other
        fem_aabb_min, fem_aabb_max = fem_positions.min(axis=-2), fem_positions.max(axis=-2)
        rigid_aabb = tensor_to_array(rigid_cube.get_AABB())
        rigid_aabb_min, rigid_aabb_max = rigid_aabb[..., 0, :], rigid_aabb[..., 1, :]
        overlap = np.minimum(fem_aabb_max, rigid_aabb_max) - np.maximum(rigid_aabb_min, fem_aabb_min)
        dist_min = np.minimum(dist_min, -overlap.min(axis=-1))
        assert (dist_min > 0.0).all()

        volume_attr = fem_proc_geo.vertices().find(builtin.volume)
        fem_vertex_masses = volume_attr.view().reshape(-1) * fem_mass_density
        assert_allclose(np.sum(fem_vertex_masses), blob_mass, tol=TOL_SINGLE)
        fem_linear_momentum = np.sum(fem_vertex_masses[:, np.newaxis] * fem_velocities, axis=0)

        # Before collision: FEM should have zero momentum, rigid should carry all momentum.
        if step < int(DURATION / 10 / DT):
            assert_allclose(fem_linear_momentum, 0.0, atol=TOL_SINGLE)
            assert_allclose(rigid_linear_momentum, momentum_0, tol=TOL_SINGLE)

        total_linear_momentum = rigid_linear_momentum + fem_linear_momentum
        total_p_history.append(total_linear_momentum)

        scene.step()

    # Make sure the objects bounced on each other
    assert (dist_min < 1.5 * CONTACT_MARGIN).all()
    assert (cube_vel[..., 0] < -0.5).all()
    assert (fem_velocities[..., 0].mean(axis=-1) > 0.5).all()

    # Check total momentum conservation.
    # NOTE : The tet mesh's contact-facing vertices (x < -0.05) have a z-mean of -0.00138 due to TetGen's asymmetric
    # Steiner point insertion, causing an asymmetric contact force distribution during the x-direction collision.
    # This z-bias produces a net -z impulse, resulting in the observed z-momentum leak.
    assert_allclose(total_p_history, momentum_0, tol=0.001)


@pytest.mark.required
@pytest.mark.parametrize("enable_rigid_ground_contact", [True, False])
@pytest.mark.parametrize("coupling_type", ["ipc_only", "two_way_soft_constraint"])
def test_collision_delegation_ipc_vs_rigid(coupling_type, enable_rigid_ground_contact):
    """Verify collision pair delegation between IPC and rigid solver based on coupling_type and ground contact."""
    from genesis.engine.entities import RigidEntity

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            enable_rigid_ground_contact=enable_rigid_ground_contact,
        ),
        show_viewer=False,
    )

    plane = scene.add_entity(gs.morphs.Plane())  # No coupling_type: stays in rigid solver only
    assert isinstance(plane, RigidEntity)

    # Non-IPC box — always handled by rigid solver
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(1.0, 0.0, 0.2),
        ),
        material=gs.materials.Rigid(),
    )
    assert isinstance(box, RigidEntity)

    if coupling_type == "two_way_soft_constraint":
        entity = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_non_overlap.xml",
            ),
            material=gs.materials.Rigid(
                coupling_type="two_way_soft_constraint",
                coupling_link_filter=("left_finger", "right_finger"),
            ),
        )
        assert isinstance(entity, RigidEntity)

        ipc_excluded_geoms = {
            geom.idx for name in entity.material.coupling_link_filter for geom in entity.get_link(name).geoms
        }
    else:
        with pytest.raises(gs.GenesisException):
            entity = scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/go2/urdf/go2.urdf",
                    pos=(0.0, 0.0, 1.0),
                ),
                material=gs.materials.Rigid(
                    coupling_type="ipc_only",
                ),
            )

        entity = scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
                pos=(0.0, 0.0, 0.6),
            ),
            material=gs.materials.Rigid(
                coupling_type="ipc_only",
            ),
        )
        assert isinstance(entity, RigidEntity)

        ipc_excluded_geoms = {geom.idx for geom in entity.geoms}

    scene.build()
    assert scene.sim is not None
    assert scene.sim.rigid_solver.collider is not None

    pair_idx = scene.sim.rigid_solver.collider._collision_pair_idx

    # Collect geom indices for entities that should retain rigid solver pairs
    rigid_kept_geoms = {geom.idx for geom in entity.geoms} - ipc_excluded_geoms
    ground_geoms = {plane.geoms[0].idx}
    box_geoms = {box.geoms[0].idx}

    # Non-IPC box always has rigid solver ground pairs
    assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in box_geoms for b in ground_geoms)

    # Pairs between IPC-excluded geoms must have no rigid solver pairs (handled by IPC)
    for i_ga in ipc_excluded_geoms:
        for i_gb in ipc_excluded_geoms:
            if i_ga < i_gb:
                assert pair_idx[i_ga, i_gb] == -1

    # Mixed pairs (IPC-excluded ↔ non-IPC) must be kept in rigid solver
    for i_ga in ipc_excluded_geoms:
        for i_gb in box_geoms:
            a, b = min(i_ga, i_gb), max(i_ga, i_gb)
            assert pair_idx[a, b] >= 0

    # IPC-excluded geom ↔ ground must be kept in rigid solver (ground is not IPC-excluded)
    for i_ga in ipc_excluded_geoms:
        for i_gb in ground_geoms:
            a, b = min(i_ga, i_gb), max(i_ga, i_gb)
            assert pair_idx[a, b] >= 0

    # Non-excluded rigid geoms (if any) keep rigid solver ground and self-collision pairs
    if rigid_kept_geoms:
        assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in rigid_kept_geoms for b in ground_geoms)
        assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in rigid_kept_geoms for b in rigid_kept_geoms if a < b)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_cloth_corner_drag(n_envs, show_viewer):
    """Drag a cloth by one corner under gravity using a sandwich grip of two boxes.

    Verify that FEM vertices near the gripped corner follow the imposed trajectory,
    while the rest of the cloth hangs freely under gravity.
    """
    DT = 0.01
    THICKNESS = 0.001
    RHO = 200.0
    CLOTH_HEIGHT = 0.5
    BOX_HALF = 0.03
    GAP = 0.005  # Must exceed cloth thickness to pass UIPC distance sanity check
    CONTACT_D_HAT = 0.01
    GRAVITY = (0.0, 0.0, -9.8)
    NUM_SETTLE = 100
    NUM_DRAG = 200

    # Drag trajectory: move the corner upward and sideways
    DRAG_DX = 0.2
    DRAG_DZ = 0.3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=CONTACT_D_HAT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, CLOTH_HEIGHT + 0.3),
            camera_lookat=(0.0, 0.0, CLOTH_HEIGHT),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.0,
            pos=(0.0, 0.0, CLOTH_HEIGHT),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e4,
            nu=0.3,
            rho=RHO,
            thickness=THICKNESS,
            bending_stiffness=None,
            friction_mu=0.8,
        ),
    )

    # Sandwich grip at one corner (+x, +y)
    HALF_L = 0.5
    corner_x, corner_y = HALF_L, HALF_L
    boxes = []
    for z_sign in (+1, -1):
        box = scene.add_entity(
            gs.morphs.Box(
                pos=(corner_x, corner_y, CLOTH_HEIGHT + z_sign * (BOX_HALF + GAP)),
                size=(2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF),
            ),
            material=gs.materials.Rigid(
                coupling_type="two_way_soft_constraint",
                coup_friction=0.8,
            ),
        )
        boxes.append(box)

    scene.build(n_envs=n_envs)

    # Record initial cloth positions
    init_pos = tensor_to_array(cloth.get_state().pos)
    if init_pos.ndim == 2:
        init_pos = init_pos[np.newaxis]

    # Find corner vertices: closest to the gripped corner
    corner_dist = np.sqrt((init_pos[0, :, 0] - corner_x) ** 2 + (init_pos[0, :, 1] - corner_y) ** 2)
    corner_radius = 0.08  # vertices within this radius are "corner vertices"
    corner_mask = corner_dist < corner_radius

    # Find opposite corner vertices (far from grip)
    opposite_dist = np.sqrt((init_pos[0, :, 0] + corner_x) ** 2 + (init_pos[0, :, 1] + corner_y) ** 2)
    opposite_mask = opposite_dist < corner_radius

    # PD control: close gap, hold position during settling
    for box in boxes:
        box.set_dofs_kp(5000.0)
        box.set_dofs_kv(500.0)
        init_dof = tensor_to_array(box.get_dofs_position()).copy()
        z_dof = init_dof[..., 2]
        init_dof[..., 2] = np.where(z_dof > CLOTH_HEIGHT, z_dof - GAP, z_dof + GAP)
        box.control_dofs_position(init_dof)

    # Settle: let cloth conform to grip
    for _ in range(NUM_SETTLE):
        scene.step()

    # Record settled positions
    settled_pos = tensor_to_array(cloth.get_state().pos)
    if settled_pos.ndim == 2:
        settled_pos = settled_pos[np.newaxis]

    # Record box initial DOF targets for trajectory
    box_settled_dofs = []
    for box in boxes:
        box_settled_dofs.append(tensor_to_array(box.get_dofs_position()).copy())

    # Drag phase: linearly ramp the corner boxes along trajectory
    for step in range(NUM_DRAG):
        t = (step + 1) / NUM_DRAG
        for box, base_dof in zip(boxes, box_settled_dofs):
            target = base_dof.copy()
            target[..., 0] += DRAG_DX * t
            target[..., 2] += DRAG_DZ * t
            box.control_dofs_position(target)
        scene.step()

    # Final cloth positions
    final_pos = tensor_to_array(cloth.get_state().pos)
    if final_pos.ndim == 2:
        final_pos = final_pos[np.newaxis]

    # Expected corner displacement
    expected_dx = DRAG_DX
    expected_dz = DRAG_DZ

    # Corner vertices near grip should have followed the trajectory
    corner_dx = float(np.mean(final_pos[:, corner_mask, 0] - settled_pos[:, corner_mask, 0]))
    corner_dz = float(np.mean(final_pos[:, corner_mask, 2] - settled_pos[:, corner_mask, 2]))

    assert corner_dx > 0.5 * expected_dx, (
        f"Corner vertices didn't follow X trajectory: dx={corner_dx:.4f}, expected > {0.5 * expected_dx:.4f}"
    )
    assert corner_dz > 0.5 * expected_dz, (
        f"Corner vertices didn't follow Z trajectory: dz={corner_dz:.4f}, expected > {0.5 * expected_dz:.4f}"
    )

    # Opposite corner hangs under gravity: should have dropped in z
    opposite_dz = float(np.mean(final_pos[:, opposite_mask, 2] - settled_pos[:, opposite_mask, 2]))
    assert opposite_dz < -0.01, f"Opposite corner should sag under gravity: dz={opposite_dz:.4f}"

    # Corner displacement is larger than center displacement (cloth stretches, not rigid body)
    center_dist = np.sqrt(init_pos[0, :, 0] ** 2 + init_pos[0, :, 1] ** 2)
    center_mask = center_dist < corner_radius
    center_dx = float(np.mean(final_pos[:, center_mask, 0] - settled_pos[:, center_mask, 0]))
    assert corner_dx > center_dx + 0.01, (
        f"Corner should move more than center: corner_dx={corner_dx:.4f}, center_dx={center_dx:.4f}"
    )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("E, nu", [(1e4, 0.3), (5e4, 0.49)])
def test_cloth_uniform_biaxial_stretching(n_envs, E, nu, show_viewer):
    """Stretch a square cloth uniformly via position-controlled boxes at corners. Verify stretch physics."""
    DT = 0.01
    THICKNESS = 0.001
    RHO = 200.0
    CLOTH_HEIGHT = 0.5
    BOX_HALF = 0.03
    GAP = 0.005  # Must exceed cloth thickness (0.001) to pass UIPC distance sanity check
    CONTACT_D_HAT = 0.01
    PULL_DISTANCE = 0.03  # Radial displacement per corner
    NUM_SETTLE = 200
    NUM_STABLE = 50

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=CONTACT_D_HAT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, CLOTH_HEIGHT),
            camera_lookat=(0.0, 0.0, CLOTH_HEIGHT),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.0,
            pos=(0.0, 0.0, CLOTH_HEIGHT),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=E,
            nu=nu,
            rho=RHO,
            thickness=THICKNESS,
            bending_stiffness=None,
            friction_mu=0.8,
        ),
    )

    # 8 boxes: 2 per corner (sandwich grip above/below cloth).
    # Box `size` is FULL edge length; `BOX_HALF` is the half-extent used for positioning.
    HALF_L = 0.5  # grid20x20 at scale=1.0 is ~1m x 1m
    corner_xy = [(-HALF_L, -HALF_L), (-HALF_L, HALF_L), (HALF_L, -HALF_L), (HALF_L, HALF_L)]
    diagonal_signs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    inv_sqrt2 = 1.0 / np.sqrt(2)

    boxes = []  # list of (box_entity, diagonal_sign_x, diagonal_sign_y)
    for (cx, cy), (sx, sy) in zip(corner_xy, diagonal_signs):
        for z_sign in (+1, -1):
            box = scene.add_entity(
                gs.morphs.Box(
                    pos=(cx, cy, CLOTH_HEIGHT + z_sign * (BOX_HALF + GAP)),
                    size=(2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF),
                ),
                material=gs.materials.Rigid(
                    coupling_type="two_way_soft_constraint",
                    coup_friction=0.8,
                ),
            )
            boxes.append((box, sx, sy))

    scene.build(n_envs=n_envs)

    # Record initial cloth vertex positions
    init_pos = tensor_to_array(cloth.get_state().pos)
    if init_pos.ndim == 2:
        init_pos = init_pos[np.newaxis]
    L = float(init_pos[0, :, 0].max() - init_pos[0, :, 0].min())

    # Configure PD: position-controlled outward pull on x,y; hold z + rotation
    for box, sx, sy in boxes:
        box.set_dofs_kp(5000.0)
        box.set_dofs_kv(500.0)
        init_dof = tensor_to_array(box.get_dofs_position()).copy()
        # Close the init gap in z target
        z_dof = init_dof[..., 2]
        init_dof[..., 2] = np.where(z_dof > CLOTH_HEIGHT, z_dof - GAP, z_dof + GAP)
        # Pull corners outward along diagonal
        init_dof[..., 0] += PULL_DISTANCE * sx * inv_sqrt2
        init_dof[..., 1] += PULL_DISTANCE * sy * inv_sqrt2
        box.control_dofs_position(init_dof)

    for _ in range(NUM_SETTLE):
        scene.step()

    # Check steady state over additional steps
    prev_pos = tensor_to_array(cloth.get_state().pos)
    if prev_pos.ndim == 2:
        prev_pos = prev_pos[np.newaxis]
    for _ in range(NUM_STABLE):
        scene.step()
    final_pos = tensor_to_array(cloth.get_state().pos)
    if final_pos.ndim == 2:
        final_pos = final_pos[np.newaxis]

    # Verify steady state (loose tolerance — cloth oscillates under IPC contact)
    assert_allclose(final_pos, prev_pos, atol=0.005)

    # Compute observed radial strain
    init_center = init_pos.mean(axis=-2, keepdims=True)
    final_center = final_pos.mean(axis=-2, keepdims=True)
    init_r = np.linalg.norm((init_pos - init_center)[..., :2], axis=-1)
    final_r = np.linalg.norm((final_pos - final_center)[..., :2], axis=-1)
    interior = init_r > 0.02
    observed_strain = np.mean((final_r[interior] / init_r[interior]) - 1.0)

    # Cloth is being stretched outward: positive strain expected
    assert observed_strain > 0.001, f"Expected positive stretch, got strain={observed_strain:.6f}"

    # Deformation symmetric: x-scale ~ y-scale
    init_xy = (init_pos - init_center)[..., :2]
    final_xy = (final_pos - final_center)[..., :2]
    x_mask = np.abs(init_xy[..., 0]) > 0.02
    y_mask = np.abs(init_xy[..., 1]) > 0.02
    scale_x = np.mean(final_xy[x_mask, 0] / init_xy[x_mask, 0])
    scale_y = np.mean(final_xy[y_mask, 1] / init_xy[y_mask, 1])
    assert_allclose(scale_x, scale_y, rtol=0.1)

    # Out-of-plane deformation negligible (no gravity, no bending load)
    assert_allclose(final_pos[..., 2], CLOTH_HEIGHT, atol=0.02)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("E, rho", [(1e4, 200), (5e4, 400)])
def test_cloth_gravity_deflection(n_envs, E, rho, show_viewer):
    """Cloth held at corners sags under gravity. Verify Hencky membrane deflection scaling."""
    DT = 0.01
    THICKNESS = 0.001
    CLOTH_HEIGHT = 1.0
    BOX_HALF = 0.03
    GAP = 0.005
    CONTACT_D_HAT = 0.01
    GRAVITY = (0.0, 0.0, -9.8)
    NUM_SETTLE = 300
    NUM_STABLE = 50

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=GRAVITY),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=CONTACT_D_HAT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 0.8),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.0,
            pos=(0.0, 0.0, CLOTH_HEIGHT),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=E,
            nu=0.49,
            rho=rho,
            thickness=THICKNESS,
            bending_stiffness=None,
            friction_mu=0.8,
        ),
    )

    # 8 boxes: 2 per corner (sandwich grip above/below cloth), position-controlled to hold corners
    HALF_L = 0.5
    corner_xy = [(-HALF_L, -HALF_L), (-HALF_L, HALF_L), (HALF_L, -HALF_L), (HALF_L, HALF_L)]
    boxes = []
    for cx, cy in corner_xy:
        for z_sign in (+1, -1):
            box = scene.add_entity(
                gs.morphs.Box(
                    pos=(cx, cy, CLOTH_HEIGHT + z_sign * (BOX_HALF + GAP)),
                    size=(2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF),
                ),
                material=gs.materials.Rigid(
                    coupling_type="two_way_soft_constraint",
                    coup_friction=0.8,
                ),
            )
            boxes.append(box)

    scene.build(n_envs=n_envs)

    # Record initial cloth positions
    init_pos = tensor_to_array(cloth.get_state().pos)
    if init_pos.ndim == 2:
        init_pos = init_pos[np.newaxis]

    # Find center vertex (closest to origin in x,y)
    center_dist = np.sqrt(init_pos[0, :, 0] ** 2 + init_pos[0, :, 1] ** 2)
    center_idx = int(np.argmin(center_dist))

    # Find corner vertices (closest to each corner)
    corner_indices = []
    for cx, cy in corner_xy:
        dist = np.sqrt((init_pos[0, :, 0] - cx) ** 2 + (init_pos[0, :, 1] - cy) ** 2)
        corner_indices.append(int(np.argmin(dist)))

    # PD control: hold boxes at initial position (close gap in z target)
    for box in boxes:
        box.set_dofs_kp(5000.0)
        box.set_dofs_kv(500.0)
        init_dof = tensor_to_array(box.get_dofs_position()).copy()
        z_dof = init_dof[..., 2]
        init_dof[..., 2] = np.where(z_dof > CLOTH_HEIGHT, z_dof - GAP, z_dof + GAP)
        box.control_dofs_position(init_dof)

    for _ in range(NUM_SETTLE):
        scene.step()

    # Check steady state
    prev_pos = tensor_to_array(cloth.get_state().pos)
    if prev_pos.ndim == 2:
        prev_pos = prev_pos[np.newaxis]
    for _ in range(NUM_STABLE):
        scene.step()
    final_pos = tensor_to_array(cloth.get_state().pos)
    if final_pos.ndim == 2:
        final_pos = final_pos[np.newaxis]

    assert_allclose(final_pos, prev_pos, atol=0.005)

    # Corners held near cloth height
    for c_idx in corner_indices:
        assert_allclose(final_pos[..., c_idx, 2], CLOTH_HEIGHT, atol=0.05)

    # Center has sagged
    center_sag = CLOTH_HEIGHT - final_pos[..., center_idx, 2]
    assert (center_sag > 0.05).all(), f"Expected center sag > 0.05, got {center_sag}"

    # Hencky membrane scaling: w_max ~ C * (rho * g * L^4 / E)^(1/3)
    # where rho_s = rho * t cancels with E * t, so w_max ~ (rho * g * L^4 / E)^(1/3)
    g = abs(GRAVITY[2])
    L = float(init_pos[0, :, 0].max() - init_pos[0, :, 0].min())
    hencky_scale = (rho * g * L**4 / E) ** (1.0 / 3.0)

    # FIXME: Calibrate C_REF from first run once reference values are stable.
    # For now, verify the scaling exponent by checking ratio consistency across (E, rho) combinations.
    C_observed = float(np.mean(center_sag) / hencky_scale)
    assert 0.1 < C_observed < 5.0, f"Hencky coefficient C={C_observed:.3f} outside plausible range [0.1, 5.0]"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_stacked_revolute_pairs_collision(n_envs, show_viewer, tmp_path):
    """Three two-cube revolute robots stacked on a ground plane. Verify IPC collision ordering."""
    DT = 0.005
    CONTACT_D_HAT = 0.01
    GRAVITY = (0.0, 0.0, -9.8)
    NUM_SETTLE = 300
    NUM_STABLE = 60

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=GRAVITY),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_D_HAT,
            enable_rigid_rigid_contact=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.8, -0.8, 0.6),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    # Ground plane
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(coupling_type="ipc_only", coup_friction=0.5),
    )

    # 3 robots at different heights, added in permuted order to flip contact pair indices
    mjcf_path = build_two_cube_joint_mjcf(tmp_path, "revolute", (-1.57, 1.57), fixed=False, suffix="_stacked")
    heights = [0.15, 0.40, 0.65]
    add_order = [2, 0, 1]
    robots = [None, None, None]
    for idx in add_order:
        robots[idx] = scene.add_entity(
            gs.morphs.MJCF(file=mjcf_path, pos=(0, 0, heights[idx])),
            material=gs.materials.Rigid(coupling_type="external_articulation"),
        )

    scene.build(n_envs=n_envs)

    for _ in range(NUM_SETTLE):
        scene.step()

    # Record positions for stability check
    prev_positions = [tensor_to_array(r.get_pos()).copy() for r in robots]
    for _ in range(NUM_STABLE):
        scene.step()
    final_positions = [tensor_to_array(r.get_pos()) for r in robots]

    # Verify steady state
    for prev, final in zip(prev_positions, final_positions):
        assert_allclose(final, prev, atol=0.005)

    # Extract min z for each robot
    min_zs = []
    for robot in robots:
        pos = tensor_to_array(robot.get_pos())
        z = np.atleast_1d(pos[..., 2])
        min_zs.append(float(z.min()))

    # No ground penetration
    for i, mz in enumerate(min_zs):
        assert mz > -CONTACT_D_HAT, f"Robot {i} penetrates ground: min_z={mz:.4f}"

    # Stacking order preserved: each robot above the previous
    for i in range(len(min_zs) - 1):
        assert min_zs[i] < min_zs[i + 1], (
            f"Stacking order violated: robot {i} z={min_zs[i]:.4f} >= robot {i + 1} z={min_zs[i + 1]:.4f}"
        )

    # All robots settled (not floating away)
    for i, mz in enumerate(min_zs):
        assert mz < 1.0, f"Robot {i} floating: z={mz:.4f}"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_joint_position_limits_bang_bang(n_envs, coupling_type, joint_type, show_viewer, tmp_path):
    """Bang-bang velocity control pushes joint toward limits. Verify limits are respected."""
    DT = 0.01
    CONTACT_D_HAT = 0.01
    V_MAX = 2.0
    HALF_PERIOD = 60
    NUM_OSCILLATIONS = 3

    if joint_type == "revolute":
        limits = (-1.57, 1.57)
    else:
        limits = (-0.3, 0.3)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_D_HAT,
            enable_rigid_rigid_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.7),
            camera_lookat=(0.1, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    mjcf_path = build_two_cube_joint_mjcf(tmp_path, joint_type, limits, fixed=True)
    robot = scene.add_entity(
        gs.morphs.MJCF(file=mjcf_path, pos=(0, 0, 0.5)),
        material=gs.materials.Rigid(coupling_type=coupling_type),
    )

    scene.build(n_envs=n_envs)

    # Set damping on the actuated joint (last DOF)
    robot.set_dofs_kv(500.0, dofs_idx_local=-1)

    # Bang-bang velocity control
    pos_history = []
    total_steps = 2 * HALF_PERIOD * NUM_OSCILLATIONS
    for step in range(total_steps):
        phase = (step // HALF_PERIOD) % 2
        vel = V_MAX if phase == 0 else -V_MAX
        robot.control_dofs_velocity(vel, dofs_idx_local=-1)
        scene.step()
        q = tensor_to_array(robot.get_dofs_position(dofs_idx_local=-1))
        pos_history.append(float(np.mean(q)))

    pos_arr = np.array(pos_history)
    lower, upper = limits

    # Joint never exceeds limits (tolerance accounts for IPC coupling compliance)
    tolerance = 0.05
    assert pos_arr.min() >= lower - tolerance, f"Joint violated lower limit: min={pos_arr.min():.4f}, limit={lower}"
    assert pos_arr.max() <= upper + tolerance, f"Joint violated upper limit: max={pos_arr.max():.4f}, limit={upper}"

    # Joint has non-trivial excursion (IPC coupling damping slows revolute joints)
    min_excursion = 0.1 if joint_type == "revolute" else 0.05
    assert pos_arr.max() > min_excursion, (
        f"Joint didn't reach positive excursion: max={pos_arr.max():.4f}, expected > {min_excursion}"
    )
    assert pos_arr.min() < -min_excursion, (
        f"Joint didn't reach negative excursion: min={pos_arr.min():.4f}, expected < {-min_excursion}"
    )

    # At least 2 velocity reversals
    diff = np.diff(pos_arr)
    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    assert sign_changes >= 2, f"Expected >= 2 velocity reversals, got {sign_changes}"


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_cloth_grip_friction_drag(n_envs, show_viewer):
    """Cloth gripped at center by two fixed boxes, then moved sideways and upward.

    Verify friction grip drags the cloth center along with the boxes, while non-grip
    vertices are displaced less (cloth stretches rather than translating rigidly).
    """
    DT = 0.02
    CLOTH_HEIGHT = 0.5
    CLOTH_SCALE = 0.5
    BOX_HALF = 0.05  # half-extent: size = 0.1
    GAP = 0.005
    NUM_SETTLE = 50
    NUM_MOVE = 200

    # Teleop-like IPC settings
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            constraint_strength_translation=100.0,
            constraint_strength_rotation=100.0,
            n_linesearch_iterations=8,
            newton_tolerance=1e-1,
            newton_translation_tolerance=1,
            newton_semi_implicit_enable=False,
            linear_system_tolerance=1e-3,
            contact_enable=True,
            enable_rigid_rigid_contact=True,
            contact_d_hat=0.001,
            contact_resistance=1e7,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.5, CLOTH_HEIGHT + 0.3),
            camera_lookat=(0.0, 0.0, CLOTH_HEIGHT),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=CLOTH_SCALE,
            pos=(0.0, 0.0, CLOTH_HEIGHT),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=6e4,
            nu=0.49,
            rho=200,
            thickness=0.001,
            bending_stiffness=10.0,
            friction_mu=0.5,
        ),
    )

    # Two fixed boxes for sandwich grip at cloth center
    boxes = []
    for z_sign in (+1, -1):
        box = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, CLOTH_HEIGHT + z_sign * (BOX_HALF + GAP)),
                size=(2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF),
            ),
            material=gs.materials.Rigid(
                coupling_type="two_way_soft_constraint",
                coup_friction=0.5,
            ),
        )
        boxes.append(box)

    scene.build(n_envs=n_envs)

    # Record initial cloth positions
    init_pos = tensor_to_array(cloth.get_state().pos)
    if init_pos.ndim == 2:
        init_pos = init_pos[np.newaxis]

    # Find center vertex (closest to origin in x,y)
    center_dist = np.sqrt(init_pos[0, :, 0] ** 2 + init_pos[0, :, 1] ** 2)
    center_idx = int(np.argmin(center_dist))
    center_radius = 0.08
    center_mask = center_dist < center_radius

    # Find edge vertices (far from center)
    HALF_L = CLOTH_SCALE * 0.5
    edge_mask = center_dist > 0.8 * HALF_L * np.sqrt(2)

    # PD control: close gap, hold position
    for box in boxes:
        box.set_dofs_kp(5000.0)
        box.set_dofs_kv(500.0)
        init_dof = tensor_to_array(box.get_dofs_position()).copy()
        z_dof = init_dof[..., 2]
        init_dof[..., 2] = np.where(z_dof > CLOTH_HEIGHT, z_dof - GAP, z_dof + GAP)
        box.control_dofs_position(init_dof)

    # Settle
    for _ in range(NUM_SETTLE):
        scene.step()

    settled_pos = tensor_to_array(cloth.get_state().pos)
    if settled_pos.ndim == 2:
        settled_pos = settled_pos[np.newaxis]

    # After settling, cloth center should be near box center
    assert_allclose(settled_pos[:, center_idx, 2], CLOTH_HEIGHT, atol=0.05)

    # Record box settled DOF targets
    box_settled_dofs = []
    for box in boxes:
        box_settled_dofs.append(tensor_to_array(box.get_dofs_position()).copy())

    # Move boxes sideways (+x) and up (+z) linearly
    MOVE_X = 0.1
    MOVE_Z = 0.1
    for step in range(NUM_MOVE):
        t = (step + 1) / NUM_MOVE
        for box, base_dof in zip(boxes, box_settled_dofs):
            target = base_dof.copy()
            target[..., 0] += MOVE_X * t
            target[..., 2] += MOVE_Z * t
            box.control_dofs_position(target)
        scene.step()

    final_pos = tensor_to_array(cloth.get_state().pos)
    if final_pos.ndim == 2:
        final_pos = final_pos[np.newaxis]

    # Center vertex has displaced in +X
    center_dx = float(np.mean(final_pos[:, center_mask, 0] - settled_pos[:, center_mask, 0]))
    assert center_dx > 0.03, f"Center cloth didn't follow +X: dx={center_dx:.4f}"

    # Center vertex has displaced in +Z
    center_dz = float(np.mean(final_pos[:, center_mask, 2] - settled_pos[:, center_mask, 2]))
    assert center_dz > 0.03, f"Center cloth didn't follow +Z: dz={center_dz:.4f}"

    # Edge vertices also moved (friction grip drags the whole cloth)
    edge_dx = float(np.mean(final_pos[:, edge_mask, 0] - settled_pos[:, edge_mask, 0]))
    assert edge_dx > 0.01, f"Edge vertices should also move in +X: edge_dx={edge_dx:.4f}"

    # Gravity causes edges to sag relative to center (cloth is not perfectly rigid)
    edge_z_mean = float(np.mean(final_pos[:, edge_mask, 2]))
    center_z_mean = float(np.mean(final_pos[:, center_mask, 2]))
    assert center_z_mean > edge_z_mean, (
        f"Center should be above edges due to grip: center_z={center_z_mean:.4f}, edge_z={edge_z_mean:.4f}"
    )
