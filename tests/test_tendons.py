import numpy as np
import mujoco
import pytest

import genesis as gs

from .utils import assert_allclose


# An actuated fixed tendon coupling a hinge and a slide joint, plus a passive spring tendon with a non-trivial resting
# length, exercising the actuation and passive-force paths. Actuation involves no constraints, so with matched
# integrator, gravity and (disabled) default armature the actuated trajectory matches MuJoCo to high precision; the
# spring is well-damped so it settles at its analytic resting length.
TENDON_SMOOTH_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="armA" pos="0 0 0">
      <joint name="hingeA" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".1 .02 .02" mass="1"/>
      <body name="armA2" pos="0.2 0 0">
        <joint name="slideA" type="slide" axis="1 0 0"/>
        <geom type="box" size=".05 .02 .02" mass="0.5"/>
      </body>
    </body>
    <body name="armB" pos="0 1 0">
      <joint name="hingeB" type="hinge" axis="0 0 1" damping="2.0"/>
      <geom type="box" size=".1 .02 .02" mass="1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="driveA">
      <joint joint="hingeA" coef="2.0"/>
      <joint joint="slideA" coef="1.0"/>
    </fixed>
    <fixed name="springB" stiffness="30.0" springlength="0.3 0.3">
      <joint joint="hingeB" coef="1.0"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="driveA_motor" tendon="driveA"/>
  </actuator>
</mujoco>
"""


# A fixed tendon with a length limit. Driving it monotonically lets the limit constraint arrest the motion.
TENDON_LIMIT_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 0 1" damping="0.01"/>
      <geom type="box" size=".1 .02 .02" mass="1"/>
      <body name="arm2" pos="0.2 0 0">
        <joint name="slide" type="slide" axis="1 0 0" damping="0.01"/>
        <geom type="box" size=".05 .02 .02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="capped" limited="true" range="-0.05 0.3">
      <joint joint="hinge" coef="1.0"/>
      <joint joint="slide" coef="1.0"/>
    </fixed>
  </tendon>
  <actuator>
    <motor name="capped_motor" tendon="capped"/>
  </actuator>
</mujoco>
"""


def _build_smooth_scene(xml_path, integrator=gs.integrator.implicitfast):
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.002,
            gravity=(0.0, 0.0, 0.0),
            integrator=integrator,
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file=xml_path, default_armature=0.0),
    )
    scene.build()
    return scene, robot


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_fixed_tendon_matches_mujoco(tmp_path):
    # Tendon actuation and the passive spring involve no constraints. With matched integrator, gravity and (disabled)
    # default armature, and a well-damped (non-oscillatory) regime, Genesis tracks MuJoCo closely.
    xml_path = str(tmp_path / "tendon_smooth.xml")
    with open(xml_path, "w") as f:
        f.write(TENDON_SMOOTH_XML)

    n_steps, ctrl = 60, 0.6
    joints = ("hingeA", "slideA")

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mj_model.opt.timestep = 0.002
    mj_data = mujoco.MjData(mj_model)
    mj_data.ctrl[mj_model.actuator("driveA_motor").id] = ctrl
    for _ in range(n_steps):
        mujoco.mj_step(mj_model, mj_data)
    mj_qpos = {j: mj_data.qpos[mj_model.jnt_qposadr[mj_model.joint(j).id]] for j in joints}
    mj_drive_length = float(mj_data.ten_length[mj_model.tendon("driveA").id])

    scene, robot = _build_smooth_scene(xml_path, integrator=gs.integrator.Euler)
    assert robot.n_tendons == 2
    drive = robot.get_tendon("driveA")
    for _ in range(n_steps):
        drive.control_force(np.array([ctrl]))
        scene.step()
    gs_qpos = {j: float(robot.get_dofs_position([robot.get_joint(j).dof_start])[0]) for j in joints}

    # The actuated joints are undamped (no constraints involved), so Genesis tracks MuJoCo tightly.
    for j in joints:
        assert_allclose(gs_qpos[j], mj_qpos[j], tol=1e-4)

    # The reported tendon length matches MuJoCo. Like MuJoCo, it reflects the configuration at the start of the last
    # step (where the dynamics is evaluated), so it is compared against MuJoCo's tendon length rather than the
    # post-step joint positions.
    assert_allclose(float(drive.get_length()), mj_drive_length, tol=1e-4)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_fixed_tendon_passive_spring(tmp_path):
    # A damped passive spring tendon (rest length 0.3, single-joint coefficient 1.0) settles its joint at the resting
    # length, regardless of the integrator.
    xml_path = str(tmp_path / "tendon_smooth.xml")
    with open(xml_path, "w") as f:
        f.write(TENDON_SMOOTH_XML)

    scene, robot = _build_smooth_scene(xml_path)
    spring = robot.get_tendon("springB")
    hinge_b = robot.get_joint("hingeB")
    for _ in range(500):
        scene.step()

    assert_allclose(float(robot.get_dofs_position([hinge_b.dof_start])[0]), 0.3, tol=1e-3)
    assert_allclose(float(spring.get_length()), 0.3, tol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_fixed_tendon_force_equals_per_dof_force(tmp_path):
    # A fixed tendon's effect must be exactly equivalent to applying 'coef_i * force' to each coupled DOF. This is
    # integrator-independent and is the defining property of a fixed-tendon transmission.
    xml_path = str(tmp_path / "tendon_smooth.xml")
    with open(xml_path, "w") as f:
        f.write(TENDON_SMOOTH_XML)

    ctrl = 0.6

    scene, robot = _build_smooth_scene(xml_path)
    drive = robot.get_tendon("driveA")
    for _ in range(80):
        drive.control_force(np.array([ctrl]))
        scene.step()
    tendon_qpos = robot.get_dofs_position()

    scene, robot = _build_smooth_scene(xml_path)
    hinge_a = robot.get_joint("hingeA").dof_start
    slide_a = robot.get_joint("slideA").dof_start
    for _ in range(80):
        # driveA couples hingeA (coef 2) and slideA (coef 1).
        robot.control_dofs_force(np.array([2.0 * ctrl, 1.0 * ctrl]), [hinge_a, slide_a])
        scene.step()
    direct_qpos = robot.get_dofs_position()

    assert_allclose(tendon_qpos, direct_qpos, tol=1e-9)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_fixed_tendon_differential_drive(tmp_path):
    # The issue's robot: each side has two tendons (r + d) and (r - d). Equal motor forces drive the wheel rotation
    # only (the slide contributions cancel); opposite forces drive the slide only.
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/tendon_diff_drive.xml"),
    )
    scene.build()

    left_r = robot.get_joint("left_r")
    left_d = robot.get_joint("left_d")
    m1, m2 = robot.get_tendon("left_m1"), robot.get_tendon("left_m2")

    # Equal forces: wheel spins, slide stays put.
    for _ in range(100):
        m1.control_force(np.array([0.2]))
        m2.control_force(np.array([0.2]))
        scene.step()
    r_equal = float(robot.get_dofs_position([left_r.dof_start])[0])
    d_equal = float(robot.get_dofs_position([left_d.dof_start])[0])
    assert abs(r_equal) > 0.5
    assert abs(d_equal) < 1e-3

    # Opposite forces: slide moves.
    scene.reset()
    for _ in range(50):
        m1.control_force(np.array([0.2]))
        m2.control_force(np.array([-0.2]))
        scene.step()
    d_opposite = float(robot.get_dofs_position([left_d.dof_start])[0])
    assert abs(d_opposite) > 1e-2


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu])
def test_fixed_tendon_length_limit(tmp_path):
    # A monotonically driven tendon must be arrested by its length limit.
    xml_path = str(tmp_path / "tendon_limit.xml")
    with open(xml_path, "w") as f:
        f.write(TENDON_LIMIT_XML)

    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.005,
            gravity=(0.0, 0.0, 0.0),
            constraint_solver=gs.constraint_solver.Newton,
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file=xml_path, default_armature=0.0),
    )
    scene.build()

    capped = robot.get_tendon("capped")
    for _ in range(300):
        capped.control_force(np.array([2.0]))
        scene.step()

    # Tendon length stays within the upper bound (small solver softness tolerance).
    assert float(capped.get_length()) < 0.3 + 5e-3
