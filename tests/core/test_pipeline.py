import functools

import pytest

import quadrants as qd

import genesis as gs
import genesis.engine.pipeline as pipeline
from genesis.engine.pipeline import System, assemble_systems, create_system, register_system
from genesis.engine.solvers.rigid.collider import Collider
from genesis.engine.solvers.rigid.collider.collider import (
    ConvexConvex,
    ConvexConvexFirstContact,
    ConvexConvexSDF,
    NonconvexNonconvex,
    NonconvexNonconvexSDF,
)
from genesis.engine.solvers.rigid.collider.narrowphase import func_narrow_phase_nonconvex_vs_nonterrain
from genesis.engine.solvers.rigid.data import DynState
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.step import Integration
from genesis.utils.data import V
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


@pytest.mark.required
def test_data_written_by_owner_alone(request, show_viewer):
    @qd.kernel
    def kernel_record_altitude(dyn_state: DynState, altitude: qd.Tensor):
        for i_b in range(dyn_state.links.pos.shape[1]):
            altitude[i_b] = dyn_state.links.pos[0, i_b][2]

    @qd.kernel
    def kernel_lift(dyn_state: DynState):
        for i_b in range(dyn_state.links.pos.shape[1]):
            dyn_state.links.pos[0, i_b][2] += 1.0

    class AltitudeRecorder(Integration):
        def __init__(self, dyn_state: DynState) -> None:
            self.dyn_state = dyn_state
            self.altitude = None

        def build(self) -> None:
            self.altitude = V(dtype=gs.qd_float, shape=(self.dyn_state.links.pos.shape[1],))

        # A kernel stage is fused from the source of its module, so a kernel defined here runs from a Python stage
        def substep_post_coupling(self, f: int) -> None:
            kernel_record_altitude(self.dyn_state, self.altitude)

    class Lifter(Integration):
        substep_post_coupling = kernel_lift

        def __init__(self, dyn_state: DynState) -> None:
            pass

    class SolverLifter(Integration):
        substep_post_coupling = kernel_lift

        def __init__(self, rigid_solver: RigidSolver) -> None:
            pass

    for system_cls, is_refused in ((AltitudeRecorder, False), (Lifter, True), (SolverLifter, True)):
        # Registered alongside the built-in Systems, which the first scene of a process registers as it imports them
        register_system(system_cls)
        request.addfinalizer(functools.partial(pipeline._SYSTEM_CLASSES.remove, system_cls))
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity=(0.0, 0.0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -1.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
            ),
            show_viewer=show_viewer,
        )
        box = scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.0, 0.0, 1.0),
            ),
        )
        if is_refused:
            with pytest.raises(TypeError, match="writes 'dyn_state'"):
                scene.build(n_envs=2)
            continue
        scene.build(n_envs=2)
        for _ in range(10):
            scene.step()
        recorder = scene.sim.get_system(AltitudeRecorder)
        assert (box.get_pos()[:, 2] < 1.0).all()
        assert_allclose(qd_to_numpy(recorder.altitude), box.get_pos()[:, 2], tol=gs.EPS)


def test_template_instances_are_distinct_systems(monkeypatch):
    class Layout: ...

    class PerEnv(Layout): ...

    class Queued(Layout): ...

    # A component: playing no role, registered by none, it exists once a System depends on it
    class Scratch[LayoutT: Layout](System): ...

    class Pass[LayoutT: Layout](System):
        def __init__(self, scratch: Scratch[LayoutT]) -> None:
            self.scratch = scratch

    assert Scratch[Queued] is Scratch[Queued] and Scratch[Queued] is not Scratch[PerEnv]
    assert list(Scratch[Queued].template_args.values()) == [Queued]
    with pytest.raises(TypeError, match="takes the arguments"):
        Scratch[Queued, PerEnv]

    monkeypatch.setattr(pipeline, "_SYSTEM_CLASSES", [Pass[PerEnv], Pass[Queued]])
    systems = assemble_systems([], [])
    systems_by_cls = {type(system): system for system in systems}
    assert set(systems_by_cls) == {Scratch[PerEnv], Scratch[Queued], Pass[PerEnv], Pass[Queued]}
    assert systems_by_cls[Pass[PerEnv]].scratch is systems_by_cls[Scratch[PerEnv]]
    assert systems_by_cls[Pass[Queued]].scratch is systems_by_cls[Scratch[Queued]]


@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_system_superseded_by_subclass(monkeypatch, show_viewer):
    class OtherNonconvexNonconvex(NonconvexNonconvex):
        detect = func_narrow_phase_nonconvex_vs_nonterrain

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Sphere(radius=0.05))
    scene.sim.get_system(NonconvexNonconvexSDF, opt_in=True)
    scene.sim.get_system(OtherNonconvexNonconvex, opt_in=True)
    with pytest.raises(gs.GenesisException, match="all supersede NonconvexNonconvex"):
        scene.build()

    # A requested subclass of a registered narrowphase replaces it in the detection, whichever layout the collider
    # picks for the convex pairs, and finds nothing in a scene without pairs of its family
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -0.5, 0.3),
            camera_lookat=(0.0, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.05,
            pos=(0.0, 0.0, 0.05),
        ),
    )
    convex_sdf = scene.sim.get_system(ConvexConvexSDF, opt_in=True)
    nonconvex_sdf = scene.sim.get_system(NonconvexNonconvexSDF, opt_in=True)
    scene.build(n_envs=2)
    collider = scene.sim.get_system(Collider)
    assert scene.sim.get_system(ConvexConvex[collider.narrowphase_layout]) is convex_sdf
    assert scene.sim.get_system(ConvexConvexFirstContact) is convex_sdf
    assert scene.sim.get_system(NonconvexNonconvex) is nonconvex_sdf
    # A func stage runs through the kernel generated around it, named after it
    narrowphase_names = [stage.fn.__name__ for stage, _ in collider.detection.slots["narrowphase"].items]
    assert "kernel_narrow_phase_convex_vs_convex_sdf" in narrowphase_names
    assert "kernel_narrow_phase_nonconvex_vs_nonterrain_sdf" in narrowphase_names
    assert "kernel_narrow_phase_nonconvex_vs_nonterrain" not in narrowphase_names
    for _ in range(10):
        scene.step()
    assert_allclose(sphere.get_pos()[:, 2], 0.05, atol=1e-3)
    with pytest.raises(gs.GenesisException, match="requested once the scene is built"):
        scene.sim.get_system(ConvexConvexSDF, opt_in=True)

    class Base(System): ...

    class Custom(Base): ...

    class Other(Base): ...

    class User(System):
        def __init__(self, base: Base) -> None:
            self.base = base

    monkeypatch.setattr(pipeline, "_SYSTEM_CLASSES", [Base, Custom, User])
    systems = assemble_systems([], [])
    assert {type(system) for system in systems} == {Custom, User}
    (user,) = [system for system in systems if isinstance(system, User)]
    assert type(user.base) is Custom

    # A request before the assembly creates the dependency it misses, which the assembly then uses
    custom, user = create_system(User, [])
    assert type(custom) is Custom and user.base is custom
    systems = assemble_systems([custom, user], [])
    assert len(systems) == 2
    (custom,) = create_system(Base, [])
    assert type(custom) is Custom
    assert {type(system) for system in assemble_systems([custom], [])} == {Custom, User}

    monkeypatch.setattr(pipeline, "_SYSTEM_CLASSES", [Base, Custom, Other])
    with pytest.raises(gs.GenesisException, match="all supersede Base"):
        assemble_systems([], [])

    # Two Systems given of distinct subclasses of a registered class conflict as well
    monkeypatch.setattr(pipeline, "_SYSTEM_CLASSES", [Base])
    with pytest.raises(gs.GenesisException, match="all supersede Base"):
        assemble_systems([Custom(), Other()], [])
