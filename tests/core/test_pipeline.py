import pytest

import quadrants as qd

import genesis as gs
import genesis.engine.pipeline as pipeline
from genesis.engine.pipeline import System
from genesis.engine.step import Step
from genesis.utils.array_class import DynState
from genesis.utils.data import V
from genesis.utils.misc import qd_to_numpy

from ..utils.assertions import assert_allclose


@pytest.mark.required
def test_data_dependency_is_read_only(monkeypatch, show_viewer):
    @qd.kernel
    def kernel_record_altitude(dyn_state: DynState, altitude: qd.Tensor):
        for i_b in range(dyn_state.links.pos.shape[1]):
            altitude[i_b] = dyn_state.links.pos[0, i_b][2]

    @qd.kernel
    def kernel_lift(dyn_state: DynState):
        for i_b in range(dyn_state.links.pos.shape[1]):
            dyn_state.links.pos[0, i_b][2] += 1.0

    class AltitudeRecorder(System):
        def __init__(self) -> None:
            self.altitude = None

        def build(self, dyn_state: DynState) -> None:
            self.altitude = V(dtype=gs.qd_float, shape=(dyn_state.links.pos.shape[1],))

        def compose(self, step: Step, dyn_state: DynState) -> None:
            # A kernel stage is fused from the source of its module, so a kernel defined here runs from a Python stage
            def record_altitude(dyn_state, altitude):
                kernel_record_altitude(dyn_state, altitude)

            step.integrations.add(record_altitude)

    class Lifter(System):
        def compose(self, step: Step, dyn_state: DynState) -> None:
            step.integrations.add(kernel_lift)

    systems_registered = pipeline._SYSTEM_CLASSES
    for system_cls, is_refused in ((AltitudeRecorder, False), (Lifter, True)):
        monkeypatch.setattr(pipeline, "_SYSTEM_CLASSES", [*systems_registered, system_cls])
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
