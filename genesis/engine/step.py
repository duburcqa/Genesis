"""The step of a scene, the phases it runs, and the roles the Systems of the scene play in them."""

from genesis.engine.pipeline import PhaseSpec, System, register_system


@register_system
class Step(System):
    """
    One step of the scene: its substeps, then the completions of the step and the observations of its outcome.

    A substep runs the preparations, the dynamics of every solver, the couplings, then the integrations. A solver runs
    its dynamics before every coupling, which reads the dynamics of all of them, and its integration after them, which
    applies the couplings. A stage of a substep takes the frame `f` and the time `t` it starts at as call arguments.
    """

    substep = PhaseSpec("preparations", "dynamics", "couplings", "integrations")
    completions = PhaseSpec()
    observations = PhaseSpec()


class Preparation(System):
    """A System preparing every substep, before any solver runs its dynamics."""

    slot = Step.substep.preparations
    entry = "preprocess"


class Dynamics(System):
    """A solver running its dynamics in every substep, before the couplings read them."""

    slot = Step.substep.dynamics
    entry = "substep_pre_coupling"


class Coupling(System):
    """A System coupling solvers in every substep, between their dynamics and their integration."""

    slot = Step.substep.couplings
    entry = "couple"


class Integration(System):
    """A solver integrating in every substep, after the couplings it applies."""

    slot = Step.substep.integrations
    entry = "substep_post_coupling"


class Completion(System):
    """A System completing every step, once its substeps are over."""

    slot = Step.completions
    entry = "complete"


class Observation(System):
    """A System observing the state every step leaves, outside of the autograd graph of a differentiable rollout."""

    slot = Step.observations
    entry = "observe"
