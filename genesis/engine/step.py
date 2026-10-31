"""The dispatcher of a step, whose phases the solvers, the couplers and the sensors of a scene add their stages to."""

from genesis.engine.pipeline import Phase, System, register_system


@register_system
class Step(System):
    """
    Dispatcher of one step: its substeps, then the completions of the step and the observations of its outcome.

    A substep runs the preparations, the dynamics of every solver, the couplings, then the integrations. A solver runs
    its dynamics before every coupling, which reads the dynamics of all of them, and its integration after them, which
    applies the couplings. A stage of these phases takes the frame of the substep `f` and the time `t` it starts at as
    call arguments. The completions run once the substeps are over, and the observations read the state they left.
    """

    def __init__(self) -> None:
        self.preparations = Phase()
        self.dynamics = Phase()
        self.couplings = Phase()
        self.integrations = Phase()
        self.completions = Phase()
        self.observations = Phase()
        # The phases of a substep run as one, whose launches are those of the stages alone (see Phase)
        self.substep = Phase()

    def compose(self) -> None:
        self.substep.add(self.preparations, self.dynamics, self.couplings, self.integrations)
