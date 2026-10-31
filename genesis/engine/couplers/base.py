from genesis.engine.pipeline import System


class Coupler(System):
    """
    A System solving the interactions of several solvers at once, which the solvers it couples arrange their substep
    around.
    """

    @property
    def is_solving_rigid_constraints(self) -> bool:
        """Whether the coupler solves the constraints of the rigid entities, between their dynamics and integration."""
        return False

    @property
    def is_simulating_rigid_first(self) -> bool:
        """Whether the coupler simulates the rigid entities it couples before the rigid solver runs its substep."""
        return False
