"""The coupler of a scene, which solves the interactions of several solvers at once."""

from genesis.engine.pipeline import System


class Coupler(System):
    """
    A System solving the interactions of several solvers at once, which the solvers it couples arrange their substep
    around.
    """

    # Whether the coupler computes the dynamics of the FEM entities itself
    is_simulating_fem: bool = False

    @property
    def is_using_rigid_coup_type(self) -> bool:
        """Whether the coupler couples every rigid entity the way the `coup_type` of its material sets."""
        return False

    @property
    def rigid_links_idx_simulated(self) -> set[int]:
        """Return the indices of the rigid links the coupler simulates alone, which take part in no rigid contact."""
        return set()

    @property
    def rigid_links_idx_contacting(self) -> set[int]:
        """Return the indices of the rigid links whose contacts with each other the coupler resolves."""
        return set()
