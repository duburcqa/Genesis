"""The couplings of the surface of the finite element method (FEM) entities with the other solvers, and its floor."""

from quadrants.lang._quadrants_callable import QuadrantsCallable

from genesis.engine.pipeline import Inline, Stages, System
from genesis.engine.solvers.fem_solver import (
    FEMSolver,
    func_fem_surface_couplings,
    kernel_fem_surface,
    kernel_fem_surface_boundary,
)
from genesis.engine.step import Coupling
from genesis.options.solvers import LegacyCouplerOptions


class FEMSurface(Coupling):
    """
    Couple the surface of the finite element method (FEM) entities with the other solvers, then impose the floor on it.

    The couplings of the surface vertices are the funcs of the Systems playing SurfaceCoupling.
    """

    surface_couplings = Inline(kernel_fem_surface, func_fem_surface_couplings, run_without_funcs=False)

    def __init__(self, fem_solver: FEMSolver, options: LegacyCouplerOptions) -> None:
        pass

    @property
    def couple(self) -> tuple[Inline, QuadrantsCallable]:
        return self.surface_couplings, kernel_fem_surface_boundary


class SurfaceCoupling(System):
    """A System coupling the velocity of the surface vertices of the FEM entities with another solver."""

    slot = FEMSurface.surface_couplings
    surface_coupling: Stages
