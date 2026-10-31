"""The coupling of the finite element method (FEM) surface with the smoothed particle hydrodynamics (SPH) particles."""

from typing import Annotated

import quadrants as qd

from genesis.engine.pipeline import Enabled, register_system
from genesis.engine.solvers.particle_solver import func_add_particles_momentum, func_get_particles_mean_vel
from genesis.engine.solvers.sph_solver import SPHSolver, SPHState
from genesis.options.solvers import LegacyCouplerOptions

from .fem_surface import FEMSurface, SurfaceCoupling


@qd.func
def func_fem_sph(
    f: int,
    i_v: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    surface_normal: qd.types.vector(3),
    sph_state: SPHState,
) -> qd.types.vector(3):
    """
    Return the velocity of a surface vertex whose normal velocity is set to the mean velocity of the SPH particles of the
    cells of the hash grid around it, if any, the tangential velocity being left out.
    """
    half_size = 0.5 * sph_state.hash_grid.cell_size[None]
    sph_vel, n_particles = func_get_particles_mean_vel(
        i_b,
        pos,
        sph_state.hash_grid,
        sph_state.particles_reordered.pos,
        sph_state.particles_reordered.vel,
        half_size=half_size,
        n_cells=2,
    )
    new_vel = vel
    if n_particles > 0:
        new_vel = sph_vel.dot(surface_normal) * surface_normal
        func_add_particles_momentum(
            i_b,
            pos,
            0.25 * mass * (vel - new_vel),
            sph_state.hash_grid,
            sph_state.particles_reordered.pos,
            sph_state.particles_reordered.vel,
            sph_state.particles_info_reordered.mass,
            sph_state.particles_reordered.is_active,
            half_size=half_size,
            n_cells=2,
        )
    return new_vel


@register_system
class FEMSPH(SurfaceCoupling):
    """
    Set the normal velocity of the surface vertices of the FEM entities to that of the SPH particles around them.

    The particles take the opposite of the momentum the vertices gain, divided among them.
    """

    is_differentiable = False
    surface_coupling = func_fem_sph

    def __init__(
        self,
        fem_surface: FEMSurface,
        sph_solver: SPHSolver,
        options: Annotated[LegacyCouplerOptions, Enabled("fem_sph")],
    ) -> None:
        pass
