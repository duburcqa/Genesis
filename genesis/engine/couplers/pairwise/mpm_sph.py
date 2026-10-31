"""The coupling of the grid of the material point method (MPM) solver with the smoothed particle hydrodynamics (SPH)."""

from typing import Annotated

import numpy as np

import quadrants as qd

import genesis.utils.array_class as array_class
from genesis.engine.pipeline import Enabled, register_system
from genesis.engine.solvers.mpm_solver import GridCoupling, MPMSolver
from genesis.engine.solvers.particle_solver import func_add_particles_momentum, func_get_particles_mean_vel
from genesis.engine.solvers.sph_solver import SPHSolver
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.data import Data, DataKind, field


class MPMSPHInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the coupling of the material point method (MPM) grid with the smoothed particle hydrodynamics
    (SPH) particles.

    A cell exchanges momentum with the particles of the cube of `stencil_size` cells of the hash grid of the SPH solver
    around its lower corner.
    """

    stencil_size: qd.Tensor = field(int, shape=())


@qd.func
def func_mpm_sph(
    f: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    sph_state: array_class.SPHState,
    mpm_info: array_class.MPMInfo,
    mpm_sph_info: MPMSPHInfo,
) -> qd.types.vector(3):
    """
    Return the velocity of a cell of the grid set to the mean velocity of the SPH particles in it, if any.

    The particles take the opposite of the momentum the cell gains, divided among them.
    """
    half_size = 0.5 * mpm_info.dx[None]
    stencil_size = mpm_sph_info.stencil_size[None]
    sph_vel, n_particles = func_get_particles_mean_vel(
        i_b,
        pos,
        sph_state.hash_grid,
        sph_state.particles_reordered.pos,
        sph_state.particles_reordered.vel,
        half_size=half_size,
        n_cells=stencil_size,
    )
    if n_particles > 0:
        func_add_particles_momentum(
            i_b,
            pos,
            mass * (vel - sph_vel),
            sph_state.hash_grid,
            sph_state.particles_reordered.pos,
            sph_state.particles_reordered.vel,
            sph_state.particles_info_reordered.mass,
            sph_state.particles_reordered.is_active,
            half_size=half_size,
            n_cells=stencil_size,
        )
        vel = sph_vel
    return vel


@register_system
class MPMSPH(GridCoupling):
    """Exchange momentum between the cells of the material point method grid and the smoothed particles in them."""

    grid_coupling = func_mpm_sph

    def __init__(
        self, mpm_solver: MPMSolver, sph_solver: SPHSolver, options: Annotated[LegacyCouplerOptions, Enabled("mpm_sph")]
    ) -> None:
        # A cell spans the cells of the hash grid closer than its size to its lower corner
        self.mpm_sph_info = MPMSPHInfo(
            values=dict(stencil_size=int(np.floor(mpm_solver.dx / sph_solver.hash_grid_cell_size) + 2))
        )
