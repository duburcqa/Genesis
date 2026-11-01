"""
The couplings of the grid of the material point method (MPM) solver with the particles of the smoothed particle
hydrodynamics (SPH) and position based dynamics (PBD) solvers.
"""

from typing import Annotated

import numpy as np

import quadrants as qd

from genesis.engine.pipeline import Enabled, register_system
from genesis.engine.solvers.mpm_solver import GridCoupling, MPMInfo, MPMSolver
from genesis.engine.solvers.particle_solver import func_add_particles_momentum, func_get_particles_mean_vel
from genesis.engine.solvers.pbd_solver import PBDSolver, PBDState
from genesis.engine.solvers.sph_solver import SPHSolver, SPHState
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.data import Data, DataKind, field
from genesis.utils.geom import SpatialHashGrid


class MPMParticlesInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the coupling of the material point method (MPM) grid with the particles of another solver.

    A cell exchanges momentum with the particles of the cube of `stencil_size` cells of the hash grid of the particle
    solver around its lower corner, which spans the cells closer than the size of a cell of the grid to that corner.
    """

    stencil_size: qd.Tensor = field(int, shape=())


@qd.func
def func_mpm_particles(
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    hash_grid: SpatialHashGrid,
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_mass: qd.Tensor,
    particles_is_moved: qd.Tensor,
    mpm_info: MPMInfo,
    mpm_particles_info: MPMParticlesInfo,
) -> qd.types.vector(3):
    """
    Return the velocity of a cell of the grid set to the mean velocity of the sorted particles of a solver in it if any.

    The particles that `particles_is_moved` selects take the opposite of the momentum the cell gains, shared among them.
    """
    half_size = 0.5 * mpm_info.dx[None]
    stencil_size = mpm_particles_info.stencil_size[None]
    particles_mean_vel, n_particles = func_get_particles_mean_vel(
        i_b, pos, hash_grid, particles_pos, particles_vel, half_size=half_size, n_cells=stencil_size
    )
    if n_particles > 0:
        func_add_particles_momentum(
            i_b,
            pos,
            mass * (vel - particles_mean_vel),
            hash_grid,
            particles_pos,
            particles_vel,
            particles_mass,
            particles_is_moved,
            half_size=half_size,
            n_cells=stencil_size,
        )
        vel = particles_mean_vel
    return vel


# The couplings take the state of their particle solver under its own name, which binds it (see 'func_mpm_particles')
@qd.func
def func_mpm_sph(
    f: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    sph_state: SPHState,
    mpm_info: MPMInfo,
    mpm_particles_info: MPMParticlesInfo,
) -> qd.types.vector(3):
    return func_mpm_particles(
        i_b,
        pos,
        vel,
        mass,
        sph_state.hash_grid,
        sph_state.particles_reordered.pos,
        sph_state.particles_reordered.vel,
        sph_state.particles_info_reordered.mass,
        sph_state.particles_reordered.is_active,
        mpm_info,
        mpm_particles_info,
    )


@qd.func
def func_mpm_pbd(
    f: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    pbd_state: PBDState,
    mpm_info: MPMInfo,
    mpm_particles_info: MPMParticlesInfo,
) -> qd.types.vector(3):
    return func_mpm_particles(
        i_b,
        pos,
        vel,
        mass,
        pbd_state.hash_grid,
        pbd_state.particles_reordered.pos,
        pbd_state.particles_reordered.vel,
        pbd_state.particles_info_reordered.mass,
        pbd_state.particles_reordered.free,
        mpm_info,
        mpm_particles_info,
    )


@register_system
class MPMSPH(GridCoupling):
    """
    Exchange momentum between the cells of the material point method grid and the smoothed particles in them.

    See func_mpm_particles.
    """

    grid_coupling = func_mpm_sph

    mpm_particles_info: MPMParticlesInfo

    def __init__(
        self, mpm_solver: MPMSolver, sph_solver: SPHSolver, options: Annotated[LegacyCouplerOptions, Enabled("mpm_sph")]
    ) -> None:
        self.mpm_particles_info = MPMParticlesInfo(
            values=dict(stencil_size=int(np.floor(mpm_solver.dx / sph_solver.hash_grid_cell_size) + 2))
        )


@register_system
class MPMPBD(GridCoupling):
    """
    Exchange momentum between the cells of the material point method grid and the free PBD particles in them.

    See func_mpm_particles.
    """

    grid_coupling = func_mpm_pbd

    mpm_particles_info: MPMParticlesInfo

    def __init__(
        self, mpm_solver: MPMSolver, pbd_solver: PBDSolver, options: Annotated[LegacyCouplerOptions, Enabled("mpm_pbd")]
    ) -> None:
        self.mpm_particles_info = MPMParticlesInfo(
            values=dict(stencil_size=int(np.floor(mpm_solver.dx / pbd_solver.hash_grid_cell_size) + 2))
        )
