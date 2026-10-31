"""The coupling of the grid of the material point method (MPM) solver with the position based dynamics (PBD)."""

from typing import Annotated

import numpy as np

import quadrants as qd

import genesis.utils.array_class as array_class
from genesis.engine.pipeline import Enabled, register_system
from genesis.engine.solvers.mpm_solver import GridCoupling, MPMSolver
from genesis.engine.solvers.particle_solver import func_add_particles_momentum, func_get_particles_mean_vel
from genesis.engine.solvers.pbd_solver import PBDSolver
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.data import Data, DataKind, field


class MPMPBDInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the coupling of the material point method (MPM) grid with the position based dynamics (PBD)
    particles.

    A cell exchanges momentum with the particles of the cube of `stencil_size` cells of the hash grid of the PBD solver
    around its lower corner.
    """

    stencil_size: qd.Tensor = field(int, shape=())


@qd.func
def func_mpm_pbd(
    f: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    pbd_state: array_class.PBDState,
    mpm_info: array_class.MPMInfo,
    mpm_pbd_info: MPMPBDInfo,
) -> qd.types.vector(3):
    """
    Return the velocity of a cell of the grid set to the mean velocity of the PBD particles in it, if any.

    The free PBD particles take the opposite of the momentum the cell gains, divided among them.
    """
    half_size = 0.5 * mpm_info.dx[None]
    stencil_size = mpm_pbd_info.stencil_size[None]
    pbd_vel, n_particles = func_get_particles_mean_vel(
        i_b,
        pos,
        pbd_state.hash_grid,
        pbd_state.particles_reordered.pos,
        pbd_state.particles_reordered.vel,
        half_size=half_size,
        n_cells=stencil_size,
    )
    if n_particles > 0:
        func_add_particles_momentum(
            i_b,
            pos,
            mass * (vel - pbd_vel),
            pbd_state.hash_grid,
            pbd_state.particles_reordered.pos,
            pbd_state.particles_reordered.vel,
            pbd_state.particles_info_reordered.mass,
            pbd_state.particles_reordered.free,
            half_size=half_size,
            n_cells=stencil_size,
        )
        vel = pbd_vel
    return vel


@register_system
class MPMPBD(GridCoupling):
    """Exchange momentum between the cells of the material point method grid and the free PBD particles in them."""

    grid_coupling = func_mpm_pbd

    def __init__(
        self, mpm_solver: MPMSolver, pbd_solver: PBDSolver, options: Annotated[LegacyCouplerOptions, Enabled("mpm_pbd")]
    ) -> None:
        # A cell spans the cells of the hash grid closer than its size to its lower corner
        self.mpm_pbd_info = MPMPBDInfo(
            values=dict(stencil_size=int(np.floor(mpm_solver.dx / pbd_solver.hash_grid_cell_size) + 2))
        )
