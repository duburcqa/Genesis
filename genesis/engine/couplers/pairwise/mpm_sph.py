"""The coupling of the grid of the material point method (MPM) solver with the smoothed particle hydrodynamics (SPH)."""

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.pipeline import System, register_system
from genesis.engine.solvers.mpm_solver import MPMSolver
from genesis.engine.solvers.sph_solver import SPHSolver
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.data import Data, DataKind, field
from genesis.utils.geom import func_hash_grid_to_slot, func_hash_pos_to_grid


class MPMSPHInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the coupling of the material point method (MPM) grid with the smoothed particle hydrodynamics
    (SPH) particles.

    A cell exchanges momentum with the particles of the cube of `stencil_size` cells of the hash grid of the SPH solver
    around its lower corner.
    """

    stencil_size: qd.Tensor = field(int, shape=())


@register_system
class MPMSPH(System):
    """Exchange momentum between the cells of the material point method grid and the smoothed particles in them."""

    def __init__(self, mpm_solver: MPMSolver, sph_solver: SPHSolver, options: LegacyCouplerOptions) -> None:
        self._mpm_solver = mpm_solver
        # A cell spans the cells of the hash grid closer than its size to its lower corner
        self.mpm_sph_info = MPMSPHInfo(
            values=dict(stencil_size=int(np.floor(mpm_solver.dx / sph_solver.hash_grid_cell_size) + 2))
        )

    @classmethod
    def create(cls, mpm_solver: MPMSolver, sph_solver: SPHSolver, options: LegacyCouplerOptions) -> "MPMSPH | None":
        return cls(mpm_solver, sph_solver, options) if options.mpm_sph else None

    def compose(self) -> None:
        self._mpm_solver.grid_couplings.add(func_mpm_sph)


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
    dx = mpm_info.dx[None]
    stencil_size = mpm_sph_info.stencil_size[None]
    base = func_hash_pos_to_grid(pos - 0.5 * dx, sph_state.hash_grid)

    # SPH -> MPM
    sph_vel = qd.Vector.zero(gs.qd_float, 3)
    n_colliding_particles = 0
    for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
        slot_idx = func_hash_grid_to_slot(base + offset, sph_state.hash_grid)
        slot_start = sph_state.hash_grid.slot_start[slot_idx, i_b]
        for i in range(slot_start, slot_start + sph_state.hash_grid.slot_size[slot_idx, i_b]):
            if qd.abs(pos - sph_state.particles_reordered.pos[i, i_b]).max() < dx * 0.5:
                sph_vel += sph_state.particles_reordered.vel[i, i_b]
                n_colliding_particles += 1
    if n_colliding_particles > 0:
        vel_old = vel
        vel = sph_vel / n_colliding_particles

        # MPM -> SPH
        delta_mv = mass * (vel - vel_old)
        for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
            slot_idx = func_hash_grid_to_slot(base + offset, sph_state.hash_grid)
            slot_start = sph_state.hash_grid.slot_start[slot_idx, i_b]
            for i in range(slot_start, slot_start + sph_state.hash_grid.slot_size[slot_idx, i_b]):
                if qd.abs(pos - sph_state.particles_reordered.pos[i, i_b]).max() < dx * 0.5:
                    sph_state.particles_reordered.vel[i, i_b] = (
                        sph_state.particles_reordered.vel[i, i_b]
                        - delta_mv / sph_state.particles_info_reordered.mass[i, i_b]
                    )
    return vel
