"""The coupling of the finite element method (FEM) surface with the smoothed particle hydrodynamics (SPH) particles."""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.pipeline import System, register_system
from genesis.engine.solvers.sph_solver import SPHSolver
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.geom import func_hash_grid_to_slot, func_hash_pos_to_grid

from .fem_surface import FEMSurface


@register_system
class FEMSPH(System):
    """
    Set the normal velocity of the surface vertices of the FEM entities to that of the SPH particles around them.

    The particles take the opposite of the momentum the vertices gain, divided among them.
    """

    is_differentiable = False

    def __init__(self, fem_surface: FEMSurface, sph_solver: SPHSolver, options: LegacyCouplerOptions) -> None:
        self._fem_surface = fem_surface

    @classmethod
    def create(cls, fem_surface: FEMSurface, sph_solver: SPHSolver, options: LegacyCouplerOptions) -> "FEMSPH | None":
        return cls(fem_surface, sph_solver, options) if options.fem_sph else None

    def compose(self) -> None:
        self._fem_surface.surface_couplings.add(func_fem_sph)


@qd.func
def func_fem_sph(
    f: int,
    i_v: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    surface_normal: qd.types.vector(3),
    sph_state: array_class.SPHState,
) -> qd.types.vector(3):
    """
    Return the velocity of a surface vertex whose normal velocity is set to the mean velocity of the SPH particles of the
    cells of the hash grid around it, if any, the tangential velocity being left out.
    """
    dx = sph_state.hash_grid.cell_size[None]
    mass_fem_sv = mass / 4.0
    base = func_hash_pos_to_grid(pos - 0.5 * dx, sph_state.hash_grid)

    # SPH -> FEM
    sph_vel = qd.Vector.zero(gs.qd_float, 3)
    n_colliding_particles = 0
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        slot_idx = func_hash_grid_to_slot(base + offset, sph_state.hash_grid)
        slot_start = sph_state.hash_grid.slot_start[slot_idx, i_b]
        for k in range(slot_start, slot_start + sph_state.hash_grid.slot_size[slot_idx, i_b]):
            if qd.abs(pos - sph_state.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                sph_vel += sph_state.particles_reordered.vel[k, i_b]
                n_colliding_particles += 1

    new_vel = vel
    if n_colliding_particles > 0:
        new_vel = (sph_vel / n_colliding_particles).dot(surface_normal) * surface_normal

        # FEM -> SPH
        delta_mv = mass_fem_sv * (new_vel - vel)
        for offset in qd.grouped(qd.ndrange(2, 2, 2)):
            slot_idx = func_hash_grid_to_slot(base + offset, sph_state.hash_grid)
            slot_start = sph_state.hash_grid.slot_start[slot_idx, i_b]
            for k in range(slot_start, slot_start + sph_state.hash_grid.slot_size[slot_idx, i_b]):
                if qd.abs(pos - sph_state.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                    sph_state.particles_reordered.vel[k, i_b] = (
                        sph_state.particles_reordered.vel[k, i_b]
                        - delta_mv / sph_state.particles_info_reordered.mass[k, i_b]
                    )
    return new_vel
