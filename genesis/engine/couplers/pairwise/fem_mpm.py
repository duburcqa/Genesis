"""The coupling of the finite element method (FEM) surface with the grid of the material point method (MPM) solver."""

from typing import Annotated

import quadrants as qd

import genesis as gs
from genesis.engine.pipeline import Enabled
from genesis.engine.solvers.fem_solver import FEMInfo
from genesis.engine.solvers.mpm_solver import (
    MPMInfo,
    MPMSolver,
    MPMState,
    func_add_grid_vel,
    func_quadratic_bspline_stencil,
)
from genesis.options.solvers import LegacyCouplerOptions

from .fem_surface import FEMSurface, SurfaceCoupling


@qd.func
def func_fem_mpm(
    f: int,
    i_v: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    surface_normal: qd.types.vector(3),
    mpm_state: MPMState,
    fem_info: FEMInfo,
    mpm_info: MPMInfo,
) -> qd.types.vector(3):
    """
    Return the velocity of a surface vertex once it exchanged momentum with the cells of the grid within one cell of the
    plane of its triangle, in proportion to the mass of each cell.
    """
    dx = mpm_info.dx[None]
    inv_dx = mpm_info.inv_dx[None]
    grid_offset = mpm_info.grid_offset[None]
    mass_fem_sv = mass / 4.0

    mass_mpm = gs.qd_float(0.0)
    mpm_base, _, mpm_w = func_quadratic_bspline_stencil(pos, inv_dx)
    grid_res = qd.Vector([mpm_state.grid.mass.shape[1], mpm_state.grid.mass.shape[2], mpm_state.grid.mass.shape[3]])
    new_vel = vel
    for mpm_offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
        cell_ijk = mpm_base - grid_offset + mpm_offset
        i_x, i_y, i_z = cell_ijk[0], cell_ijk[1], cell_ijk[2]
        # A vertex outside the domain of the grid exchanges nothing with the cells the grid lacks
        mpm_grid_mass = gs.qd_float(0.0)
        if (cell_ijk >= 0).all() and (cell_ijk < grid_res).all():
            mpm_grid_mass = mpm_state.grid.mass[f, i_x, i_y, i_z, i_b] / mpm_info.particle_volume_scale[None]

        mpm_weight = gs.qd_float(1.0)
        for d in qd.static(range(3)):
            mpm_weight *= mpm_w[mpm_offset[d], d]

        # FEM -> MPM, within one cell of the plane of the triangle
        mpm_grid_pos = (cell_ijk + grid_offset) * dx
        signed_dist = (mpm_grid_pos - pos).dot(surface_normal)
        if signed_dist <= dx:
            mass_mpm_at_cell = mpm_weight * mpm_grid_mass
            mass_mpm += mass_mpm_at_cell

            if mass_mpm_at_cell > fem_info.EPS[None]:
                # The momentum is exchanged in proportion to the mass of the whole cell
                delta_mpm_vel_at_cell = (vel * mpm_weight - mpm_state.grid.vel_out[f, i_x, i_y, i_z, i_b]) * (
                    mpm_grid_mass / mass_fem_sv
                )
                func_add_grid_vel(f, i_x, i_y, i_z, i_b, delta_mpm_vel_at_cell, mpm_state)
                new_vel -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

    # MPM -> FEM
    if mass_mpm <= fem_info.EPS[None]:
        new_vel = vel
    return new_vel


class FEMMPM(SurfaceCoupling):
    """
    Exchange momentum between the surface vertices of the FEM entities and the cells of the MPM grid near their surface.

    The grid is coupled rather than the particles, after its boundary is imposed, and the surface is searched for from
    the vertices, which spares searching it from every particle.
    """

    is_differentiable = False
    surface_coupling = func_fem_mpm

    def __init__(
        self,
        fem_surface: FEMSurface,
        mpm_solver: MPMSolver,
        options: Annotated[LegacyCouplerOptions, Enabled("fem_mpm")],
    ) -> None:
        pass
