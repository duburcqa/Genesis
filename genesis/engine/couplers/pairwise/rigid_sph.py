"""The coupling of the rigid geoms with the smoothed particle hydrodynamics (SPH) particles."""

from typing import Annotated

import quadrants as qd

from genesis.engine.pipeline import Enabled
from genesis.engine.solvers.particle_solver import func_set_particle_vec3
from genesis.engine.solvers.rigid.data import DynInfo, DynState, RigidInfo
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.solvers.sph_solver import SPHInfo, SPHSolver, SPHState
from genesis.engine.step import Coupling
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.sdf import SDF, SDFInfo

from .rigid_contact import func_collide_with_rigid_geom


@qd.kernel(fastcache=True)
def kernel_sph_rigid(
    sph_state: SPHState,
    dyn_state: DynState,
    sph_info: SPHInfo,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    rigid_config: qd.template(),
):
    """Collide the sorted smoothed particle hydrodynamics particles with the coupled rigid geoms.

    See func_collide_with_rigid_geom.
    """
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            for i_g in range(dyn_info.geoms.needs_coup.shape[0]):
                if dyn_info.geoms.needs_coup[i_g]:
                    vel = func_collide_with_rigid_geom(
                        i_g,
                        i_b,
                        sph_state.particles_reordered.pos[i_p, i_b],
                        sph_state.particles_reordered.vel[i_p, i_b],
                        sph_state.particles_info_reordered.mass[i_p, i_b],
                        sph_state.particles_reordered.p[i_p, i_b],
                        sph_info.particle_volume[None],
                        sph_info.support_radius[None],
                        dyn_state.geoms,
                        dyn_state.links,
                        dyn_info.geoms,
                        dyn_info.links,
                        rigid_info,
                        sdf_info,
                        rigid_config,
                        has_pressure=True,
                    )
                    func_set_particle_vec3(i_p, i_b, vel, sph_state.particles_reordered.vel)


class RigidSPH(Coupling):
    """Collide the smoothed particle hydrodynamics particles with the coupled rigid geoms, which their pressure pushes."""

    couple = kernel_sph_rigid

    def __init__(
        self,
        rigid_solver: RigidSolver,
        sph_solver: SPHSolver,
        sdf: SDF,
        options: Annotated[LegacyCouplerOptions, Enabled("rigid_sph")],
    ) -> None:
        pass
