"""The coupling of the rigid geoms with the smoothed particle hydrodynamics (SPH) particles."""

from typing import Annotated

import quadrants as qd

import genesis.utils.sdf as sdf
from genesis.engine.pipeline import Enabled
from genesis.engine.solvers.particle_solver import func_set_particle_vel
from genesis.engine.solvers.rigid.abd import func_apply_coupling_force, func_collide_in_rigid_geom
from genesis.engine.solvers.rigid.data import DynInfo, DynState, GeomsInfo, GeomsState, LinksInfo, LinksState, RigidInfo
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.solvers.sph_solver import SPHInfo, SPHSolver, SPHState
from genesis.engine.step import Coupling
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.geom import func_cubic_kernel_plane_integral
from genesis.utils.sdf import SDF, SDFInfo


@qd.func
def func_collide_with_rigid_geom_pressure(
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    pressure: float,
    geoms_state: GeomsState,
    links_state: LinksState,
    sph_info: SPHInfo,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    rigid_config: qd.template(),
):
    """
    Return the velocity of a smoothed particle hydrodynamics (SPH) particle colliding with a coupled rigid geom.

    On top of the collision response (see func_collide_with_rigid_geom), the pressure of the particle pushes on the
    geom.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)
    normal_rigid = sdf.sdf_func_normal_world(i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info)

    # A larger coupling softness extends the influence of the geom further away from it
    influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[i_g])), 1)

    if influence > 0.1:
        vel = func_collide_in_rigid_geom(
            i_g, i_b, pos_world, vel, mass, normal_rigid, influence, links_state, geoms_info, rigid_info
        )

    # Static fluid pressure pushes on the geom even at rest, where the velocity-gated collision response above transfers
    # nothing, which makes submerged geoms buoyant. Mirroring the particle pressure across the surface and integrating
    # the symmetric pressure force over the truncated kernel support yields the factor 2 * sigma(signed_dist), with
    # sigma the kernel plane integral (see func_cubic_kernel_plane_integral in geom.py). Sigma integrates to 1/2 across
    # the support band, so a covering particle layer transmits exactly p per unit area: Archimedes buoyancy with no
    # tuning constant. Fixed links are exempt: they cannot respond to the force, and the reaction is a conservative
    # stiff kick that keeps fluid resting on them ringing forever, pumped by its acoustic pressure fluctuations.
    i_l = geoms_info.link_idx[i_g]
    I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
    support_radius = sph_info.support_radius[None]
    if signed_dist < support_radius and pressure > 0 and not links_info.is_fixed[I_l]:
        pressure_force = (
            -2.0
            * pressure
            * sph_info.particle_volume[None]
            * func_cubic_kernel_plane_integral(signed_dist, support_radius)
            * normal_rigid
        )
        func_apply_coupling_force(i_l, i_b, pos_world, pressure_force, links_state)
        vel = vel - pressure_force * (rigid_info.substep_dt[None] / mass)

    return vel


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

    See func_collide_with_rigid_geom_pressure.
    """
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            for i_g in range(dyn_info.geoms.needs_coup.shape[0]):
                if dyn_info.geoms.needs_coup[i_g]:
                    vel = func_collide_with_rigid_geom_pressure(
                        i_g,
                        i_b,
                        sph_state.particles_reordered.pos[i_p, i_b],
                        sph_state.particles_reordered.vel[i_p, i_b],
                        sph_state.particles_info_reordered.mass[i_p, i_b],
                        sph_state.particles_reordered.p[i_p, i_b],
                        dyn_state.geoms,
                        dyn_state.links,
                        sph_info,
                        dyn_info.geoms,
                        dyn_info.links,
                        rigid_info,
                        sdf_info,
                        rigid_config,
                    )
                    func_set_particle_vel(i_p, i_b, vel, sph_state.particles_reordered.vel)


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
