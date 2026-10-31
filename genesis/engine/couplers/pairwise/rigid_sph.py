"""The coupling of the rigid geoms with the smoothed particle hydrodynamics (SPH) particles."""

from typing import Annotated

import quadrants as qd

import genesis.utils.array_class as array_class
import genesis.utils.sdf as sdf
from genesis.engine.pipeline import Enabled, register_system
from genesis.engine.solvers.rigid.abd import func_apply_coupling_force, func_collide_in_rigid_geom
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.solvers.sph_solver import SPHSolver
from genesis.engine.step import Coupling
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.geom import func_cubic_kernel_plane_integral


@qd.func
def func_collide_with_rigid_geom_pressure(
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    pressure: float,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    sph_info: array_class.SPHInfo,
    geoms_info: array_class.GeomsInfo,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_config: qd.template(),
):
    """
    Return the velocity of a smoothed particle hydrodynamics (SPH) particle colliding with a coupled rigid geom.

    On top of the collision response (see func_collide_with_rigid_geom), the pressure of the particle pushes on the
    geom.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)
    normal_rigid = sdf.sdf_func_normal_world(
        i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
    )

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
    sph_state: array_class.SPHState,
    dyn_state: array_class.DynState,
    sph_info: array_class.SPHInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    rigid_config: qd.template(),
    collider_config: qd.template(),
):
    """Collide the sorted smoothed particle hydrodynamics particles with the coupled rigid geoms.

    See func_collide_with_rigid_geom_pressure.
    """
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            for i_g in range(dyn_info.geoms.needs_coup.shape[0]):
                if dyn_info.geoms.needs_coup[i_g]:
                    sph_state.particles_reordered.vel[i_p, i_b] = func_collide_with_rigid_geom_pressure(
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
                        collider_info.sdf,
                        rigid_config,
                        collider_config,
                    )


@register_system
class RigidSPH(Coupling):
    """Collide the smoothed particle hydrodynamics particles with the coupled rigid geoms, which their pressure pushes."""

    couple = kernel_sph_rigid

    def __init__(
        self,
        rigid_solver: RigidSolver,
        sph_solver: SPHSolver,
        options: Annotated[LegacyCouplerOptions, Enabled("rigid_sph")],
    ) -> None:
        # The particles collide with the rigid geoms through their SDF
        rigid_solver.opt_ins.add(array_class.SDFInfo)
