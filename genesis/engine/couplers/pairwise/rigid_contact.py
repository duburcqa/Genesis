"""The response of a point colliding with the rigid geoms, which the couplings of the rigid solver share."""

import quadrants as qd

import genesis.utils.sdf as sdf
from genesis.engine.solvers.rigid.abd import func_apply_coupling_force, func_collide_in_rigid_geom
from genesis.engine.solvers.rigid.data import DynInfo, DynState, GeomsInfo, GeomsState, LinksInfo, LinksState, RigidInfo
from genesis.utils.geom import func_cubic_kernel_plane_integral
from genesis.utils.sdf import SDFInfo


@qd.func
def func_collide_with_rigid_geom(
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    pressure: float,
    particle_volume: float,
    support_radius: float,
    geoms_state: GeomsState,
    links_state: LinksState,
    geoms_info: GeomsInfo,
    links_info: LinksInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    rigid_config: qd.template(),
    has_pressure: qd.template(),
):
    """
    Return the velocity of a point colliding with a coupled rigid geom, applying the reaction to its link.

    The influence of the geom fades with the distance to its surface, over a length its coupling softness sets. With
    `has_pressure`, the point is a fluid particle of the given `pressure` and `particle_volume`, whose pressure pushes on
    the geom within the smoothing kernel support `support_radius` on top of the collision response.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)

    # A larger coupling softness extends the influence of the geom further away from it
    influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[i_g])), 1)
    is_colliding = influence > 0.1

    # Static fluid pressure pushes on the geom even at rest, where the velocity-gated collision response transfers
    # nothing, which makes submerged geoms buoyant. Fixed links are exempt: they cannot respond to the force, and the
    # reaction is a conservative stiff kick that keeps fluid resting on them ringing forever, pumped by its acoustic
    # pressure fluctuations.
    i_l = geoms_info.link_idx[i_g]
    is_pushing = False
    if qd.static(has_pressure):
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        is_pushing = signed_dist < support_radius and pressure > 0 and not links_info.is_fixed[I_l]

    if is_colliding or is_pushing:
        normal_rigid = sdf.sdf_func_normal_world(i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info)
        if is_colliding:
            vel = func_collide_in_rigid_geom(
                i_g, i_b, pos_world, vel, mass, normal_rigid, influence, links_state, geoms_info, rigid_info
            )
        if qd.static(has_pressure):
            if is_pushing:
                # Mirroring the particle pressure across the surface and integrating the symmetric pressure force over
                # the truncated kernel support yields the factor 2 * sigma(signed_dist), with sigma the kernel plane
                # integral (see func_cubic_kernel_plane_integral in geom.py). Sigma integrates to 1/2 across the
                # support band, so a covering particle layer transmits exactly p per unit area: Archimedes buoyancy
                # with no tuning constant.
                pressure_force = (
                    -2.0
                    * pressure
                    * particle_volume
                    * func_cubic_kernel_plane_integral(signed_dist, support_radius)
                    * normal_rigid
                )
                func_apply_coupling_force(i_l, i_b, pos_world, pressure_force, links_state)
                vel = vel - pressure_force * (rigid_info.substep_dt[None] / mass)

    return vel


@qd.func
def func_collide_with_rigid(
    f: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    rigid_config: qd.template(),
) -> qd.types.vector(3):
    """
    Return the velocity of a point of the given mass at `pos` colliding with every coupled rigid geom in turn.

    It couples a cell of the grid of the material point method (MPM) solver as it is (see 'MPMSolver.grid_couplings').
    See func_collide_with_rigid_geom.
    """
    for i_g in range(dyn_info.geoms.needs_coup.shape[0]):
        if dyn_info.geoms.needs_coup[i_g]:
            vel = func_collide_with_rigid_geom(
                i_g,
                i_b,
                pos,
                vel,
                mass,
                0.0,
                0.0,
                0.0,
                dyn_state.geoms,
                dyn_state.links,
                dyn_info.geoms,
                dyn_info.links,
                rigid_info,
                sdf_info,
                rigid_config,
                has_pressure=False,
            )
    return vel
