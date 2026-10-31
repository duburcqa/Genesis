"""The response of a point colliding with the rigid geoms, which the couplings of the rigid solver share."""

import quadrants as qd

import genesis.utils.sdf as sdf
from genesis.engine.solvers.rigid.abd import func_collide_in_rigid_geom
from genesis.engine.solvers.rigid.data import GeomsInfo, GeomsState, LinksState, RigidInfo
from genesis.utils.sdf import SDFInfo


@qd.func
def func_collide_with_rigid_geom(
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    geoms_state: GeomsState,
    links_state: LinksState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """
    Return the velocity of a point colliding with a coupled rigid geom, applying the reaction to its link.

    The influence of the geom fades with the distance to its surface, over a length its coupling softness sets.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)

    # A larger coupling softness extends the influence of the geom further away from it
    influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[i_g])), 1)

    if influence > 0.1:
        normal_rigid = sdf.sdf_func_normal_world(
            i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
        )
        vel = func_collide_in_rigid_geom(
            i_g, i_b, pos_world, vel, mass, normal_rigid, influence, links_state, geoms_info, rigid_info
        )

    return vel


@qd.func
def func_collide_with_rigid(
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    geoms_state: GeomsState,
    links_state: LinksState,
    geoms_info: GeomsInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
):
    """Return the velocity of a point colliding with every coupled rigid geom in turn.

    See func_collide_with_rigid_geom.
    """
    for i_g in range(geoms_info.needs_coup.shape[0]):
        if geoms_info.needs_coup[i_g]:
            vel = func_collide_with_rigid_geom(
                i_g,
                i_b,
                pos_world,
                vel,
                mass,
                geoms_state,
                links_state,
                geoms_info,
                rigid_info,
                sdf_info,
                collider_config,
            )
    return vel
