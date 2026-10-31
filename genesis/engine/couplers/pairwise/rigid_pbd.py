"""The coupling of the rigid geoms with the position based dynamics (PBD) particles."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.sdf as sdf
from genesis.engine.pipeline import System, register_system
from genesis.engine.solvers.pbd_solver import PBDSolver
from genesis.engine.solvers.rigid.abd import func_apply_coupling_force
from genesis.engine.solvers.rigid.collider.collider import Collider
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.step import Step
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.array_class import ENVS, PARTICLES
from genesis.utils.data import Data, DataItem, DataKind, field, iter_data, opt_in
from genesis.utils.geom import qd_inv_transform_by_trans_quat, qd_transform_by_trans_quat

# The largest inverse timestep at which the particles carried by a link catch up with it, which spares them jerks
CLAMPED_INV_DT = 50.0


class RigidPBDInfo(Data, kind=DataKind.INFO, shape=(PARTICLES, ENVS)):
    """The link carrying each position based dynamics particle, -1 for none, and the point of its frame it stays at."""

    link_idx: qd.Tensor = field(int)
    local_pos: qd.Tensor = field((float, 3))


@register_system
class RigidPBD(System):
    """
    Push the position based dynamics (PBD) particles out of the coupled rigid geoms, and carry some along with links.

    A carried particle catches up with the point of the link it follows over `1 / clamped_inv_dt`, clamped to spare it
    jerks (see 'attach_particles').
    """

    def __init__(self, rigid_solver: RigidSolver, pbd_solver: PBDSolver, options: LegacyCouplerOptions) -> None:
        self._pbd_solver = pbd_solver
        self._rigid_solver = rigid_solver
        self.clamped_inv_dt = min(1.0 / pbd_solver.substep_dt, CLAMPED_INV_DT)
        self.rigid_pbd_info = RigidPBDInfo(sizes=dict(particles=pbd_solver.n_particles, envs=pbd_solver.sim._B))
        self.rigid_pbd_info.link_idx.fill(-1)
        # The particles collide with the rigid geoms through their SDF
        opt_in(array_class.SDFInfo)

    @classmethod
    def create(
        cls, rigid_solver: RigidSolver, pbd_solver: PBDSolver, options: LegacyCouplerOptions
    ) -> "RigidPBD | None":
        return cls(rigid_solver, pbd_solver, options) if options.rigid_pbd else None

    def compose(self, step: Step, collider: Collider) -> None:
        # The carried particles catch up with their link after both the rigid and the PBD solvers updated, so the
        # correction lags by one substep
        step.couplings.add(kernel_pbd_rigid_collide, kernel_drive_pbd_particles_by_link)

    @property
    def data(self) -> Iterator[DataItem]:
        yield from iter_data(self, "rigid_pbd_info")

    def attach_particles(self, particles_idx, envs_idx, link_idx):
        """Carry some PBD particles along with a rigid link, where they lie now in its frame."""
        kernel_attach_pbd_to_rigid_link(
            particles_idx,
            envs_idx,
            link_idx,
            self._pbd_solver.pbd_state,
            self._rigid_solver.dyn_state.links,
            self.rigid_pbd_info,
        )

    def detach_particles(self, particles_idx, envs_idx):
        """Release some PBD particles from the link carrying them, if any."""
        kernel_detach_pbd_from_rigid_link(particles_idx, envs_idx, self.rigid_pbd_info)


@qd.func
def func_pbd_collide_with_rigid_geom(
    i_p: int,
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    pbd_state: array_class.PBDState,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    pbd_info: array_class.PBDInfo,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_config: qd.template(),
):
    """
    Return the position and velocity of a sorted position based dynamics (PBD) particle pushed out of a rigid geom.

    The particle is projected onto the surface of the geom, offset by its radius, and its velocity follows from its
    displacement over the substep. The momentum it gains is taken from the link of the geom, friction aside.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)
    contact_normal = sdf.sdf_func_normal_world(
        i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_config
    )
    new_pos = pos_world
    new_vel = vel
    particle_radius = pbd_info.particle_size[None] / 2
    if signed_dist < particle_radius:
        new_pos = pos_world + contact_normal * (particle_radius - signed_dist)
        new_vel = (new_pos - pbd_state.particles_reordered.ipos[i_p, i_b]) / pbd_info.substep_dt[None]

        force = -mass * (new_vel - vel) / rigid_info.substep_dt[None]
        i_l = geoms_info.link_idx[i_g]
        func_apply_coupling_force(i_l, i_b, pos_world, force, links_state)

    return new_pos, new_vel


@qd.kernel(fastcache=True)
def kernel_pbd_rigid_collide(
    pbd_state: array_class.PBDState,
    dyn_state: array_class.DynState,
    pbd_info: array_class.PBDInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_config: qd.template(),
):
    """Push the sorted position based dynamics particles out of the coupled rigid geoms.

    See func_pbd_collide_with_rigid_geom.
    """
    for i_p, i_b in qd.ndrange(pbd_state.particles_reordered.pos.shape[0], pbd_state.particles_reordered.pos.shape[1]):
        if pbd_state.particles_reordered.is_active[i_p, i_b]:
            for i_g in range(dyn_info.geoms.needs_coup.shape[0]):
                if dyn_info.geoms.needs_coup[i_g]:
                    (
                        pbd_state.particles_reordered.pos[i_p, i_b],
                        pbd_state.particles_reordered.vel[i_p, i_b],
                    ) = func_pbd_collide_with_rigid_geom(
                        i_p,
                        i_g,
                        i_b,
                        pbd_state.particles_reordered.pos[i_p, i_b],
                        pbd_state.particles_reordered.vel[i_p, i_b],
                        pbd_state.particles_info_reordered.mass[i_p, i_b],
                        pbd_state,
                        dyn_state.geoms,
                        dyn_state.links,
                        pbd_info,
                        dyn_info.geoms,
                        rigid_info,
                        collider_info.sdf,
                        collider_config,
                    )


@qd.kernel(fastcache=True)
def kernel_attach_pbd_to_rigid_link(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: int,
    pbd_state: array_class.PBDState,
    links_state: array_class.LinksState,
    rigid_pbd_info: RigidPBDInfo,
):
    """Carry some position based dynamics (PBD) particles along with a link, where they lie now in its frame."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        local_pos = qd_inv_transform_by_trans_quat(
            pbd_state.particles.pos[i_p, i_b], links_state.pos[link_idx, i_b], links_state.quat[link_idx, i_b]
        )
        pbd_state.particles.free[i_p, i_b] = False
        rigid_pbd_info.link_idx[i_p, i_b] = link_idx
        rigid_pbd_info.local_pos[i_p, i_b] = local_pos


@qd.kernel(fastcache=True)
def kernel_detach_pbd_from_rigid_link(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), rigid_pbd_info: RigidPBDInfo
):
    """Release some position based dynamics (PBD) particles from the link carrying them."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        rigid_pbd_info.link_idx[i_p, i_b] = -1
        rigid_pbd_info.local_pos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_drive_pbd_particles_by_link(
    clamped_inv_dt: float,
    pbd_state: array_class.PBDState,
    dyn_state: array_class.DynState,
    rigid_pbd_info: RigidPBDInfo,
):
    """
    Drive the position based dynamics (PBD) particles carried by a link toward the point of the link they follow.

    A particle takes the velocity of that point plus the velocity catching up with it over `1 / clamped_inv_dt`.
    """
    for i_p, i_b in qd.ndrange(pbd_state.particles.pos.shape[0], pbd_state.particles.pos.shape[1]):
        i_l = rigid_pbd_info.link_idx[i_p, i_b]
        if i_l >= 0:
            target_world_pos = qd_transform_by_trans_quat(
                rigid_pbd_info.local_pos[i_p, i_b], dyn_state.links.pos[i_l, i_b], dyn_state.links.quat[i_l, i_b]
            )
            link_com_in_world = dyn_state.links.root_COM[i_l, i_b] + dyn_state.links.i_pos[i_l, i_b]
            world_arm = target_world_pos - link_com_in_world
            target_world_vel = dyn_state.links.cd_vel[i_l, i_b] + dyn_state.links.cd_ang[i_l, i_b].cross(world_arm)

            i_r = pbd_state.particles.reordered_idx[i_p, i_b]
            pos_correction = target_world_pos - pbd_state.particles_reordered.pos[i_r, i_b]
            pbd_state.particles_reordered.vel[i_r, i_b] = pos_correction * clamped_inv_dt + target_world_vel
