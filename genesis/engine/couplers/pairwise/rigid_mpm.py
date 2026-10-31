"""The coupling of the rigid geoms with the material point method (MPM) solver."""

from collections.abc import Iterator
from typing import Annotated

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.entities.hybrid_entity import HybridEntity
from genesis.engine.pipeline import Enabled, register_system
from genesis.engine.solvers.mpm_solver import CellSeparation, CellVelocity, GridCoupling, MPMSolver
from genesis.engine.solvers.rigid.abd import func_apply_coupling_force, func_collide_in_rigid_geom
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.state_change import StateChange, Subscriber
from genesis.engine.step import Integration, Preparation
from genesis.options.solvers import LegacyCouplerOptions, MPMOptions
from genesis.utils.array_class import ENVS, GEOMS, PARTICLES
from genesis.utils.data import Data, DataItem, DataKind, field, iter_data
from genesis.utils.misc import broadcast_tensor
from genesis.utils.sdf import SDF

from .rigid_contact import func_collide_with_rigid


class RigidMPMCPICState(Data, kind=DataKind.STATE):
    """
    The thin rigid geoms separating each particle from the cells of its stencil, for Compatible Particle-In-Cell (CPIC).

    `rigid_normal` is the last outward normal of each geom seen by each particle, and `cells_geom_idx` the geom
    separating each particle from each of the 27 cells of its stencil, -1 for none.
    """

    rigid_normal: qd.Tensor = field((float, 3), shape=(PARTICLES, GEOMS, ENVS))
    cells_geom_idx: qd.Tensor = field(int, shape=(PARTICLES, 27, ENVS), kind=DataKind.SCRATCH)


@qd.func
def func_mpm_rigid(
    f: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_config: qd.template(),
) -> qd.types.vector(3):
    """Return the velocity of a cell of the grid colliding with the coupled rigid geoms (see func_collide_with_rigid)."""
    return func_collide_with_rigid(
        i_b,
        pos,
        vel,
        mass,
        dyn_state.geoms,
        dyn_state.links,
        dyn_info.geoms,
        rigid_info,
        sdf_info,
        collider_config,
    )


@qd.func
def func_cpic_cell_separation(
    f: int,
    i_p: int,
    i_b: int,
    i_c: int,
    cell_pos: qd.types.vector(3),
    i_sep: int,
    rigid_mpm_cpic: RigidMPMCPICState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_config: qd.template(),
) -> int:
    """
    Return the first coupled rigid geom separating a particle from a cell of its stencil, -1 for none, unless an earlier
    coupling found a separation.

    A geom separates them where the outward normal of the geom at the cell points against the one the particle saw last.
    """
    i_g_sep = -1
    if i_sep == -1:
        for i_g in range(rigid_mpm_cpic.rigid_normal.shape[1]):
            if i_g_sep == -1 and dyn_info.geoms.needs_coup[i_g]:
                sdf_normal_cell = sdf.sdf_func_normal_world(
                    i_g, i_b, cell_pos, dyn_state.geoms, dyn_info.geoms, rigid_info, sdf_info, collider_config
                )
                if rigid_mpm_cpic.rigid_normal[i_p, i_g, i_b].dot(sdf_normal_cell) < 0:
                    i_g_sep = i_g
    rigid_mpm_cpic.cells_geom_idx[i_p, i_c, i_b] = i_g_sep
    if i_sep == -1:
        i_sep = i_g_sep
    return i_sep


@qd.func
def func_cpic_cell_velocity(
    f: int,
    i_p: int,
    i_b: int,
    i_c: int,
    weight: float,
    grid_vel: qd.types.vector(3),
    mpm_state: array_class.MPMState,
    rigid_mpm_cpic: RigidMPMCPICState,
    dyn_state: array_class.DynState,
    mpm_info: array_class.MPMInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
) -> qd.types.vector(3):
    """
    Return the velocity of a particle colliding with the thin rigid geom separating it from a cell of its stencil, in
    place of the velocity of the cell, which is kept where no geom separates them.
    """
    i_g_sep = rigid_mpm_cpic.cells_geom_idx[i_p, i_c, i_b]
    if i_g_sep != -1:
        grid_vel = func_collide_in_rigid_geom(
            i_g_sep,
            i_b,
            mpm_state.particles.pos[f, i_p, i_b],
            mpm_state.particles.vel[f, i_p, i_b],
            mpm_info.particles.mass[i_p] * weight / mpm_info.particle_volume_scale[None],
            rigid_mpm_cpic.rigid_normal[i_p, i_g_sep, i_b],
            influence=1.0,
            links_state=dyn_state.links,
            geoms_info=dyn_info.geoms,
            rigid_info=rigid_info,
        )
    return grid_vel


@qd.kernel(fastcache=True)
def kernel_update_cpic_rigid_normal(
    f: int,
    mpm_state: array_class.MPMState,
    rigid_mpm_cpic: RigidMPMCPICState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_config: qd.template(),
):
    """Record the outward normal of every coupled rigid geom at the active material point method particles.

    A particle keeps the normal of the side it came from when it crosses a geom.
    """
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            for i_g in range(rigid_mpm_cpic.rigid_normal.shape[1]):
                if dyn_info.geoms.needs_coup[i_g]:
                    sdf_normal = sdf.sdf_func_normal_world(
                        i_g,
                        i_b,
                        mpm_state.particles.pos[f, i_p, i_b],
                        dyn_state.geoms,
                        dyn_info.geoms,
                        rigid_info,
                        sdf_info,
                        collider_config,
                    )
                    # The normal of a geom a particle penetrates keeps the side the particle came from
                    if sdf_normal.dot(rigid_mpm_cpic.rigid_normal[i_p, i_g, i_b]) >= 0:
                        rigid_mpm_cpic.rigid_normal[i_p, i_g, i_b] = sdf_normal


@qd.kernel(fastcache=True)
def kernel_reset_cpic_rigid_normal(envs_idx: qd.types.ndarray(), rigid_mpm_cpic: RigidMPMCPICState):
    """Zero the normals of the thin rigid geoms seen by the particles of some environments."""
    for i_p, i_g, i_b_ in qd.ndrange(
        rigid_mpm_cpic.rigid_normal.shape[0], rigid_mpm_cpic.rigid_normal.shape[1], envs_idx.shape[0]
    ):
        i_b = envs_idx[i_b_]
        rigid_mpm_cpic.rigid_normal[i_p, i_g, i_b] = qd.Vector.zero(gs.qd_float, 3)


@register_system
class RigidMPM(GridCoupling):
    """Collide the cells of the material point method (MPM) grid holding mass with the coupled rigid geoms."""

    grid_coupling = func_mpm_rigid

    def __init__(
        self,
        rigid_solver: RigidSolver,
        mpm_solver: MPMSolver,
        sdf: SDF,
        options: Annotated[LegacyCouplerOptions, Enabled("rigid_mpm")],
    ) -> None:
        pass


@register_system
class RigidMPMCPIC(Preparation, CellSeparation, CellVelocity):
    """
    Separate the particles and the cells of the material point method (MPM) grid that a thin coupled rigid geom splits.

    With Compatible Particle-In-Cell (CPIC), a particle and a cell of its stencil that a thin rigid geom separates
    exchange nothing: the cell gets nothing from the particle, and gives it the velocity of the particle colliding with
    the geom instead. CPIC carries no gradient.
    """

    is_differentiable = False
    preprocess = kernel_update_cpic_rigid_normal
    cell_separation = func_cpic_cell_separation
    cell_velocity = func_cpic_cell_velocity

    rigid_mpm_cpic: RigidMPMCPICState

    def __init__(
        self,
        rigid_mpm: RigidMPM,
        rigid_solver: RigidSolver,
        mpm_solver: MPMSolver,
        sdf: SDF,
        options: Annotated[MPMOptions, Enabled("enable_CPIC")],
    ) -> None:
        self._mpm_solver = mpm_solver
        self.rigid_mpm_cpic = RigidMPMCPICState(
            sizes=dict(particles=mpm_solver.n_particles, geoms=max(1, rigid_solver.n_geoms), envs=mpm_solver.sim._B)
        )
        # The thin rigid geoms the particles saw belong to the trajectory they leave
        mpm_solver.subscribe(
            Subscriber(frozenset((StateChange.GEOMETRY,)), callback=lambda _, envs_idx: self.reset(envs_idx))
        )

    def reset(self, envs_idx=None) -> None:
        """Forget the thin rigid geoms the particles of some environments saw, every environment by default."""
        if envs_idx is None:
            self.rigid_mpm_cpic.rigid_normal.fill(0)
        else:
            kernel_reset_cpic_rigid_normal(self._mpm_solver.scene._sanitize_envs_idx(envs_idx), self.rigid_mpm_cpic)


@qd.kernel(fastcache=True)
def kernel_update_soft_part_mpm(
    f: int,
    particle_start: int,
    dt: float,
    damping: float,
    soft_dv_coef: float,
    mpm_state: array_class.MPMState,
    links_state: array_class.LinksState,
    soft_part_info: array_class.HybridSoftPartInfo,
    mpm_info: array_class.MPMInfo,
    geoms_info: array_class.GeomsInfo,
):
    """
    Couple the particles of the soft part of a hybrid entity with the links of its rigid part, once g2p advected them.

    Each active particle is driven toward the pose its muscle group takes with its link, the velocity correction being
    damped by `damping` over the timestep `dt` shared by both solvers, partly applied to the particle by
    `soft_dv_coef`, and its momentum applied to the link.
    """
    for i_p_, i_b in qd.ndrange(soft_part_info.particles_init_pos.shape[0], mpm_state.particles.pos.shape[2]):
        i_p = i_p_ + particle_start
        if mpm_state.particles.is_active[f, i_p, i_b]:
            i_mg = mpm_info.particles.muscle_group[i_p]
            i_l = soft_part_info.groups_link_idx[i_mg]
            i_g = soft_part_info.groups_geom_idx[i_mg]

            # Target position of the particle, rigidly attached to its link
            local_pos = gu.qd_inv_transform_by_trans_quat(
                soft_part_info.particles_init_pos[i_p_],
                soft_part_info.groups_frame_pos[i_mg],
                soft_part_info.groups_frame_quat[i_mg],
            )
            target_pos, target_quat = gu.qd_transform_pos_quat_by_trans_quat(
                geoms_info.pos[i_g], geoms_info.quat[i_g], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
            )
            new_x_pos = gu.qd_transform_by_trans_quat(local_pos, trans=target_pos, quat=target_quat)

            # The velocity covers the distance to the target over a whole step, so each substep moves the particle part
            # of the way
            x_pos = mpm_state.particles.pos[f + 1, i_p, i_b]
            xd_vel = (new_x_pos - x_pos) / dt

            vel_d = xd_vel - mpm_state.particles.vel[f + 1, i_p, i_b]
            vel_d *= qd.exp(-dt * damping)

            # Soft to rigid, over the timestep of the rigid solver since the update integrates within a substep of the
            # soft solver
            mass_real = mpm_info.particles.mass[i_p] / mpm_info.particle_volume_scale[None]
            func_apply_coupling_force(i_l, i_b, x_pos, -mass_real * (vel_d / dt), links_state)

            # Rigid to soft, which may destabilize the loop
            mpm_state.particles.vel[f + 1, i_p, i_b] += vel_d * soft_dv_coef


@register_system
class RigidMPMHybrid(Integration):
    """
    Drive the particles of the soft part of the hybrid entities toward the links of their rigid part, after the grid
    advects them (see kernel_update_soft_part_mpm).
    """

    def __init__(
        self, mpm_solver: Annotated[MPMSolver, Enabled("has_hybrid_entities")], rigid_solver: RigidSolver
    ) -> None:
        self._mpm_solver = mpm_solver
        self._rigid_solver = rigid_solver
        self._entities = [
            entity
            for entity in mpm_solver.sim.entities
            if isinstance(entity, HybridEntity) and not entity.material.use_default_coupling
        ]

    def build(self) -> None:
        for entity in self._entities:
            entity.build()

    def substep_post_coupling(self, f: int) -> None:
        mpm_solver, dyn_state, dyn_info = self._mpm_solver, self._rigid_solver.dyn_state, self._rigid_solver.dyn_info
        for entity in self._entities:
            kernel_update_soft_part_mpm(
                f,
                entity.part_soft.particle_start,
                mpm_solver.dt,
                entity.material.damping,
                entity.material.soft_dv_coef,
                mpm_solver.mpm_state,
                dyn_state.links,
                entity.soft_part_info,
                mpm_solver.mpm_info,
                dyn_info.geoms,
            )


class RigidMPMAttachmentInfo(Data, kind=DataKind.INFO, shape=(PARTICLES, ENVS)):
    """
    The critically damped springs pulling material point method particles toward points of rigid links.

    A particle attached to a link (`link_idx` other than -1) is pulled toward the point `link_local_pos` of the frame of
    the link, with the spring of the given `stiffness`.
    """

    stiffness: qd.Tensor = field(float)
    link_idx: qd.Tensor = field(int)
    link_local_pos: qd.Tensor = field((float, 3))


@register_system
class RigidMPMAttachment(Integration):
    """
    Pull material point method (MPM) particles toward points of rigid links with critically damped springs.

    The springs are allocated the first time a particle is attached, which most scenes never do (see 'attach').
    """

    rigid_mpm_attachment_info: RigidMPMAttachmentInfo | None

    def __init__(self, rigid_solver: RigidSolver, mpm_solver: MPMSolver) -> None:
        self._rigid_solver = rigid_solver
        self._mpm_solver = mpm_solver
        self.rigid_mpm_attachment_info = None

    def substep_post_coupling(self, f: int) -> None:
        """Pull the attached particles of the next frame toward their link."""
        if self.rigid_mpm_attachment_info is not None:
            kernel_apply_rigid_mpm_attachment(
                f,
                self._mpm_solver.mpm_state,
                self._rigid_solver.dyn_state,
                self.rigid_mpm_attachment_info,
                self._mpm_solver.mpm_info,
            )

    def attach(self, particles_mask, link_idx, stiffness, *, particles_start=0):
        """
        Attach some particles to a rigid link, each pulled toward the point of the frame of the link where it lies now.

        `particles_mask` selects consecutive particles of the MPM solver from the global index `particles_start` on.
        """
        mpm_solver = self._mpm_solver
        n_envs = mpm_solver.sim._B
        if not isinstance(link_idx, int):
            gs.raise_exception("link_idx must be an integer.")
        if self.rigid_mpm_attachment_info is None:
            # The flat index of a vector array fits in int32
            if mpm_solver.n_particles * n_envs * 3 > np.iinfo(np.int32).max:
                gs.raise_exception(
                    f"Particle constraint shape (n_envs={n_envs}, n_particles={mpm_solver.n_particles}, 3) is too "
                    "large. Consider reducing n_envs or n_particles."
                )
            self.rigid_mpm_attachment_info = RigidMPMAttachmentInfo(
                sizes=dict(particles=mpm_solver.n_particles, envs=n_envs)
            )
            self.rigid_mpm_attachment_info.link_idx.fill(-1)
        particles_mask = broadcast_tensor(particles_mask, gs.tc_bool, (n_envs, -1), ("envs_idx", "particles_idx"))
        if particles_start < 0 or particles_start + particles_mask.shape[1] > mpm_solver.n_particles:
            gs.raise_exception("The particle mask overruns the particles of the solver.")
        # The attachment is a configuration of the constraint, which carries no gradient
        with torch.no_grad():
            link_pos = self._rigid_solver.get_links_pos(links_idx=link_idx)[..., 0, :]
            link_quat = self._rigid_solver.get_links_quat(links_idx=link_idx)[..., 0, :]
        link_pos = broadcast_tensor(link_pos, gs.tc_float, (n_envs, 3)).contiguous()
        link_quat = broadcast_tensor(link_quat, gs.tc_float, (n_envs, 4)).contiguous()
        kernel_attach_mpm_particles(
            particles_start,
            link_idx,
            stiffness,
            particles_mask.contiguous(),
            link_pos,
            link_quat,
            mpm_solver.mpm_state,
            self.rigid_mpm_attachment_info,
        )

    def detach(self, particles_mask=None, *, particles_start=0):
        """Detach some particles from their link, every particle from `particles_start` on by default."""
        if self.rigid_mpm_attachment_info is None:
            return
        n_envs = self._mpm_solver.sim._B
        n_particles = -1
        if particles_mask is None:
            particles_mask = True
            n_particles = self._mpm_solver.n_particles - particles_start
        particles_mask = broadcast_tensor(
            particles_mask, gs.tc_bool, (n_envs, n_particles), ("envs_idx", "particles_idx")
        ).contiguous()
        if particles_start < 0 or particles_start + particles_mask.shape[1] > self._mpm_solver.n_particles:
            gs.raise_exception("The particle mask overruns the particles of the solver.")
        kernel_detach_mpm_particles(particles_start, particles_mask, self.rigid_mpm_attachment_info)


@qd.kernel(fastcache=True)
def kernel_attach_mpm_particles(
    particles_start: int,
    link_idx: int,
    stiffness: float,
    particles_mask: qd.types.ndarray(),
    link_pos: qd.types.ndarray(element_dim=1),
    link_quat: qd.types.ndarray(element_dim=1),
    mpm_state: array_class.MPMState,
    rigid_mpm_attachment_info: RigidMPMAttachmentInfo,
):
    """Attach the masked particles to a link, at the point of its frame where each one lies now."""
    for i_p_, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
        if particles_mask[i_b, i_p_]:
            i_p = i_p_ + particles_start
            # Between two steps, the state sits in frame 0 (see 'Solver.has_frames')
            pos = mpm_state.particles.pos[0, i_p, i_b]
            rigid_mpm_attachment_info.stiffness[i_p, i_b] = stiffness
            rigid_mpm_attachment_info.link_idx[i_p, i_b] = link_idx
            rigid_mpm_attachment_info.link_local_pos[i_p, i_b] = gu.qd_inv_transform_by_trans_quat(
                pos, link_pos[i_b], link_quat[i_b]
            )


@qd.kernel(fastcache=True)
def kernel_detach_mpm_particles(
    particles_start: int, particles_mask: qd.types.ndarray(), rigid_mpm_attachment_info: RigidMPMAttachmentInfo
):
    """Detach the masked particles from the link they are attached to."""
    for i_p_, i_b in qd.ndrange(particles_mask.shape[1], particles_mask.shape[0]):
        if particles_mask[i_b, i_p_]:
            rigid_mpm_attachment_info.link_idx[i_p_ + particles_start, i_b] = -1


@qd.kernel(fastcache=True)
def kernel_apply_rigid_mpm_attachment(
    f: int,
    mpm_state: array_class.MPMState,
    dyn_state: array_class.DynState,
    rigid_mpm_attachment_info: RigidMPMAttachmentInfo,
    mpm_info: array_class.MPMInfo,
):
    """Pull the attached particles of the next frame toward the point of their link with a critically damped spring."""
    for i_p, i_b in qd.ndrange(
        rigid_mpm_attachment_info.link_idx.shape[0], rigid_mpm_attachment_info.link_idx.shape[1]
    ):
        i_l = rigid_mpm_attachment_info.link_idx[i_p, i_b]
        if i_l >= 0:
            target_pos = gu.qd_transform_by_trans_quat(
                rigid_mpm_attachment_info.link_local_pos[i_p, i_b],
                dyn_state.links.pos[i_l, i_b],
                dyn_state.links.quat[i_l, i_b],
            )
            stiffness = rigid_mpm_attachment_info.stiffness[i_p, i_b]
            mass = mpm_info.particles.mass[i_p] / mpm_info.particle_volume_scale[None]

            pos_error = mpm_state.particles.pos[f + 1, i_p, i_b] - target_pos
            spring_force = -stiffness * pos_error
            damping_force = -2.0 * qd.math.sqrt(stiffness * mass) * mpm_state.particles.vel[f + 1, i_p, i_b]

            # The velocity is incremented in place, which autodiff reverses, since it cannot reverse an overwrite of a
            # field the same kernel reads
            mpm_state.particles.vel[f + 1, i_p, i_b] += (
                mpm_info.substep_dt[None] * (spring_force + damping_force) / mass
            )
