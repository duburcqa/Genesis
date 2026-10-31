from typing import TYPE_CHECKING

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.sdf as sdf
from genesis.engine.boundaries import func_impose_cube_boundary, func_impose_floor_boundary
from genesis.engine.solvers.mpm_solver import func_quadratic_bspline_stencil
from genesis.engine.solvers.rigid.abd import func_apply_coupling_force, func_collide_in_rigid_geom
from genesis.options.solvers import LegacyCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.data import opt_in
from genesis.utils.geom import (
    func_cubic_kernel_plane_integral,
    func_hash_grid_to_slot,
    func_hash_pos_to_grid,
    qd_inv_transform_by_trans_quat,
    qd_transform_by_trans_quat,
)

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator

# The largest inverse timestep at which the particles carried by a link catch up with it, which spares them jerks
CLAMPED_INV_DT = 50.0


class LegacyCoupler(RBC):
    """
    This class handles all the coupling between different solvers. LegacyCoupler will be deprecated in the future.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, simulator: "Simulator", options: "LegacyCouplerOptions") -> None:
        self.sim = simulator
        self.options = options

        self.rigid_solver = self.sim.rigid_solver
        self.mpm_solver = self.sim.mpm_solver
        self.sph_solver = self.sim.sph_solver
        self.pbd_solver = self.sim.pbd_solver
        self.fem_solver = self.sim.fem_solver
        self.sf_solver = self.sim.sf_solver

        self.coupler_info: array_class.LegacyCouplerInfo | None = None
        self.coupler_static_config: array_class.LegacyCouplerStaticConfig | None = None

        # The particles and the FEM vertices collide with the rigid geoms through their SDF
        opt_in(array_class.SDFInfo, cond=lambda: any(self.rigid_couplings))

    @property
    def rigid_couplings(self) -> tuple[bool, bool, bool, bool]:
        """Whether the rigid geoms couple with the MPM, SPH, PBD and FEM solvers, in this order."""
        return tuple(
            [
                self.rigid_solver.is_active and solver.is_active and is_enabled
                for solver, is_enabled in (
                    (self.mpm_solver, self.options.rigid_mpm),
                    (self.sph_solver, self.options.rigid_sph),
                    (self.pbd_solver, self.options.rigid_pbd),
                    (self.fem_solver, self.options.rigid_fem),
                )
            ]
        )

    def build(self) -> None:
        enable_rigid_mpm, enable_rigid_sph, enable_rigid_pbd, enable_rigid_fem = self.rigid_couplings
        self.coupler_static_config = array_class.LegacyCouplerStaticConfig(
            enable_rigid_mpm=enable_rigid_mpm,
            enable_rigid_sph=enable_rigid_sph,
            enable_rigid_pbd=enable_rigid_pbd,
            enable_rigid_fem=enable_rigid_fem,
            enable_mpm_sph=self.mpm_solver.is_active and self.sph_solver.is_active and self.options.mpm_sph,
            enable_mpm_pbd=self.mpm_solver.is_active and self.pbd_solver.is_active and self.options.mpm_pbd,
            enable_fem_mpm=self.fem_solver.is_active and self.mpm_solver.is_active and self.options.fem_mpm,
            enable_fem_sph=self.fem_solver.is_active and self.sph_solver.is_active and self.options.fem_sph,
        )

        # A material point method (MPM) cell spans the cells of the hash grid of the other solver closer than its size
        # to its lower corner
        mpm_sph_stencil_size = mpm_pbd_stencil_size = 0
        if self.coupler_static_config.enable_mpm_sph:
            mpm_sph_stencil_size = int(np.floor(self.mpm_solver.dx / self.sph_solver.hash_grid_cell_size) + 2)
        if self.coupler_static_config.enable_mpm_pbd:
            mpm_pbd_stencil_size = int(np.floor(self.mpm_solver.dx / self.pbd_solver.hash_grid_cell_size) + 2)

        self.coupler_info = array_class.LegacyCouplerInfo(
            sizes=dict(particles=self.pbd_solver.n_particles, envs=self.sim._B),
            values=dict(
                mpm_sph_stencil_size=mpm_sph_stencil_size, mpm_pbd_stencil_size=mpm_pbd_stencil_size, EPS=gs.EPS
            ),
            unused={array_class.RIGID_PBD: not self.coupler_static_config.enable_rigid_pbd},
        )
        if self.coupler_static_config.enable_rigid_pbd:
            self.coupler_info.pbd_attach_link_idx.fill(-1)

    def reset(self, envs_idx=None) -> None:
        """Restart the coupler, which holds no state of its own."""

    # ------------------------------------------------------------------------------------
    # ------------------------------------ coupling --------------------------------------
    # ------------------------------------------------------------------------------------

    def preprocess(self, f):
        # The material point method (MPM) particles see the thin rigid geoms between them and the cells of their stencil
        # (see MPMCPIC)
        if self.coupler_static_config.enable_rigid_mpm and self.mpm_solver.is_cpic_active:
            kernel_update_cpic_rigid_normal(
                f,
                self.mpm_solver.mpm_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.rigid_solver.collider.collider_config,
            )

    def couple(self, f):
        # MPM <-> all others
        if self.mpm_solver.is_active:
            kernel_mpm_grid_op(
                f,
                self.sim.cur_t,
                self.mpm_solver.mpm_state,
                self.sph_solver.sph_state,
                self.pbd_solver.pbd_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.mpm_solver.mpm_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.coupler_info,
                self.coupler_static_config,
                self.rigid_solver.collider.collider_config,
                self.mpm_solver.force_fields_dispatcher,
            )

        # SPH <-> Rigid
        if self.coupler_static_config.enable_rigid_sph:
            kernel_sph_rigid(
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.sph_solver.sph_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.dyn_info.links,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.rigid_solver.rigid_config,
                self.rigid_solver.collider.collider_config,
            )

        # PBD <-> Rigid
        if self.coupler_static_config.enable_rigid_pbd:
            kernel_pbd_rigid_collide(
                self.pbd_solver.pbd_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.pbd_solver.pbd_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.rigid_solver.collider.collider_config,
            )

            # One-way, the particles carried by links catching up with them over a substep, as fast as the clamp allows
            clamped_inv_dt = min(1.0 / self.pbd_solver.substep_dt, CLAMPED_INV_DT)
            kernel_drive_pbd_particles_by_link(
                clamped_inv_dt, self.pbd_solver.pbd_state, self.rigid_solver.dyn_state.links, self.coupler_info
            )

        if self.fem_solver.is_active:
            kernel_fem_surface_force(
                f,
                self.fem_solver.fem_state,
                self.mpm_solver.mpm_state,
                self.sph_solver.sph_state,
                self.rigid_solver.dyn_state.geoms,
                self.rigid_solver.dyn_state.links,
                self.fem_solver.fem_info,
                self.mpm_solver.mpm_info,
                self.rigid_solver.dyn_info.geoms,
                self.rigid_solver.rigid_info,
                self.rigid_solver.collider.collider_info.sdf,
                self.coupler_info,
                self.coupler_static_config,
                self.rigid_solver.collider.collider_config,
            )
            if self.rigid_solver.is_active:
                self.fem_solver.update_linked_vertex_constraints(self.rigid_solver.dyn_state.links)

    def attach_pbd_particles_to_link(self, particles_idx, envs_idx, link_idx):
        """Carry some position based dynamics (PBD) particles along with a rigid link, where they lie in its frame."""
        if not self.coupler_static_config.enable_rigid_pbd:
            gs.raise_exception(
                "Attaching particles to a link requires the PBD solver to be coupled with the rigid solver."
            )
        kernel_attach_pbd_to_rigid_link(
            particles_idx,
            envs_idx,
            link_idx,
            self.pbd_solver.pbd_state,
            self.rigid_solver.dyn_state.links,
            self.coupler_info,
        )

    def detach_pbd_particles(self, particles_idx, envs_idx):
        """Release some position based dynamics (PBD) particles from the link carrying them, if any."""
        if self.coupler_static_config.enable_rigid_pbd:
            kernel_detach_pbd_from_rigid_link(particles_idx, envs_idx, self.coupler_info)

    @property
    def is_differentiable(self) -> bool:
        """
        Whether the coupler supports a scene with `requires_grad` (see 'Solver.is_differentiable').

        The couplings of the finite element method (FEM) solver with the rigid, material point method and smoothed
        particle hydrodynamics solvers carry no gradient: each surface triangle of an FEM entity overwrites the velocity
        of the vertices it shares with its neighbours, which the reverse pass cannot undo without a race.
        """
        return not (
            self.coupler_static_config.enable_rigid_fem
            or self.coupler_static_config.enable_fem_mpm
            or self.coupler_static_config.enable_fem_sph
        )

    @property
    def active_solvers(self):
        """All the active solvers managed by the scene's simulator."""
        return self.sim.active_solvers


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_collide_with_rigid_geom(
    i_g: int,
    i_b: int,
    pos_world: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
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
            i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
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
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
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
                collider_static_config,
            )
    return vel


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
    collider_static_config: qd.template(),
):
    """
    Return the velocity of a smoothed particle hydrodynamics (SPH) particle colliding with a coupled rigid geom.

    On top of the collision response (see func_collide_with_rigid_geom), the pressure of the particle pushes on the
    geom.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)
    normal_rigid = sdf.sdf_func_normal_world(
        i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
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
    collider_static_config: qd.template(),
):
    """
    Return the position and velocity of a sorted position based dynamics (PBD) particle pushed out of a rigid geom.

    The particle is projected onto the surface of the geom, offset by its radius, and its velocity follows from its
    displacement over the substep. The momentum it gains is taken from the link of the geom, friction aside.
    """
    signed_dist = sdf.sdf_func_world(i_g, i_b, pos_world, geoms_state, geoms_info, sdf_info)
    contact_normal = sdf.sdf_func_normal_world(
        i_g, i_b, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
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
def kernel_update_cpic_rigid_normal(
    f: int,
    mpm_state: array_class.MPMState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """Record the outward normal of every coupled rigid geom at the active material point method particles.

    A particle keeps the normal of the side it came from when it crosses a geom (see MPMCPIC).
    """
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            for i_g in range(mpm_state.cpic.rigid_normal.shape[1]):
                if geoms_info.needs_coup[i_g]:
                    sdf_normal = sdf.sdf_func_normal_world(
                        i_g,
                        i_b,
                        mpm_state.particles.pos[f, i_p, i_b],
                        geoms_state,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        collider_static_config,
                    )
                    # The normal of a geom a particle penetrates keeps the side the particle came from
                    if sdf_normal.dot(mpm_state.cpic.rigid_normal[i_p, i_g, i_b]) >= 0:
                        mpm_state.cpic.rigid_normal[i_p, i_g, i_b] = sdf_normal


@qd.kernel
def kernel_mpm_grid_op(
    f: int,
    t: float,
    mpm_state: array_class.MPMState,
    sph_state: array_class.SPHState,
    pbd_state: array_class.PBDState,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    mpm_info: array_class.MPMInfo,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    coupler_info: array_class.LegacyCouplerInfo,
    coupler_static_config: qd.template(),
    collider_static_config: qd.template(),
    force_fields: qd.template(),
):
    """
    Update the velocity of the cells of the material point method (MPM) grid holding mass, coupling every solver.

    The momentum of a cell turns into a velocity, which gravity and the force fields accelerate, the rigid geoms, the
    smoothed particle hydrodynamics (SPH) and the position based dynamics (PBD) particles collide with, and the
    boundary of the solver clamps.
    """
    for i_x, i_y, i_z, i_b in qd.ndrange(
        mpm_state.grid.mass.shape[1],
        mpm_state.grid.mass.shape[2],
        mpm_state.grid.mass.shape[3],
        mpm_state.grid.mass.shape[4],
    ):
        if mpm_state.grid.mass[f, i_x, i_y, i_z, i_b] > coupler_info.EPS[None]:
            substep_dt = mpm_info.substep_dt[None]
            dx = mpm_info.dx[None]

            # Momentum to velocity, plus gravity
            vel_mpm = (1 / mpm_state.grid.mass[f, i_x, i_y, i_z, i_b]) * mpm_state.grid.vel_in[f, i_x, i_y, i_z, i_b]
            vel_mpm += substep_dt * mpm_info.gravity[i_b]

            pos = (qd.Vector([i_x, i_y, i_z], dt=gs.qd_int) + mpm_info.grid_offset[None]) * dx
            mass_mpm = mpm_state.grid.mass[f, i_x, i_y, i_z, i_b] / mpm_info.particle_volume_scale[None]

            # External force fields
            for i_ff in qd.static(range(len(force_fields.fields))):
                vel_mpm += force_fields.fields[i_ff].get_acc(pos, vel_mpm, t, i=-1) * substep_dt

            # MPM <-> Rigid
            if qd.static(coupler_static_config.enable_rigid_mpm):
                vel_mpm = func_collide_with_rigid(
                    i_b,
                    pos,
                    vel_mpm,
                    mass_mpm,
                    geoms_state,
                    links_state,
                    geoms_info,
                    rigid_info,
                    sdf_info,
                    collider_static_config,
                )

            # MPM <-> SPH, through the particles of the cells of the hash grid around the lower corner of the cell
            if qd.static(coupler_static_config.enable_mpm_sph):
                stencil_size = coupler_info.mpm_sph_stencil_size[None]
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
                    vel_old = vel_mpm
                    vel_mpm = sph_vel / n_colliding_particles

                    # MPM -> SPH
                    delta_mv = mass_mpm * (vel_mpm - vel_old)
                    for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                        slot_idx = func_hash_grid_to_slot(base + offset, sph_state.hash_grid)
                        slot_start = sph_state.hash_grid.slot_start[slot_idx, i_b]
                        for i in range(slot_start, slot_start + sph_state.hash_grid.slot_size[slot_idx, i_b]):
                            if qd.abs(pos - sph_state.particles_reordered.pos[i, i_b]).max() < dx * 0.5:
                                sph_state.particles_reordered.vel[i, i_b] = (
                                    sph_state.particles_reordered.vel[i, i_b]
                                    - delta_mv / sph_state.particles_info_reordered.mass[i, i_b]
                                )

            # MPM <-> PBD, likewise
            if qd.static(coupler_static_config.enable_mpm_pbd):
                stencil_size = coupler_info.mpm_pbd_stencil_size[None]
                base = func_hash_pos_to_grid(pos - 0.5 * dx, pbd_state.hash_grid)

                # PBD -> MPM
                pbd_vel = qd.Vector.zero(gs.qd_float, 3)
                n_colliding_particles = 0
                for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                    slot_idx = func_hash_grid_to_slot(base + offset, pbd_state.hash_grid)
                    slot_start = pbd_state.hash_grid.slot_start[slot_idx, i_b]
                    for i in range(slot_start, slot_start + pbd_state.hash_grid.slot_size[slot_idx, i_b]):
                        if qd.abs(pos - pbd_state.particles_reordered.pos[i, i_b]).max() < dx * 0.5:
                            pbd_vel += pbd_state.particles_reordered.vel[i, i_b]
                            n_colliding_particles += 1
                if n_colliding_particles > 0:
                    vel_old = vel_mpm
                    vel_mpm = pbd_vel / n_colliding_particles

                    # MPM -> PBD
                    delta_mv = mass_mpm * (vel_mpm - vel_old)
                    for offset in qd.grouped(qd.ndrange(stencil_size, stencil_size, stencil_size)):
                        slot_idx = func_hash_grid_to_slot(base + offset, pbd_state.hash_grid)
                        slot_start = pbd_state.hash_grid.slot_start[slot_idx, i_b]
                        for i in range(slot_start, slot_start + pbd_state.hash_grid.slot_size[slot_idx, i_b]):
                            if qd.abs(pos - pbd_state.particles_reordered.pos[i, i_b]).max() < dx * 0.5:
                                if pbd_state.particles_reordered.free[i, i_b]:
                                    pbd_state.particles_reordered.vel[i, i_b] = (
                                        pbd_state.particles_reordered.vel[i, i_b]
                                        - delta_mv / pbd_state.particles_info_reordered.mass[i, i_b]
                                    )

            # MPM boundary
            _, mpm_state.grid.vel_out[f, i_x, i_y, i_z, i_b] = func_impose_cube_boundary(
                pos, vel_mpm, mpm_info.boundary
            )


@qd.kernel
def kernel_fem_surface_force(
    f: int,
    fem_state: array_class.FEMState,
    mpm_state: array_class.MPMState,
    sph_state: array_class.SPHState,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    fem_info: array_class.FEMInfo,
    mpm_info: array_class.MPMInfo,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    coupler_info: array_class.LegacyCouplerInfo,
    coupler_static_config: qd.template(),
    collider_static_config: qd.template(),
):
    """
    Couple the vertices of the surface triangles of the finite element method (FEM) entities with the other solvers.

    Each vertex of a surface triangle collides with the rigid geoms, exchanges momentum with the cells of the material
    point method (MPM) grid within one cell of the plane of the triangle, and with the smoothed particle hydrodynamics
    (SPH) particles of the cells of its hash grid around it. The floor is imposed on the surface vertices last.
    """
    for i_s, i_b in qd.ndrange(fem_info.surface.tri2el.shape[0], fem_state.vertices.pos.shape[2]):
        i_e = fem_info.surface.tri2el[i_s]
        mass = fem_info.elements.mass_scaled[i_e] / fem_info.vol_scale[None]
        tri2v = fem_info.surface.tri2v[i_s]

        i_v0, i_v1, i_v2 = tri2v[0], tri2v[1], tri2v[2]
        p1 = fem_state.vertices.pos[f, i_v0, i_b]
        p2 = fem_state.vertices.pos[f, i_v1, i_b]
        p3 = fem_state.vertices.pos[f, i_v2, i_b]
        surface_normal = qd.math.cross(p2 - p1, p3 - p1)
        surface_normal = surface_normal / surface_normal.norm(coupler_info.EPS[None])

        # FEM <-> Rigid, on the surface vertices alone, the mass of the element spread evenly over them
        if qd.static(coupler_static_config.enable_rigid_fem):
            for j in range(3):
                i_v = tri2v[j]
                fem_state.vertices.vel[f + 1, i_v, i_b] = func_collide_with_rigid(
                    i_b,
                    fem_state.vertices.pos[f, i_v, i_b],
                    fem_state.vertices.vel[f + 1, i_v, i_b],
                    mass / 3.0,
                    geoms_state,
                    links_state,
                    geoms_info,
                    rigid_info,
                    sdf_info,
                    collider_static_config,
                )

        # FEM <-> MPM, with the grid rather than the particles, after the boundary of the MPM grid is imposed. The
        # surface is searched for from the vertices, which spares searching it from every particle.
        if qd.static(coupler_static_config.enable_fem_mpm):
            dx = mpm_info.dx[None]
            inv_dx = mpm_info.inv_dx[None]
            grid_offset = mpm_info.grid_offset[None]
            for j in range(3):
                i_v = tri2v[j]
                pos = fem_state.vertices.pos[f, i_v, i_b]
                vel_fem_sv = fem_state.vertices.vel[f + 1, i_v, i_b]
                mass_fem_sv = mass / 4.0

                mass_mpm = gs.qd_float(0.0)
                mpm_base, _, mpm_w = func_quadratic_bspline_stencil(pos, inv_dx)
                new_vel_fem_sv = vel_fem_sv
                for mpm_offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                    cell_ijk = mpm_base - grid_offset + mpm_offset
                    i_x, i_y, i_z = cell_ijk[0], cell_ijk[1], cell_ijk[2]
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

                        if mass_mpm_at_cell > coupler_info.EPS[None]:
                            # The momentum is exchanged in proportion to the mass of the whole cell
                            delta_mpm_vel_at_cell = (
                                vel_fem_sv * mpm_weight - mpm_state.grid.vel_out[f, i_x, i_y, i_z, i_b]
                            ) * (mpm_grid_mass / mass_fem_sv)
                            mpm_state.grid.vel_out[f, i_x, i_y, i_z, i_b] += delta_mpm_vel_at_cell
                            new_vel_fem_sv -= delta_mpm_vel_at_cell * mass_mpm_at_cell / mass_fem_sv

                # MPM -> FEM
                if mass_mpm > coupler_info.EPS[None]:
                    fem_state.vertices.vel[f + 1, i_v, i_b] = new_vel_fem_sv

        # FEM <-> SPH, through the particles of the cells of the hash grid of SPH around the vertex
        if qd.static(coupler_static_config.enable_fem_sph):
            dx = sph_state.hash_grid.cell_size[None]
            for j in range(3):
                i_v = tri2v[j]
                pos = fem_state.vertices.pos[f, i_v, i_b]
                vel_fem_sv = fem_state.vertices.vel[f + 1, i_v, i_b]
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

                if n_colliding_particles > 0:
                    # The tangential velocity is left out
                    vel_old = vel_fem_sv
                    vel_fem_sv = (sph_vel / n_colliding_particles).dot(surface_normal) * surface_normal

                    # FEM -> SPH
                    delta_mv = mass_fem_sv * (vel_fem_sv - vel_old)
                    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
                        slot_idx = func_hash_grid_to_slot(base + offset, sph_state.hash_grid)
                        slot_start = sph_state.hash_grid.slot_start[slot_idx, i_b]
                        for k in range(slot_start, slot_start + sph_state.hash_grid.slot_size[slot_idx, i_b]):
                            if qd.abs(pos - sph_state.particles_reordered.pos[k, i_b]).max() < dx * 0.5:
                                sph_state.particles_reordered.vel[k, i_b] = (
                                    sph_state.particles_reordered.vel[k, i_b]
                                    - delta_mv / sph_state.particles_info_reordered.mass[k, i_b]
                                )

                    fem_state.vertices.vel[f + 1, i_v, i_b] = vel_fem_sv

    # The boundary condition runs once per surface vertex: the reverse pass of an overwrite of the velocity of a
    # vertex by each of the surface triangles sharing it would race on its adjoint
    for i_sv, i_b in qd.ndrange(fem_info.surface.vertices_idx.shape[0], fem_state.vertices.pos.shape[2]):
        i_v = fem_info.surface.vertices_idx[i_sv]
        _, fem_state.vertices.vel[f + 1, i_v, i_b] = func_impose_floor_boundary(
            fem_state.vertices.pos[f, i_v, i_b], fem_state.vertices.vel[f + 1, i_v, i_b], fem_info.boundary
        )


@qd.kernel(fastcache=True)
def kernel_sph_rigid(
    sph_state: array_class.SPHState,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    sph_info: array_class.SPHInfo,
    geoms_info: array_class.GeomsInfo,
    links_info: array_class.LinksInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
):
    """Collide the sorted smoothed particle hydrodynamics particles with the coupled rigid geoms.

    See func_collide_with_rigid_geom_pressure.
    """
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            for i_g in range(geoms_info.needs_coup.shape[0]):
                if geoms_info.needs_coup[i_g]:
                    sph_state.particles_reordered.vel[i_p, i_b] = func_collide_with_rigid_geom_pressure(
                        i_g,
                        i_b,
                        sph_state.particles_reordered.pos[i_p, i_b],
                        sph_state.particles_reordered.vel[i_p, i_b],
                        sph_state.particles_info_reordered.mass[i_p, i_b],
                        sph_state.particles_reordered.p[i_p, i_b],
                        geoms_state,
                        links_state,
                        sph_info,
                        geoms_info,
                        links_info,
                        rigid_info,
                        sdf_info,
                        rigid_config,
                        collider_static_config,
                    )


@qd.kernel(fastcache=True)
def kernel_pbd_rigid_collide(
    pbd_state: array_class.PBDState,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    pbd_info: array_class.PBDInfo,
    geoms_info: array_class.GeomsInfo,
    rigid_info: array_class.RigidInfo,
    sdf_info: array_class.SDFInfo,
    collider_static_config: qd.template(),
):
    """Push the sorted position based dynamics particles out of the coupled rigid geoms.

    See func_pbd_collide_with_rigid_geom.
    """
    for i_p, i_b in qd.ndrange(pbd_state.particles_reordered.pos.shape[0], pbd_state.particles_reordered.pos.shape[1]):
        if pbd_state.particles_reordered.is_active[i_p, i_b]:
            for i_g in range(geoms_info.needs_coup.shape[0]):
                if geoms_info.needs_coup[i_g]:
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
                        geoms_state,
                        links_state,
                        pbd_info,
                        geoms_info,
                        rigid_info,
                        sdf_info,
                        collider_static_config,
                    )


@qd.kernel(fastcache=True)
def kernel_attach_pbd_to_rigid_link(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: int,
    pbd_state: array_class.PBDState,
    links_state: array_class.LinksState,
    coupler_info: array_class.LegacyCouplerInfo,
):
    """Carry some position based dynamics (PBD) particles along with a link, where they lie now in its frame."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        local_pos = qd_inv_transform_by_trans_quat(
            pbd_state.particles.pos[i_p, i_b], links_state.pos[link_idx, i_b], links_state.quat[link_idx, i_b]
        )
        pbd_state.particles.free[i_p, i_b] = False
        coupler_info.pbd_attach_link_idx[i_p, i_b] = link_idx
        coupler_info.pbd_attach_local_pos[i_p, i_b] = local_pos


@qd.kernel(fastcache=True)
def kernel_detach_pbd_from_rigid_link(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), coupler_info: array_class.LegacyCouplerInfo
):
    """Release some position based dynamics (PBD) particles from the link carrying them."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        coupler_info.pbd_attach_link_idx[i_p, i_b] = -1
        coupler_info.pbd_attach_local_pos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_drive_pbd_particles_by_link(
    clamped_inv_dt: float,
    pbd_state: array_class.PBDState,
    links_state: array_class.LinksState,
    coupler_info: array_class.LegacyCouplerInfo,
):
    """
    Drive the position based dynamics (PBD) particles carried by a link toward the point of the link they follow.

    A particle takes the velocity of that point plus the velocity catching up with it over `1 / clamped_inv_dt`. It
    runs after both the rigid and the PBD solvers update, so the correction lags by one substep.
    """
    for i_p, i_b in qd.ndrange(pbd_state.particles.pos.shape[0], pbd_state.particles.pos.shape[1]):
        i_l = coupler_info.pbd_attach_link_idx[i_p, i_b]
        if i_l >= 0:
            target_world_pos = qd_transform_by_trans_quat(
                coupler_info.pbd_attach_local_pos[i_p, i_b], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
            )
            link_com_in_world = links_state.root_COM[i_l, i_b] + links_state.i_pos[i_l, i_b]
            world_arm = target_world_pos - link_com_in_world
            target_world_vel = links_state.cd_vel[i_l, i_b] + links_state.cd_ang[i_l, i_b].cross(world_arm)

            i_r = pbd_state.particles.reordered_idx[i_p, i_b]
            pos_correction = target_world_pos - pbd_state.particles_reordered.pos[i_r, i_b]
            pbd_state.particles_reordered.vel[i_r, i_b] = pos_correction * clamped_inv_dt + target_world_vel
