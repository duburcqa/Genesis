"""The coupling of the rigid geoms with the material point method (MPM) solver."""

from collections.abc import Iterator

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.sdf as sdf
from genesis.engine.pipeline import System, register_system
from genesis.engine.solvers.mpm_solver import MPMSolver
from genesis.engine.solvers.rigid.abd import func_collide_in_rigid_geom
from genesis.engine.solvers.rigid.collider.collider import Collider
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.state_change import StateChange, Subscriber
from genesis.engine.step import Step
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.array_class import ENVS, GEOMS, PARTICLES
from genesis.utils.data import Data, DataItem, DataKind, field, iter_data, opt_in

from .rigid_contact import func_collide_with_rigid


class RigidMPMCPIC(Data, kind=DataKind.STATE):
    """
    The thin rigid geoms separating each particle from the cells of its stencil, for Compatible Particle-In-Cell (CPIC).

    `rigid_normal` is the last outward normal of each geom seen by each particle, and `cells_geom_idx` the geom
    separating each particle from each of the 27 cells of its stencil, -1 for none.
    """

    rigid_normal: qd.Tensor = field((float, 3), shape=(PARTICLES, GEOMS, ENVS))
    cells_geom_idx: qd.Tensor = field(int, shape=(PARTICLES, 27, ENVS), kind=DataKind.SCRATCH)


@register_system
class RigidMPM(System):
    """
    Collide the cells of the material point method (MPM) grid holding mass with the coupled rigid geoms.

    With Compatible Particle-In-Cell (CPIC), which the MPM solver enables, a particle and a cell of its stencil that a
    thin rigid geom separates exchange nothing: the cell gets nothing from the particle, and gives it the velocity of the
    particle colliding with the geom instead. CPIC carries no gradient.
    """

    def __init__(self, rigid_solver: RigidSolver, mpm_solver: MPMSolver, options: LegacyCouplerOptions) -> None:
        self._mpm_solver = mpm_solver
        self._is_cpic = mpm_solver.enable_CPIC
        self.is_differentiable = not self._is_cpic
        self.rigid_mpm_cpic: RigidMPMCPIC | None = None
        if self._is_cpic:
            self.rigid_mpm_cpic = RigidMPMCPIC(
                sizes=dict(particles=mpm_solver.n_particles, geoms=max(1, rigid_solver.n_geoms), envs=mpm_solver.sim._B)
            )
        # The cells and the particles collide with the rigid geoms through their SDF
        opt_in(array_class.SDFInfo)

    @classmethod
    def create(
        cls, rigid_solver: RigidSolver, mpm_solver: MPMSolver, options: LegacyCouplerOptions
    ) -> "RigidMPM | None":
        return cls(rigid_solver, mpm_solver, options) if options.rigid_mpm else None

    def compose(self, step: Step, collider: Collider) -> None:
        self._mpm_solver.grid_couplings.add(func_mpm_rigid)
        if self._is_cpic:
            step.preparations.add(kernel_update_cpic_rigid_normal)
            self._mpm_solver.p2g.add(func_cpic_cell_separation)
            self._mpm_solver.g2p.add(func_cpic_cell_velocity)
            # The thin rigid geoms the particles saw belong to the trajectory they leave
            self._mpm_solver.subscribe(
                Subscriber(frozenset((StateChange.GEOMETRY,)), callback=lambda _, envs_idx: self.reset(envs_idx))
            )

    def reset(self, envs_idx=None) -> None:
        """Forget the thin rigid geoms the particles of some environments saw, every environment by default."""
        if envs_idx is None:
            self.rigid_mpm_cpic.rigid_normal.fill(0)
        else:
            kernel_reset_cpic_rigid_normal(self._mpm_solver.scene._sanitize_envs_idx(envs_idx), self.rigid_mpm_cpic)

    @property
    def data(self) -> Iterator[DataItem]:
        if self.rigid_mpm_cpic is not None:
            yield from iter_data(self, "rigid_mpm_cpic")


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
    collider_info: array_class.ColliderInfo,
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
        collider_info.sdf,
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
    rigid_mpm_cpic: RigidMPMCPIC,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
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
                    i_g, i_b, cell_pos, dyn_state.geoms, dyn_info.geoms, rigid_info, collider_info.sdf, collider_config
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
    rigid_mpm_cpic: RigidMPMCPIC,
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
    rigid_mpm_cpic: RigidMPMCPIC,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
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
                        collider_info.sdf,
                        collider_config,
                    )
                    # The normal of a geom a particle penetrates keeps the side the particle came from
                    if sdf_normal.dot(rigid_mpm_cpic.rigid_normal[i_p, i_g, i_b]) >= 0:
                        rigid_mpm_cpic.rigid_normal[i_p, i_g, i_b] = sdf_normal


@qd.kernel(fastcache=True)
def kernel_reset_cpic_rigid_normal(envs_idx: qd.types.ndarray(), rigid_mpm_cpic: RigidMPMCPIC):
    """Zero the normals of the thin rigid geoms seen by the particles of some environments."""
    for i_p, i_g, i_b_ in qd.ndrange(
        rigid_mpm_cpic.rigid_normal.shape[0], rigid_mpm_cpic.rigid_normal.shape[1], envs_idx.shape[0]
    ):
        i_b = envs_idx[i_b_]
        rigid_mpm_cpic.rigid_normal[i_p, i_g, i_b] = qd.Vector.zero(gs.qd_float, 3)
