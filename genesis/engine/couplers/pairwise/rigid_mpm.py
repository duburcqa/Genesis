"""The coupling of the rigid geoms with the grid of the material point method (MPM) solver."""

import quadrants as qd

import genesis.utils.array_class as array_class
import genesis.utils.sdf as sdf
from genesis.engine.pipeline import System, register_system
from genesis.engine.solvers.mpm_solver import MPMSolver
from genesis.engine.solvers.rigid.collider.collider import Collider
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.step import Step
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.data import opt_in

from .rigid_contact import func_collide_with_rigid


@register_system
class RigidMPM(System):
    """
    Collide the cells of the material point method (MPM) grid holding mass with the coupled rigid geoms.

    The particles also see the thin rigid geoms between them and the cells of their stencil, when the MPM solver
    separates them (see MPMCPIC).
    """

    def __init__(self, rigid_solver: RigidSolver, mpm_solver: MPMSolver, options: LegacyCouplerOptions) -> None:
        self._mpm_solver = mpm_solver
        # The cells and the particles collide with the rigid geoms through their SDF
        opt_in(array_class.SDFInfo)

    @classmethod
    def create(
        cls, rigid_solver: RigidSolver, mpm_solver: MPMSolver, options: LegacyCouplerOptions
    ) -> "RigidMPM | None":
        return cls(rigid_solver, mpm_solver, options) if options.rigid_mpm else None

    def compose(self, step: Step, collider: Collider) -> None:
        if self._mpm_solver.is_cpic_active:
            step.preparations.add(kernel_update_cpic_rigid_normal)
        self._mpm_solver.grid_couplings.add(func_mpm_rigid)


@qd.kernel(fastcache=True)
def kernel_update_cpic_rigid_normal(
    f: int,
    mpm_state: array_class.MPMState,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    collider_config: qd.template(),
):
    """Record the outward normal of every coupled rigid geom at the active material point method particles.

    A particle keeps the normal of the side it came from when it crosses a geom (see MPMCPIC).
    """
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            for i_g in range(mpm_state.cpic.rigid_normal.shape[1]):
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
                    if sdf_normal.dot(mpm_state.cpic.rigid_normal[i_p, i_g, i_b]) >= 0:
                        mpm_state.cpic.rigid_normal[i_p, i_g, i_b] = sdf_normal


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
