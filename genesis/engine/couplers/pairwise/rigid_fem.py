"""The couplings of the rigid solver with the finite element method (FEM) solver."""

from collections.abc import Callable
from typing import Annotated

import torch

import quadrants as qd

from genesis.engine.pipeline import Enabled, register_system
from genesis.engine.solvers.fem_solver import FEMSolver
from genesis.engine.solvers.rigid.data import DynInfo, DynState, RigidInfo
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.step import Coupling
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.sdf import SDF, SDFInfo

from .fem_surface import FEMSurface, SurfaceCoupling
from .rigid_contact import func_collide_with_rigid


@register_system
class RigidFEMAttachment(Coupling):
    """
    Move the targets of the finite element method (FEM) vertices following a rigid link with the link.

    The FEM solver constrains its vertices to follow a link through it (see 'FEMSolver.set_vertex_constraints').
    """

    def __init__(self, rigid_solver: RigidSolver, fem_solver: FEMSolver) -> None:
        self._rigid_solver = rigid_solver
        self._fem_solver = fem_solver

    @property
    def couple(self) -> Callable:
        return self._fem_solver.update_linked_vertex_constraints

    def get_link_pose(self, link_idx: int, envs_idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the position and quaternion of a link in some environments, which carry no gradient."""
        links_envs_idx = envs_idx if self._rigid_solver.n_envs > 0 else None
        with torch.no_grad():
            link_pos = self._rigid_solver.get_links_pos(links_idx=link_idx, envs_idx=links_envs_idx)[..., 0, :]
            link_quat = self._rigid_solver.get_links_quat(links_idx=link_idx, envs_idx=links_envs_idx)[..., 0, :]
        return link_pos, link_quat


@qd.func
def func_fem_rigid(
    f: int,
    i_v: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    surface_normal: qd.types.vector(3),
    dyn_state: DynState,
    dyn_info: DynInfo,
    rigid_info: RigidInfo,
    sdf_info: SDFInfo,
    collider_config: qd.template(),
) -> qd.types.vector(3):
    """
    Return the velocity of a surface vertex colliding with the coupled rigid geoms (see func_collide_with_rigid).

    The mass of the element of its triangle is spread evenly over the vertices of the triangle.
    """
    return func_collide_with_rigid(
        i_b,
        pos,
        vel,
        mass / 3.0,
        dyn_state.geoms,
        dyn_state.links,
        dyn_info.geoms,
        rigid_info,
        sdf_info,
        collider_config,
    )


@register_system
class RigidFEM(SurfaceCoupling):
    """
    Collide the surface vertices of the finite element method (FEM) entities with the coupled rigid geoms.

    The mass of the element of a surface triangle is spread evenly over its vertices.
    """

    is_differentiable = False
    surface_coupling = func_fem_rigid

    def __init__(
        self,
        rigid_solver: RigidSolver,
        fem_surface: FEMSurface,
        sdf: SDF,
        options: Annotated[LegacyCouplerOptions, Enabled("rigid_fem")],
    ) -> None:
        pass
