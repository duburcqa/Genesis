"""The couplings of the rigid solver with the finite element method (FEM) solver."""

from typing import Annotated


import quadrants as qd

from genesis.engine.pipeline import Enabled
from genesis.engine.solvers.fem_solver import (
    VERTICES,
    FEMSolver,
    FEMState,
    FEMVertexConstraintsInfo,
    func_set_constraint_target,
)
from genesis.engine.solvers.rigid.data import DynInfo, DynState, LinksState, RigidInfo
from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
from genesis.engine.step import Coupling
from genesis.options.solvers import LegacyCouplerOptions
from genesis.utils.data import ENVS, Data, DataKind, field
from genesis.utils.geom import qd_transform_by_quat, qd_transform_quat_by_quat
from genesis.utils.sdf import SDF, SDFInfo

from .fem_surface import FEMSurface, SurfaceCoupling
from .rigid_contact import func_collide_with_rigid


class RigidFEMAttachmentInfo(Data, kind=DataKind.INFO, shape=(VERTICES, ENVS)):
    """
    The rigid links the targets of the constrained finite element method (FEM) vertices follow.

    The target of a vertex following the link `link_idx` (-1 for none) keeps the offset `link_offset_pos` of the vertex
    from the link, rotated by the rotation of the link since `link_init_quat`.
    """

    link_idx: qd.Tensor = field(int)
    link_offset_pos: qd.Tensor = field((float, 3))
    link_init_quat: qd.Tensor = field((float, 4))


@qd.kernel(fastcache=True)
def kernel_attach_fem_vertices(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: int,
    fem_state: FEMState,
    links_state: LinksState,
    rigid_fem_attachment_info: RigidFEMAttachmentInfo,
):
    """Make the targets of some vertices follow a link, -1 for none, keeping the offset of each vertex from it."""
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_values' in base_solver.py
        if i_v >= 0:
            rigid_fem_attachment_info.link_idx[i_v, i_b] = link_idx
            if link_idx >= 0:
                link_pos = links_state.pos[link_idx, i_b]
                rigid_fem_attachment_info.link_offset_pos[i_v, i_b] = fem_state.vertices.pos[0, i_v, i_b] - link_pos
                rigid_fem_attachment_info.link_init_quat[i_v, i_b] = links_state.quat[link_idx, i_b]


@qd.kernel(fastcache=True)
def kernel_update_fem_attachment(
    vertex_constraints: FEMVertexConstraintsInfo, dyn_state: DynState, rigid_fem_attachment_info: RigidFEMAttachmentInfo
):
    """Move the target of the constrained vertices following a rigid link with the link."""
    for i_v, i_b in qd.ndrange(vertex_constraints.is_constrained.shape[0], vertex_constraints.is_constrained.shape[1]):
        i_l = rigid_fem_attachment_info.link_idx[i_v, i_b]
        if vertex_constraints.is_constrained[i_v, i_b] and i_l >= 0:
            offset_quat = qd_transform_quat_by_quat(
                rigid_fem_attachment_info.link_init_quat[i_v, i_b], dyn_state.links.quat[i_l, i_b]
            )
            target_pos = dyn_state.links.pos[i_l, i_b] + qd_transform_by_quat(
                rigid_fem_attachment_info.link_offset_pos[i_v, i_b], offset_quat
            )
            func_set_constraint_target(i_v, i_b, target_pos, vertex_constraints)


class RigidFEMAttachment(Coupling):
    """
    Move the targets of the constrained finite element method (FEM) vertices following a rigid link with the link.

    The FEM solver constrains its vertices, which follow a link through this coupling (see
    'FEMSolver.set_vertex_constraints').
    """

    couple = kernel_update_fem_attachment

    rigid_fem_attachment_info: RigidFEMAttachmentInfo

    def __init__(self, rigid_solver: RigidSolver, fem_solver: FEMSolver, options: LegacyCouplerOptions) -> None:
        self._rigid_solver = rigid_solver
        self._fem_solver = fem_solver

    def build(self) -> None:
        self.rigid_fem_attachment_info = RigidFEMAttachmentInfo(
            sizes=dict(vertices=self._fem_solver.n_vertices, envs=self._fem_solver.sim._B)
        )
        self.rigid_fem_attachment_info.link_idx.fill(-1)

    def attach(self, verts_idx, envs_idx, link_idx: int) -> None:
        """Make the targets of some constrained vertices follow a rigid link from its current pose, -1 for none."""
        kernel_attach_fem_vertices(
            verts_idx,
            envs_idx,
            link_idx,
            self._fem_solver.fem_state,
            self._rigid_solver.dyn_state.links,
            self.rigid_fem_attachment_info,
        )


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
    rigid_config: qd.template(),
) -> qd.types.vector(3):
    """
    Return the velocity of a surface vertex colliding with the coupled rigid geoms (see func_collide_with_rigid).

    The mass of the element of its triangle is spread evenly over the vertices of the triangle.
    """
    return func_collide_with_rigid(
        f, i_b, pos, vel, mass / 3.0, dyn_state, dyn_info, rigid_info, sdf_info, rigid_config
    )


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
