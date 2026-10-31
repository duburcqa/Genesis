from collections.abc import Iterator

import numpy as np
import torch

import igl

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.linalg as lu
from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities.fem_entity import FEMEntity
from genesis.engine.materials import FEM
from genesis.engine.pipeline import Coupler
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import FEMSolverState
from genesis.engine.step import Step
from genesis.grad.bridge import no_grad
from genesis.utils.data import ConfigItem, DataItem, iter_data
from genesis.utils.geom import qd_transform_by_quat, qd_transform_quat_by_quat
from genesis.utils.misc import broadcast_tensor, qd_to_torch, qd_zero_grad, tensor_to_array

from .base_solver import GravityMixin, Solver, TimeBasedMixin


@qd.data_oriented
class FEMMaterialDispatcher:
    """
    Dispatcher of the elements to the constitutive model of their material.

    The kernels take it as a template, which compiles one branch per distinct material of the scene, the index of a
    material being its position in `materials`.
    """

    def __init__(self, materials: "list[FEM.Base]"):
        self.materials = tuple(materials)

    @qd.func
    def update_stress(
        self,
        material_idx: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        """Return the first Piola-Kirchhoff stress of the material."""
        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i_mat in qd.static(range(len(self.materials))):
            if material_idx == i_mat:
                stress = self.materials[i_mat].update_stress(mu, lam, J, F, actu, m_dir)
        return stress

    @qd.func
    def pre_compute(self, material_idx: int, F: qd.types.matrix(3, 3)):
        """Return the rotation of the deformation gradient the energy of the material linearizes around."""
        R = qd.Matrix.identity(gs.qd_float, 3)
        for i_mat in qd.static(range(len(self.materials))):
            if qd.static(self.materials[i_mat].has_rotation):
                if material_idx == i_mat:
                    R = self.materials[i_mat].pre_compute(F)
        return R

    @qd.func
    def compute_energy_gradient(
        self,
        material_idx: int,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        R: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
        hessian: qd.template(),
        is_hessian_ready: bool,
    ):
        """
        Return the energy density of the material and its gradient, writing the blocks of its Hessian.

        The blocks go into `hessian`. The Hessian of a material whose Hessian does not depend on the deformation is
        written once, the solve passing `is_hessian_ready` once it is.
        """
        energy = gs.qd_float(0.0)
        gradient = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i_mat in qd.static(range(len(self.materials))):
            if material_idx == i_mat:
                is_hessian_skipped = False
                if qd.static(self.materials[i_mat].hessian_invariant):
                    is_hessian_skipped = is_hessian_ready
                if is_hessian_skipped:
                    energy, gradient = self.materials[i_mat].compute_energy_gradient(mu, lam, J, F, R, actu, m_dir)
                else:
                    energy, gradient = self.materials[i_mat].compute_energy_gradient_hessian(
                        i_e, i_b, mu, lam, J, F, R, actu, m_dir, hessian
                    )
        return energy, gradient

    @qd.func
    def compute_energy(
        self,
        material_idx: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        R: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        """Return the energy density of the material."""
        energy = gs.qd_float(0.0)
        for i_mat in qd.static(range(len(self.materials))):
            if material_idx == i_mat:
                energy = self.materials[i_mat].compute_energy(mu, lam, J, F, R, actu, m_dir)
        return energy


class FEMSolver(GravityMixin, TimeBasedMixin, Solver):
    material_cls = FEM.Base
    has_frames = True

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
        self._damping = options.damping
        self._use_implicit_solver = options.use_implicit_solver
        self._n_newton_iterations = options.n_newton_iterations
        self._newton_dx_threshold = options.newton_dx_threshold
        self._n_pcg_iterations = options.n_pcg_iterations
        self._pcg_threshold = options.pcg_threshold
        self._n_linesearch_iterations = options.n_linesearch_iterations
        self._linesearch_c = options.linesearch_c
        self._linesearch_tau = options.linesearch_tau
        self._damping_alpha = options.damping_alpha
        self._damping_beta = options.damping_beta
        self._enable_vertex_constraints = options.enable_vertex_constraints

        # See 'vol_scale' in FEMInfo
        self._vol_scale = 1e4

        # The distinct materials of the entities, the kernels dispatching each element to the one its index names
        self._materials: list[FEM.Base] = []

        # boundary
        self.boundary = FloorBoundary(height=self._floor_height)

        self.fem_state: array_class.FEMState | None = None
        self.fem_info: array_class.FEMInfo | None = None
        self.fem_static_config: array_class.FEMStaticConfig | None = None
        self.vertex_constraints: array_class.FEMVertexConstraintsInfo | None = None
        # See 'newton_vertex_constraints'
        self._newton_vertex_constraints: array_class.FEMVertexConstraintsInfo | None = None
        self._material_dispatcher: FEMMaterialDispatcher | None = None
        self._n_surface_elements = 0
        # Whether the Hessian of the materials whose Hessian does not depend on the deformation is computed already
        self._is_hessian_ready = False

    def reset_grad(self):
        # Every adjoint the backward pass writes accumulates, so a pass starts from zeros
        if not self._sim.requires_grad:
            return
        qd_zero_grad(self.fem_state.vertices)
        qd_zero_grad(self.fem_state.actu)
        qd_zero_grad(self.fem_state.explicit_corners)
        # One flush for the zeroing batch, see qd_zero_grad in misc.py
        if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
            torch.mps.synchronize()

    def build(self):
        super().build()

        self.n_envs = self.sim.n_envs
        self._B = self.sim._B

        # The surface triangles of every entity, with the element each one belongs to (itself for a cloth, whose
        # elements are its surface triangles), and the corners of the tetrahedra, as i_e * 4 + i_corner, touching each
        # vertex in increasing order
        surfaces_tri2v, surfaces_tri2el, elements_el2v = [], [], []
        for entity in self._entities:
            surfaces_tri2v.append(entity.surface_triangles + entity.v_start)
            if isinstance(entity.material, FEM.Cloth):
                surfaces_tri2el.append(entity.s_start + np.arange(entity.n_surfaces, dtype=gs.np_int))
            else:
                surfaces_tri2el.append(entity.surface_triangles_element + entity.el_start)
                elements_el2v.append((entity.el_start, entity.elems + entity.v_start))
        tri2v = np.concatenate(surfaces_tri2v, dtype=gs.np_int) if surfaces_tri2v else np.zeros((0, 3), gs.np_int)
        tri2el = np.concatenate(surfaces_tri2el, dtype=gs.np_int) if surfaces_tri2el else np.zeros(0, gs.np_int)
        el2v = np.full((self.n_elements, 4), -1, dtype=gs.np_int)
        for el_start, entity_el2v in elements_el2v:
            el2v[el_start : el_start + len(entity_el2v)] = entity_el2v
        corners = np.flatnonzero(el2v.reshape(-1) >= 0)
        corners_vert = el2v.reshape(-1)[corners]
        vertices_corner = corners[np.argsort(corners_vert, kind="stable")].astype(gs.np_int, copy=False)
        vertices_corner_start = np.zeros(self.n_vertices + 1, dtype=gs.np_int)
        vertices_corner_start[1:] = np.cumsum(np.bincount(corners_vert, minlength=self.n_vertices))

        # The vertices of the surface triangles, their lumped surface mass, and the tetrahedra touching them
        surface_vertices_idx = np.unique(tri2v)
        is_vertex_on_surface = np.zeros(self.n_vertices, dtype=np.bool_)
        is_vertex_on_surface[surface_vertices_idx] = True
        is_element_on_surface = np.any(is_vertex_on_surface[np.maximum(el2v, 0)] & (el2v >= 0), axis=1)
        surface_elements_idx = np.flatnonzero(is_element_on_surface).astype(gs.np_int, copy=False)
        surface_vertices_mass = np.zeros(0, dtype=gs.np_float)
        if self.is_active:
            verts = np.concatenate(
                [tensor_to_array(entity.init_positions, dtype=gs.np_float) for entity in self._entities]
            )
            surface_vertices_mapping = np.full(self.n_vertices, -1, dtype=gs.np_int)
            surface_vertices_mapping[surface_vertices_idx] = np.arange(len(surface_vertices_idx))
            mass = igl.massmatrix(verts[surface_vertices_idx], surface_vertices_mapping[tri2v])
            surface_vertices_mass = mass.diagonal().astype(gs.np_float, copy=False)

        # The couplers take the data of the solver whether it holds entities or not
        self.fem_static_config = array_class.FEMStaticConfig(
            enable_vertex_constraints=self._enable_vertex_constraints,
            has_rotation=any([material.has_rotation for material in self._materials]),
            requires_grad=self._sim.requires_grad,
        )
        sizes = dict(envs=self._B, vertices=self.n_vertices, elements=self.n_elements, vverts=self.n_vverts)
        # The arrays of the integrator the solver runs alone, or none of them without entities
        unused = {None: True}
        if self.is_active:
            is_implicit = self._use_implicit_solver
            unused = {
                array_class.FEMState.explicit_corners: is_implicit,
                array_class.FEMState.newton: not is_implicit,
                array_class.FEMState.newton.rotation: not (is_implicit and self.fem_static_config.has_rotation),
            }
        self.fem_state = array_class.FEMState(
            sizes={
                **sizes,
                "frames": self.n_frames,
                "corner_frames": self.n_frames - 1 if self._sim.requires_grad else 1,
                array_class.FEMState.newton.sums_v: array_class.FixedOrderSums.sizes(self.n_vertices, 2),
                array_class.FEMState.newton.sums_el: array_class.FixedOrderSums.sizes(self.n_elements, 1),
            },
            unused=unused,
            requires_grad=self._sim.requires_grad,
        )
        self.fem_info = array_class.FEMInfo(
            sizes=dict(
                sizes,
                vfaces=self.n_vfaces,
                corners=len(vertices_corner),
                surfaces=self.n_surfaces,
                surface_vertices=len(surface_vertices_idx),
                surface_elements=len(surface_elements_idx),
            ),
            values=dict(
                boundary=dict(height=self.boundary.height, restitution=self.boundary.restitution),
                substep_dt=self.substep_dt,
                damping=self.damping,
                damping_alpha=self._damping_alpha,
                damping_beta=self._damping_beta,
                pcg_threshold=self._pcg_threshold,
                linesearch_c=self._linesearch_c,
                linesearch_tau=self._linesearch_tau,
                vol_scale=self.vol_scale,
                EPS=gs.EPS,
            ),
            # The constants are filled whether the solver holds entities or not, since the couplers read them
            unused={
                (
                    array_class.FEMInfo.elements,
                    array_class.FEMInfo.vertices,
                    array_class.FEMInfo.surface,
                    array_class.FEMInfo.vverts,
                    array_class.FEMInfo.gravity,
                ): not self.is_active
            },
        )
        self._n_surface_elements = len(surface_elements_idx)
        self._material_dispatcher = FEMMaterialDispatcher(self._materials)
        self._is_hessian_ready = False
        # The explicit solver allocates the constraints the first time a vertex is constrained, unless they are enabled
        # (see 'set_vertex_constraints')
        constraints_sizes = dict(envs=self._B, vertices=self.n_vertices)
        if self.is_active and self._enable_vertex_constraints:
            self.vertex_constraints = array_class.FEMVertexConstraintsInfo(sizes=constraints_sizes)
            self.vertex_constraints.link_idx.fill(-1)
        self._newton_vertex_constraints = self.vertex_constraints
        if self._newton_vertex_constraints is None:
            self._newton_vertex_constraints = array_class.FEMVertexConstraintsInfo(
                sizes=constraints_sizes, unused={None: True}
            )
        self._build_gravity(self.fem_info.gravity)

        if self.is_active:
            surface = self.fem_info.surface
            if len(tri2v):
                surface.tri2v.from_numpy(tri2v)
                surface.tri2el.from_numpy(tri2el)
                surface.vertices_idx.from_numpy(surface_vertices_idx)
                surface.vertices_mass.from_numpy(surface_vertices_mass)
            if len(surface_elements_idx):
                surface.elements_idx.from_numpy(surface_elements_idx)
            self.fem_info.vertices.corner_start.from_numpy(vertices_corner_start)
            if len(vertices_corner):
                self.fem_info.vertices.corner.from_numpy(vertices_corner)

            for entity in self._entities:
                entity._add_to_solver()

            kernel_compute_vertices_mass(self.fem_info)

    @property
    def is_active(self):
        # Counted from the entities rather than from what build recorded, so the answer stands before it: the rate
        # every solver integrates at is settled from the active ones, and that is settled before any of them builds.
        return self.n_elements > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "FEMEntity":
        # A material equal to one already registered shares its index, and so the kernels it compiles into
        for material_registered in self._materials:
            if material == material_registered:
                material.idx = material_registered.idx
                break
        else:
            material.idx = len(self._materials)
            self._materials.append(material)

        # create entity
        entity = FEMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            el_start=self.n_elements,
            s_start=self.n_surfaces,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )

        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def compose(self, step: Step, coupler: Coupler | None) -> None:
        """Add the substep of the solver to the step, its dynamics left to a coupler computing them itself if any."""
        if coupler is None or not coupler.is_simulating_fem:
            step.dynamics.add(self.substep_pre_coupling)
        step.integrations.add(self.substep_post_coupling)

    def substep_pre_coupling(self, f):
        if not self.is_active:
            return
        if self._use_implicit_solver:
            if self.fem_static_config.has_rotation:
                kernel_precompute_material_data(f, self.fem_state, self.fem_info, self._material_dispatcher)
            kernel_init_pos_and_inertia(
                f, self._newton_vertex_constraints, self.fem_state, self.fem_info, self.fem_static_config
            )
            # Newton iterations, each one a preconditioned conjugate gradient solve followed by a backtracking line
            # search (see https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm)
            for _ in range(self._n_newton_iterations):
                kernel_compute_ele_hessian_gradient(
                    f,
                    self.fem_state,
                    self.fem_info,
                    self.fem_static_config,
                    self._material_dispatcher,
                    self._is_hessian_ready,
                )
                self._is_hessian_ready = True
                kernel_accumulate_vertex_force_preconditioner(
                    f, self._newton_vertex_constraints, self.fem_state, self.fem_info, self.fem_static_config
                )

                kernel_init_pcg_solve(self.fem_state, self.fem_info)
                for _ in range(self._n_pcg_iterations):
                    kernel_one_pcg_iter(
                        self._newton_vertex_constraints, self.fem_state, self.fem_info, self.fem_static_config
                    )

                if self._n_linesearch_iterations <= 0:
                    kernel_skip_linesearch(f, self.fem_state)
                else:
                    kernel_init_linesearch(f, self.fem_state, self.fem_info)
                    for _ in range(self._n_linesearch_iterations):
                        kernel_one_linesearch_iter(
                            f,
                            self._newton_vertex_constraints,
                            self.fem_state,
                            self.fem_info,
                            self.fem_static_config,
                            self._material_dispatcher,
                        )
            kernel_compute_vertices_vel(f, self.fem_state, self.fem_info)
        else:
            kernel_copy_frame(f, f + 1, self.fem_state)
            kernel_compute_elements_force(
                f, self.fem_state, self.fem_info, self.fem_static_config, self._material_dispatcher
            )
            kernel_update_vertices_vel(f, self.fem_state, self.fem_info, self.fem_static_config)
            if self.vertex_constraints is not None:
                kernel_apply_soft_constraints(f, self.vertex_constraints, self.fem_state, self.fem_info)

    def substep_post_coupling(self, f):
        if not self.is_active:
            return
        kernel_compute_pos(f, self.fem_state, self.fem_info, self.fem_static_config)
        if self.vertex_constraints is not None and not self._use_implicit_solver:
            kernel_apply_hard_constraints(f, self.vertex_constraints, self.fem_state)

    def copy_frame(self, source, target):
        kernel_copy_frame(source, target, self.fem_state)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- gradient ---------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_differentiable(self) -> bool:
        return not self._use_implicit_solver

    @property
    def data(self) -> Iterator[DataItem | ConfigItem]:
        for name in ("fem_static_config", "fem_state", "fem_info"):
            yield from iter_data(self, name)
        if self.vertex_constraints is not None:
            yield from iter_data(self, "vertex_constraints")

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            kernel_set_state(f, state.pos, state.vel, self.fem_state)

    def get_state(self, f):
        if not self.is_active:
            return None
        vertices = self.fem_state.vertices
        pos, vel = self._read_data((vertices.pos, vertices.vel), None, None, copy=True)
        return FEMSolverState(self._scene, pos, vel)

    def get_state_render(self, f):
        """
        Refresh and return the render geometry of every visual geom, laid out contiguously.

        Returns
        -------
        tuple
            (vverts_pos, vverts_uvs, vfaces_indices) - environment-offset render vertex positions with shape
            (n_vverts, B), their UV coordinates, and the render triangles in global render vertex space.
        """
        if not self.is_active or self.n_vverts == 0:
            return None, None, None

        kernel_get_state_render(f, self.fem_state, self.fem_info)
        return self.fem_state.vverts_render, self.fem_info.vverts.uvs, self.fem_info.vverts.vfaces_indices

    def get_forces(self):
        """
        Get forces on all vertices.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        if not self.is_active:
            return None

        return qd_to_torch(self.fem_state.force, transpose=True, copy=True)

    def add_elements(self, f, material_idx, mat_mu, mat_lam, mat_rho, mat_friction_mu, v_start, el_start, verts, elems):
        """Add the tetrahedra of an entity at rest to the given frame, with the parameters of its material."""
        # The vertices of every tetrahedron are ordered so that its rest shape matrix has a negative determinant
        rest_shape = verts[elems[:, :3]] - verts[elems[:, 3:4]]
        if (np.linalg.det(rest_shape) >= 0.0).any():
            gs.raise_exception(
                "The order of vertices in the tetrahedral elements is not correct. Please check the input mesh or the "
                "FEM solver implementation."
            )
        kernel_add_elements(
            f,
            material_idx,
            v_start,
            el_start,
            mat_mu,
            mat_lam,
            mat_rho,
            mat_friction_mu,
            verts,
            elems,
            self.fem_state,
            self.fem_info,
        )

    def add_cloth(self, f, v_start, verts):
        """
        Add the vertices of a cloth at rest to the given frame, for position tracking and coupling only.

        The IPC coupler owns the elements and the mass of a cloth, so its vertices hold placeholder values.
        """
        kernel_add_cloth(f, v_start, verts, self.fem_state, self.fem_info)

    def add_vverts(self, vvert_start, vface_start, v_start, verts_idx, uvs, vfaces):
        kernel_add_vverts(vvert_start, vface_start, v_start, verts_idx, uvs, vfaces, self.fem_info)

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_verts_pos(self, poss, verts_idx=None, envs_idx=None):
        """
        Set the position of some vertices.

        Parameters
        ----------
        poss : array_like, shape ([M,] [N,] 3)
            Target position of each vertex.
        verts_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the vertices, once for every environment or one row per environment. If None, all vertices
            will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        poss, verts_idx, envs_idx = self._sanitize_io_variables(
            poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        # Between two steps, the state sits in frame 0 (see 'Solver.has_frames')
        kernel_set_verts_pos(f=0, verts_idx=verts_idx, envs_idx=envs_idx, poss=poss, fem_state=self.fem_state)

    def get_verts_pos(self, verts_idx=None, envs_idx=None):
        """Get the position of some vertices (see 'set_verts_pos')."""
        poss, verts_idx, envs_idx = self._sanitize_io_variables(
            None, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        # See 'set_verts_pos'
        kernel_get_verts_pos(f=0, verts_idx=verts_idx, envs_idx=envs_idx, poss=poss, fem_state=self.fem_state)
        return poss[0] if self._scene.n_envs == 0 else poss

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_verts_vel(self, vels, verts_idx=None, envs_idx=None):
        """Set the velocity of some vertices (see 'set_verts_pos')."""
        vels, verts_idx, envs_idx = self._sanitize_io_variables(
            vels, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        # See 'set_verts_pos'
        kernel_set_verts_vel(f=0, verts_idx=verts_idx, envs_idx=envs_idx, vels=vels, fem_state=self.fem_state)

    @gs.assert_built
    @mutates()
    def set_elements_actu(self, actus, elements_idx=None, envs_idx=None, *, n_groups=1):
        """
        Set the actuation of some elements, each element taking the actuation of its muscle group.

        Parameters
        ----------
        actus : array_like, shape ([M,] [N,] [n_groups,])
            Actuation of each muscle group, for each element.
        elements_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the elements (see 'set_verts_pos'). If None, all elements will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        n_groups : int, optional
            The number of muscle groups. Defaults to 1.
        """
        actus, elements_idx, envs_idx = self._sanitize_io_variables(
            actus, elements_idx, self.n_elements, "elements_idx", envs_idx, (n_groups,), batched_idx=True
        )
        # See 'set_verts_pos'
        kernel_set_elements_actu(
            f=0,
            elements_idx=elements_idx,
            envs_idx=envs_idx,
            actus=actus,
            fem_state=self.fem_state,
            fem_info=self.fem_info,
        )

    @mutates(StateChange.MODEL)
    def set_elements_muscle_group(self, muscle_group, elements_idx=None):
        """Set the muscle group of some elements, shared by every environment."""
        muscle_group, elements_idx, _ = self._sanitize_io_variables(
            muscle_group, elements_idx, self.n_elements, "elements_idx", dtype=gs.tc_int, batched=False
        )
        kernel_set_elements_muscle_group(elements_idx, muscle_group, self.fem_info)

    @mutates(StateChange.MODEL)
    def set_elements_muscle_direction(self, muscle_direction, elements_idx=None):
        """Set the unit direction of the muscle fibers of some elements, shared by every environment."""
        muscle_direction, elements_idx, _ = self._sanitize_io_variables(
            muscle_direction, elements_idx, self.n_elements, "elements_idx", element_shape=(3,), batched=False
        )
        kernel_set_elements_muscle_direction(elements_idx, muscle_direction, self.fem_info)

    def get_elements_el2v(self, elements_idx=None):
        """Get the global index of the four vertices of some elements."""
        el2v, elements_idx, _ = self._sanitize_io_variables(
            None, elements_idx, self.n_elements, "elements_idx", element_shape=(4,), dtype=gs.tc_int, batched=False
        )
        kernel_get_elements_el2v(elements_idx, el2v, self.fem_info)
        return el2v

    # ------------------------------------------------------------------------------------
    # -------------------------------- vertex constraints --------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_vertex_constraints(
        self, verts_idx=None, target_poss=None, link_idx=-1, is_soft_constraint=False, stiffness=0.0, envs_idx=None
    ):
        """
        Constrain some vertices to target positions, or to follow a rigid link.

        Parameters
        ----------
        verts_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the vertices (see 'set_verts_pos'). If None, all vertices will be considered.
        target_poss : None | array_like, shape ([M,] [N,] 3), optional
            Target position of each vertex. If None, the current positions are used. Defaults to None.
        link_idx : int, optional
            Global index of the rigid link the vertices follow, keeping their offset to it, or -1 for none.
        is_soft_constraint : bool, optional
            Whether to pull the vertices toward their targets with critically damped springs, rather than setting their
            position and zeroing their velocity. Defaults to False.
        stiffness : float, optional
            Stiffness of the springs of a soft constraint. Defaults to 0.0.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        if self._use_implicit_solver and not self._enable_vertex_constraints:
            gs.raise_exception(
                "This feature is disabled. Please set 'enable_vertex_constraints=True' when using FEM implicit solver."
            )
        if self._sim.coupler is not None and self._sim.coupler.is_simulating_fem:
            gs.raise_exception(
                "Vertex constraints are not supported by a coupler computing the dynamics of FEM entities."
            )
        if self.vertex_constraints is None:
            self.vertex_constraints = array_class.FEMVertexConstraintsInfo(
                sizes=dict(vertices=self.n_vertices, envs=self._B)
            )
            self.vertex_constraints.link_idx.fill(-1)

        is_target_current = target_poss is None
        target_poss, verts_idx, envs_idx = self._sanitize_io_variables(
            target_poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        if is_target_current:
            # See 'set_verts_pos'
            kernel_get_verts_pos(
                f=0, verts_idx=verts_idx, envs_idx=envs_idx, poss=target_poss, fem_state=self.fem_state
            )

        if link_idx < 0:
            link_init_pos = torch.zeros((len(envs_idx), 3), dtype=gs.tc_float, device=gs.device)
            link_init_quat = torch.zeros((len(envs_idx), 4), dtype=gs.tc_float, device=gs.device)
        else:
            # Imported here to break the import cycle between the couplings and the solvers they couple
            from genesis.engine.couplers.pairwise import RigidFEMAttachment

            attachment = self._sim.get_system(RigidFEMAttachment)
            if attachment is None:
                gs.raise_exception("Constraining vertices to follow a link requires a scene simulating rigid entities.")
            # The attachment is a configuration of the constraint, which carries no gradient
            link_init_pos, link_init_quat = attachment.get_link_pose(link_idx, envs_idx)
            link_init_pos = broadcast_tensor(link_init_pos, gs.tc_float, (len(envs_idx), 3)).contiguous()
            link_init_quat = broadcast_tensor(link_init_quat, gs.tc_float, (len(envs_idx), 4)).contiguous()

        # See 'set_verts_pos'
        kernel_set_vertex_constraints(
            f=0,
            verts_idx=verts_idx,
            envs_idx=envs_idx,
            link_idx=link_idx,
            stiffness=stiffness,
            target_poss=target_poss,
            link_init_pos=link_init_pos,
            link_init_quat=link_init_quat,
            vertex_constraints=self.vertex_constraints,
            fem_state=self.fem_state,
            is_soft_constraint=is_soft_constraint,
        )

    @mutates(StateChange.MODEL)
    def update_constraint_targets(self, target_poss, verts_idx=None, envs_idx=None):
        """Update the target position of some constrained vertices (see 'set_vertex_constraints')."""
        if target_poss is None:
            gs.raise_exception("The target positions of the constraints must be specified.")
        if self.vertex_constraints is None:
            gs.logger.warning("Ignoring update_constraint_targets; constraints have not been initialized.")
            return
        target_poss, verts_idx, envs_idx = self._sanitize_io_variables(
            target_poss, verts_idx, self.n_vertices, "verts_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_update_constraint_targets(verts_idx, envs_idx, target_poss, self.vertex_constraints)

    @mutates(StateChange.MODEL)
    def remove_vertex_constraints(self, verts_idx=None, envs_idx=None):
        """Remove the constraints of some vertices (see 'set_vertex_constraints'), of all of them if None."""
        if self.vertex_constraints is None:
            gs.logger.warning("Ignoring remove_vertex_constraints; constraints have not been initialized.")
            return
        if verts_idx is None and envs_idx is None:
            self.vertex_constraints.is_constrained.fill(False)
            return
        _, verts_idx, envs_idx = self._sanitize_io_variables(
            None, verts_idx, self.n_vertices, "verts_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        kernel_remove_specific_constraints(verts_idx, envs_idx, self.vertex_constraints)

    def update_linked_vertex_constraints(self, dyn_state: array_class.DynState):
        """Move the target of the vertices following a rigid link with the link."""
        if self.vertex_constraints is not None:
            kernel_update_linked_vertex_constraints(self.vertex_constraints, dyn_state.links)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def floor_height(self):
        return self._floor_height

    @property
    def damping(self):
        return self._damping

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_elements(self):
        return sum([entity.n_elements for entity in self._entities])

    @property
    def n_surfaces(self):
        return sum([entity.n_surfaces for entity in self.entities])

    @property
    def n_vverts(self):
        return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_vfaces(self):
        return sum([entity.n_vfaces for entity in self._entities])

    @property
    def vol_scale(self):
        return self._vol_scale

    @property
    def n_surface_elements(self):
        return self._n_surface_elements

    @property
    def use_implicit_solver(self):
        return self._use_implicit_solver

    @property
    def newton_vertex_constraints(self):
        """The vertex constraints the implicit solver reads, a placeholder when they are disabled."""
        return self._newton_vertex_constraints


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_compute_ele_J_F(f: int, i_e: int, i_b: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """Return the determinant and the deformation gradient of an element in a frame."""
    i_v0, i_v1, i_v2, i_v3 = fem_info.elements.el2v[i_e]
    pos_v0 = fem_state.vertices.pos[f, i_v0, i_b]
    pos_v1 = fem_state.vertices.pos[f, i_v1, i_b]
    pos_v2 = fem_state.vertices.pos[f, i_v2, i_b]
    pos_v3 = fem_state.vertices.pos[f, i_v3, i_b]
    D = qd.Matrix.cols([pos_v0 - pos_v3, pos_v1 - pos_v3, pos_v2 - pos_v3])
    F = D @ fem_info.elements.B[i_e]
    J = F.determinant()
    return J, F


@qd.func
def func_compute_element_mapping_matrix(
    i_e: int,
    i_b: int,
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
    fem_info: array_class.FEMInfo,
    static_config: qd.template(),
):
    """
    Return the 4x3 matrix mapping the positions of the vertices of an element to its deformation gradient.

    The rows of the constrained vertices are zero, which removes them from the Newton solve.
    """
    B = fem_info.elements.B[i_e]
    S = qd.Matrix.zero(gs.qd_float, 4, 3)
    S[:3, :] = B
    S[3, :] = -B[0, :] - B[1, :] - B[2, :]

    if qd.static(static_config.enable_vertex_constraints):
        i_vs = fem_info.elements.el2v[i_e]
        for i in qd.static(range(4)):
            if vertex_constraints.is_constrained[i_vs[i], i_b]:
                S[i, :] = qd.Vector.zero(gs.qd_float, 3)
    return S


@qd.func
def func_damping_force_term(
    f: int, i_e: int, i_b: int, S: qd.types.matrix(4, 3), fem_state: array_class.FEMState, fem_info: array_class.FEMInfo
):
    """
    Return the displacement of an element over the substep in deformation gradient space, and its Hessian product.

    The displacement dx of the vertices maps through `S` (see func_compute_element_mapping_matrix), giving S^T dx and
    H S^T dx.
    """
    i_vs = fem_info.elements.el2v[i_e]
    x_diff = qd.Vector.zero(gs.qd_float, 12)
    for i in qd.static(range(4)):
        i_v = i_vs[i]
        x_diff[i * 3 : i * 3 + 3] = fem_state.vertices.pos[f + 1, i_v, i_b] - fem_state.vertices.pos[f, i_v, i_b]
    St_x_diff = qd.Vector.zero(gs.qd_float, 9)
    for i, j in qd.static(qd.ndrange(3, 4)):
        St_x_diff[i * 3 : i * 3 + 3] += S[j, i] * x_diff[j * 3 : j * 3 + 3]

    H_St_x_diff = qd.Vector.zero(gs.qd_float, 9)
    for i, j in qd.static(qd.ndrange(3, 3)):
        H_St_x_diff[i * 3 : i * 3 + 3] += fem_state.newton.hessian[i, j, i_e, i_b] @ St_x_diff[j * 3 : j * 3 + 3]
    return St_x_diff, H_St_x_diff


@qd.kernel
def kernel_compute_elements_force(
    f: int,
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
    static_config: qd.template(),
    material_dispatcher: qd.template(),
):
    """Compute the internal force of every element of the explicit solver, and its contribution to each vertex."""
    for i_e, i_b in qd.ndrange(fem_info.elements.el2v.shape[0], fem_state.vertices.pos.shape[2]):
        J, F = func_compute_ele_J_F(f, i_e, i_b, fem_state, fem_info)
        stress = material_dispatcher.update_stress(
            fem_info.elements.material_idx[i_e],
            fem_info.elements.mu[i_e],
            fem_info.elements.lam[i_e],
            J,
            F,
            fem_state.actu[f, i_e, i_b],
            fem_info.elements.muscle_direction[i_e],
        )

        # The vertices sum the contributions of their corners in kernel_update_vertices_vel
        i_f = 0
        if qd.static(static_config.requires_grad):
            i_f = f
        substep_dt = fem_info.substep_dt[None]
        mass_scaled = fem_info.elements.mass_scaled[i_e]
        H_scaled = -fem_info.elements.V_scaled[i_e] * stress @ fem_info.elements.B[i_e].transpose()
        force_sum = qd.Vector.zero(gs.qd_float, 3)
        for k in qd.static(range(3)):
            force_scaled = H_scaled[:, k]
            fem_state.explicit_corners.force[i_f, i_e, k, i_b] = force_scaled
            fem_state.explicit_corners.dv[i_f, i_e, k, i_b] = substep_dt * force_scaled / mass_scaled
            force_sum += force_scaled
        fem_state.explicit_corners.force[i_f, i_e, 3, i_b] = -force_sum
        fem_state.explicit_corners.dv[i_f, i_e, 3, i_b] = -substep_dt * force_sum / mass_scaled


@qd.kernel(fastcache=True)
def kernel_update_vertices_vel(
    f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo, static_config: qd.template()
):
    """Add the internal force of the elements to the velocity of the vertices, then damp it and add gravity."""
    for i_v, i_b in qd.ndrange(fem_state.vertices.pos.shape[1], fem_state.vertices.pos.shape[2]):
        i_f = 0
        if qd.static(static_config.requires_grad):
            i_f = f
        # The internal force and velocity change of the vertex, summed over its corners (see
        # kernel_compute_elements_force), the internal force being stored so that it can be read out
        force = qd.Vector.zero(gs.qd_float, 3)
        dv = qd.Vector.zero(gs.qd_float, 3)
        for i_entry in range(fem_info.vertices.corner_start[i_v], fem_info.vertices.corner_start[i_v + 1]):
            i_corner = fem_info.vertices.corner[i_entry]
            force += fem_state.explicit_corners.force[i_f, i_corner // 4, i_corner % 4, i_b]
            dv += fem_state.explicit_corners.dv[i_f, i_corner // 4, i_corner % 4, i_b]
        fem_state.force[i_v, i_b] = force
        fem_state.vertices.vel[f + 1, i_v, i_b] += dv
        # The damping applies to the velocity the internal force gives, which damps gravity too, so gravity comes after
        substep_dt = fem_info.substep_dt[None]
        fem_state.vertices.vel[f + 1, i_v, i_b] *= qd.exp(-substep_dt * fem_info.damping[None])
        fem_state.vertices.vel[f + 1, i_v, i_b] += substep_dt * fem_info.gravity[i_b]


@qd.kernel(fastcache=True)
def kernel_compute_pos(
    f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo, static_config: qd.template()
):
    """Advect the vertices into the next frame, carrying the actuation of the elements over with gradients."""
    for i_v, i_b in qd.ndrange(fem_state.vertices.pos.shape[1], fem_state.vertices.pos.shape[2]):
        fem_state.vertices.pos[f + 1, i_v, i_b] = (
            fem_info.substep_dt[None] * fem_state.vertices.vel[f + 1, i_v, i_b] + fem_state.vertices.pos[f, i_v, i_b]
        )
    # With gradients, the substeps of a step run in successive frames, to which the actuation set in frame 0 carries
    # over
    if qd.static(static_config.requires_grad):
        for i_e, i_b in qd.ndrange(fem_state.actu.shape[1], fem_state.actu.shape[2]):
            fem_state.actu[f + 1, i_e, i_b] = fem_state.actu[f, i_e, i_b]


@qd.kernel
def kernel_precompute_material_data(
    f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo, material_dispatcher: qd.template()
):
    """Compute the rotation the linear corotated materials linearize around, from the deformation of the last frame."""
    for i_e, i_b in qd.ndrange(fem_state.newton.rotation.shape[0], fem_state.newton.rotation.shape[1]):
        J, F = func_compute_ele_J_F(f, i_e, i_b, fem_state, fem_info)
        fem_state.newton.rotation[i_e, i_b] = material_dispatcher.pre_compute(fem_info.elements.material_idx[i_e], F)


@qd.kernel(fastcache=True)
def kernel_init_pos_and_inertia(
    f: int,
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
    static_config: qd.template(),
):
    """Start the Newton solve from the last position, toward the inertial target of the free vertices."""
    for i_v, i_b in qd.ndrange(fem_state.vertices.pos.shape[1], fem_state.vertices.pos.shape[2]):
        substep_dt = fem_info.substep_dt[None]
        is_constrained = False
        target_pos = qd.Vector.zero(gs.qd_float, 3)
        if qd.static(static_config.enable_vertex_constraints):
            is_constrained = vertex_constraints.is_constrained[i_v, i_b]
            target_pos = vertex_constraints.target_pos[i_v, i_b]
        if is_constrained:
            fem_state.vertices.pos[f + 1, i_v, i_b] = target_pos
            fem_state.newton.inertia[i_v, i_b] = target_pos
        else:
            fem_state.newton.inertia[i_v, i_b] = (
                fem_state.vertices.pos[f, i_v, i_b]
                + fem_state.vertices.vel[f, i_v, i_b] * substep_dt
                + fem_info.gravity[i_b] * substep_dt**2
            )
            fem_state.vertices.pos[f + 1, i_v, i_b] = fem_state.vertices.pos[f, i_v, i_b]


@qd.kernel
def kernel_compute_ele_hessian_gradient(
    f: int,
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
    static_config: qd.template(),
    material_dispatcher: qd.template(),
    is_hessian_ready: bool,
):
    """Compute the elastic energy density of the elements, its gradient and its Hessian, at the current iterate."""
    for i_e, i_b in qd.ndrange(fem_state.newton.energy.shape[0], fem_state.newton.energy.shape[1]):
        J, F = func_compute_ele_J_F(f + 1, i_e, i_b, fem_state, fem_info)
        R = qd.Matrix.identity(gs.qd_float, 3)
        if qd.static(static_config.has_rotation):
            R = fem_state.newton.rotation[i_e, i_b]
        fem_state.newton.energy[i_e, i_b], fem_state.newton.gradient[i_e, i_b] = (
            material_dispatcher.compute_energy_gradient(
                fem_info.elements.material_idx[i_e],
                i_e,
                i_b,
                fem_info.elements.mu[i_e],
                fem_info.elements.lam[i_e],
                J,
                F,
                R,
                fem_state.actu[f, i_e, i_b],
                fem_info.elements.muscle_direction[i_e],
                fem_state.newton.hessian,
                is_hessian_ready,
            )
        )


@qd.kernel(fastcache=True)
def kernel_accumulate_vertex_force_preconditioner(
    f: int,
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
    static_config: qd.template(),
):
    """Compute the force on the vertices and the diagonal Hessian blocks, whose inverse preconditions the solve."""
    # Elastic force and diagonal 3-by-3 block of the Hessian, per corner of the elements
    for i_e, i_b in qd.ndrange(fem_state.newton.energy.shape[0], fem_state.newton.energy.shape[1]):
        damping_beta_over_dt = fem_info.damping_beta[None] / fem_info.substep_dt[None]
        V = fem_info.elements.V[i_e]
        S = func_compute_element_mapping_matrix(i_e, i_b, vertex_constraints, fem_info, static_config)
        force = -V * fem_state.newton.gradient[i_e, i_b] @ S.transpose()

        S_H_St_x_diff = qd.Vector.zero(gs.qd_float, 12)
        if fem_info.damping_beta[None] > fem_info.EPS[None]:
            St_x_diff, H_St_x_diff = func_damping_force_term(f, i_e, i_b, S, fem_state, fem_info)
            for i, j in qd.static(qd.ndrange(4, 3)):
                S_H_St_x_diff[i * 3 : i * 3 + 3] += S[i, j] * H_St_x_diff[j * 3 : j * 3 + 3]

        for k in qd.static(range(4)):
            fem_state.newton.corners.force[i_e, k, i_b] = (
                force[:, k] - damping_beta_over_dt * V * S_H_St_x_diff[k * 3 : k * 3 + 3]
            )
            diag3x3 = qd.Matrix.zero(gs.qd_float, 3, 3)
            for i, j in qd.static(qd.ndrange(3, 3)):
                diag3x3 += S[k, i] * S[k, j] * fem_state.newton.hessian[i, j, i_e, i_b]
            fem_state.newton.corners.diag3x3[i_e, k, i_b] = V * (damping_beta_over_dt + 1.0) * diag3x3

    # Inertia, then the corners touching the vertex, and the inverse of the diagonal block as preconditioner
    for i_v, i_b in qd.ndrange(fem_state.newton.inertia.shape[0], fem_state.newton.inertia.shape[1]):
        damping_alpha_dt = fem_info.damping_alpha[None] * fem_info.substep_dt[None]
        mass_over_dt2 = fem_info.vertices.mass_over_dt2[i_v]
        force = -mass_over_dt2 * (
            (fem_state.vertices.pos[f + 1, i_v, i_b] - fem_state.newton.inertia[i_v, i_b])
            + (fem_state.vertices.pos[f + 1, i_v, i_b] - fem_state.vertices.pos[f, i_v, i_b]) * damping_alpha_dt
        )
        diag3x3 = qd.Matrix.identity(gs.qd_float, 3) * (mass_over_dt2 * (damping_alpha_dt + 1.0))
        for i_entry in range(fem_info.vertices.corner_start[i_v], fem_info.vertices.corner_start[i_v + 1]):
            i_corner = fem_info.vertices.corner[i_entry]
            force += fem_state.newton.corners.force[i_corner // 4, i_corner % 4, i_b]
            diag3x3 += fem_state.newton.corners.diag3x3[i_corner // 4, i_corner % 4, i_b]
        fem_state.force[i_v, i_b] = force
        fem_state.newton.diag3x3[i_v, i_b] = diag3x3
        fem_state.newton.prec[i_v, i_b] = diag3x3.inverse()


@qd.kernel(fastcache=True)
def kernel_init_pcg_solve(fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """Start the conjugate gradient solve of the Newton step from zero, with the force as residual."""
    for i_b in range(fem_state.newton.is_pcg_active.shape[0]):
        fem_state.newton.is_pcg_active[i_b] = True
    for i_v, i_b in qd.ndrange(fem_state.newton.x.shape[0], fem_state.newton.x.shape[1]):
        r = fem_state.force[i_v, i_b]
        z = fem_state.newton.prec[i_v, i_b] @ r
        fem_state.newton.x[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
        fem_state.newton.r[i_v, i_b] = r
        fem_state.newton.z[i_v, i_b] = z
        fem_state.newton.p[i_v, i_b] = z
        fem_state.newton.sums_v.terms[i_v, 0, i_b] = r.dot(r)
        fem_state.newton.sums_v.terms[i_v, 1, i_b] = r.dot(z)
    lu.func_reduce_terms(fem_state.newton.sums_v, fem_state.newton.is_pcg_active)
    for i_b in range(fem_state.newton.is_pcg_active.shape[0]):
        fem_state.newton.rTr[i_b] = lu.func_sum_chunks(i_b, k=0, sums=fem_state.newton.sums_v)
        fem_state.newton.rTz[i_b] = lu.func_sum_chunks(i_b, k=1, sums=fem_state.newton.sums_v)
        fem_state.newton.is_pcg_active[i_b] = fem_state.newton.rTr[i_b] > fem_info.pcg_threshold[None]


@qd.kernel(fastcache=True)
def kernel_one_pcg_iter(
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
    static_config: qd.template(),
):
    """Run one iteration of the preconditioned conjugate gradient solve of the Newton step."""
    # A * p, per corner of the elements
    for i_e, i_b in qd.ndrange(fem_state.newton.energy.shape[0], fem_state.newton.energy.shape[1]):
        if fem_state.newton.is_pcg_active[i_b]:
            damping_beta_factor = fem_info.damping_beta[None] / fem_info.substep_dt[None] + 1.0
            V = fem_info.elements.V[i_e]
            i_vs = fem_info.elements.el2v[i_e]
            S = func_compute_element_mapping_matrix(i_e, i_b, vertex_constraints, fem_info, static_config)

            p9 = qd.Vector.zero(gs.qd_float, 9)
            for i, j in qd.static(qd.ndrange(3, 4)):
                i_v = i_vs[j]
                p9[i * 3 : i * 3 + 3] = p9[i * 3 : i * 3 + 3] + S[j, i] * fem_state.newton.p[i_v, i_b]

            new_p9 = qd.Vector.zero(gs.qd_float, 9)
            for i, j in qd.static(qd.ndrange(3, 3)):
                new_p9[i * 3 : i * 3 + 3] = (
                    new_p9[i * 3 : i * 3 + 3] + fem_state.newton.hessian[i, j, i_e, i_b] @ p9[j * 3 : j * 3 + 3]
                )

            for i in qd.static(range(4)):
                fem_state.newton.corners.Ap[i_e, i, i_b] = (
                    (S[i, 0] * new_p9[0:3] + S[i, 1] * new_p9[3:6] + S[i, 2] * new_p9[6:9]) * V * damping_beta_factor
                )

    # A * p per vertex, and the terms of p^T A p
    for i_v, i_b in qd.ndrange(fem_state.newton.x.shape[0], fem_state.newton.x.shape[1]):
        if fem_state.newton.is_pcg_active[i_b]:
            damping_alpha_factor = fem_info.damping_alpha[None] * fem_info.substep_dt[None] + 1.0
            Ap = fem_info.vertices.mass_over_dt2[i_v] * damping_alpha_factor * fem_state.newton.p[i_v, i_b]
            for i_entry in range(fem_info.vertices.corner_start[i_v], fem_info.vertices.corner_start[i_v + 1]):
                i_corner = fem_info.vertices.corner[i_entry]
                Ap += fem_state.newton.corners.Ap[i_corner // 4, i_corner % 4, i_b]
            fem_state.newton.Ap[i_v, i_b] = Ap
            fem_state.newton.sums_v.terms[i_v, 0, i_b] = fem_state.newton.p[i_v, i_b].dot(Ap)
    lu.func_reduce_terms(fem_state.newton.sums_v, fem_state.newton.is_pcg_active)

    # Step size, then x, r, z and the terms of r^T r and r^T z
    for i_b in range(fem_state.newton.is_pcg_active.shape[0]):
        if fem_state.newton.is_pcg_active[i_b]:
            fem_state.newton.pTAp[i_b] = lu.func_sum_chunks(i_b, k=0, sums=fem_state.newton.sums_v)
            fem_state.newton.alpha[i_b] = fem_state.newton.rTz[i_b] / fem_state.newton.pTAp[i_b]
    for i_v, i_b in qd.ndrange(fem_state.newton.x.shape[0], fem_state.newton.x.shape[1]):
        if fem_state.newton.is_pcg_active[i_b]:
            alpha = fem_state.newton.alpha[i_b]
            fem_state.newton.x[i_v, i_b] += alpha * fem_state.newton.p[i_v, i_b]
            fem_state.newton.r[i_v, i_b] -= alpha * fem_state.newton.Ap[i_v, i_b]
            r = fem_state.newton.r[i_v, i_b]
            z = fem_state.newton.prec[i_v, i_b] @ r
            fem_state.newton.z[i_v, i_b] = z
            fem_state.newton.sums_v.terms[i_v, 0, i_b] = r.dot(r)
            fem_state.newton.sums_v.terms[i_v, 1, i_b] = r.dot(z)
    lu.func_reduce_terms(fem_state.newton.sums_v, fem_state.newton.is_pcg_active)

    # Convergence check, then beta, r^T r and r^T z
    for i_b in range(fem_state.newton.is_pcg_active.shape[0]):
        if fem_state.newton.is_pcg_active[i_b]:
            fem_state.newton.rTr_new[i_b] = lu.func_sum_chunks(i_b, k=0, sums=fem_state.newton.sums_v)
            fem_state.newton.rTz_new[i_b] = lu.func_sum_chunks(i_b, k=1, sums=fem_state.newton.sums_v)
            fem_state.newton.is_pcg_active[i_b] = fem_state.newton.rTr_new[i_b] > fem_info.pcg_threshold[None]
            if fem_state.newton.is_pcg_active[i_b]:
                fem_state.newton.beta[i_b] = fem_state.newton.rTz_new[i_b] / fem_state.newton.rTz[i_b]
                fem_state.newton.rTr[i_b] = fem_state.newton.rTr_new[i_b]
                fem_state.newton.rTz[i_b] = fem_state.newton.rTz_new[i_b]

    # Search direction
    for i_v, i_b in qd.ndrange(fem_state.newton.x.shape[0], fem_state.newton.x.shape[1]):
        if fem_state.newton.is_pcg_active[i_b]:
            fem_state.newton.p[i_v, i_b] = (
                fem_state.newton.z[i_v, i_b] + fem_state.newton.beta[i_b] * fem_state.newton.p[i_v, i_b]
            )


@qd.func(requires_top_level=True)
def func_sum_elements_energy(fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """Compute the partial sums of the elastic energy of the elements of each environment (see FixedOrderSums)."""
    for i_e, i_b in qd.ndrange(fem_state.newton.energy.shape[0], fem_state.newton.energy.shape[1]):
        if fem_state.newton.is_linesearch_active[i_b]:
            fem_state.newton.sums_el.terms[i_e, 0, i_b] = fem_state.newton.energy[i_e, i_b] * fem_info.elements.V[i_e]
    lu.func_reduce_terms(fem_state.newton.sums_el, fem_state.newton.is_linesearch_active)


@qd.kernel(fastcache=True)
def kernel_init_linesearch(f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """Start the line search of the Newton step from a unit step, with the energy and its slope at the iterate."""
    for i_b in range(fem_state.newton.is_linesearch_active.shape[0]):
        fem_state.newton.is_linesearch_active[i_b] = True
        fem_state.newton.step_size[i_b] = 1.0

    # Inertia, x_prev, m
    for i_v, i_b in qd.ndrange(fem_state.newton.x.shape[0], fem_state.newton.x.shape[1]):
        diff = fem_state.vertices.pos[f + 1, i_v, i_b] - fem_state.newton.inertia[i_v, i_b]
        fem_state.newton.sums_v.terms[i_v, 0, i_b] = 0.5 * fem_info.vertices.mass_over_dt2[i_v] * diff.dot(diff)
        fem_state.newton.sums_v.terms[i_v, 1, i_b] = -fem_state.newton.x[i_v, i_b].dot(fem_state.force[i_v, i_b])
        fem_state.newton.x_prev[i_v, i_b] = fem_state.vertices.pos[f + 1, i_v, i_b]
    lu.func_reduce_terms(fem_state.newton.sums_v, fem_state.newton.is_linesearch_active)
    # Elastic
    func_sum_elements_energy(fem_state, fem_info)
    for i_b in range(fem_state.newton.is_linesearch_active.shape[0]):
        inertia_energy = lu.func_sum_chunks(i_b, k=0, sums=fem_state.newton.sums_v)
        elastic_energy = lu.func_sum_chunks(i_b, k=0, sums=fem_state.newton.sums_el)
        fem_state.newton.prev_energy[i_b] = inertia_energy + elastic_energy
        fem_state.newton.m[i_b] = lu.func_sum_chunks(i_b, k=1, sums=fem_state.newton.sums_v)


@qd.kernel
def kernel_one_linesearch_iter(
    f: int,
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
    static_config: qd.template(),
    material_dispatcher: qd.template(),
):
    """Try the current step of the line search, then shrink it unless the energy decreased enough (Armijo)."""
    # Position at the step, and inertia energy
    for i_v, i_b in qd.ndrange(fem_state.newton.x.shape[0], fem_state.newton.x.shape[1]):
        if fem_state.newton.is_linesearch_active[i_b]:
            fem_state.vertices.pos[f + 1, i_v, i_b] = (
                fem_state.newton.x_prev[i_v, i_b] + fem_state.newton.step_size[i_b] * fem_state.newton.x[i_v, i_b]
            )
            mass_over_dt2 = fem_info.vertices.mass_over_dt2[i_v]
            diff = fem_state.vertices.pos[f + 1, i_v, i_b] - fem_state.newton.inertia[i_v, i_b]
            energy = 0.5 * mass_over_dt2 * diff.dot(diff)
            if fem_info.damping_alpha[None] > 0.0:
                damping_alpha_dt = fem_info.damping_alpha[None] * fem_info.substep_dt[None]
                diff = fem_state.vertices.pos[f + 1, i_v, i_b] - fem_state.vertices.pos[f, i_v, i_b]
                energy += 0.5 * mass_over_dt2 * diff.dot(diff) * damping_alpha_dt
            fem_state.newton.sums_v.terms[i_v, 0, i_b] = energy
    lu.func_reduce_terms(fem_state.newton.sums_v, fem_state.newton.is_linesearch_active)

    # Elastic energy, plus the linearized damping energy
    for i_e, i_b in qd.ndrange(fem_state.newton.energy.shape[0], fem_state.newton.energy.shape[1]):
        if fem_state.newton.is_linesearch_active[i_b]:
            J, F = func_compute_ele_J_F(f + 1, i_e, i_b, fem_state, fem_info)
            R = qd.Matrix.identity(gs.qd_float, 3)
            if qd.static(static_config.has_rotation):
                R = fem_state.newton.rotation[i_e, i_b]
            fem_state.newton.energy[i_e, i_b] = material_dispatcher.compute_energy(
                fem_info.elements.material_idx[i_e],
                fem_info.elements.mu[i_e],
                fem_info.elements.lam[i_e],
                J,
                F,
                R,
                fem_state.actu[f, i_e, i_b],
                fem_info.elements.muscle_direction[i_e],
            )
            if fem_info.damping_beta[None] > fem_info.EPS[None]:
                damping_beta_over_dt = fem_info.damping_beta[None] / fem_info.substep_dt[None]
                S = func_compute_element_mapping_matrix(i_e, i_b, vertex_constraints, fem_info, static_config)
                St_x_diff, H_St_x_diff = func_damping_force_term(f, i_e, i_b, S, fem_state, fem_info)
                fem_state.newton.energy[i_e, i_b] += 0.5 * damping_beta_over_dt * St_x_diff.dot(H_St_x_diff)
    func_sum_elements_energy(fem_state, fem_info)

    # Sufficient decrease condition
    for i_b in range(fem_state.newton.is_linesearch_active.shape[0]):
        if fem_state.newton.is_linesearch_active[i_b]:
            inertia_energy = lu.func_sum_chunks(i_b, k=0, sums=fem_state.newton.sums_v)
            elastic_energy = lu.func_sum_chunks(i_b, k=0, sums=fem_state.newton.sums_el)
            fem_state.newton.linesearch_energy[i_b] = inertia_energy + elastic_energy
            fem_state.newton.is_linesearch_active[i_b] = (
                fem_state.newton.linesearch_energy[i_b]
                > fem_state.newton.prev_energy[i_b]
                + fem_info.linesearch_c[None] * fem_state.newton.step_size[i_b] * fem_state.newton.m[i_b]
            )
            if fem_state.newton.is_linesearch_active[i_b]:
                fem_state.newton.step_size[i_b] *= fem_info.linesearch_tau[None]


@qd.kernel(fastcache=True)
def kernel_skip_linesearch(f: int, fem_state: array_class.FEMState):
    """Take the full Newton step."""
    for i_v, i_b in qd.ndrange(fem_state.newton.x.shape[0], fem_state.newton.x.shape[1]):
        fem_state.vertices.pos[f + 1, i_v, i_b] += fem_state.newton.x[i_v, i_b]


@qd.kernel(fastcache=True)
def kernel_compute_vertices_vel(f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """Derive the velocity of the vertices from their displacement over the substep."""
    for i_v, i_b in qd.ndrange(fem_state.vertices.pos.shape[1], fem_state.vertices.pos.shape[2]):
        fem_state.vertices.vel[f + 1, i_v, i_b] = (
            fem_state.vertices.pos[f + 1, i_v, i_b] - fem_state.vertices.pos[f, i_v, i_b]
        ) / fem_info.substep_dt[None]


@qd.kernel(fastcache=True)
def kernel_copy_frame(source: int, target: int, fem_state: array_class.FEMState):
    """Copy the position and velocity of the vertices from one frame to another."""
    for i_v, i_b in qd.ndrange(fem_state.vertices.pos.shape[1], fem_state.vertices.pos.shape[2]):
        fem_state.vertices.pos[target, i_v, i_b] = fem_state.vertices.pos[source, i_v, i_b]
        fem_state.vertices.vel[target, i_v, i_b] = fem_state.vertices.vel[source, i_v, i_b]


@qd.kernel(fastcache=True)
def kernel_compute_vertices_mass(fem_info: array_class.FEMInfo):
    """Lump a quarter of the mass of every element onto each of its vertices, summed in the order of corners."""
    for i_v in range(fem_info.vertices.mass.shape[0]):
        i_start = fem_info.vertices.corner_start[i_v]
        i_end = fem_info.vertices.corner_start[i_v + 1]
        if i_end > i_start:
            mass = gs.qd_float(0.0)
            for i_entry in range(i_start, i_end):
                i_e = fem_info.vertices.corner[i_entry] // 4
                mass += 0.25 * fem_info.elements.mass_scaled[i_e] / fem_info.vol_scale[None]
            fem_info.vertices.mass[i_v] = mass
            fem_info.vertices.mass_over_dt2[i_v] = mass * (1.0 / (fem_info.substep_dt[None] ** 2))
            fem_info.vertices.mass_inv[i_v] = 1.0 / mass


@qd.kernel(fastcache=True)
def kernel_add_elements(
    f: int,
    material_idx: int,
    v_start: int,
    el_start: int,
    mat_mu: float,
    mat_lam: float,
    mat_rho: float,
    mat_friction_mu: float,
    verts: qd.types.ndarray(),
    elems: qd.types.ndarray(),
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
):
    """Add the vertices and the tetrahedra of an entity at rest to a frame, with the parameters of its material."""
    for i_v, i_b in qd.ndrange(verts.shape[0], fem_state.vertices.pos.shape[2]):
        i_global = i_v + v_start
        for j in qd.static(range(3)):
            fem_state.vertices.pos[f, i_global, i_b][j] = verts[i_v, j]
        fem_state.vertices.vel[f, i_global, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_v in range(verts.shape[0]):
        fem_info.vertices.friction_mu[i_v + v_start] = mat_friction_mu

    for i_e in range(elems.shape[0]):
        i_global = i_e + el_start
        i_v0, i_v1, i_v2, i_v3 = elems[i_e, 0], elems[i_e, 1], elems[i_e, 2], elems[i_e, 3]
        a = qd.Vector([verts[i_v0, 0], verts[i_v0, 1], verts[i_v0, 2]], dt=gs.qd_float)
        b = qd.Vector([verts[i_v1, 0], verts[i_v1, 1], verts[i_v1, 2]], dt=gs.qd_float)
        c = qd.Vector([verts[i_v2, 0], verts[i_v2, 1], verts[i_v2, 2]], dt=gs.qd_float)
        d = qd.Vector([verts[i_v3, 0], verts[i_v3, 1], verts[i_v3, 2]], dt=gs.qd_float)
        B_inv = qd.Matrix.cols([a - d, b - d, c - d])
        fem_info.elements.B[i_global] = B_inv.inverse()
        V = qd.abs(B_inv.determinant()) / 6.0
        fem_info.elements.V[i_global] = V
        V_scaled = V * fem_info.vol_scale[None]
        fem_info.elements.V_scaled[i_global] = V_scaled

        fem_info.elements.el2v[i_global] = qd.Vector(
            [i_v0 + v_start, i_v1 + v_start, i_v2 + v_start, i_v3 + v_start], dt=gs.qd_int
        )
        fem_info.elements.material_idx[i_global] = material_idx
        fem_info.elements.mu[i_global] = mat_mu
        fem_info.elements.lam[i_global] = mat_lam
        fem_info.elements.friction_mu[i_global] = mat_friction_mu
        fem_info.elements.mass_scaled[i_global] = mat_rho * V_scaled
        fem_info.elements.muscle_group[i_global] = 0
        fem_info.elements.muscle_direction[i_global] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

    for i_e, i_b in qd.ndrange(elems.shape[0], fem_state.actu.shape[2]):
        fem_state.actu[f, i_e + el_start, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_add_cloth(
    f: int, v_start: int, verts: qd.types.ndarray(), fem_state: array_class.FEMState, fem_info: array_class.FEMInfo
):
    """Add the vertices of a cloth at rest to a frame, with placeholder mass and friction."""
    for i_v, i_b in qd.ndrange(verts.shape[0], fem_state.vertices.pos.shape[2]):
        i_global = i_v + v_start
        for j in qd.static(range(3)):
            fem_state.vertices.pos[f, i_global, i_b][j] = verts[i_v, j]
        fem_state.vertices.vel[f, i_global, i_b] = qd.Vector.zero(gs.qd_float, 3)

    for i_v in range(verts.shape[0]):
        i_global = i_v + v_start
        fem_info.vertices.mass[i_global] = 1.0
        fem_info.vertices.mass_over_dt2[i_global] = 0.0
        fem_info.vertices.friction_mu[i_global] = 0.0


@qd.kernel(fastcache=True)
def kernel_add_vverts(
    vvert_start: int,
    vface_start: int,
    v_start: int,
    verts_idx: qd.types.ndarray(),
    uvs: qd.types.ndarray(element_dim=1),
    vfaces: qd.types.ndarray(element_dim=1),
    fem_info: array_class.FEMInfo,
):
    """Bind the visual vertices of a visual geom to the simulated vertices they stand for, with their UVs and faces."""
    for i_vv_ in range(verts_idx.shape[0]):
        fem_info.vverts.vert_idx[i_vv_ + vvert_start] = verts_idx[i_vv_] + v_start
    for i_vv_ in range(uvs.shape[0]):
        fem_info.vverts.uvs[i_vv_ + vvert_start] = uvs[i_vv_]
    for i_vf_ in range(vfaces.shape[0]):
        fem_info.vverts.vfaces_indices[i_vf_ + vface_start] = vfaces[i_vf_] + vvert_start


@qd.kernel(fastcache=True)
def kernel_get_state_render(f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """Copy the position of the simulated vertices of a frame to the visual vertices standing for them."""
    for i_vv, i_b in qd.ndrange(fem_info.vverts.vert_idx.shape[0], fem_state.vverts_render.shape[1]):
        i_v = fem_info.vverts.vert_idx[i_vv]
        fem_state.vverts_render[i_vv, i_b] = fem_state.vertices.pos[f, i_v, i_b]


@qd.kernel(fastcache=True)
def kernel_set_state(f: int, pos: qd.types.ndarray(), vel: qd.types.ndarray(), fem_state: array_class.FEMState):
    """Write the position and velocity of every vertex of every environment into a frame."""
    for i_v, i_b in qd.ndrange(pos.shape[1], pos.shape[0]):
        for j in qd.static(range(3)):
            fem_state.vertices.pos[f, i_v, i_b][j] = pos[i_b, i_v, j]
            fem_state.vertices.vel[f, i_v, i_b][j] = vel[i_b, i_v, j]


@qd.kernel(fastcache=True)
def kernel_set_verts_pos(
    f: int,
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    fem_state: array_class.FEMState,
):
    """Set the position of some vertices in a frame."""
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        # The earlier occurrences of a repeated vertex are -1 (see 'drop_repeated_index')
        if i_v >= 0:
            for k in qd.static(range(3)):
                fem_state.vertices.pos[f, i_v, i_b][k] = poss[i_b_, i_v_, k]


@qd.kernel(fastcache=True)
def kernel_set_verts_vel(
    f: int,
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
    fem_state: array_class.FEMState,
):
    """Set the velocity of some vertices in a frame."""
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_verts_pos'
        if i_v >= 0:
            for k in qd.static(range(3)):
                fem_state.vertices.vel[f, i_v, i_b][k] = vels[i_b_, i_v_, k]


@qd.kernel(fastcache=True)
def kernel_get_verts_pos(
    f: int,
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    fem_state: array_class.FEMState,
):
    """Get the position of some vertices in a frame."""
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_v = verts_idx[i_b_, i_v_]
        i_b = envs_idx[i_b_]
        for k in qd.static(range(3)):
            poss[i_b_, i_v_, k] = fem_state.vertices.pos[f, i_v, i_b][k]


@qd.kernel(fastcache=True)
def kernel_set_elements_actu(
    f: int,
    elements_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actus: qd.types.ndarray(),
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
):
    """Set the actuation of some elements in a frame, each one taking that of its muscle group."""
    for i_e_, i_g, i_b_ in qd.ndrange(elements_idx.shape[1], actus.shape[2], envs_idx.shape[0]):
        i_e = elements_idx[i_b_, i_e_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_verts_pos'
        if i_e >= 0:
            if fem_info.elements.muscle_group[i_e] == i_g:
                fem_state.actu[f, i_e, i_b] = actus[i_b_, i_e_, i_g]


@qd.kernel(fastcache=True)
def kernel_set_elements_muscle_group(
    elements_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray(), fem_info: array_class.FEMInfo
):
    """Set the muscle group of some elements."""
    for i_e_ in range(elements_idx.shape[0]):
        i_e = elements_idx[i_e_]
        # See 'kernel_set_verts_pos'
        if i_e >= 0:
            fem_info.elements.muscle_group[i_e] = muscle_group[i_e_]


@qd.kernel(fastcache=True)
def kernel_set_elements_muscle_direction(
    elements_idx: qd.types.ndarray(), muscle_direction: qd.types.ndarray(), fem_info: array_class.FEMInfo
):
    """Set the direction of the muscle fibers of some elements."""
    for i_e_ in range(elements_idx.shape[0]):
        i_e = elements_idx[i_e_]
        # See 'kernel_set_verts_pos'
        if i_e >= 0:
            for j in qd.static(range(3)):
                fem_info.elements.muscle_direction[i_e][j] = muscle_direction[i_e_, j]


@qd.kernel(fastcache=True)
def kernel_get_elements_el2v(elements_idx: qd.types.ndarray(), el2v: qd.types.ndarray(), fem_info: array_class.FEMInfo):
    """Get the global index of the four vertices of some elements."""
    for i_e_ in range(elements_idx.shape[0]):
        i_e = elements_idx[i_e_]
        for j in qd.static(range(4)):
            el2v[i_e_, j] = fem_info.elements.el2v[i_e][j]


@qd.kernel(fastcache=True)
def kernel_update_linked_vertex_constraints(
    vertex_constraints: array_class.FEMVertexConstraintsInfo, links_state: array_class.LinksState
):
    """Move the target of the vertices following a rigid link with the link."""
    for i_v, i_b in qd.ndrange(vertex_constraints.is_constrained.shape[0], vertex_constraints.is_constrained.shape[1]):
        i_l = vertex_constraints.link_idx[i_v, i_b]
        if vertex_constraints.is_constrained[i_v, i_b] and i_l >= 0:
            offset_quat = qd_transform_quat_by_quat(
                vertex_constraints.link_init_quat[i_v, i_b], links_state.quat[i_l, i_b]
            )
            vertex_constraints.target_pos[i_v, i_b] = links_state.pos[i_l, i_b] + qd_transform_by_quat(
                vertex_constraints.link_offset_pos[i_v, i_b], offset_quat
            )


@no_grad
@qd.kernel(fastcache=True)
def kernel_apply_hard_constraints(
    f: int, vertex_constraints: array_class.FEMVertexConstraintsInfo, fem_state: array_class.FEMState
):
    """Hold the hard-constrained vertices of the next frame at their target, at rest."""
    for i_v, i_b in qd.ndrange(vertex_constraints.is_constrained.shape[0], vertex_constraints.is_constrained.shape[1]):
        if vertex_constraints.is_constrained[i_v, i_b] and not vertex_constraints.is_soft_constraint[i_v, i_b]:
            fem_state.vertices.pos[f + 1, i_v, i_b] = vertex_constraints.target_pos[i_v, i_b]
            fem_state.vertices.vel[f + 1, i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)


@no_grad
@qd.kernel(fastcache=True)
def kernel_apply_soft_constraints(
    f: int,
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
    fem_state: array_class.FEMState,
    fem_info: array_class.FEMInfo,
):
    """Pull the soft-constrained vertices of the next frame toward their target with a critically damped spring."""
    for i_v, i_b in qd.ndrange(vertex_constraints.is_constrained.shape[0], vertex_constraints.is_constrained.shape[1]):
        if vertex_constraints.is_constrained[i_v, i_b] and vertex_constraints.is_soft_constraint[i_v, i_b]:
            stiffness = vertex_constraints.stiffness[i_v, i_b]
            pos_error = fem_state.vertices.pos[f, i_v, i_b] - vertex_constraints.target_pos[i_v, i_b]
            vel_error = fem_state.vertices.vel[f + 1, i_v, i_b] - fem_state.vertices.vel[f, i_v, i_b]
            spring_force = -stiffness * pos_error
            damping_force = -2.0 * qd.math.sqrt(stiffness) * vel_error
            fem_state.vertices.vel[f + 1, i_v, i_b] += fem_info.substep_dt[None] * (spring_force + damping_force)


@qd.kernel(fastcache=True)
def kernel_set_vertex_constraints(
    f: int,
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    link_idx: int,
    stiffness: float,
    target_poss: qd.types.ndarray(),
    link_init_pos: qd.types.ndarray(),
    link_init_quat: qd.types.ndarray(),
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
    fem_state: array_class.FEMState,
    is_soft_constraint: bool,
):
    """Constrain some vertices to their targets, keeping their offset from a link of the given initial pose."""
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_b_, i_v_]
        # See 'kernel_set_verts_pos'
        if i_v >= 0:
            vertex_constraints.is_constrained[i_v, i_b] = True
            vertex_constraints.is_soft_constraint[i_v, i_b] = is_soft_constraint
            vertex_constraints.stiffness[i_v, i_b] = stiffness
            vertex_constraints.link_idx[i_v, i_b] = link_idx

            cur_pos = fem_state.vertices.pos[f, i_v, i_b]
            for j in qd.static(range(3)):
                vertex_constraints.target_pos[i_v, i_b][j] = target_poss[i_b_, i_v_, j]
                vertex_constraints.link_offset_pos[i_v, i_b][j] = cur_pos[j] - link_init_pos[i_b_, j]
            for j in qd.static(range(4)):
                vertex_constraints.link_init_quat[i_v, i_b][j] = link_init_quat[i_b_, j]


@qd.kernel(fastcache=True)
def kernel_update_constraint_targets(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    new_target_poss: qd.types.ndarray(),
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
):
    """Set the target position of some constrained vertices."""
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_b_, i_v_]
        # See 'kernel_set_verts_pos'
        if i_v >= 0:
            for j in qd.static(range(3)):
                vertex_constraints.target_pos[i_v, i_b][j] = new_target_poss[i_b_, i_v_, j]


@qd.kernel(fastcache=True)
def kernel_remove_specific_constraints(
    verts_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vertex_constraints: array_class.FEMVertexConstraintsInfo,
):
    """Release some constrained vertices."""
    for i_v_, i_b_ in qd.ndrange(verts_idx.shape[1], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_v = verts_idx[i_b_, i_v_]
        vertex_constraints.is_constrained[i_v, i_b] = False
