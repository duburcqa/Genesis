import math
from functools import partial

import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary, CubeBoundaryBox, func_impose_cube_boundary
from genesis.engine.entities import PBD2DEntity, PBD3DEntity, PBDFreeParticleEntity, PBDParticleEntity
from genesis.engine.entities.particle_entity import PARTICLES
from genesis.engine.entities.pbd_entity import PBDTetEntity
from genesis.engine.materials import PBD
from genesis.engine.solvers.base_solver import EDGES, VFACES, VVERTS
from genesis.engine.solvers.particle_solver import (
    Particles,
    ParticlesParams,
    ParticlesRender,
    VVertsRender,
    VVertsSupport,
)
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import PBDSolverState
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.geom import (
    SpatialHashGrid,
    func_hash_compute_reordered_idx,
    func_hash_grid_to_slot,
    func_hash_pos_to_grid,
)

from .particle_solver import ParticleSolver, kernel_add_vverts, kernel_get_particles_state, kernel_set_particles_state


class PBDParticles(Particles):
    """
    Runtime state of the PBD particles, one value per particle and environment.

    A particle that is not `free` follows the user or a rigid link alone. `ipos` is the position at the start of the
    substep, `dpos` the correction the constraint projections accumulate, and `lam` the multiplier of the density
    constraint. `reordered_idx` is the index of the particle once sorted by hash grid cell.
    """

    free: qd.Tensor = field(bool)
    ipos: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH)
    dpos: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH)
    lam: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    rho: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    reordered_idx: qd.Tensor = field(int, kind=DataKind.DERIVED)


class PBDParticlesParams(ParticlesParams):
    """Model parameters of the PBD particles, one value per particle (and per environment once sorted)."""

    pos_rest: qd.Tensor = field((float, 3))
    rho_rest: qd.Tensor = field(float)
    material_type: qd.Tensor = field(int)
    mu_s: qd.Tensor = field(float)
    mu_k: qd.Tensor = field(float)
    air_resistance: qd.Tensor = field(float)
    density_relaxation: qd.Tensor = field(float)
    viscosity_relaxation: qd.Tensor = field(float)


# Sizes of the PBD solver: its bending constraints and its volume constraints
INNER_EDGES = Size("inner_edges")
ELEMS = Size("elems")


class PBDEdges(Data, kind=DataKind.CONSTANT, shape=(EDGES,)):
    """The stretch constraints, one per edge between the particles `v1` and `v2`."""

    len_rest: qd.Tensor = field(float)
    stretch_compliance: qd.Tensor = field(float)
    stretch_relaxation: qd.Tensor = field(float)
    v1: qd.Tensor = field(int)
    v2: qd.Tensor = field(int)


class PBDInnerEdges(Data, kind=DataKind.CONSTANT, shape=(INNER_EDGES,)):
    """The bending constraints, one per edge `v1`-`v2` shared by the triangles `v1, v2, v3` and `v1, v2, v4`."""

    len_rest: qd.Tensor = field(float)
    bending_compliance: qd.Tensor = field(float)
    bending_relaxation: qd.Tensor = field(float)
    v1: qd.Tensor = field(int)
    v2: qd.Tensor = field(int)
    v3: qd.Tensor = field(int)
    v4: qd.Tensor = field(int)


class PBDElems(Data, kind=DataKind.CONSTANT, shape=(ELEMS,)):
    """The volume constraints, one per tetrahedron of the particles `v1` to `v4`."""

    vol_rest: qd.Tensor = field(float)
    volume_compliance: qd.Tensor = field(float)
    volume_relaxation: qd.Tensor = field(float)
    v1: qd.Tensor = field(int)
    v2: qd.Tensor = field(int)
    v3: qd.Tensor = field(int)
    v4: qd.Tensor = field(int)


class PBDState(Data, kind=DataKind.STATE):
    """
    Runtime state of the PBD solver.

    The constraint projections that look for neighbours run on the copies of the particles sorted by hash grid cell,
    `particles_reordered` and `particles_info_reordered`, which the particles take back at the end of the substep.
    """

    particles: PBDParticles
    particles_reordered: PBDParticles = field(kind=DataKind.SCRATCH)
    particles_info_reordered: PBDParticlesParams = field(shape=(PARTICLES, ENVS), kind=DataKind.SCRATCH)
    hash_grid: SpatialHashGrid
    particles_render: ParticlesRender
    vverts_render: VVertsRender


class PBDInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the PBD solver.

    `dist_scale` maps a distance between particles to the unit support of the smoothing kernels.
    """

    particles: PBDParticlesParams
    edges: PBDEdges
    inner_edges: PBDInnerEdges
    elems: PBDElems
    vverts: VVertsSupport
    vverts_uvs: qd.Tensor = field((float, 2), shape=(VVERTS,))
    vfaces_indices: qd.Tensor = field((int, 3), shape=(VFACES,))
    boundary: CubeBoundaryBox
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float, shape=())
    particle_size: qd.Tensor = field(float, shape=())
    dist_scale: qd.Tensor = field(float, shape=())
    EPS: qd.Tensor = field(float, shape=())


class PBDMaterial(gs.IntEnum):
    CLOTH = 0
    ELASTIC = 1
    LIQUID = 2
    # Particles without physics
    PARTICLE = 3


class PBDSolver(ParticleSolver):
    material_cls = PBD.Base
    MATERIAL = PBDMaterial

    pbd_state: PBDState | None = None
    pbd_info: PBDInfo | None = None

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._max_stretch_solver_iterations = options.max_stretch_solver_iterations
        self._max_bending_solver_iterations = options.max_bending_solver_iterations
        self._max_volume_solver_iterations = options.max_volume_solver_iterations
        self._max_density_solver_iterations = options.max_density_solver_iterations
        self._max_viscosity_solver_iterations = options.max_viscosity_solver_iterations
        self._hash_grid_cell_size = options.hash_grid_cell_size
        self._hash_grid_res = options._hash_grid_res

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors

        self.boundary = CubeBoundary(lower=self._lower_bound, upper=self._upper_bound)

    def build(self):
        super().build()

        self._n_fluid_particles = self.n_fluid_particles
        self._n_edges = self.n_edges
        self._n_inner_edges = self.n_inner_edges
        self._n_elems = self.n_elems
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        # The couplers take the data of the solver whether it holds entities or not
        sizes = dict(envs=self._B, particles=self._n_particles, vverts=self._n_vverts)
        unused = {None: not self.is_active}
        self.pbd_state = PBDState(
            sizes=dict(sizes, hash_slots=np.prod(self._hash_grid_res)),
            values=dict(hash_grid=dict(cell_size=self._hash_grid_cell_size, grid_res=self._hash_grid_res)),
            unused=unused,
        )
        self.pbd_info = PBDInfo(
            sizes=dict(
                sizes,
                edges=self._n_edges,
                inner_edges=self._n_inner_edges,
                elems=self._n_elems,
                vvert_supports=self._n_vvert_supports,
                vfaces=self._n_vfaces,
            ),
            values=dict(
                boundary=self.boundary.values,
                substep_dt=self.substep_dt,
                particle_size=self._particle_size,
                dist_scale=self.particle_radius / 0.4,
                EPS=gs.EPS,
            ),
            unused=unused,
        )
        self._particles = self.pbd_state.particles
        self._particles_params = self.pbd_info.particles
        self._vverts_support = self.pbd_info.vverts
        self.particles_render = self.pbd_state.particles_render
        self.vverts_render = self.pbd_state.vverts_render

        if self.is_active:
            for entity in self._entities:
                entity._add_to_solver()

        self._build_gravity(self.pbd_info.gravity)

    def add_uvs_and_faces(self, *args):
        """Add the render UVs and faces of an entity, an empty `uvs` leaving its slice of the UV buffer zeroed."""
        kernel_add_uvs_and_faces(*args, self.pbd_info)

    def add_vverts(self, *args):
        """Bind the given vverts to their supporting particles (see 'kernel_add_vverts')."""
        kernel_add_vverts(*args, self.pbd_info.vverts)

    def add_particles(self, *args, **kwargs):
        """Add the particles of an entity at rest, each property a material leaves out being given zero."""
        kernel_add_particles(*args, **kwargs, pbd_state=self.pbd_state, pbd_info=self.pbd_info)

    def add_edges(self, *args):
        """Add the stretch constraints of the edges of an entity, given by the local indices of their particles."""
        kernel_add_edges(*args, self.pbd_info)

    def add_inner_edges(self, *args):
        """Add the bending constraints of the inner edges of an entity, given by the local indices of its particles."""
        kernel_add_inner_edges(*args, self.pbd_info)

    def add_elems(self, *args):
        """Add the volume constraints of the tetrahedra of an entity, given by the local indices of their particles."""
        kernel_add_elems(*args, self.pbd_info)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "PBD2DEntity | PBD3DEntity | PBDParticleEntity | PBDFreeParticleEntity":
        # The constructor of the entity, given the start of its slices of the buffers beyond the particles
        if isinstance(material, gs.materials.PBD.Cloth):
            entity_cls = partial(
                PBD2DEntity,
                edge_start=self.n_edges,
                inner_edge_start=self.n_inner_edges,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
            )
        elif isinstance(material, gs.materials.PBD.Elastic):
            entity_cls = partial(
                PBD3DEntity,
                edge_start=self.n_edges,
                elem_start=self.n_elems,
                vvert_start=self.n_vverts,
                vface_start=self.n_vfaces,
            )
        elif isinstance(material, gs.materials.PBD.Liquid):
            entity_cls = PBDParticleEntity
        elif isinstance(material, gs.materials.PBD.Particle):
            entity_cls = PBDFreeParticleEntity
        else:
            raise NotImplementedError()

        entity = entity_cls(
            scene=self.scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            name=name,
        )
        self._entities.append(entity)

        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if self.is_active:
            kernel_apply_external_force(self._sim.cur_t, self.pbd_state, self.pbd_info, self.force_fields_dispatcher)

            # topology constraints (doesn't require spatial hashing)
            if self._n_edges > 0:
                for _ in range(self._max_stretch_solver_iterations):
                    kernel_solve_stretch(self.pbd_state, self.pbd_info)

            if self._n_inner_edges > 0:
                for _ in range(self._max_bending_solver_iterations):
                    kernel_solve_bending(self.pbd_state, self.pbd_info)

            if self._n_elems > 0:
                for _ in range(self._max_volume_solver_iterations):
                    kernel_solve_volume(self.pbd_state, self.pbd_info)

            # perform spatial hashing
            kernel_reorder_particles(self.pbd_state, self.pbd_info)

            # spatial constraints
            for _ in range(self._max_density_solver_iterations):
                kernel_solve_density(self.pbd_state, self.pbd_info)
            for _ in range(self._max_viscosity_solver_iterations):
                kernel_solve_viscosity(self.pbd_state, self.pbd_info)

            kernel_solve_collision(self.pbd_state, self.pbd_info)

            # compute effective velocity
            kernel_compute_velocity(self.pbd_state, self.pbd_info)

    def substep_post_coupling(self, f):
        if self.is_active:
            kernel_copy_from_reordered(self.pbd_state)

            # boundary collision
            kernel_solve_boundary_collision(self.pbd_state, self.pbd_info)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            particles = self.pbd_state.particles
            kernel_set_particles_state(state.pos, state.vel, state.free, particles.pos, particles.vel, particles.free)

    def get_state(self, f):
        if self.is_active:
            state = PBDSolverState(self.scene)
            particles = self.pbd_state.particles
            kernel_get_particles_state(state.pos, state.vel, state.free, particles.pos, particles.vel, particles.free)
        else:
            state = None
        return state

    @gs.assert_built
    @mutates(StateChange.DYNAMICS, differentiable=False)
    def fix_particles(self, particles_idx=None, envs_idx=None):
        """
        Fix some particles where they are, until they are released (see 'release_particles').

        Parameters
        ----------
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles (see 'set_particles_pos'). If None, all particles will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        _, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        kernel_set_particles_free(particles_idx, envs_idx, self.pbd_state, free=False)

    @gs.assert_built
    @mutates(StateChange.DYNAMICS, differentiable=False)
    def fix_particles_to_link(self, link_idx, particles_idx=None, envs_idx=None):
        """
        Attach some particles to a rigid link, which carries them along until they are released (see
        'release_particles').

        Parameters
        ----------
        link_idx : int
            Global index of the rigid link.
        particles_idx, envs_idx : see 'fix_particles'.
        """
        _, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        # Imported here to break the import cycle between the couplings and the solvers they couple
        from genesis.engine.couplers.pairwise import RigidPBD

        rigid_pbd = self._sim.get_system(RigidPBD)
        if rigid_pbd is None:
            gs.raise_exception(
                "Attaching particles to a link requires the PBD solver to be coupled with the rigid solver."
            )
        rigid_pbd.attach_particles(particles_idx, envs_idx, link_idx)

    @gs.assert_built
    @mutates(StateChange.DYNAMICS, differentiable=False)
    def release_particles(self, particles_idx=None, envs_idx=None):
        """Release some particles fixed in place or to a rigid link, which then move freely (see 'fix_particles')."""
        _, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True, skip_allocation=True
        )
        kernel_set_particles_free(particles_idx, envs_idx, self.pbd_state, free=True)
        # Imported here, see 'fix_particles_to_link'
        from genesis.engine.couplers.pairwise import RigidPBD

        rigid_pbd = self._sim.get_system(RigidPBD)
        if rigid_pbd is not None:
            rigid_pbd.detach_particles(particles_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_fluid_particles(self):
        if self.is_built:
            return self._n_fluid_particles
        return sum(entity.n_fluid_particles for entity in self._entities if isinstance(entity, PBDParticleEntity))

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum(entity.n_edges for entity in self._entities if isinstance(entity, PBDTetEntity))

    @property
    def n_inner_edges(self):
        if self.is_built:
            return self._n_inner_edges
        return sum(entity.n_inner_edges for entity in self._entities if isinstance(entity, PBD2DEntity))

    @property
    def n_elems(self):
        if self.is_built:
            return self._n_elems
        return sum(entity.n_elems for entity in self._entities if isinstance(entity, PBD3DEntity))

    @property
    def n_vvert_supports(self):
        """
        Number of particles supporting each visual vertex.
        """
        return self._n_vvert_supports

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def hash_grid_res(self):
        return self._hash_grid_res

    @property
    def hash_grid_cell_size(self):
        return self._hash_grid_cell_size


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_add_uvs_and_faces(
    vvert_start: int,
    vface_start: int,
    uvs: qd.types.ndarray(element_dim=1),
    faces: qd.types.ndarray(element_dim=1),
    pbd_info: PBDInfo,
):
    for i_vv_ in range(uvs.shape[0]):
        pbd_info.vverts_uvs[i_vv_ + vvert_start] = uvs[i_vv_]
    for i_vf_ in range(faces.shape[0]):
        pbd_info.vfaces_indices[i_vf_ + vface_start] = faces[i_vf_] + vvert_start


@qd.kernel(fastcache=True)
def kernel_add_particles(
    particle_start: int,
    active: bool,
    material_type: int,
    mass: float,
    rho_rest: float,
    mu_s: float,
    mu_k: float,
    air_resistance: float,
    density_relaxation: float,
    viscosity_relaxation: float,
    particles: qd.types.ndarray(),
    pbd_state: PBDState,
    pbd_info: PBDInfo,
):
    for i_p_ in range(particles.shape[0]):
        i_p = i_p_ + particle_start
        pbd_info.particles.pos_rest[i_p] = qd.Vector([particles[i_p_, 0], particles[i_p_, 1], particles[i_p_, 2]])
        pbd_info.particles.material_type[i_p] = material_type
        pbd_info.particles.mass[i_p] = mass
        pbd_info.particles.rho_rest[i_p] = rho_rest
        pbd_info.particles.mu_s[i_p] = mu_s
        pbd_info.particles.mu_k[i_p] = mu_k
        pbd_info.particles.air_resistance[i_p] = air_resistance
        pbd_info.particles.density_relaxation[i_p] = density_relaxation
        pbd_info.particles.viscosity_relaxation[i_p] = viscosity_relaxation

    for i_p_, i_b in qd.ndrange(particles.shape[0], pbd_state.particles.pos.shape[1]):
        i_p = i_p_ + particle_start
        pbd_state.particles.pos[i_p, i_b] = qd.Vector([particles[i_p_, 0], particles[i_p_, 1], particles[i_p_, 2]])
        pbd_state.particles.vel[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        pbd_state.particles.dpos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        pbd_state.particles.free[i_p, i_b] = True
        pbd_state.particles.is_active[i_p, i_b] = active


@qd.kernel(fastcache=True)
def kernel_add_edges(
    edge_start: int,
    particle_start: int,
    stretch_compliance: float,
    stretch_relaxation: float,
    edges: qd.types.ndarray(),
    edges_len_rest: qd.types.ndarray(),
    pbd_info: PBDInfo,
):
    for i_e_ in range(edges.shape[0]):
        i_e = i_e_ + edge_start
        pbd_info.edges.stretch_compliance[i_e] = stretch_compliance
        pbd_info.edges.stretch_relaxation[i_e] = stretch_relaxation
        pbd_info.edges.len_rest[i_e] = edges_len_rest[i_e_]
        pbd_info.edges.v1[i_e] = particle_start + edges[i_e_, 0]
        pbd_info.edges.v2[i_e] = particle_start + edges[i_e_, 1]


@qd.kernel(fastcache=True)
def kernel_add_inner_edges(
    inner_edge_start: int,
    particle_start: int,
    bending_compliance: float,
    bending_relaxation: float,
    inner_edges: qd.types.ndarray(),
    inner_edges_len_rest: qd.types.ndarray(),
    pbd_info: PBDInfo,
):
    for i_ie_ in range(inner_edges.shape[0]):
        i_ie = i_ie_ + inner_edge_start
        pbd_info.inner_edges.bending_compliance[i_ie] = bending_compliance
        pbd_info.inner_edges.bending_relaxation[i_ie] = bending_relaxation
        pbd_info.inner_edges.len_rest[i_ie] = inner_edges_len_rest[i_ie_]
        pbd_info.inner_edges.v1[i_ie] = particle_start + inner_edges[i_ie_, 0]
        pbd_info.inner_edges.v2[i_ie] = particle_start + inner_edges[i_ie_, 1]
        pbd_info.inner_edges.v3[i_ie] = particle_start + inner_edges[i_ie_, 2]
        pbd_info.inner_edges.v4[i_ie] = particle_start + inner_edges[i_ie_, 3]


@qd.kernel(fastcache=True)
def kernel_add_elems(
    elem_start: int,
    particle_start: int,
    volume_compliance: float,
    volume_relaxation: float,
    elems: qd.types.ndarray(),
    elems_vol_rest: qd.types.ndarray(),
    pbd_info: PBDInfo,
):
    for i_el_ in range(elems.shape[0]):
        i_el = i_el_ + elem_start
        pbd_info.elems.volume_compliance[i_el] = volume_compliance
        pbd_info.elems.volume_relaxation[i_el] = volume_relaxation
        pbd_info.elems.vol_rest[i_el] = elems_vol_rest[i_el_]
        pbd_info.elems.v1[i_el] = particle_start + elems[i_el_, 0]
        pbd_info.elems.v2[i_el] = particle_start + elems[i_el_, 1]
        pbd_info.elems.v3[i_el] = particle_start + elems[i_el_, 2]
        pbd_info.elems.v4[i_el] = particle_start + elems[i_el_, 3]


@qd.func
def func_poly6_scalar(d: float):
    """Return the poly6 smoothing kernel of a scaled distance between two particles."""
    result = gs.qd_float(0.0)
    if 0 < d < 1.0:
        rhs = (1.0 - d * d) * (1.0 - d * d) * (1.0 - d * d)
        result = 315.0 / (64 * math.pi) * rhs
    return result


@qd.func
def func_poly6(d_ij: qd.types.vector(3), dist_scale: float):
    """Return the poly6 smoothing kernel of the vector between two particles, of unit support once scaled."""
    return func_poly6_scalar(d_ij.norm() / dist_scale)


@qd.func
def func_spiky(d_ij: qd.types.vector(3), dist_scale: float):
    """Return the gradient of the spiky smoothing kernel of the vector between two particles, of unit support."""
    result = qd.Vector.zero(gs.qd_float, 3)
    d = d_ij.norm() / dist_scale
    if 0 < d < 1.0:
        m = (1.0 - d) * (1.0 - d)
        result = (-45.0 / math.pi * m / d) * d_ij / dist_scale
    return result


@qd.func
def func_s_corr(d_ij: qd.types.vector(3), dist_scale: float):
    """Return the artificial pressure that keeps the particles of a liquid from clustering (tensile instability)."""
    upper = func_poly6(d_ij, dist_scale)
    lower = func_poly6_scalar(0.3)
    m = upper / lower
    return -0.0001 * m * m * m * m


# FIXME: quadrants#1036 - a func takes no subclass of its struct annotation, so the neighbour search, the sort and the
# copy back of the sorted particles repeat those of the SPH solver.
@qd.func
def func_for_all_neighbors(
    i_p: int,
    i_b: int,
    ret: qd.template(),
    pbd_state: PBDState,
    pbd_info: PBDInfo,
    radius: float,
    task: qd.template(),
):
    """
    Run a task on every sorted particle closer than a radius to a sorted particle, the particle itself excluded.

    `task` takes the indices of both particles, the environment, `ret`, the state and the info, in this order, and
    accumulates its result in `ret`.
    """
    base = func_hash_pos_to_grid(pbd_state.particles_reordered.pos[i_p, i_b], pbd_state.hash_grid)
    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
        i_s = func_hash_grid_to_slot(base + offset, pbd_state.hash_grid)
        slot_start = pbd_state.hash_grid.slot_start[i_s, i_b]
        for j_p in range(slot_start, slot_start + pbd_state.hash_grid.slot_size[i_s, i_b]):
            d_ij = pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.pos[j_p, i_b]
            if i_p != j_p and d_ij.norm() < radius:
                task(i_p, j_p, i_b, ret, pbd_state, pbd_info)


@qd.kernel(fastcache=True)
def kernel_reorder_particles(pbd_state: PBDState, pbd_info: PBDInfo):
    """Sort the particles by the cell of the hash grid they lie in, into the copies the neighbour queries read."""
    func_hash_compute_reordered_idx(
        pbd_state.particles.pos, pbd_state.particles.is_active, pbd_state.particles.reordered_idx, pbd_state.hash_grid
    )

    for i_p, i_b in qd.ndrange(pbd_state.particles.pos.shape[0], pbd_state.particles.pos.shape[1]):
        pbd_state.particles_reordered.is_active[i_p, i_b] = False
    for i_p, i_b in qd.ndrange(pbd_state.particles.pos.shape[0], pbd_state.particles.pos.shape[1]):
        if pbd_state.particles.is_active[i_p, i_b]:
            i_r = pbd_state.particles.reordered_idx[i_p, i_b]
            func_copy_particle(i_p, i_r, i_b, pbd_state.particles, pbd_state.particles_reordered)
            pbd_state.particles_info_reordered.mass[i_r, i_b] = pbd_info.particles.mass[i_p]
            pbd_state.particles_info_reordered.pos_rest[i_r, i_b] = pbd_info.particles.pos_rest[i_p]
            pbd_state.particles_info_reordered.rho_rest[i_r, i_b] = pbd_info.particles.rho_rest[i_p]
            pbd_state.particles_info_reordered.material_type[i_r, i_b] = pbd_info.particles.material_type[i_p]
            pbd_state.particles_info_reordered.mu_s[i_r, i_b] = pbd_info.particles.mu_s[i_p]
            pbd_state.particles_info_reordered.mu_k[i_r, i_b] = pbd_info.particles.mu_k[i_p]
            pbd_state.particles_info_reordered.air_resistance[i_r, i_b] = pbd_info.particles.air_resistance[i_p]
            pbd_state.particles_info_reordered.density_relaxation[i_r, i_b] = pbd_info.particles.density_relaxation[i_p]
            pbd_state.particles_info_reordered.viscosity_relaxation[i_r, i_b] = pbd_info.particles.viscosity_relaxation[
                i_p
            ]
            pbd_state.particles_reordered.is_active[i_r, i_b] = True


@qd.func
def func_copy_particle(
    i_p_src: int,
    i_p_dst: int,
    i_b: int,
    particles_src: PBDParticles,
    particles_dst: PBDParticles,
):
    """Copy the dynamic state of a particle onto another slot of the same environment."""
    particles_dst.free[i_p_dst, i_b] = particles_src.free[i_p_src, i_b]
    particles_dst.pos[i_p_dst, i_b] = particles_src.pos[i_p_src, i_b]
    particles_dst.ipos[i_p_dst, i_b] = particles_src.ipos[i_p_src, i_b]
    particles_dst.dpos[i_p_dst, i_b] = particles_src.dpos[i_p_src, i_b]
    particles_dst.vel[i_p_dst, i_b] = particles_src.vel[i_p_src, i_b]
    particles_dst.lam[i_p_dst, i_b] = particles_src.lam[i_p_src, i_b]
    particles_dst.rho[i_p_dst, i_b] = particles_src.rho[i_p_src, i_b]


@qd.kernel
def kernel_apply_external_force(t: float, pbd_state: PBDState, pbd_info: PBDInfo, force_fields: qd.template()):
    """
    Advance the particles over the substep, the free ones accelerated by gravity, the force fields and air resistance.

    The position of each particle at the start of the substep is kept as `ipos`, from which the velocity is recomputed
    once the constraints are solved.
    """
    substep_dt = pbd_info.substep_dt[None]
    for i_p, i_b in qd.ndrange(pbd_state.particles.pos.shape[0], pbd_state.particles.pos.shape[1]):
        pbd_state.particles.ipos[i_p, i_b] = pbd_state.particles.pos[i_p, i_b]
        if pbd_state.particles.free[i_p, i_b]:
            pbd_state.particles.vel[i_p, i_b] = pbd_state.particles.vel[i_p, i_b] + pbd_info.gravity[i_b] * substep_dt

            acc = force_fields.add_acc(
                qd.Vector.zero(gs.qd_float, 3),
                pbd_state.particles.pos[i_p, i_b],
                pbd_state.particles.vel[i_p, i_b],
                t,
                i_p,
            )
            pbd_state.particles.vel[i_p, i_b] = pbd_state.particles.vel[i_p, i_b] + acc * substep_dt

            if pbd_info.particles.material_type[i_p] == PBDMaterial.CLOTH:
                f_air_resistance = (
                    pbd_info.particles.air_resistance[i_p]
                    * pbd_state.particles.vel[i_p, i_b].norm()
                    * pbd_state.particles.vel[i_p, i_b]
                )
                pbd_state.particles.vel[i_p, i_b] = (
                    pbd_state.particles.vel[i_p, i_b] - f_air_resistance / pbd_info.particles.mass[i_p] * substep_dt
                )

        # An attached particle, held by the link rather than free, still moves with the link
        pbd_state.particles.pos[i_p, i_b] = (
            pbd_state.particles.pos[i_p, i_b] + pbd_state.particles.vel[i_p, i_b] * substep_dt
        )


@qd.func(requires_top_level=True)
def func_apply_dpos(pbd_state: PBDState, pbd_info: PBDInfo):
    """Move the free particles of physics by the correction the constraint projections accumulated, then reset it."""
    for i_p, i_b in qd.ndrange(pbd_state.particles.pos.shape[0], pbd_state.particles.pos.shape[1]):
        if pbd_state.particles.free[i_p, i_b] and pbd_info.particles.material_type[i_p] != PBDMaterial.PARTICLE:
            pbd_state.particles.pos[i_p, i_b] = pbd_state.particles.pos[i_p, i_b] + pbd_state.particles.dpos[i_p, i_b]
            pbd_state.particles.dpos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_solve_stretch(pbd_state: PBDState, pbd_info: PBDInfo):
    """Run one projection of the stretch constraints of the edges (extended position-based dynamics)."""
    for i_e, i_b in qd.ndrange(pbd_info.edges.v1.shape[0], pbd_state.particles.pos.shape[1]):
        i_p1 = pbd_info.edges.v1[i_e]
        i_p2 = pbd_info.edges.v2[i_e]

        w1 = pbd_state.particles.free[i_p1, i_b] / pbd_info.particles.mass[i_p1]
        w2 = pbd_state.particles.free[i_p2, i_b] / pbd_info.particles.mass[i_p2]
        n = pbd_state.particles.pos[i_p1, i_b] - pbd_state.particles.pos[i_p2, i_b]
        C = n.norm() - pbd_info.edges.len_rest[i_e]
        alpha = pbd_info.edges.stretch_compliance[i_e] / (pbd_info.substep_dt[None] ** 2)
        dp = -C / (w1 + w2 + alpha) * n / n.norm(pbd_info.EPS[None]) * pbd_info.edges.stretch_relaxation[i_e]
        pbd_state.particles.dpos[i_p1, i_b] += dp * w1
        pbd_state.particles.dpos[i_p2, i_b] -= dp * w2

    func_apply_dpos(pbd_state, pbd_info)


@qd.kernel(fastcache=True)
def kernel_solve_bending(pbd_state: PBDState, pbd_info: PBDInfo):
    """Run one projection of the bending constraints of the inner edges (extended position-based dynamics)."""
    for i_ie, i_b in qd.ndrange(pbd_info.inner_edges.v1.shape[0], pbd_state.particles.pos.shape[1]):
        i_p1 = pbd_info.inner_edges.v1[i_ie]
        i_p2 = pbd_info.inner_edges.v2[i_ie]
        i_p3 = pbd_info.inner_edges.v3[i_ie]
        i_p4 = pbd_info.inner_edges.v4[i_ie]

        w1 = pbd_state.particles.free[i_p1, i_b] / pbd_info.particles.mass[i_p1]
        w2 = pbd_state.particles.free[i_p2, i_b] / pbd_info.particles.mass[i_p2]
        w3 = pbd_state.particles.free[i_p3, i_b] / pbd_info.particles.mass[i_p3]
        w4 = pbd_state.particles.free[i_p4, i_b] / pbd_info.particles.mass[i_p4]

        if w1 + w2 + w3 + w4 > 0.0:
            # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
            # Appendix A: Bending Constraint Projection
            p2 = pbd_state.particles.pos[i_p2, i_b] - pbd_state.particles.pos[i_p1, i_b]
            p3 = pbd_state.particles.pos[i_p3, i_b] - pbd_state.particles.pos[i_p1, i_b]
            p4 = pbd_state.particles.pos[i_p4, i_b] - pbd_state.particles.pos[i_p1, i_b]
            l23 = p2.cross(p3).norm()
            l24 = p2.cross(p4).norm()
            n1 = p2.cross(p3) / l23
            n2 = p2.cross(p4) / l24
            d = qd.math.clamp(n1.dot(n2), -1.0, 1.0)

            # eq. (25), (26) and (27)
            q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23
            q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24
            q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 - (p4.cross(n1) + n2.cross(p4) * d) / l24
            q1 = -q2 - q3 - q4
            # eq. (29)
            sum_wq = w1 * q1.norm_sqr() + w2 * q2.norm_sqr() + w3 * q3.norm_sqr() + w4 * q4.norm_sqr()
            constraint = qd.acos(d) - qd.acos(-1.0)

            # Extended position-based dynamics (XPBD)
            alpha = pbd_info.inner_edges.bending_compliance[i_ie] / (pbd_info.substep_dt[None] ** 2)
            constraint = (
                -qd.sqrt(1 - d**2) * constraint / (sum_wq + alpha) * pbd_info.inner_edges.bending_relaxation[i_ie]
            )

            pbd_state.particles.dpos[i_p1, i_b] += w1 * constraint * q1
            pbd_state.particles.dpos[i_p2, i_b] += w2 * constraint * q2
            pbd_state.particles.dpos[i_p3, i_b] += w3 * constraint * q3
            pbd_state.particles.dpos[i_p4, i_b] += w4 * constraint * q4

    func_apply_dpos(pbd_state, pbd_info)


@qd.kernel(fastcache=True)
def kernel_solve_volume(pbd_state: PBDState, pbd_info: PBDInfo):
    """Run one projection of the volume constraints of the tetrahedra (extended position-based dynamics)."""
    for i_el, i_b in qd.ndrange(pbd_info.elems.v1.shape[0], pbd_state.particles.pos.shape[1]):
        i_p1 = pbd_info.elems.v1[i_el]
        i_p2 = pbd_info.elems.v2[i_el]
        i_p3 = pbd_info.elems.v3[i_el]
        i_p4 = pbd_info.elems.v4[i_el]

        p1 = pbd_state.particles.pos[i_p1, i_b]
        p2 = pbd_state.particles.pos[i_p2, i_b]
        p3 = pbd_state.particles.pos[i_p3, i_b]
        p4 = pbd_state.particles.pos[i_p4, i_b]

        grad1 = (p4 - p2).cross(p3 - p2) / 6.0
        grad2 = (p3 - p1).cross(p4 - p1) / 6.0
        grad3 = (p4 - p1).cross(p2 - p1) / 6.0
        grad4 = (p2 - p1).cross(p3 - p1) / 6.0

        w1 = pbd_state.particles.free[i_p1, i_b] / pbd_info.particles.mass[i_p1] * grad1.norm_sqr()
        w2 = pbd_state.particles.free[i_p2, i_b] / pbd_info.particles.mass[i_p2] * grad2.norm_sqr()
        w3 = pbd_state.particles.free[i_p3, i_b] / pbd_info.particles.mass[i_p3] * grad3.norm_sqr()
        w4 = pbd_state.particles.free[i_p4, i_b] / pbd_info.particles.mass[i_p4] * grad4.norm_sqr()

        if w1 + w2 + w3 + w4 > 0.0:
            vol = gu.qd_tet_vol(p1, p2, p3, p4)
            C = vol - pbd_info.elems.vol_rest[i_el]
            alpha = pbd_info.elems.volume_compliance[i_el] / (pbd_info.substep_dt[None] ** 2)
            s = -C / (w1 + w2 + w3 + w4 + alpha) * pbd_info.elems.volume_relaxation[i_el]

            pbd_state.particles.dpos[i_p1, i_b] += s * w1 * grad1
            pbd_state.particles.dpos[i_p2, i_b] += s * w2 * grad2
            pbd_state.particles.dpos[i_p3, i_b] += s * w3 * grad3
            pbd_state.particles.dpos[i_p4, i_b] += s * w4 * grad4

    func_apply_dpos(pbd_state, pbd_info)


@qd.func
def func_task_solve_collision(i_p: int, j_p: int, i_b: int, ret: qd.template(), pbd_state: PBDState, pbd_info: PBDInfo):
    """
    Push a sorted particle out of a neighbour, with friction (unified particle physics).

    The pair is skipped when both particles are fixed or both are of a liquid. `ret` accumulates the correction of the
    position of the particle `i_p`.
    """
    is_pair_valid = (
        pbd_state.particles_reordered.free[i_p, i_b] or pbd_state.particles_reordered.free[j_p, i_b]
    ) and not (
        pbd_state.particles_info_reordered.material_type[i_p, i_b] == PBDMaterial.LIQUID
        and pbd_state.particles_info_reordered.material_type[j_p, i_b] == PBDMaterial.LIQUID
    )
    if is_pair_valid:
        cur_dist = (pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.pos[j_p, i_b]).norm(
            pbd_info.EPS[None]
        )
        rest_dist = (
            pbd_state.particles_info_reordered.pos_rest[i_p, i_b]
            - pbd_state.particles_info_reordered.pos_rest[j_p, i_b]
        ).norm(pbd_info.EPS[None])
        # The target distance between two particles is twice their radius
        target_dist = pbd_info.particle_size[None]
        if cur_dist < target_dist and rest_dist > target_dist:
            wi = pbd_state.particles_reordered.free[i_p, i_b] / pbd_state.particles_info_reordered.mass[i_p, i_b]
            wj = pbd_state.particles_reordered.free[j_p, i_b] / pbd_state.particles_info_reordered.mass[j_p, i_b]
            n = (pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.pos[j_p, i_b]) / cur_dist

            ### resolve collision ###
            ret += wi / (wi + wj) * (target_dist - cur_dist) * n

            ### apply friction ###
            # https://mmacklin.com/uppfrta_preprint.pdf
            # equation (23)
            dv = (pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.ipos[i_p, i_b]) - (
                pbd_state.particles_reordered.pos[j_p, i_b] - pbd_state.particles_reordered.ipos[j_p, i_b]
            )
            dpos = -(dv - n * n.dot(dv))
            # equation (24)
            d = target_dist - cur_dist
            mu_s = qd.max(
                pbd_state.particles_info_reordered.mu_s[i_p, i_b], pbd_state.particles_info_reordered.mu_s[j_p, i_b]
            )
            mu_k = qd.max(
                pbd_state.particles_info_reordered.mu_k[i_p, i_b], pbd_state.particles_info_reordered.mu_k[j_p, i_b]
            )
            if dpos.norm() < mu_s * d:
                ret += wi / (wi + wj) * dpos
            else:
                ret += wi / (wi + wj) * dpos * qd.min(1.0, mu_k * d / dpos.norm(pbd_info.EPS[None]))


@qd.kernel(fastcache=True)
def kernel_solve_collision(pbd_state: PBDState, pbd_info: PBDInfo):
    """Resolve the collisions between the sorted particles that are not both of a liquid."""
    for i_p, i_b in qd.ndrange(pbd_state.particles_reordered.pos.shape[0], pbd_state.particles_reordered.pos.shape[1]):
        if pbd_state.particles_info_reordered.material_type[i_p, i_b] != PBDMaterial.PARTICLE:
            dpos = pbd_state.particles_reordered.dpos[i_p, i_b]
            func_for_all_neighbors(
                i_p, i_b, dpos, pbd_state, pbd_info, pbd_info.particle_size[None], func_task_solve_collision
            )
            pbd_state.particles_reordered.dpos[i_p, i_b] = dpos

    func_apply_sorted_dpos(pbd_state, liquid_only=False)


@qd.kernel(fastcache=True)
def kernel_solve_boundary_collision(pbd_state: PBDState, pbd_info: PBDInfo):
    """Keep every particle inside the boundary, whether free or not."""
    for i_p, i_b in qd.ndrange(pbd_state.particles.pos.shape[0], pbd_state.particles.pos.shape[1]):
        pos_new, vel_new = func_impose_cube_boundary(
            pbd_state.particles.pos[i_p, i_b], pbd_state.particles.vel[i_p, i_b], pbd_info.boundary
        )
        pbd_state.particles.pos[i_p, i_b] = pos_new
        pbd_state.particles.vel[i_p, i_b] = vel_new


@qd.func
def func_task_compute_lambda(i_p: int, j_p: int, i_b: int, ret: qd.template(), pbd_state: PBDState, pbd_info: PBDInfo):
    d_ij = pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.pos[j_p, i_b]
    dist_scale = pbd_info.dist_scale[None]
    ret.rho += func_poly6(d_ij, dist_scale) * pbd_state.particles_info_reordered.mass[j_p, i_b]
    s = func_spiky(d_ij, dist_scale) / pbd_state.particles_info_reordered.rho_rest[i_p, i_b]
    ret.spiky_sum += s
    ret.spiky_norm_sqr_sum += s.dot(s)


@qd.func
def func_task_compute_density_dpos(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), pbd_state: PBDState, pbd_info: PBDInfo
):
    d_ij = pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.pos[j_p, i_b]
    dist_scale = pbd_info.dist_scale[None]
    scorr = func_s_corr(d_ij, dist_scale)
    left = pbd_state.particles_reordered.lam[i_p, i_b] + pbd_state.particles_reordered.lam[j_p, i_b] + scorr
    right = func_spiky(d_ij, dist_scale)
    ret += (
        left
        * right
        / pbd_state.particles_info_reordered.rho_rest[i_p, i_b]
        * dist_scale
        * pbd_state.particles_info_reordered.density_relaxation[i_p, i_b]
    )


@qd.kernel(fastcache=True)
def kernel_solve_density(pbd_state: PBDState, pbd_info: PBDInfo):
    """Run one projection of the density constraints of the sorted liquid particles (position based fluids)."""
    for i_p, i_b in qd.ndrange(pbd_state.particles_reordered.pos.shape[0], pbd_state.particles_reordered.pos.shape[1]):
        if pbd_state.particles_info_reordered.material_type[i_p, i_b] == PBDMaterial.LIQUID:
            ret = qd.Struct(
                rho=gs.qd_float(0.0), spiky_sum=qd.Vector.zero(gs.qd_float, 3), spiky_norm_sqr_sum=gs.qd_float(0.0)
            )
            func_for_all_neighbors(
                i_p, i_b, ret, pbd_state, pbd_info, pbd_info.dist_scale[None], func_task_compute_lambda
            )
            constraint = (ret.rho / pbd_state.particles_info_reordered.rho_rest[i_p, i_b]) - 1.0
            lower_sum = ret.spiky_norm_sqr_sum + ret.spiky_sum.dot(ret.spiky_sum)
            pbd_state.particles_reordered.lam[i_p, i_b] = -1.0 * (constraint / (lower_sum + 100.0))

    for i_p, i_b in qd.ndrange(pbd_state.particles_reordered.pos.shape[0], pbd_state.particles_reordered.pos.shape[1]):
        if pbd_state.particles_info_reordered.material_type[i_p, i_b] == PBDMaterial.LIQUID:
            dpos = pbd_state.particles_reordered.dpos[i_p, i_b]
            func_for_all_neighbors(
                i_p, i_b, dpos, pbd_state, pbd_info, pbd_info.dist_scale[None], func_task_compute_density_dpos
            )
            pbd_state.particles_reordered.dpos[i_p, i_b] = dpos

    func_apply_sorted_dpos(pbd_state, liquid_only=True)


@qd.func(requires_top_level=True)
def func_apply_sorted_dpos(pbd_state: PBDState, liquid_only: qd.template()):
    """
    Move the free sorted particles of physics by the correction their projections accumulated, then reset it.

    `liquid_only` restricts the move to the particles of a liquid (see func_apply_dpos).
    """
    for i_p, i_b in qd.ndrange(pbd_state.particles_reordered.pos.shape[0], pbd_state.particles_reordered.pos.shape[1]):
        material_type = pbd_state.particles_info_reordered.material_type[i_p, i_b]
        is_moved = material_type != PBDMaterial.PARTICLE
        if qd.static(liquid_only):
            is_moved = material_type == PBDMaterial.LIQUID
        if is_moved and pbd_state.particles_reordered.free[i_p, i_b]:
            pbd_state.particles_reordered.pos[i_p, i_b] = (
                pbd_state.particles_reordered.pos[i_p, i_b] + pbd_state.particles_reordered.dpos[i_p, i_b]
            )
            pbd_state.particles_reordered.dpos[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.func
def func_task_compute_xsph(i_p: int, j_p: int, i_b: int, ret: qd.template(), pbd_state: PBDState, pbd_info: PBDInfo):
    v_ij = (pbd_state.particles_reordered.pos[j_p, i_b] - pbd_state.particles_reordered.ipos[j_p, i_b]) - (
        pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.ipos[i_p, i_b]
    )
    d_ij = pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.pos[j_p, i_b]
    ret += func_poly6(d_ij, pbd_info.dist_scale[None]) * v_ij


@qd.kernel(fastcache=True)
def kernel_solve_viscosity(pbd_state: PBDState, pbd_info: PBDInfo):
    """
    Smooth the velocity of the sorted liquid particles by their neighbours once.

    This is the viscosity of extended smoothed particle hydrodynamics (XSPH).
    """
    for i_p, i_b in qd.ndrange(pbd_state.particles_reordered.pos.shape[0], pbd_state.particles_reordered.pos.shape[1]):
        if pbd_state.particles_info_reordered.material_type[i_p, i_b] == PBDMaterial.LIQUID:
            xsph_sum = qd.Vector.zero(gs.qd_float, 3)
            func_for_all_neighbors(
                i_p, i_b, xsph_sum, pbd_state, pbd_info, pbd_info.dist_scale[None], func_task_compute_xsph
            )
            pbd_state.particles_reordered.dpos[i_p, i_b] = (
                pbd_state.particles_reordered.dpos[i_p, i_b]
                + xsph_sum * pbd_state.particles_info_reordered.viscosity_relaxation[i_p, i_b]
            )

    func_apply_sorted_dpos(pbd_state, liquid_only=True)


@qd.kernel(fastcache=True)
def kernel_compute_velocity(pbd_state: PBDState, pbd_info: PBDInfo):
    """Set the velocity of the sorted particles to their displacement over the substep."""
    for i_p, i_b in qd.ndrange(pbd_state.particles_reordered.pos.shape[0], pbd_state.particles_reordered.pos.shape[1]):
        pbd_state.particles_reordered.vel[i_p, i_b] = (
            pbd_state.particles_reordered.pos[i_p, i_b] - pbd_state.particles_reordered.ipos[i_p, i_b]
        ) / pbd_info.substep_dt[None]


@qd.kernel(fastcache=True)
def kernel_copy_from_reordered(pbd_state: PBDState):
    """Give the active particles back the state of their sorted copy."""
    for i_p, i_b in qd.ndrange(pbd_state.particles.pos.shape[0], pbd_state.particles.pos.shape[1]):
        if pbd_state.particles.is_active[i_p, i_b]:
            i_r = pbd_state.particles.reordered_idx[i_p, i_b]
            func_copy_particle(i_r, i_p, i_b, pbd_state.particles_reordered, pbd_state.particles)


@qd.kernel(fastcache=True)
def kernel_set_particles_free(
    particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), pbd_state: PBDState, free: bool
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        pbd_state.particles.free[i_p, i_b] = free
