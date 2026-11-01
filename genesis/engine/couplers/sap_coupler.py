import math
from typing import TYPE_CHECKING, Annotated

import numpy as np
import torch

import igl

import quadrants as qd

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.linalg as lu
from genesis.engine.bvh import (
    BVHLeaves,
    BVHQueryFilter,
    BVHQueryResults,
    BVHState,
    BVHStaticConfig,
    build_bvh,
    func_bvh_query_leaves,
    func_no_filter,
    get_bvh_data,
)
from genesis.engine.pipeline import Coupler, Enabled, Phase, RoleOf, System, register_system
from genesis.engine.solvers.fem_solver import (
    ELEMENTS,
    VERTICES,
    FEMInfo,
    FEMState,
    FEMVertexConstraintsInfo,
    func_compute_element_mapping_matrix,
)
from genesis.engine.solvers.rigid.abd.forward_dynamics import (
    func_solve_mass_batch,
    update_qacc_from_qvel_delta,
    update_qvel,
)
from genesis.engine.solvers.rigid.abd.forward_kinematics import kernel_update_all_verts
from genesis.engine.solvers.rigid.collider import Collider
from genesis.engine.solvers.rigid.constraint import ConstraintSolver
from genesis.engine.solvers.rigid.data import (
    DOFS,
    DofsState,
    DynState,
    EntitiesInfo,
    EqualitiesInfo,
    FacesInfo,
    GeomsInfo,
    GeomsState,
    JointsInfo,
    LinksInfo,
    LinksState,
    RigidInfo,
    VertsInfo,
    VertsState,
)
from genesis.engine.solvers.rigid.rigid_solver import RigidCollisionDetection, RigidConstraints, RigidSolver
from genesis.engine.step import Coupling, Integration, Preparation
from genesis.options.solvers import SAPCouplerOptions
from genesis.repr_base import RBC
from genesis.utils.data import ENVS, Data, DataKind, Size, StaticConfig, field
from genesis.utils.linalg import SUM_MAX_CHUNKS, FixedOrderSums
from genesis.utils.misc import qd_to_numpy, qd_to_torch, tensor_to_array
from genesis.utils.sort import SegmentSortBuffer, SegmentSortStaticConfig, func_sort_segments, get_segment_sort_data

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


# Most FEM vertices a contact pair of the SAP coupler applies its impulse to, the vertices of two tetrahedra. The
# kernels read it as the size of the vector of slots of a pair (`fem_slots_vert` in SAPContactPairs), since a module
# global read in a kernel breaks its fast cache.
SAP_N_FEM_SLOTS = 8


SAP_VOLUME_VERTS = Size("volume_verts")
SAP_VOLUME_ELEMS = Size("volume_elems")
SAP_EQUALITIES = Size("equality_constraints")
SAP_PAIRS = Size("contact_pairs")
SAP_RIGID_PAIRS = Size("rigid_contact_pairs")
SAP_HANDLERS = Size("handlers")
SAP_RIGID_HANDLERS = Size("rigid_handlers")
SAP_ENTRIES = Size("entries")
SAP_PAIR_CHUNKS = Size("pair_chunks")
SAP_CANDIDATES = Size("handler_candidates")
SAP_MARCHING_TETS_CASES = Size("marching_tets_cases")
SAP_TET_EDGES = Size("tet_edges")


class SAPContactQueries(Data, kind=DataKind.SCRATCH):
    """
    The box queries of the contact handlers of the SAP coupler traversing a tree of SAPState.

    The leaves of the surface tets of the FEM entities serve both the self-contact query and the query of the rigid
    triangles. The filter of the self-contact query maps each leaf to the vertices of its tet, and the filter of the
    rigid tets maps each leaf to its geom in its first item, its table holding the collision pair of each pair of geoms,
    -1 for none (see collision_pair_idx in ColliderInfo). The query of the rigid triangles keeps every pair, its filter
    reading nothing.
    """

    fem_self_results: BVHQueryResults
    rigid_tri_results: BVHQueryResults
    rigid_tet_results: BVHQueryResults
    fem_self_filter: BVHQueryFilter
    rigid_tri_filter: BVHQueryFilter
    rigid_tet_filter: BVHQueryFilter


class SAPHydroelasticFEM(Data, kind=DataKind.CONSTANT):
    """The pressure field of the FEM entities for hydroelastic contact, its value at each vertex and its gradient in
    each element."""

    pressure: qd.Tensor = field(float, shape=(VERTICES,))
    pressure_gradient: qd.Tensor = field((float, 3), shape=(ELEMENTS, ENVS), kind=DataKind.SCRATCH)


class SAPRigidVolume(Data, kind=DataKind.CONSTANT):
    """
    The tetrahedral volume meshes of the rigid geoms for hydroelastic contact, and their pressure field.

    The vertices move with their geom in every environment from their rest position `verts_rest`, and so does the
    gradient of the pressure field in each element from `pressure_gradient_rest`.
    """

    verts_rest: qd.Tensor = field((float, 3), shape=(SAP_VOLUME_VERTS,))
    verts: qd.Tensor = field((float, 3), shape=(SAP_VOLUME_VERTS, ENVS), kind=DataKind.SCRATCH)
    elems: qd.Tensor = field((int, 4), shape=(SAP_VOLUME_ELEMS,))
    verts_geom_idx: qd.Tensor = field(int, shape=(SAP_VOLUME_VERTS,))
    elems_geom_idx: qd.Tensor = field(int, shape=(SAP_VOLUME_ELEMS,))
    pressure: qd.Tensor = field(float, shape=(SAP_VOLUME_VERTS,))
    pressure_gradient_rest: qd.Tensor = field((float, 3), shape=(SAP_VOLUME_ELEMS,))
    pressure_gradient: qd.Tensor = field((float, 3), shape=(SAP_VOLUME_ELEMS, ENVS), kind=DataKind.SCRATCH)


class SAPFEMVertices(Data, kind=DataKind.SCRATCH, shape=(VERTICES, ENVS)):
    """
    The unknowns of the SAP solve at the FEM vertices, and the scratch of its conjugate gradient and line search.

    `v` is the velocity being solved for, `v_diff` its change from the free velocity, `gradient` the gradient of the
    cost and `impulse` the contact impulse. `corners` holds the elastic contribution of each element to each of its
    four vertices, which the vertices sum in a fixed order (see FEMVerticesProperties).
    """

    v: qd.Tensor = field((float, 3))
    v_diff: qd.Tensor = field((float, 3))
    gradient: qd.Tensor = field((float, 3))
    impulse: qd.Tensor = field((float, 3))
    diag3x3: qd.Tensor = field((float, (3, 3)))
    prec: qd.Tensor = field((float, (3, 3)))
    x: qd.Tensor = field((float, 3))
    r: qd.Tensor = field((float, 3))
    z: qd.Tensor = field((float, 3))
    p: qd.Tensor = field((float, 3))
    Ap: qd.Tensor = field((float, 3))
    x_prev: qd.Tensor = field((float, 3))
    dp: qd.Tensor = field((float, 3))
    corners: qd.Tensor = field((float, 3), shape=(ELEMENTS, 4, ENVS))
    sums_v: FixedOrderSums
    sums_el: FixedOrderSums


class SAPRigidDofs(Data, kind=DataKind.SCRATCH, shape=(DOFS, ENVS)):
    """The unknowns of the SAP solve at the rigid dofs, and the scratch of its conjugate gradient and line search (see
    SAPFEMVertices)."""

    v: qd.Tensor = field(float)
    v_diff: qd.Tensor = field(float)
    mass_v_diff: qd.Tensor = field(float)
    gradient: qd.Tensor = field(float)
    impulse: qd.Tensor = field(float)
    x: qd.Tensor = field(float)
    r: qd.Tensor = field(float)
    z: qd.Tensor = field(float)
    p: qd.Tensor = field(float)
    Ap: qd.Tensor = field(float)
    x_prev: qd.Tensor = field(float)
    dp: qd.Tensor = field(float)
    sums: FixedOrderSums


class SAPSolverScalars(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    """
    The scalars of the SAP solve of each environment: its Newton iterations, its conjugate gradient and its exact line
    search.

    The line search finds the root of the derivative of the cost along the search direction by safeguarded Newton
    steps within the bracket [`alpha_min`, `alpha_max`], `f_upper` being that derivative scaled by `dell_scale` at
    `alpha_max`, unless the slope at the start is small enough for the full step (`is_full_step`).
    """

    is_active: qd.Tensor = field(bool)
    is_pcg_active: qd.Tensor = field(bool)
    is_linesearch_active: qd.Tensor = field(bool)
    is_full_step: qd.Tensor = field(bool)
    rTz: qd.Tensor = field(float)
    alpha: qd.Tensor = field(float)
    beta: qd.Tensor = field(float)
    energy_prev: qd.Tensor = field(float)
    step_size: qd.Tensor = field(float)
    m: qd.Tensor = field(float)
    dell_dalpha: qd.Tensor = field(float)
    d2ellA_dalpha2: qd.Tensor = field(float)
    d2ell_dalpha2: qd.Tensor = field(float)
    dell_scale: qd.Tensor = field(float)
    alpha_min: qd.Tensor = field(float)
    alpha_max: qd.Tensor = field(float)
    alpha_tol: qd.Tensor = field(float)
    f_upper: qd.Tensor = field(float)
    minus_dalpha: qd.Tensor = field(float)
    minus_dalpha_prev: qd.Tensor = field(float)


class SAPEqualityConstraints(Data, kind=DataKind.CONSTANT, shape=(SAP_EQUALITIES,)):
    """
    The joint equality constraints of the rigid solver, which the SAP coupler regularizes and solves with the contacts.

    The constraints are ordered by environment, those of environment i_b being from `batch_start[i_b]` to
    `batch_start[i_b + 1]`. Each one ties the dof `dof1_idx` to the dof `dof2_idx`, its Jacobian row being +1 at the
    first and -1 at the second. The SAP regularization of a constraint is `R` and `v_hat`, its Hessian being `R_inv`.
    `vec` holds the impulse each constraint applies, and `sums` the partial sums over the chunks of the constraints of
    each environment, whose terms are unused.
    """

    batch_idx: qd.Tensor = field(int)
    dof1_idx: qd.Tensor = field(int)
    dof2_idx: qd.Tensor = field(int)
    batch_start: qd.Tensor = field(int, shape=(ENVS + 1,))
    R: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    R_inv: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    v_hat: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    dvc: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    vec: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    sums: FixedOrderSums = field(kind=DataKind.SCRATCH)


class SAPContactPairs(Data, kind=DataKind.SCRATCH, shape=(SAP_PAIRS,)):
    """
    The contact pairs every contact handler of the SAP coupler detects, which one generic solve handles.

    Handler i_h owns the pairs from `handlers_offset[i_h]` to `handlers_offset[i_h] + handlers_capacity[i_h]`, the
    first `handlers_n_pairs[i_h]` of which are detected, `handlers_is_overflow[i_h]` telling whether its candidates or
    its box query overflowed. The handlers applying impulses to rigid dofs come first, so that their pairs alone take a
    Jacobian `Jt` (see SAPContactsLayout). A pair applies its impulse to up to SAP_N_FEM_SLOTS FEM vertices
    `fem_slots_vert` (-1 for an unused slot), with the weights `fem_slots_weight`, and to the chains of the links
    `link_idx` (-1 for none), with the signs `link_sign`, at `contact_pos`. A pair solves in its contact frame `frame`,
    whose columns are its two tangents and its normal, the world frame for the floor contacts.

    The pairs are appended concurrently during the detection, in an order that changes from run to run. The key of a
    pair identifies it uniquely among the pairs of its handler and environment, and `sort` orders the pairs by
    environment, then by handler, then by key, as one segment per environment and handler. Every sum over the pairs,
    whether into a scalar of an environment or into a vertex or a dof, runs in that order, chunk by chunk for each
    handler (see SUM_MAX_CHUNKS), so that the result is bitwise reproducible. The slots applying to vertex i_v of
    environment i_b, encoded as i_s * SAP_N_FEM_SLOTS + i_slot for the slot i_slot of the pair at the sorted position
    i_s, are `vertices_entry[vertices_entry_start[i_v, i_b]:vertices_entry_start[i_v + 1, i_b]]` in increasing order.

    The SAP regularization of a pair is `k`, `phi0`, `Rn`, `Rt`, `vn_hat`, `mu` and their derived values, its impulse
    `gamma` and its Hessian `G` in its frame, and `dvc` the change of its velocity along the search direction. `vec` and
    `mat` hold its impulse and its Hessian in the world frame, and `pairs_partials` and `dofs_partials` the partial sums
    over the chunks of the sorted pairs of each handler and environment.
    """

    handler_idx: qd.Tensor = field(int)
    batch_idx: qd.Tensor = field(int)
    key: qd.Tensor = field(qd.u64)
    frame: qd.Tensor = field((float, (3, 3)))
    contact_pos: qd.Tensor = field((float, 3))
    fem_slots_vert: qd.Tensor = field((int, SAP_N_FEM_SLOTS))
    fem_slots_weight: qd.Tensor = field((float, SAP_N_FEM_SLOTS))
    link_idx: qd.Tensor = field((int, 2))
    link_sign: qd.Tensor = field((float, 2))
    k: qd.Tensor = field(float)
    phi0: qd.Tensor = field(float)
    Rn: qd.Tensor = field(float)
    Rt: qd.Tensor = field(float)
    Rn_inv: qd.Tensor = field(float)
    Rt_inv: qd.Tensor = field(float)
    vn_hat: qd.Tensor = field(float)
    mu: qd.Tensor = field(float)
    mu_hat: qd.Tensor = field(float)
    mu_factor: qd.Tensor = field(float)
    gamma: qd.Tensor = field((float, 3))
    G: qd.Tensor = field((float, (3, 3)))
    dvc: qd.Tensor = field((float, 3))
    vec: qd.Tensor = field((float, 3))
    mat: qd.Tensor = field((float, (3, 3)))
    Jt: qd.Tensor = field((float, 3), shape=(SAP_RIGID_PAIRS, DOFS))
    M_inv_Jt: qd.Tensor = field((float, 3), shape=(SAP_RIGID_PAIRS, DOFS))
    handlers_n_pairs: qd.Tensor = field(int, shape=(SAP_HANDLERS,))
    handlers_is_overflow: qd.Tensor = field(bool, shape=(SAP_HANDLERS,))
    segments_n_pairs: qd.Tensor = field(int, shape=(SAP_HANDLERS * ENVS,))
    sort: SegmentSortBuffer
    vertices_n_entries: qd.Tensor = field(int, shape=(VERTICES, ENVS))
    vertices_entry_start: qd.Tensor = field(int, shape=(VERTICES + 1, ENVS))
    vertices_entry: qd.Tensor = field(int, shape=(SAP_ENTRIES,))
    pairs_partials: qd.Tensor = field(float, shape=(SAP_PAIR_CHUNKS, 2, SAP_HANDLERS, ENVS))
    dofs_partials: qd.Tensor = field(float, shape=(SAP_PAIR_CHUNKS, DOFS, SAP_RIGID_HANDLERS, ENVS))


class SAPContactsLayout(Data, kind=DataKind.CONSTANT, shape=(SAP_HANDLERS,)):
    """
    The layout of the contact pairs of every handler of the SAP coupler (see SAPContactPairs).

    `handlers_n_chunks` is the number of chunks the sorted pairs of a handler split into for the sums over them,
    `handlers_rigid_rank` the rank of a handler among those applying impulses to rigid dofs (-1 for none), and
    `key_bits` the bits of the keys of the pairs of a handler.
    """

    handlers_offset: qd.Tensor = field(int)
    handlers_capacity: qd.Tensor = field(int)
    handlers_n_chunks: qd.Tensor = field(int)
    handlers_rigid_rank: qd.Tensor = field(int)
    key_bits: qd.Tensor = field(int, shape=())


class SAPTetPlaneCandidates(Data, kind=DataKind.SCRATCH, shape=(SAP_CANDIDATES,)):
    """The tetrahedra crossing a plane, with the signed distance of their vertices to it and the code of the edges it
    cuts (see the marching tetrahedra table of SAPInfo)."""

    batch_idx: qd.Tensor = field(int)
    elem_idx: qd.Tensor = field(int)
    intersection_code: qd.Tensor = field(int)
    distance: qd.Tensor = field((float, 4))
    n_candidates: qd.Tensor = field(int, shape=(1,))


class SAPTetTetCandidates(Data, kind=DataKind.SCRATCH, shape=(SAP_CANDIDATES,)):
    """
    The pairs of tetrahedra crossing their plane of equal pressure, with the normal of the plane, the signed distance of
    the vertices of the first one to it and the code of the edges it cuts.
    """

    batch_idx: qd.Tensor = field(int)
    elem_idx0: qd.Tensor = field(int)
    elem_idx1: qd.Tensor = field(int)
    intersection_code0: qd.Tensor = field(int)
    normal: qd.Tensor = field((float, 3))
    distance0: qd.Tensor = field((float, 4))
    n_candidates: qd.Tensor = field(int, shape=(1,))


class SAPTriTetCandidates(Data, kind=DataKind.SCRATCH, shape=(SAP_CANDIDATES,)):
    """The pairs of a FEM tetrahedron and a rigid triangle crossing the plane of the triangle, of normal `normal`."""

    batch_idx: qd.Tensor = field(int)
    elem_idx: qd.Tensor = field(int)
    face_idx: qd.Tensor = field(int)
    normal: qd.Tensor = field((float, 3))
    n_candidates: qd.Tensor = field(int, shape=(1,))


class SAPContactCandidates(Data, kind=DataKind.SCRATCH):
    """The candidates of the contact handlers detecting their pairs in two passes, one member per handler."""

    fem_floor_tet: SAPTetPlaneCandidates
    fem_self_tet: SAPTetTetCandidates
    rigid_floor_tet: SAPTetPlaneCandidates
    rigid_fem_tri_tet: SAPTriTetCandidates
    rigid_rigid_tet: SAPTetTetCandidates


class SAPState(Data, kind=DataKind.SCRATCH):
    """Runtime state of the SAP coupler, with the trees of the surface tets of the FEM entities, of the triangles and of
    the tets of the rigid geoms its box queries traverse."""

    fem_surface_tet_bvh: BVHState
    rigid_tri_bvh: BVHState
    rigid_tet_bvh: BVHState
    fem: SAPFEMVertices
    rigid: SAPRigidDofs
    solver: SAPSolverScalars
    hydroelastic_fem: SAPHydroelasticFEM
    rigid_volume: SAPRigidVolume
    equalities: SAPEqualityConstraints
    pairs: SAPContactPairs
    candidates: SAPContactCandidates
    queries: SAPContactQueries


class SAPInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """
    Model parameters of the SAP coupler.

    The SAP solve stops once the norm of the gradient of its cost falls below `convergence_atol` plus
    `convergence_rtol` times the norm of the momentum or of the impulse. `taud`, `beta` and `sigma` set the
    regularization of the contacts and of the constraints, and `hydroelastic_stiffness`, `point_contact_stiffness` and
    `rigid_floor_vert_stiffness` the stiffness of the contacts of each model. A hydroelastic contact smaller than
    `hydroelastic_eps` in area, stiffness or gap is dropped. The tables `marching_tets_edge_table` and `tet_edges` give
    the edges a plane cuts in a tetrahedron from the sign of the distance of its vertices to it.
    """

    contacts: SAPContactsLayout
    substep_dt: qd.Tensor = field(float)
    convergence_atol: qd.Tensor = field(float)
    convergence_rtol: qd.Tensor = field(float)
    taud: qd.Tensor = field(float)
    beta: qd.Tensor = field(float)
    sigma: qd.Tensor = field(float)
    pcg_threshold: qd.Tensor = field(float)
    linesearch_ftol: qd.Tensor = field(float)
    linesearch_max_step_size: qd.Tensor = field(float)
    hydroelastic_stiffness: qd.Tensor = field(float)
    point_contact_stiffness: qd.Tensor = field(float)
    equality_stiffness: qd.Tensor = field(float)
    rigid_floor_vert_stiffness: qd.Tensor = field(float)
    hydroelastic_eps: qd.Tensor = field(float)
    floor_height: qd.Tensor = field(float)
    marching_tets_edge_table: qd.Tensor = field((int, 4), shape=(SAP_MARCHING_TETS_CASES,))
    tet_edges: qd.Tensor = field((int, 2), shape=(SAP_TET_EDGES,))
    EPS: qd.Tensor = field(float)


class SAPStaticConfig(StaticConfig):
    """The compile-time configuration of the kernels of the SAP coupler."""

    # Whether the solve has FEM vertices, rigid dofs and equality constraints among its unknowns
    has_fem: bool
    has_rigid: bool
    has_equalities: bool
    # Whether some pairs apply their impulse to rigid dofs, through a rigid Jacobian
    has_rigid_contacts: bool
    # Whether the vertices of the FEM solver may be constrained, which drops them from the solve. The name matches the
    # one of FEMStaticConfig, the FEM funcs the SAP kernels call reading it from either config.
    enable_vertex_constraints: bool


MARCHING_TETS_EDGE_TABLE = (
    (-1, -1, -1, -1),
    (0, 3, 2, -1),
    (0, 1, 4, -1),
    (4, 3, 2, 1),
    (1, 2, 5, -1),
    (0, 3, 5, 1),
    (0, 2, 5, 4),
    (3, 5, 4, -1),
    (3, 4, 5, -1),
    (4, 5, 2, 0),
    (1, 5, 3, 0),
    (1, 5, 2, -1),
    (1, 2, 3, 4),
    (0, 4, 1, -1),
    (0, 2, 3, -1),
    (-1, -1, -1, -1),
)

TET_EDGES = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
)

# An estimate of the maximum number of contact pairs per AABB query
MAX_N_QUERY_RESULT_PER_AABB = 32

# An estimate of the maximum number of candidates per contact pair of the handlers detecting their pairs in two passes
MAX_N_CANDIDATES_PER_PAIR = 8

# The largest box query result list, its count being a 32-bit integer
MAX_N_QUERY_RESULTS = 0x7FFFFFFF

# Stiffness of the joint equality constraints
EQUALITY_STIFFNESS = 1e8

# Stiffness of the point contacts of the rigid vertices with the floor
RIGID_FLOOR_VERT_STIFFNESS = 1.0e6

# Smallest area, stiffness and gap a hydroelastic contact pair is kept with
HYDROELASTIC_EPS = 1e-10


def compute_pressure_field(
    verts: np.ndarray, surface_verts: np.ndarray, surface_faces: np.ndarray, modulus: float
) -> np.ndarray:
    """
    Return the pressure field of a volume mesh at its vertices for hydroelastic contact, its distance to the surface.

    The field grows linearly with the distance of a vertex to the surface mesh `surface_verts`, `surface_faces`, from
    zero on the surface to `modulus` at the deepest vertex. See
    https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
    """
    # The unsigned query returns bitwise identical distances for identical inputs, whereas the signed ones differ in
    # their last bits from one call to the next, which makes the whole simulation irreproducible
    distance, *_ = igl.signed_distance(verts, surface_verts, surface_faces, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)
    distance_max = np.max(distance)
    if distance_max < gs.EPS:
        gs.raise_exception(
            f"Pressure field max distance is too small: {distance_max}. "
            "This might be due to a mesh having no internal vertices."
        )
    return distance / distance_max * modulus


class SAPCoupler(RBC, Coupler, Preparation, Coupling, Integration, RigidConstraints):
    """
    Couple the FEM and rigid solvers of a scene through the Semi-Analytic Primal (SAP) contact solver of Drake.

    Paper reference: https://arxiv.org/abs/2110.10107
    Drake reference: https://drake.mit.edu/release_notes/v1.5.0.html
    """

    # Whether the coupler supports a scene with `requires_grad` (see 'Solver.is_differentiable')
    is_differentiable = False

    # SAP solves the constraints of the rigid entities itself, its own contacts included, on the velocities their
    # dynamics reach: their velocities are predicted in place of their constraint force, and their acceleration is
    # recovered from the velocities the couplings solve before they are integrated
    rigid_integration = Phase()
    relocations = {
        RoleOf(Collider, RigidCollisionDetection): None,
        RoleOf(ConstraintSolver, RigidConstraints): None,
        RigidSolver.integration: rigid_integration,
    }
    solve_rigid_constraints = update_qvel

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, simulator: "Simulator", options: "SAPCouplerOptions") -> None:
        self.sim = simulator
        self.options = options
        self.rigid_solver = self.sim.rigid_solver
        self.fem_solver = self.sim.fem_solver
        self._n_sap_iterations = options.n_sap_iterations
        self._n_pcg_iterations = options.n_pcg_iterations
        self._n_linesearch_iterations = options.n_linesearch_iterations
        if gs.qd_float == qd.f32:
            gs.raise_exception(
                "SAPCoupler does not support 32bits precision. Please specify precision='64' when initializing Genesis."
            )
        if options.rigid_rigid_contact_type == "vert":
            gs.raise_exception("Invalid rigid-rigid contact type: 'vert'. Must be one of 'tet' or 'none'.")

        self.contact_handlers: list[SAPContactHandler] = []
        # The static configs of the trees of SAPState
        self.fem_surface_tet_bvh_config: BVHStaticConfig | None = None
        self.rigid_tri_bvh_config: BVHStaticConfig | None = None
        self.rigid_tet_bvh_config: BVHStaticConfig | None = None
        self.sap_state: SAPState | None = None
        self.sap_info: SAPInfo | None = None
        self.sap_static_config: SAPStaticConfig | None = None
        self._pairs_sort_config: SegmentSortStaticConfig | None = None
        self._is_fem_hydroelastic = False
        self._is_rigid_hydroelastic = False
        self._has_fem_surface_tet_tree = False
        self._has_contact = False

    def build(self) -> None:
        if self.rigid_solver.is_active:
            self.rigid_integration.add(update_qacc_from_qvel_delta)
        B = self.sim._B
        fem_solver, rigid_solver = self.fem_solver, self.rigid_solver

        for equality in rigid_solver.equalities:
            if equality.type == gs.EQUALITY_TYPE.JOINT and equality.eq_obj2id < 0:
                gs.raise_exception("SAPCoupler does not support JOINT equality constraints without `joint2`.")
        if fem_solver.is_active and not fem_solver.use_implicit_solver:
            gs.raise_exception(
                "SAPCoupler requires FEM to use implicit solver. Please set `use_implicit_solver=True` in FEM options."
            )

        # The tetrahedral volume meshes of the colliding rigid geoms, the geoms of their vertices and elements, and their
        # pressure field
        options = self.options
        rigid_contact_types = (options.rigid_floor_contact_type, options.rigid_rigid_contact_type)
        self._is_rigid_hydroelastic = rigid_solver.is_active and "tet" in rigid_contact_types
        volume_verts, volume_elems, volume_verts_geom_idx, volume_elems_geom_idx, volume_pressure = [], [], [], [], []
        n_rigid_volume_verts = 0
        if self._is_rigid_hydroelastic:
            for geom in rigid_solver.geoms:
                if geom.contype or geom.conaffinity:
                    if geom.type == gs.GEOM_TYPE.PLANE:
                        gs.raise_exception("Primitive plane not supported as user-specified collision geometries.")
                    tet_cfg = {"nobisect": False, "maxvolume": geom.trimesh.volume / 100}
                    verts, elems = eu.split_all_surface_tets(*eu.mesh_to_elements(geom.trimesh, tet_cfg=tet_cfg))
                    volume_verts.append(verts)
                    volume_elems.append(elems + n_rigid_volume_verts)
                    volume_verts_geom_idx.append(np.full(len(verts), geom.idx, dtype=gs.np_int))
                    volume_elems_geom_idx.append(np.full(len(elems), geom.idx, dtype=gs.np_int))
                    volume_pressure.append(
                        compute_pressure_field(verts, geom.init_verts, geom.init_faces, options.hydroelastic_stiffness)
                    )
                    n_rigid_volume_verts += len(verts)
            if not volume_verts:
                gs.raise_exception("No rigid collision geometries found.")
        n_rigid_volume_elems = sum([len(elems) for elems in volume_elems])
        n_vertices, n_elements = fem_solver.n_vertices, fem_solver.n_elements
        n_surface_elements = fem_solver.n_surface_elements if fem_solver.is_active else 0
        n_dofs, n_faces = rigid_solver.n_dofs, rigid_solver.n_faces

        # Every handler, in the order every sum over them runs, the active ones alone detecting pairs
        fem_floor_tet = FEMFloorTetContactHandler(
            n_surface_elements,
            n_elements,
            B,
            is_active=fem_solver.is_active and options.fem_floor_contact_type == "tet",
        )
        fem_floor_vert = FEMFloorVertContactHandler(
            n_surface_elements,
            n_vertices,
            B,
            is_active=fem_solver.is_active and options.fem_floor_contact_type == "vert",
        )
        fem_self_tet = FEMSelfTetContactHandler(
            n_surface_elements, n_elements, B, is_active=fem_solver.is_active and options.enable_fem_self_tet_contact
        )
        rigid_floor_vert = RigidFloorVertContactHandler(
            rigid_solver.n_free_verts,
            rigid_solver.n_verts,
            B,
            is_active=rigid_solver.is_active and options.rigid_floor_contact_type == "vert",
        )
        rigid_floor_tet = RigidFloorTetContactHandler(
            n_rigid_volume_elems,
            B,
            is_active=rigid_solver.is_active and options.rigid_floor_contact_type == "tet",
        )
        rigid_rigid_tet = RigidRigidTetContactHandler(
            n_rigid_volume_elems,
            B,
            is_active=rigid_solver.is_active and options.rigid_rigid_contact_type == "tet",
        )
        rigid_fem_tri_tet = RigidFemTriTetContactHandler(
            n_surface_elements,
            n_elements,
            n_faces,
            B,
            is_active=rigid_solver.is_active and fem_solver.is_active and options.enable_rigid_fem_contact,
        )
        self._is_fem_hydroelastic = fem_floor_tet.is_active or fem_self_tet.is_active or rigid_fem_tri_tet.is_active
        self.contact_handlers.clear()
        self.contact_handlers.extend(
            [
                handler
                for handler in (
                    fem_floor_tet,
                    fem_floor_vert,
                    fem_self_tet,
                    rigid_floor_vert,
                    rigid_floor_tet,
                    rigid_rigid_tet,
                    rigid_fem_tri_tet,
                )
                if handler.is_active
            ]
        )

        # The handlers applying impulses to rigid dofs hold the first pairs, so that their pairs alone take a Jacobian.
        # The arrays of the handlers keep one slot when there is none, so that they are allocated.
        n_handlers = max(len(self.contact_handlers), 1)
        handlers_offset = np.zeros(n_handlers, dtype=gs.np_int)
        handlers_capacity = np.zeros(n_handlers, dtype=gs.np_int)
        handlers_rigid_rank = np.full(n_handlers, -1, dtype=gs.np_int)
        n_pairs = n_rigid_handlers = 0
        for is_rigid in (True, False):
            for i_h, handler in enumerate(self.contact_handlers):
                if handler.HAS_RIGID_JACOBIAN == is_rigid:
                    handler.idx = i_h
                    handlers_offset[i_h] = n_pairs
                    handlers_capacity[i_h] = handler.capacity
                    if is_rigid:
                        handlers_rigid_rank[i_h] = n_rigid_handlers
                        n_rigid_handlers += 1
                    n_pairs += handler.capacity
            if is_rigid:
                n_rigid_pairs = n_pairs
        n_entries = SAP_N_FEM_SLOTS * max(n_pairs, 1) if fem_solver.is_active else 0
        handlers_n_chunks = np.clip(handlers_capacity, 1, SUM_MAX_CHUNKS)

        # The pairs sort by environment, then by handler, then by key, one segment per environment and handler
        key_bits = max([max(1, (handler.n_keys - 1).bit_length()) for handler in self.contact_handlers] + [1])
        n_segments = B * n_handlers
        pairs_sort = get_segment_sort_data(
            n_segments,
            max(n_pairs, 1),
            max(handlers_capacity.max(), 1),
            (n_segments - 1).bit_length() + key_bits,
            has_values=True,
        )
        self._pairs_sort_config = pairs_sort.config

        # The trees the box queries traverse, the surface tets of the FEM entities serving both the self-contact tree
        # and the query of the rigid triangles
        self._has_fem_surface_tet_tree = fem_self_tet.is_active or rigid_fem_tri_tet.is_active
        bvhs = {
            SAPState.fem_surface_tet_bvh: get_bvh_data(
                B, n_surface_elements if self._has_fem_surface_tet_tree else 0, is_active=self._has_fem_surface_tet_tree
            ),
            SAPState.rigid_tri_bvh: get_bvh_data(B, n_faces, is_active=rigid_fem_tri_tet.is_active),
            SAPState.rigid_tet_bvh: get_bvh_data(
                B, n_rigid_volume_elems if rigid_rigid_tet.is_active else 0, is_active=rigid_rigid_tet.is_active
            ),
        }
        self.fem_surface_tet_bvh_config, self.rigid_tri_bvh_config, self.rigid_tet_bvh_config = [
            bvh.config for bvh in bvhs.values()
        ]

        # TODO: The joint equality constraints added after the build are not supported
        has_equalities = rigid_solver.is_active and rigid_solver.n_equalities > 0
        candidates_handler = {
            SAPState.candidates.fem_floor_tet: fem_floor_tet,
            SAPState.candidates.fem_self_tet: fem_self_tet,
            SAPState.candidates.rigid_floor_tet: rigid_floor_tet,
            SAPState.candidates.rigid_fem_tri_tet: rigid_fem_tri_tet,
            SAPState.candidates.rigid_rigid_tet: rigid_rigid_tet,
        }
        results_handler = {
            SAPState.queries.fem_self_results: fem_self_tet,
            SAPState.queries.rigid_tri_results: rigid_fem_tri_tet,
            SAPState.queries.rigid_tet_results: rigid_rigid_tet,
        }
        self.sap_state = SAPState(
            sizes={
                "envs": B,
                "vertices": n_vertices,
                "elements": n_elements,
                "dofs": n_dofs,
                "volume_verts": n_rigid_volume_verts,
                "volume_elems": n_rigid_volume_elems,
                "equality_constraints": rigid_solver.n_equalities * B,
                "contact_pairs": n_pairs,
                "rigid_contact_pairs": n_rigid_pairs,
                "handlers": n_handlers,
                "rigid_handlers": n_rigid_handlers,
                "entries": n_entries,
                "pair_chunks": handlers_n_chunks.max(),
                SAPState.fem.sums_v: FixedOrderSums.sizes(n_vertices, 3),
                SAPState.fem.sums_el: FixedOrderSums.sizes(n_elements, 1),
                SAPState.rigid.sums: FixedOrderSums.sizes(n_dofs, 3),
                SAPState.equalities.sums: FixedOrderSums.sizes(rigid_solver.n_equalities, 2),
                SAPState.pairs.sort: pairs_sort.sizes,
                **{ref: bvh.sizes for ref, bvh in bvhs.items()},
                **{ref: dict(handler_candidates=handler.n_candidates) for ref, handler in candidates_handler.items()},
                **{ref: dict(query_results=handler.n_query_results) for ref, handler in results_handler.items()},
                SAPState.queries.fem_self_filter: dict(filter_leaves=n_surface_elements),
                SAPState.queries.rigid_tet_filter: dict(
                    filter_leaves=n_rigid_volume_elems, filter_items=rigid_solver.n_geoms
                ),
            },
            unused={
                SAPState.pairs.sort: pairs_sort.unused,
                **{ref: bvh.unused for ref, bvh in bvhs.items()},
                SAPState.fem: not fem_solver.is_active,
                SAPState.rigid: not rigid_solver.is_active,
                SAPState.hydroelastic_fem: not self._is_fem_hydroelastic,
                SAPState.rigid_volume: not self._is_rigid_hydroelastic,
                SAPState.equalities: not has_equalities,
                SAPState.equalities.sums.terms: True,
                (SAPState.pairs.Jt, SAPState.pairs.M_inv_Jt): n_rigid_pairs == 0,
                (
                    SAPState.pairs.vertices_n_entries,
                    SAPState.pairs.vertices_entry_start,
                    SAPState.pairs.vertices_entry,
                ): n_entries == 0,
                SAPState.pairs.dofs_partials: n_rigid_handlers == 0,
                **{ref: not handler.is_active for ref, handler in candidates_handler.items()},
                **{ref: not handler.is_active for ref, handler in results_handler.items()},
                (SAPState.queries.fem_self_filter.pairs_table, SAPState.queries.rigid_tri_filter): True,
                SAPState.queries.fem_self_filter.leaves_map: not fem_self_tet.is_active,
                SAPState.queries.rigid_tet_filter: not rigid_rigid_tet.is_active,
            },
        )
        # The filters hold their own copy of the data the detection kernels also read under another argument (see
        # BVHQueryFilter)
        queries = self.sap_state.queries
        if fem_self_tet.is_active:
            elements_idx = qd_to_numpy(fem_solver.fem_info.surface.elements_idx)
            queries.fem_self_filter.leaves_map.from_numpy(qd_to_numpy(fem_solver.fem_info.elements.el2v)[elements_idx])
        if rigid_rigid_tet.is_active:
            leaves_map = np.zeros((n_rigid_volume_elems, 4), dtype=gs.np_int)
            leaves_map[:, 0] = np.concatenate(volume_elems_geom_idx)
            queries.rigid_tet_filter.leaves_map.from_numpy(leaves_map)

        self.sap_info = SAPInfo(
            sizes=dict(
                handlers=n_handlers,
                marching_tets_cases=len(MARCHING_TETS_EDGE_TABLE),
                tet_edges=len(TET_EDGES),
            ),
            values=dict(
                contacts=dict(key_bits=key_bits),
                substep_dt=self.sim.substep_dt,
                convergence_atol=options.sap_convergence_atol,
                convergence_rtol=options.sap_convergence_rtol,
                taud=options.sap_taud,
                beta=options.sap_beta,
                sigma=options.sap_sigma,
                pcg_threshold=options.pcg_threshold,
                linesearch_ftol=options.linesearch_ftol,
                linesearch_max_step_size=options.linesearch_max_step_size,
                hydroelastic_stiffness=options.hydroelastic_stiffness,
                point_contact_stiffness=options.point_contact_stiffness,
                equality_stiffness=EQUALITY_STIFFNESS,
                rigid_floor_vert_stiffness=RIGID_FLOOR_VERT_STIFFNESS,
                hydroelastic_eps=HYDROELASTIC_EPS,
                floor_height=fem_solver.floor_height,
                EPS=gs.EPS,
            ),
        )
        self.sap_static_config = SAPStaticConfig(
            has_fem=fem_solver.is_active,
            has_rigid=rigid_solver.is_active,
            has_rigid_contacts=n_rigid_pairs > 0,
            has_equalities=has_equalities,
            enable_vertex_constraints=fem_solver.is_active and fem_solver.fem_static_config.enable_vertex_constraints,
        )

        contacts_layout = self.sap_info.contacts
        contacts_layout.handlers_offset.from_numpy(handlers_offset)
        contacts_layout.handlers_capacity.from_numpy(handlers_capacity)
        contacts_layout.handlers_n_chunks.from_numpy(handlers_n_chunks)
        contacts_layout.handlers_rigid_rank.from_numpy(handlers_rigid_rank)
        self.sap_info.marching_tets_edge_table.from_numpy(np.array(MARCHING_TETS_EDGE_TABLE, dtype=gs.np_int))
        self.sap_info.tet_edges.from_numpy(np.array(TET_EDGES, dtype=gs.np_int))

        if self._is_fem_hydroelastic:
            fem_pressure = []
            for entity in fem_solver.entities:
                init_positions = tensor_to_array(entity.init_positions)
                fem_pressure.append(
                    compute_pressure_field(
                        init_positions, init_positions, entity.surface_triangles, entity.material.hydroelastic_modulus
                    )
                )
            self.sap_state.hydroelastic_fem.pressure.from_numpy(np.concatenate(fem_pressure))
        if self._is_rigid_hydroelastic:
            rigid_volume = self.sap_state.rigid_volume
            rigid_volume.verts_rest.from_numpy(np.concatenate(volume_verts, dtype=gs.np_float))
            rigid_volume.elems.from_numpy(np.concatenate(volume_elems, dtype=gs.np_int))
            rigid_volume.verts_geom_idx.from_numpy(np.concatenate(volume_verts_geom_idx))
            rigid_volume.elems_geom_idx.from_numpy(np.concatenate(volume_elems_geom_idx))
            rigid_volume.pressure.from_numpy(np.concatenate(volume_pressure))
            kernel_compute_rigid_pressure_gradient_rest(self.sap_state, self.sap_info)
        if has_equalities:
            kernel_build_equality_constraints(
                self.sap_state,
                rigid_solver.dyn_info.joints,
                rigid_solver.dyn_info.equalities,
                rigid_solver.rigid_config,
                rigid_solver.n_equalities,
            )

    def reset(self, envs_idx=None):
        pass

    # ------------------------------------------------------------------------------------
    # -------------------------------------- Main ----------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_post_coupling(self, f: int) -> None:
        """Integrate the rigid entities from the velocities the couplings solved."""
        self.rigid_integration(**self.rigid_solver.get_substep_values(is_recording=False))

    def preprocess(self, f):
        fem_solver, rigid_solver = self.fem_solver, self.rigid_solver
        sap_state, sap_info = self.sap_state, self.sap_info

        # Pressure fields, the poses of the vertices of the rigid geoms, and the leaves of the tree of the surface tets
        # of the FEM entities, which the handlers share
        if self._is_fem_hydroelastic:
            kernel_compute_fem_pressure_gradient(f, sap_state, fem_solver.fem_state, sap_info, fem_solver.fem_info)
        if rigid_solver.is_active:
            kernel_update_all_verts(rigid_solver.dyn_state, rigid_solver.dyn_info, rigid_solver.rigid_config)
        if self._is_rigid_hydroelastic:
            kernel_update_rigid_volume(sap_state, rigid_solver.dyn_state.geoms, sap_info)
        if self._has_fem_surface_tet_tree:
            kernel_compute_fem_surface_tet_aabb(
                f, self.sap_state.fem_surface_tet_bvh.leaves, fem_solver.fem_state, fem_solver.fem_info
            )

        # Contact detection, then the pairs in a fixed order, their Jacobian and their regularization
        if gs.use_zerocopy:
            handlers_n_pairs = qd_to_torch(sap_state.pairs.handlers_n_pairs, copy=False)
            handlers_n_pairs.zero_()
            handlers_is_overflow = qd_to_torch(sap_state.pairs.handlers_is_overflow, copy=False)
            handlers_is_overflow.zero_()
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_reset_contacts(sap_state)
        for handler in self.contact_handlers:
            handler.detect(f, self)
        self._has_contact, is_overflow = kernel_sort_contacts(
            sap_state, sap_info, self.sap_static_config, self._pairs_sort_config
        )
        if is_overflow:
            handlers_n_pairs = qd_to_numpy(sap_state.pairs.handlers_n_pairs)
            handlers_is_overflow = qd_to_numpy(sap_state.pairs.handlers_is_overflow)
            message = "Overflowed In Contact Query: \n"
            for handler in self.contact_handlers:
                name = type(handler).__name__
                if handlers_n_pairs[handler.idx] > handler.capacity:
                    message += f"{name} max contact pairs: {handler.capacity}, using {handlers_n_pairs[handler.idx]}\n"
                if handlers_is_overflow[handler.idx]:
                    message += f"{name} overflowed its candidates or its box query\n"
            gs.raise_exception(message)
        if self.sap_static_config.has_rigid_contacts:
            kernel_compute_jacobian(
                sap_state,
                rigid_solver.dyn_state.dofs,
                rigid_solver.dyn_state.links,
                rigid_solver.dyn_info.links,
                rigid_solver.rigid_config,
            )
        kernel_compute_regularization(
            sap_state,
            fem_solver.fem_state,
            rigid_solver.dyn_state.dofs,
            sap_info,
            rigid_solver.dyn_info.entities,
            rigid_solver.rigid_info,
            self.sap_static_config,
        )

    def couple(self, f):
        """
        Solve the SAP problem of the substep by Newton iterations on the velocities, then write the velocities back.

        Each iteration solves its Newton step by preconditioned conjugate gradient, then searches the step size along it
        by an exact line search (the rtsafe method of the SAP solver of Drake).
        """
        if not self._has_contact:
            return

        fem_solver, rigid_solver = self.fem_solver, self.rigid_solver
        sap_state, sap_info, static_config = self.sap_state, self.sap_info, self.sap_static_config
        fem_state, fem_info = fem_solver.fem_state, fem_solver.fem_info
        dyn_state, dofs_state = rigid_solver.dyn_state, rigid_solver.dyn_state.dofs
        rigid_info, rigid_config = rigid_solver.rigid_info, rigid_solver.rigid_config
        vertex_constraints = fem_solver.newton_vertex_constraints
        kernel_init_sap_solve(f, sap_state, fem_state, dofs_state, static_config)
        for i_iter in range(self._n_sap_iterations):
            # The free motion part of the gradient, zero at the first iteration where the velocities are free
            kernel_init_unconstrained_gradient_diag(f, sap_state, fem_state, dofs_state, fem_info, static_config)
            if i_iter > 0:
                kernel_compute_unconstrained_gradient(
                    vertex_constraints, sap_state, fem_state, sap_info, fem_info, rigid_info, static_config
                )

            # The constraints and the contacts, then the convergence
            kernel_compute_constraint_contact_gradient_hessian_diag_prec(
                vertex_constraints, sap_state, sap_info, static_config
            )
            kernel_check_sap_convergence(sap_state, sap_info, fem_info, rigid_info, static_config)

            # The Newton step
            kernel_init_pcg_solve(sap_state, dyn_state, sap_info, rigid_info, static_config, rigid_config)
            for _ in range(self._n_pcg_iterations):
                kernel_one_pcg_iter(
                    vertex_constraints,
                    sap_state,
                    fem_state,
                    dyn_state,
                    sap_info,
                    fem_info,
                    rigid_info,
                    static_config,
                    rigid_config,
                )

            # The step size
            kernel_init_exact_linesearch(
                f,
                vertex_constraints,
                sap_state,
                fem_state,
                dofs_state,
                sap_info,
                fem_info,
                rigid_info,
                static_config,
            )
            for _ in range(self._n_linesearch_iterations):
                kernel_one_exact_linesearch_iter(f, sap_state, fem_state, dofs_state, sap_info, static_config)
        kernel_update_vel(f, sap_state, fem_state, dofs_state, static_config)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- Properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def has_rigid_tet_contacts(self) -> bool:
        """Whether the coupler detects the contacts between the tetrahedra of the rigid geoms."""
        return any([isinstance(handler, RigidRigidTetContactHandler) for handler in self.contact_handlers])


@register_system
class SAPRigidContactFilter(System):
    """Restrict the contacts between the tetrahedra of rigid geoms to the pairs of geoms the collider lets collide."""

    def __init__(
        self, sap_coupler: Annotated[SAPCoupler, Enabled("has_rigid_tet_contacts")], collider: Collider
    ) -> None:
        self._sap_coupler = sap_coupler
        self._collider = collider

    def build(self) -> None:
        # The filter holds its own copy of the table the collider reads (see BVHQueryFilter)
        pairs_table = self._sap_coupler.sap_state.queries.rigid_tet_filter.pairs_table
        pairs_table.from_numpy(qd_to_numpy(self._collider.collider_info.collision_pair_idx))


# ------------------------------------------------------------------------------------
# -------------------------------- Contact handlers ----------------------------------
# ------------------------------------------------------------------------------------


class SAPContactHandler:
    """
    A kind of contact pairs of the SAP coupler, which it detects and appends to its segment of the pairs.

    The pairs of every handler go through one generic solve (see SAPContactPairs). A handler states the capacity of its
    segment, the number of distinct keys of its pairs within an environment, the capacity of its candidates and of its
    box query when it detects its pairs in two passes or through a tree, and whether its pairs apply impulses to rigid
    dofs, then detects them. An inactive handler detects nothing, its buffers taking the empty shape. `idx` is its
    index among the active handlers of the coupler.
    """

    # Whether the pairs apply their impulse to rigid dofs through a rigid Jacobian
    HAS_RIGID_JACOBIAN = False

    def __init__(
        self, capacity: int, n_keys: int, is_active: bool, n_candidates: int = 0, n_query_results: int = 0
    ) -> None:
        self.capacity = capacity
        self.n_keys = n_keys
        self.is_active = is_active
        self.n_candidates = n_candidates
        self.n_query_results = min(n_query_results, MAX_N_QUERY_RESULTS)
        self.idx = -1

    def detect(self, f: int, coupler: SAPCoupler) -> None:
        """Append the pairs of the substep to the segment of the handler."""
        raise NotImplementedError


class FEMFloorTetContactHandler(SAPContactHandler):
    """Contact of the tetrahedra of the FEM entities with the floor, by the hydroelastic model."""

    def __init__(self, n_surface_elements: int, n_elements: int, n_envs: int, is_active: bool) -> None:
        capacity = n_surface_elements * n_envs
        super().__init__(capacity, n_elements, is_active, n_candidates=capacity)

    def detect(self, f, coupler):
        fem_solver = coupler.fem_solver
        kernel_detect_fem_floor_tet(
            f, self.idx, coupler.sap_state, fem_solver.fem_state, coupler.sap_info, fem_solver.fem_info
        )


class FEMSelfTetContactHandler(SAPContactHandler):
    """Contact between the tetrahedra of the FEM entities, by the hydroelastic model."""

    def __init__(self, n_surface_elements: int, n_elements: int, n_envs: int, is_active: bool) -> None:
        capacity = n_surface_elements * n_envs
        super().__init__(
            capacity,
            n_elements**2,
            is_active,
            n_candidates=capacity * MAX_N_CANDIDATES_PER_PAIR,
            n_query_results=n_surface_elements * MAX_N_QUERY_RESULT_PER_AABB * n_envs,
        )

    def detect(self, f, coupler):
        fem_solver = coupler.fem_solver
        build_bvh(coupler.sap_state.fem_surface_tet_bvh, coupler.fem_surface_tet_bvh_config, eps=gs.EPS)
        kernel_detect_fem_self_tet(
            f, self.idx, coupler.sap_state, fem_solver.fem_state, coupler.sap_info, fem_solver.fem_info
        )


class FEMFloorVertContactHandler(SAPContactHandler):
    """Contact of the surface vertices of the FEM entities with the floor, by the point contact model."""

    def __init__(self, n_surface_elements: int, n_vertices: int, n_envs: int, is_active: bool) -> None:
        super().__init__(n_surface_elements * n_envs, n_vertices, is_active)

    def detect(self, f, coupler):
        fem_solver = coupler.fem_solver
        kernel_detect_fem_floor_vert(
            f, self.idx, coupler.sap_state, fem_solver.fem_state, coupler.sap_info, fem_solver.fem_info
        )


class RigidFloorVertContactHandler(SAPContactHandler):
    """Contact of the free vertices of the rigid geoms with the floor, by the point contact model."""

    HAS_RIGID_JACOBIAN = True

    def __init__(self, n_free_verts: int, n_verts: int, n_envs: int, is_active: bool) -> None:
        super().__init__(n_free_verts * n_envs, n_verts, is_active)

    def detect(self, f, coupler):
        rigid_solver = coupler.rigid_solver
        kernel_detect_rigid_floor_vert(
            self.idx,
            coupler.sap_state,
            rigid_solver.dyn_state.free_verts,
            coupler.sap_info,
            rigid_solver.dyn_info.verts,
            rigid_solver.dyn_info.geoms,
        )


class RigidFloorTetContactHandler(SAPContactHandler):
    """Contact of the tetrahedra of the volume meshes of the rigid geoms with the floor, by the hydroelastic model."""

    HAS_RIGID_JACOBIAN = True

    def __init__(self, n_rigid_volume_elems: int, n_envs: int, is_active: bool) -> None:
        capacity = n_rigid_volume_elems * n_envs
        super().__init__(capacity, n_rigid_volume_elems, is_active, n_candidates=capacity * MAX_N_CANDIDATES_PER_PAIR)

    def detect(self, f, coupler):
        rigid_solver = coupler.rigid_solver
        kernel_detect_rigid_floor_tet(
            self.idx,
            coupler.sap_state,
            coupler.sap_info,
            rigid_solver.dyn_info.links,
            rigid_solver.dyn_info.geoms,
            rigid_solver.rigid_config,
        )


class RigidRigidTetContactHandler(SAPContactHandler):
    """Contact between the tetrahedra of the volume meshes of the rigid geoms, by the hydroelastic model."""

    HAS_RIGID_JACOBIAN = True

    def __init__(self, n_rigid_volume_elems: int, n_envs: int, is_active: bool) -> None:
        capacity = n_rigid_volume_elems * n_envs
        super().__init__(
            capacity,
            n_rigid_volume_elems**2,
            is_active,
            n_candidates=capacity * MAX_N_CANDIDATES_PER_PAIR,
            n_query_results=n_rigid_volume_elems * MAX_N_QUERY_RESULT_PER_AABB * n_envs,
        )

    def detect(self, f, coupler):
        kernel_compute_rigid_tet_aabb(coupler.sap_state)
        build_bvh(coupler.sap_state.rigid_tet_bvh, coupler.rigid_tet_bvh_config, eps=gs.EPS)
        kernel_detect_rigid_rigid_tet(
            self.idx, coupler.sap_state, coupler.sap_info, coupler.rigid_solver.dyn_info.geoms
        )


class RigidFemTriTetContactHandler(SAPContactHandler):
    """
    Contact of the triangles of the rigid geoms with the tetrahedra of the FEM entities, by the hydroelastic model.
    """

    HAS_RIGID_JACOBIAN = True

    def __init__(self, n_surface_elements: int, n_elements: int, n_faces: int, n_envs: int, is_active: bool) -> None:
        capacity = max(n_surface_elements, n_faces) * n_envs
        super().__init__(
            capacity,
            n_faces * n_elements,
            is_active,
            n_candidates=capacity * MAX_N_CANDIDATES_PER_PAIR,
            n_query_results=max(n_surface_elements, n_faces) * MAX_N_QUERY_RESULT_PER_AABB * n_envs,
        )

    def detect(self, f, coupler):
        fem_solver, rigid_solver = coupler.fem_solver, coupler.rigid_solver
        kernel_compute_rigid_tri_aabb(
            coupler.sap_state.rigid_tri_bvh.leaves,
            rigid_solver.dyn_state.free_verts,
            rigid_solver.dyn_state.fixed_verts,
            rigid_solver.dyn_info.verts,
            rigid_solver.dyn_info.faces,
        )
        build_bvh(coupler.sap_state.rigid_tri_bvh, coupler.rigid_tri_bvh_config, eps=gs.EPS)
        kernel_detect_rigid_fem_tri_tet(
            f,
            self.idx,
            coupler.sap_state,
            fem_solver.fem_state,
            rigid_solver.dyn_state.free_verts,
            rigid_solver.dyn_state.fixed_verts,
            coupler.sap_info,
            fem_solver.fem_info,
            rigid_solver.dyn_info.verts,
            rigid_solver.dyn_info.faces,
            rigid_solver.dyn_info.geoms,
        )


# ------------------------------------------------------------------------------------
# ---------------------------------- Geometry funcs ----------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_tet_barycentric(p: qd.types.vector(3), tet_vertices: qd.types.matrix(3, 4)):
    """Return the barycentric coordinates of a point in the tetrahedron whose vertices are the columns of a matrix."""
    v0 = tet_vertices[:, 0]
    v1 = tet_vertices[:, 1]
    v2 = tet_vertices[:, 2]
    v3 = tet_vertices[:, 3]

    # The volumes of the tetrahedra formed by the point and the vertices
    vol_tet_inv = 1.0 / ((v1 - v0).dot((v2 - v0).cross(v3 - v0)))

    b0 = (p - v1).dot((v3 - v1).cross(v2 - v1)) * vol_tet_inv
    b1 = (p - v2).dot((v3 - v2).cross(v0 - v2)) * vol_tet_inv
    b2 = (p - v3).dot((v1 - v3).cross(v0 - v3)) * vol_tet_inv
    b3 = 1.0 - b0 - b1 - b2

    return qd.Vector([b0, b1, b2, b3], dt=gs.qd_float)


@qd.func
def func_polygon_area_centroid(polygon_vertices, polygon_n_vertices: int, area_start: float):
    """
    Return the area of a convex polygon and its area-weighted centroid, the area starting from a given value.

    The polygon is the first `polygon_n_vertices` columns of a matrix, split into the triangles fanning out of its first
    vertex.
    """
    total_area = area_start
    total_area_weighted_centroid = qd.Vector.zero(gs.qd_float, 3)
    for i_v in range(2, polygon_n_vertices):
        e1 = polygon_vertices[:, i_v - 1] - polygon_vertices[:, 0]
        e2 = polygon_vertices[:, i_v] - polygon_vertices[:, 0]
        area = 0.5 * e1.cross(e2).norm()
        total_area += area
        total_area_weighted_centroid += (
            area * (polygon_vertices[:, 0] + polygon_vertices[:, i_v - 1] + polygon_vertices[:, i_v]) / 3.0
        )
    return total_area, total_area_weighted_centroid


@qd.func
def func_intersection_code(distance: qd.types.vector(4)) -> int:
    """Return the code of the edges a plane cuts in a tetrahedron, one bit per vertex above it (see SAPInfo)."""
    intersection_code = 0
    for i in qd.static(range(4)):
        if distance[i] > 0.0:
            intersection_code |= 1 << i
    return intersection_code


@qd.func
def func_tet_plane_section(
    intersection_code: int,
    tet_vertices: qd.types.matrix(3, 4),
    distance: qd.types.vector(4),
    sap_info: SAPInfo,
):
    """
    Return the section of a tetrahedron by a plane, and its number of vertices.

    The section is the first columns of a matrix, wide enough for the section to be clipped by another tetrahedron (see
    func_clip_polygon_by_tet). `distance` holds the signed distance of the vertices to the plane.
    """
    intersected_edges = sap_info.marching_tets_edge_table[intersection_code]
    polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 8)
    polygon_n_vertices = 0
    for i in range(4):
        i_edge = intersected_edges[i]
        if i_edge >= 0:
            edge = sap_info.tet_edges[i_edge]
            pos_v0 = tet_vertices[:, edge[0]]
            pos_v1 = tet_vertices[:, edge[1]]
            d_v0 = distance[edge[0]]
            d_v1 = distance[edge[1]]
            t = d_v0 / (d_v0 - d_v1)
            polygon_vertices[:, polygon_n_vertices] = pos_v0 + t * (pos_v1 - pos_v0)
            polygon_n_vertices += 1
    return polygon_vertices, polygon_n_vertices


@qd.func
def func_tet_pressure_gradient(
    pos_v0: qd.types.vector(3),
    pos_v1: qd.types.vector(3),
    pos_v2: qd.types.vector(3),
    pos_v3: qd.types.vector(3),
    pressure_v0: float,
    EPS: float,
):
    """Return the contribution of the first vertex of a tetrahedron to the gradient of a linear field over it."""
    grad = qd.Vector.zero(gs.qd_float, 3)
    area_vector = (pos_v2 - pos_v1).cross(pos_v3 - pos_v1)
    signed_volume = area_vector.dot(pos_v0 - pos_v1)
    if qd.abs(signed_volume) > EPS:
        grad = area_vector / signed_volume * pressure_v0
    return grad


@qd.func
def func_equal_pressure_plane(
    x0: qd.types.vector(3),
    x1: qd.types.vector(3),
    p0: float,
    p1: float,
    g0: qd.types.vector(3),
    g1: qd.types.vector(3),
    EPS: float,
):
    """
    Return whether two linear pressure fields meet on a plane, its unit normal along g0 - g1, and a point of it.

    The fields take the pressure p0 at x0 with the gradient g0 and the pressure p1 at x1 with the gradient g1. They meet
    on the plane p0 + g0.dot(x - x0) = p1 + g1.dot(x - x1), which exists unless a gradient or their difference is
    smaller than EPS.
    """
    normal = g0 - g1
    x = qd.Vector.zero(gs.qd_float, 3)
    is_valid = not (g0.norm() < EPS or g1.norm() < EPS)
    if is_valid:
        magnitude = normal.norm()
        is_valid = not magnitude < EPS
        if is_valid:
            normal /= magnitude
            b = p1 - p0 - g1.dot(x1) + g0.dot(x0)
            x = b / magnitude * normal
    return is_valid, normal, x


@qd.func
def func_filter_fem_surface_tets(i_t: int, i_a: int, i_q: int, filter_data: BVHQueryFilter) -> bool:
    """
    Drop a pair of FEM surface tets once per unordered pair, and the pairs sharing a vertex.

    The leaves map to the vertices of their tet (see SAPContactQueries).
    """
    is_dropped = i_a >= i_q
    el2v_a = filter_data.leaves_map[i_a]
    el2v_q = filter_data.leaves_map[i_q]
    for i, j in qd.static(qd.ndrange(4, 4)):
        if el2v_a[i] == el2v_q[j]:
            is_dropped = True
    return is_dropped


@qd.func
def func_filter_rigid_tets(i_t: int, i_a: int, i_q: int, filter_data: BVHQueryFilter) -> bool:
    """
    Drop a pair of rigid tets whose geoms hold no collision pair.

    The leaves map to their geom in their first item, and the table holds the collision pair of each pair of geoms (see
    SAPContactQueries).
    The tree and the queries hold the same tets, so every pair of geoms is found in both orders, the one of increasing
    geom index alone being kept.
    """
    i_ga = filter_data.leaves_map[i_a][0]
    i_gq = filter_data.leaves_map[i_q][0]
    return i_ga >= i_gq or filter_data.pairs_table[i_ga, i_gq] == -1


# ------------------------------------------------------------------------------------
# ------------------------------------ Pair funcs ------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_append_pair(
    i_h: int,
    i_b: int,
    key: qd.u64,
    k: float,
    phi0: float,
    mu: float,
    frame: qd.types.matrix(3, 3),
    contact_pos: qd.types.vector(3),
    sap_state: SAPState,
    sap_info: SAPInfo,
):
    """
    Append a pair to the segment of a handler, and return its index, -1 when the segment is full.

    The pair applies its impulse to no FEM vertex and no link until the caller sets them.
    """
    i_p = -1
    i_p_ = qd.atomic_add(sap_state.pairs.handlers_n_pairs[i_h], 1)
    if i_p_ < sap_info.contacts.handlers_capacity[i_h]:
        i_p = sap_info.contacts.handlers_offset[i_h] + i_p_
        sap_state.pairs.handler_idx[i_p] = i_h
        sap_state.pairs.batch_idx[i_p] = i_b
        sap_state.pairs.key[i_p] = key
        sap_state.pairs.frame[i_p] = frame
        sap_state.pairs.contact_pos[i_p] = contact_pos
        fem_slots_vert = sap_state.pairs.fem_slots_vert[i_p]
        sap_state.pairs.fem_slots_vert[i_p] = qd.Vector.zero(gs.qd_int, fem_slots_vert.n) - 1
        sap_state.pairs.fem_slots_weight[i_p] = qd.Vector.zero(gs.qd_float, fem_slots_vert.n)
        sap_state.pairs.link_idx[i_p] = qd.Vector([-1, -1], dt=gs.qd_int)
        sap_state.pairs.link_sign[i_p] = qd.Vector.zero(gs.qd_float, 2)
        sap_state.pairs.k[i_p] = k
        sap_state.pairs.phi0[i_p] = phi0
        sap_state.pairs.mu[i_p] = mu
    return i_p


@qd.func
def func_contact_velocity(
    i_p: int,
    x_fem: qd.template(),
    x_rigid: qd.template(),
    sap_state: SAPState,
    static_config: qd.template(),
):
    """
    Return the Jacobian of a pair times the velocities of the FEM vertices and of the rigid dofs, in its frame.

    The FEM side sums the velocities of the vertices of the pair, weighted, and the rigid side the Jacobian rows of its
    links (see kernel_compute_jacobian).
    """
    i_b = sap_state.pairs.batch_idx[i_p]
    Jx = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(static_config.has_fem):
        fem_slots_vert = sap_state.pairs.fem_slots_vert[i_p]
        fem_slots_weight = sap_state.pairs.fem_slots_weight[i_p]
        for i_slot in qd.static(range(fem_slots_vert.n)):
            i_v = fem_slots_vert[i_slot]
            if i_v >= 0:
                Jx = Jx + fem_slots_weight[i_slot] * x_fem[i_v, i_b]
    if qd.static(static_config.has_rigid_contacts):
        if sap_state.pairs.link_idx[i_p][0] >= 0:
            for i_d in range(x_rigid.shape[0]):
                Jx = Jx + sap_state.pairs.Jt[i_p, i_d] * x_rigid[i_d, i_b]
    return sap_state.pairs.frame[i_p].transpose() @ Jx


@qd.func
def func_contact_gamma(
    i_p: int,
    vc: qd.types.vector(3),
    sap_state: SAPState,
    sap_info: SAPInfo,
    compute_hessian: qd.template(),
):
    """
    Compute the impulse of a pair in its frame from its velocity, and return its energy.

    With `compute_hessian`, the Hessian of the energy of the pair in its frame is computed too.
    """
    y = qd.Vector([0.0, 0.0, sap_state.pairs.vn_hat[i_p]]) - vc
    y[0] *= sap_state.pairs.Rt_inv[i_p]
    y[1] *= sap_state.pairs.Rt_inv[i_p]
    y[2] *= sap_state.pairs.Rn_inv[i_p]
    yr = y[:2].norm(sap_info.EPS[None])
    yn = y[2]

    t_hat = y[:2] / yr
    mu = sap_state.pairs.mu[i_p]
    mu_hat = sap_state.pairs.mu_hat[i_p]
    mu_factor = sap_state.pairs.mu_factor[i_p]
    gamma = qd.Vector.zero(gs.qd_float, 3)
    G = qd.Matrix.zero(gs.qd_float, 3, 3)
    # The contact sticks, slides or separates, from its friction and its velocity scaled by its regularization
    if yr <= mu * yn:
        gamma = y
        if qd.static(compute_hessian):
            G[0, 0] = sap_state.pairs.Rt_inv[i_p]
            G[1, 1] = sap_state.pairs.Rt_inv[i_p]
            G[2, 2] = sap_state.pairs.Rn_inv[i_p]
    elif -mu_hat * yr < yn and yn < yr / mu:
        gn = (yn + mu_hat * yr) * mu_factor
        gt = mu * gn * t_hat
        gamma = qd.Vector([gt[0], gt[1], gn])
        if qd.static(compute_hessian):
            P = t_hat.outer_product(t_hat)
            Pperp = qd.Matrix.identity(gs.qd_float, 2) - P
            dgt_dyt = mu * (gn / yr * Pperp + mu_hat * mu_factor * P)
            dgt_dyn = mu * mu_factor * t_hat
            dgn_dyt = mu_hat * mu_factor * t_hat
            dgn_dyn = mu_factor

            G[:2, :2] = dgt_dyt * sap_state.pairs.Rt_inv[i_p]
            G[:2, 2] = dgt_dyn * sap_state.pairs.Rn_inv[i_p]
            G[2, :2] = dgn_dyt * sap_state.pairs.Rt_inv[i_p]
            G[2, 2] = dgn_dyn * sap_state.pairs.Rn_inv[i_p]
    sap_state.pairs.gamma[i_p] = gamma
    if qd.static(compute_hessian):
        sap_state.pairs.G[i_p] = G

    R_gamma = gamma
    R_gamma[0] *= sap_state.pairs.Rt[i_p]
    R_gamma[1] *= sap_state.pairs.Rt[i_p]
    R_gamma[2] *= sap_state.pairs.Rn[i_p]
    return 0.5 * gamma.dot(R_gamma)


@qd.func
def func_fem_contacts_Jt(
    i_v: int,
    i_b: int,
    vertex_constraints: FEMVertexConstraintsInfo,
    sap_state: SAPState,
    static_config: qd.template(),
    compute_hessian: qd.template(),
):
    """
    Return the transposed FEM Jacobian of the pairs times their impulses at a FEM vertex, zero if it is constrained.

    With `compute_hessian`, the 3x3 diagonal block of the contact Hessian at the vertex is returned too, zero otherwise.
    The vertex sums its slots in the sorted order (see SAPContactPairs), the slots of each handler apart, then the
    handlers in order.
    """
    Jt_vec = qd.Vector.zero(gs.qd_float, 3)
    Jt_A_J = qd.Matrix.zero(gs.qd_float, 3, 3)
    is_free = True
    if qd.static(static_config.enable_vertex_constraints):
        is_free = not vertex_constraints.is_constrained[i_v, i_b]
    if is_free:
        Jt_vec_handler = qd.Vector.zero(gs.qd_float, 3)
        Jt_A_J_handler = qd.Matrix.zero(gs.qd_float, 3, 3)
        i_h_cur = -1
        # See SAP_N_FEM_SLOTS for the slots of a pair
        fem_slots_vert = sap_state.pairs.fem_slots_vert[0]
        n_slots = fem_slots_vert.n
        for i_entry in range(
            sap_state.pairs.vertices_entry_start[i_v, i_b], sap_state.pairs.vertices_entry_start[i_v + 1, i_b]
        ):
            i_slot_entry = sap_state.pairs.vertices_entry[i_entry]
            i_p = sap_state.pairs.sort.values[i_slot_entry // n_slots]
            i_h = sap_state.pairs.handler_idx[i_p]
            if i_h != i_h_cur:
                Jt_vec += Jt_vec_handler
                Jt_vec_handler = qd.Vector.zero(gs.qd_float, 3)
                if qd.static(compute_hessian):
                    Jt_A_J += Jt_A_J_handler
                    Jt_A_J_handler = qd.Matrix.zero(gs.qd_float, 3, 3)
                i_h_cur = i_h
            weight = sap_state.pairs.fem_slots_weight[i_p][i_slot_entry % n_slots]
            Jt_vec_handler += weight * sap_state.pairs.vec[i_p]
            if qd.static(compute_hessian):
                Jt_A_J_handler += weight**2 * sap_state.pairs.mat[i_p]
        Jt_vec += Jt_vec_handler
        if qd.static(compute_hessian):
            Jt_A_J += Jt_A_J_handler
    return Jt_vec, Jt_A_J


# ------------------------------------------------------------------------------------
# --------------------------------- Geometry kernels ---------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_compute_fem_pressure_gradient(
    f: int,
    sap_state: SAPState,
    fem_state: FEMState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
):
    """Compute the gradient of the pressure field of the FEM entities in every element, at the positions of a frame."""
    for i_e, i_b in qd.ndrange(
        sap_state.hydroelastic_fem.pressure_gradient.shape[0], sap_state.hydroelastic_fem.pressure_gradient.shape[1]
    ):
        el2v = fem_info.elements.el2v[i_e]
        grad = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v0 = el2v[i]
            i_v1 = el2v[(i + 1) % 4]
            i_v2 = el2v[(i + 2) % 4]
            i_v3 = el2v[(i + 3) % 4]
            grad += func_tet_pressure_gradient(
                fem_state.vertices.pos[f, i_v0, i_b],
                fem_state.vertices.pos[f, i_v1, i_b],
                fem_state.vertices.pos[f, i_v2, i_b],
                fem_state.vertices.pos[f, i_v3, i_b],
                sap_state.hydroelastic_fem.pressure[i_v0],
                sap_info.EPS[None],
            )
        sap_state.hydroelastic_fem.pressure_gradient[i_e, i_b] = grad


@qd.kernel(fastcache=True)
def kernel_compute_rigid_pressure_gradient_rest(sap_state: SAPState, sap_info: SAPInfo):
    """Compute the gradient of the pressure field of the rigid geoms in every element of their volume, at rest."""
    for i_e in range(sap_state.rigid_volume.elems.shape[0]):
        elem = sap_state.rigid_volume.elems[i_e]
        grad = qd.Vector.zero(gs.qd_float, 3)
        for i in qd.static(range(4)):
            i_v0 = elem[i]
            i_v1 = elem[(i + 1) % 4]
            i_v2 = elem[(i + 2) % 4]
            i_v3 = elem[(i + 3) % 4]
            grad += func_tet_pressure_gradient(
                sap_state.rigid_volume.verts_rest[i_v0],
                sap_state.rigid_volume.verts_rest[i_v1],
                sap_state.rigid_volume.verts_rest[i_v2],
                sap_state.rigid_volume.verts_rest[i_v3],
                sap_state.rigid_volume.pressure[i_v0],
                sap_info.EPS[None],
            )
        sap_state.rigid_volume.pressure_gradient_rest[i_e] = grad


@qd.kernel(fastcache=True)
def kernel_update_rigid_volume(sap_state: SAPState, geoms_state: GeomsState, sap_info: SAPInfo):
    """Move the volume meshes of the rigid geoms and the gradient of their pressure field to the pose of their geom."""
    for i_v, i_b in qd.ndrange(sap_state.rigid_volume.verts.shape[0], sap_state.rigid_volume.verts.shape[1]):
        i_g = sap_state.rigid_volume.verts_geom_idx[i_v]
        R = gu.qd_quat_to_R(geoms_state.quat[i_g, i_b], sap_info.EPS[None])
        sap_state.rigid_volume.verts[i_v, i_b] = R @ sap_state.rigid_volume.verts_rest[i_v] + geoms_state.pos[i_g, i_b]

    for i_e, i_b in qd.ndrange(
        sap_state.rigid_volume.pressure_gradient.shape[0], sap_state.rigid_volume.pressure_gradient.shape[1]
    ):
        i_g = sap_state.rigid_volume.elems_geom_idx[i_e]
        R = gu.qd_quat_to_R(geoms_state.quat[i_g, i_b], sap_info.EPS[None])
        sap_state.rigid_volume.pressure_gradient[i_e, i_b] = R @ sap_state.rigid_volume.pressure_gradient_rest[i_e]


@qd.kernel(fastcache=True)
def kernel_compute_fem_surface_tet_aabb(f: int, bvh_leaves: BVHLeaves, fem_state: FEMState, fem_info: FEMInfo):
    """Compute the boxes of the surface tetrahedra of the FEM entities in a frame, the leaves of their tree."""
    for i_b, i_se in qd.ndrange(bvh_leaves.aabbs_min.shape[0], bvh_leaves.aabbs_min.shape[1]):
        i_e = fem_info.surface.elements_idx[i_se]
        el2v = fem_info.elements.el2v[i_e]
        leaf_min = qd.Vector([qd.math.inf, qd.math.inf, qd.math.inf], dt=gs.qd_float)
        leaf_max = -leaf_min
        for i in qd.static(range(4)):
            pos_v = fem_state.vertices.pos[f, el2v[i], i_b]
            leaf_min = qd.min(leaf_min, pos_v)
            leaf_max = qd.max(leaf_max, pos_v)
        bvh_leaves.aabbs_min[i_b, i_se] = leaf_min
        bvh_leaves.aabbs_max[i_b, i_se] = leaf_max


@qd.func
def func_rigid_face_vertices(
    i_f: int,
    i_b: int,
    free_verts_state: VertsState,
    fixed_verts_state: VertsState,
    verts_info: VertsInfo,
    faces_info: FacesInfo,
):
    """Return the vertices of a face of the rigid geoms in an environment, as the columns of a matrix."""
    tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        i_v = faces_info.verts_idx[i_f][i]
        i_fv = verts_info.verts_state_idx[i_v]
        if verts_info.is_fixed[i_v]:
            tri_vertices[:, i] = fixed_verts_state.pos[i_fv]
        else:
            tri_vertices[:, i] = free_verts_state.pos[i_fv, i_b]
    return tri_vertices


@qd.kernel(fastcache=True)
def kernel_compute_rigid_tri_aabb(
    bvh_leaves: BVHLeaves,
    free_verts_state: VertsState,
    fixed_verts_state: VertsState,
    verts_info: VertsInfo,
    faces_info: FacesInfo,
):
    """Compute the boxes of the faces of the rigid geoms, the leaves of their tree."""
    for i_b, i_f in qd.ndrange(bvh_leaves.aabbs_min.shape[0], bvh_leaves.aabbs_min.shape[1]):
        tri_vertices = func_rigid_face_vertices(i_f, i_b, free_verts_state, fixed_verts_state, verts_info, faces_info)
        pos_v0, pos_v1, pos_v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
        bvh_leaves.aabbs_min[i_b, i_f] = qd.min(pos_v0, pos_v1, pos_v2)
        bvh_leaves.aabbs_max[i_b, i_f] = qd.max(pos_v0, pos_v1, pos_v2)


@qd.kernel(fastcache=True)
def kernel_compute_rigid_tet_aabb(sap_state: SAPState):
    """Compute the boxes of the elements of the volume meshes of the rigid geoms, the leaves of their tree."""
    for i_b, i_e in qd.ndrange(
        sap_state.rigid_tet_bvh.leaves.aabbs_min.shape[0], sap_state.rigid_tet_bvh.leaves.aabbs_min.shape[1]
    ):
        elem = sap_state.rigid_volume.elems[i_e]
        pos_v0 = sap_state.rigid_volume.verts[elem[0], i_b]
        pos_v1 = sap_state.rigid_volume.verts[elem[1], i_b]
        pos_v2 = sap_state.rigid_volume.verts[elem[2], i_b]
        pos_v3 = sap_state.rigid_volume.verts[elem[3], i_b]
        sap_state.rigid_tet_bvh.leaves.aabbs_min[i_b, i_e] = qd.min(pos_v0, pos_v1, pos_v2, pos_v3)
        sap_state.rigid_tet_bvh.leaves.aabbs_max[i_b, i_e] = qd.max(pos_v0, pos_v1, pos_v2, pos_v3)


# ------------------------------------------------------------------------------------
# -------------------------------- Detection funcs -----------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_clamp_to_capacity(i_h: int, n_items: int, capacity: int, sap_state: SAPState):
    """Return the items of a handler within the capacity of their buffer, flagging the handler on overflow."""
    if n_items > capacity:
        sap_state.pairs.handlers_is_overflow[i_h] = True
    return qd.min(n_items, capacity)


@qd.func
def func_append_tet_plane_candidate(
    i_e: int,
    i_b: int,
    intersection_code: int,
    distance: qd.types.vector(4),
    candidates: SAPTetPlaneCandidates,
):
    """Append a tetrahedron crossing a plane to the candidates of a handler, dropped when they are full."""
    i_c = qd.atomic_add(candidates.n_candidates[0], 1)
    if i_c < candidates.batch_idx.shape[0]:
        candidates.batch_idx[i_c] = i_b
        candidates.elem_idx[i_c] = i_e
        candidates.intersection_code[i_c] = intersection_code
        candidates.distance[i_c] = distance


@qd.func
def func_append_tet_tet_candidate(
    i_e0: int,
    i_e1: int,
    i_b: int,
    intersection_code0: int,
    normal: qd.types.vector(3),
    distance0: qd.types.vector(4),
    candidates: SAPTetTetCandidates,
):
    """Append a pair of tetrahedra crossing their plane of equal pressure to the candidates of a handler."""
    i_c = qd.atomic_add(candidates.n_candidates[0], 1)
    if i_c < candidates.batch_idx.shape[0]:
        candidates.batch_idx[i_c] = i_b
        candidates.normal[i_c] = normal
        candidates.elem_idx0[i_c] = i_e0
        candidates.intersection_code0[i_c] = intersection_code0
        candidates.distance0[i_c] = distance0
        candidates.elem_idx1[i_c] = i_e1


@qd.func
def func_tet_floor_contact(
    intersection_code: int,
    tet_vertices: qd.types.matrix(3, 4),
    tet_pressures: qd.types.vector(4),
    distance: qd.types.vector(4),
    sap_info: SAPInfo,
):
    """
    Return the area and the centroid of the section of a tetrahedron by the floor, and the barycentric coordinates and
    the pressure of the centroid.

    The area starts from EPS, which keeps the centroid finite for a degenerate section.
    """
    polygon_vertices, polygon_n_vertices = func_tet_plane_section(intersection_code, tet_vertices, distance, sap_info)
    total_area, total_area_weighted_centroid = func_polygon_area_centroid(
        polygon_vertices, polygon_n_vertices, sap_info.EPS[None]
    )
    centroid = total_area_weighted_centroid / total_area
    barycentric = func_tet_barycentric(centroid, tet_vertices)
    return total_area, centroid, barycentric, barycentric.dot(tet_pressures)


@qd.func
def func_tet_tet_contact(
    intersection_code0: int,
    tet_vertices0: qd.types.matrix(3, 4),
    tet_pressures0: qd.types.vector(4),
    distance0: qd.types.vector(4),
    tet_vertices1: qd.types.matrix(3, 4),
    normal: qd.types.vector(3),
    sap_info: SAPInfo,
):
    """
    Return whether two tetrahedra contact on their plane of equal pressure, over the section of the first one clipped by
    the second, and the area, the centroid, the barycentric coordinates in the first one, the pressure and the frame
    of the contact (see SAPContactPairs).

    `distance0` and `intersection_code0` give the section of the first tetrahedron by the plane. The tetrahedra contact
    when the clipped section holds 3 vertices and an area of `hydroelastic_eps` at least.
    """
    polygon_vertices, polygon_n_vertices = func_tet_plane_section(
        intersection_code0, tet_vertices0, distance0, sap_info
    )
    polygon_vertices, polygon_n_vertices = func_clip_polygon_by_tet(
        polygon_vertices, polygon_n_vertices, tet_vertices1, strict=False
    )
    total_area = gs.qd_float(0.0)
    centroid = qd.Vector.zero(gs.qd_float, 3)
    barycentric0 = qd.Vector.zero(gs.qd_float, 4)
    frame = qd.Matrix.zero(gs.qd_float, 3, 3)
    is_contact = polygon_n_vertices >= 3
    if is_contact:
        total_area, total_area_weighted_centroid = func_polygon_area_centroid(polygon_vertices, polygon_n_vertices, 0.0)
        is_contact = not total_area < sap_info.hydroelastic_eps[None]
        if is_contact:
            centroid = total_area_weighted_centroid / total_area
            barycentric0 = func_tet_barycentric(centroid, tet_vertices0)
            tangent0 = polygon_vertices[:, 0] - centroid
            tangent0 /= tangent0.norm()
            frame = qd.Matrix.cols([tangent0, normal.cross(tangent0), normal])
    return is_contact, total_area, centroid, barycentric0, barycentric0.dot(tet_pressures0), frame


@qd.func
def func_clip_polygon_by_tet(
    polygon_vertices, polygon_n_vertices: int, tet_vertices: qd.types.matrix(3, 4), strict: qd.template()
):
    """
    Clip a polygon by the four half-spaces bounding a tetrahedron, and return it with its number of vertices.

    The polygon is the first columns of a matrix. With `strict`, an edge is cut at a face only if its ends lie strictly
    on either side of it, otherwise as soon as one of its ends lies outside.
    """
    clipped_vertices = polygon_vertices
    normal_signs = qd.Vector([1.0, -1.0, 1.0, -1.0], dt=gs.qd_float)
    distances = qd.Vector.zero(gs.qd_float, polygon_vertices.m)
    for i_f in range(4):
        clipped_n_vertices = 0
        x = tet_vertices[:, (i_f + 1) % 4]
        normal = (tet_vertices[:, (i_f + 2) % 4] - x).cross(tet_vertices[:, (i_f + 3) % 4] - x) * normal_signs[i_f]
        normal /= normal.norm()

        for i_v in range(polygon_n_vertices):
            distances[i_v] = (polygon_vertices[:, i_v] - x).dot(normal)

        for i_v in range(polygon_n_vertices):
            j_v = (i_v + 1) % polygon_n_vertices
            is_inside = distances[i_v] <= 0.0
            is_cut = False
            if qd.static(strict):
                is_cut = distances[i_v] * distances[j_v] < 0.0
            elif is_inside:
                is_cut = distances[j_v] > 0.0
            else:
                is_cut = distances[j_v] <= 0.0
            if is_inside:
                clipped_vertices[:, clipped_n_vertices] = polygon_vertices[:, i_v]
                clipped_n_vertices += 1
            if is_cut:
                wa = distances[j_v] / (distances[j_v] - distances[i_v])
                wb = 1.0 - wa
                clipped_vertices[:, clipped_n_vertices] = wa * polygon_vertices[:, i_v] + wb * polygon_vertices[:, j_v]
                clipped_n_vertices += 1
        polygon_n_vertices = clipped_n_vertices
        polygon_vertices = clipped_vertices

        # A polygon of less than 3 vertices is no contact
        if polygon_n_vertices < 3:
            break
    return polygon_vertices, polygon_n_vertices


# ------------------------------------------------------------------------------------
# -------------------------------- Detection kernels ---------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_reset_contacts(sap_state: SAPState):
    """Empty the pairs of every handler, each handler emptying its candidates and its box query itself."""
    for i_h in range(sap_state.pairs.handlers_n_pairs.shape[0]):
        sap_state.pairs.handlers_n_pairs[i_h] = 0
        sap_state.pairs.handlers_is_overflow[i_h] = False


@qd.kernel(fastcache=True)
def kernel_detect_fem_floor_tet(
    f: int,
    i_h: int,
    sap_state: SAPState,
    fem_state: FEMState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
):
    """
    Detect the contact of the tetrahedra of the FEM entities with the floor, by the hydroelastic model.

    A tetrahedron crossing the floor contacts it at the centroid of its section, with the stiffness of the harmonic mean
    of the pressure gradient of the entity and of the hydroelastic stiffness of the floor, applying its impulse to its
    vertices by their barycentric weights.
    """
    sap_state.candidates.fem_floor_tet.n_candidates[0] = 0
    # TODO: Check the surface elements only instead of all the elements
    for i_e, i_b in qd.ndrange(fem_info.elements.el2v.shape[0], fem_state.vertices.pos.shape[2]):
        el2v = fem_info.elements.el2v[i_e]
        distance = qd.Vector.zero(gs.qd_float, 4)
        for i in qd.static(range(4)):
            distance[i] = fem_state.vertices.pos[f, el2v[i], i_b].z - sap_info.floor_height[None]
        intersection_code = func_intersection_code(distance)
        if intersection_code != 0 and intersection_code != 15:
            func_append_tet_plane_candidate(i_e, i_b, intersection_code, distance, sap_state.candidates.fem_floor_tet)

    n_candidates = func_clamp_to_capacity(
        i_h,
        sap_state.candidates.fem_floor_tet.n_candidates[0],
        sap_state.candidates.fem_floor_tet.batch_idx.shape[0],
        sap_state,
    )
    for i_c in range(n_candidates):
        i_b = sap_state.candidates.fem_floor_tet.batch_idx[i_c]
        i_e = sap_state.candidates.fem_floor_tet.elem_idx[i_c]
        el2v = fem_info.elements.el2v[i_e]
        tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)
        tet_pressures = qd.Vector.zero(gs.qd_float, 4)
        for i in qd.static(range(4)):
            tet_vertices[:, i] = fem_state.vertices.pos[f, el2v[i], i_b]
            tet_pressures[i] = sap_state.hydroelastic_fem.pressure[el2v[i]]
        total_area, centroid, barycentric, pressure = func_tet_floor_contact(
            sap_state.candidates.fem_floor_tet.intersection_code[i_c],
            tet_vertices,
            tet_pressures,
            sap_state.candidates.fem_floor_tet.distance[i_c],
            sap_info,
        )

        floor_g = sap_info.hydroelastic_stiffness[None]
        fem_g = sap_state.hydroelastic_fem.pressure_gradient[i_e, i_b].z
        # TODO A better way to handle corner cases where pressure and pressure gradient are ill defined
        if total_area < sap_info.hydroelastic_eps[None] or fem_g < sap_info.hydroelastic_eps[None]:
            continue
        # Harmonic mean
        g = 1.0 / (1.0 / floor_g + 1.0 / fem_g)
        k = total_area * g
        phi0 = -pressure / g
        if k < sap_info.hydroelastic_eps[None] or phi0 > sap_info.hydroelastic_eps[None]:
            continue
        mu = fem_info.elements.friction_mu[i_e]
        i_p = func_append_pair(
            i_h, i_b, qd.u64(i_e), k, phi0, mu, qd.Matrix.identity(gs.qd_float, 3), centroid, sap_state, sap_info
        )
        if i_p >= 0:
            for i in qd.static(range(4)):
                sap_state.pairs.fem_slots_vert[i_p][i] = el2v[i]
                sap_state.pairs.fem_slots_weight[i_p][i] = barycentric[i]


@qd.kernel(fastcache=True)
def kernel_detect_fem_floor_vert(
    f: int,
    i_h: int,
    sap_state: SAPState,
    fem_state: FEMState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
):
    """
    Detect the contact of the surface vertices of the FEM entities with the floor, by the point contact model.

    The stiffness of a vertex scales with its lumped surface mass.
    """
    for i_sv, i_b in qd.ndrange(fem_info.surface.vertices_idx.shape[0], fem_state.vertices.pos.shape[2]):
        i_v = fem_info.surface.vertices_idx[i_sv]
        pos_v = fem_state.vertices.pos[f, i_v, i_b]
        distance = pos_v.z - sap_info.floor_height[None]
        if distance <= 0.0:
            i_p = func_append_pair(
                i_h,
                i_b,
                qd.u64(i_v),
                sap_info.point_contact_stiffness[None] * fem_info.surface.vertices_mass[i_sv],
                distance,
                fem_info.vertices.friction_mu[i_v],
                qd.Matrix.identity(gs.qd_float, 3),
                pos_v,
                sap_state,
                sap_info,
            )
            if i_p >= 0:
                sap_state.pairs.fem_slots_vert[i_p][0] = i_v
                sap_state.pairs.fem_slots_weight[i_p][0] = 1.0


@qd.kernel(fastcache=True)
def kernel_detect_fem_self_tet(
    f: int,
    i_h: int,
    sap_state: SAPState,
    fem_state: FEMState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
):
    """
    Detect the contact between the surface tetrahedra of the FEM entities, by the hydroelastic model.

    Two tetrahedra contact on their plane of equal pressure, over the section of the first one clipped by the second,
    the vertices of the second one moving against those of the first. The plane follows the field intersection of the
    hydroelastic contact of Drake.
    """
    sap_state.queries.fem_self_results.count[0] = 0
    sap_state.candidates.fem_self_tet.n_candidates[0] = 0
    func_bvh_query_leaves(
        sap_state.fem_surface_tet_bvh.leaves,
        sap_state.fem_surface_tet_bvh.tree,
        sap_state.queries.fem_self_results,
        func_filter_fem_surface_tets,
        sap_state.queries.fem_self_filter,
    )

    # The pairs of tetrahedra crossing their plane of equal pressure
    n_results = func_clamp_to_capacity(
        i_h,
        sap_state.queries.fem_self_results.count[0],
        sap_state.queries.fem_self_results.triplets.shape[0],
        sap_state,
    )
    for i_r in range(n_results):
        i_b, i_sa, i_sq = sap_state.queries.fem_self_results.triplets[i_r]
        i_e0 = fem_info.surface.elements_idx[i_sa]
        i_e1 = fem_info.surface.elements_idx[i_sq]
        el2v0 = fem_info.elements.el2v[i_e0]
        el2v1 = fem_info.elements.el2v[i_e1]
        i_v0 = el2v0[0]
        i_v1 = el2v1[0]
        g0 = sap_state.hydroelastic_fem.pressure_gradient[i_e0, i_b]
        g1 = sap_state.hydroelastic_fem.pressure_gradient[i_e1, i_b]
        is_valid, normal, x = func_equal_pressure_plane(
            fem_state.vertices.pos[f, i_v0, i_b],
            fem_state.vertices.pos[f, i_v1, i_b],
            sap_state.hydroelastic_fem.pressure[i_v0],
            sap_state.hydroelastic_fem.pressure[i_v1],
            g0,
            g1,
            sap_info.EPS[None],
        )
        if not is_valid:
            continue
        # The normal must point along g0 and against g1, up to an allowance as in Drake
        threshold = qd.static(math.cos(math.pi * 5.0 / 8.0))
        if normal.dot(g0) < threshold * g0.norm() or normal.dot(g1) > -threshold * g1.norm():
            continue
        distance0 = qd.Vector.zero(gs.qd_float, 4)
        distance1 = qd.Vector.zero(gs.qd_float, 4)
        for i in qd.static(range(4)):
            distance0[i] = (fem_state.vertices.pos[f, el2v0[i], i_b] - x).dot(normal)
            distance1[i] = (fem_state.vertices.pos[f, el2v1[i], i_b] - x).dot(normal)
        intersection_code0 = func_intersection_code(distance0)
        intersection_code1 = func_intersection_code(distance1)
        # Both tetrahedra cross the plane
        if intersection_code0 == 0 or intersection_code1 == 0 or intersection_code0 == 15 or intersection_code1 == 15:
            continue
        func_append_tet_tet_candidate(
            i_e0, i_e1, i_b, intersection_code0, normal, distance0, sap_state.candidates.fem_self_tet
        )

    n_candidates = func_clamp_to_capacity(
        i_h,
        sap_state.candidates.fem_self_tet.n_candidates[0],
        sap_state.candidates.fem_self_tet.batch_idx.shape[0],
        sap_state,
    )
    for i_c in range(n_candidates):
        i_b = sap_state.candidates.fem_self_tet.batch_idx[i_c]
        i_e0 = sap_state.candidates.fem_self_tet.elem_idx0[i_c]
        i_e1 = sap_state.candidates.fem_self_tet.elem_idx1[i_c]
        normal = sap_state.candidates.fem_self_tet.normal[i_c]
        el2v0 = fem_info.elements.el2v[i_e0]
        el2v1 = fem_info.elements.el2v[i_e1]
        tet_vertices0 = qd.Matrix.zero(gs.qd_float, 3, 4)
        tet_pressures0 = qd.Vector.zero(gs.qd_float, 4)
        tet_vertices1 = qd.Matrix.zero(gs.qd_float, 3, 4)
        for i in qd.static(range(4)):
            tet_vertices0[:, i] = fem_state.vertices.pos[f, el2v0[i], i_b]
            tet_pressures0[i] = sap_state.hydroelastic_fem.pressure[el2v0[i]]
            tet_vertices1[:, i] = fem_state.vertices.pos[f, el2v1[i], i_b]
        is_contact, total_area, centroid, barycentric0, pressure, frame = func_tet_tet_contact(
            sap_state.candidates.fem_self_tet.intersection_code0[i_c],
            tet_vertices0,
            tet_pressures0,
            sap_state.candidates.fem_self_tet.distance0[i_c],
            tet_vertices1,
            normal,
            sap_info,
        )
        if not is_contact:
            continue

        barycentric1 = func_tet_barycentric(centroid, tet_vertices1)
        g0 = sap_state.hydroelastic_fem.pressure_gradient[i_e0, i_b].dot(normal)
        g1 = -sap_state.hydroelastic_fem.pressure_gradient[i_e1, i_b].dot(normal)
        # TODO: This is an approximated value, different from Drake, which actually calculates the distance
        phi0 = -pressure / g0 - pressure / g1
        if phi0 > sap_info.EPS[None]:
            continue

        n_elements = fem_info.elements.el2v.shape[0]
        i_p = func_append_pair(
            i_h,
            i_b,
            qd.u64(i_e0) * qd.u64(n_elements) + qd.u64(i_e1),
            total_area * sap_info.hydroelastic_stiffness[None],
            phi0,
            qd.sqrt(fem_info.elements.friction_mu[i_e0] * fem_info.elements.friction_mu[i_e1]),
            frame,
            centroid,
            sap_state,
            sap_info,
        )
        if i_p >= 0:
            for i in qd.static(range(4)):
                sap_state.pairs.fem_slots_vert[i_p][i] = el2v0[i]
                sap_state.pairs.fem_slots_weight[i_p][i] = barycentric0[i]
                sap_state.pairs.fem_slots_vert[i_p][4 + i] = el2v1[i]
                sap_state.pairs.fem_slots_weight[i_p][4 + i] = -barycentric1[i]


@qd.kernel(fastcache=True)
def kernel_detect_rigid_floor_vert(
    i_h: int,
    sap_state: SAPState,
    free_verts_state: VertsState,
    sap_info: SAPInfo,
    verts_info: VertsInfo,
    geoms_info: GeomsInfo,
):
    """Detect the contact of the free vertices of the rigid geoms with the floor, by the point contact model."""
    for i_v, i_b in qd.ndrange(verts_info.is_fixed.shape[0], free_verts_state.pos.shape[1]):
        if not verts_info.is_fixed[i_v]:
            i_fv = verts_info.verts_state_idx[i_v]
            pos_v = free_verts_state.pos[i_fv, i_b]
            distance = pos_v.z - sap_info.floor_height[None]
            if distance <= 0.0:
                i_g = verts_info.geom_idx[i_v]
                i_p = func_append_pair(
                    i_h,
                    i_b,
                    qd.u64(i_v),
                    sap_info.rigid_floor_vert_stiffness[None],
                    distance,
                    geoms_info.coup_friction[i_g],
                    qd.Matrix.identity(gs.qd_float, 3),
                    pos_v,
                    sap_state,
                    sap_info,
                )
                if i_p >= 0:
                    sap_state.pairs.link_idx[i_p][0] = geoms_info.link_idx[i_g]
                    sap_state.pairs.link_sign[i_p][0] = 1.0


@qd.kernel(fastcache=True)
def kernel_detect_rigid_floor_tet(
    i_h: int,
    sap_state: SAPState,
    sap_info: SAPInfo,
    links_info: LinksInfo,
    geoms_info: GeomsInfo,
    rigid_config: qd.template(),
):
    """
    Detect the contact of the tetrahedra of the volume meshes of the rigid geoms with the floor, by the hydroelastic
    model.

    A tetrahedron of a moving link crossing the floor contacts it at the centroid of its section, with the stiffness of
    the pressure gradient of its geom.
    """
    sap_state.candidates.rigid_floor_tet.n_candidates[0] = 0
    # TODO: Check the surface elements only instead of all the elements
    for i_e, i_b in qd.ndrange(sap_state.rigid_volume.elems.shape[0], sap_state.rigid_volume.verts.shape[1]):
        i_g = sap_state.rigid_volume.elems_geom_idx[i_e]
        i_l = geoms_info.link_idx[i_g]
        I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
        if not links_info.is_fixed[I_l]:
            elem = sap_state.rigid_volume.elems[i_e]
            distance = qd.Vector.zero(gs.qd_float, 4)
            for i in qd.static(range(4)):
                distance[i] = sap_state.rigid_volume.verts[elem[i], i_b].z - sap_info.floor_height[None]
            intersection_code = func_intersection_code(distance)
            if intersection_code != 0 and intersection_code != 15:
                func_append_tet_plane_candidate(
                    i_e, i_b, intersection_code, distance, sap_state.candidates.rigid_floor_tet
                )

    n_candidates = func_clamp_to_capacity(
        i_h,
        sap_state.candidates.rigid_floor_tet.n_candidates[0],
        sap_state.candidates.rigid_floor_tet.batch_idx.shape[0],
        sap_state,
    )
    for i_c in range(n_candidates):
        i_b = sap_state.candidates.rigid_floor_tet.batch_idx[i_c]
        i_e = sap_state.candidates.rigid_floor_tet.elem_idx[i_c]
        elem = sap_state.rigid_volume.elems[i_e]
        tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)
        tet_pressures = qd.Vector.zero(gs.qd_float, 4)
        for i in qd.static(range(4)):
            tet_vertices[:, i] = sap_state.rigid_volume.verts[elem[i], i_b]
            tet_pressures[i] = sap_state.rigid_volume.pressure[elem[i]]
        total_area, centroid, barycentric, pressure = func_tet_floor_contact(
            sap_state.candidates.rigid_floor_tet.intersection_code[i_c],
            tet_vertices,
            tet_pressures,
            sap_state.candidates.rigid_floor_tet.distance[i_c],
            sap_info,
        )

        g = sap_state.rigid_volume.pressure_gradient[i_e, i_b].z
        k = total_area * g
        phi0 = -pressure / g
        if k < sap_info.hydroelastic_eps[None] or phi0 > sap_info.hydroelastic_eps[None]:
            continue
        i_g = sap_state.rigid_volume.elems_geom_idx[i_e]
        mu = geoms_info.coup_friction[i_g]
        i_p = func_append_pair(
            i_h, i_b, qd.u64(i_e), k, phi0, mu, qd.Matrix.identity(gs.qd_float, 3), centroid, sap_state, sap_info
        )
        if i_p >= 0:
            sap_state.pairs.link_idx[i_p][0] = geoms_info.link_idx[i_g]
            sap_state.pairs.link_sign[i_p][0] = 1.0


@qd.kernel(fastcache=True)
def kernel_detect_rigid_rigid_tet(i_h: int, sap_state: SAPState, sap_info: SAPInfo, geoms_info: GeomsInfo):
    """
    Detect the contact between the tetrahedra of the volume meshes of the rigid geoms, by the hydroelastic model.

    Two tetrahedra contact on their plane of equal pressure, over the section of the first one clipped by the second,
    the link of the second one moving against the link of the first.
    """
    sap_state.queries.rigid_tet_results.count[0] = 0
    sap_state.candidates.rigid_rigid_tet.n_candidates[0] = 0
    func_bvh_query_leaves(
        sap_state.rigid_tet_bvh.leaves,
        sap_state.rigid_tet_bvh.tree,
        sap_state.queries.rigid_tet_results,
        func_filter_rigid_tets,
        sap_state.queries.rigid_tet_filter,
    )

    # The pairs of tetrahedra crossing their plane of equal pressure
    n_results = func_clamp_to_capacity(
        i_h,
        sap_state.queries.rigid_tet_results.count[0],
        sap_state.queries.rigid_tet_results.triplets.shape[0],
        sap_state,
    )
    for i_r in range(n_results):
        i_b, i_e0, i_e1 = sap_state.queries.rigid_tet_results.triplets[i_r]
        elem0 = sap_state.rigid_volume.elems[i_e0]
        elem1 = sap_state.rigid_volume.elems[i_e1]
        i_v0 = elem0[0]
        i_v1 = elem1[0]
        g0 = sap_state.rigid_volume.pressure_gradient[i_e0, i_b]
        g1 = sap_state.rigid_volume.pressure_gradient[i_e1, i_b]
        is_valid, normal, x = func_equal_pressure_plane(
            sap_state.rigid_volume.verts[i_v0, i_b],
            sap_state.rigid_volume.verts[i_v1, i_b],
            sap_state.rigid_volume.pressure[i_v0],
            sap_state.rigid_volume.pressure[i_v1],
            g0,
            g1,
            sap_info.EPS[None],
        )
        if not is_valid:
            continue
        # The normal must point along g0 and against g1
        if normal.dot(g0) < sap_info.hydroelastic_eps[None] or normal.dot(g1) > -sap_info.hydroelastic_eps[None]:
            continue
        distance0 = qd.Vector.zero(gs.qd_float, 4)
        distance1 = qd.Vector.zero(gs.qd_float, 4)
        for i in qd.static(range(4)):
            distance0[i] = (sap_state.rigid_volume.verts[elem0[i], i_b] - x).dot(normal)
            distance1[i] = (sap_state.rigid_volume.verts[elem1[i], i_b] - x).dot(normal)
        intersection_code0 = func_intersection_code(distance0)
        intersection_code1 = func_intersection_code(distance1)
        # Both tetrahedra cross the plane
        if intersection_code0 == 0 or intersection_code1 == 0 or intersection_code0 == 15 or intersection_code1 == 15:
            continue
        func_append_tet_tet_candidate(
            i_e0, i_e1, i_b, intersection_code0, normal, distance0, sap_state.candidates.rigid_rigid_tet
        )

    n_candidates = func_clamp_to_capacity(
        i_h,
        sap_state.candidates.rigid_rigid_tet.n_candidates[0],
        sap_state.candidates.rigid_rigid_tet.batch_idx.shape[0],
        sap_state,
    )
    for i_c in range(n_candidates):
        i_b = sap_state.candidates.rigid_rigid_tet.batch_idx[i_c]
        i_e0 = sap_state.candidates.rigid_rigid_tet.elem_idx0[i_c]
        i_e1 = sap_state.candidates.rigid_rigid_tet.elem_idx1[i_c]
        normal = sap_state.candidates.rigid_rigid_tet.normal[i_c]
        elem0 = sap_state.rigid_volume.elems[i_e0]
        elem1 = sap_state.rigid_volume.elems[i_e1]
        tet_vertices0 = qd.Matrix.zero(gs.qd_float, 3, 4)
        tet_pressures0 = qd.Vector.zero(gs.qd_float, 4)
        tet_vertices1 = qd.Matrix.zero(gs.qd_float, 3, 4)
        for i in qd.static(range(4)):
            tet_vertices0[:, i] = sap_state.rigid_volume.verts[elem0[i], i_b]
            tet_pressures0[i] = sap_state.rigid_volume.pressure[elem0[i]]
            tet_vertices1[:, i] = sap_state.rigid_volume.verts[elem1[i], i_b]
        is_contact, total_area, centroid, barycentric0, pressure, frame = func_tet_tet_contact(
            sap_state.candidates.rigid_rigid_tet.intersection_code0[i_c],
            tet_vertices0,
            tet_pressures0,
            sap_state.candidates.rigid_rigid_tet.distance0[i_c],
            tet_vertices1,
            normal,
            sap_info,
        )
        if not is_contact:
            continue

        g0 = sap_state.rigid_volume.pressure_gradient[i_e0, i_b].dot(normal)
        g1 = -sap_state.rigid_volume.pressure_gradient[i_e1, i_b].dot(normal)
        # Harmonic mean, which handles an infinite gradient
        g = 1.0 / (1.0 / g0 + 1.0 / g1)
        phi0 = -pressure / g
        if phi0 > sap_info.hydroelastic_eps[None]:
            continue
        i_g0 = sap_state.rigid_volume.elems_geom_idx[i_e0]
        i_g1 = sap_state.rigid_volume.elems_geom_idx[i_e1]
        n_elements = sap_state.rigid_volume.elems.shape[0]
        i_p = func_append_pair(
            i_h,
            i_b,
            qd.u64(i_e0) * qd.u64(n_elements) + qd.u64(i_e1),
            total_area * g,
            phi0,
            qd.sqrt(geoms_info.friction[i_g0] * geoms_info.friction[i_g1]),
            frame,
            centroid,
            sap_state,
            sap_info,
        )
        if i_p >= 0:
            sap_state.pairs.link_idx[i_p] = qd.Vector(
                [geoms_info.link_idx[i_g0], geoms_info.link_idx[i_g1]], dt=gs.qd_int
            )
            sap_state.pairs.link_sign[i_p] = qd.Vector([1.0, -1.0], dt=gs.qd_float)


@qd.kernel(fastcache=True)
def kernel_detect_rigid_fem_tri_tet(
    f: int,
    i_h: int,
    sap_state: SAPState,
    fem_state: FEMState,
    free_verts_state: VertsState,
    fixed_verts_state: VertsState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
    verts_info: VertsInfo,
    faces_info: FacesInfo,
    geoms_info: GeomsInfo,
):
    """
    Detect the contact of the triangles of the rigid geoms with the surface tetrahedra of the FEM entities, by the
    hydroelastic model.

    A triangle contacts a tetrahedron crossing its plane over the triangle clipped by the tetrahedron, the link of the
    triangle moving against the vertices of the tetrahedron, as in the mesh intersection of the hydroelastic contact of
    Drake.
    """
    sap_state.queries.rigid_tri_results.count[0] = 0
    sap_state.candidates.rigid_fem_tri_tet.n_candidates[0] = 0
    func_bvh_query_leaves(
        sap_state.fem_surface_tet_bvh.leaves,
        sap_state.rigid_tri_bvh.tree,
        sap_state.queries.rigid_tri_results,
        func_no_filter,
        sap_state.queries.rigid_tri_filter,
    )

    # The pairs of a triangle and a tetrahedron crossing its plane
    n_results = func_clamp_to_capacity(
        i_h,
        sap_state.queries.rigid_tri_results.count[0],
        sap_state.queries.rigid_tri_results.triplets.shape[0],
        sap_state,
    )
    for i_r in range(n_results):
        i_b, i_f, i_sq = sap_state.queries.rigid_tri_results.triplets[i_r]
        i_e = fem_info.surface.elements_idx[i_sq]
        tri_vertices = func_rigid_face_vertices(i_f, i_b, free_verts_state, fixed_verts_state, verts_info, faces_info)
        pos_v0, pos_v1, pos_v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

        normal = (pos_v1 - pos_v0).cross(pos_v2 - pos_v0)
        magnitude_sqr = normal.norm_sqr()
        if magnitude_sqr < sap_info.EPS[None]:
            continue
        normal *= qd.rsqrt(magnitude_sqr)
        if sap_state.hydroelastic_fem.pressure_gradient[i_e, i_b].dot(normal) < sap_info.EPS[None]:
            continue

        el2v = fem_info.elements.el2v[i_e]
        distance = qd.Vector.zero(gs.qd_float, 4)
        for i in qd.static(range(4)):
            distance[i] = (fem_state.vertices.pos[f, el2v[i], i_b] - pos_v0).dot(normal)
        intersection_code = func_intersection_code(distance)
        if intersection_code == 0 or intersection_code == 15:
            continue

        i_c = qd.atomic_add(sap_state.candidates.rigid_fem_tri_tet.n_candidates[0], 1)
        if i_c < sap_state.candidates.rigid_fem_tri_tet.batch_idx.shape[0]:
            sap_state.candidates.rigid_fem_tri_tet.batch_idx[i_c] = i_b
            sap_state.candidates.rigid_fem_tri_tet.normal[i_c] = normal
            sap_state.candidates.rigid_fem_tri_tet.elem_idx[i_c] = i_e
            sap_state.candidates.rigid_fem_tri_tet.face_idx[i_c] = i_f

    n_candidates = func_clamp_to_capacity(
        i_h,
        sap_state.candidates.rigid_fem_tri_tet.n_candidates[0],
        sap_state.candidates.rigid_fem_tri_tet.batch_idx.shape[0],
        sap_state,
    )
    for i_c in range(n_candidates):
        i_b = sap_state.candidates.rigid_fem_tri_tet.batch_idx[i_c]
        i_e = sap_state.candidates.rigid_fem_tri_tet.elem_idx[i_c]
        i_f = sap_state.candidates.rigid_fem_tri_tet.face_idx[i_c]
        normal = sap_state.candidates.rigid_fem_tri_tet.normal[i_c]
        el2v = fem_info.elements.el2v[i_e]

        tri_vertices = func_rigid_face_vertices(i_f, i_b, free_verts_state, fixed_verts_state, verts_info, faces_info)
        tet_vertices = qd.Matrix.zero(gs.qd_float, 3, 4)
        tet_pressures = qd.Vector.zero(gs.qd_float, 4)
        for i in qd.static(range(4)):
            tet_vertices[:, i] = fem_state.vertices.pos[f, el2v[i], i_b]
            tet_pressures[i] = sap_state.hydroelastic_fem.pressure[el2v[i]]

        # The triangle, clipped by the tetrahedron
        polygon_vertices = qd.Matrix.zero(gs.qd_float, 3, 7)
        for i in qd.static(range(3)):
            polygon_vertices[:, i] = tri_vertices[:, i]
        polygon_vertices, polygon_n_vertices = func_clip_polygon_by_tet(polygon_vertices, 3, tet_vertices, strict=True)
        if polygon_n_vertices < 3:
            continue

        total_area, total_area_weighted_centroid = func_polygon_area_centroid(polygon_vertices, polygon_n_vertices, 0.0)
        centroid = total_area_weighted_centroid / total_area
        barycentric = func_tet_barycentric(centroid, tet_vertices)
        tangent0 = (polygon_vertices[:, 0] - centroid).normalized()
        rigid_g = sap_info.hydroelastic_stiffness[None]
        fem_g = sap_state.hydroelastic_fem.pressure_gradient[i_e, i_b].dot(normal)
        pressure = barycentric.dot(tet_pressures)
        if total_area < sap_info.hydroelastic_eps[None] or fem_g < sap_info.hydroelastic_eps[None]:
            continue
        # Harmonic mean
        g = 1.0 / (1.0 / rigid_g + 1.0 / fem_g)
        i_g = faces_info.geom_idx[i_f]
        n_elements = fem_info.elements.el2v.shape[0]
        i_p = func_append_pair(
            i_h,
            i_b,
            qd.u64(i_f) * qd.u64(n_elements) + qd.u64(i_e),
            total_area * g,
            -pressure / g,
            qd.sqrt(fem_info.elements.friction_mu[i_e] * geoms_info.coup_friction[i_g]),
            qd.Matrix.cols([tangent0, normal.cross(tangent0), normal]),
            centroid,
            sap_state,
            sap_info,
        )
        if i_p >= 0:
            for i in qd.static(range(4)):
                sap_state.pairs.fem_slots_vert[i_p][i] = el2v[i]
                sap_state.pairs.fem_slots_weight[i_p][i] = barycentric[i]
            # The link of the triangle moves against the vertices of the tetrahedron
            sap_state.pairs.link_idx[i_p][0] = geoms_info.link_idx[i_g]
            sap_state.pairs.link_sign[i_p][0] = -1.0


# ------------------------------------------------------------------------------------
# ---------------------------------- Pairs kernels -----------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_sort_contacts(
    sap_state: SAPState,
    sap_info: SAPInfo,
    static_config: qd.template(),
    sort_config: qd.template(),
) -> tuple[bool, bool]:
    """
    Order the pairs of every handler by environment, then by handler, then by key, and index the slots applying to each
    FEM vertex in that order (see SAPContactPairs).

    The pairs are grouped by segment by counting, then the segment sort orders the pairs of each segment by key. The
    keys being unique within a segment, the order does not depend on the order in which the detection appended the
    pairs. Return whether any pair was detected, and whether a handler, its candidates or its box query overflowed.
    """
    n_handlers = sap_state.pairs.handlers_n_pairs.shape[0]
    for i_seg in range(sap_state.pairs.segments_n_pairs.shape[0]):
        sap_state.pairs.segments_n_pairs[i_seg] = 0
    for i_h, i_p_ in qd.ndrange(n_handlers, sap_state.pairs.handler_idx.shape[0]):
        if i_p_ < qd.min(sap_state.pairs.handlers_n_pairs[i_h], sap_info.contacts.handlers_capacity[i_h]):
            i_p = sap_info.contacts.handlers_offset[i_h] + i_p_
            i_b = sap_state.pairs.batch_idx[i_p]
            qd.atomic_add(sap_state.pairs.segments_n_pairs[i_b * n_handlers + i_h], 1)

    # The bounds of the segments, the overflows, and whether any pair was detected
    has_contact = False
    is_overflow = False
    qd.loop_config(serialize=True)
    for i_seg in range(sap_state.pairs.segments_n_pairs.shape[0]):
        if i_seg == 0:
            sap_state.pairs.sort.segments_start[0] = 0
            for i_h in range(n_handlers):
                if sap_state.pairs.handlers_n_pairs[i_h] > 0:
                    has_contact = True
                if (
                    sap_state.pairs.handlers_n_pairs[i_h] > sap_info.contacts.handlers_capacity[i_h]
                    or sap_state.pairs.handlers_is_overflow[i_h]
                ):
                    is_overflow = True
        sap_state.pairs.sort.segments_start[i_seg + 1] = (
            sap_state.pairs.sort.segments_start[i_seg] + sap_state.pairs.segments_n_pairs[i_seg]
        )
        if i_seg == sap_state.pairs.segments_n_pairs.shape[0] - 1:
            sap_state.pairs.sort.n_keys[()] = sap_state.pairs.sort.segments_start[i_seg + 1]
    for i_h, i_p_ in qd.ndrange(n_handlers, sap_state.pairs.handler_idx.shape[0]):
        if i_p_ < qd.min(sap_state.pairs.handlers_n_pairs[i_h], sap_info.contacts.handlers_capacity[i_h]):
            i_p = sap_info.contacts.handlers_offset[i_h] + i_p_
            i_seg = sap_state.pairs.batch_idx[i_p] * n_handlers + i_h
            i_s = (
                sap_state.pairs.sort.segments_start[i_seg]
                + qd.atomic_sub(sap_state.pairs.segments_n_pairs[i_seg], 1)
                - 1
            )
            sap_state.pairs.sort.keys[i_s] = (qd.u64(i_seg) << qd.u64(sap_info.contacts.key_bits[None])) | (
                sap_state.pairs.key[i_p]
            )
            sap_state.pairs.sort.values[i_s] = i_p
    func_sort_segments(sap_state.pairs.sort, sort_config)

    # The slots of the sorted pairs applying to each FEM vertex, a handful at most per vertex, sorted by insertion
    if qd.static(static_config.has_fem):
        n_vertices = sap_state.pairs.vertices_n_entries.shape[0]
        # See SAP_N_FEM_SLOTS for the slots of a pair
        fem_slots_vert = sap_state.pairs.fem_slots_vert[0]
        n_slots = fem_slots_vert.n
        for i_v, i_b in qd.ndrange(n_vertices, sap_state.pairs.vertices_n_entries.shape[1]):
            sap_state.pairs.vertices_n_entries[i_v, i_b] = 0
        for i_s, i_slot in qd.ndrange(sap_state.pairs.sort.n_keys[()], n_slots):
            i_p = sap_state.pairs.sort.values[i_s]
            i_v = sap_state.pairs.fem_slots_vert[i_p][i_slot]
            if i_v >= 0:
                i_b = sap_state.pairs.batch_idx[i_p]
                qd.atomic_add(sap_state.pairs.vertices_n_entries[i_v, i_b], 1)
        for i_b in range(sap_state.pairs.vertices_n_entries.shape[1]):
            i_entry = n_slots * sap_state.pairs.sort.segments_start[i_b * n_handlers]
            for i_v in range(n_vertices):
                sap_state.pairs.vertices_entry_start[i_v, i_b] = i_entry
                i_entry += sap_state.pairs.vertices_n_entries[i_v, i_b]
            sap_state.pairs.vertices_entry_start[n_vertices, i_b] = i_entry
        for i_s, i_slot in qd.ndrange(sap_state.pairs.sort.n_keys[()], n_slots):
            i_p = sap_state.pairs.sort.values[i_s]
            i_v = sap_state.pairs.fem_slots_vert[i_p][i_slot]
            if i_v >= 0:
                i_b = sap_state.pairs.batch_idx[i_p]
                i_entry = (
                    sap_state.pairs.vertices_entry_start[i_v, i_b]
                    + qd.atomic_sub(sap_state.pairs.vertices_n_entries[i_v, i_b], 1)
                    - 1
                )
                sap_state.pairs.vertices_entry[i_entry] = i_s * n_slots + i_slot
        for i_v, i_b in qd.ndrange(n_vertices, sap_state.pairs.vertices_n_entries.shape[1]):
            i_start = sap_state.pairs.vertices_entry_start[i_v, i_b]
            for i_entry in range(i_start + 1, sap_state.pairs.vertices_entry_start[i_v + 1, i_b]):
                i_slot_entry = sap_state.pairs.vertices_entry[i_entry]
                is_shifting = True
                for j_ in range(i_entry - i_start):
                    j_entry = i_entry - j_
                    if is_shifting:
                        i_slot_entry_prev = sap_state.pairs.vertices_entry[j_entry - 1]
                        if i_slot_entry_prev > i_slot_entry:
                            sap_state.pairs.vertices_entry[j_entry] = i_slot_entry_prev
                        else:
                            sap_state.pairs.vertices_entry[j_entry] = i_slot_entry
                            is_shifting = False
                if is_shifting:
                    sap_state.pairs.vertices_entry[i_start] = i_slot_entry
    return has_contact, is_overflow


# TODO: The Jacobian duplicates the one of the contact constraints of the rigid solver (see
# add_collision_constraints in rigid/constraint/solver.py).
@qd.kernel(fastcache=True)
def kernel_compute_jacobian(
    sap_state: SAPState,
    dofs_state: DofsState,
    links_state: LinksState,
    links_info: LinksInfo,
    rigid_config: qd.template(),
):
    """
    Compute the rigid Jacobian of the pairs applying impulses to rigid dofs, at their contact position.

    A pair moves with the chain of each of its links, times the sign of the link.
    """
    for i_s, i_d in qd.ndrange(sap_state.pairs.sort.n_keys[()], sap_state.pairs.Jt.shape[1]):
        i_p = sap_state.pairs.sort.values[i_s]
        if sap_state.pairs.link_idx[i_p][0] >= 0:
            sap_state.pairs.Jt[i_p, i_d] = qd.Vector.zero(gs.qd_float, 3)
    for i_s in range(sap_state.pairs.sort.n_keys[()]):
        i_p = sap_state.pairs.sort.values[i_s]
        i_b = sap_state.pairs.batch_idx[i_p]
        for i_side in qd.static(range(2)):
            i_l = sap_state.pairs.link_idx[i_p][i_side]
            sign = sap_state.pairs.link_sign[i_p][i_side]
            while i_l > -1:
                I_l = [i_l, i_b] if qd.static(rigid_config.batch_links_info) else i_l
                t_pos = sap_state.pairs.contact_pos[i_p] - links_state.root_COM[i_l, i_b]
                # In reverse order, so that the dofs of each row are strictly descending
                for i_d_ in range(links_info.n_dofs[I_l]):
                    i_d = links_info.dof_end[I_l] - 1 - i_d_
                    _, vel = gu.qd_transform_motion_by_trans_quat(
                        dofs_state.cdof_ang[i_d, i_b], dofs_state.cdof_vel[i_d, i_b], t_pos, gu.qd_identity_quat()
                    )
                    sap_state.pairs.Jt[i_p, i_d] = sap_state.pairs.Jt[i_p, i_d] + sign * vel
                i_l = links_info.parent_idx[I_l]


@qd.kernel(fastcache=True)
def kernel_build_equality_constraints(
    sap_state: SAPState,
    joints_info: JointsInfo,
    equalities_info: EqualitiesInfo,
    rigid_config: qd.template(),
    n_equalities: int,
):
    """
    Set up the joint equality constraints of every environment, ordered by environment then by equality.

    A constraint ties the first dof of a joint to the first dof of another.
    """
    # Serial, so that the constraints are ordered by environment then by equality
    qd.loop_config(serialize=True)
    for i_b in range(sap_state.equalities.batch_start.shape[0] - 1):
        n_constraints = 0
        if i_b > 0:
            n_constraints = sap_state.equalities.batch_start[i_b]
        sap_state.equalities.batch_start[i_b] = n_constraints
        for i_eq in range(n_equalities):
            if equalities_info.eq_type[i_eq, i_b] == gs.EQUALITY_TYPE.JOINT:
                i_c = n_constraints
                n_constraints += 1
                i_j1 = equalities_info.eq_obj1id[i_eq, i_b]
                i_j2 = equalities_info.eq_obj2id[i_eq, i_b]
                I_j1 = [i_j1, i_b] if qd.static(rigid_config.batch_joints_info) else i_j1
                I_j2 = [i_j2, i_b] if qd.static(rigid_config.batch_joints_info) else i_j2
                i_d1 = joints_info.dof_start[I_j1]
                i_d2 = joints_info.dof_start[I_j2]
                sap_state.equalities.batch_idx[i_c] = i_b
                sap_state.equalities.dof1_idx[i_c] = i_d1
                sap_state.equalities.dof2_idx[i_c] = i_d2
        sap_state.equalities.batch_start[i_b + 1] = n_constraints


# TODO: The solve duplicates func_solve_mass_block in rigid/abd/forward_dynamics.py, which solves one right-hand side
# per environment instead of the three columns of the rigid Jacobian of every pair.
@qd.func(requires_top_level=True)
def func_rigid_solve_jacobian(sap_state: SAPState, entities_info: EntitiesInfo, rigid_info: RigidInfo):
    """
    Compute M^-1 J^T for the rigid Jacobian of the pairs, by the LDL^T factor of the mass matrix of their environment.
    """
    n_pairs = sap_state.pairs.sort.n_keys[()]
    # Solve w such that L^T @ w = y
    for i_s, i_e, k in qd.ndrange(n_pairs, entities_info.dof_start.shape[0], 3):
        i_p = sap_state.pairs.sort.values[i_s]
        if sap_state.pairs.link_idx[i_p][0] >= 0:
            i_b = sap_state.pairs.batch_idx[i_p]
            entity_dof_end = entities_info.dof_end[i_e]
            for i_d_ in range(entities_info.n_dofs[i_e]):
                i_d = entity_dof_end - i_d_ - 1
                sap_state.pairs.M_inv_Jt[i_p, i_d][k] = sap_state.pairs.Jt[i_p, i_d][k]
                for j_d in range(i_d + 1, entity_dof_end):
                    sap_state.pairs.M_inv_Jt[i_p, i_d][k] -= (
                        rigid_info.mass_mat_L[j_d, i_d, i_b] * sap_state.pairs.M_inv_Jt[i_p, j_d][k]
                    )

    # z = D^-1 w
    for i_s, i_d, k in qd.ndrange(n_pairs, sap_state.pairs.Jt.shape[1], 3):
        i_p = sap_state.pairs.sort.values[i_s]
        if sap_state.pairs.link_idx[i_p][0] >= 0:
            i_b = sap_state.pairs.batch_idx[i_p]
            sap_state.pairs.M_inv_Jt[i_p, i_d][k] *= rigid_info.mass_mat_D_inv[i_d, i_b]

    # Solve x such that L @ x = z
    for i_s, i_e, k in qd.ndrange(n_pairs, entities_info.dof_start.shape[0], 3):
        i_p = sap_state.pairs.sort.values[i_s]
        if sap_state.pairs.link_idx[i_p][0] >= 0:
            i_b = sap_state.pairs.batch_idx[i_p]
            entity_dof_start = entities_info.dof_start[i_e]
            for i_d in range(entity_dof_start, entities_info.dof_end[i_e]):
                for j_d in range(entity_dof_start, i_d):
                    sap_state.pairs.M_inv_Jt[i_p, i_d][k] -= (
                        rigid_info.mass_mat_L[i_d, j_d, i_b] * sap_state.pairs.M_inv_Jt[i_p, j_d][k]
                    )


@qd.kernel(fastcache=True)
def kernel_compute_regularization(
    sap_state: SAPState,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_info: SAPInfo,
    entities_info: EntitiesInfo,
    rigid_info: RigidInfo,
    static_config: qd.template(),
):
    """
    Regularize the pairs and the equality constraints from their Delassus operator, as SAP does.

    The Delassus operator of a pair is J A^-1 J^T, A being the mass matrix of the rigid dofs and the preconditioner of
    the FEM vertices of the substep, in the frame of the pair. The equality constraints are regularized from their
    stiffness alone.
    """
    if qd.static(static_config.has_rigid_contacts):
        func_rigid_solve_jacobian(sap_state, entities_info, rigid_info)

    dt = sap_info.substep_dt[None]
    for i_s in range(sap_state.pairs.sort.n_keys[()]):
        i_p = sap_state.pairs.sort.values[i_s]
        i_b = sap_state.pairs.batch_idx[i_p]
        W = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(static_config.has_rigid_contacts):
            if sap_state.pairs.link_idx[i_p][0] >= 0:
                for i_d in range(sap_state.pairs.Jt.shape[1]):
                    W += sap_state.pairs.M_inv_Jt[i_p, i_d].outer_product(sap_state.pairs.Jt[i_p, i_d])
        if qd.static(static_config.has_fem):
            dt2_inv = 1.0 / dt**2
            fem_slots_vert = sap_state.pairs.fem_slots_vert[i_p]
            fem_slots_weight = sap_state.pairs.fem_slots_weight[i_p]
            for i_slot in qd.static(range(fem_slots_vert.n)):
                i_v = fem_slots_vert[i_slot]
                if i_v >= 0:
                    W += fem_slots_weight[i_slot] ** 2 * dt2_inv * fem_state.newton.prec[i_v, i_b]
        frame = sap_state.pairs.frame[i_p]
        W = frame.transpose() @ W @ frame

        w_rms = W.norm() / 3.0
        beta_factor = sap_info.beta[None] ** 2 / (4.0 * qd.math.pi**2)
        taud = sap_info.taud[None]
        k = sap_state.pairs.k[i_p]
        Rn = max(beta_factor * w_rms, 1.0 / (dt * k * (dt + taud)))
        Rt = sap_info.sigma[None] * w_rms
        sap_state.pairs.Rn[i_p] = Rn
        sap_state.pairs.Rt[i_p] = Rt
        sap_state.pairs.Rn_inv[i_p] = 1.0 / Rn
        sap_state.pairs.Rt_inv[i_p] = 1.0 / Rt
        sap_state.pairs.vn_hat[i_p] = -sap_state.pairs.phi0[i_p] / (dt + taud)
        mu = sap_state.pairs.mu[i_p]
        sap_state.pairs.mu_hat[i_p] = mu * Rt * sap_state.pairs.Rn_inv[i_p]
        sap_state.pairs.mu_factor[i_p] = 1.0 / (1.0 + mu * sap_state.pairs.mu_hat[i_p])

    if qd.static(static_config.has_equalities):
        for i_c in range(sap_state.equalities.batch_start[sap_state.equalities.batch_start.shape[0] - 1]):
            i_b = sap_state.equalities.batch_idx[i_c]
            i_d1 = sap_state.equalities.dof1_idx[i_c]
            i_d2 = sap_state.equalities.dof2_idx[i_c]
            g0 = dofs_state.pos[i_d1, i_b] - dofs_state.pos[i_d2, i_b]
            sap_state.equalities.v_hat[i_c] = -g0 * (1.0 / dt)
            R = 1.0 / (dt**2 * sap_info.equality_stiffness[None])
            sap_state.equalities.R[i_c] = R
            sap_state.equalities.R_inv[i_c] = 1.0 / R


# ------------------------------------------------------------------------------------
# ---------------------------------- Solve funcs -------------------------------------
# ------------------------------------------------------------------------------------

# Every sum over the vertices, the elements or the dofs of an environment runs in fixed order, so that it is bitwise
# reproducible (see FixedOrderSums in linalg.py): the loop updating the items writes their terms, and the loop
# over the environments that follows sums the partial sums in order. The sums over the constraints or the pairs of an
# environment, a segment of a flat array, run one thread per chunk of the segment, which sums its items in order into
# a partial sum (see lu.func_chunk_range).


@qd.func
def func_sum_vertices_dofs(i_b: int, k: int, sap_state: SAPState, static_config: qd.template()):
    """Return the sum k over the FEM vertices and the rigid dofs of an environment."""
    value_sum = gs.qd_float(0.0)
    if qd.static(static_config.has_fem):
        value_sum += lu.func_sum_chunks(i_b, k, sap_state.fem.sums_v)
    if qd.static(static_config.has_rigid):
        value_sum += lu.func_sum_chunks(i_b, k, sap_state.rigid.sums)
    return value_sum


@qd.func
def func_sum_constraints_contacts(
    i_b: int, k: int, sap_state: SAPState, sap_info: SAPInfo, static_config: qd.template()
):
    """Return the sum k over the equality constraints and the pairs of an environment, the handlers in order."""
    value_sum = gs.qd_float(0.0)
    if qd.static(static_config.has_equalities):
        value_sum += lu.func_sum_chunks(i_b, k, sap_state.equalities.sums)
    for i_h in range(sap_state.pairs.handlers_n_pairs.shape[0]):
        handler_sum = gs.qd_float(0.0)
        for i_chunk in range(sap_info.contacts.handlers_n_chunks[i_h]):
            handler_sum += sap_state.pairs.pairs_partials[i_chunk, k, i_h, i_b]
        value_sum += handler_sum
    return value_sum


@qd.func
def func_rigid_contacts_Jt_vec(
    i_d: int, i_b: int, sap_state: SAPState, sap_info: SAPInfo, static_config: qd.template()
):
    """
    Return the transposed Jacobian of the equality constraints and of the pairs times their impulses at a rigid dof.

    The dof sums the constraints of its environment in their order, few enough to go serially, then the partial sums
    over the sorted pairs of each handler in order (see func_sum_rigid_Jt_vec).
    """
    Jt_vec = gs.qd_float(0.0)
    if qd.static(static_config.has_equalities):
        equalities_Jt_vec = gs.qd_float(0.0)
        for i_c in range(sap_state.equalities.batch_start[i_b], sap_state.equalities.batch_start[i_b + 1]):
            if i_d == sap_state.equalities.dof1_idx[i_c]:
                equalities_Jt_vec += sap_state.equalities.vec[i_c]
            elif i_d == sap_state.equalities.dof2_idx[i_c]:
                equalities_Jt_vec -= sap_state.equalities.vec[i_c]
        Jt_vec += equalities_Jt_vec
    if qd.static(static_config.has_rigid_contacts):
        for i_h in range(sap_state.pairs.handlers_n_pairs.shape[0]):
            i_r = sap_info.contacts.handlers_rigid_rank[i_h]
            if i_r >= 0:
                handler_Jt_vec = gs.qd_float(0.0)
                for i_chunk in range(sap_info.contacts.handlers_n_chunks[i_h]):
                    handler_Jt_vec += sap_state.pairs.dofs_partials[i_chunk, i_d, i_r, i_b]
                Jt_vec += handler_Jt_vec
    return Jt_vec


@qd.func(requires_top_level=True)
def func_sum_rigid_Jt_vec(sap_state: SAPState, sap_info: SAPInfo):
    """
    Compute the partial sums of the transposed rigid Jacobian times the impulses of the pairs over chunks of the sorted
    pairs of each handler, per dof.

    One thread per chunk and dof sums the pairs of the chunk in order (see func_rigid_contacts_Jt_vec).
    """
    n_handlers = sap_state.pairs.handlers_n_pairs.shape[0]
    for i_chunk, i_d, i_h, i_b in qd.ndrange(
        sap_state.pairs.dofs_partials.shape[0],
        sap_state.pairs.dofs_partials.shape[1],
        n_handlers,
        sap_state.pairs.dofs_partials.shape[3],
    ):
        i_r = sap_info.contacts.handlers_rigid_rank[i_h]
        n_chunks = sap_info.contacts.handlers_n_chunks[i_h]
        if i_r >= 0 and i_chunk < n_chunks:
            Jt_vec = gs.qd_float(0.0)
            i_seg = i_b * n_handlers + i_h
            i_s_start = sap_state.pairs.sort.segments_start[i_seg]
            n_segment_pairs = sap_state.pairs.sort.segments_start[i_seg + 1] - i_s_start
            i_s_start, i_s_end = lu.func_chunk_range(i_s_start, i_chunk, n_segment_pairs, n_chunks)
            for i_s in range(i_s_start, i_s_end):
                i_p = sap_state.pairs.sort.values[i_s]
                Jt_vec += sap_state.pairs.Jt[i_p, i_d].dot(sap_state.pairs.vec[i_p])
            sap_state.pairs.dofs_partials[i_chunk, i_d, i_r, i_b] = Jt_vec


@qd.func
def func_compute_elastic_products(
    i_e: int,
    i_b: int,
    S: qd.types.matrix(4, 3),
    src: qd.template(),
    fem_state: FEMState,
    fem_info: FEMInfo,
):
    """Return S^T x of an element, x being the values of a field at its vertices, and its Hessian times it."""
    el2v = fem_info.elements.el2v[i_e]
    p9 = qd.Vector.zero(gs.qd_float, 9)
    for i, j in qd.static(qd.ndrange(3, 4)):
        p9[i * 3 : i * 3 + 3] = p9[i * 3 : i * 3 + 3] + (S[j, i] * src[el2v[j], i_b])

    H9_p9 = qd.Vector.zero(gs.qd_float, 9)
    for i, j in qd.static(qd.ndrange(3, 3)):
        H9_p9[i * 3 : i * 3 + 3] = H9_p9[i * 3 : i * 3 + 3] + (
            fem_state.newton.hessian[i, j, i_e, i_b] @ p9[j * 3 : j * 3 + 3]
        )
    return p9, H9_p9


@qd.func(requires_top_level=True)
def func_compute_fem_matrix_vector_product(
    src: qd.template(),
    dst: qd.template(),
    active: qd.template(),
    vertex_constraints: FEMVertexConstraintsInfo,
    sap_state: SAPState,
    fem_state: FEMState,
    fem_info: FEMInfo,
    static_config: qd.template(),
    add_contacts: qd.template(),
    sum_dot: qd.template(),
):
    """
    Compute the product of the FEM matrix with a field of the vertices, the mass matrix plus the elastic stiffness.

    With `add_contacts`, the product includes the contact Hessian, the pairs holding the contact part of the product in
    their impulses. With `sum_dot`, the partial sums of src.dot(dst) over the vertices go to slot 0 of the sums of the
    vertices.
    """
    # Elasticity, per corner of the elements
    for i_e, i_b in qd.ndrange(sap_state.fem.corners.shape[0], sap_state.fem.corners.shape[2]):
        if active[i_b]:
            dt = fem_info.substep_dt[None]
            S = func_compute_element_mapping_matrix(i_e, i_b, vertex_constraints, fem_info, static_config)
            _, new_p9 = func_compute_elastic_products(i_e, i_b, S, src, fem_state, fem_info)
            scale = fem_info.elements.V[i_e] * dt**2 * (fem_info.damping_beta[None] / dt + 1.0)
            for i in qd.static(range(4)):
                sap_state.fem.corners[i_e, i, i_b] = (
                    S[i, 0] * new_p9[0:3] + S[i, 1] * new_p9[3:6] + S[i, 2] * new_p9[6:9]
                ) * scale

    # Inertia, and the elastic contributions of the corners touching the vertex in their order
    for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
        if active[i_b]:
            dt = fem_info.substep_dt[None]
            damping_alpha_factor = fem_info.damping_alpha[None] * dt + 1.0
            dst_v = fem_info.vertices.mass_over_dt2[i_v] * src[i_v, i_b] * dt**2 * damping_alpha_factor
            for i_entry in range(fem_info.vertices.corner_start[i_v], fem_info.vertices.corner_start[i_v + 1]):
                i_corner = fem_info.vertices.corner[i_entry]
                dst_v += sap_state.fem.corners[i_corner // 4, i_corner % 4, i_b]
            if qd.static(add_contacts):
                Jt_vec, _ = func_fem_contacts_Jt(
                    i_v, i_b, vertex_constraints, sap_state, static_config, compute_hessian=False
                )
                dst_v += Jt_vec
            dst[i_v, i_b] = dst_v
            if qd.static(sum_dot):
                sap_state.fem.sums_v.terms[i_v, 0, i_b] = src[i_v, i_b].dot(dst_v)
    if qd.static(sum_dot):
        lu.func_reduce_terms(sap_state.fem.sums_v, active)


@qd.func(requires_top_level=True)
def func_compute_rigid_mass_mat_vec_product(
    vec: qd.template(),
    out: qd.template(),
    active: qd.template(),
    sap_state: SAPState,
    sap_info: SAPInfo,
    rigid_info: RigidInfo,
    static_config: qd.template(),
    add_contacts: qd.template(),
    sum_dot: qd.template(),
):
    """
    Compute the product of the rigid mass matrix with a field of the dofs.

    With `add_contacts`, the product includes the Hessian of the equality constraints and of the pairs, which hold the
    contact part of the product in their impulses. With `sum_dot`, the partial sums of vec.dot(out) over the dofs go to
    slot 0 of the sums of the dofs.
    """
    for i_d1, i_b in qd.ndrange(vec.shape[0], vec.shape[1]):
        out_d1 = gs.qd_float(0.0)
        if active[i_b]:
            for i_d0 in range(vec.shape[0]):
                out_d1 += rigid_info.mass_mat[i_d1, i_d0, i_b] * vec[i_d0, i_b]
            if qd.static(add_contacts):
                out_d1 += func_rigid_contacts_Jt_vec(i_d1, i_b, sap_state, sap_info, static_config)
        out[i_d1, i_b] = out_d1
        if qd.static(sum_dot):
            sap_state.rigid.sums.terms[i_d1, 0, i_b] = vec[i_d1, i_b] * out_d1
    if qd.static(sum_dot):
        lu.func_reduce_terms(sap_state.rigid.sums, active)


@qd.func(requires_top_level=True)
def func_compute_constraints_contacts_line_terms(
    sap_state: SAPState,
    sap_info: SAPInfo,
    static_config: qd.template(),
    compute_energy: qd.template(),
):
    """
    Compute the partial sums over the equality constraints and over the pairs of each environment along the search
    direction, chunk by chunk.

    With `compute_energy`, slot 0 holds the energy. Otherwise slots 0 and 1 hold the first and the second derivative of
    the energy along the search direction, the impulses and the Hessians being updated on the way.
    """
    if qd.static(static_config.has_equalities):
        n_chunks = sap_state.equalities.sums.partials.shape[0]
        for i_chunk, i_b in qd.ndrange(n_chunks, sap_state.equalities.sums.partials.shape[2]):
            if sap_state.solver.is_linesearch_active[i_b]:
                term0 = gs.qd_float(0.0)
                term1 = gs.qd_float(0.0)
                i_c_start = sap_state.equalities.batch_start[i_b]
                n_constraints = sap_state.equalities.batch_start[i_b + 1] - i_c_start
                i_c_start, i_c_end = lu.func_chunk_range(i_c_start, i_chunk, n_constraints, n_chunks)
                for i_c in range(i_c_start, i_c_end):
                    i_d1 = sap_state.equalities.dof1_idx[i_c]
                    i_d2 = sap_state.equalities.dof2_idx[i_c]
                    vc = sap_state.rigid.v[i_d1, i_b] - sap_state.rigid.v[i_d2, i_b]
                    y = (sap_state.equalities.v_hat[i_c] - vc) * sap_state.equalities.R_inv[i_c]
                    if qd.static(compute_energy):
                        term0 += 0.5 * y**2 * sap_state.equalities.R[i_c]
                    else:
                        dvc = sap_state.equalities.dvc[i_c]
                        term0 -= dvc * y
                        term1 += dvc**2 * sap_state.equalities.R_inv[i_c]
                sap_state.equalities.sums.partials[i_chunk, 0, i_b] = term0
                if qd.static(not compute_energy):
                    sap_state.equalities.sums.partials[i_chunk, 1, i_b] = term1

    n_handlers = sap_state.pairs.handlers_n_pairs.shape[0]
    for i_chunk, i_h, i_b in qd.ndrange(
        sap_state.pairs.pairs_partials.shape[0], n_handlers, sap_state.pairs.pairs_partials.shape[3]
    ):
        n_chunks = sap_info.contacts.handlers_n_chunks[i_h]
        if sap_state.solver.is_linesearch_active[i_b] and i_chunk < n_chunks:
            term0 = gs.qd_float(0.0)
            term1 = gs.qd_float(0.0)
            i_seg = i_b * n_handlers + i_h
            i_s_start = sap_state.pairs.sort.segments_start[i_seg]
            n_segment_pairs = sap_state.pairs.sort.segments_start[i_seg + 1] - i_s_start
            i_s_start, i_s_end = lu.func_chunk_range(i_s_start, i_chunk, n_segment_pairs, n_chunks)
            for i_s in range(i_s_start, i_s_end):
                i_p = sap_state.pairs.sort.values[i_s]
                vc = func_contact_velocity(i_p, sap_state.fem.v, sap_state.rigid.v, sap_state, static_config)
                if qd.static(compute_energy):
                    term0 += func_contact_gamma(i_p, vc, sap_state, sap_info, compute_hessian=False)
                else:
                    func_contact_gamma(i_p, vc, sap_state, sap_info, compute_hessian=True)
                    dvc = sap_state.pairs.dvc[i_p]
                    term0 -= dvc.dot(sap_state.pairs.gamma[i_p])
                    term1 += dvc.dot(sap_state.pairs.G[i_p] @ dvc)
            sap_state.pairs.pairs_partials[i_chunk, 0, i_h, i_b] = term0
            if qd.static(not compute_energy):
                sap_state.pairs.pairs_partials[i_chunk, 1, i_h, i_b] = term1


# ------------------------------------------------------------------------------------
# --------------------------------- Solve kernels ------------------------------------
# ------------------------------------------------------------------------------------


@qd.kernel(fastcache=True)
def kernel_init_sap_solve(
    f: int,
    sap_state: SAPState,
    fem_state: FEMState,
    dofs_state: DofsState,
    static_config: qd.template(),
):
    """Start the SAP solve of every environment from the free velocities of the substep."""
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            sap_state.fem.v[i_v, i_b] = fem_state.vertices.vel[f + 1, i_v, i_b]
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            sap_state.rigid.v[i_d, i_b] = dofs_state.vel[i_d, i_b]
    for i_b in range(sap_state.solver.is_active.shape[0]):
        sap_state.solver.is_active[i_b] = True


@qd.kernel(fastcache=True)
def kernel_init_unconstrained_gradient_diag(
    f: int,
    sap_state: SAPState,
    fem_state: FEMState,
    dofs_state: DofsState,
    fem_info: FEMInfo,
    static_config: qd.template(),
):
    """Reset the gradient, and set the diagonal blocks of the free motion Hessian and the change from the free
    velocities."""
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            sap_state.fem.gradient[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
            # The Hessian of the FEM solver is in positions, which turns into velocities times dt^2
            sap_state.fem.diag3x3[i_v, i_b] = fem_state.newton.diag3x3[i_v, i_b] * fem_info.substep_dt[None] ** 2
            sap_state.fem.v_diff[i_v, i_b] = sap_state.fem.v[i_v, i_b] - fem_state.vertices.vel[f + 1, i_v, i_b]
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            sap_state.rigid.gradient[i_d, i_b] = 0.0
            sap_state.rigid.v_diff[i_d, i_b] = sap_state.rigid.v[i_d, i_b] - dofs_state.vel[i_d, i_b]


@qd.kernel(fastcache=True)
def kernel_compute_unconstrained_gradient(
    vertex_constraints: FEMVertexConstraintsInfo,
    sap_state: SAPState,
    fem_state: FEMState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
    rigid_info: RigidInfo,
    static_config: qd.template(),
):
    """Compute the free motion part of the gradient, A (v - v*), A being the mass matrix of the free motion."""
    if qd.static(static_config.has_fem):
        func_compute_fem_matrix_vector_product(
            sap_state.fem.v_diff,
            sap_state.fem.gradient,
            sap_state.solver.is_active,
            vertex_constraints,
            sap_state,
            fem_state,
            fem_info,
            static_config,
            add_contacts=False,
            sum_dot=False,
        )
    if qd.static(static_config.has_rigid):
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid.v_diff,
            sap_state.rigid.gradient,
            sap_state.solver.is_active,
            sap_state,
            sap_info,
            rigid_info,
            static_config,
            add_contacts=False,
            sum_dot=False,
        )


@qd.kernel(fastcache=True)
def kernel_compute_constraint_contact_gradient_hessian_diag_prec(
    vertex_constraints: FEMVertexConstraintsInfo,
    sap_state: SAPState,
    sap_info: SAPInfo,
    static_config: qd.template(),
):
    """
    Compute the impulses of the equality constraints and of the pairs, and add them to the gradient.

    Then compute the diagonal blocks of the contact Hessian of the FEM vertices and their inverse, which preconditions
    the conjugate gradient.
    """
    if qd.static(static_config.has_equalities):
        for i_c in range(sap_state.equalities.batch_start[sap_state.equalities.batch_start.shape[0] - 1]):
            i_b = sap_state.equalities.batch_idx[i_c]
            i_d1 = sap_state.equalities.dof1_idx[i_c]
            i_d2 = sap_state.equalities.dof2_idx[i_c]
            vc = sap_state.rigid.v[i_d1, i_b] - sap_state.rigid.v[i_d2, i_b]
            sap_state.equalities.vec[i_c] = (sap_state.equalities.v_hat[i_c] - vc) * sap_state.equalities.R_inv[i_c]
    # The impulse and the Hessian of each pair, in the world frame
    for i_s in range(sap_state.pairs.sort.n_keys[()]):
        i_p = sap_state.pairs.sort.values[i_s]
        vc = func_contact_velocity(i_p, sap_state.fem.v, sap_state.rigid.v, sap_state, static_config)
        func_contact_gamma(i_p, vc, sap_state, sap_info, compute_hessian=True)
        frame = sap_state.pairs.frame[i_p]
        sap_state.pairs.vec[i_p] = frame @ sap_state.pairs.gamma[i_p]
        sap_state.pairs.mat[i_p] = frame @ sap_state.pairs.G[i_p] @ frame.transpose()
    if qd.static(static_config.has_rigid_contacts):
        func_sum_rigid_Jt_vec(sap_state, sap_info)
    if qd.static(static_config.has_fem):
        # The impulses, the gradient and the preconditioner of the vertices, gathering the pairs
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_active[i_b]:
                Jt_gamma, Jt_A_J = func_fem_contacts_Jt(
                    i_v, i_b, vertex_constraints, sap_state, static_config, compute_hessian=True
                )
                sap_state.fem.impulse[i_v, i_b] = Jt_gamma
                sap_state.fem.gradient[i_v, i_b] -= Jt_gamma
                diag3x3 = sap_state.fem.diag3x3[i_v, i_b] + Jt_A_J
                sap_state.fem.diag3x3[i_v, i_b] = diag3x3
                sap_state.fem.prec[i_v, i_b] = diag3x3.inverse()
    if qd.static(static_config.has_rigid):
        # The impulses and the gradient of the dofs, gathering the equality constraints and the pairs
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_active[i_b]:
                Jt_gamma = func_rigid_contacts_Jt_vec(i_d, i_b, sap_state, sap_info, static_config)
                sap_state.rigid.impulse[i_d, i_b] = Jt_gamma
                sap_state.rigid.gradient[i_d, i_b] -= Jt_gamma


@qd.kernel(fastcache=True)
def kernel_check_sap_convergence(
    sap_state: SAPState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
    rigid_info: RigidInfo,
    static_config: qd.template(),
):
    """Stop the SAP solve of the environments whose gradient is small enough relative to their momentum or impulse."""
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_active[i_b]:
                mass = fem_info.vertices.mass[i_v]
                sap_state.fem.sums_v.terms[i_v, 0, i_b] = sap_state.fem.gradient[i_v, i_b].norm_sqr() / mass
                sap_state.fem.sums_v.terms[i_v, 1, i_b] = sap_state.fem.v[i_v, i_b].norm_sqr() * mass
                sap_state.fem.sums_v.terms[i_v, 2, i_b] = sap_state.fem.impulse[i_v, i_b].norm_sqr() / mass
        lu.func_reduce_terms(sap_state.fem.sums_v, sap_state.solver.is_active)
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_active[i_b]:
                mass = rigid_info.mass_mat[i_d, i_d, i_b]
                sap_state.rigid.sums.terms[i_d, 0, i_b] = sap_state.rigid.gradient[i_d, i_b] ** 2 / mass
                sap_state.rigid.sums.terms[i_d, 1, i_b] = sap_state.rigid.v[i_d, i_b] ** 2 * mass
                sap_state.rigid.sums.terms[i_d, 2, i_b] = sap_state.rigid.impulse[i_d, i_b] ** 2 / mass
        lu.func_reduce_terms(sap_state.rigid.sums, sap_state.solver.is_active)
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_active[i_b]:
            gradient_norm = func_sum_vertices_dofs(i_b, 0, sap_state, static_config)
            momentum_norm = func_sum_vertices_dofs(i_b, 1, sap_state, static_config)
            impulse_norm = func_sum_vertices_dofs(i_b, 2, sap_state, static_config)
            norm_threshold = sap_info.convergence_atol[None] + sap_info.convergence_rtol[None] * qd.max(
                momentum_norm, impulse_norm
            )
            sap_state.solver.is_active[i_b] = gradient_norm >= norm_threshold


@qd.kernel(fastcache=True)
def kernel_init_pcg_solve(
    sap_state: SAPState,
    dyn_state: DynState,
    sap_info: SAPInfo,
    rigid_info: RigidInfo,
    static_config: qd.template(),
    rigid_config: qd.template(),
):
    """Start the conjugate gradient solve of the Newton step from zero, with minus the gradient as residual."""
    for i_b in range(sap_state.solver.is_active.shape[0]):
        sap_state.solver.is_pcg_active[i_b] = sap_state.solver.is_active[i_b]
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                sap_state.fem.x[i_v, i_b] = qd.Vector.zero(gs.qd_float, 3)
                r = -sap_state.fem.gradient[i_v, i_b]
                z = sap_state.fem.prec[i_v, i_b] @ r
                sap_state.fem.r[i_v, i_b] = r
                sap_state.fem.z[i_v, i_b] = z
                sap_state.fem.p[i_v, i_b] = z
                sap_state.fem.sums_v.terms[i_v, 0, i_b] = r.dot(r)
                sap_state.fem.sums_v.terms[i_v, 1, i_b] = r.dot(z)
        lu.func_reduce_terms(sap_state.fem.sums_v, sap_state.solver.is_pcg_active)
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                sap_state.rigid.x[i_d, i_b] = 0.0
                sap_state.rigid.r[i_d, i_b] = -sap_state.rigid.gradient[i_d, i_b]
                sap_state.rigid.sums.terms[i_d, 0, i_b] = sap_state.rigid.r[i_d, i_b] ** 2
        for i_b in range(sap_state.rigid.r.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                func_solve_mass_batch(i_b, sap_state.rigid.r, sap_state.rigid.z, dyn_state, rigid_info, rigid_config)
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                sap_state.rigid.p[i_d, i_b] = sap_state.rigid.z[i_d, i_b]
                sap_state.rigid.sums.terms[i_d, 1, i_b] = sap_state.rigid.r[i_d, i_b] * sap_state.rigid.z[i_d, i_b]
        lu.func_reduce_terms(sap_state.rigid.sums, sap_state.solver.is_pcg_active)
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_pcg_active[i_b]:
            rTr = func_sum_vertices_dofs(i_b, 0, sap_state, static_config)
            sap_state.solver.rTz[i_b] = func_sum_vertices_dofs(i_b, 1, sap_state, static_config)
            sap_state.solver.is_pcg_active[i_b] = rTr > sap_info.pcg_threshold[None]


@qd.kernel(fastcache=True)
def kernel_one_pcg_iter(
    vertex_constraints: FEMVertexConstraintsInfo,
    sap_state: SAPState,
    fem_state: FEMState,
    dyn_state: DynState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
    rigid_info: RigidInfo,
    static_config: qd.template(),
    rigid_config: qd.template(),
):
    """
    Run one iteration of the preconditioned conjugate gradient solve of the Newton step.

    See https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    """
    # A @ p, the equality constraints and the pairs holding their part J^T G J p in their impulses, and p^T A p
    if qd.static(static_config.has_equalities):
        for i_c in range(sap_state.equalities.batch_start[sap_state.equalities.batch_start.shape[0] - 1]):
            i_b = sap_state.equalities.batch_idx[i_c]
            i_d1 = sap_state.equalities.dof1_idx[i_c]
            i_d2 = sap_state.equalities.dof2_idx[i_c]
            vc = sap_state.rigid.p[i_d1, i_b] - sap_state.rigid.p[i_d2, i_b]
            sap_state.equalities.vec[i_c] = sap_state.equalities.R_inv[i_c] * vc
    for i_s in range(sap_state.pairs.sort.n_keys[()]):
        i_p = sap_state.pairs.sort.values[i_s]
        x = func_contact_velocity(i_p, sap_state.fem.p, sap_state.rigid.p, sap_state, static_config)
        sap_state.pairs.vec[i_p] = sap_state.pairs.frame[i_p] @ (sap_state.pairs.G[i_p] @ x)
    if qd.static(static_config.has_rigid_contacts):
        func_sum_rigid_Jt_vec(sap_state, sap_info)
    if qd.static(static_config.has_fem):
        func_compute_fem_matrix_vector_product(
            sap_state.fem.p,
            sap_state.fem.Ap,
            sap_state.solver.is_pcg_active,
            vertex_constraints,
            sap_state,
            fem_state,
            fem_info,
            static_config,
            add_contacts=True,
            sum_dot=True,
        )
    if qd.static(static_config.has_rigid):
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid.p,
            sap_state.rigid.Ap,
            sap_state.solver.is_pcg_active,
            sap_state,
            sap_info,
            rigid_info,
            static_config,
            add_contacts=True,
            sum_dot=True,
        )

    # The step
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_pcg_active[i_b]:
            pTAp = func_sum_vertices_dofs(i_b, 0, sap_state, static_config)
            sap_state.solver.alpha[i_b] = sap_state.solver.rTz[i_b] / pTAp

    # x, r, z, and the terms of r^T r and r^T z
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                alpha = sap_state.solver.alpha[i_b]
                sap_state.fem.x[i_v, i_b] = sap_state.fem.x[i_v, i_b] + alpha * sap_state.fem.p[i_v, i_b]
                r = sap_state.fem.r[i_v, i_b] - alpha * sap_state.fem.Ap[i_v, i_b]
                z = sap_state.fem.prec[i_v, i_b] @ r
                sap_state.fem.r[i_v, i_b] = r
                sap_state.fem.z[i_v, i_b] = z
                sap_state.fem.sums_v.terms[i_v, 1, i_b] = r.norm_sqr()
                sap_state.fem.sums_v.terms[i_v, 2, i_b] = r.dot(z)
        lu.func_reduce_terms(sap_state.fem.sums_v, sap_state.solver.is_pcg_active)
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                alpha = sap_state.solver.alpha[i_b]
                sap_state.rigid.x[i_d, i_b] = sap_state.rigid.x[i_d, i_b] + alpha * sap_state.rigid.p[i_d, i_b]
                sap_state.rigid.r[i_d, i_b] = sap_state.rigid.r[i_d, i_b] - alpha * sap_state.rigid.Ap[i_d, i_b]
                sap_state.rigid.sums.terms[i_d, 1, i_b] = sap_state.rigid.r[i_d, i_b] ** 2
        for i_b in range(sap_state.rigid.r.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                func_solve_mass_batch(i_b, sap_state.rigid.r, sap_state.rigid.z, dyn_state, rigid_info, rigid_config)
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                sap_state.rigid.sums.terms[i_d, 2, i_b] = sap_state.rigid.r[i_d, i_b] * sap_state.rigid.z[i_d, i_b]
        lu.func_reduce_terms(sap_state.rigid.sums, sap_state.solver.is_pcg_active)

    # Convergence, then beta, r^T r and r^T z
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_pcg_active[i_b]:
            rTr = func_sum_vertices_dofs(i_b, 1, sap_state, static_config)
            rTz = func_sum_vertices_dofs(i_b, 2, sap_state, static_config)
            sap_state.solver.is_pcg_active[i_b] = rTr > sap_info.pcg_threshold[None]
            if sap_state.solver.is_pcg_active[i_b]:
                sap_state.solver.beta[i_b] = rTz / sap_state.solver.rTz[i_b]
                sap_state.solver.rTz[i_b] = rTz

    # The search direction
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                sap_state.fem.p[i_v, i_b] = (
                    sap_state.fem.z[i_v, i_b] + sap_state.solver.beta[i_b] * sap_state.fem.p[i_v, i_b]
                )
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_pcg_active[i_b]:
                sap_state.rigid.p[i_d, i_b] = (
                    sap_state.rigid.z[i_d, i_b] + sap_state.solver.beta[i_b] * sap_state.rigid.p[i_d, i_b]
                )


@qd.func(requires_top_level=True)
def func_compute_line_gradient_hessian(
    f: int,
    sap_state: SAPState,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_info: SAPInfo,
    static_config: qd.template(),
):
    """
    Move the velocities along the search direction, and compute the partial sums of the line energy derivatives.

    The partial sums are those of the first and the second derivative of the energy along the search direction (see
    func_sum_line_gradient_hessian).
    """
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_linesearch_active[i_b]:
                alpha = sap_state.solver.step_size[i_b]
                v = sap_state.fem.x_prev[i_v, i_b] + alpha * sap_state.fem.x[i_v, i_b]
                sap_state.fem.v[i_v, i_b] = v
                dell = sap_state.fem.dp[i_v, i_b].dot(v - fem_state.vertices.vel[f + 1, i_v, i_b])
                sap_state.fem.sums_v.terms[i_v, 0, i_b] = dell
        lu.func_reduce_terms(sap_state.fem.sums_v, sap_state.solver.is_linesearch_active)
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_linesearch_active[i_b]:
                alpha = sap_state.solver.step_size[i_b]
                v = sap_state.rigid.x_prev[i_d, i_b] + alpha * sap_state.rigid.x[i_d, i_b]
                sap_state.rigid.v[i_d, i_b] = v
                dell = sap_state.rigid.dp[i_d, i_b] * (v - dofs_state.vel[i_d, i_b])
                sap_state.rigid.sums.terms[i_d, 0, i_b] = dell
        lu.func_reduce_terms(sap_state.rigid.sums, sap_state.solver.is_linesearch_active)
    func_compute_constraints_contacts_line_terms(sap_state, sap_info, static_config, compute_energy=False)


@qd.func
def func_sum_line_gradient_hessian(i_b: int, sap_state: SAPState, sap_info: SAPInfo, static_config: qd.template()):
    """Sum the first and the second derivative of the energy along the search direction of an environment."""
    d2ellA_dalpha2 = sap_state.solver.d2ellA_dalpha2[i_b]
    dellA_dalpha = func_sum_vertices_dofs(i_b, 0, sap_state, static_config)
    dell_contacts_dalpha = func_sum_constraints_contacts(i_b, 0, sap_state, sap_info, static_config)
    sap_state.solver.dell_dalpha[i_b] = dellA_dalpha + dell_contacts_dalpha
    sap_state.solver.d2ell_dalpha2[i_b] = d2ellA_dalpha2 + func_sum_constraints_contacts(
        i_b, 1, sap_state, sap_info, static_config
    )


@qd.kernel(fastcache=True)
def kernel_init_exact_linesearch(
    f: int,
    vertex_constraints: FEMVertexConstraintsInfo,
    sap_state: SAPState,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_info: SAPInfo,
    fem_info: FEMInfo,
    rigid_info: RigidInfo,
    static_config: qd.template(),
):
    """
    Start the exact line search of the Newton step, by the rtsafe method as in Drake.

    It records the energy at the current velocities and the slope along the search direction, takes the full step when
    the slope is small enough, and brackets the root of the derivative of the energy otherwise.
    """
    # The slope m along the search direction, and the velocities the search starts from
    for i_b in range(sap_state.solver.is_active.shape[0]):
        sap_state.solver.is_linesearch_active[i_b] = sap_state.solver.is_active[i_b]
        if sap_state.solver.is_linesearch_active[i_b]:
            sap_state.solver.step_size[i_b] = sap_info.linesearch_max_step_size[None]
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_linesearch_active[i_b]:
                sap_state.fem.sums_v.terms[i_v, 0, i_b] = sap_state.fem.x[i_v, i_b].dot(
                    sap_state.fem.gradient[i_v, i_b]
                )
                sap_state.fem.x_prev[i_v, i_b] = sap_state.fem.v[i_v, i_b]
        lu.func_reduce_terms(sap_state.fem.sums_v, sap_state.solver.is_linesearch_active)
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_linesearch_active[i_b]:
                sap_state.rigid.sums.terms[i_d, 0, i_b] = (
                    sap_state.rigid.x[i_d, i_b] * sap_state.rigid.gradient[i_d, i_b]
                )
                sap_state.rigid.x_prev[i_d, i_b] = sap_state.rigid.v[i_d, i_b]
        lu.func_reduce_terms(sap_state.rigid.sums, sap_state.solver.is_linesearch_active)
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_linesearch_active[i_b]:
            sap_state.solver.m[i_b] = func_sum_vertices_dofs(i_b, 0, sap_state, static_config)

    # The energy at the current velocities: inertia and elasticity, kinetic energy of the dofs, constraints and pairs
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_linesearch_active[i_b]:
                dt = fem_info.substep_dt[None]
                v_diff = sap_state.fem.v[i_v, i_b] - fem_state.vertices.vel[f + 1, i_v, i_b]
                sap_state.fem.v_diff[i_v, i_b] = v_diff
                sap_state.fem.sums_v.terms[i_v, 0, i_b] = (
                    0.5
                    * fem_info.vertices.mass_over_dt2[i_v]
                    * v_diff.norm_sqr()
                    * dt**2
                    * (fem_info.damping_alpha[None] * dt + 1.0)
                )
        lu.func_reduce_terms(sap_state.fem.sums_v, sap_state.solver.is_linesearch_active)
        for i_e, i_b in qd.ndrange(sap_state.fem.corners.shape[0], sap_state.fem.corners.shape[2]):
            if sap_state.solver.is_linesearch_active[i_b]:
                dt = fem_info.substep_dt[None]
                S = func_compute_element_mapping_matrix(i_e, i_b, vertex_constraints, fem_info, static_config)
                p9, H9_p9 = func_compute_elastic_products(i_e, i_b, S, sap_state.fem.v_diff, fem_state, fem_info)
                sap_state.fem.sums_el.terms[i_e, 0, i_b] = (
                    0.5 * p9.dot(H9_p9) * (fem_info.damping_beta[None] / dt + 1.0) * fem_info.elements.V[i_e] * dt**2
                )
        lu.func_reduce_terms(sap_state.fem.sums_el, sap_state.solver.is_linesearch_active)
    if qd.static(static_config.has_rigid):
        # Kinetic energy, twice its value going to the partial sums
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_linesearch_active[i_b]:
                sap_state.rigid.v_diff[i_d, i_b] = sap_state.rigid.v[i_d, i_b] - dofs_state.vel[i_d, i_b]
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid.v_diff,
            sap_state.rigid.mass_v_diff,
            sap_state.solver.is_linesearch_active,
            sap_state,
            sap_info,
            rigid_info,
            static_config,
            add_contacts=False,
            sum_dot=True,
        )
    func_compute_constraints_contacts_line_terms(sap_state, sap_info, static_config, compute_energy=True)
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_linesearch_active[i_b]:
            energy = gs.qd_float(0.0)
            if qd.static(static_config.has_fem):
                energy += lu.func_sum_chunks(i_b, k=0, sums=sap_state.fem.sums_v)
                energy += lu.func_sum_chunks(i_b, k=0, sums=sap_state.fem.sums_el)
            if qd.static(static_config.has_rigid):
                energy += 0.5 * lu.func_sum_chunks(i_b, k=0, sums=sap_state.rigid.sums)
            sap_state.solver.energy_prev[i_b] = energy + func_sum_constraints_contacts(
                i_b, 0, sap_state, sap_info, static_config
            )

    # A @ x along the search direction, and the change of the velocity of the constraints and of the pairs
    if qd.static(static_config.has_fem):
        func_compute_fem_matrix_vector_product(
            sap_state.fem.x,
            sap_state.fem.dp,
            sap_state.solver.is_linesearch_active,
            vertex_constraints,
            sap_state,
            fem_state,
            fem_info,
            static_config,
            add_contacts=False,
            sum_dot=True,
        )
    if qd.static(static_config.has_rigid):
        func_compute_rigid_mass_mat_vec_product(
            sap_state.rigid.x,
            sap_state.rigid.dp,
            sap_state.solver.is_linesearch_active,
            sap_state,
            sap_info,
            rigid_info,
            static_config,
            add_contacts=False,
            sum_dot=True,
        )
    if qd.static(static_config.has_equalities):
        for i_c in range(sap_state.equalities.batch_start[sap_state.equalities.batch_start.shape[0] - 1]):
            i_b = sap_state.equalities.batch_idx[i_c]
            if sap_state.solver.is_linesearch_active[i_b]:
                i_d1 = sap_state.equalities.dof1_idx[i_c]
                i_d2 = sap_state.equalities.dof2_idx[i_c]
                sap_state.equalities.dvc[i_c] = sap_state.rigid.x[i_d1, i_b] - sap_state.rigid.x[i_d2, i_b]
    for i_s in range(sap_state.pairs.sort.n_keys[()]):
        i_p = sap_state.pairs.sort.values[i_s]
        i_b = sap_state.pairs.batch_idx[i_p]
        if sap_state.solver.is_linesearch_active[i_b]:
            sap_state.pairs.dvc[i_p] = func_contact_velocity(
                i_p, sap_state.fem.x, sap_state.rigid.x, sap_state, static_config
            )
    # The second derivative of the dynamic energy along the search direction
    for i_b in range(sap_state.solver.is_active.shape[0]):
        d2ellA_dalpha2 = gs.qd_float(0.0)
        if sap_state.solver.is_linesearch_active[i_b]:
            d2ellA_dalpha2 = func_sum_vertices_dofs(i_b, 0, sap_state, static_config)
        sap_state.solver.d2ellA_dalpha2[i_b] = d2ellA_dalpha2

    func_compute_line_gradient_hessian(f, sap_state, fem_state, dofs_state, sap_info, static_config)
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_linesearch_active[i_b]:
            func_sum_line_gradient_hessian(i_b, sap_state, sap_info, static_config)

    # The full step when the slope is already small enough, a rare case in the exact line search of Drake as well
    for i_b in range(sap_state.solver.is_active.shape[0]):
        is_full_step = False
        if sap_state.solver.is_linesearch_active[i_b]:
            sap_state.solver.is_linesearch_active[i_b] = sap_state.solver.dell_dalpha[i_b] > 0.0
        if sap_state.solver.is_linesearch_active[i_b]:
            err_threshold = (
                sap_info.convergence_atol[None] + sap_info.convergence_rtol[None] * sap_state.solver.energy_prev[i_b]
            )
            is_full_step = -sap_state.solver.m[i_b] < err_threshold
            if is_full_step:
                sap_state.solver.is_linesearch_active[i_b] = False
                sap_state.solver.step_size[i_b] = 1.0
        sap_state.solver.is_full_step[i_b] = is_full_step
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            if sap_state.solver.is_full_step[i_b]:
                sap_state.fem.v[i_v, i_b] = sap_state.fem.x_prev[i_v, i_b] + sap_state.fem.x[i_v, i_b]
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            if sap_state.solver.is_full_step[i_b]:
                sap_state.rigid.v[i_d, i_b] = sap_state.rigid.x_prev[i_d, i_b] + sap_state.rigid.x[i_d, i_b]

    # The bracket of the root of the derivative of the energy along the search direction
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_linesearch_active[i_b]:
            max_step_size = sap_info.linesearch_max_step_size[None]
            ftol = sap_info.linesearch_ftol[None]
            sap_state.solver.dell_scale[i_b] = -sap_state.solver.m[i_b]
            sap_state.solver.step_size[i_b] = qd.min(
                -sap_state.solver.m[i_b] / sap_state.solver.d2ell_dalpha2[i_b], max_step_size
            )
            sap_state.solver.alpha_min[i_b] = 0.0
            sap_state.solver.alpha_max[i_b] = max_step_size
            sap_state.solver.f_upper[i_b] = sap_state.solver.dell_dalpha[i_b] / sap_state.solver.dell_scale[i_b]
            sap_state.solver.alpha_tol[i_b] = ftol * sap_state.solver.step_size[i_b]
            sap_state.solver.minus_dalpha[i_b] = sap_state.solver.alpha_min[i_b] - sap_state.solver.alpha_max[i_b]
            sap_state.solver.minus_dalpha_prev[i_b] = sap_state.solver.minus_dalpha[i_b]
            # The scaled derivative is -1 at the start of the bracket, a root for a tolerance above 1
            if ftol > 1.0:
                sap_state.solver.is_linesearch_active[i_b] = False
                sap_state.solver.step_size[i_b] = sap_state.solver.alpha_min[i_b]
            if qd.abs(sap_state.solver.f_upper[i_b]) < ftol:
                sap_state.solver.is_linesearch_active[i_b] = False
                sap_state.solver.step_size[i_b] = sap_state.solver.alpha_max[i_b]


@qd.kernel(fastcache=True)
def kernel_one_exact_linesearch_iter(
    f: int,
    sap_state: SAPState,
    fem_state: FEMState,
    dofs_state: DofsState,
    sap_info: SAPInfo,
    static_config: qd.template(),
):
    """
    Run one iteration of the exact line search: a Newton step on the derivative of the energy along the search
    direction, or a bisection of its bracket when Newton is slow or leaves the bracket.
    """
    func_compute_line_gradient_hessian(f, sap_state, fem_state, dofs_state, sap_info, static_config)
    for i_b in range(sap_state.solver.is_active.shape[0]):
        if sap_state.solver.is_linesearch_active[i_b]:
            func_sum_line_gradient_hessian(i_b, sap_state, sap_info, static_config)

            # The derivative of the energy, scaled, and its own derivative, which update the bracket
            f_value = sap_state.solver.dell_dalpha[i_b] / sap_state.solver.dell_scale[i_b]
            df_value = sap_state.solver.d2ell_dalpha2[i_b] / sap_state.solver.dell_scale[i_b]
            f_upper = sap_state.solver.f_upper[i_b]
            if (f_value > 0.0) != (f_upper > 0.0) or (f_value < 0.0) != (f_upper < 0.0):
                sap_state.solver.alpha_min[i_b] = sap_state.solver.step_size[i_b]
            else:
                sap_state.solver.alpha_max[i_b] = sap_state.solver.step_size[i_b]
                sap_state.solver.f_upper[i_b] = f_value
            if qd.abs(f_value) < sap_info.linesearch_ftol[None]:
                sap_state.solver.is_linesearch_active[i_b] = False
            if sap_state.solver.is_linesearch_active[i_b]:
                is_newton_slow = 2.0 * qd.abs(f_value) > qd.abs(sap_state.solver.minus_dalpha_prev[i_b] * df_value)
                sap_state.solver.minus_dalpha_prev[i_b] = sap_state.solver.minus_dalpha[i_b]
                is_bisected = is_newton_slow
                if not is_newton_slow:
                    sap_state.solver.minus_dalpha[i_b] = f_value / df_value
                    sap_state.solver.step_size[i_b] = (
                        sap_state.solver.step_size[i_b] - sap_state.solver.minus_dalpha[i_b]
                    )
                    is_bisected = (
                        sap_state.solver.step_size[i_b] <= sap_state.solver.alpha_min[i_b]
                        or sap_state.solver.step_size[i_b] >= sap_state.solver.alpha_max[i_b]
                    )
                if is_bisected:
                    sap_state.solver.minus_dalpha[i_b] = 0.5 * (
                        sap_state.solver.alpha_min[i_b] - sap_state.solver.alpha_max[i_b]
                    )
                    sap_state.solver.step_size[i_b] = (
                        sap_state.solver.alpha_min[i_b] - sap_state.solver.minus_dalpha[i_b]
                    )
                if qd.abs(sap_state.solver.minus_dalpha[i_b]) < sap_state.solver.alpha_tol[i_b]:
                    sap_state.solver.is_linesearch_active[i_b] = False


@qd.kernel(fastcache=True)
def kernel_update_vel(
    f: int,
    sap_state: SAPState,
    fem_state: FEMState,
    dofs_state: DofsState,
    static_config: qd.template(),
):
    """Write the velocities the SAP solve found back to the FEM vertices and the rigid dofs."""
    if qd.static(static_config.has_fem):
        for i_v, i_b in qd.ndrange(sap_state.fem.v.shape[0], sap_state.fem.v.shape[1]):
            fem_state.vertices.vel[f + 1, i_v, i_b] = sap_state.fem.v[i_v, i_b]
    if qd.static(static_config.has_rigid):
        for i_d, i_b in qd.ndrange(sap_state.rigid.v.shape[0], sap_state.rigid.v.shape[1]):
            dofs_state.vel[i_d, i_b] = sap_state.rigid.v[i_d, i_b]
