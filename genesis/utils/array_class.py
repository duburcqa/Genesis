from enum import IntEnum

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.utils.data import ENVS, Data, DataKind, Label, Size, StaticConfig, field, requires


def _timer_dtype():
    """Return the type of the profiling timers, 64-bit where the backend supports it."""
    return qd.i32 if gs.backend == gs.metal else qd.i64


# =========================================== ErrorCode ===========================================


class ErrorCode(IntEnum):
    SUCCESS = 0b000000000000000000000000000000000
    OVERFLOW_COLLISION_PAIRS = 0b00000000000000000000000000000001
    OVERFLOW_CANDIDATE_CONTACTS = 0b00000000000000000000000000000010
    OVERFLOW_CONTACTS = 0b00000000000000000000000000000100
    UNSUPPORTED_COLLISION_PAIR = 0b00000000000000000000000000001000
    INVALID_CONTACT_NAN = 0b00000000000000000000000000010000
    INVALID_FORCE_NAN = 0b00000000000000000000000000100000
    INVALID_ACC_NAN = 0b00000000000000000000000001000000
    INVALID_GRAD_NAN = 0b00000000000000000000000010000000
    INVALID_PARTICLE_NAN = 0b00000000000000000000000100000000


# =========================================== RigidInfo ===========================================


QS = Size("qs")
DOFS = Size("dofs")
LINKS = Size("links")
JOINTS = Size("joints")
GEOMS = Size("geoms")
ROOTS = Size("roots")
TREES = Size("trees")
# The links of every tree level by level, and the children of every link (see trees_levels_links_idx in RigidInfo)
TREES_LEVELS_LINKS = Size("trees_levels_links")
CHILD_LINKS = Size("child_links")

# The arrays of the dynamics, which a kinematic-only solver never reads
DYNAMICS = Label("dynamics")
# The tables of the level sweep (see trees_levels_links_idx)
LEVEL_SWEEP = Label("level_sweep")
# The scratch of the register-tiled mass factor (see mass_mat_tiled_scratch)
TILED_MASS = Label("tiled_mass")


class RigidInfo(Data, kind=DataKind.CONSTANT):
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO, labels=DYNAMICS)
    meaninertia: qd.Tensor = field(float, shape=(ENVS,), kind=DataKind.INFO, labels=DYNAMICS)
    # Awake dofs per env under hibernation, counted down where an island sleeps and up where one wakes. An env at zero
    # skips the passes gated on it. An env at the dof count skips the passes looking for a sleeper (the chain edges and
    # the wake pass of the island build, the pair filter of the broad phase, the contact advection, the inert rows). A
    # dof-less scene pads its dof buffers to one slot and takes the slow path.
    n_awake_dofs: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)
    qpos0: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.INFO)
    qpos: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.STATE, grad=True)
    qpos_next: qd.Tensor = field(float, shape=(QS, ENVS), kind=DataKind.SCRATCH, grad=True)
    links_T: qd.Tensor = field((float, (4, 4)), shape=(LINKS,), kind=DataKind.DERIVED)
    geoms_init_AABB: qd.Tensor = field((float, 3), shape=(GEOMS, 8), labels=DYNAMICS)
    # Stored reversed under the cooperative constraint kernels (see DataManager), i_d1 innermost, for their readers
    # striding i_d1 across lanes. The writers striding i_b have cooperative variants under the same flag.
    mass_mat: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), kind=DataKind.DERIVED, labels=DYNAMICS, grad=True)
    # Stored in declared order whatever the kernels: its main reader, a serial back-substitution with lanes varying
    # i_b, would lose more from the reversed order than the tiled factor gains.
    mass_mat_L: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), kind=DataKind.DERIVED, labels=DYNAMICS, grad=True)
    mass_mat_D_inv: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.DERIVED, labels=DYNAMICS, grad=True)
    # Scratch of the register-tiled mass factor, environment-first like the tile operations. The constraint Hessian
    # aliases it (see nt_H in ConstraintState).
    mass_mat_tiled_scratch: qd.Tensor = field(float, shape=(ENVS, DOFS, DOFS), kind=DataKind.SCRATCH, labels=TILED_MASS)
    # Kinematic roots: the links sharing a root link (links.root_idx), static ones included. A root spans the links
    # [root, links_root_end[root]), which may interleave links of other roots, so the walks gate each link on its root.
    # The composite inertia and the center of mass are per root.
    roots_link_idx: qd.Tensor = field(int, shape=(ROOTS,))
    links_root_end: qd.Tensor = field(int, shape=(LINKS,))
    # Kinematic trees: the links a chain of moving joints connects, a static link belonging to none (links_tree_idx -1)
    # and each branch of a fixed base being a tree. Tree i_t is rooted at trees_root_idx[i_t], spans the links
    # [trees_root_idx[i_t], trees_link_end[i_t]) mapped to it and the contiguous dofs [trees_dof_start[i_t],
    # trees_dof_start[i_t] + trees_n_dofs[i_t]). The islands are built on the trees. A scene without any tree keeps
    # one padded slot at root 0 and link end 0, so every tree walk is empty.
    trees_root_idx: qd.Tensor = field(int, shape=(TREES,))
    trees_link_end: qd.Tensor = field(int, shape=(TREES,))
    trees_n_links: qd.Tensor = field(int, shape=(TREES,))
    trees_dof_start: qd.Tensor = field(int, shape=(TREES,))
    trees_n_dofs: qd.Tensor = field(int, shape=(TREES,))
    links_tree_idx: qd.Tensor = field(int, shape=(LINKS,))
    # The links of every tree level by level, for the level sweep (see func_sweep_links_by_level). The levels of tree
    # i_t start at entry trees_level_start[i_t] and number trees_n_levels[i_t], and the level of entry i_k ends before
    # entry trees_levels_links_end[i_k]. The root tables list the links of every root the same way, the level of entry
    # i_k starting at entry roots_levels_links_start[i_k].
    trees_level_start: qd.Tensor = field(int, shape=(TREES,), labels=LEVEL_SWEEP)
    trees_n_levels: qd.Tensor = field(int, shape=(TREES,), labels=LEVEL_SWEEP)
    trees_levels_links_idx: qd.Tensor = field(int, shape=(TREES_LEVELS_LINKS,), labels=LEVEL_SWEEP)
    trees_levels_links_end: qd.Tensor = field(int, shape=(TREES_LEVELS_LINKS,), labels=LEVEL_SWEEP)
    roots_level_start: qd.Tensor = field(int, shape=(ROOTS,), labels=LEVEL_SWEEP)
    roots_n_levels: qd.Tensor = field(int, shape=(ROOTS,), labels=LEVEL_SWEEP)
    roots_levels_links_idx: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    roots_levels_links_start: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    roots_levels_links_end: qd.Tensor = field(int, shape=(LINKS,), labels=LEVEL_SWEEP)
    # The children of link i_l are links_child_idx[links_child_start[i_l]:links_child_start[i_l + 1]], in descending
    # order: the leaf-to-root folds of the level sweep add them into their parent in the order of the serial folds.
    links_child_start: qd.Tensor = field(int, shape=(LINKS + 1,), labels=LEVEL_SWEEP)
    links_child_idx: qd.Tensor = field(int, shape=(CHILD_LINKS,), labels=LEVEL_SWEEP)
    # Bounds of the mass block of each dof: the dofs of its branch rooted where the fixed structure ends, deeper
    # branches staying in the enclosing block, merged across entities and kept contiguous by attach(). The blocks
    # partition the dof range of a kinematic tree (an aligned free body splits into one block per dof), and the
    # assembly, the factor and the solve of the mass matrix walk them.
    dofs_mass_block_start: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    dofs_mass_block_end: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    dofs_mass_envelope_start: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    # Mass parent of each dof within its mass block, -1 for none: the previous dof of its link, else the last dof of
    # its nearest ancestor link holding one.
    dofs_mass_parent: qd.Tensor = field(int, shape=(DOFS,), labels=DYNAMICS)
    # Mask of the (dof, dof) pairs the mass matrix couples: a dof with its ancestors along the kinematic chain and the
    # dofs of its own link, within its mass block.
    mass_parent_mask: qd.Tensor = field(float, shape=(DOFS, DOFS), labels=DYNAMICS)
    # Runtime constants
    substep_dt: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    ls_iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    ls_tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    noslip_iterations: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    noslip_tolerance: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    impratio: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    n_equalities: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    n_candidate_equalities: qd.Tensor = field(int, shape=(), labels=DYNAMICS)
    hibernation_thresh_vel: qd.Tensor = field(float, shape=(), labels=DYNAMICS)
    EPS: qd.Tensor = field(float, shape=())


# =========================================== Constraint ===========================================


# Sizes of the constraint solve
CONSTRAINTS = Size("constraints")
CONTACTS = Size("contacts")
# The rows of each elliptic cone, and of all of them
ROWS_PER_CONTACT = Size("rows_per_contact")
CONE_ROWS = CONTACTS * ROWS_PER_CONTACT
# Fused updates of the per-island factor, one slot each (see nt_vec)
NT_VEC_SLOTS = Size("nt_vec_slots")
# Dofs a Jacobian row reaches at most (see jac_dofs_idx)
JAC_DOFS = Size("jac_dofs")
# Lanes of the cooperative noslip sweep, one without it (see noslip_MinvJT)
NOSLIP_LANES = Size("noslip_lanes")
# Size classes of the work-lists of the tiled island seed (see factor_worklist_i_b)
TILE_CLASSES = Size("tile_classes")

# The rows of the constraint state and of the islands the cooperative kernels read, one env per block, which they
# read from consecutive addresses once stored environment-first
COOPERATIVE_ROWS = Label("cooperative_rows")
# The Jacobian rows, which the cooperative kernels and serialized execution read whole for one env (see jac)
JACOBIAN_ROWS = Label("jacobian_rows")
# The rows serialized execution reads env by env, in the outermost loop of every constraint kernel
SERIAL_ROWS = Label("serial_rows")
NEWTON = Label("newton")
CONE_INCREMENTAL = Label("cone_incremental")
SIGNORINI = Label("signorini")
CONE_FREE_REUSE = Label("cone_free_reuse")
NOSLIP = Label("noslip")
NOSLIP_COOPERATIVE = Label("noslip_cooperative")
HIBERNATION = Label("hibernation")
# The reverse Cuthill-McKee tree reorder of the CPU per-island skyline path (see rcm_tree_pos)
RCM = Label("rcm")
# The contact lists of the islands, of the skyline reorder or of the cooperative contact sort (see contact_id)
ISLAND_CONTACTS = Label("island_contacts")
WORKLISTS = Label("worklists")
# The buffers of the backward pass of the constraint solve
BACKWARD = Label("backward")


class IslandSlices(Data, kind=DataKind.SCRATCH, shape=(TREES, ENVS)):
    # Slices of a packed id array per (island, env): the items of island i_island are id[start[i_island, i_b] :
    # start[i_island, i_b] + n[i_island, i_b]]. curr is the write cursor of the partition build, start + n once built.
    # An island holds at least one tree, hence one slice per tree.
    curr: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)
    n: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)
    start: qd.Tensor = field(int, labels=COOPERATIVE_ROWS)


class IslandState(Data, kind=DataKind.SCRATCH):
    # Partition of the dof-carrying kinematic trees (see trees_root_idx in RigidInfo) into islands, the connected
    # components of the trees under the contact, equality and hibernation couplings, rebuilt every step by island.py.
    # trees_parent_idx is the union-find forest over the trees, trees_island_idx the island of each tree and
    # links_island_idx that of each link, -1 for a static link and for a dof-less tree and its links. link_slices maps
    # an island to its slice of link_id, dof_slices to its slice of dof_id (island-local dof -> global dof, ascending
    # unless the CPU skyline path reorders it by contact adjacency).
    trees_parent_idx: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    trees_island_idx: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    links_island_idx: qd.Tensor = field(int, shape=(LINKS, ENVS), labels=COOPERATIVE_ROWS)
    n_islands: qd.Tensor = field(int, shape=(ENVS,))
    link_slices: IslandSlices = field(labels=HIBERNATION)
    link_id: qd.Tensor = field(int, shape=(LINKS, ENVS), labels=(COOPERATIVE_ROWS, HIBERNATION))
    dof_slices: IslandSlices
    dof_id: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Inverse of dof_id: dof_local_pos[d] is the island-local position of global dof d
    dof_local_pos: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    dofs_island_idx: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # First dof of an island whose dof list holds consecutive dofs in ascending order, -1 otherwise (see func_list_item
    # in linesearch.py).
    dof_range_start: qd.Tensor = field(int, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    # Skyline envelope of the CPU per-island path: dof_env_start_local[dof_slices.start[i] + ld] is the smallest
    # island-local column structurally nonzero in local row ld of island i's Hessian block, 0 (dense) when uncomputed.
    # dof_env_col_end is its transpose, the largest local row whose envelope reaches column ld, which bounds the
    # column-oriented factor and solve sweeps to the band: a tall island of a few hundred dofs otherwise tests every
    # row below the diagonal against its envelope, an O(n^2) scan per sweep. It has no safe uncomputed default (0
    # truncates), so only the CPU per-island path reads it.
    dof_env_start_local: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    dof_env_col_end: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Contact lists of the islands, for the CPU skyline reorder (see rcm_tree_pos). contact_id also carries the input
    # order of the cooperative contact sort (see func_sort_contacts_coop in island.py).
    contact_slices: IslandSlices = field(labels=RCM)
    contact_id: qd.Tensor = field(int, shape=(CONTACTS, ENVS), labels=(COOPERATIVE_ROWS, ISLAND_CONTACTS))
    constraint_slices: IslandSlices
    constraint_id: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Island of each constraint, -1 for one touching no island
    constraint_island_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # is_hibernated marks an island whose every link is asleep. hibernated_next_link chains the links of a sleeping
    # component, which the partition build re-unions before labeling, so the component stays one island across steps
    # although its bodies generate no live contacts.
    is_hibernated: qd.Tensor = field(int, shape=(TREES, ENVS), kind=DataKind.STATE, labels=HIBERNATION)
    hibernated_next_link: qd.Tensor = field(int, shape=(LINKS, ENVS), kind=DataKind.STATE, labels=HIBERNATION)
    # (env, island) work-lists of the cooperative per-island factor+solve, one per island size class (see
    # island_tile_caps): class c holds its factor_worklist_size[c] awake islands in its own region of the two index
    # lists, of n_trees * _B slots each, in the order of the atomic reservation of the partition pass. Read where an env
    # can hold several islands (see func_island_tiled_factor_solve_all in solver.py).
    factor_worklist_i_b: qd.Tensor = field(int, shape=(TILE_CLASSES * TREES * ENVS,), labels=WORKLISTS)
    factor_worklist_i_island: qd.Tensor = field(int, shape=(TILE_CLASSES * TREES * ENVS,), labels=WORKLISTS)
    factor_worklist_size: qd.Tensor = field(int, shape=(TILE_CLASSES,), labels=WORKLISTS)
    # Scratch of the reverse Cuthill-McKee tree reorder of the CPU per-island skyline path: the island-local slot, the
    # contact degree and the ordered flag of each tree, and the resulting visit order.
    rcm_tree_pos: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_degree: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_is_ordered: qd.Tensor = field(bool, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    rcm_tree_order: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, RCM))
    # Per-island Newton / CG iteration state (see linesearch.py), indexed [i_island, i_b]: inertia is the trace of the
    # mass matrix over the island's dofs, the scale of its convergence tests, and improved says whether the island
    # still iterates, every pass of an iteration being confined to the islands that do. ls_improvement carries
    # -cost(alpha_accepted) in shifted form, cost(alpha) - cost(0), which stays resolvable in float32 near convergence.
    inertia: qd.Tensor = field(float, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    improved: qd.Tensor = field(bool, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)
    ls_improvement: qd.Tensor = field(float, shape=(TREES, ENVS), labels=COOPERATIVE_ROWS)


# =========================================== IKState ===========================================


# The counts of the largest entity of the solver, each taken over every entity
ENTITY_QS = Size("entity_qs")
ENTITY_DOFS = Size("entity_dofs")
ENTITY_LINKS = Size("entity_links")
ENTITY_JOINTS = Size("entity_joints")
# The cap of inverse-kinematics targets
IK_TARGETS = Size("ik_targets")
# The pose residuals of the targets stacked, six per target
IK_ERRORS = 6 * IK_TARGETS


class IKState(Data, kind=DataKind.SCRATCH):
    """Damped-least-squares inverse-kinematics scratch, one column per environment or per restart of the planner."""

    # Spatial Jacobian of one target, rebuilt per target link and copied into the stacked block
    jacobian: qd.Tensor = field(float, shape=(6, ENTITY_DOFS, ENVS))
    # Jacobian of all targets stacked, and its transpose
    jacobian_stacked: qd.Tensor = field(float, shape=(IK_ERRORS, ENTITY_DOFS, ENVS))
    jacobian_stacked_t: qd.Tensor = field(float, shape=(ENTITY_DOFS, IK_ERRORS, ENVS))
    # Damped normal matrix J J^T + damping^2 I, its LU factors with their forward-substitution scratch, and its inverse
    mat: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    lu_lower: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    lu_upper: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    lu_y: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    inv: qd.Tensor = field(float, shape=(IK_ERRORS, IK_ERRORS, ENVS))
    # Pose residual of all targets stacked, the best one across restarts, and inv @ residual
    err_pose: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    err_pose_best: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    vec: qd.Tensor = field(float, shape=(IK_ERRORS, ENVS))
    # Joint-space step, and the best configuration across restarts
    delta_qpos: qd.Tensor = field(float, shape=(ENTITY_DOFS, ENVS))
    qpos_best: qd.Tensor = field(float, shape=(ENTITY_QS, ENVS))


class IKScratchFK(Data, kind=DataKind.SCRATCH):
    """Forward kinematics of the inverse-kinematics solve of an entity, apart from the state of the solver.

    It holds the configuration the solve iterates and the link and joint frames of that configuration, one column per
    environment.
    """

    qpos: qd.Tensor = field(float, shape=(ENTITY_QS, ENVS))
    # Written by func_forward_kinematics_scratch
    links_pos: qd.Tensor = field((float, 3), shape=(ENTITY_LINKS, ENVS))
    links_quat: qd.Tensor = field((float, 4), shape=(ENTITY_LINKS, ENVS))
    joints_xanchor: qd.Tensor = field((float, 3), shape=(ENTITY_JOINTS, ENVS))
    joints_xaxis: qd.Tensor = field((float, 3), shape=(ENTITY_JOINTS, ENVS))


class IKTargets(Data, kind=DataKind.SCRATCH):
    """Targets and dofs of an inverse-kinematics solve, one column per environment or per restart of the planner.

    The caller writes every field before the solve.
    """

    # Global indices of the target links, of the dofs to move and of the columns to solve
    links_idx: qd.Tensor = field(int, shape=(IK_TARGETS,))
    dofs_idx: qd.Tensor = field(int, shape=(ENTITY_DOFS,))
    envs_idx: qd.Tensor = field(int, shape=(ENVS,))
    # Target pose per (link, column) in the world frame, and the link-local point it drives
    pos: qd.Tensor = field((float, 3), shape=(IK_TARGETS, ENVS))
    quat: qd.Tensor = field((float, 4), shape=(IK_TARGETS, ENVS))
    local_point: qd.Tensor = field((float, 3), shape=(IK_TARGETS,))
    # Initial configuration per column, read when custom_init_qpos is set
    init_qpos: qd.Tensor = field(float, shape=(ENVS, ENTITY_QS))
    # The position and rotation axes to solve for all links, and the position and rotation switches of each link
    pos_mask: qd.Tensor = field(float, shape=(3,))
    rot_mask: qd.Tensor = field(float, shape=(3,))
    link_pos_mask: qd.Tensor = field(int, shape=(IK_TARGETS,))
    link_rot_mask: qd.Tensor = field(int, shape=(IK_TARGETS,))


class IKScratch(Data, kind=DataKind.SCRATCH):
    """The scratch of the inverse-kinematics solves of every entity of a solver.

    It spans the largest entity and the target cap, and a solve writes the leading rows its entity spans.
    """

    state: IKState
    fk: IKScratchFK
    targets: IKTargets


# The scratch of the kinematics queries, over the dofs of the widest entity of the solver (see ENTITY_DOFS)
KINEMATICS = Label("kinematics")


class KinematicsScratch(Data, kind=DataKind.SCRATCH):
    """The scratch of the kinematics queries of every entity of a solver, one column per environment.

    The Jacobian spans the widest entity, and the configuration cache spans the configuration of the solver, indexed
    by global q indices.
    """

    jacobian: qd.Tensor = field(float, shape=(6, ENTITY_DOFS, ENVS), labels=KINEMATICS)
    qpos_cache: qd.Tensor = field(float, shape=(QS, ENVS), labels=KINEMATICS)


class WeightScratch(Data, kind=DataKind.SCRATCH, shape=(DOFS, ENVS)):
    """The scratch of the inverse weights of a solver: a row of a link Jacobian and the mass-matrix solve against it."""

    jac_row: qd.Tensor = field(float)
    solve_out: qd.Tensor = field(float)


class ConstraintState(Data, kind=DataKind.SCRATCH):
    # /!\ The declaration order is the allocation order, which changing may reduce runtime speed by >10% /!\
    n_constraints: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    qd_n_equalities: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)
    n_constraints_equality: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    n_constraints_frictionloss: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    # Elliptic-cone contact rows, rows_per_contact per contact, zero for the pyramidal cone. They occupy
    # [ne + n_frictionloss, ne + n_frictionloss + n_cone), the start of the collision segment.
    n_constraints_cone: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    is_warmstart: qd.Tensor = field(bool, shape=(ENVS,), kind=DataKind.STATE)
    improved: qd.Tensor = field(bool, shape=(ENVS,))
    cost_ws: qd.Tensor = field(float, shape=(ENVS,))
    cost: qd.Tensor = field(float, shape=(ENVS,))
    Ma: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    Ma_ws: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    grad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    Mgrad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Cone-row residuals of the previous iteration, for the CPU elliptic cone: the incremental factor downdates the
    # previous coupled cone block by a rank-3 update, after the linesearch overwrote Jaref.
    cone_prev_jaref: qd.Tensor = field(float, shape=(CONE_ROWS, ENVS), labels=(SERIAL_ROWS, CONE_INCREMENTAL))
    # Per contact under 'signorini': the latched normal force scaling its friction discs (negative until seeded, see
    # func_relatch_cone in solver.py), and the gap between the iterate's normal force and the latch at the last update.
    cone_latch: qd.Tensor = field(float, shape=(CONTACTS, ENVS), labels=SIGNORINI)
    cone_latch_gap: qd.Tensor = field(float, shape=(CONTACTS, ENVS), labels=SIGNORINI)
    search: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    qfrc_constraint: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    qacc: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    qacc_ws: qd.Tensor = field(float, shape=(DOFS, ENVS), kind=DataKind.STATE, labels=COOPERATIVE_ROWS)
    mv: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    cg_prev_grad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    cg_prev_Mgrad: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=COOPERATIVE_ROWS)
    # Rank-1 vectors of the incremental Cholesky update, slot-minor at [i_d * n_slots + i_u]: one slot per fused update
    # of the per-island factor (see func_rank_batch_update_island), one for the factor of a single-island scene. The
    # flat index keeps the rank and the layout of the other dof vectors.
    nt_vec: qd.Tensor = field(float, shape=(DOFS * NT_VEC_SLOTS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    # Cholesky factor L of the Hessian H = L @ L.T, in its lower triangle. The strictly upper triangle is undefined,
    # except under enable_cone_free_hessian_reuse: the mirror of each used slot holds the Hessian without its cone
    # blocks (M + J^T D J of the active rows), kept across the Newton iterations of a step by signed flip scatters,
    # its diagonal in nt_H_cone_free_diag. A rebuild restores the mirror into the lower triangle and adds the current
    # cone blocks, skipping the full J^T D J assembly. It aliases the scratch of the register-tiled mass factor when
    # there is one (see the constraint solver).
    # TODO: Allocate the lower triangle alone.
    nt_H: qd.Tensor = field(float, shape=(ENVS, DOFS, DOFS), labels=NEWTON)
    # Jacobi scale of each dof under enable_jacobi_equilibration, s_i = 1 / sqrt(diag(M + J^T D J)_i), 1 on an empty
    # diagonal, refreshed before every rebuild (see func_jacobi_scale). The assembly writes H scaled, nt_H holds the
    # factor of S H S, the update vectors are scaled when built, and the batch solve scales grad and Mgrad, so Mgrad =
    # H^-1 grad exactly. The unit diagonal keeps the mixed units of the Hessian within float32 conditioning at any
    # geometry scale and makes the EPS pivot floor relative.
    nt_jacobi: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    # The constraints whose active state changed, which the incremental Cholesky update iterates over
    incr_changed_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    incr_n_changed: qd.Tensor = field(int, shape=(ENVS,))
    active: qd.Tensor = field(bool, shape=(CONSTRAINTS, ENVS), kind=DataKind.DERIVED, labels=COOPERATIVE_ROWS)
    prev_active: qd.Tensor = field(bool, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NEWTON))
    diag: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    aref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    Jaref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # The friction loss of a frictionloss row. The normal row of an elliptic cone holds the sliding friction coefficient
    # of its contact and its spin row the torsional one, its tangent rows 0.
    efc_frictionloss: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    efc_force: qd.Tensor = field(
        float, shape=(CONSTRAINTS, ENVS), kind=DataKind.DERIVED, labels=COOPERATIVE_ROWS, grad=True
    )
    efc_D: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    jv: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Stored environment-first for the cooperative kernels and serialized execution (see JACOBIAN_ROWS). The lanes of
    # the cooperative J^T @ efc_force and Hessian staging then read consecutive dofs of a row, whose support spans a
    # cache line or two.
    jac: qd.Tensor = field(float, shape=(CONSTRAINTS, DOFS, ENVS), labels=JACOBIAN_ROWS)
    # The support of each row: the dofs of the kinematic chains of the two links it couples, or the two dofs of a joint
    # equality
    jac_dofs_idx: qd.Tensor = field(int, shape=(CONSTRAINTS, JAC_DOFS, ENVS), labels=JACOBIAN_ROWS)
    jac_n_dofs: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=COOPERATIVE_ROWS)
    # Backward gradients
    dL_dqacc: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    dL_dM: qd.Tensor = field(float, shape=(DOFS, DOFS, ENVS), labels=BACKWARD)
    dL_djac: qd.Tensor = field(float, shape=(CONSTRAINTS, DOFS, ENVS), labels=BACKWARD)
    dL_daref: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    dL_defc_D: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    dL_dforce: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    # Scratch of the linear solve of the backward pass
    bw_u: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_r: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_p: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_Ap: qd.Tensor = field(float, shape=(DOFS, ENVS), labels=BACKWARD)
    bw_Ju: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    bw_y: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    bw_w: qd.Tensor = field(float, shape=(CONSTRAINTS, ENVS), labels=BACKWARD)
    timers: qd.Tensor = field(_timer_dtype, shape=(10, ENVS))
    # Iteration of the solver loop, from 0
    solver_iter_counter: qd.Tensor = field(qd.i32, shape=())
    # An ndarray on every backend, since graph.do_while takes the same ndarray on every call
    graph_counter: qd.Tensor = field(qd.i32, shape=(), ndarray=True)
    # Iterations left to the graph loop past the current one, resumed once an env iterates on
    graph_counter_next: qd.Tensor = field(qd.i32, shape=())
    # Diagonal of the Hessian without its cone blocks (see nt_H)
    nt_H_cone_free_diag: qd.Tensor = field(float, shape=(ENVS, DOFS), labels=CONE_FREE_REUSE)
    # M^-1 J^T of the two rows of the pair the noslip sweep updates, in the two columns of the env, or of the lane of
    # the cooperative sweep at [i_d, 2 * (i_b * 32 + tid) + i_r] (see func_noslip_update_row in noslip.py)
    noslip_MinvJT: qd.Tensor = field(float, shape=(DOFS, 2 * NOSLIP_LANES * ENVS), labels=NOSLIP)
    # Row coloring of the cooperative noslip sweep (see func_color_rows_batch in noslip.py): the color of each row, the
    # color count of each island and the next free color of the mass block starting at each dof
    noslip_rows_color: qd.Tensor = field(int, shape=(CONSTRAINTS, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    noslip_islands_n_colors: qd.Tensor = field(int, shape=(TREES, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    noslip_blocks_n_colors: qd.Tensor = field(int, shape=(DOFS, ENVS), labels=(COOPERATIVE_ROWS, NOSLIP_COOPERATIVE))
    island: IslandState


# =========================================== Collider ===========================================


# Sizes of the collision detection
VERTS = Size("verts")
# Link pairs holding a valid geom pair, the possible geom pairs and those the broad phase keeps at most
VALID_LINK_PAIRS = Size("valid_link_pairs")
POSSIBLE_PAIRS = Size("possible_pairs")
BROAD_PAIRS = Size("broad_pairs")
# Contacts the narrow phase detects at most per environment, and per nonconvex pair
CANDIDATE_CONTACTS = Size("candidate_contacts")
NONCONVEX_PAIR_CONTACTS = Size("nonconvex_pair_contacts")
# A pair holds its first contact and its four perturbed ones (see N_PERTURBATIONS in narrowphase.py), or its first
# contact alone under the contact patch
CONTACT_SLOTS = Size("contact_slots")
# Threads of the terrain pass, each testing the prism under one triangle
TERRAIN_THREADS = Size("terrain_threads")

# The scratch of the box-box detection
BOX_BOX = Label("box_box")
# The contact cache, which the convex-convex detection alone reads
CONTACT_CACHE = Label("contact_cache")
# The inputs of the differentiable contact detection, which the backward pass reads
DIFF_CONTACTS = Label("diff_contacts")
# The work queue of the split narrowphase and of the terrain pass, its contact slots, and the status of each slot
WORK_QUEUE = Label("work_queue")
SPLIT_NARROWPHASE = Label("split_narrowphase")
PERTURBED_SLOTS = Label("perturbed_slots")
# The scratch of the link-pair contact pruning, read by the threads of the block running an environment
PRUNING = Label("pruning")


class ContactData(Data, kind=DataKind.DERIVED, shape=(CANDIDATE_CONTACTS, ENVS)):
    geom_a: qd.Tensor = field(int)
    geom_b: qd.Tensor = field(int)
    normal: qd.Tensor = field((float, 3), grad=True)
    pos: qd.Tensor = field((float, 3), grad=True)
    penetration: qd.Tensor = field(float, grad=True)
    friction: qd.Tensor = field(float)
    friction_torsional: qd.Tensor = field(float)
    friction_rolling: qd.Tensor = field(float)
    sol_params: qd.Tensor = field((float, 7))
    force: qd.Tensor = field((float, 3))
    link_a: qd.Tensor = field(int)
    link_b: qd.Tensor = field(int)
    pair_idx: qd.Tensor = field(int)


class DiffContactInput(Data, kind=DataKind.SCRATCH):
    geom_a: qd.Tensor = field(int)
    geom_b: qd.Tensor = field(int)
    # Local positions of the vertices of the two geoms defining the face of the Minkowski difference
    local_pos1_a: qd.Tensor = field((float, 3))
    local_pos1_b: qd.Tensor = field((float, 3))
    local_pos1_c: qd.Tensor = field((float, 3))
    local_pos2_a: qd.Tensor = field((float, 3))
    local_pos2_b: qd.Tensor = field((float, 3))
    local_pos2_c: qd.Tensor = field((float, 3))
    # Local positions of the vertices of the two geoms defining the support point of that face
    w_local_pos1: qd.Tensor = field((float, 3))
    w_local_pos2: qd.Tensor = field((float, 3))
    # Support core of the convex geom of a plane-convex contact (box vertex, sphere center, nearest capsule endpoint),
    # in its local frame
    core_local: qd.Tensor = field((float, 3))
    ref_id: qd.Tensor = field(int)
    # Whether the forward and backward passes compute the contact stably
    valid: qd.Tensor = field(int)
    # Penetration depth weighting the contact, the input the gradients flow to
    ref_penetration: qd.Tensor = field(float, grad=True)


class SortBuffer(Data, kind=DataKind.STATE, shape=(2 * LINKS, ENVS)):
    value: qd.Tensor = field(float)
    i_l: qd.Tensor = field(int)
    is_max: qd.Tensor = field(bool)


class ContactCache(Data, kind=DataKind.STATE, shape=(POSSIBLE_PAIRS, ENVS)):
    normal: qd.Tensor = field((float, 3))
    # Penetration of each pair at the previous step, 0 out of contact, which gates the fallback from MPR to GJK
    penetration: qd.Tensor = field(float)


class NarrowphaseWorkQueue(Data, kind=DataKind.SCRATCH, shape=(BROAD_PAIRS * ENVS,)):
    # The pair of every entry (see collision_pair_idx in ColliderInfo), queued for the multicontact pass, then for the
    # terrain pass, which runs after it
    i_b: qd.Tensor = field(int, labels=WORK_QUEUE)
    i_ga: qd.Tensor = field(int, labels=WORK_QUEUE)
    i_gb: qd.Tensor = field(int, labels=WORK_QUEUE)
    i_pair: qd.Tensor = field(int, labels=WORK_QUEUE)
    # Candidate contacts of the multicontact pass, one slot per contact a pair can hold. The contact0 kernel stores the
    # first contact of the pair in slot 0, the detections of the pass store theirs in their own slot, and the gather
    # accepts them in slot order according to their status (see MULTICONTACT_SLOT in collider/constants.py). Under the
    # contact patch an entry holds slot 0 alone and no status.
    contact_pos: qd.Tensor = field((float, 3), shape=(BROAD_PAIRS * ENVS, CONTACT_SLOTS), labels=SPLIT_NARROWPHASE)
    contact_normal: qd.Tensor = field((float, 3), shape=(BROAD_PAIRS * ENVS, CONTACT_SLOTS), labels=SPLIT_NARROWPHASE)
    contact_penetration: qd.Tensor = field(float, shape=(BROAD_PAIRS * ENVS, CONTACT_SLOTS), labels=SPLIT_NARROWPHASE)
    contact_status: qd.Tensor = field(int, shape=(BROAD_PAIRS * ENVS, CONTACT_SLOTS), labels=PERTURBED_SLOTS)
    # Where contact 0 of the entry stands (see MULTICONTACT_ENTRY in collider/constants.py). The contact0 kernel leaves
    # it to Gilbert-Johnson-Keerthi (GJK) when the per-pair gate from Minkowski portal refinement (MPR) to GJK fired,
    # and each perturbed contact otherwise falls back from MPR to GJK.
    contact0_status: qd.Tensor = field(int, labels=SPLIT_NARROWPHASE)
    # The number of pairs queued and claimed so far
    n_pairs: qd.Tensor = field(int, shape=(1,), labels=WORK_QUEUE)
    n_claimed: qd.Tensor = field(int, shape=(1,), labels=WORK_QUEUE)


class ColliderState(Data, kind=DataKind.SCRATCH):
    sort_buffer: SortBuffer
    active_buffer: qd.Tensor = field(int, shape=(LINKS, ENVS))
    # Bounding box of every link, the box of its frame moved with it (see func_collision_clear in broadphase.py)
    links_aabb_min: qd.Tensor = field((float, 3), shape=(LINKS, ENVS))
    links_aabb_max: qd.Tensor = field((float, 3), shape=(LINKS, ENVS))
    # Valid link pairs whose boxes overlap, by index into ColliderInfo.valid_link_pairs
    n_broad_link_pairs: qd.Tensor = field(int, shape=(ENVS,))
    broad_link_pairs: qd.Tensor = field(int, shape=(VALID_LINK_PAIRS, ENVS))
    n_broad_pairs: qd.Tensor = field(int, shape=(ENVS,))
    box_depth: qd.Tensor = field(float, shape=(NONCONVEX_PAIR_CONTACTS, ENVS), labels=BOX_BOX)
    box_points: qd.Tensor = field((float, 3), shape=(NONCONVEX_PAIR_CONTACTS, ENVS), labels=BOX_BOX)
    box_pts: qd.Tensor = field((float, 3), shape=(6, ENVS), labels=BOX_BOX)
    box_lines: qd.Tensor = field((float, 6), shape=(4, ENVS), labels=BOX_BOX)
    box_linesu: qd.Tensor = field((float, 6), shape=(4, ENVS), labels=BOX_BOX)
    box_axi: qd.Tensor = field((float, 3), shape=(3, ENVS), labels=BOX_BOX)
    box_ppts2: qd.Tensor = field(float, shape=(4, 2, ENVS), labels=BOX_BOX)
    box_pu: qd.Tensor = field((float, 3), shape=(4, ENVS), labels=BOX_BOX)
    # The prism under the triangle every thread of the terrain pass tests, which reads its thread count from its shape
    prism: qd.Tensor = field((float, 3), shape=(6, TERRAIN_THREADS))
    n_contacts: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    # Kept contacts of the sleepers, at the front of the contact buffer with the identity permutation: a hibernated
    # link's contacts against fixed links and other sleepers are carried from step to step with the force of the last
    # solve they took part in, for the contact getters and the per-link contact force alone. Every per-step pass
    # (prune, sort, island edges, constraint rows, noslip) walks the live contacts after them,
    # [n_contacts_hibernated, n_contacts).
    n_contacts_hibernated: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.DERIVED)
    first_time: qd.Tensor = field(bool, shape=(ENVS,), kind=DataKind.STATE)
    contact_cache: ContactCache = field(labels=CONTACT_CACHE)
    broad_collision_pairs: qd.Tensor = field((int, 2), shape=(BROAD_PAIRS, ENVS))
    contact_data: ContactData
    # Input data for differentiable contact detection used in the backward pass
    diff_contact_input: DiffContactInput = field(shape=(ENVS, CANDIDATE_CONTACTS), labels=DIFF_CONTACTS)
    work_queue: NarrowphaseWorkQueue
    contact_sort_key: qd.Tensor = field(float, shape=(CANDIDATE_CONTACTS, ENVS))
    contact_sort_idx: qd.Tensor = field(int, shape=(CANDIDATE_CONTACTS, ENVS), kind=DataKind.DERIVED)
    contact_proj_v: qd.Tensor = field(float, shape=(CANDIDATE_CONTACTS, ENVS), labels=PRUNING)
    contact_keep: qd.Tensor = field(int, shape=(CANDIDATE_CONTACTS, ENVS), labels=PRUNING)
    contact_hull_stack: qd.Tensor = field(int, shape=(CANDIDATE_CONTACTS, ENVS), labels=PRUNING)
    # Permutation of the (u, v) lexicographic sort of the cooperative contact deduplication (see
    # func_clamp_prune_contacts_coop)
    contact_lex_idx: qd.Tensor = field(int, shape=(CANDIDATE_CONTACTS, ENVS), labels=PRUNING)


class VertsSpatialGrid(Data, kind=DataKind.CONSTANT):
    # Per-geom 8x8x8 grid over collision verts in the local AABB: a permutation of vert indices sorted by grid
    # cell (z fastest), the matching vert positions duplicated in that order for sequential streaming, and per-cell
    # vert ranges (8^3 + 1 entries per geom), so a scan visits only the cells overlapping a query box. The cell
    # mapping is anchored by geoms_origin / geoms_inv_cell_size in the geom frame.
    verts_idx: qd.Tensor = field(int, shape=(VERTS,))
    verts_pos: qd.Tensor = field((float, 3), shape=(VERTS,))
    cells_vert_start: qd.Tensor = field(int, shape=(GEOMS * (8**3 + 1),))
    geoms_origin: qd.Tensor = field((float, 3), shape=(GEOMS,))
    geoms_inv_cell_size: qd.Tensor = field((float, 3), shape=(GEOMS,))


@qd.data_oriented
class ColliderStaticConfig(StaticConfig):
    has_terrain: bool
    # Whether a convex-convex pair reaches the general narrowphase past the box-box and plane-box specializations: the
    # split narrowphase on GPU, func_narrow_phase_convex_vs_convex on CPU. It also gates the deterministic contact sort
    # (see func_sort_contacts). See its resolution in collider.py.
    has_non_box_plane_convex_convex: bool
    has_convex_specialization: bool
    has_nonconvex_nonterrain: bool
    # Whether the link-pair contact pruning pass is compiled in, see its resolution in collider.py
    has_prunable_contacts: bool
    # Contacts a convex pair and a nonconvex (vertex-versus-SDF) pair hold at most, the second one at least the first
    n_contacts_per_convex_pair: int
    n_contacts_per_nonconvex_pair: int
    ccd_algorithm: int
    # GPU core count, sizing the split narrow phase launches and scratch states
    gpu_cores: int


# =========================================== MPR ===========================================


class MPRSimplexSupport(Data, kind=DataKind.SCRATCH, shape=(4, ENVS)):
    v1: qd.Tensor = field((float, 3))
    v2: qd.Tensor = field((float, 3))
    v: qd.Tensor = field((float, 3))


class MPRState(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    simplex_support: MPRSimplexSupport
    simplex_size: qd.Tensor = field(int)
    # The PORTAL_STATUS of the contact depth in simplex_support[1..3], zero without a portal. A VALID portal seeds the
    # perturbation reconstruction and the EPA, an INVALID one forces a GJK refinement.
    portal_status: qd.Tensor = field(int)


class MPRInfo(Data, kind=DataKind.CONSTANT, shape=()):
    CCD_EPS: qd.Tensor = field(float)
    CCD_TOLERANCE: qd.Tensor = field(float)
    CCD_ITERATIONS: qd.Tensor = field(float)


# The states MPR runs on: one per env, and those of the passes of the split narrowphase and of the terrain pass, one
# per thread, which the collider opts in when it runs them
class MPRScratch(Data, kind=DataKind.SCRATCH):
    main: MPRState
    contact0: MPRState = field(opt_in=True)
    multicontact: MPRState = field(opt_in=True)


# =========================================== GJK ===========================================


# Sizes of the scratch of Gilbert-Johnson-Keerthi (GJK) and of the expanding polytope algorithm (EPA): the vertices,
# faces and horizon edges of the polytope, the contacts of a pair and the vertices of a contact polygon
POLYTOPE_VERTS = Size("polytope_verts")
POLYTOPE_FACES = Size("polytope_faces")
POLYTOPE_HORIZONS = POLYTOPE_VERTS + 1
PAIR_CONTACTS = Size("pair_contacts")
POLYGON_VERTS = Size("polygon_verts")
# The arrays of the EPA, of the multi-contact detection and of their results, which the overlap test never reads
EPA = Label("epa")


class MDVertex(Data, kind=DataKind.SCRATCH):
    """A vertex of the Minkowski difference."""

    obj1: qd.Tensor = field((float, 3))
    obj2: qd.Tensor = field((float, 3))
    local_obj1: qd.Tensor = field((float, 3))
    local_obj2: qd.Tensor = field((float, 3))
    id1: qd.Tensor = field(int)
    id2: qd.Tensor = field(int)
    mink: qd.Tensor = field((float, 3))


class GJKSimplex(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    nverts: qd.Tensor = field(int)


class GJKSimplexBuffer(Data, kind=DataKind.SCRATCH, shape=(ENVS, 4)):
    normal: qd.Tensor = field((float, 3))
    sdist: qd.Tensor = field(float)


class EPAPolytope(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    nverts: qd.Tensor = field(int)
    nfaces: qd.Tensor = field(int)
    nfaces_map: qd.Tensor = field(int)
    horizon_nedges: qd.Tensor = field(int)
    horizon_w: qd.Tensor = field((float, 3))


class EPAPolytopeFace(Data, kind=DataKind.SCRATCH, shape=(ENVS, POLYTOPE_FACES)):
    verts_idx: qd.Tensor = field((int, 3))
    adj_idx: qd.Tensor = field((int, 3))
    normal: qd.Tensor = field((float, 3))
    dist2: qd.Tensor = field(float)
    map_idx: qd.Tensor = field(int)
    visited: qd.Tensor = field(int)


class EPAPolytopeHorizonData(Data, kind=DataKind.SCRATCH):
    face_idx: qd.Tensor = field(int)
    edge_idx: qd.Tensor = field(int)


class ContactFace(Data, kind=DataKind.SCRATCH, shape=(ENVS, POLYGON_VERTS)):
    vert1: qd.Tensor = field((float, 3))
    vert2: qd.Tensor = field((float, 3))
    endverts: qd.Tensor = field((float, 3))
    normal1: qd.Tensor = field((float, 3))
    normal2: qd.Tensor = field((float, 3))
    id1: qd.Tensor = field(int)
    id2: qd.Tensor = field(int)


class ContactNormal(Data, kind=DataKind.SCRATCH, shape=(ENVS, POLYGON_VERTS)):
    endverts: qd.Tensor = field((float, 3))
    normal: qd.Tensor = field((float, 3))
    id: qd.Tensor = field(int)


class ContactHalfspace(Data, kind=DataKind.SCRATCH, shape=(ENVS, POLYGON_VERTS)):
    normal: qd.Tensor = field((float, 3))
    dist: qd.Tensor = field(float)


class Witness(Data, kind=DataKind.SCRATCH, shape=(ENVS, PAIR_CONTACTS)):
    point_obj1: qd.Tensor = field((float, 3))
    point_obj2: qd.Tensor = field((float, 3))


class GJKState(Data, kind=DataKind.SCRATCH):
    """
    The scratch of a detection by Gilbert-Johnson-Keerthi (GJK) and the expanding polytope algorithm (EPA), one column
    per thread or environment.

    The state of the overlap test alone leaves the arrays labeled EPA unused.
    """

    # GJK simplex
    support_mesh_prev_vertex_id: qd.Tensor = field(int, shape=(ENVS, 2))
    simplex_vertex: MDVertex = field(shape=(ENVS, 4))
    simplex_buffer: GJKSimplexBuffer
    simplex: GJKSimplex
    last_searched_simplex_vertex_id: qd.Tensor = field(int, shape=(ENVS,))
    simplex_vertex_intersect: MDVertex = field(shape=(ENVS, 4))
    simplex_buffer_intersect: GJKSimplexBuffer
    nsimplex: qd.Tensor = field(int, shape=(ENVS,))
    # EPA polytope
    polytope: EPAPolytope = field(labels=EPA)
    polytope_verts: MDVertex = field(shape=(ENVS, POLYTOPE_VERTS), labels=EPA)
    polytope_faces: EPAPolytopeFace = field(labels=EPA)
    polytope_faces_map: qd.Tensor = field(int, shape=(ENVS, POLYTOPE_FACES), labels=EPA)
    polytope_horizon_data: EPAPolytopeHorizonData = field(shape=(ENVS, POLYTOPE_HORIZONS), labels=EPA)
    polytope_horizon_stack: EPAPolytopeHorizonData = field(shape=(ENVS, 3 * POLYTOPE_FACES), labels=EPA)
    # Multi-contact detection by the contact patch
    contact_faces: ContactFace = field(labels=EPA)
    contact_normals: ContactNormal = field(labels=EPA)
    contact_halfspaces: ContactHalfspace = field(labels=EPA)
    contact_clipped_polygons: qd.Tensor = field((float, 3), shape=(ENVS, 2, POLYGON_VERTS), labels=EPA)
    multi_contact_flag: qd.Tensor = field(bool, shape=(ENVS,))
    # Results
    witness: Witness
    n_witness: qd.Tensor = field(int, shape=(ENVS,))
    n_contacts: qd.Tensor = field(int, shape=(ENVS,), labels=EPA)
    contact_pos: qd.Tensor = field((float, 3), shape=(ENVS, PAIR_CONTACTS), labels=EPA)
    normal: qd.Tensor = field((float, 3), shape=(ENVS, PAIR_CONTACTS), labels=EPA)
    is_col: qd.Tensor = field(bool, shape=(ENVS,), labels=EPA)
    penetration: qd.Tensor = field(float, shape=(ENVS,), labels=EPA)
    distance: qd.Tensor = field(float, shape=(ENVS,))
    # The EPA polytope face nearest to the origin, -1 when EPA did not run, from whose support simplex the multi-contact
    # perturbation recovers the exact unperturbed normal
    nearest_face: qd.Tensor = field(int, shape=(ENVS,))
    # Differentiable contact detection
    diff_contact_input: DiffContactInput = field(shape=(ENVS, PAIR_CONTACTS), labels=DIFF_CONTACTS)
    n_diff_contact_input: qd.Tensor = field(int, shape=(ENVS,), labels=EPA)
    diff_penetration: qd.Tensor = field(float, shape=(ENVS, PAIR_CONTACTS), labels=DIFF_CONTACTS)


class GJKInfo(Data, kind=DataKind.CONSTANT, shape=()):
    max_contacts_per_pair: qd.Tensor = field(int)
    max_contact_polygon_verts: qd.Tensor = field(int)
    # Iteration caps of GJK and EPA
    gjk_max_iterations: qd.Tensor = field(int)
    epa_max_iterations: qd.Tensor = field(int)
    FLOAT_MIN: qd.Tensor = field(float)
    FLOAT_MIN_SQ: qd.Tensor = field(float)
    FLOAT_MAX: qd.Tensor = field(float)
    FLOAT_MAX_SQ: qd.Tensor = field(float)
    # Convergence tolerance of GJK and EPA on smooth geometries
    tolerance: qd.Tensor = field(float)
    # Distance below which two geoms collide
    collision_eps: qd.Tensor = field(float)
    # Degeneracy measure (colinearity, coplanarity) below which safe GJK discards a simplex, degenerate simplices being
    # the main source of EPA errors
    simplex_max_degeneracy_sq: qd.Tensor = field(float)
    polytope_max_faces: qd.Tensor = field(int)
    # Reprojection errors above which the projection of the origin onto a polytope face is rejected when computing the
    # witness points: the point rebuilt from its barycentric coordinates must match it. The relative error catches
    # degenerate faces, and the absolute one keeps the tiny faces of smooth geometries near convergence.
    polytope_max_rel_reprojection_error: qd.Tensor = field(float)
    polytope_max_abs_reprojection_error: qd.Tensor = field(float)
    # Alignment tolerances of the normals of a face-face and an edge-face parallel contact
    contact_face_tol: qd.Tensor = field(float)
    contact_edge_tol: qd.Tensor = field(float)
    # Tolerances of the differentiable contact: the distance along its normal from a face to the support point, zero on
    # the boundary of the Minkowski difference, the excess of the distance from the origin to the face over the EPA
    # depth, and the excursion of the affine coordinates of the projection of the origin outside [0, 1].
    # FIXME: Calibrate these values on real scenes.
    diff_contact_eps_boundary: qd.Tensor = field(float)
    diff_contact_eps_distance: qd.Tensor = field(float)
    diff_contact_eps_affine: qd.Tensor = field(float)
    # Minimum norm of a valid normal of the differentiable contact
    diff_contact_min_normal_norm: qd.Tensor = field(float)
    # Minimum penetration of a differentiable contact. It must exceed the rounding gap between the depths of the forward
    # and backward passes, since a depth the backward pass computes as zero yields a NaN normal.
    diff_contact_min_penetration: qd.Tensor = field(float)


# The states GJK runs on: one per env for the single-kernel narrowphase, and those of the passes of the split
# narrowphase, one per thread, which the collider opts in as it runs them
class GJKScratch(Data, kind=DataKind.SCRATCH):
    main: GJKState = field(opt_in=True)
    contact0: GJKState = field(opt_in=True)
    multicontact: GJKState = field(opt_in=True)


@qd.data_oriented
class GJKStaticConfig(StaticConfig):
    enable_contact_patch: bool


# =========================================== SupportField ===========================================


# The cells of the support tables and their candidate vertices (see SupportFieldInfo)
SUPPORT_CELLS = Size("support_cells")
SUPPORT_CANDIDATES = Size("support_candidates")


class SupportFieldInfo(Data, kind=DataKind.CONSTANT):
    # First cell of the table of each geom, the geoms of one trimesh sharing one table
    support_cell_start: qd.Tensor = field(int, shape=(GEOMS,))
    # First extra candidate of each cell, past those of its record, its extras ending where those of the next start
    support_list_start: qd.Tensor = field(int, shape=(SUPPORT_CELLS + 1,))
    # Mesh-frame position and index of each candidate vertex, among the vertices of its geom: the record of every cell
    # at twice the index of the cell, holding the first two candidates of the cell (a single candidate twice), then the
    # extra candidates of all cells
    support_v: qd.Tensor = field((float, 3), shape=(SUPPORT_CANDIDATES,))
    support_vid: qd.Tensor = field(int, shape=(SUPPORT_CANDIDATES,))
    support_res: qd.Tensor = field(int, shape=())


# =========================================== SDF ===========================================


# The nodes of the signed distance field (SDF) grids and of their coarse min-grids, all geoms together
SDF_CELLS = Size("sdf_cells")
SDF_COARSE_CELLS = Size("sdf_coarse_cells")


class SDFGeomInfo(Data, kind=DataKind.CONSTANT, shape=(GEOMS,)):
    T_mesh_to_sdf: qd.Tensor = field((float, (4, 4)))
    sdf_res: qd.Tensor = field((int, 3))
    sdf_max: qd.Tensor = field(float)
    sdf_cell_size: qd.Tensor = field((float, 3))
    sdf_cell_start: qd.Tensor = field(int)
    # Coarse grid of the block minima of the nodes, a lower bound of the trilinear signed distance
    sdf_coarse_res: qd.Tensor = field((int, 3))
    sdf_coarse_cell_start: qd.Tensor = field(int)


class SDFInfo(Data, kind=DataKind.CONSTANT, shape=(SDF_CELLS,)):
    geoms_info: SDFGeomInfo
    geoms_sdf_val: qd.Tensor = field(float)
    geoms_sdf_grad: qd.Tensor = field(
        (float, 3),
        oversize_hint="Consider setting a larger 'sdf_cell_size', or a smaller 'sdf_max_res', in 'gs.materials.Rigid' "
        "options.",
    )
    geoms_sdf_closest_vert: qd.Tensor = field(int)
    geoms_sdf_coarse_val: qd.Tensor = field(float, shape=(SDF_COARSE_CELLS,))


# =========================================== ColliderInfo ===========================================


VERT_NEIGHBORS = Size("vert_neighbors")
# The samples of the height field of the terrain
TERRAIN_ROWS = Size("terrain_rows")
TERRAIN_COLS = Size("terrain_cols")


class ColliderInfo(Data, kind=DataKind.CONSTANT):
    sdf: SDFInfo = requires(opt_in=True)
    mpr: MPRInfo = requires()
    gjk: GJKInfo = requires()
    support_field: SupportFieldInfo = requires(opt_in=True)
    vert_neighbors: qd.Tensor = field(int, shape=(VERT_NEIGHBORS,))
    vert_neighbor_start: qd.Tensor = field(int, shape=(VERTS,))
    vert_n_neighbors: qd.Tensor = field(int, shape=(VERTS,))
    verts_spatial_grid: VertsSpatialGrid
    # Index of the pair of geoms (i_ga, i_gb) among the possible pairs, -1 for an invalid pair
    collision_pair_idx: qd.Tensor = field(int, shape=(GEOMS, GEOMS))
    # Index of the pair of links (i_la, i_lb) in valid_link_pairs, -1 for a pair without a valid geom pair
    link_pair_idx: qd.Tensor = field(int, shape=(LINKS, LINKS))
    # Lower and upper corners of the box of every link in its own frame, bounding the boxes of its geoms
    links_init_aabb_min: qd.Tensor = field((float, 3), shape=(LINKS,))
    links_init_aabb_max: qd.Tensor = field((float, 3), shape=(LINKS,))
    max_possible_pairs: qd.Tensor = field(int, shape=())
    max_collision_pairs: qd.Tensor = field(int, shape=())
    max_candidate_contacts: qd.Tensor = field(int, shape=())
    max_collision_pairs_broad: qd.Tensor = field(int, shape=())
    # Contacts per environment after pruning, which sizes the contact rows of the constraint solve. Below
    # max_candidate_contacts under contact pruning or the option 'max_contacts'.
    max_contacts: qd.Tensor = field(int, shape=())
    # Link pairs holding at least one valid geom pair, the link carrying most geoms first (see func_broad_phase_geoms)
    n_valid_link_pairs: qd.Tensor = field(int, shape=())
    valid_link_pairs: qd.Tensor = field((int, 2), shape=(VALID_LINK_PAIRS,))
    terrain_hf: qd.Tensor = field(float, shape=(TERRAIN_ROWS, TERRAIN_COLS))
    terrain_rc: qd.Tensor = field(int, shape=(2,))
    terrain_scale: qd.Tensor = field(float, shape=(2,))
    terrain_xyz_maxmin: qd.Tensor = field(float, shape=(6,))
    # Perturbation and tolerance of the multi-contact detection
    mc_perturbation: qd.Tensor = field(float, shape=())
    mc_tolerance: qd.Tensor = field(float, shape=())
    # Tolerances of the differentiable contact
    diff_pos_tolerance: qd.Tensor = field(float, shape=())
    diff_normal_tolerance: qd.Tensor = field(float, shape=())
    # Link-pair contact pruning
    contact_pruning_tolerance: qd.Tensor = field(float, shape=())
    prune_deep_penetration_ratio: qd.Tensor = field(float, shape=())


# =========================================== DynInfo and DynState ===========================================


# Sizes of the elements of the rigid solver, and of the environments of the model parameters the options batch (see
# 'batch_dofs_info', 'batch_links_info' and 'batch_joints_info' in RigidOptions)
ENTITIES = Size("entities")
FACES = Size("faces")
EDGES = Size("edges")
FREE_VERTS = Size("free_verts")
FIXED_VERTS = Size("fixed_verts")
VVERTS = Size("vverts")
CUSTOM_VVERTS = Size("custom_vverts")
VFACES = Size("vfaces")
VGEOMS = Size("vgeoms")
CANDIDATE_EQUALITIES = Size("candidate_equalities")
DOFS_INFO_ENVS = Size("dofs_info_envs")
LINKS_INFO_ENVS = Size("links_info_envs")
JOINTS_INFO_ENVS = Size("joints_info_envs")
# The backward slots of a link, one per joint plus the link itself
LINK_BW_SLOTS = Size("link_bw_slots")

# The arrays the midpoint integration of free bodies reads (see func_midpoint_free_body)
MIDPOINT = Label("midpoint")


class DofsInfo(Data, kind=DataKind.INFO, shape=(DOFS, DOFS_INFO_ENVS)):
    entity_idx: qd.Tensor = field(int)
    stiffness: qd.Tensor = field(float)
    invweight: qd.Tensor = field(float)
    armature: qd.Tensor = field(float)
    damping: qd.Tensor = field(float)
    frictionloss: qd.Tensor = field(float)
    motion_ang: qd.Tensor = field((float, 3))
    motion_vel: qd.Tensor = field((float, 3))
    limit: qd.Tensor = field((float, 2))
    act_gain: qd.Tensor = field(float)
    act_bias: qd.Tensor = field((float, 3))
    force_range: qd.Tensor = field((float, 2))
    dof_length: qd.Tensor = field(float)


class DofsState(Data, kind=DataKind.DERIVED, shape=(DOFS, ENVS)):
    # A *_bw array keeps what the backward pass reads and the forward pass overwrites
    force: qd.Tensor = field(float, grad=True)
    qf_bias: qd.Tensor = field(float, grad=True)
    qf_passive: qd.Tensor = field(float, grad=True)
    qf_applied: qd.Tensor = field(float, grad=True)
    pos: qd.Tensor = field(float, grad=True)
    vel: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    vel_prev: qd.Tensor = field(float, kind=DataKind.SCRATCH, grad=True)
    vel_next: qd.Tensor = field(float, kind=DataKind.SCRATCH, grad=True)
    acc: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    acc_smooth: qd.Tensor = field(float, grad=True)
    acc_smooth_bw: qd.Tensor = field(float, shape=(2, DOFS, ENVS), labels=BACKWARD, grad=True)
    qf_smooth: qd.Tensor = field(float, grad=True)
    qf_constraint: qd.Tensor = field(float, grad=True)
    # Implicit damping (see func_implicit_damping in forward_dynamics.py): the implicit share h D qacc of the damping
    # force, D being the diagonal the damped mass factor adds, and the correction (M + hD)^-1 h D qacc subtracted from
    # the acceleration
    qf_damping_implicit: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    qacc_damping_implicit: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    cdof_ang: qd.Tensor = field((float, 3), grad=True)
    cdof_vel: qd.Tensor = field((float, 3), grad=True)
    cdofd_ang: qd.Tensor = field((float, 3), grad=True)
    cdofd_vel: qd.Tensor = field((float, 3), grad=True)
    f_vel: qd.Tensor = field((float, 3), grad=True)
    f_ang: qd.Tensor = field((float, 3), grad=True)
    ctrl_force: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    ctrl_pos: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    ctrl_vel: qd.Tensor = field(float, kind=DataKind.STATE, grad=True)
    ctrl_mode: qd.Tensor = field(int, kind=DataKind.STATE)
    is_hibernated: qd.Tensor = field(int, kind=DataKind.STATE)


class LinksInfo(Data, kind=DataKind.INFO, shape=(LINKS, LINKS_INFO_ENVS)):
    parent_idx: qd.Tensor = field(int)
    root_idx: qd.Tensor = field(int)
    q_start: qd.Tensor = field(int)
    dof_start: qd.Tensor = field(int)
    joint_start: qd.Tensor = field(int)
    q_end: qd.Tensor = field(int)
    dof_end: qd.Tensor = field(int)
    joint_end: qd.Tensor = field(int)
    n_dofs: qd.Tensor = field(int)
    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    invweight: qd.Tensor = field((float, 2))
    is_fixed: qd.Tensor = field(bool)
    inertial_pos: qd.Tensor = field((float, 3))
    inertial_quat: qd.Tensor = field((float, 4))
    inertial_i: qd.Tensor = field((float, (3, 3)))
    inertial_mass: qd.Tensor = field(float)
    entity_idx: qd.Tensor = field(int)
    # The geoms and visual geoms of each link, which heterogeneous environments vary
    geom_start: qd.Tensor = field(int)
    geom_end: qd.Tensor = field(int)
    vgeom_start: qd.Tensor = field(int)
    vgeom_end: qd.Tensor = field(int)


class LinksState(Data, kind=DataKind.DERIVED, shape=(LINKS, ENVS)):
    # A *_bw array keeps what the backward pass reads and the forward pass overwrites
    cinr_inertial: qd.Tensor = field((float, (3, 3)), grad=True)
    cinr_pos: qd.Tensor = field((float, 3), grad=True)
    cinr_quat: qd.Tensor = field((float, 4), grad=True)
    cinr_mass: qd.Tensor = field(float, grad=True)
    crb_inertial: qd.Tensor = field((float, (3, 3)), grad=True)
    crb_pos: qd.Tensor = field((float, 3), grad=True)
    crb_quat: qd.Tensor = field((float, 4), grad=True)
    crb_mass: qd.Tensor = field(float, grad=True)
    cdd_vel: qd.Tensor = field((float, 3), grad=True)
    cdd_ang: qd.Tensor = field((float, 3), grad=True)
    # The pose of a fixed root link is state, which forward kinematics leaves as set (see
    # func_forward_kinematics_link_bw), the poses of the other links being recomputed from the positions of the dofs.
    pos: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    quat: qd.Tensor = field((float, 4), kind=DataKind.STATE, grad=True)
    pos_bw: qd.Tensor = field((float, 3), shape=(LINKS, LINK_BW_SLOTS, ENVS), grad=True)
    quat_bw: qd.Tensor = field((float, 4), shape=(LINKS, LINK_BW_SLOTS, ENVS), grad=True)
    i_pos: qd.Tensor = field((float, 3), grad=True)
    i_pos_bw: qd.Tensor = field((float, 3), grad=True)
    i_quat: qd.Tensor = field((float, 4), grad=True)
    cd_ang: qd.Tensor = field((float, 3), grad=True)
    cd_vel: qd.Tensor = field((float, 3), grad=True)
    # Whether a contact or a connect or weld equality row involves the link this step, written by the constraint
    # assembly from the sorted contacts and read by the midpoint integration
    is_constrained: qd.Tensor = field(bool)
    # Midpoint angular velocity a free body eligible for midpoint integration converged to this substep, in its inertial
    # frame, and the residual of the midpoint equation there. The integration the backward pass reverses resumes from
    # them, since autodiff cannot reverse the Newton iteration (see func_midpoint_free_body).
    midpoint_ang: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, labels=MIDPOINT)
    midpoint_res: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH, labels=MIDPOINT)
    cd_ang_bw: qd.Tensor = field((float, 3), shape=(LINKS, LINK_BW_SLOTS, ENVS), grad=True)
    cd_vel_bw: qd.Tensor = field((float, 3), shape=(LINKS, LINK_BW_SLOTS, ENVS), grad=True)
    mass_sum: qd.Tensor = field(float, grad=True)
    # Center of mass of the kinematic tree
    root_COM: qd.Tensor = field((float, 3), grad=True)
    root_COM_bw: qd.Tensor = field((float, 3), grad=True)
    cacc_ang: qd.Tensor = field((float, 3), grad=True)
    cacc_lin: qd.Tensor = field((float, 3), grad=True)
    cfrc_ang: qd.Tensor = field((float, 3), grad=True)
    cfrc_vel: qd.Tensor = field((float, 3), grad=True)
    cfrc_applied_ang: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    cfrc_applied_vel: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    # The wrench the couplers accumulate during a substep, which the next substep consumes
    cfrc_coupling_ang: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    cfrc_coupling_vel: qd.Tensor = field((float, 3), kind=DataKind.STATE, grad=True)
    contact_force: qd.Tensor = field((float, 3), grad=True)
    # Hibernation: is_hibernated flags a sleeping link, awake_steps counts the consecutive substeps an awake link has
    # spent below the hibernation speed tolerance, up to hibernation_min_steps (see func_count_settled_step), zero
    # again the step it exceeds the tolerance or wakes.
    is_hibernated: qd.Tensor = field(int, kind=DataKind.STATE)
    awake_steps: qd.Tensor = field(int, kind=DataKind.STATE)


class JointsInfo(Data, kind=DataKind.INFO, shape=(JOINTS, JOINTS_INFO_ENVS)):
    type: qd.Tensor = field(int)
    sol_params: qd.Tensor = field((float, 7))
    q_start: qd.Tensor = field(int)
    dof_start: qd.Tensor = field(int)
    q_end: qd.Tensor = field(int)
    dof_end: qd.Tensor = field(int)
    n_dofs: qd.Tensor = field(int)
    pos: qd.Tensor = field((float, 3))


class JointsState(Data, kind=DataKind.DERIVED, shape=(JOINTS, ENVS)):
    xanchor: qd.Tensor = field((float, 3), grad=True)
    xaxis: qd.Tensor = field((float, 3), grad=True)


class GeomsInfo(Data, kind=DataKind.INFO, shape=(GEOMS,)):
    pos: qd.Tensor = field((float, 3))
    center: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    data: qd.Tensor = field((float, 7))
    link_idx: qd.Tensor = field(int)
    type: qd.Tensor = field(int)
    friction: qd.Tensor = field(float)
    friction_torsional: qd.Tensor = field(float)
    friction_rolling: qd.Tensor = field(float)
    sol_params: qd.Tensor = field((float, 7))
    vert_num: qd.Tensor = field(int)
    vert_start: qd.Tensor = field(int)
    vert_end: qd.Tensor = field(int)
    verts_state_start: qd.Tensor = field(int)
    verts_state_end: qd.Tensor = field(int)
    face_num: qd.Tensor = field(int)
    face_start: qd.Tensor = field(int)
    face_end: qd.Tensor = field(int)
    edge_num: qd.Tensor = field(int)
    edge_start: qd.Tensor = field(int)
    edge_end: qd.Tensor = field(int)
    is_convex: qd.Tensor = field(bool)
    contype: qd.Tensor = field(int)
    conaffinity: qd.Tensor = field(int)
    is_fixed: qd.Tensor = field(bool)
    is_decomposed: qd.Tensor = field(bool)
    is_hollow: qd.Tensor = field(bool)
    needs_coup: qd.Tensor = field(int)
    coup_friction: qd.Tensor = field(float)
    coup_softness: qd.Tensor = field(float)
    coup_restitution: qd.Tensor = field(float)


class GeomsState(Data, kind=DataKind.DERIVED, shape=(GEOMS, ENVS)):
    pos: qd.Tensor = field((float, 3), grad=True)
    quat: qd.Tensor = field((float, 4), grad=True)
    aabb_min: qd.Tensor = field((float, 3))
    aabb_max: qd.Tensor = field((float, 3))
    verts_updated: qd.Tensor = field(bool)
    is_hibernated: qd.Tensor = field(int, kind=DataKind.STATE)
    friction_ratio: qd.Tensor = field(float, kind=DataKind.INFO)


class VertsInfo(Data, kind=DataKind.CONSTANT, shape=(VERTS,)):
    init_pos: qd.Tensor = field((float, 3))
    init_normal: qd.Tensor = field((float, 3))
    geom_idx: qd.Tensor = field(int)
    init_center_pos: qd.Tensor = field((float, 3))
    verts_state_idx: qd.Tensor = field(int)
    is_fixed: qd.Tensor = field(bool)


class FacesInfo(Data, kind=DataKind.CONSTANT, shape=(FACES,)):
    verts_idx: qd.Tensor = field((int, 3))
    geom_idx: qd.Tensor = field(int)


class EdgesInfo(Data, kind=DataKind.CONSTANT, shape=(EDGES,)):
    v0: qd.Tensor = field(int)
    v1: qd.Tensor = field(int)
    length: qd.Tensor = field(float)


class VertsState(Data, kind=DataKind.DERIVED):
    pos: qd.Tensor = field((float, 3))


class VVertsInfo(Data, kind=DataKind.CONSTANT, shape=(VVERTS,)):
    init_pos: qd.Tensor = field((float, 3))
    init_vnormal: qd.Tensor = field((float, 3))
    vgeom_idx: qd.Tensor = field(int)
    vverts_state_idx: qd.Tensor = field(int)


class VVertsState(Data, kind=DataKind.DERIVED, shape=(CUSTOM_VVERTS, ENVS)):
    pos: qd.Tensor = field(
        (float, 3),
        oversize_hint="Consider opting fewer kinematic entities into 'enable_custom_vverts=True', or reducing "
        "'n_envs'.",
    )


class VFacesInfo(Data, kind=DataKind.CONSTANT, shape=(VFACES,)):
    vverts_idx: qd.Tensor = field((int, 3))
    vgeom_idx: qd.Tensor = field(int)


class VGeomsInfo(Data, kind=DataKind.CONSTANT, shape=(VGEOMS,)):
    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))
    link_idx: qd.Tensor = field(int)
    vvert_num: qd.Tensor = field(int)
    vvert_start: qd.Tensor = field(int)
    vvert_end: qd.Tensor = field(int)
    vface_num: qd.Tensor = field(int)
    vface_start: qd.Tensor = field(int)
    vface_end: qd.Tensor = field(int)
    color: qd.Tensor = field((float, 4))


class VGeomsState(Data, kind=DataKind.DERIVED, shape=(VGEOMS, ENVS)):
    pos: qd.Tensor = field((float, 3))
    quat: qd.Tensor = field((float, 4))


class EqualitiesInfo(Data, kind=DataKind.INFO, shape=(CANDIDATE_EQUALITIES, ENVS)):
    eq_obj1id: qd.Tensor = field(int)
    eq_obj2id: qd.Tensor = field(int)
    # The data of a screw constraint, the largest one, sizes the entries of all equalities (see func_screw_frame in
    # constraint/solver.py)
    eq_data: qd.Tensor = field((float, 18))
    eq_type: qd.Tensor = field(int)
    sol_params: qd.Tensor = field((float, 7))


class EntitiesInfo(Data, kind=DataKind.INFO, shape=(ENTITIES,)):
    dof_start: qd.Tensor = field(int)
    dof_end: qd.Tensor = field(int)
    n_dofs: qd.Tensor = field(int)
    link_start: qd.Tensor = field(int)
    link_end: qd.Tensor = field(int)
    n_links: qd.Tensor = field(int)
    geom_start: qd.Tensor = field(int)
    geom_end: qd.Tensor = field(int)
    n_geoms: qd.Tensor = field(int)
    gravity_compensation: qd.Tensor = field(float)
    is_local_collision_mask: qd.Tensor = field(bool)


class EntitiesState(Data, kind=DataKind.STATE, shape=(ENTITIES, ENVS)):
    is_hibernated: qd.Tensor = field(int)


class DynInfo(Data):
    """The model of the kinematics and dynamics, one struct per element type."""

    entities: EntitiesInfo
    links: LinksInfo
    joints: JointsInfo
    dofs: DofsInfo
    geoms: GeomsInfo = field(labels=DYNAMICS)
    verts: VertsInfo = field(labels=DYNAMICS)
    faces: FacesInfo = field(labels=DYNAMICS)
    edges: EdgesInfo = field(labels=DYNAMICS)
    vverts: VVertsInfo
    vfaces: VFacesInfo
    vgeoms: VGeomsInfo
    equalities: EqualitiesInfo = field(labels=DYNAMICS)


class DynState(Data):
    """The state of the kinematics and dynamics, one struct per element type."""

    entities: EntitiesState
    links: LinksState
    joints: JointsState
    dofs: DofsState
    geoms: GeomsState = field(labels=DYNAMICS)
    free_verts: VertsState = field(shape=(FREE_VERTS, ENVS), labels=DYNAMICS)
    fixed_verts: VertsState = field(shape=(FIXED_VERTS,), labels=DYNAMICS)
    vverts: VVertsState
    vgeoms: VGeomsState


# =================================== RigidSimStaticConfig ===================================


@qd.data_oriented
class RigidSimStaticConfig(StaticConfig):
    backend: int
    para_level: int
    enable_collision: bool
    use_hibernation: bool
    batch_links_info: bool
    batch_dofs_info: bool
    batch_joints_info: bool
    enable_mujoco_compatibility: bool
    enable_elliptic_friction: bool
    # Whether contact friction is bounded by the disc of radius mu * f_n, f_n being the normal force at the current
    # iterate, rather than jointly with it through the coupled cone (see gs.contact_resolution). It implies
    # enable_elliptic_friction, whose rows separate the normal force from the friction disc.
    enable_signorini_contact: bool
    enable_multi_contact: bool
    enable_joint_limit: bool
    box_box_detection: bool
    sparse_solve: bool
    integrator: int
    solver_type: int
    requires_grad: bool
    # -1 to let the dispatch pick, 0 or 1 to force it
    prefer_decomposed_solver: int = -1
    # Whether the Hessian without its cone blocks persists in the mirror slots of nt_H (see nt_H in ConstraintState)
    enable_cone_free_hessian_reuse: bool = False
    # Whether contacts carry torsional friction rows resisting relative spin about the contact normal: one extra
    # opposing pyramid pair per contact with the pyramidal cone, one extra cone row with the elliptic cone.
    enable_torsional_friction: bool = False
    # Whether contacts also carry rolling friction rows resisting relative rotation about the two tangent axes: two
    # extra opposing pyramid pairs per contact with the pyramidal cone, two extra cone rows with the elliptic cone.
    # It requires enable_torsional_friction, the rolling rows following the spin row.
    enable_rolling_friction: bool = False
    # Consecutive steps the dof velocities of a body must stay below the tolerance before it hibernates, so a body
    # momentarily slow (at the apex of a toss) stays awake
    hibernation_min_steps: int = 10
    # Whether the initialization of the solve runs over (constraint, env) pairs, for envs too few to fill the GPU
    parallel_init: bool = False
    # Whether the walks of the kinematic trees and roots sweep them level by level (see func_sweep_links_by_level)
    enable_level_sweep: bool = False
    # Lanes of the level sweep per env, a power of two dividing its blocks of 32 lanes. Four envs per block balance the
    # lanes a narrow level leaves idle against the links a wide level serializes.
    level_sweep_n_lanes_per_env: int = 8
    broadphase_traversal: int = 0
    enable_tiled_cholesky_mass_matrix: bool = False
    mass_matrix_fits_shared: bool = False
    # Register-tile width of the mass-matrix Cholesky kernels, 16 or 32 (see cholesky_tile_size_for)
    cholesky_tile_size: int = 32
    # Whether func_factor_mass factors the mass block of each entity in registers with the tiled Cholesky of the
    # Hessian, where the block exceeds shared memory. It requires every entity to be one mass block, and its tiles are
    # 32 wide, a block exceeding shared memory spanning well over 48 dofs.
    enable_register_tiled_mass: bool = False
    # Whether the shared-memory mass factor also solves the smooth acceleration against the factor it holds, after the
    # force passes, in place of func_compute_qacc. See its resolution in the rigid solver.
    enable_fused_smooth_acc_solve: bool = False
    # Whether func_solve_init seeds the factor of every island with the tiled per-island kernels, else one env per
    # thread (see has_per_env_solve_seed). See its resolution in the rigid solver.
    enable_tiled_island_seed: bool = False
    # Whether the scene holds one dof-carrying kinematic tree, an env then forming one island at most, whose passes read
    # the dof and row ranges of the env
    is_single_island: bool = False
    # Whether the seed kernel assembles the Hessian of a single-island scene too large for the cooperative factor, which
    # the monolith factors with the scalar dense Cholesky (see _kernel_solve_monolith)
    has_scalar_seed_factor: bool = False
    # Whether the constraint solver runs the subgroup-cooperative GPU kernels (linesearch refinement, friction rows,
    # mass-matrix assembly) on environment-first arrays (see COOPERATIVE_ROWS)
    enable_cooperative_constraint_kernels: bool = False
    # Whether the noslip sweep of an island runs on a block of 32 lanes: its rows are colored so that the rows of a
    # color touch disjoint mass blocks, and the lanes update the rows of a color in parallel, color after color (see
    # func_noslip in noslip.py). The visit order differs from the one-thread sweep, and so do the iterates. See its
    # resolution in the rigid solver.
    enable_cooperative_noslip: bool = False
    # Whether the arrays labeled COOPERATIVE_ROWS are stored environment-first, under the cooperative constraint
    # kernels or serialized execution. Loops whose order follows the memory order key on it, and the choice of the
    # cooperative algorithms on enable_cooperative_constraint_kernels.
    constraint_layout_batch_first: bool = False
    # Whether the Newton system is Jacobi-equilibrated (see nt_jacobi in ConstraintState)
    enable_jacobi_equilibration: bool = False
    tiled_n_dofs_per_block: int = -1
    # Dof caps of the shared tiles of the cooperative per-island factor and solve, one launch per cap (see
    # island_tile_caps): the doublings of the first cap below the last one. The first cap is the tile size doubled up
    # to the smallest dof-carrying tree, the last one the largest multiple of the tile size fitting GPU shared memory.
    # An island above the last cap factors in global memory.
    island_tile_cap_first: int = 0
    island_tile_cap_last: int = 0
    # Whether an island can hold more dofs than the last cap, which compiles the factor paths above it
    has_island_above_tile_cap: bool = False

    @property
    def rows_per_contact(self) -> int:
        """Constraint rows per contact.

        The pyramidal cone has two opposing edges per friction axis, the elliptic cone the normal row plus one row per
        friction axis. Torsional friction adds the spin axis, and rolling friction the two tangent axes.
        """
        n_extra_axes = int(self.enable_torsional_friction) + 2 * int(self.enable_rolling_friction)
        if self.enable_elliptic_friction:
            return 3 + n_extra_axes
        return 4 + 2 * n_extra_axes

    @property
    def hessian_rank_update_batch(self) -> int:
        """Rank-1 Cholesky updates the per-island incremental factor fuses into one column sweep.

        It sizes the nt_vec slots and the unroll of func_rank_batch_update_island, and covers the two slots per cone row
        of the coupled elliptic-cone update (see func_cone_rank_update_island).
        """
        return max(8, 2 * self.rows_per_contact)

    @property
    def has_per_env_solve_seed(self) -> bool:
        """Whether func_solve_init seeds the Newton factor, the gradient and the convergence test one env per thread.

        It holds on GPU wherever the tiled seed is off or leaves the factor to the scalar dense Cholesky (see
        has_scalar_seed_factor). The decomposed arm then factors one env per thread at every iteration too, its tiled
        factor maintaining the Hessian the tiled seed assembles.
        """
        return (
            self.solver_type == gs.constraint_solver.Newton
            and self.backend != gs.cpu
            and (not self.enable_tiled_island_seed or self.has_scalar_seed_factor)
        )


# =========================================== RigidData ===========================================


def cholesky_tile_size_for(n_dofs):
    """Return the tile size of the register Cholesky kernels for a system of dofs.

    It is 16 where one tile of 16 lanes holds the system, or where tiles of 32 lanes would leave most lanes idle (33 to
    48 dofs), and 32 otherwise.
    """
    return 16 if (n_dofs <= 16 or 32 < n_dofs <= 48) else 32


def cholesky_packed_tile_shape(max_dofs):
    """Return the shape of the shared tile of the tiled factor of a block of dofs.

    The tile stacks the blocks of the lower triangle, each a square of the tile size (see cholesky_tile_size_for)
    padded by one column against bank conflicts (see func_packed_tile_pos in solver.py). The sizes are native
    integers, as the shared array takes them.
    """
    tile_size = cholesky_tile_size_for(max_dofs)
    n_blocks = (max_dofs + tile_size - 1) // tile_size
    n_rows = min(tile_size, max_dofs)
    return int(n_blocks * (n_blocks + 1) // 2 * n_rows), int(n_rows + 1)


def island_tile_caps(cap_first, cap_last):
    """Return the ascending dof caps of the shared tiles of the per-island factor (see island_tile_cap_first)."""
    caps = []
    cap = cap_first
    while cap < cap_last:
        caps.append(cap)
        cap *= 2
    caps.append(cap_last)
    return tuple(caps)


class DataManager(Data):
    """Every array of a kinematic or rigid solver, allocated from the solver once its scene is built."""

    rigid_info: RigidInfo
    dyn_info: DynInfo
    dyn_state: DynState
    kinematics_scratch: KinematicsScratch
    errno: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.STATE)
    # Scratch of the inverse-weight computation, which borrows the buffers of the constraint solver at build time, so a
    # scene that never writes a mass, a center of mass or an inertia allocates none of it
    weight_scratch: WeightScratch = field(lazy=True)
    # The damped normal matrix of inverse kinematics is quadratic in RigidOptions.IK_max_targets, so a scene that never
    # solves inverse kinematics allocates none of it
    ik_scratch: IKScratch = field(lazy=True)

    def __init__(self, solver, kinematic_only: bool):
        is_dynamic = not kinematic_only
        config, options = solver.rigid_config, solver._options
        entities, n_targets = solver.entities, options.IK_max_targets
        entity_dofs = max([entity.n_dofs for entity in entities], default=0)
        sizes = dict(
            envs=solver._B,
            qs=solver.n_qs_,
            dofs=solver.n_dofs_,
            links=solver.n_links_,
            joints=solver.n_joints_,
            roots=solver.n_roots_,
            trees=solver.n_trees_,
            entities=solver.n_entities_,
            vverts=solver.n_vverts_,
            custom_vverts=solver.n_custom_vverts_,
            vfaces=solver.n_vfaces_,
            vgeoms=solver.n_vgeoms_,
            dofs_info_envs=solver._B if options.batch_dofs_info else (),
            links_info_envs=solver._B if options.batch_links_info else (),
            joints_info_envs=solver._B if options.batch_joints_info else (),
            link_bw_slots=max([link.n_joints for link in solver.links], default=0) + 1 if solver._requires_grad else 1,
            entity_dofs=entity_dofs,
            entity_qs=max([entity.n_qs for entity in entities], default=0),
            entity_links=max([entity.n_links for entity in entities], default=0),
            entity_joints=max([entity.n_joints for entity in entities], default=0),
            ik_targets=n_targets,
        )
        values = dict(EPS=gs.EPS)
        if is_dynamic:
            sizes.update(
                geoms=solver.n_geoms_,
                verts=solver.n_verts_,
                faces=solver.n_faces_,
                edges=solver.n_edges_,
                free_verts=solver.n_free_verts_,
                fixed_verts=solver.n_fixed_verts_,
                candidate_equalities=solver.n_candidate_equalities_,
                trees_levels_links=np.count_nonzero(solver._links_tree_root_idx >= 0),
                child_links=np.count_nonzero(solver._links_parent_idx >= 0),
            )
            values.update(
                substep_dt=solver._substep_dt,
                iterations=options.iterations,
                tolerance=options.tolerance,
                ls_iterations=options.ls_iterations,
                ls_tolerance=options.ls_tolerance,
                noslip_iterations=options.noslip_iterations,
                noslip_tolerance=options.noslip_tolerance,
                impratio=options.impratio,
                n_equalities=solver._n_equalities,
                n_candidate_equalities=solver.n_candidate_equalities_,
                hibernation_thresh_vel=solver._hibernation_thresh_vel,
            )
        super().__init__(
            sizes=sizes,
            values=dict(rigid_info=values),
            layout={
                DataManager.rigid_info.mass_mat: (2, 1, 0)
                if is_dynamic and config.enable_cooperative_constraint_kernels
                else None
            },
            unused=dict(
                dynamics=kinematic_only,
                level_sweep=not (is_dynamic and config.enable_level_sweep),
                tiled_mass=not (is_dynamic and config.enable_register_tiled_mass),
                midpoint=not (is_dynamic and config.integrator != gs.integrator.Euler),
                backward=not solver._requires_grad,
                kinematics=entity_dofs == 0,
            ),
            requires_grad=solver._requires_grad,
        )


# =========================================== RaycastResult ===========================================


class RaycastResult(Data, kind=DataKind.SCRATCH, shape=(ENVS,)):
    distance: qd.Tensor = field(float)
    geom_idx: qd.Tensor = field(int)
    hit_point: qd.Tensor = field((float, 3))
    normal: qd.Tensor = field((float, 3))


GeomsInitAABB = qd.Tensor


# =========================================== Particle solvers ===========================================


# Sizes of the particle solvers: the particles, the slots of the hash grid and the support particles of a visual vertex
PARTICLES = Size("particles")
HASH_SLOTS = Size("hash_slots")
VVERT_SUPPORTS = Size("vvert_supports")


class SpatialHashGrid(Data, kind=DataKind.DERIVED, shape=(HASH_SLOTS, ENVS)):
    """
    The particles of every environment sorted by the cell of a uniform grid they lie in, for neighbour queries.

    A cell of integer coordinates hashes to one of the slots, whose particles take the indices from `slot_start` to
    `slot_start + slot_size` once sorted (see func_hash_compute_reordered_idx in genesis/utils/geom.py).
    """

    slot_size: qd.Tensor = field(int)
    slot_start: qd.Tensor = field(int)
    cur_cnt: qd.Tensor = field(int, shape=(ENVS,), kind=DataKind.SCRATCH)
    cell_size: qd.Tensor = field(float, shape=(), kind=DataKind.CONSTANT)
    grid_res: qd.Tensor = field((int, 3), shape=(), kind=DataKind.CONSTANT)


class CubeBoundaryBox(Data, kind=DataKind.CONSTANT, shape=()):
    """An axis-aligned box confining the particles, which bounce off its walls with the given restitution."""

    lower: qd.Tensor = field((float, 3))
    upper: qd.Tensor = field((float, 3))
    restitution: qd.Tensor = field(float)


class Particles(Data, kind=DataKind.STATE, shape=(PARTICLES, ENVS)):
    """
    Runtime state shared by the particles of every particle-based solver, one value per particle and environment.

    The struct of each solver extends it with its own fields, and the accessors of 'ParticleSolver' take any of them.
    """

    pos: qd.Tensor = field((float, 3))
    vel: qd.Tensor = field((float, 3))
    is_active: qd.Tensor = field(bool)


class ParticlesParams(Data, kind=DataKind.CONSTANT, shape=(PARTICLES,)):
    """Model parameters shared by the particles of every particle-based solver, one value per particle."""

    mass: qd.Tensor = field(float)


class ParticlesRender(Data, kind=DataKind.SCRATCH, shape=(PARTICLES, ENVS)):
    """The particles as drawn: their position and velocity, an inactive particle moved out of sight."""

    pos: qd.Tensor = field((float, 3))
    vel: qd.Tensor = field((float, 3))
    is_active: qd.Tensor = field(bool)


class VVertsSupport(Data, kind=DataKind.CONSTANT, shape=(VVERTS,)):
    """The visual vertices skinned by particles: each one the weighted sum of the positions of its support particles."""

    support_idxs: qd.Tensor = field((int, VVERT_SUPPORTS))
    support_weights: qd.Tensor = field((float, VVERT_SUPPORTS))


class VVertsRender(Data, kind=DataKind.SCRATCH, shape=(VVERTS, ENVS)):
    """The visual vertices as drawn, active when their first support particle is."""

    pos: qd.Tensor = field((float, 3))
    is_active: qd.Tensor = field(bool)


# =========================================== PBDSolver ===========================================


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


# =========================================== SPHSolver ===========================================


class SPHParticles(Particles):
    """
    Runtime state of the SPH particles, one value per particle and environment.

    `p` is the pressure, `dfsph_factor` the factor of the divergence-free and density solves of divergence-free smoothed
    particle hydrodynamics (DFSPH), and `drho` the rate of change of the density they correct. `reordered_idx` is the
    index of the particle once sorted by hash grid cell.
    """

    acc: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH)
    rho: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    p: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    dfsph_factor: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    drho: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    reordered_idx: qd.Tensor = field(int, kind=DataKind.DERIVED)


class SPHParticlesParams(ParticlesParams):
    """
    Model parameters of the SPH particles, one value per particle (and per environment once sorted).

    `rho` is the rest density, `stiffness` and `exponent` the parameters of the equation of state, `mu` the viscosity
    and `gamma` the surface tension.
    """

    rho: qd.Tensor = field(float)
    stiffness: qd.Tensor = field(float)
    exponent: qd.Tensor = field(float)
    mu: qd.Tensor = field(float)
    gamma: qd.Tensor = field(float)


class SPHState(Data, kind=DataKind.STATE):
    """
    Runtime state of the SPH solver.

    The substep runs on the copies of the particles sorted by hash grid cell, `particles_reordered` and
    `particles_info_reordered`, which the particles take back at its end.
    """

    particles: SPHParticles
    particles_reordered: SPHParticles = field(kind=DataKind.SCRATCH)
    particles_info_reordered: SPHParticlesParams = field(shape=(PARTICLES, ENVS), kind=DataKind.SCRATCH)
    hash_grid: SpatialHashGrid
    particles_render: ParticlesRender


class SPHInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the SPH solver.

    `support_radius` is the radius of the smoothing kernels, `particle_volume` the volume a particle stands for, and
    `density0` the rest density the incompressibility solves of divergence-free smoothed particle hydrodynamics (DFSPH)
    target.
    """

    particles: SPHParticlesParams
    boundary: CubeBoundaryBox
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float, shape=())
    particle_size: qd.Tensor = field(float, shape=())
    particle_volume: qd.Tensor = field(float, shape=())
    support_radius: qd.Tensor = field(float, shape=())
    density0: qd.Tensor = field(float, shape=())


# =========================================== SFSolver ===========================================


# Sizes of the stable fluid: the cells along each axis of its grid, and its jets
SF_GRID = Size("sf_grid")
JETS = Size("jets")


class SFState(Data, kind=DataKind.STATE, shape=(SF_GRID, SF_GRID, SF_GRID)):
    """
    Runtime state of the stable fluid, one value per cell of its grid.

    `smoke` holds the concentration of the smoke of each jet. `pressure` holds the two buffers the Jacobi iterations of
    the pressure projection alternate between, the first one keeping the pressure of the last substep as warm start.
    """

    vel: qd.Tensor = field((float, 3))
    vel_advected: qd.Tensor = field((float, 3), kind=DataKind.SCRATCH)
    div: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    pressure: qd.Tensor = field(float, shape=(2, SF_GRID, SF_GRID, SF_GRID))
    smoke: qd.Tensor = field((float, JETS))


class SFInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """Model parameters of the stable fluid: the jets inject their smoke at speed `inlet_s`, which `decay` fades."""

    substep_dt: qd.Tensor = field(float)
    inlet_s: qd.Tensor = field(float)
    decay: qd.Tensor = field(float)


# =========================================== MPMSolver ===========================================


# The frames of a step the differentiable MPM and FEM solvers keep, and the cells of the grid of the material point
# method (MPM) solver along each axis
FRAMES = Size("frames")
GRID_X = Size("grid_x")
GRID_Y = Size("grid_y")
GRID_Z = Size("grid_z")
# The scratch of the sparse reset of the grid, without gradient alone (see MPMGrid)
SPARSE_GRID_RESET = Label("sparse_grid_reset")
# The thin rigid geoms between the particles and the cells of their stencil (see MPMCPIC)
CPIC = Label("cpic")


class MPMParticles(Data, kind=DataKind.STATE, shape=(FRAMES, PARTICLES, ENVS)):
    """
    Runtime state of the material point method (MPM) particles, one value per frame of a step, particle and environment.

    `C` is the affine velocity field, `F` the deformation gradient, `Jp` the plastic volume ratio and `actu` the
    actuation of the muscles. `F_tmp` is the deformation gradient advected over the substep, before the plasticity of
    the material applies, and `U`, `S`, `V` its singular value decomposition.
    """

    pos: qd.Tensor = field((float, 3), grad=True)
    vel: qd.Tensor = field((float, 3), grad=True)
    C: qd.Tensor = field((float, (3, 3)), grad=True)
    F: qd.Tensor = field((float, (3, 3)), grad=True)
    F_tmp: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    U: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    V: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    S: qd.Tensor = field((float, (3, 3)), kind=DataKind.SCRATCH, grad=True)
    actu: qd.Tensor = field(float, grad=True)
    Jp: qd.Tensor = field(float, grad=True)
    is_active: qd.Tensor = field(bool)


class MPMParticlesParams(Data, kind=DataKind.INFO, shape=(PARTICLES,)):
    """
    Model parameters of the material point method (MPM) particles, one value per particle.

    `default_Jp` is the plastic volume ratio a particle takes back when its position is set, and `is_free` whether it
    moves or acts as a boundary condition of the grid.
    """

    material_idx: qd.Tensor = field(int, kind=DataKind.CONSTANT)
    mass: qd.Tensor = field(float, kind=DataKind.CONSTANT)
    default_Jp: qd.Tensor = field(float, kind=DataKind.CONSTANT)
    is_free: qd.Tensor = field(bool)
    muscle_group: qd.Tensor = field(int)
    muscle_direction: qd.Tensor = field((float, 3))


class MPMGrid(Data, kind=DataKind.SCRATCH, shape=(FRAMES - 1, GRID_X, GRID_Y, GRID_Z, ENVS)):
    """
    The background grid of the material point method (MPM), one value per frame of a step, cell and environment.

    `vel_in` holds the momentum the particles scatter, and `vel_out` the velocity they gather back once the grid is
    updated. Without gradient, the grid of a substep is zeroed right after use (the backward pass reads the grid of
    every substep back), the cells to zero being listed once across the environments: `dirty_flag` marks a listed cell,
    and `dirty_list` holds the flat indices of the `dirty_count` listed cells, at most one per cell of the grid.
    """

    mass: qd.Tensor = field(float, grad=True)
    vel_in: qd.Tensor = field((float, 3), grad=True)
    vel_out: qd.Tensor = field((float, 3), grad=True)
    dirty_flag: qd.Tensor = field(int, shape=(GRID_X, GRID_Y, GRID_Z), labels=SPARSE_GRID_RESET)
    dirty_list: qd.Tensor = field(int, shape=(GRID_X * GRID_Y * GRID_Z,), labels=SPARSE_GRID_RESET)
    dirty_count: qd.Tensor = field(int, shape=(1,), labels=SPARSE_GRID_RESET)


class MPMCPIC(Data, kind=DataKind.STATE):
    """
    The thin rigid geoms separating each particle from the cells of its stencil, for Compatible Particle-In-Cell (CPIC).

    `rigid_normal` is the last outward normal of each geom seen by each particle, and `cells_geom_idx` the geom
    separating each particle from each cell of its stencil, -1 for none.
    """

    rigid_normal: qd.Tensor = field((float, 3), shape=(PARTICLES, GEOMS, ENVS), labels=CPIC)
    cells_geom_idx: qd.Tensor = field(int, shape=(PARTICLES, 3, 3, 3, ENVS), kind=DataKind.SCRATCH, labels=CPIC)


class MPMState(Data, kind=DataKind.STATE):
    """Runtime state of the material point method (MPM) solver."""

    particles: MPMParticles
    grid: MPMGrid
    cpic: MPMCPIC
    particles_render: ParticlesRender
    vverts_render: VVertsRender
    # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
    errno: qd.Tensor = field(int, shape=(ENVS,))


class MPMParticleConstraintsInfo(Data, kind=DataKind.INFO, shape=(PARTICLES, ENVS)):
    """
    The critically damped springs pulling particles toward targets, one per particle and environment.

    A particle attached to a link (`link_idx` other than -1) follows the point `link_local_pos` of the frame of the
    link, the others the fixed point `target_pos`.
    """

    is_constrained: qd.Tensor = field(bool)
    target_pos: qd.Tensor = field((float, 3))
    stiffness: qd.Tensor = field(float)
    link_idx: qd.Tensor = field(int)
    link_local_pos: qd.Tensor = field((float, 3))


class MPMInfo(Data, kind=DataKind.CONSTANT):
    """
    Model parameters of the material point method (MPM) solver.

    The grid spans the cells from `grid_offset` on, of size `dx`. `particle_volume` is the volume a particle stands
    for, scaled up by `particle_volume_scale` for numerical robustness, which the masses exchanged with the other
    solvers divide out.
    """

    particles: MPMParticlesParams
    vverts: VVertsSupport
    boundary: CubeBoundaryBox
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float, shape=())
    dx: qd.Tensor = field(float, shape=())
    inv_dx: qd.Tensor = field(float, shape=())
    grid_offset: qd.Tensor = field((int, 3), shape=())
    particle_volume: qd.Tensor = field(float, shape=())
    particle_volume_scale: qd.Tensor = field(float, shape=())


class MPMStaticConfig(StaticConfig):
    # Whether the singular value decomposition of the deformation gradient is computed, which a material reading it
    # and the backward pass require. Otherwise J is the determinant of the advected deformation gradient.
    is_svd_needed: bool
    # Whether the particles and the cells of their stencil are checked for a thin rigid geom between them (see MPMCPIC)
    enable_CPIC: bool
    # Whether the grid of every substep is kept for the backward pass, instead of being zeroed sparsely (see MPMGrid)
    requires_grad: bool


# =========================================== Fixed-order sums ===========================================


# Most chunks the terms of the items of one environment split into for a fixed-order sum (see FixedOrderSums). It
# bounds the serial sum over the partial sums that closes the sum, while leaving enough threads to the chunks for many
# thousand items.
SUM_MAX_CHUNKS = 256


# The items of a fixed-order sum, its sums, and the chunks of items it sums apart (see FixedOrderSums)
SUM_ITEMS = Size("sum_items")
SUMS = Size("sums")
SUM_CHUNKS = Size("sum_chunks")


class FixedOrderSums(Data, kind=DataKind.SCRATCH):
    """
    The terms of fixed-order sums over the items of each environment, and their partial sums.

    A sum runs in three steps, so that it is bitwise reproducible whatever the scheduling of the threads. Each item
    writes its term of sum k into `terms[i_item, k, i_b]`, func_reduce_terms in genesis/utils/linalg.py sums the terms
    over contiguous chunks of the items into `partials[i_chunk, k, i_b]`, and func_sum_chunks sums the partials of an
    environment in order. Items no more numerous than SUM_MAX_CHUNKS are their own chunks.
    """

    terms: qd.Tensor = field(float, shape=(SUM_ITEMS, SUMS, ENVS))
    partials: qd.Tensor = field(float, shape=(SUM_CHUNKS, SUMS, ENVS))

    @staticmethod
    def sizes(n_items: int, n_sums: int) -> dict[str, int]:
        """Return the sizes of the sums over some items, under the name of their member (see 'sizes' in Data)."""
        return dict(sum_items=n_items, sums=n_sums, sum_chunks=min(n_items, SUM_MAX_CHUNKS))


# =========================================== FEMSolver ===========================================


# Sizes of the FEM solver: the vertices and the elements, the frames of the element contributions of the explicit
# solver (one per substep with gradients), the corners of the elements, and the surface triangles with their vertices
# and elements, its states spanning the frames of a step (see FRAMES)
VERTICES = Size("vertices")
ELEMENTS = Size("elements")
CORNER_FRAMES = Size("corner_frames")
CORNERS = Size("corners")
SURFACES = Size("surfaces")
SURFACE_VERTICES = Size("surface_vertices")
SURFACE_ELEMENTS = Size("surface_elements")


class FloorBoundaryPlane(Data, kind=DataKind.CONSTANT, shape=()):
    """The floor of the FEM solver, at a height and with a restitution (see FloorBoundary in boundaries.py)."""

    height: qd.Tensor = field(float)
    restitution: qd.Tensor = field(float)


class FEMVerticesKinematics(Data, kind=DataKind.STATE, shape=(FRAMES, VERTICES, ENVS)):
    """The position and velocity of the vertices, one per frame of the substeps of a step, vertex and environment."""

    pos: qd.Tensor = field((float, 3), grad=True)
    vel: qd.Tensor = field((float, 3), grad=True)


class FEMElements(Data, kind=DataKind.CONSTANT, shape=(ELEMENTS,)):
    """
    Model parameters of the elements, one value per element.

    `el2v` holds the four vertices of an element, `B` the inverse of its rest shape matrix, `V` its rest volume and
    `mu`, `lam` its Lame parameters. The masses and volumes `*_scaled` are scaled up by `vol_scale` (see FEMInfo).
    `material_idx` is the index of the material (see FEMMaterialDispatcher in fem_solver.py).
    """

    el2v: qd.Tensor = field((int, 4))
    mu: qd.Tensor = field(float)
    lam: qd.Tensor = field(float)
    mass_scaled: qd.Tensor = field(float)
    material_idx: qd.Tensor = field(int)
    B: qd.Tensor = field((float, (3, 3)))
    V: qd.Tensor = field(float)
    V_scaled: qd.Tensor = field(float)
    friction_mu: qd.Tensor = field(float)
    muscle_group: qd.Tensor = field(int, kind=DataKind.INFO)
    muscle_direction: qd.Tensor = field((float, 3), kind=DataKind.INFO)


class FEMVerticesProperties(Data, kind=DataKind.CONSTANT, shape=(VERTICES,)):
    """
    Model parameters of the vertices.

    `mass`, `mass_inv`, `mass_over_dt2` and `friction_mu` hold one value per vertex, the mass of a vertex lumping a
    quarter of the mass of the elements it belongs to. `corner` lists the corners of the elements vertex after vertex,
    each one encoded as i_e * 4 + i_corner in increasing order, the corners touching vertex i_v being
    `corner[corner_start[i_v]:corner_start[i_v + 1]]`.
    """

    mass: qd.Tensor = field(float)
    mass_inv: qd.Tensor = field(float)
    mass_over_dt2: qd.Tensor = field(float)
    friction_mu: qd.Tensor = field(float)
    corner_start: qd.Tensor = field(int, shape=(VERTICES + 1,))
    corner: qd.Tensor = field(int, shape=(CORNERS,))


class FEMSurface(Data, kind=DataKind.CONSTANT):
    """
    The surface of the entities, which the other solvers couple with.

    `tri2v` holds the vertices of each surface triangle and `tri2el` its element. `vertices_idx` lists the vertices of
    the surface triangles, `vertices_mass` their lumped surface mass, and `elements_idx` the elements touching them.
    """

    tri2v: qd.Tensor = field((int, 3), shape=(SURFACES,))
    tri2el: qd.Tensor = field(int, shape=(SURFACES,))
    vertices_idx: qd.Tensor = field(int, shape=(SURFACE_VERTICES,))
    vertices_mass: qd.Tensor = field(float, shape=(SURFACE_VERTICES,))
    elements_idx: qd.Tensor = field(int, shape=(SURFACE_ELEMENTS,))


class FEMVVerts(Data, kind=DataKind.CONSTANT):
    """The visual vertices, each one drawn at the position of the simulated vertex `vert_idx`, and their faces."""

    vert_idx: qd.Tensor = field(int, shape=(VVERTS,))
    uvs: qd.Tensor = field((float, 2), shape=(VVERTS,))
    vfaces_indices: qd.Tensor = field((int, 3), shape=(VFACES,))


class FEMExplicitCorners(Data, kind=DataKind.SCRATCH, shape=(CORNER_FRAMES, ELEMENTS, 4, ENVS)):
    """
    The contribution of each element to each of its four vertices, for the explicit solver.

    A vertex sums the contributions of its corners in the order of FEMVerticesProperties, so that the result does not
    depend on the scheduling of the threads. With gradients, every substep of a step writes its own, which the
    backward pass reads back.
    """

    force: qd.Tensor = field((float, 3), grad=True)
    dv: qd.Tensor = field((float, 3), grad=True)


class FEMImplicitCorners(Data, kind=DataKind.SCRATCH, shape=(ELEMENTS, 4, ENVS)):
    """The contribution of each element to the force, the diagonal block of the Hessian and A * p of its vertices."""

    force: qd.Tensor = field((float, 3))
    diag3x3: qd.Tensor = field((float, (3, 3)))
    Ap: qd.Tensor = field((float, 3))


class FEMNewtonScratch(Data, kind=DataKind.SCRATCH, shape=(VERTICES, ENVS)):
    """
    The scratch of the Newton solve of the implicit solver, its conjugate gradient and its line search.

    Per element: the elastic `energy` density, its `gradient` with respect to the deformation gradient, the blocks
    `hessian[i, j]` of its Hessian, and the `rotation` the linear corotated materials linearize around. Per vertex: the
    `inertia` target, the `diag3x3` block of the Hessian and its inverse `prec`, the conjugate gradient vectors and the
    position `x_prev` the line search starts from. Per environment: whether the conjugate gradient and the line search
    are still running, and their scalars.

    The Hessian of a material whose Hessian does not depend on the deformation is written once and read by every later
    solve, hence its kind.
    """

    energy: qd.Tensor = field(float, shape=(ELEMENTS, ENVS))
    gradient: qd.Tensor = field((float, (3, 3)), shape=(ELEMENTS, ENVS))
    hessian: qd.Tensor = field((float, (3, 3)), shape=(3, 3, ELEMENTS, ENVS), kind=DataKind.STATE)
    rotation: qd.Tensor = field((float, (3, 3)), shape=(ELEMENTS, ENVS))
    inertia: qd.Tensor = field((float, 3))
    diag3x3: qd.Tensor = field((float, (3, 3)))
    prec: qd.Tensor = field((float, (3, 3)))
    x: qd.Tensor = field((float, 3))
    r: qd.Tensor = field((float, 3))
    z: qd.Tensor = field((float, 3))
    p: qd.Tensor = field((float, 3))
    Ap: qd.Tensor = field((float, 3))
    x_prev: qd.Tensor = field((float, 3))
    is_pcg_active: qd.Tensor = field(bool, shape=(ENVS,))
    is_linesearch_active: qd.Tensor = field(bool, shape=(ENVS,))
    rTr: qd.Tensor = field(float, shape=(ENVS,))
    rTz: qd.Tensor = field(float, shape=(ENVS,))
    rTr_new: qd.Tensor = field(float, shape=(ENVS,))
    rTz_new: qd.Tensor = field(float, shape=(ENVS,))
    pTAp: qd.Tensor = field(float, shape=(ENVS,))
    alpha: qd.Tensor = field(float, shape=(ENVS,))
    beta: qd.Tensor = field(float, shape=(ENVS,))
    prev_energy: qd.Tensor = field(float, shape=(ENVS,))
    linesearch_energy: qd.Tensor = field(float, shape=(ENVS,))
    step_size: qd.Tensor = field(float, shape=(ENVS,))
    m: qd.Tensor = field(float, shape=(ENVS,))
    corners: FEMImplicitCorners
    sums_v: FixedOrderSums
    sums_el: FixedOrderSums


class FEMState(Data, kind=DataKind.STATE):
    """
    Runtime state of the FEM solver.

    `vertices` and `actu`, the actuation of each element, hold one frame per substep of a step. `force` is the internal
    force of each vertex, which the last substep computed.
    """

    vertices: FEMVerticesKinematics
    actu: qd.Tensor = field(float, shape=(FRAMES, ELEMENTS, ENVS), grad=True)
    force: qd.Tensor = field((float, 3), shape=(VERTICES, ENVS), kind=DataKind.DERIVED)
    explicit_corners: FEMExplicitCorners
    newton: FEMNewtonScratch = field(kind=DataKind.SCRATCH)
    vverts_render: qd.Tensor = field((float, 3), shape=(VVERTS, ENVS), kind=DataKind.SCRATCH)


class FEMVertexConstraintsInfo(Data, kind=DataKind.INFO, shape=(VERTICES, ENVS)):
    """
    The constraints of the vertices, one per vertex and environment.

    A constrained vertex is held at `target_pos`, or pulled toward it by a critically damped spring of the given
    stiffness when the constraint is soft. A vertex following the link `link_idx` (-1 for none) keeps the offset
    `link_offset_pos` from it, rotated by the rotation of the link since `link_init_quat`.
    """

    is_constrained: qd.Tensor = field(bool)
    target_pos: qd.Tensor = field((float, 3))
    is_soft_constraint: qd.Tensor = field(bool)
    stiffness: qd.Tensor = field(float)
    link_idx: qd.Tensor = field(int)
    link_offset_pos: qd.Tensor = field((float, 3))
    link_init_quat: qd.Tensor = field((float, 4))


class FEMInfo(Data, kind=DataKind.CONSTANT, shape=()):
    """
    Model parameters of the FEM solver.

    `damping` damps the velocity of the explicit solver, and `damping_alpha`, `damping_beta` are the Rayleigh damping
    coefficients of the implicit one, proportional to the mass and to the stiffness. `vol_scale` scales the volumes up
    for numerical robustness, which the masses exchanged with the other solvers divide out.
    """

    elements: FEMElements
    vertices: FEMVerticesProperties
    surface: FEMSurface
    vverts: FEMVVerts
    boundary: FloorBoundaryPlane
    gravity: qd.Tensor = field((float, 3), shape=(ENVS,), kind=DataKind.INFO)
    substep_dt: qd.Tensor = field(float)
    damping: qd.Tensor = field(float)
    damping_alpha: qd.Tensor = field(float)
    damping_beta: qd.Tensor = field(float)
    pcg_threshold: qd.Tensor = field(float)
    linesearch_c: qd.Tensor = field(float)
    linesearch_tau: qd.Tensor = field(float)
    vol_scale: qd.Tensor = field(float)
    EPS: qd.Tensor = field(float)


class FEMStaticConfig(StaticConfig):
    # Whether the vertices of the implicit solver may be constrained, which drops their rows from its Newton solve
    enable_vertex_constraints: bool
    # Whether a material linearizes its energy around the rotation of the deformation gradient of each element
    has_rotation: bool
    # Whether every substep of a step keeps its own element contributions (see FEMExplicitCorners)
    requires_grad: bool


# =========================================== LegacyCoupler ===========================================


# The particles of the position based dynamics (PBD) solver a rigid link carries (see LegacyCouplerInfo)
RIGID_PBD = Label("rigid_pbd")


class LegacyCouplerInfo(Data, kind=DataKind.CONSTANT, shape=(PARTICLES, ENVS)):
    """
    Model parameters of the legacy coupler.

    A material point method (MPM) cell exchanges momentum with the smoothed particle hydrodynamics (SPH) and position
    based dynamics (PBD) particles of the cube of `mpm_sph_stencil_size` and `mpm_pbd_stencil_size` cells of their hash
    grid around its lower corner. A PBD particle carried by the link `pbd_attach_link_idx` (-1 for none) stays at the
    point `pbd_attach_local_pos` of the frame of the link.
    """

    mpm_sph_stencil_size: qd.Tensor = field(int, shape=())
    mpm_pbd_stencil_size: qd.Tensor = field(int, shape=())
    pbd_attach_link_idx: qd.Tensor = field(int, kind=DataKind.INFO, labels=RIGID_PBD)
    pbd_attach_local_pos: qd.Tensor = field((float, 3), kind=DataKind.INFO, labels=RIGID_PBD)
    EPS: qd.Tensor = field(float, shape=())


class LegacyCouplerStaticConfig(StaticConfig):
    # The pairs of solvers coupled with each other
    enable_rigid_mpm: bool
    enable_rigid_sph: bool
    enable_rigid_pbd: bool
    enable_rigid_fem: bool
    enable_mpm_sph: bool
    enable_mpm_pbd: bool
    enable_fem_mpm: bool
    enable_fem_sph: bool


# =========================================== HybridEntity ===========================================


# The muscle groups of the soft part of a hybrid entity
GROUPS = Size("groups")


class HybridSoftPartInfo(Data, kind=DataKind.CONSTANT, shape=(GROUPS,)):
    """
    The binding of the particles of the soft part of a hybrid entity to the links of its rigid part.

    The particles of each muscle group follow the geom `groups_geom_idx` of the link `groups_link_idx`. The rest
    position `particles_init_pos` of each particle is expressed in the frame `groups_frame_pos`, `groups_frame_quat` of
    its group.
    """

    groups_link_idx: qd.Tensor = field(int)
    groups_geom_idx: qd.Tensor = field(int)
    groups_frame_pos: qd.Tensor = field((float, 3))
    groups_frame_quat: qd.Tensor = field((float, 4))
    particles_init_pos: qd.Tensor = field((float, 3), shape=(PARTICLES,))


# ======================================= SEGMENT SORT =======================================


class SEGMENT_SORT_KIND(IntEnum):
    """The sort ordering the keys of a segment sort.

    See genesis.utils.sort.
    """

    # The device-wide radix sort of quadrants over every key of the set (GPU, large segments)
    DEVICE_RADIX = 0
    # The rank of each key among the keys of its segment (small segments)
    RANK = 1
    # A radix sort per segment, its chunks in parallel (CPU, large segments)
    SEGMENT_RADIX = 2


@qd.data_oriented
class SegmentSortStaticConfig(StaticConfig):
    """The compile-time shape of a sort of 64-bit keys per segment.

    See genesis.utils.sort.
    """

    # The bits a sort must order, a multiple of 16 up to 64
    end_bit: int
    # Scan depth of the device-wide radix sort (see quadrants.algorithms.sort), 1 to 8
    log256_max_n: int
    # SEGMENT_SORT_KIND
    sort_kind: int
    # Whether each key carries a value the sort moves along
    has_values: bool


# Sizes of a sort per segment: its segments, its keys, the slots of the device-wide radix sort and the chunks of the
# radix sort per segment
SORT_SEGMENTS = Size("sort_segments")
SORT_KEYS = Size("sort_keys")
SORT_SLOTS = Size("sort_slots")
SORT_CHUNKS = Size("sort_chunks")
# The values moved along the keys, and the scratch of each sort (see SEGMENT_SORT_KIND)
SORT_VALUES = Label("sort_values")
DEVICE_RADIX_SORT = Label("device_radix_sort")
SEGMENT_RADIX_SORT = Label("segment_radix_sort")


class SegmentSortBuffer(Data, kind=DataKind.SCRATCH, shape=(SORT_KEYS,)):
    """The keys of a sort per segment, their values, the segment bounds and the scratch of the sort.

    See genesis.utils.sort. The first n_keys keys are sorted, segment i_s holding the keys from segments_start[i_s] to
    segments_start[i_s + 1].
    """

    keys: qd.Tensor = field(qd.u64)
    values: qd.Tensor = field(int, labels=SORT_VALUES)
    keys_scratch: qd.Tensor = field(qd.u64)
    values_scratch: qd.Tensor = field(int, labels=SORT_VALUES)
    segments_start: qd.Tensor = field(int, shape=(SORT_SEGMENTS + 1,))
    sort_scratch: qd.Tensor = field(qd.u32, shape=(SORT_SLOTS,), labels=DEVICE_RADIX_SORT)
    sort_hist: qd.Tensor = field(int, shape=(SORT_SEGMENTS, SORT_CHUNKS, 256), labels=SEGMENT_RADIX_SORT)
    n_keys: qd.Tensor = field(int, shape=())


# =========================================== BVH ===========================================


class BVH_FIT_KIND(IntEnum):
    """The bottom-up fit of the node boxes of a BVH set.

    See genesis.engine.bvh.
    """

    # One thread per leaf walking to the root, the arrival counters sequentially consistent (CPU)
    LEAF_WALK = 0
    # The leaf walk with a device fence on each side of the relaxed arrival atomic (GPU)
    FENCED_LEAF_WALK = 1
    # One block per tree sweeping the nodes level by level, every barrier uniform (Metal, see quadrants#925)
    BLOCK_SWEEP = 2


@qd.data_oriented
class BVHStaticConfig(StaticConfig):
    """The compile-time shape of a set of linear bounding volume hierarchies (LBVH).

    See genesis.engine.bvh. The leaf keys pack the tree, the morton code and the leaf index in that order, so one flat
    sort orders every tree.
    """

    # The bits of a morton code, three interleaved coordinates of the same width
    morton_bits: int
    # The sort of the leaf keys, one segment per tree
    sort_config: SegmentSortStaticConfig
    # BVH_FIT_KIND
    fit_kind: int


# Sizes of a set of linear bounding volume hierarchies (LBVH): its trees, the leaves of each tree and the lanes of its
# extent reduction
BVH_TREES = Size("bvh_trees")
BVH_LEAVES = Size("bvh_leaves")
EXTENT_LANES = Size("extent_lanes")
# The scratch of the fit sweeping the nodes level by level (see BVH_FIT_KIND)
BLOCK_SWEEP_FIT = Label("block_sweep_fit")


class BVHTreeState(Data, kind=DataKind.SCRATCH, shape=(BVH_TREES, 2 * BVH_LEAVES - 1)):
    """What a traversal reads of a set of linear bounding volume hierarchies (LBVH).

    See genesis.engine.bvh for the build. The n_leaves - 1 internal nodes come first, then the leaves, so node
    n_leaves - 1 + i holds the i-th sorted leaf, whose index is leaves_idx[i_t, i]. A leaf node holds -1 in nodes_left.
    """

    nodes_left: qd.Tensor = field(int)
    nodes_right: qd.Tensor = field(int)
    nodes_min: qd.Tensor = field((float, 3))
    nodes_max: qd.Tensor = field((float, 3))
    leaves_idx: qd.Tensor = field(int, shape=(BVH_TREES, BVH_LEAVES))


class BVHLeaves(Data, kind=DataKind.SCRATCH, shape=(BVH_TREES, BVH_LEAVES)):
    """The boxes of the leaves of a set of linear bounding volume hierarchies (LBVH).

    See genesis.engine.bvh. The caller writes them before a build, and a box query tests them against a tree.
    """

    aabbs_min: qd.Tensor = field((float, 3))
    aabbs_max: qd.Tensor = field((float, 3))


class BVHState(Data, kind=DataKind.SCRATCH):
    """The linear bounding volume hierarchies (LBVH) of a set of trees.

    A build (see genesis.engine.bvh) takes the boxes of the leaves (see BVHLeaves) and produces the tree a traversal
    reads (see BVHTreeState), with the sort of the leaf keys (one segment per tree), the parent of each node and the
    scratch of the extent and the fit.
    """

    sort: SegmentSortBuffer
    tree: BVHTreeState
    leaves: BVHLeaves
    nodes_parent: qd.Tensor = field(int, shape=(BVH_TREES, 2 * BVH_LEAVES - 1))
    nodes_fitted: qd.Tensor = field(int, shape=(BVH_TREES, BVH_LEAVES - 1))
    fit_frontier: qd.Tensor = field(int, shape=(BVH_TREES, 2, BVH_LEAVES - 1), labels=BLOCK_SWEEP_FIT)
    lanes_min: qd.Tensor = field((float, 3), shape=(BVH_TREES, EXTENT_LANES))
    lanes_max: qd.Tensor = field((float, 3), shape=(BVH_TREES, EXTENT_LANES))
    trees_min: qd.Tensor = field((float, 3), shape=(BVH_TREES,))
    trees_max: qd.Tensor = field((float, 3), shape=(BVH_TREES,))


# The (tree, leaf, query) triplets a box query holds, and the leaves and the items a query filter reads
QUERY_RESULTS = Size("query_results")
FILTER_LEAVES = Size("filter_leaves")
FILTER_ITEMS = Size("filter_items")


class BVHQueryResults(Data, kind=DataKind.SCRATCH):
    """The (tree, leaf, query) triplets a box query of a BVH collects, and their count, above the capacity on overflow.

    See func_bvh_query_aabb in genesis.engine.bvh.
    """

    triplets: qd.Tensor = field((int, 3), shape=(QUERY_RESULTS,))
    count: qd.Tensor = field(int, shape=(1,))


class BVHQueryFilter(Data, kind=DataKind.CONSTANT):
    """
    The data the filter of a box query reads to drop the pairs it finds (see func_bvh_query_aabb in genesis.engine.bvh).

    `leaves_map` holds up to four items of each leaf, and `pairs_table` what the filter reads of each pair of items. A
    kernel taking the filter passes none of its buffers under another argument, so the filter owns them. Every filter
    is of this one type, since the funcs of a query take the filter data by its declared type.
    """

    leaves_map: qd.Tensor = field((int, 4), shape=(FILTER_LEAVES,))
    pairs_table: qd.Tensor = field(int, shape=(FILTER_ITEMS, FILTER_ITEMS))


# =========================================== SAPCoupler ===========================================


# Most FEM vertices a contact pair of the SAP coupler applies its impulse to, the vertices of two tetrahedra. The
# kernels read it as the size of the vector of slots of a pair (`fem_slots_vert` in SAPContactPairs), since a module
# global read in a kernel breaks its fast cache.
SAP_N_FEM_SLOTS = 8


# Sizes of the SAP coupler, besides the vertices and the elements of the FEM solver (see VERTICES): the tetrahedral
# volume of the rigid geoms, the equality constraints of every environment and their chunks, the contact pairs, those
# applying impulses to rigid dofs, the contact handlers, those applying impulses to rigid dofs, the vertex slots the
# pairs apply to, the chunks of the pairs of a handler, the candidates of a handler, and the rows of the marching
# tetrahedra tables
SAP_VOLUME_VERTS = Size("volume_verts")
SAP_VOLUME_ELEMS = Size("volume_elems")
SAP_EQUALITIES = Size("equality_constraints")
SAP_EQUALITY_CHUNKS = Size("equality_chunks")
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
    steps within the bracket [`alpha_min`, `alpha_max`], `f` being that derivative scaled by `dell_scale`, unless the
    slope at the start is small enough for the full step (`is_full_step`).
    """

    is_active: qd.Tensor = field(bool)
    is_pcg_active: qd.Tensor = field(bool)
    is_linesearch_active: qd.Tensor = field(bool)
    is_full_step: qd.Tensor = field(bool)
    gradient_norm: qd.Tensor = field(float)
    momentum_norm: qd.Tensor = field(float)
    impulse_norm: qd.Tensor = field(float)
    rTr: qd.Tensor = field(float)
    rTz: qd.Tensor = field(float)
    rTr_new: qd.Tensor = field(float)
    rTz_new: qd.Tensor = field(float)
    pTAp: qd.Tensor = field(float)
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
    f_lower: qd.Tensor = field(float)
    f_upper: qd.Tensor = field(float)
    f: qd.Tensor = field(float)
    df: qd.Tensor = field(float)
    minus_dalpha: qd.Tensor = field(float)
    minus_dalpha_prev: qd.Tensor = field(float)


class SAPEqualityConstraints(Data, kind=DataKind.CONSTANT, shape=(SAP_EQUALITIES,)):
    """
    The joint equality constraints of the rigid solver, which the SAP coupler regularizes and solves with the contacts.

    The constraints are ordered by environment, those of environment i_b being from `batch_start[i_b]` to
    `batch_start[i_b + 1]`. Each one ties the dof `dof1_idx` to the dof `dof2_idx`, its Jacobian row being `Jt`. The SAP
    regularization of a constraint is `k`, `R` and `v_hat`, its impulse `gamma` and its Hessian `G`. `vec` holds the
    impulse each constraint applies, and `partials` the partial sums over the chunks of the constraints of each
    environment.
    """

    batch_idx: qd.Tensor = field(int)
    dof1_idx: qd.Tensor = field(int)
    dof2_idx: qd.Tensor = field(int)
    batch_start: qd.Tensor = field(int, shape=(ENVS + 1,))
    Jt: qd.Tensor = field(float, shape=(SAP_EQUALITIES, DOFS))
    k: qd.Tensor = field(float)
    R: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    R_inv: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    v_hat: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    gamma: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    G: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    dvc: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    vec: qd.Tensor = field(float, kind=DataKind.SCRATCH)
    partials: qd.Tensor = field(float, shape=(SAP_EQUALITY_CHUNKS, 2, ENVS), kind=DataKind.SCRATCH)


class SAPContactPairs(Data, kind=DataKind.SCRATCH, shape=(SAP_PAIRS,)):
    """
    The contact pairs every contact handler of the SAP coupler detects, which one generic solve handles.

    Handler i_h owns the pairs from `handlers_offset[i_h]` to `handlers_offset[i_h] + handlers_capacity[i_h]`, the
    first `handlers_n_pairs[i_h]` of which are detected, `handlers_is_overflow[i_h]` telling whether its candidates or
    its box query overflowed. The handlers applying impulses to rigid dofs come first, so that their pairs alone take a
    Jacobian `Jt` (see SAPContactsLayout). A pair applies its impulse to up to SAP_N_FEM_SLOTS FEM vertices
    `fem_slots_vert` (-1 for an unused slot), with the weights `fem_slots_weight`, and to the chains of the links
    `link_idx` (-1 for none), with the signs `link_sign`. A pair with a contact frame (`normal`, `tangent0`, `tangent1`)
    solves in that frame, the others in the world frame.

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
    has_frame: qd.Tensor = field(bool)
    normal: qd.Tensor = field((float, 3))
    tangent0: qd.Tensor = field((float, 3))
    tangent1: qd.Tensor = field((float, 3))
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
