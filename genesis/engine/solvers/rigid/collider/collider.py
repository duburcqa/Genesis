"""
Collider module for rigid body collision detection.

This module provides collision detection functionality for the rigid body solver,
including broad-phase (sweep-and-prune), narrow-phase (convex-convex, SDF-based,
terrain), and contact management.
"""

from collections.abc import Callable
from typing import Annotated

import numpy as np
import torch

import trimesh


import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.pipeline import Enabled, Phase, Stages, System, register_system
from genesis.engine.solvers.kinematic_solver import _balanced_variant_mapping
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_update_geom_aabbs
from genesis.engine.solvers.rigid.data import ColliderInfo, ColliderState, ColliderStaticConfig, RigidErrorCode
from genesis.engine.solvers.rigid.rigid_solver import (
    RigidCollisionDetection,
    RigidCollisionDetectionBackward,
    RigidSolver,
)
from genesis.engine.state_change import StateChange, Subscriber
from genesis.engine.states import SystemCheckpoint
from genesis.options.solvers import RigidOptions
from genesis.utils.data import DataKind
from genesis.utils.misc import (
    get_gpu_core_count,
    has_zerocopy_view,
    qd_to_numpy,
    qd_to_torch,
    tensor_to_array,
    unique_rows,
)
from genesis.utils.sdf import SDF

from .broadphase import func_broad_phase
from .constants import CCD_ALGORITHM_CODE, Layout, PerEnv, Queued, WithEPA, WithoutEPA
from .contact import (
    collider_kernel_get_contacts,
    func_clamp_prune_contacts,
    func_clamp_prune_contacts_coop,
    func_contact_order_set,
    func_promote_woken_contacts,
    func_set_upstream_grad,
    kernel_collider_clear,
    kernel_masked_collider_clear,
    kernel_set_links_contact_force,
    kernel_wake_up_entities_resting_on_links,
)
from .gjk import GJK
from .mpr import MPR
from .narrowphase import (
    func_fill_diff_contact_input_analytic,
    func_narrow_phase_any_vs_terrain,
    func_narrow_phase_convex_specializations,
    func_narrow_phase_convex_vs_convex,
    func_narrow_phase_convex_vs_convex_sdf,
    func_narrow_phase_diff_convex_vs_convex,
    func_narrow_phase_nonconvex_vs_nonterrain,
    func_narrow_phase_nonconvex_vs_nonterrain_sdf,
    func_narrowphase_contact0,
    func_narrowphase_multicontact,
)
from .support_field import SupportField

NEUTRAL_COLLISION_RES_ABS = 0.01
NEUTRAL_COLLISION_RES_REL = 0.05


@register_system
class Collider(RigidCollisionDetection, RigidCollisionDetectionBackward):
    """
    Detect the contacts of the rigid geoms: their broad phase, the narrow phase of each family of pairs, the pruning.

    The stages of the detection are the Systems filling its slots (see Broadphase, Narrowphase and Pruning), which a
    scene holds when it has pairs of their family. A scene disabling the collisions holds the collider still, which
    then detects no contact.
    """

    # FIXME: quadrants#946 - a graph refuses the ndarrays that own a gradient, so a scene tracking gradients runs the
    # detection as no graph, inline with the substep around it.
    detection = Phase(
        "broadphase", "narrowphase", "pruning", is_graph=lambda collider: not collider._solver.requires_grad
    )

    collider_config: ColliderStaticConfig
    collider_info: ColliderInfo
    collider_state: ColliderState

    def __init__(self, rigid_solver: RigidSolver, options: RigidOptions) -> None:
        self._solver = rigid_solver
        self._options = options

    def build(self) -> None:
        self._init_static_config()

        # The threads of the narrow phase of the terrain pairs: two per core fill the warp slots the register use of the
        # terrain pass leaves
        self.n_terrain_threads = 1
        if self.collider_config.has_terrain and gs.backend != gs.cpu:
            self.n_terrain_threads = 2 * self.collider_config.gpu_cores

        self._init_collision_fields()

        # The query of the contacts reads these arrays through zero-copy views when they all have one, else gathers them
        # through a kernel (see 'get_contacts'). The views of n_contacts and contact_sort_idx are cached on their fields.
        collider_state = self.collider_state
        contact_data = collider_state.contact_data
        contact_arrays = dict(
            link_a=contact_data.link_a,
            link_b=contact_data.link_b,
            geom_a=contact_data.geom_a,
            geom_b=contact_data.geom_b,
            penetration=contact_data.penetration,
            position=contact_data.pos,
            normal=contact_data.normal,
            force=contact_data.force,
        )
        query_arrays = [collider_state.n_contacts, collider_state.contact_sort_idx, collider_state.first_time]
        if self.collider_config.has_non_box_plane_convex_convex:
            query_arrays += [collider_state.contact_cache.normal, collider_state.contact_cache.penetration]
        self._contact_data: dict[str, torch.Tensor] | None = None
        if all([has_zerocopy_view(array) for array in (*query_arrays, *contact_arrays.values())]):
            qd_to_torch(collider_state.n_contacts, copy=False)
            qd_to_torch(collider_state.contact_sort_idx, transpose=True, copy=False)
            self._contact_data = {
                key: qd_to_torch(array, transpose=True, copy=False) for key, array in contact_arrays.items()
            }

        # Make sure that the initial state is clean
        self.clear()

        collider_info = self.collider_info
        self._solver.register_errors(
            {
                RigidErrorCode.OVERFLOW_CANDIDATE_CONTACTS: (
                    "Exceeding max number of broad phase candidate contact pairs "
                    f"({collider_info.max_collision_pairs_broad[None]}). Please increase the value of RigidSolver's "
                    "option 'multiplier_collision_broad_phase'."
                ),
                RigidErrorCode.OVERFLOW_COLLISION_PAIRS: (
                    f"Exceeding max number of candidate contact points ({collider_info.max_candidate_contacts[None]}). "
                    "Please increase the value of RigidSolver's option 'max_collision_pairs'."
                ),
                RigidErrorCode.OVERFLOW_CONTACTS: (
                    f"Exceeding max number of post-pruning contact points ({collider_info.max_contacts[None]}) "
                    "supported by the constraint solver. Please increase the value of RigidSolver's option "
                    "'max_contacts'."
                ),
                RigidErrorCode.UNSUPPORTED_COLLISION_PAIR: (
                    "Collision between a non-convex geom and a terrain is not supported. Set 'convexify=True' or "
                    "filter this pair out via 'contype'/'conaffinity'."
                ),
                RigidErrorCode.INVALID_CONTACT_NAN: (
                    "Collision detection reported a contact whose position, normal or penetration is not finite. This "
                    "is a solver-internal error, please report it."
                ),
            },
        )

        # The contacts and their forces were found at the previous pose, so a change of the geometry moving a link
        # clears them, the sleepers alone keeping theirs (see 'clear'), while a change of the visual geometry alone
        # reaches no link of the filter. The checkpoint of the collider is restored after that of the rigid solver,
        # whose restore changes the geometry, so the contacts it holds are the ones that remain.
        self._fixed_links_idx = np.array([link.idx for link in self._solver.links if link.is_fixed], dtype=gs.np_int)
        self._solver.subscribe(
            Subscriber(frozenset((StateChange.GEOMETRY,)), callback=self._clear_moved, links_filter=slice(None))
        )
        self.detection.add(self._clear_contacts_readout)

    def load_checkpoint(self, checkpoint: SystemCheckpoint, *, boundary_only: bool = False) -> None:
        super().load_checkpoint(checkpoint, boundary_only=boundary_only)
        # The restore of the rigid solver, which precedes this one, clears the contacts and their forces (see 'build'),
        # so the contact force of the links is summed again from the contacts restored
        if DataKind.DERIVED in checkpoint.kinds:
            kernel_set_links_contact_force(self._solver.dyn_state, self.collider_state, self._solver.rigid_config)

    def _clear_contacts_readout(self) -> None:
        """Drop the readout of the contacts the last query cached, which the next detection invalidates."""
        self._contact_data_cache.clear()

    def _clear_moved(self, change: StateChange, envs_idx, links_idx: np.ndarray | None) -> None:
        """Clear the contacts of the envs whose geometry changed, waking the sleepers on moved fixed links first.

        A fixed link never sleeps, so the links resting on it would stay asleep where it was otherwise. A change of
        unbounded reach resets or restores the state of every link, sleepers included, which leaves none to wake.
        """
        if self._solver.rigid_config.use_hibernation and links_idx is not None:
            fixed_links_idx = np.intersect1d(links_idx, self._fixed_links_idx)
            if fixed_links_idx.size:
                kernel_wake_up_entities_resting_on_links(
                    fixed_links_idx,
                    self._solver._scene._sanitize_envs_idx(envs_idx),
                    self._solver.dyn_state,
                    self._solver.hibernation_state,
                    self.collider_state,
                    self._solver.dyn_info,
                    self._solver.rigid_info,
                    self._solver.rigid_config,
                )
        self.clear(envs_idx)

    @property
    def has_possible_pairs(self) -> bool:
        """Whether some pair of geoms can collide."""
        return self._n_possible_pairs > 0

    @property
    def has_convex_pairs(self) -> bool:
        """Whether some pair of convex geoms that no primitive specialization handles can collide."""
        return self.has_possible_pairs and self.collider_config.has_non_box_plane_convex_convex

    @property
    def has_primitive_pairs(self) -> bool:
        """Whether some pair of primitive geoms that a specialization handles can collide."""
        return self.has_possible_pairs and self._has_convex_specialization

    @property
    def has_terrain_pairs(self) -> bool:
        """Whether some geom can collide with a terrain."""
        return self.has_possible_pairs and self.collider_config.has_terrain

    @property
    def has_nonconvex_pairs(self) -> bool:
        """Whether some pair of geoms holding a non-convex one, a terrain aside, can collide."""
        return self.has_possible_pairs and self._has_nonconvex_nonterrain

    @property
    def narrowphase_layout(self) -> type[Layout]:
        """The layout of the convex narrow phase: a queue of pairs on GPU, a worker per environment otherwise."""
        if (
            self.collider_config.has_non_box_plane_convex_convex
            and gs.backend != gs.cpu
            and not self._solver.requires_grad
        ):
            return Queued
        return PerEnv

    def _init_static_config(self) -> None:
        # Identify the convex collision detection (ccd) algorithm
        if self._options.use_gjk_collision:
            if self._solver.rigid_config.enable_mujoco_compatibility:
                ccd_algorithm = CCD_ALGORITHM_CODE.MJ_GJK
            else:
                ccd_algorithm = CCD_ALGORITHM_CODE.GJK
        else:
            if self._solver.rigid_config.enable_mujoco_compatibility:
                ccd_algorithm = CCD_ALGORITHM_CODE.MJ_MPR
            else:
                ccd_algorithm = CCD_ALGORITHM_CODE.MPR

        # The contact patch can hand back the full clipped polygon of a box-box pair, up to 8 contacts (see
        # func_clip_polygon), which the convex budget must hold or the extra witnesses of the patch are lost.
        if self._solver.rigid_config.requires_grad:
            n_contacts_per_convex_pair = 20
        elif self._options.enable_contact_patch:
            n_contacts_per_convex_pair = 8
        else:
            n_contacts_per_convex_pair = 5

        # Nonconvex pairs and box-box pairs emit a whole contact ring or face patch, so they share a cap of their own,
        # which keeps the buffer of ordinary convex pairs small. Fewer points let parts of an extended patch drop out
        # as the geometry moves, losing constraint directions and letting bodies slip.
        n_contacts_per_nonconvex_pair = 40
        if self._options.box_box_detection and sum(g.type == gs.GEOM_TYPE.BOX for g in self._solver.geoms) > 1:
            n_contacts_per_nonconvex_pair = max(n_contacts_per_nonconvex_pair, 16)

        self._geoms_link_idx = np.array([geom.link.idx for geom in self._solver.geoms], dtype=gs.np_int)
        self._links_n_geoms = np.bincount(self._geoms_link_idx, minlength=self._solver.n_links_)

        # Compute collision pairs and algorithm flags in a single pass
        (
            self._n_possible_pairs,
            self._collision_pair_idx,
            self._valid_collision_pairs,
            has_terrain,
            has_non_box_plane_convex_convex,
            has_convex_specialization,
            has_nonconvex_nonterrain,
            self._large_contact_pair_mask,
        ) = self._compute_collision_pair_idx()
        # The link pairs holding a valid geom pair and the one of each valid geom pair. A link pair is oriented with the
        # link carrying the most geoms first, whose geoms the broad phase spreads its work over (see
        # func_broad_phase_geoms).
        valid_link_pairs, _, self._pairs_link_pair_idx, _ = unique_rows(
            np.sort(self._geoms_link_idx[self._valid_collision_pairs], axis=1)
        )
        is_swapped = self._links_n_geoms[valid_link_pairs[:, 1]] > self._links_n_geoms[valid_link_pairs[:, 0]]
        valid_link_pairs[is_swapped] = valid_link_pairs[is_swapped, ::-1]
        self._valid_link_pairs = valid_link_pairs

        # The hull prune drops a contact only from a (link_a, link_b) bucket of at least 3 contacts, the middle one of 3
        # collinear points being the smallest case, so the pass is compiled in whenever some pair of links can emit 3
        # contacts, and leaves every contact as it is otherwise. Nonconvex geoms and terrain emit many contacts per pair
        # through their vertex-based narrowphase. A convex pair emits several through multi-contact detection or a box
        # specialization, unless it holds a sphere or an ellipsoid, which leaves it a single point. Outside performance
        # mode the pass is always compiled in, being a no-op wherever it would be left out.
        has_prunable_contacts = gs.use_ndarray or has_nonconvex_nonterrain or has_terrain
        if not has_prunable_contacts and len(self._valid_collision_pairs):
            geoms_type = np.array([geom.type for geom in self._solver.geoms])
            pairs_type = geoms_type[self._valid_collision_pairs]
            is_point_pair = np.isin(pairs_type, (gs.GEOM_TYPE.SPHERE, gs.GEOM_TYPE.ELLIPSOID)).any(axis=1)
            is_multi_point_pair = ~is_point_pair & (
                self._options.enable_multi_contact | (pairs_type == gs.GEOM_TYPE.BOX).any(axis=1)
            )
            links_pair_n_contacts = np.bincount(self._pairs_link_pair_idx, weights=np.where(is_multi_point_pair, 3, 1))
            has_prunable_contacts = bool(links_pair_n_contacts.max() >= 3)

        # Outside performance mode the box specializations are always compiled in, their pass finding nothing in a scene
        # without box pairs, which every other path leaves to it by geom type.
        self._has_convex_specialization = has_convex_specialization or gs.use_ndarray
        self._has_nonconvex_nonterrain = has_nonconvex_nonterrain
        self.collider_config = ColliderStaticConfig(
            gpu_cores=get_gpu_core_count(),
            has_terrain=has_terrain,
            # Outside performance mode the convex narrowphase and the contact sort it gates are always compiled in, the
            # narrowphase finding nothing in a scene whose convex pairs all take a box specialization.
            has_non_box_plane_convex_convex=has_non_box_plane_convex_convex or gs.use_ndarray,
            has_prunable_contacts=has_prunable_contacts,
            n_contacts_per_convex_pair=n_contacts_per_convex_pair,
            n_contacts_per_nonconvex_pair=n_contacts_per_nonconvex_pair,
            ccd_algorithm=ccd_algorithm,
        )

    def _init_collision_fields(self) -> None:
        # Pre-compute fields, as they are needed to initialize the collider state and info.
        vert_neighbors, vert_neighbor_start, vert_n_neighbors = self._compute_verts_connectivity()
        n_vert_neighbors = len(vert_neighbors)

        n_valid_link_pairs = len(self._valid_link_pairs)
        link_pair_idx = np.full((self._solver.n_links_, self._solver.n_links_), fill_value=-1, dtype=gs.np_int)
        links_a_idx, links_b_idx = self._valid_link_pairs.T
        link_pair_idx[links_a_idx, links_b_idx] = np.arange(n_valid_link_pairs, dtype=gs.np_int)
        link_pair_idx[links_b_idx, links_a_idx] = np.arange(n_valid_link_pairs, dtype=gs.np_int)

        n_possible_nonconvex_pairs = int(np.count_nonzero(self._large_contact_pair_mask))
        max_collision_pairs = min(self._solver.max_collision_pairs, self._n_possible_pairs)
        # The worst case fills the pair budget with as many nonconvex pairs as exist, then with convex and terrain
        # pairs, each at its cap. The two vertex scans of a nonconvex pair share its cap: the second one only displaces
        # the least-penetrating contact once the pair is full.
        cap_nonconvex = self.collider_config.n_contacts_per_nonconvex_pair
        cap_convex = self.collider_config.n_contacts_per_convex_pair
        n_nonconvex = min(n_possible_nonconvex_pairs, max_collision_pairs)
        n_convex = min(self._n_possible_pairs - n_possible_nonconvex_pairs, max_collision_pairs - n_nonconvex)
        max_candidate_contacts = n_nonconvex * cap_nonconvex + n_convex * cap_convex
        # The broad phase reports each possible pair at most once per environment
        max_collision_pairs_broad = min(
            max_collision_pairs * self._options.multiplier_collision_broad_phase, self._n_possible_pairs
        )

        # The constraint solver only sees the contacts surviving link-pair pruning, roughly the support polygon of each
        # link pair, so its budget caps a link pair at 32 points. A conforming patch is left unpruned and can exceed it,
        # which halts the simulation with a request to increase 'max_contacts'. The floor of 512 keeps that risk away
        # from small scenes, where the cap would save no meaningful memory.
        max_contacts = max_candidate_contacts
        if (
            self.collider_config.has_prunable_contacts
            and not self._solver.requires_grad
            and self._options.contact_pruning_tolerance is not None
        ):
            pairs_n_contacts = np.where(self._large_contact_pair_mask, cap_nonconvex, cap_convex)
            link_pairs_n_contacts = np.bincount(self._pairs_link_pair_idx, weights=pairs_n_contacts)
            max_contacts_pruned_total = max(int(np.minimum(link_pairs_n_contacts, 32).sum()), 512)
            max_contacts = min(max_contacts, max_contacts_pruned_total)

        # The option 'max_contacts' replaces the post-pruning budget, capped by the candidate budget. It is resolved in
        # place, since the constraint solver reads it.
        options = self._options
        if options.max_contacts is None:
            options.max_contacts = max_contacts
        else:
            options.max_contacts = min(options.max_contacts, max_candidate_contacts)

        terrain_hf_shape = (1, 1)
        for geom in self._solver.geoms:
            if geom.type == gs.GEOM_TYPE.TERRAIN:
                terrain_hf_shape = geom.entity.terrain_hf.shape
                break

        # Initialize [info], which stores every data that must be considered mutable from Quadrants's perspective,
        # i.e. unknown at compile time, but IMMUTABLE from Genesis scene's perspective after build.
        self.collider_info = ColliderInfo(
            sizes=dict(
                geoms=self._solver.n_geoms,
                links=self._solver.n_links,
                verts=self._solver.n_verts,
                vert_neighbors=n_vert_neighbors,
                valid_link_pairs=n_valid_link_pairs,
                terrain_rows=terrain_hf_shape[0],
                terrain_cols=terrain_hf_shape[1],
            ),
            values=dict(
                max_candidate_contacts=max_candidate_contacts,
                max_collision_pairs_broad=max_collision_pairs_broad,
                max_contacts=options.max_contacts,
                n_valid_link_pairs=n_valid_link_pairs,
                mc_perturbation=1e-3 if self._solver.rigid_config.enable_mujoco_compatibility else 5e-3,
                mc_tolerance=1e-3 if self._solver.rigid_config.enable_mujoco_compatibility else 1.5e-2,
                diff_pos_tolerance=1e-2,
                diff_normal_tolerance=1e-2,
                diff_contact_eps_boundary=1e-2,
                diff_contact_eps_distance=1e-2,
                contact_pruning_tolerance=self._options.contact_pruning_tolerance or 0.0,
                prune_deep_penetration_ratio=3.0,
            ),
        )
        self._init_collision_pair_idx(self._collision_pair_idx)
        self.collider_info.link_pair_idx.from_numpy(link_pair_idx)

        # The box of every link in its own frame, bounding the boxes of its geoms, whose pose in the link is fixed
        if self._solver.n_geoms > 0:
            geoms_corners = gu.transform_by_trans_quat(
                qd_to_numpy(self._solver.rigid_info.geoms_init_AABB)[: self._solver.n_geoms],
                np.array([geom.init_pos for geom in self._solver.geoms])[:, None],
                np.array([geom.init_quat for geom in self._solver.geoms])[:, None].repeat(8, axis=1),
            )
            links_init_aabb_min = np.full((self._solver.n_links_, 3), np.inf)
            links_init_aabb_max = np.full((self._solver.n_links_, 3), -np.inf)
            np.minimum.at(links_init_aabb_min, self._geoms_link_idx, geoms_corners.min(axis=1))
            np.maximum.at(links_init_aabb_max, self._geoms_link_idx, geoms_corners.max(axis=1))
            self.collider_info.links_init_aabb_min.from_numpy(links_init_aabb_min.astype(gs.np_float))
            self.collider_info.links_init_aabb_max.from_numpy(links_init_aabb_max.astype(gs.np_float))
        if n_valid_link_pairs > 0:
            self.collider_info.valid_link_pairs.from_numpy(self._valid_link_pairs)
        self._init_verts_connectivity(vert_neighbors, vert_neighbor_start, vert_n_neighbors)
        self._init_verts_spatial_grid()
        self._init_terrain_state()

        # Initialize [state], which stores every data that are may be updated at every single simulation step
        config = self._solver.rigid_config
        is_split = self.narrowphase_layout is Queued
        self.collider_state = ColliderState(
            sizes=dict(
                envs=self._solver._B,
                links=self._solver.n_links_,
                valid_link_pairs=n_valid_link_pairs,
                possible_pairs=self._n_possible_pairs,
                broad_pairs=max_collision_pairs_broad,
                candidate_contacts=max_candidate_contacts,
                nonconvex_pair_contacts=self.collider_config.n_contacts_per_nonconvex_pair,
                contact_slots=1 if self._options.enable_contact_patch else 5,
                terrain_threads=self.n_terrain_threads,
            ),
            layout={ColliderState.contact_sort_key: True, "pruning": True},
            unused=dict(
                box_box=not config.box_box_detection,
                contact_cache=not self.collider_config.has_non_box_plane_convex_convex,
                diff_contacts=not config.requires_grad,
                work_queue=not (is_split or self.collider_config.has_terrain),
                split_narrowphase=not is_split,
                perturbed_slots=not (is_split and not self._options.enable_contact_patch),
                pruning=not self.collider_config.has_prunable_contacts,
            ),
            requires_grad=config.requires_grad,
        )

        # 'contact_data_cache' is not used in Quadrants kernels, so keep it outside of the collider state / info
        self._contact_data_cache: dict[tuple[bool, bool], dict[str, torch.Tensor | tuple[torch.Tensor]]] = {}

        # Whether a warp prunes the contacts of an environment, rather than a thread. The warp-per-env cooperative dedup
        # kernel beats the one-env-per-thread fused kernel only while the GPU keeps spare occupancy: past half the cores
        # in envs, the launch oversubscribes the SMs and the fused kernel wins.
        self.is_pruning_cooperative = (
            gs.backend != gs.cpu
            and not self._solver.rigid_config.requires_grad
            and self.collider_config.has_prunable_contacts
            and (self._options.contact_pruning_tolerance or 0.0) > 0.0
            and self._solver._B <= 0.5 * self.collider_config.gpu_cores
        )

    def _compute_collision_pair_idx(self):
        """
        Compute flat indices of all valid collision pairs, plus algorithm flags.

        For each pair of geoms, determine if they can collide based on their properties and the solver configuration.
        Pairs that are already colliding at the initial configuration (qpos0) are filtered out with a warning.

        Returns (n_possible_pairs, collision_pair_idx, pair_flags) where pair_flags is a dict of booleans
        for has_terrain, has_non_box_plane_convex_convex, has_convex_specialization, has_nonconvex_nonterrain.
        """
        n_geoms = self._solver.n_geoms
        geoms = self._solver.geoms

        if n_geoms == 0:
            empty_pairs = np.empty((0, 2), dtype=gs.np_int)
            empty_mask = np.zeros((0,), dtype=bool)
            return 0, np.full((0, 0), -1, dtype=gs.np_int), empty_pairs, False, False, False, False, empty_mask

        # A coupler resolving the contacts of some links leaves the rigid solver the pairs it does not resolve
        coupler = self._solver.sim.coupler
        links_idx_contacting = set() if coupler is None else coupler.rigid_links_idx_contacting
        links_idx_simulated = set() if coupler is None else coupler.rigid_links_idx_simulated

        # Pre-compute per-geom properties into numpy arrays for vectorized filtering
        geom_root_idx = np.array([g.link.root_idx for g in geoms], dtype=np.int32)
        geom_is_fixed = np.array([g.link.is_fixed for g in geoms], dtype=bool)
        geom_entity_id = np.array([id(g.entity) for g in geoms], dtype=np.int64)
        geom_contype = np.array([g.contype for g in geoms], dtype=np.int64)
        geom_conaffinity = np.array([g.conaffinity for g in geoms], dtype=np.int64)
        geom_local_mask = np.array([g.entity.is_local_collision_mask for g in geoms], dtype=bool)
        geom_is_simulated = np.array([g.link.idx in links_idx_simulated for g in geoms], dtype=bool)
        geom_is_contacting = np.array([g.link.idx in links_idx_contacting for g in geoms], dtype=bool)
        geom_type = np.array([g.type for g in geoms], dtype=np.int32)
        geom_is_convex = np.array([g.is_convex for g in geoms], dtype=bool)

        # Build weld pairs set for O(1) lookup (use sorted tuple keys)
        weld_pairs = set()
        for eq in self._solver.equalities:
            if eq.type == gs.EQUALITY_TYPE.WELD:
                a, b = eq.eq_obj1id, eq.eq_obj2id
                weld_pairs.add((min(a, b), max(a, b)))

        # --- Vectorized filtering: build upper-triangular valid-pair mask ---
        row, col = np.triu_indices(n_geoms, k=1)

        link_a = self._geoms_link_idx[row]
        link_b = self._geoms_link_idx[col]

        # geoms in the same link
        valid = link_a != link_b

        valid &= ~geom_is_simulated[row]
        valid &= ~geom_is_simulated[col]
        valid &= ~(geom_is_contacting[row] & geom_is_contacting[col])

        # pair of fixed links wrt the world
        valid &= ~(geom_is_fixed[row] & geom_is_fixed[col])

        # contype and conaffinity
        same_entity = geom_entity_id[row] == geom_entity_id[col]
        has_local_mask = geom_local_mask[row] | geom_local_mask[col]
        con_match = (geom_contype[row] & geom_conaffinity[col]) | (geom_contype[col] & geom_conaffinity[row])
        con_skip = (same_entity | ~has_local_mask) & (con_match == 0)
        valid &= ~con_skip

        # self collision (same root), disabled
        same_root = geom_root_idx[row] == geom_root_idx[col]
        if not self._solver._enable_self_collision:
            valid &= ~same_root

        # Weld constraint filtering
        if weld_pairs:
            link_min = np.minimum(link_a, link_b)
            link_max = np.maximum(link_a, link_b)
            is_weld = np.array([(link_min[i], link_max[i]) in weld_pairs for i in range(len(row))], dtype=bool)
            valid &= ~is_weld

        # Heterogeneous variants that no environment carries together never collide, and keeping their pairs would
        # grow the pair count quadratically with the pool size. An entity dispatches the same variant to all of its
        # links in a given environment, so a pair is kept only if some environment carries the variant of both geoms.
        # Each variant of every heterogeneous entity gets its own key, and the key 0 stands for every homogeneous geom.
        geoms_variant_key = np.zeros((n_geoms,), dtype=gs.np_int)
        entities_envs_variant_key = [np.zeros((self._solver._B,), dtype=gs.np_int)]
        n_variant_keys = 1
        for entity in self._solver.entities:
            if not entity.desc.variants:
                continue
            for link in entity.links:
                for i_variant, (geom_start, geom_end) in enumerate(link._variant_geom_ranges):
                    geoms_variant_key[geom_start:geom_end] = n_variant_keys + i_variant
            n_variants = len(entity.desc.variants)
            entities_envs_variant_key.append(n_variant_keys + _balanced_variant_mapping(n_variants, self._solver._B))
            n_variant_keys += n_variants
        envs_variant_key = np.stack(entities_envs_variant_key)
        variant_keys_is_co_carried = np.zeros((n_variant_keys, n_variant_keys), dtype=bool)
        variant_keys_is_co_carried[envs_variant_key[:, None], envs_variant_key[None, :]] = True
        valid &= variant_keys_is_co_carried[geoms_variant_key[row], geoms_variant_key[col]]

        # --- Self-collision: adjacent and neutral overlap checks (Python loop, only same-root pairs) ---
        # These checks only apply when self_collision is enabled and the pair passed all vectorized filters
        self_colliding_pairs: list[tuple[int, int]] = []

        needs_self_check = self._solver._enable_self_collision and np.any(valid & same_root)
        needs_neutral_check = needs_self_check and not self._solver._enable_neutral_collision

        # Lazily compute geom vertices only for geoms that need neutral overlap checks
        geoms_verts: dict[int, np.ndarray] = {}
        # Voxel grid and filled-voxel centers of a geom, built the first time it enters a pair with overlapping bounds
        # and shared by every other pair it enters, since voxelization dominates this check and a link usually overlaps
        # several others in the neutral configuration
        geoms_voxels: dict[int, tuple[trimesh.voxel.VoxelGrid, np.ndarray]] = {}
        if needs_neutral_check:
            self_root_indices = np.where(valid & same_root)[0]
            self_root_geom_idxs = np.unique(np.concatenate([row[self_root_indices], col[self_root_indices]]))
            # Compute vertices only for geoms involved in self-collision pairs, shrunk by 0.1% to avoid false positive
            # when detecting self-collision. The pose of a heterogeneous geom is only meaningful in the environments
            # carrying its variant. The pair filter above leaves only variants that some environment carries, and every
            # same-root pair left belongs to a single variant of one entity, so the first environment carrying a geom
            # carries the other geom of its pair as well.
            for i_g in self_root_geom_idxs:
                geom = geoms[i_g]
                i_b = geom.active_envs_idx[0] if geom.active_envs_idx is not None else 0
                env_idx = i_b if self._solver.n_envs > 0 else None
                geom_pos = tensor_to_array(geom.get_pos(env_idx, relative=False).reshape((3,)))
                geom_quat = tensor_to_array(geom.get_quat(env_idx, relative=False).reshape((4,)))
                verts = gu.transform_by_trans_quat(geom.init_verts, geom_pos, geom_quat)
                centroid = verts.mean(axis=0, keepdims=True)
                verts = centroid + (1.0 - 1e-3) * (verts - centroid)
                geoms_verts[i_g] = verts

        if needs_self_check:
            self_root_indices = np.where(valid & same_root)[0]
            for idx in self_root_indices:
                i_ga, i_gb = row[idx], col[idx]
                link_ga = geoms[i_ga].link
                link_gb = geoms[i_gb].link

                # adjacent links
                # FIXME: Links should be considered adjacent if connected by only fixed joints.
                if not self._solver._enable_adjacent_collision:
                    is_adjacent = False
                    link_a_, link_b_ = (link_ga, link_gb) if link_ga.idx < link_gb.idx else (link_gb, link_ga)
                    while link_b_.parent_idx != -1:
                        if link_b_.parent_idx == link_a_.idx:
                            is_adjacent = True
                            break
                        if not all(joint.type is gs.JOINT_TYPE.FIXED for joint in link_b_.joints):
                            break
                        link_b_ = self._solver.links[link_b_.parent_idx]
                    if is_adjacent:
                        valid[idx] = False
                        continue

                # active in neutral configuration (qpos0)
                if needs_neutral_check:
                    verts_a = geoms_verts[i_ga]
                    mesh_a = trimesh.Trimesh(vertices=verts_a, faces=geoms[i_ga].init_faces, process=False)
                    verts_b = geoms_verts[i_gb]
                    mesh_b = trimesh.Trimesh(vertices=verts_b, faces=geoms[i_gb].init_faces, process=False)
                    bounds_a, bounds_b = mesh_a.bounds, mesh_b.bounds
                    if not ((bounds_a[1] < bounds_b[0]).any() or (bounds_b[1] < bounds_a[0]).any()):
                        for i_g, mesh in ((i_ga, mesh_a), (i_gb, mesh_b)):
                            if i_g not in geoms_voxels:
                                voxels = mesh.voxelized(
                                    pitch=min(NEUTRAL_COLLISION_RES_ABS, NEUTRAL_COLLISION_RES_REL * max(mesh.extents))
                                )
                                geoms_voxels[i_g] = (voxels, voxels.indices_to_points(np.argwhere(voxels.matrix)))
                        voxels_a, coords_a = geoms_voxels[i_ga]
                        voxels_b, coords_b = geoms_voxels[i_gb]
                        if voxels_a.is_filled(coords_b).any() or voxels_b.is_filled(coords_a).any():
                            self_colliding_pairs.append((i_ga, i_gb))
                            valid[idx] = False
                            continue

        # Emit warning for self-collision pairs
        if self_colliding_pairs:
            pairs = ", ".join((f"({i_ga}, {i_gb})") for i_ga, i_gb in self_colliding_pairs)
            gs.logger.warning(
                f"Filtered out geometry pairs causing self-collision for the neutral configuration (qpos0): {pairs}. "
                "Consider tuning Morph option 'decompose_robot_error_threshold' or specify dedicated collision meshes. "
                "This behavior can be disabled by setting Morph option 'enable_neutral_collision=True'."
            )

        # --- Build collision_pair_idx, valid pairs list, and count ---
        valid_indices = np.where(valid)[0]
        n_possible_pairs = len(valid_indices)
        # Symmetric, so a pair finds its index in whichever order it comes (see func_broad_phase_geoms)
        collision_pair_idx = np.full((n_geoms, n_geoms), fill_value=-1, dtype=gs.np_int)
        collision_pair_idx[row[valid_indices], col[valid_indices]] = np.arange(n_possible_pairs, dtype=gs.np_int)
        collision_pair_idx[col[valid_indices], row[valid_indices]] = np.arange(n_possible_pairs, dtype=gs.np_int)

        valid_collision_pairs = np.stack([row[valid_indices], col[valid_indices]], axis=1).astype(gs.np_int)

        # --- Compute algorithm flags from valid pairs ---
        valid_type_a = geom_type[row[valid_indices]]
        valid_type_b = geom_type[col[valid_indices]]
        valid_convex_a = geom_is_convex[row[valid_indices]]
        valid_convex_b = geom_is_convex[col[valid_indices]]

        has_any_vs_terrain = bool(
            np.any((valid_type_a == gs.GEOM_TYPE.TERRAIN) | (valid_type_b == gs.GEOM_TYPE.TERRAIN))
        )
        both_convex = valid_convex_a & valid_convex_b
        if np.any(both_convex):
            is_box_a = valid_type_a == gs.GEOM_TYPE.BOX
            is_box_b = valid_type_b == gs.GEOM_TYPE.BOX
            is_plane_a = valid_type_a == gs.GEOM_TYPE.PLANE
            is_plane_b = valid_type_b == gs.GEOM_TYPE.PLANE
            is_plane_box = (is_plane_a & is_box_b) | (is_box_a & is_plane_b)
            specialized = is_plane_box
            if self._options.box_box_detection:
                specialized = specialized | (is_box_a & is_box_b)
            has_non_box_plane_convex_convex = bool(np.any(both_convex & ~specialized))

            # Differentiable contact detection (diff_gjk) reconstructs each contact from a triangular face of the
            # Minkowski difference. A sphere or ellipsoid has no flat facet, so a pair of them yields an everywhere
            # smoothly curved Minkowski boundary on which EPA never converges, and no contact is ever generated -
            # the bodies silently tunnel. Faceted partners (box, mesh) and the analytical plane branch are unaffected,
            # and sphere-sphere pairs are reconstructed in closed form (func_differentiable_sphere_contact).
            if self._solver.requires_grad:
                is_smooth_a = (valid_type_a == gs.GEOM_TYPE.SPHERE) | (valid_type_a == gs.GEOM_TYPE.ELLIPSOID)
                is_smooth_b = (valid_type_b == gs.GEOM_TYPE.SPHERE) | (valid_type_b == gs.GEOM_TYPE.ELLIPSOID)
                is_sphere_sphere = (valid_type_a == gs.GEOM_TYPE.SPHERE) & (valid_type_b == gs.GEOM_TYPE.SPHERE)
                if np.any(both_convex & ~specialized & is_smooth_a & is_smooth_b & ~is_sphere_sphere):
                    gs.raise_exception(
                        "Differentiable contact detection is not supported for sphere-ellipsoid or "
                        "ellipsoid-ellipsoid collision pairs (requires_grad=True). Approximate them with a faceted "
                        "geometry (e.g. a convex mesh) or disable requires_grad."
                    )
        else:
            has_non_box_plane_convex_convex = False

        if self._options.box_box_detection:
            spec_types = [gs.GEOM_TYPE.TERRAIN, gs.GEOM_TYPE.BOX]
            has_convex_specialization = bool(
                np.any(np.isin(valid_type_a, spec_types) | np.isin(valid_type_b, spec_types))
            )
        else:
            has_convex_specialization = bool(
                np.any(
                    ((valid_type_a == gs.GEOM_TYPE.BOX) & (valid_type_b == gs.GEOM_TYPE.PLANE))
                    | ((valid_type_a == gs.GEOM_TYPE.PLANE) & (valid_type_b == gs.GEOM_TYPE.BOX))
                )
            )

        # Pairs routed to the vertex-vs-SDF nonconvex narrowphase (at least one nonconvex geom, neither terrain).
        nonconvex_pair_mask = (
            ~(valid_convex_a & valid_convex_b)
            & (valid_type_a != gs.GEOM_TYPE.TERRAIN)
            & (valid_type_b != gs.GEOM_TYPE.TERRAIN)
        )
        has_nonconvex_vs_nonterrain = bool(np.any(nonconvex_pair_mask))
        # Pairs that emit many contacts per pair, used to size the contact buffer. Box-box pairs (when the specialized
        # detector is enabled) emit up to box_MAXCONPAIR contacts, so they join the nonconvex pairs in the large-cap
        # bucket rather than being sized as generic convex pairs.
        large_contact_mask = nonconvex_pair_mask
        if self._options.box_box_detection:
            large_contact_mask = large_contact_mask | (
                (valid_type_a == gs.GEOM_TYPE.BOX) & (valid_type_b == gs.GEOM_TYPE.BOX)
            )

        return (
            n_possible_pairs,
            collision_pair_idx,
            valid_collision_pairs,
            has_any_vs_terrain,
            has_non_box_plane_convex_convex,
            has_convex_specialization,
            has_nonconvex_vs_nonterrain,
            large_contact_mask,
        )

    def _compute_verts_connectivity(self):
        """
        Compute the vertex connectivity.
        """
        vert_neighbors = []
        vert_neighbor_start = []
        vert_n_neighbors = []
        offset = 0
        for geom in self._solver.geoms:
            vert_neighbors.append(geom.vert_neighbors + geom.vert_start)
            vert_neighbor_start.append(geom.vert_neighbor_start + offset)
            vert_n_neighbors.append(geom.vert_n_neighbors)
            offset = offset + len(geom.vert_neighbors)

        if self._solver.n_verts > 0:
            vert_neighbors = np.concatenate(vert_neighbors, dtype=gs.np_int)
            vert_neighbor_start = np.concatenate(vert_neighbor_start, dtype=gs.np_int)
            vert_n_neighbors = np.concatenate(vert_n_neighbors, dtype=gs.np_int)

        return vert_neighbors, vert_neighbor_start, vert_n_neighbors

    def _init_verts_spatial_grid(self):
        """
        Sort each geom's collision verts into a fixed 8x8x8 grid over its local AABB, as a permutation of vert
        indices ordered by grid cell (z fastest) plus per-cell vert ranges.

        The nonconvex narrowphase visits only the cells overlapping the other geom's pulled-back AABB, skipping far
        verts wholesale. Binning uses the same single-precision cell mapping as the kernel side; that mapping is
        monotone, so a vert inside a (padded) query box always lands inside the visited cell range.
        """
        if self._solver.n_verts == 0:
            return
        n_geoms = len(self._solver.geoms)
        verts_idx = []
        verts_pos = []
        cells_vert_start = []
        geoms_origin = np.zeros((n_geoms, 3), dtype=gs.np_float)
        geoms_inv_cell_size = np.zeros((n_geoms, 3), dtype=gs.np_float)
        offset_vert = 0
        for i_g, geom in enumerate(self._solver.geoms):
            verts = geom.init_verts.astype(gs.np_float)
            if len(verts) == 0:
                verts_idx.append(np.zeros(0, dtype=gs.np_int))
                verts_pos.append(np.zeros((0, 3), dtype=gs.np_float))
                cells_vert_start.append(np.full(8**3 + 1, offset_vert, dtype=gs.np_int))
                continue
            origin = verts.min(axis=0)
            extent = verts.max(axis=0) - origin
            inv_cell_size = np.where(extent > 0.0, 8 / np.maximum(extent, gs.EPS), 0.0).astype(gs.np_float)
            verts_cell = np.clip(np.floor((verts - origin) * inv_cell_size), 0, 7).astype(gs.np_int)
            verts_cell_flat = (verts_cell[:, 0] * 8 + verts_cell[:, 1]) * 8 + verts_cell[:, 2]
            order = np.argsort(verts_cell_flat, kind="stable")
            verts_idx.append(order + geom.vert_start)
            # The positions themselves are duplicated in spatial order: the scan streams them sequentially, where
            # gathering through the permutation would defeat the prefetcher on the hot path.
            verts_pos.append(verts[order])
            counts = np.bincount(verts_cell_flat, minlength=8**3)
            cells_vert_start.append(np.concatenate(([0], counts.cumsum())) + offset_vert)
            geoms_origin[i_g] = origin
            geoms_inv_cell_size[i_g] = inv_cell_size
            offset_vert = offset_vert + len(verts)

        verts_spatial_grid = self.collider_info.verts_spatial_grid
        verts_spatial_grid.verts_idx.from_numpy(np.concatenate(verts_idx, dtype=gs.np_int))
        verts_spatial_grid.verts_pos.from_numpy(np.concatenate(verts_pos, dtype=gs.np_float))
        verts_spatial_grid.cells_vert_start.from_numpy(np.concatenate(cells_vert_start, dtype=gs.np_int))
        verts_spatial_grid.geoms_origin.from_numpy(geoms_origin)
        verts_spatial_grid.geoms_inv_cell_size.from_numpy(geoms_inv_cell_size)

    def _init_collision_pair_idx(self, collision_pair_idx):
        if self._n_possible_pairs == 0:
            self.collider_info.collision_pair_idx.fill(-1)
            return
        self.collider_info.collision_pair_idx.from_numpy(collision_pair_idx)

    def _init_verts_connectivity(self, vert_neighbors, vert_neighbor_start, vert_n_neighbors):
        if self._solver.n_verts > 0:
            self.collider_info.vert_neighbors.from_numpy(vert_neighbors)
            self.collider_info.vert_neighbor_start.from_numpy(vert_neighbor_start)
            self.collider_info.vert_n_neighbors.from_numpy(vert_n_neighbors)

    def _init_terrain_state(self):
        if self.collider_config.has_terrain:
            links_idx = self._solver.dyn_info.geoms.link_idx.to_numpy()[
                self._solver.dyn_info.geoms.type.to_numpy() == gs.GEOM_TYPE.TERRAIN
            ]
            entity_idx = self._solver.dyn_info.links.entity_idx.to_numpy()[links_idx[0]]
            if isinstance(entity_idx, np.ndarray):
                entity_idx = entity_idx[0]
            entity = self._solver._entities[entity_idx]

            rc = np.array(entity.terrain_hf.shape, dtype=gs.np_int)
            xyz_maxmin = np.array(
                [
                    rc[0] * entity.terrain_scale[0],
                    rc[1] * entity.terrain_scale[0],
                    entity.terrain_hf.max(),
                    0,
                    0,
                    entity.terrain_hf.min() - 1.0,
                ],
                dtype=gs.np_float,
            )

            self.collider_info.terrain_hf.from_numpy(entity.terrain_hf)
            self.collider_info.terrain_rc.from_numpy(rc)
            self.collider_info.terrain_scale.from_numpy(entity.terrain_scale)
            self.collider_info.terrain_xyz_maxmin.from_numpy(xyz_maxmin)

    def clear(self, envs_idx=None):
        """Remove the contacts of the environments 'envs_idx' (None for all of them), with their forces.

        A hibernated entity keeps the contacts it rests on, with the forces of the last solve it took part in, for as
        long as it sleeps. The contact force of every link is then the sum of the forces of the contacts it keeps.
        """
        self._contact_data_cache.clear()
        # A kernel wipes the live contacts alone, while views over the contact buffers would wipe every slot of them
        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = self._solver._scene._sanitize_envs_idx(envs_idx)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            fn = kernel_masked_collider_clear
        else:
            fn = kernel_collider_clear
        fn(
            envs_idx,
            self._solver.dyn_state,
            self.collider_state,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self._solver.rigid_config,
            self.collider_config,
        )

    def detect_collision(self, env_idx: int = 0) -> np.ndarray:
        """Detect the contacts of the current poses afresh and return the pairs of geoms in contact in an environment."""
        self.clear()
        self.detection()
        n_collision = qd_to_numpy(self.collider_state.n_contacts)[env_idx]
        collision_pairs = np.empty((n_collision, 2), dtype=gs.np_int)
        collision_pairs[:, 0] = qd_to_numpy(self.collider_state.contact_data.geom_a)[:n_collision, env_idx]
        collision_pairs[:, 1] = qd_to_numpy(self.collider_state.contact_data.geom_b)[:n_collision, env_idx]
        return collision_pairs

    def get_contacts(
        self, as_tensor: bool = True, to_torch: bool = True, keep_batch_dim: bool = False, is_padded: bool = False
    ):
        # Early return if already pre-computed
        contact_data = self._contact_data_cache.setdefault((as_tensor, to_torch, is_padded), {})
        if contact_data:
            return contact_data.copy()

        n_envs = self._solver.n_envs
        # When pruning and spatial sort are both statically disabled, contact_sort_idx is guaranteed to stay at the
        # identity permutation, so the physical layout of contact_data already matches the logical order. In that
        # case the zero-copy fast path returns torch views over contact_data storage truncated to n_contacts_max.
        # Otherwise the permutation must be applied: read both contact_data and contact_sort_idx as zero-copy torch
        # views, then materialize each field via a single torch.gather along the contact axis. This still avoids the
        # Quadrants gather kernel and produces a contiguous output suitable for downstream consumers.
        zerocopy_aligned = (
            not self.collider_config.has_prunable_contacts and not self.collider_config.has_non_box_plane_convex_convex
        )
        if gs.use_zerocopy and self._contact_data is not None:
            n_contacts = qd_to_torch(self.collider_state.n_contacts, copy=False)
            # 'is_padded' returns fixed-capacity tensors (plus per-env 'n_contacts'), skipping the
            # 'n_contacts.max().item()' device->host sync that trimming to the live count needs.
            padded = is_padded
            if (as_tensor or n_envs == 0) and not padded:
                n_contacts_max = (n_contacts if n_envs == 0 else n_contacts.max()).item()

            if not zerocopy_aligned:
                # Build a (_B, n_contacts_max) index tensor once, expanded to (_B, n_contacts_max, 3) for vector
                # fields. n_contacts_max comes from the max-across-envs reduction so the same index drives every
                # field; per-env trimming to n_contacts[i] happens in the ragged split below.
                if not (as_tensor or n_envs == 0) and not padded:
                    n_contacts_max = n_contacts.max().item()
                sort_idx_view = qd_to_torch(self.collider_state.contact_sort_idx, transpose=True, copy=False)
                if padded:
                    # Gather the full capacity so no host-side trim (sync) is needed. Sort indices past the live
                    # range may be stale; clamp (out-of-place, preserving the zero-copy view) keeps them in-bounds.
                    gather_idx_flat = sort_idx_view.clamp(0, sort_idx_view.shape[1] - 1)
                else:
                    gather_idx_flat = sort_idx_view[:, :n_contacts_max]
                # FIXME: MPS gathers zeros through an expanded index reading a zero-copy view of the quadrants
                # buffers, while a contiguous one reads correctly. The copy costs one small allocation per call.
                gather_idx_vec = gather_idx_flat.unsqueeze(-1).expand(-1, -1, 3).contiguous()
                # Gather indices past each env's n_contacts are stale (the permutation only fills the live range), so
                # the dense (n_envs, n_contacts_max) tensor has padding columns to reset to the per-field sentinel.
                # The mask is field-independent, so build it once and broadcast over scalar and vector fields alike.
                pad_mask = None
                if as_tensor and n_envs > 0 and not padded:
                    pad_mask = torch.arange(n_contacts_max, device=sort_idx_view.device)[None, :] >= n_contacts[:, None]

            for key, data in self._contact_data.items():
                if zerocopy_aligned:
                    if n_envs == 0:
                        if padded:
                            data = data if keep_batch_dim else data[0]
                        else:
                            data = data[0, :n_contacts_max] if not keep_batch_dim else data[:, :n_contacts_max]
                    elif as_tensor and not padded:
                        data = data[:, :n_contacts_max]
                    if to_torch:
                        if gs.backend == gs.cpu:
                            data = data.clone()
                    else:
                        data = tensor_to_array(data)
                else:
                    # data shape is (_B, max_candidate_contacts) for scalars, with a trailing 3 axis for vectors.
                    gidx = gather_idx_vec if data.dim() == 3 else gather_idx_flat
                    data = data.gather(dim=1, index=gidx)
                    if pad_mask is not None:
                        mask = pad_mask if data.dim() == 2 else pad_mask[..., None]
                        data.masked_fill_(mask, -1 if data.dtype == gs.tc_int else 0)
                    if n_envs == 0 and not keep_batch_dim:
                        data = data[0]
                    if not to_torch:
                        data = tensor_to_array(data)

                if n_envs > 0 and not as_tensor:
                    if padded:
                        data = tuple(data[i : i + 1] if keep_batch_dim else data[i] for i in range(n_envs))
                    elif keep_batch_dim:
                        data = tuple(data[i : i + 1, :j] for i, j in enumerate(n_contacts.tolist()))
                    else:
                        data = tuple(data[i, :j] for i, j in enumerate(n_contacts.tolist()))

                contact_data[key] = data

            if padded:
                contact_data["n_contacts"] = n_contacts if to_torch else tensor_to_array(n_contacts)

            return contact_data.copy()

        # Find out how much dynamic memory must be allocated
        n_contacts = qd_to_numpy(self.collider_state.n_contacts)
        n_contacts_max = n_contacts.max().item()
        # 'is_padded' is honored here too so the result matches the zero-copy path on every backend (this path
        # already syncs to size its buffers, so padding only fixes the shape). Padding needs the dense rectangular
        # layout, so force it even for a ragged (as_tensor=False) request; per-env slices are taken from it below.
        padded = is_padded
        dense = as_tensor or padded
        n_contacts_arr = n_contacts
        if dense:
            out_size = n_contacts_max * max(n_envs, 1)
        else:
            *n_contacts_starts, out_size = np.cumsum(n_contacts)
        n_contacts = n_contacts.tolist()

        # Allocate output buffer
        if to_torch:
            iout = torch.full((out_size, 4), -1, dtype=gs.tc_int, device=gs.device)
            fout = torch.zeros((out_size, 10), dtype=gs.tc_float, device=gs.device)
        else:
            iout = np.full((out_size, 4), -1, dtype=gs.np_int)
            fout = np.zeros((out_size, 10), dtype=gs.np_float)

        # Copy contact data
        if n_contacts_max > 0:
            collider_kernel_get_contacts(iout, fout, self.collider_state, self._solver.rigid_config, dense)

        # Build structured view (no copy)
        if dense:
            nb = max(n_envs, 1)
            iout = iout.reshape((nb, n_contacts_max, 4))
            fout = fout.reshape((nb, n_contacts_max, 10))
            if padded:
                # Widen from the live count to the fixed capacity the zero-copy path returns (contact buffers are
                # sized max(max_candidate_contacts, 1)). New slots keep the sentinel (-1 ints / 0 floats).
                capacity = max(self.collider_info.max_candidate_contacts[None], 1)
                if to_torch:
                    iout_full = torch.full((nb, capacity, 4), -1, dtype=gs.tc_int, device=gs.device)
                    fout_full = torch.zeros((nb, capacity, 10), dtype=gs.tc_float, device=gs.device)
                else:
                    iout_full = np.full((nb, capacity, 4), -1, dtype=gs.np_int)
                    fout_full = np.zeros((nb, capacity, 10), dtype=gs.np_float)
                iout_full[:, :n_contacts_max] = iout
                fout_full[:, :n_contacts_max] = fout
                iout, fout = iout_full, fout_full
            if n_envs == 0 and not keep_batch_dim:
                iout, fout = iout[0], fout[0]
            if as_tensor or n_envs == 0:
                iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
                fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
                values = (*iout_chunks, *fout_chunks)
            else:
                # Ragged + padded, batched: one fixed-capacity slice per env (no host-side trim, no sync).
                iout_envs = (iout[i : i + 1] if keep_batch_dim else iout[i] for i in range(n_envs))
                fout_envs = (fout[i : i + 1] if keep_batch_dim else fout[i] for i in range(n_envs))
                iout_chunks = ((io[..., 0], io[..., 1], io[..., 2], io[..., 3]) for io in iout_envs)
                fout_chunks = ((fo[..., 0], fo[..., 1:4], fo[..., 4:7], fo[..., 7:]) for fo in fout_envs)
                values = (*zip(*iout_chunks), *zip(*fout_chunks))
        else:
            # Split smallest dimension first, then largest dimension
            if n_envs == 0:
                iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
                fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
                values = (*iout_chunks, *fout_chunks)
            elif n_contacts_max >= n_envs:
                if to_torch:
                    iout_chunks = torch.split(iout, n_contacts)
                    fout_chunks = torch.split(fout, n_contacts)
                else:
                    iout_chunks = np.split(iout, n_contacts_starts)
                    fout_chunks = np.split(fout, n_contacts_starts)
                iout_chunks = ((out[..., 0], out[..., 1], out[..., 2], out[..., 3]) for out in iout_chunks)
                fout_chunks = ((out[..., 0], out[..., 1:4], out[..., 4:7], out[..., 7:]) for out in fout_chunks)
                values = (*zip(*iout_chunks), *zip(*fout_chunks))
            else:
                iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
                fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
                if n_envs == 1:
                    values = [(value,) for value in (*iout_chunks, *fout_chunks)]
                else:
                    if to_torch:
                        iout_chunks = (torch.split(out, n_contacts) for out in iout_chunks)
                        fout_chunks = (torch.split(out, n_contacts) for out in fout_chunks)
                    else:
                        iout_chunks = (np.split(out, n_contacts_starts) for out in iout_chunks)
                        fout_chunks = (np.split(out, n_contacts_starts) for out in fout_chunks)
                    values = (*iout_chunks, *fout_chunks)

        # Store contact information in cache
        contact_data.update(
            zip(("link_a", "link_b", "geom_a", "geom_b", "penetration", "position", "normal", "force"), values)
        )

        if padded:
            contact_data["n_contacts"] = (
                torch.as_tensor(n_contacts_arr, device=gs.device) if to_torch else n_contacts_arr
            )

        return contact_data.copy()

    def backward(self, dL_dposition, dL_dnormal, dL_dpenetration):
        func_set_upstream_grad(dL_dposition, dL_dnormal, dL_dpenetration, self.collider_state)
        self.backward_narrowphase()

    def backward_narrowphase(self):
        func_narrow_phase_diff_convex_vs_convex.grad(
            self._solver.dyn_state,
            self.collider_state,
            self.collider_state.diff_contact_input,
            self._solver.dyn_info,
            self._solver.rigid_info,
            self.collider_info,
            self._solver.rigid_config,
            self._solver.errno,
        )


# The order of the contacts kept by the sleepers and of those of the islands woken, which the constraint solver keeps
# when it builds its islands
Collider.api(func_contact_order_set, func_promote_woken_contacts)


@register_system
class Broadphase(System):
    """Update the bounding boxes of the geoms, then find the pairs whose boxes overlap, the first detection stage."""

    slot = Collider.detection.broadphase
    detect: Stages

    def __init__(self, collider: Collider, rigid_solver: RigidSolver) -> None:
        self._collider = collider

    @property
    def detect(self) -> tuple[Callable, ...]:
        if self._collider.has_possible_pairs:
            return func_update_geom_aabbs, func_broad_phase
        return (func_update_geom_aabbs,)


class Narrowphase(System):
    """A System detecting the contacts of a family of pairs of geoms, between the broad phase and the pruning."""

    slot = Collider.detection.narrowphase
    detect: Stages


@register_system
class ConvexConvexFirstContact(Narrowphase):
    """
    Detect the first contact of the convex pairs on queued workers, queueing those holding several (see ConvexConvex).

    The pass runs one thread per core, rounded up to whole environments, an upper-bound estimate of the core count
    sizing it (Genesis-Embodied-AI/Genesis#2616).
    """

    detect = func_narrowphase_contact0

    def __init__(
        self,
        collider: Annotated[Collider, Enabled("has_convex_pairs", narrowphase_layout=Queued)],
        rigid_solver: RigidSolver,
        mpr: MPR[Queued],
        gjk: GJK[Queued, WithoutEPA],
        support_field: SupportField,
    ) -> None:
        gpu_cores, n_envs = collider.collider_config.gpu_cores, rigid_solver._B
        n_threads = n_envs * ((gpu_cores + n_envs - 1) // n_envs)
        mpr.reserve(n_threads)
        gjk.reserve(n_threads)


class ConvexConvex[LayoutT: Layout](Narrowphase):
    """
    Detect the contacts of the convex pairs that no primitive specialization handles, by MPR or GJK.

    A worker per environment detects every contact of its pairs. Queued workers, one thread per core, detect the
    contacts of the pairs that the first-contact pass queued (see ConvexConvexFirstContact).
    """

    def __init__(
        self,
        collider: Annotated[Collider, Enabled("has_convex_pairs", narrowphase_layout=LayoutT)],
        rigid_solver: RigidSolver,
        mpr: MPR[LayoutT],
        gjk: GJK[LayoutT, WithEPA],
        support_field: SupportField,
    ) -> None:
        if self.template_args[LayoutT] is Queued:
            mpr.reserve(collider.collider_config.gpu_cores)
            gjk.reserve(collider.collider_config.gpu_cores)

    @property
    def detect(self) -> Callable:
        if self.template_args[LayoutT] is Queued:
            return func_narrowphase_multicontact
        return func_narrow_phase_convex_vs_convex


register_system(ConvexConvex[PerEnv])
register_system(ConvexConvex[Queued])


@register_system
class PrimitivePrimitive(Narrowphase):
    """Detect the contacts of the pairs of primitive geoms that an analytic specialization handles."""

    detect = func_narrow_phase_convex_specializations

    def __init__(
        self, collider: Annotated[Collider, Enabled("has_primitive_pairs")], rigid_solver: RigidSolver
    ) -> None:
        pass


@register_system
class ConvexTerrain(Narrowphase):
    """Detect the contacts of the geoms with a terrain, by MPR on queued workers (see 'Collider.n_terrain_threads')."""

    detect = func_narrow_phase_any_vs_terrain

    def __init__(
        self,
        collider: Annotated[Collider, Enabled("has_terrain_pairs")],
        rigid_solver: RigidSolver,
        mpr: MPR[Queued],
        support_field: SupportField,
    ) -> None:
        mpr.reserve(collider.n_terrain_threads)


@register_system
class NonconvexNonconvex(Narrowphase):
    """Detect the contacts of the pairs holding a non-convex geom, by the signed distance field (SDF) of the geoms."""

    detect = func_narrow_phase_nonconvex_vs_nonterrain

    def __init__(
        self, collider: Annotated[Collider, Enabled("has_nonconvex_pairs")], rigid_solver: RigidSolver, sdf: SDF
    ) -> None:
        pass


class ConvexConvexSDF(ConvexConvexFirstContact, ConvexConvex[PerEnv], ConvexConvex[Queued]):
    """
    Detect the contacts of the convex pairs of ConvexConvex by the signed distance field (SDF) of the geoms, in place of
    the convex narrowphases of both layouts, which it supersedes once requested (see 'Simulator.get_system').

    A pair touches at the deepest vertex of either geom in the SDF of the other, found by descending that SDF over the
    edge graph of the geom, a sphere touching at the exact point wherever the SDF of the other geom is exact (see
    func_contact_sdf). The contacts carry no input for the differentiable reverse of the convex narrowphase.
    """

    is_differentiable = False
    detect = func_narrow_phase_convex_vs_convex_sdf

    def __init__(
        self, collider: Annotated[Collider, Enabled("has_convex_pairs")], rigid_solver: RigidSolver, sdf: SDF
    ) -> None:
        pass


class NonconvexNonconvexSDF(NonconvexNonconvex):
    """
    Detect the contacts of the pairs holding a non-convex geom by the deepest vertex of the geom with the smaller
    bounding box in the signed distance field (SDF) of the other, superseding NonconvexNonconvex once requested (see
    'Simulator.get_system' and func_narrow_phase_nonconvex_vs_nonterrain_sdf).
    """

    detect = func_narrow_phase_nonconvex_vs_nonterrain_sdf


@register_system
class Pruning(System):
    """Clamp the contacts of every environment to its budget and prune those of each pair of links, the last stage."""

    slot = Collider.detection.pruning
    prune: Stages

    def __init__(self, collider: Annotated[Collider, Enabled("has_possible_pairs")], rigid_solver: RigidSolver) -> None:
        self._collider = collider
        self._requires_grad = rigid_solver.requires_grad

    @property
    def prune(self) -> tuple[Callable, ...]:
        prune = func_clamp_prune_contacts_coop if self._collider.is_pruning_cooperative else func_clamp_prune_contacts
        # Plane-convex and sphere-sphere contacts come from analytic paths that leave diff_contact_input unfilled, so it
        # is filled here for the differentiable narrow-phase reverse (see func_fill_diff_contact_input_analytic)
        if self._requires_grad:
            return prune, func_fill_diff_contact_input_analytic
        return (prune,)


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.collider_decomp")
