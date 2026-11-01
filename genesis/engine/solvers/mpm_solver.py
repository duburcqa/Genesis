from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.boundaries import CubeBoundary, func_impose_cube_boundary
from genesis.engine.boundaries.boundaries import CubeBoundaryBox
from genesis.engine.entities import MPMEntity
from genesis.engine.materials import MPM
from genesis.engine.pipeline import Inline, Stages, System
from genesis.engine.solvers.base_solver import FRAMES, func_flag_error
from genesis.engine.solvers.particle_solver import PARTICLES, ParticlesRender, VVertsRender, VVertsSupport
from genesis.engine.state_change import StateChange, mutates
from genesis.engine.states.solvers import MPMSolverState
from genesis.engine.step import Coupling
from genesis.grad.bridge import no_grad, register_custom_backward
from genesis.options.solvers import MPMOptions
from genesis.utils.data import ENVS, Data, DataKind, Label, Size, StaticConfig, field
from genesis.utils.misc import make_static_tuple, qd_isfinite, qd_to_torch, qd_zero_grad

from .particle_solver import ParticleSolver, kernel_add_vverts

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class MPMErrorCode(IntEnum):
    """The error flags the material point method (MPM) solver writes in its errno (see 'Solver.register_errors')."""

    INVALID_PARTICLE_NAN = 0b00000000000000000000000100000000


# The cells of the grid of the material point method (MPM) solver along each axis
GRID_X = Size("grid_x")
GRID_Y = Size("grid_y")
GRID_Z = Size("grid_z")
# The scratch of the sparse reset of the grid, without gradient alone (see MPMGrid)
SPARSE_GRID_RESET = Label("sparse_grid_reset")


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


class MPMState(Data, kind=DataKind.STATE):
    """Runtime state of the material point method (MPM) solver."""

    particles: MPMParticles
    grid: MPMGrid
    particles_render: ParticlesRender
    vverts_render: VVertsRender
    # The errors of each env, raised at the beginning of the next step (see 'Solver.register_errors')
    errno: qd.Tensor = field(int, shape=(ENVS,))


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
    EPS: qd.Tensor = field(float, shape=())


class MPMStaticConfig(StaticConfig):
    """The compile-time configuration of the kernels of the material point method (MPM) solver."""

    # Whether the singular value decomposition of the deformation gradient is computed, which a material reading it
    # and the backward pass require. Otherwise J is the determinant of the advected deformation gradient.
    is_svd_needed: bool
    # Whether the grid of every substep is kept for the backward pass, instead of being zeroed sparsely (see MPMGrid)
    requires_grad: bool


@qd.data_oriented
class MPMMaterialDispatcher:
    """
    Dispatcher of the particles to the update of the deformation and the stress of their material.

    The kernels take it as a template, which compiles one branch per distinct material of the scene, the index of a
    material being its position in `materials`.
    """

    def __init__(self, materials: "list[MPM.Base]"):
        self.materials = make_static_tuple(materials)

    @qd.func
    def update_F_S_Jp(
        self,
        i_mat: int,
        J: float,
        F_tmp: qd.types.matrix(3, 3),
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        Jp: float,
    ):
        """Return the deformation gradient, its singular values and the plastic volume ratio once plasticity applies.

        The particle is of the material `i_mat`, the other arguments being those of 'MPM.Base.update_F_S_Jp'.
        """
        F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        Jp_new = gs.qd_float(1.0)
        for j_mat in qd.static(range(len(self.materials))):
            if i_mat == j_mat:
                F_new, S_new, Jp_new = self.materials[j_mat].update_F_S_Jp(J, F_tmp, U, S, V, Jp)
        return F_new, S_new, Jp_new

    @qd.func
    def update_stress(
        self,
        i_mat: int,
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        F_tmp: qd.types.matrix(3, 3),
        F_new: qd.types.matrix(3, 3),
        J: float,
        Jp: float,
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        """Return the stress of a particle.

        The particle is of the material `i_mat`, the other arguments being those of 'MPM.Base.update_stress'.
        """
        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        for j_mat in qd.static(range(len(self.materials))):
            if i_mat == j_mat:
                stress = self.materials[j_mat].update_stress(U, S, V, F_tmp, F_new, J, Jp, actu, m_dir)
        return stress


@qd.func
def func_mpm_cell_separation(f: int, i_p: int, i_b: int, i_c: int, cell_pos: qd.types.vector(3), i_sep: int) -> int:
    """
    Return what separates a particle from a cell of its stencil, -1 for nothing, the cell then getting nothing from it.

    `i_c` indexes the cell among the 27 of the stencil, at `cell_pos`. The placeholder of the extension point of the
    transfer from the particles to the grid, which the coupling Systems add their funcs to (see 'MPMSolver.p2g'), each
    leaving a separation found by an earlier one as it is.
    """
    return i_sep


@qd.func
def func_mpm_cell_velocity(
    f: int, i_p: int, i_b: int, i_c: int, weight: float, grid_vel: qd.types.vector(3)
) -> qd.types.vector(3):
    """
    Return the velocity a cell of the stencil of a particle contributes to it, of the given weight.

    `i_c` indexes the cell among the 27 of the stencil. The placeholder of the extension point of the transfer from the
    grid to the particles, which the coupling Systems add their funcs to (see 'MPMSolver.g2p').
    """
    return grid_vel


@qd.kernel
def kernel_p2g(
    f: int,
    mpm_state: MPMState,
    mpm_info: MPMInfo,
    mpm_static_config: qd.template(),
    material_dispatcher: qd.template(),
):
    """
    Update the deformation of the active particles of a frame, then scatter their momentum and mass to the grid.

    The material of each particle updates its deformation gradient and plastic volume ratio into the next frame and
    gives its stress, which enters the momentum. A cell separated from a particle gets nothing from it (see
    func_mpm_cell_separation). Without gradient, the cells first touched are listed for the sparse reset (see
    MPMGrid).
    """
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            substep_dt = mpm_info.substep_dt[None]
            dx = mpm_info.dx[None]
            inv_dx = mpm_info.inv_dx[None]
            grid_offset = mpm_info.grid_offset[None]

            # The determinant of F_tmp equals that of S, since U and V of the singular value decomposition are proper
            # rotations (see MPMStaticConfig). J is declared before the branches, which quadrants scopes locally.
            J = gs.qd_float(0.0)
            if qd.static(mpm_static_config.is_svd_needed):
                J = mpm_state.particles.S[f, i_p, i_b].determinant()
            else:
                J = mpm_state.particles.F_tmp[f, i_p, i_b].determinant()
            i_mat = mpm_info.particles.material_idx[i_p]
            F_new, S_new, Jp_new = material_dispatcher.update_F_S_Jp(
                i_mat,
                J,
                mpm_state.particles.F_tmp[f, i_p, i_b],
                mpm_state.particles.U[f, i_p, i_b],
                mpm_state.particles.S[f, i_p, i_b],
                mpm_state.particles.V[f, i_p, i_b],
                mpm_state.particles.Jp[f, i_p, i_b],
            )
            mpm_state.particles.F[f + 1, i_p, i_b] = F_new
            mpm_state.particles.Jp[f + 1, i_p, i_b] = Jp_new
            # The actuation holds over every substep of the step it is set at, and until it is set again
            mpm_state.particles.actu[f + 1, i_p, i_b] = mpm_state.particles.actu[f, i_p, i_b]

            # The stress takes both the advected deformation gradient and the one once plasticity applies: the elastic
            # materials read the latter, while a viscous liquid reads the former, whose shear the plasticity removes.
            # The plastic volume ratio is that of the frame the substep starts from.
            stress = material_dispatcher.update_stress(
                i_mat,
                mpm_state.particles.U[f, i_p, i_b],
                S_new,
                mpm_state.particles.V[f, i_p, i_b],
                mpm_state.particles.F_tmp[f, i_p, i_b],
                F_new,
                J,
                mpm_state.particles.Jp[f, i_p, i_b],
                mpm_state.particles.actu[f, i_p, i_b],
                mpm_info.particles.muscle_direction[i_p],
            )
            stress = (-substep_dt * mpm_info.particle_volume[None] * 4 * inv_dx * inv_dx) * stress
            affine = stress + mpm_info.particles.mass[i_p] * mpm_state.particles.C[f, i_p, i_b]

            base, fx, w = func_quadratic_bspline_stencil(mpm_state.particles.pos[f, i_p, i_b], inv_dx)
            for offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                dpos = (offset.cast(gs.qd_float) - fx) * dx
                weight = gs.qd_float(1.0)
                for d in qd.static(range(3)):
                    weight *= w[offset[d], d]

                i_sep = -1
                i_sep = func_mpm_cell_separation(
                    f, i_p, i_b, offset[0] * 9 + offset[1] * 3 + offset[2], (base + offset) * dx, i_sep
                )
                cell_ijk = base - grid_offset + offset
                if i_sep == -1:
                    mpm_state.grid.vel_in[f, cell_ijk[0], cell_ijk[1], cell_ijk[2], i_b] += weight * (
                        mpm_info.particles.mass[i_p] * mpm_state.particles.vel[f, i_p, i_b] + affine @ dpos
                    )
                    mass_contrib = weight * mpm_info.particles.mass[i_p]
                    prev_mass = qd.atomic_add(
                        mpm_state.grid.mass[f, cell_ijk[0], cell_ijk[1], cell_ijk[2], i_b], mass_contrib
                    )
                    # The first contribution of an env to a cell claims the cell in the dirty flag shared by the envs,
                    # and the first env claiming it lists it (see MPMGrid). The masses are non-negative.
                    if qd.static(not mpm_static_config.requires_grad):
                        if prev_mass <= 0.0 and mass_contrib > 0.0:
                            dirty_flag_prev = qd.atomic_or(
                                mpm_state.grid.dirty_flag[cell_ijk[0], cell_ijk[1], cell_ijk[2]], gs.qd_int(1)
                            )
                            if dirty_flag_prev == 0:
                                i_slot = qd.atomic_add(mpm_state.grid.dirty_count[0], 1)
                                grid_res_y, grid_res_z = mpm_state.grid.mass.shape[2], mpm_state.grid.mass.shape[3]
                                i_cell = (cell_ijk[0] * grid_res_y + cell_ijk[1]) * grid_res_z + cell_ijk[2]
                                mpm_state.grid.dirty_list[i_slot] = i_cell

                # The particles held in place act as boundary conditions of the grid
                if not mpm_info.particles.is_free[i_p]:
                    mpm_state.grid.vel_in[f, cell_ijk[0], cell_ijk[1], cell_ijk[2], i_b] = qd.Vector.zero(
                        gs.qd_float, 3
                    )


@qd.kernel(fastcache=True)
def kernel_g2p(f: int, mpm_state: MPMState, mpm_info: MPMInfo, mpm_static_config: qd.template()):
    """
    Gather the velocity of the grid back to the active particles of a frame and advect them into the next frame.

    A cell contributes the velocity the couplings give it (see func_mpm_cell_velocity). The inactive particles carry
    their state over to the next frame unchanged.
    """
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            substep_dt = mpm_info.substep_dt[None]
            inv_dx = mpm_info.inv_dx[None]
            grid_offset = mpm_info.grid_offset[None]

            base, fx, w = func_quadratic_bspline_stencil(mpm_state.particles.pos[f, i_p, i_b], inv_dx)
            new_vel = qd.Vector.zero(gs.qd_float, 3)
            new_C = qd.Matrix.zero(gs.qd_float, 3, 3)
            for offset in qd.static(qd.grouped(qd.ndrange(3, 3, 3))):
                dpos = offset.cast(gs.qd_float) - fx
                cell_ijk = base - grid_offset + offset
                grid_vel = mpm_state.grid.vel_out[f, cell_ijk[0], cell_ijk[1], cell_ijk[2], i_b]
                weight = gs.qd_float(1.0)
                for d in qd.static(range(3)):
                    weight *= w[offset[d], d]

                grid_vel = func_mpm_cell_velocity(
                    f, i_p, i_b, offset[0] * 9 + offset[1] * 3 + offset[2], weight, grid_vel
                )

                new_vel += weight * grid_vel
                new_C += 4 * inv_dx * weight * grid_vel.outer_product(dpos)

            new_pos = mpm_state.particles.pos[f, i_p, i_b] + substep_dt * new_vel

            # The boundary keeps an exploding simulation from reading cells out of the grid
            new_pos, new_vel = func_impose_cube_boundary(new_pos, new_vel, mpm_info.boundary)

            mpm_state.particles.vel[f + 1, i_p, i_b] = new_vel
            mpm_state.particles.C[f + 1, i_p, i_b] = new_C
            mpm_state.particles.pos[f + 1, i_p, i_b] = new_pos
        else:
            mpm_state.particles.vel[f + 1, i_p, i_b] = mpm_state.particles.vel[f, i_p, i_b]
            mpm_state.particles.pos[f + 1, i_p, i_b] = mpm_state.particles.pos[f, i_p, i_b]
            mpm_state.particles.C[f + 1, i_p, i_b] = mpm_state.particles.C[f, i_p, i_b]
            mpm_state.particles.F[f + 1, i_p, i_b] = mpm_state.particles.F[f, i_p, i_b]
            mpm_state.particles.Jp[f + 1, i_p, i_b] = mpm_state.particles.Jp[f, i_p, i_b]

        mpm_state.particles.is_active[f + 1, i_p, i_b] = mpm_state.particles.is_active[f, i_p, i_b]


@qd.func
def func_mpm_grid_couplings(
    f: int, i_b: int, pos: qd.types.vector(3), vel: qd.types.vector(3), mass: float
) -> qd.types.vector(3):
    """
    Return the velocity of a cell of the grid holding mass, at the given position and of the given mass, once coupled
    with the other solvers.

    The placeholder of the extension point of the couplings of the grid, which the coupling Systems add their funcs to
    (see 'MPMSolver.grid_couplings').
    """
    return vel


@qd.kernel
def kernel_mpm_grid_op(
    f: int,
    t: float,
    mpm_state: MPMState,
    mpm_info: MPMInfo,
    force_fields_dispatcher: qd.template(),
):
    """
    Update the velocity of the cells of the material point method (MPM) grid holding mass.

    The momentum of a cell turns into a velocity, which gravity and the force fields accelerate, the couplings with the
    other solvers update (see func_mpm_grid_couplings), and the boundary of the solver clamps.
    """
    for i_x, i_y, i_z, i_b in qd.ndrange(
        mpm_state.grid.mass.shape[1],
        mpm_state.grid.mass.shape[2],
        mpm_state.grid.mass.shape[3],
        mpm_state.grid.mass.shape[4],
    ):
        if mpm_state.grid.mass[f, i_x, i_y, i_z, i_b] > mpm_info.EPS[None]:
            substep_dt = mpm_info.substep_dt[None]
            vel = (1 / mpm_state.grid.mass[f, i_x, i_y, i_z, i_b]) * mpm_state.grid.vel_in[f, i_x, i_y, i_z, i_b]
            vel += substep_dt * mpm_info.gravity[i_b]
            pos = (qd.Vector([i_x, i_y, i_z], dt=gs.qd_int) + mpm_info.grid_offset[None]) * mpm_info.dx[None]
            mass = mpm_state.grid.mass[f, i_x, i_y, i_z, i_b] / mpm_info.particle_volume_scale[None]
            # TODO: each field acts on the velocity the previous one updated, while SPH and PBD sum them at the same
            # state. Several velocity-dependent fields tell the schemes apart, and which one to keep is undecided.
            for i_ff in qd.static(range(len(force_fields_dispatcher.fields))):
                vel += force_fields_dispatcher.fields[i_ff].get_acc(pos, vel, t, i=-1) * substep_dt
            vel = func_mpm_grid_couplings(f, i_b, pos, vel, mass)
            _, mpm_state.grid.vel_out[f, i_x, i_y, i_z, i_b] = func_impose_cube_boundary(pos, vel, mpm_info.boundary)


class MPMSolver(ParticleSolver, Coupling):
    material_cls = MPM.Base
    has_frames = True

    # The extension points of the transfers between the particles and the grid, and of the update of the grid, filled
    # by the Systems playing CellSeparation, GridCoupling and CellVelocity
    p2g = Inline(kernel_p2g, func_mpm_cell_separation)
    grid_couplings = Inline(kernel_mpm_grid_op, func_mpm_grid_couplings)
    g2p = Inline(kernel_g2p, func_mpm_cell_velocity)

    mpm_static_config: MPMStaticConfig | None
    mpm_state: MPMState | None
    mpm_info: MPMInfo | None

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: "MPMOptions"):
        super().__init__(scene, sim, options)

        # options
        self._grid_density = options.grid_density
        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)
        self._enable_CPIC = options.enable_CPIC

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors

        # The particle volume is scaled up for numerical robustness (see MPMInfo)
        self._particle_volume_real = float(self._particle_size**3)
        self._particle_volume_scale = 1e3
        self._particle_volume = self._particle_volume_real * self._particle_volume_scale

        # Other derived parameters
        self._dx = float(1.0 / self._grid_density)
        self._inv_dx = float(self._grid_density)
        self._lower_bound_cell = np.round(self._grid_density * self._lower_bound).astype(gs.np_int)
        self._upper_bound_cell = np.round(self._grid_density * self._upper_bound).astype(gs.np_int)
        self._grid_res = self._upper_bound_cell - self._lower_bound_cell + 1  # +1 to include both corner
        if np.prod(self._grid_res) > 1e9:
            gs.raise_exception(
                "Grid size larger than 1e9 not supported by MPM solver. Please reduce 'grid_density', or set tighter "
                "boundaries via 'lower_bound' / 'upper_bound'."
            )

        # The distinct materials of the entities, the kernels dispatching each particle to the one its index names
        self._materials: list[MPM.Base] = []

        # boundary
        self.setup_boundary()

        self.mpm_state = None
        self.mpm_info = None
        self.mpm_static_config = None
        self.material_dispatcher: MPMMaterialDispatcher | None = None

    def setup_boundary(self):
        # safety padding
        self.boundary_padding = 3 * self._dx
        self.boundary = CubeBoundary(
            lower=self._lower_bound + self.boundary_padding, upper=self._upper_bound - self.boundary_padding
        )

    def reset_grad(self):
        # Every adjoint the backward pass writes accumulates, so a pass starts from zeros
        if not self._sim.requires_grad:
            return
        qd_zero_grad(self.mpm_state.particles)
        qd_zero_grad(self.mpm_state.grid)
        # One flush for the zeroing batch, see qd_zero_grad in misc.py
        if gs.use_zerocopy.is_supported and gs.backend == gs.metal:
            torch.mps.synchronize()

    def build(self):
        super().build()

        # particles and entities
        self._B = self._sim._B
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

        # The singular value decomposition always runs under requires_grad, its backward being registered with the
        # kernel computing it (see MPMStaticConfig).
        # TODO: validate the J = det(F_tmp) shortcut composes with quadrants autograd before enabling it there.
        is_svd_needed = self._sim.requires_grad or any([material.is_svd_needed for material in self._materials])

        # The couplers take the data of the solver whether it holds entities or not, an inactive solver leaving its
        # arrays unused
        requires_grad = self._sim.requires_grad
        unused = {None: True}
        if self.is_active:
            unused = {SPARSE_GRID_RESET: requires_grad}
        grid_x, grid_y, grid_z = self._grid_res
        self.mpm_state = MPMState(
            sizes=dict(
                frames=self.n_frames,
                particles=self._n_particles,
                envs=self._B,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_z=grid_z,
                vverts=self._n_vverts,
            ),
            unused=unused,
            requires_grad=requires_grad,
        )
        # The arrays of the model an inactive solver leaves unused, its boundary and constants staying
        info_arrays = (MPMInfo.particles, MPMInfo.vverts, MPMInfo.gravity)
        self.mpm_info = MPMInfo(
            sizes=dict(
                particles=self._n_particles, envs=self._B, vverts=self._n_vverts, vvert_supports=self._n_vvert_supports
            ),
            values=dict(
                boundary=dict(
                    lower=self.boundary.lower, upper=self.boundary.upper, restitution=self.boundary.restitution
                ),
                substep_dt=self.substep_dt,
                dx=self.dx,
                inv_dx=self.inv_dx,
                grid_offset=self._lower_bound_cell,
                particle_volume=self.particle_volume,
                particle_volume_scale=self.particle_volume_scale,
                EPS=gs.EPS,
            ),
            unused={info_arrays: not self.is_active},
        )
        self.mpm_static_config = MPMStaticConfig(is_svd_needed=is_svd_needed, requires_grad=self._sim.requires_grad)
        self.material_dispatcher = MPMMaterialDispatcher(self._materials)
        self._build_gravity(self.mpm_info.gravity)

        if self.is_active:
            self.register_errors(
                self.mpm_state.errno,
                {
                    MPMErrorCode.INVALID_PARTICLE_NAN: (
                        "NaN detected in MPM states. Try reducing the time step size or adjusting simulation "
                        "parameters."
                    ),
                },
            )

            for entity in self._entities:
                entity._add_to_solver()

            # Empirical bound on the substep of an explicit MPM integration, as a fraction of the cell size
            suggested_dt = 2e-2 * self._dx
            if self.substep_dt > suggested_dt:
                gs.logger.warning(
                    f"Current `substep_dt` ({self.substep_dt:.6g}) is greater than suggested_dt ({suggested_dt:.6g}, "
                    "calculated based on `grid_density`). Simulation might be unstable."
                )

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_particles > 0

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "MPMEntity":
        self.add_material(material)

        # create entity
        entity = MPMEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            particle_size=self._particle_size,
            idx=idx,
            particle_start=self.n_particles,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            name=name,
        )
        self._entities.append(entity)

        return entity

    def add_material(self, material):
        # A material equal to one already registered shares its index, and so the kernels it compiles into
        for material_i in self._materials:
            if material == material_i:
                material.idx = material_i.idx
                break
        else:
            material.idx = len(self._materials)
            self._materials.append(material)

    def add_vverts(self, vvert_start, particle_start, vverts, particles, P_invs, support_idxs_local):
        kernel_add_vverts(
            vvert_start, particle_start, vverts, particles, P_invs, support_idxs_local, self.mpm_info.vverts
        )

    def add_particles(self, f, active, particle_start, n_particles, material_idx, default_Jp, rho, pos):
        """Add the particles of an entity at rest to the given frame, with the parameters of its material."""
        kernel_add_particles(
            f, particle_start, material_idx, n_particles, active, default_Jp, rho, pos, self.mpm_state, self.mpm_info
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if self._sim.requires_grad:
            kernel_reset_grid_and_grad(f, self.mpm_state)
            kernel_compute_F_tmp(f, self.mpm_state, self.mpm_info, compute_svd=False)
            kernel_svd(f, self.mpm_state)
        else:
            # The grid is left zeroed by kernel_reset_dirty_cells at the end of the previous substep, and is zero
            # initialized before the first one
            kernel_compute_F_tmp(f, self.mpm_state, self.mpm_info, compute_svd=self.mpm_static_config.is_svd_needed)
        self.p2g(f=f)

    @property
    def couple(self) -> Inline:
        """The update of the velocity of the grid, which runs the couplings with the other solvers (see GridCoupling)."""
        return self.grid_couplings

    @property
    def has_hybrid_entities(self) -> bool:
        """Whether the solver holds the soft part of hybrid entities that their rigid part drives (see RigidMPMHybrid)."""
        return any(
            [
                isinstance(entity.material, gs.materials.Hybrid) and not entity.material.use_default_coupling
                for entity in self._sim.entities
            ]
        )

    def substep_post_coupling(self, f):
        self.g2p(f=f)

        # The grid is read no more in this substep (see MPMGrid)
        if not self._sim.requires_grad:
            kernel_reset_dirty_cells(f, self.mpm_state)
            if gs.use_zerocopy:
                dirty_count = qd_to_torch(self.mpm_state.grid.dirty_count, copy=False)
                dirty_count.zero_()
                # The next kernel reads the count (see qd_zero_grad in misc.py)
                if gs.backend == gs.metal:
                    torch.mps.synchronize()
            else:
                self.mpm_state.grid.dirty_count.fill(0)

        # The positions a step ends with are checked once, an error being raised at the beginning of the next step
        if (self._sim._cur_substep_global + 1) % self._substeps == 0:
            kernel_flag_non_finite_pos(f + 1, self.mpm_state, self._errno)

    def copy_frame(self, source, target):
        kernel_copy_frame(source, target, self.mpm_state)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    is_differentiable = True

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            kernel_set_state(
                f, state.pos, state.vel, state.C, state.F, state.Jp, state.actu, state.active, self.mpm_state
            )
            self.clear_errno(envs_idx)

    def get_state(self, f):
        if not self.is_active:
            return None

        particles = self.mpm_state.particles
        pos, vel, C, F, Jp = self._read_data(
            (particles.pos, particles.vel, particles.C, particles.F, particles.Jp), envs_idx=None, idx=None, copy=True
        )
        # The actuation and the activity are inputs of the step, which carry no gradient back
        actu, active = [
            qd_to_torch(field, row_mask=0, keepdim=False).movedim(1, 0).clone(memory_format=torch.contiguous_format)
            for field in (particles.actu, particles.is_active)
        ]
        return MPMSolverState(self._scene, pos, vel, C, F, Jp, actu, active)

    def update_render_fields(self):
        kernel_update_render_fields(self.sim.cur_substep_local, self.mpm_state, self.mpm_info)

    @gs.assert_built
    @mutates()
    def set_particles_actu(self, actus, particles_idx=None, envs_idx=None, *, n_groups=1):
        """
        Set the actuation of some particles, each particle taking the actuation of its muscle group.

        Parameters
        ----------
        actus : array_like, shape ([M,] [N,] [n_groups,])
            Actuation of each muscle group, for each particle.
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles (see 'set_particles_pos'). If None, all particles will be considered.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        n_groups : int, optional
            The number of muscle groups. Defaults to 1.
        """
        actus, particles_idx, envs_idx = self._sanitize_io_variables(
            actus, particles_idx, self.n_particles, "particles_idx", envs_idx, (n_groups,), batched_idx=True
        )
        kernel_set_particles_actu(particles_idx, envs_idx, actus, self.mpm_state, self.mpm_info)

    def get_particles_actu(self, particles_idx=None, envs_idx=None):
        """Get the actuation of some particles (see 'set_particles_actu')."""
        actus, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True
        )
        kernel_get_particles_actu(particles_idx, envs_idx, actus, self.mpm_state)
        return actus[0] if self._scene.n_envs == 0 else actus

    @mutates(StateChange.MODEL)
    def set_particles_muscle_group(self, muscle_group, particles_idx=None):
        """Set the muscle group of some particles, shared by every environment."""
        muscle_group, particles_idx, _ = self._sanitize_io_variables(
            muscle_group, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_int, batched=False
        )
        kernel_set_particles_muscle_group(particles_idx, muscle_group, self.mpm_info)

    def get_particles_muscle_group(self, particles_idx=None):
        """Get the muscle group of some particles (see 'set_particles_muscle_group')."""
        muscle_group, particles_idx, _ = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_int, batched=False
        )
        kernel_get_particles_muscle_group(particles_idx, muscle_group, self.mpm_info)
        return muscle_group

    @mutates(StateChange.MODEL)
    def set_particles_muscle_direction(self, muscle_direction, particles_idx=None):
        """Set the unit direction of the muscle fibers of some particles, shared by every environment."""
        muscle_direction, particles_idx, _ = self._sanitize_io_variables(
            muscle_direction, particles_idx, self.n_particles, "particles_idx", element_shape=(3,), batched=False
        )
        kernel_set_particles_muscle_direction(particles_idx, muscle_direction, self.mpm_info)

    @mutates(StateChange.MODEL)
    def set_particles_free(self, free, particles_idx=None):
        """Set whether some particles move freely or stay fixed, shared by every environment."""
        free, particles_idx, _ = self._sanitize_io_variables(
            free, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_bool, batched=False
        )
        kernel_set_particles_free(particles_idx, free, self.mpm_info)

    def get_particles_free(self, particles_idx=None):
        """Get whether some particles move freely (see 'set_particles_free')."""
        free, particles_idx, _ = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", dtype=gs.tc_bool, batched=False
        )
        kernel_get_particles_free(particles_idx, free, self.mpm_info)
        return free

    def _kernel_set_particles_pos(self, particles_idx, envs_idx, poss):
        kernel_set_particles_pos(particles_idx, envs_idx, poss, self.mpm_state, self.mpm_info)

    def _kernel_get_particles_pos(self, particles_idx, envs_idx, poss):
        kernel_get_particles_pos(particles_idx, envs_idx, poss, self.mpm_state)

    def _kernel_set_particles_vel(self, particles_idx, envs_idx, vels):
        kernel_set_particles_vel(particles_idx, envs_idx, vels, self.mpm_state)

    def _kernel_get_particles_vel(self, particles_idx, envs_idx, vels):
        kernel_get_particles_vel(particles_idx, envs_idx, vels, self.mpm_state)

    def _kernel_set_particles_active(self, particles_idx, envs_idx, actives):
        kernel_set_particles_active(particles_idx, envs_idx, actives, self.mpm_state)

    def _kernel_get_particles_active(self, particles_idx, envs_idx, actives):
        kernel_get_particles_active(particles_idx, envs_idx, actives, self.mpm_state)

    def _kernel_get_particles_mass(self, particles_idx, envs_idx, masses):
        kernel_get_particles_mass(particles_idx, envs_idx, masses, self.mpm_info)

    # ------------------------------------------------------------------------------------
    # -------------------------------- particle constraints ------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_particle_constraints(self, particles_mask, link_idx, stiffness, *, particles_start=0):
        """
        Attach some particles to a rigid link with critically damped springs.

        Each particle is pulled toward the position it holds now, expressed in the frame of the link.

        Parameters
        ----------
        particles_mask : array_like, shape ([n_envs,] n_particles)
            Boolean mask over consecutive particles of the solver, selecting those to attach.
        link_idx : int
            Global index of the rigid link.
        stiffness : float
            Stiffness of the springs.
        particles_start : int, optional
            Global index of the particle the mask starts at. Defaults to 0.
        """
        # Imported here to break the import cycle between the couplings and the solvers they couple
        from genesis.engine.couplers.pairwise import RigidMPMAttachment

        attachment = self._sim.get_system(RigidMPMAttachment)
        if attachment is None:
            gs.raise_exception("Attaching particles to a link requires a scene simulating rigid entities.")
        attachment.attach(particles_mask, link_idx, stiffness, particles_start=particles_start)

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def remove_particle_constraints(self, particles_mask=None, *, particles_start=0):
        """
        Detach some particles from the rigid link they are attached to (see 'set_particle_constraints').

        Parameters
        ----------
        particles_mask : None | array_like, shape ([n_envs,] n_particles), optional
            Boolean mask over consecutive particles of the solver, selecting those to detach. If None, all particles
            from `particles_start` on will be detached. Defaults to None.
        particles_start : int, optional
            Global index of the particle the mask starts at. Defaults to 0.
        """
        # Imported here to break the import cycle between the couplings and the solvers they couple
        from genesis.engine.couplers.pairwise import RigidMPMAttachment

        attachment = self._sim.get_system(RigidMPMAttachment)
        if attachment is not None:
            attachment.detach(particles_mask, particles_start=particles_start)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum(entity.n_particles for entity in self._entities)

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
    def grid_density(self):
        return self._grid_density

    @property
    def particle_size(self):
        return self._particle_size

    @property
    def particle_radius(self):
        return self._particle_size / 2.0

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def dx(self):
        return self._dx

    @property
    def inv_dx(self):
        return self._inv_dx

    @property
    def particle_volume_real(self):
        return self._particle_volume_real

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def particle_volume_scale(self):
        return self._particle_volume_scale

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def lower_bound_cell(self):
        return self._lower_bound_cell

    @property
    def upper_bound_cell(self):
        return self._upper_bound_cell

    @property
    def grid_res(self):
        return self._grid_res

    @property
    def enable_CPIC(self):
        return self._enable_CPIC


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_quadratic_bspline_stencil(pos: qd.types.vector(3), inv_dx: float):
    """Return the quadratic B-spline stencil of the grid around a position.

    The stencil spans the 3x3x3 cells from the returned lower cell on. The position lies at the returned offset from
    that cell, in units of cells, and the weight of the cell at offset (i, j, k) from it is the product of the entries
    (i, 0), (j, 1) and (k, 2) of the returned matrix.
    """
    base = qd.floor(pos * inv_dx - 0.5).cast(gs.qd_int)
    fx = pos * inv_dx - base.cast(gs.qd_float)
    w = qd.Matrix.rows([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2])
    return base, fx, w


@qd.kernel(fastcache=True)
def kernel_compute_F_tmp(f: int, mpm_state: MPMState, mpm_info: MPMInfo, compute_svd: qd.template()):
    """
    Advect the deformation gradient of the active particles of a frame over the substep, then optionally decompose it.

    The decomposition in the same kernel keeps the advected deformation gradient in registers. A pass recording
    gradients decomposes it in kernel_svd instead, whose backward is registered with it.
    """
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            F_tmp = (
                qd.Matrix.identity(gs.qd_float, 3) + mpm_info.substep_dt[None] * mpm_state.particles.C[f, i_p, i_b]
            ) @ mpm_state.particles.F[f, i_p, i_b]
            mpm_state.particles.F_tmp[f, i_p, i_b] = F_tmp
            if qd.static(compute_svd):
                U, S, V = qd.svd(F_tmp, gs.qd_float)
                mpm_state.particles.U[f, i_p, i_b] = U
                mpm_state.particles.S[f, i_p, i_b] = S
                mpm_state.particles.V[f, i_p, i_b] = V


@qd.kernel(fastcache=True)
def kernel_svd_grad(f: int, mpm_state: MPMState):
    """Reverse the singular value decomposition of a frame, consuming the adjoint of its factors like autodiff."""
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            mpm_state.particles.F_tmp.grad[f, i_p, i_b] += func_backward_svd(
                mpm_state.particles.U.grad[f, i_p, i_b],
                mpm_state.particles.S.grad[f, i_p, i_b],
                mpm_state.particles.V.grad[f, i_p, i_b],
                mpm_state.particles.U[f, i_p, i_b],
                mpm_state.particles.S[f, i_p, i_b],
                mpm_state.particles.V[f, i_p, i_b],
            )
            mpm_state.particles.U.grad[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            mpm_state.particles.S.grad[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            mpm_state.particles.V.grad[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)


# The decomposition is reversed in closed form, which autodiff cannot derive
@register_custom_backward(kernel_svd_grad)
@qd.kernel(fastcache=True)
def kernel_svd(f: int, mpm_state: MPMState):
    """Decompose the advected deformation gradient of the active particles of a frame into its singular values."""
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            U, S, V = qd.svd(mpm_state.particles.F_tmp[f, i_p, i_b], gs.qd_float)
            mpm_state.particles.U[f, i_p, i_b] = U
            mpm_state.particles.S[f, i_p, i_b] = S
            mpm_state.particles.V[f, i_p, i_b] = V


@no_grad
@qd.kernel(fastcache=True)
def kernel_flag_non_finite_pos(f: int, mpm_state: MPMState, errno: qd.Tensor):
    """Flag INVALID_PARTICLE_NAN for every env with a particle whose position in the given frame is not finite."""
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if not qd_isfinite(mpm_state.particles.pos[f, i_p, i_b]):
            func_flag_error(i_b, MPMErrorCode.INVALID_PARTICLE_NAN, errno)


@qd.kernel(fastcache=True)
def kernel_copy_frame(source: int, target: int, mpm_state: MPMState):
    """Copy the state of the particles from one frame to another."""
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        mpm_state.particles.pos[target, i_p, i_b] = mpm_state.particles.pos[source, i_p, i_b]
        mpm_state.particles.vel[target, i_p, i_b] = mpm_state.particles.vel[source, i_p, i_b]
        mpm_state.particles.F[target, i_p, i_b] = mpm_state.particles.F[source, i_p, i_b]
        mpm_state.particles.C[target, i_p, i_b] = mpm_state.particles.C[source, i_p, i_b]
        mpm_state.particles.Jp[target, i_p, i_b] = mpm_state.particles.Jp[source, i_p, i_b]
        mpm_state.particles.actu[target, i_p, i_b] = mpm_state.particles.actu[source, i_p, i_b]
        mpm_state.particles.is_active[target, i_p, i_b] = mpm_state.particles.is_active[source, i_p, i_b]


@no_grad
@qd.kernel(fastcache=True)
def kernel_reset_grid_and_grad(f: int, mpm_state: MPMState):
    """Zero every cell of the grid of a frame for every environment, with its adjoint."""
    for i_x, i_y, i_z, i_b in qd.ndrange(
        mpm_state.grid.mass.shape[1],
        mpm_state.grid.mass.shape[2],
        mpm_state.grid.mass.shape[3],
        mpm_state.grid.mass.shape[4],
    ):
        mpm_state.grid.vel_in[f, i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)
        mpm_state.grid.mass[f, i_x, i_y, i_z, i_b] = gs.qd_float(0.0)
        mpm_state.grid.vel_out[f, i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)

        mpm_state.grid.vel_in.grad[f, i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)
        mpm_state.grid.mass.grad[f, i_x, i_y, i_z, i_b] = gs.qd_float(0.0)
        mpm_state.grid.vel_out.grad[f, i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_reset_dirty_cells(f: int, mpm_state: MPMState):
    """
    Zero the cells of the grid of a frame that p2g touched, across all envs, along with their dirty flags.

    The dirty list is shared across envs (see MPMGrid), so threads past the dirty count idle. The thread of the first
    env alone zeroes the flag of its cell, for the next substep to list it again.
    """
    for i_slot, i_b in qd.ndrange(mpm_state.grid.dirty_list.shape[0], mpm_state.grid.mass.shape[4]):
        if i_slot < mpm_state.grid.dirty_count[0]:
            grid_res_y, grid_res_z = mpm_state.grid.mass.shape[2], mpm_state.grid.mass.shape[3]
            i_cell = mpm_state.grid.dirty_list[i_slot]
            i_z = i_cell % grid_res_z
            i_xy = i_cell // grid_res_z
            i_y = i_xy % grid_res_y
            i_x = i_xy // grid_res_y
            mpm_state.grid.mass[f, i_x, i_y, i_z, i_b] = gs.qd_float(0.0)
            mpm_state.grid.vel_in[f, i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)
            mpm_state.grid.vel_out[f, i_x, i_y, i_z, i_b] = qd.Vector.zero(gs.qd_float, 3)
            if i_b == 0:
                mpm_state.grid.dirty_flag[i_x, i_y, i_z] = gs.qd_int(0)


@qd.kernel(fastcache=True)
def kernel_set_state(
    f: int,
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    C: qd.types.ndarray(),
    F: qd.types.ndarray(),
    Jp: qd.types.ndarray(),
    actu: qd.types.ndarray(),
    active: qd.types.ndarray(),
    mpm_state: MPMState,
):
    """Write the state of every particle of every environment into a frame, from arrays of the environment first."""
    for i_p, i_b in qd.ndrange(pos.shape[1], pos.shape[0]):
        for j in qd.static(range(3)):
            mpm_state.particles.pos[f, i_p, i_b][j] = pos[i_b, i_p, j]
            mpm_state.particles.vel[f, i_p, i_b][j] = vel[i_b, i_p, j]
            for k in qd.static(range(3)):
                mpm_state.particles.C[f, i_p, i_b][j, k] = C[i_b, i_p, j, k]
                mpm_state.particles.F[f, i_p, i_b][j, k] = F[i_b, i_p, j, k]
        mpm_state.particles.Jp[f, i_p, i_b] = Jp[i_b, i_p]
        mpm_state.particles.actu[f, i_p, i_b] = actu[i_b, i_p]
        mpm_state.particles.is_active[f, i_p, i_b] = active[i_b, i_p]


@qd.kernel(fastcache=True)
def kernel_update_render_fields(f: int, mpm_state: MPMState, mpm_info: MPMInfo):
    """Copy the particles of a frame to their rendered state, and skin the visual vertices on them."""
    for i_p, i_b in qd.ndrange(mpm_state.particles.pos.shape[1], mpm_state.particles.pos.shape[2]):
        if mpm_state.particles.is_active[f, i_p, i_b]:
            mpm_state.particles_render.pos[i_p, i_b] = mpm_state.particles.pos[f, i_p, i_b]
            mpm_state.particles_render.vel[i_p, i_b] = mpm_state.particles.vel[f, i_p, i_b]
        else:
            mpm_state.particles_render.pos[i_p, i_b] = gu.qd_nowhere()
        mpm_state.particles_render.is_active[i_p, i_b] = mpm_state.particles.is_active[f, i_p, i_b]

    for i_v, i_b in qd.ndrange(mpm_state.vverts_render.pos.shape[0], mpm_state.vverts_render.pos.shape[1]):
        support_idxs = mpm_info.vverts.support_idxs[i_v]
        support_weights = mpm_info.vverts.support_weights[i_v]
        vvert_pos = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(support_idxs.n)):
            i_p = support_idxs[j]
            vvert_pos += mpm_state.particles.pos[f, i_p, i_b] * support_weights[j]
        mpm_state.vverts_render.pos[i_v, i_b] = vvert_pos
        i_p_first = support_idxs[0]
        mpm_state.vverts_render.is_active[i_v, i_b] = mpm_state.particles_render.is_active[i_p_first, i_b]


@qd.kernel(fastcache=True)
def kernel_add_particles(
    f: int,
    particle_start: int,
    material_idx: int,
    n_particles: int,
    active: bool,
    default_Jp: float,
    rho: float,
    pos: qd.types.ndarray(),
    mpm_state: MPMState,
    mpm_info: MPMInfo,
):
    """Add particles at rest to a frame, with the parameters of their material."""
    for i_p_ in range(n_particles):
        i_p = i_p_ + particle_start

        mpm_info.particles.material_idx[i_p] = material_idx
        mpm_info.particles.default_Jp[i_p] = default_Jp
        mpm_info.particles.mass[i_p] = mpm_info.particle_volume[None] * rho
        mpm_info.particles.is_free[i_p] = True
        mpm_info.particles.muscle_group[i_p] = 0
        mpm_info.particles.muscle_direction[i_p] = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)

    for i_p_, i_b in qd.ndrange(n_particles, mpm_state.particles.pos.shape[2]):
        i_p = i_p_ + particle_start

        mpm_state.particles.is_active[f, i_p, i_b] = active
        for i in qd.static(range(3)):
            mpm_state.particles.pos[f, i_p, i_b][i] = pos[i_p_, i]

        mpm_state.particles.vel[f, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        mpm_state.particles.F[f, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
        mpm_state.particles.C[f, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
        mpm_state.particles.Jp[f, i_p, i_b] = default_Jp
        mpm_state.particles.actu[f, i_p, i_b] = gs.qd_float(0.0)


@qd.kernel(fastcache=True)
def kernel_set_particles_pos(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    mpm_state: MPMState,
    mpm_info: MPMInfo,
):
    """Set the position of some particles between two steps, at rest and undeformed."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # The earlier occurrences of a repeated particle are -1 (see 'drop_repeated_index'). Between two steps, the
        # state sits in frame 0 (see 'Solver.has_frames').
        if i_p >= 0:
            for i in qd.static(range(3)):
                mpm_state.particles.pos[0, i_p, i_b][i] = poss[i_b_, i_p_, i]

            # Reset these attributes whenever overwriting particle positions manually
            mpm_state.particles.vel[0, i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
            mpm_state.particles.F[0, i_p, i_b] = qd.Matrix.identity(gs.qd_float, 3)
            mpm_state.particles.C[0, i_p, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)
            mpm_state.particles.Jp[0, i_p, i_b] = mpm_info.particles.default_Jp[i_p]


@qd.kernel(fastcache=True)
def kernel_get_particles_pos(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    mpm_state: MPMState,
):
    """Get the position of some particles between two steps."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            poss[i_b_, i_p_, i] = mpm_state.particles.pos[0, i_p, i_b][i]


@qd.kernel(fastcache=True)
def kernel_set_particles_vel(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
    mpm_state: MPMState,
):
    """Set the velocity of some particles between two steps."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_particles_pos'
        if i_p >= 0:
            for i in qd.static(range(3)):
                mpm_state.particles.vel[0, i_p, i_b][i] = vels[i_b_, i_p_, i]


@qd.kernel(fastcache=True)
def kernel_get_particles_vel(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
    mpm_state: MPMState,
):
    """Get the velocity of some particles between two steps."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for i in qd.static(range(3)):
            vels[i_b_, i_p_, i] = mpm_state.particles.vel[0, i_p, i_b][i]


@qd.kernel(fastcache=True)
def kernel_set_particles_active(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),
    mpm_state: MPMState,
):
    """Set whether some particles take part in the simulation between two steps."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_particles_pos'
        if i_p >= 0:
            mpm_state.particles.is_active[0, i_p, i_b] = qd.cast(actives[i_b_, i_p_], gs.qd_bool)


@qd.kernel(fastcache=True)
def kernel_get_particles_active(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),
    mpm_state: MPMState,
):
    """Get whether some particles take part in the simulation between two steps."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        actives[i_b_, i_p_] = mpm_state.particles.is_active[0, i_p, i_b]


@qd.kernel(fastcache=True)
def kernel_set_particles_actu(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actus: qd.types.ndarray(),
    mpm_state: MPMState,
    mpm_info: MPMInfo,
):
    """Set the actuation of some particles between two steps, each one taking that of its muscle group."""
    for i_p_, i_mg, i_b_ in qd.ndrange(particles_idx.shape[1], actus.shape[2], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_particles_pos'
        if i_p >= 0:
            if mpm_info.particles.muscle_group[i_p] == i_mg:
                mpm_state.particles.actu[0, i_p, i_b] = actus[i_b_, i_p_, i_mg]


@qd.kernel(fastcache=True)
def kernel_get_particles_actu(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actus: qd.types.ndarray(),
    mpm_state: MPMState,
):
    """Get the actuation of some particles between two steps."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        actus[i_b_, i_p_] = mpm_state.particles.actu[0, i_p, i_b]


@qd.kernel(fastcache=True)
def kernel_set_particles_muscle_group(
    particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray(), mpm_info: MPMInfo
):
    """Set the muscle group of some particles."""
    for i_p_ in range(particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        # See 'kernel_set_particles_pos'
        if i_p >= 0:
            mpm_info.particles.muscle_group[i_p] = muscle_group[i_p_]


@qd.kernel(fastcache=True)
def kernel_get_particles_muscle_group(
    particles_idx: qd.types.ndarray(), muscle_group: qd.types.ndarray(), mpm_info: MPMInfo
):
    """Get the muscle group of some particles."""
    for i_p_ in range(particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        muscle_group[i_p_] = mpm_info.particles.muscle_group[i_p]


@qd.kernel(fastcache=True)
def kernel_set_particles_muscle_direction(
    particles_idx: qd.types.ndarray(), muscle_direction: qd.types.ndarray(), mpm_info: MPMInfo
):
    """Set the direction of the muscle fibers of some particles."""
    for i_p_ in range(particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        # See 'kernel_set_particles_pos'
        if i_p >= 0:
            for i in qd.static(range(3)):
                mpm_info.particles.muscle_direction[i_p][i] = muscle_direction[i_p_, i]


@qd.kernel(fastcache=True)
def kernel_set_particles_free(particles_idx: qd.types.ndarray(), free: qd.types.ndarray(), mpm_info: MPMInfo):
    """Set whether some particles move freely or act as boundary conditions of the grid."""
    for i_p_ in range(particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        # See 'kernel_set_particles_pos'
        if i_p >= 0:
            mpm_info.particles.is_free[i_p] = free[i_p_]


@qd.kernel(fastcache=True)
def kernel_get_particles_free(particles_idx: qd.types.ndarray(), free: qd.types.ndarray(), mpm_info: MPMInfo):
    """Get whether some particles move freely."""
    for i_p_ in range(particles_idx.shape[0]):
        i_p = particles_idx[i_p_]
        free[i_p_] = mpm_info.particles.is_free[i_p]


@qd.kernel(fastcache=True)
def kernel_get_particles_mass(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    masses: qd.types.ndarray(),
    mpm_info: MPMInfo,
):
    """Get the physical mass of some particles, the volume scale divided out."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        masses[i_b_, i_p_] = mpm_info.particles.mass[i_p] / mpm_info.particle_volume_scale[None]


@qd.func
def func_backward_svd(
    grad_U: qd.types.matrix(3, 3),
    grad_S: qd.types.matrix(3, 3),
    grad_V: qd.types.matrix(3, 3),
    U: qd.types.matrix(3, 3),
    S: qd.types.matrix(3, 3),
    V: qd.types.matrix(3, 3),
):
    """
    Return the adjoint of a matrix from the adjoint of the factors of its singular value decomposition.

    The decomposition is A = U S V^T, with S diagonal. The adjoint is that of the closed-form derivative of the
    decomposition, F_ij = 1 / (s_j^2 - s_i^2) coupling the singular values s, its denominator kept away from zero.
    """
    vt = V.transpose()
    ut = U.transpose()
    S_term = U @ grad_S @ vt

    s = qd.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
    F = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if i == j:
            F[i, j] = 0.0
        else:
            s_diff = s[j] - s[i]
            F[i, j] = 1.0 / (qd.select(s_diff >= 0, 1.0, -1.0) * qd.max(qd.abs(s_diff), 1e-6))
    u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
    v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
    return u_term + v_term + S_term


@MPMSolver.api
@qd.func
def func_add_grid_vel(f: int, i_x: int, i_y: int, i_z: int, i_b: int, vel: qd.types.vector(3), mpm_state: MPMState):
    """Add a velocity to the velocity a cell of the grid gives back to the particles."""
    mpm_state.grid.vel_out[f, i_x, i_y, i_z, i_b] += vel


@MPMSolver.api
@qd.func
def func_add_particle_vel(f: int, i_p: int, i_b: int, vel: qd.types.vector(3), mpm_state: MPMState):
    """Add a velocity to a particle at a frame, in place, which autodiff reverses."""
    mpm_state.particles.vel[f, i_p, i_b] += vel


class CellSeparation(System):
    """A System adding the separation of the particles of a cell to the transfer from the particles to the grid."""

    slot = MPMSolver.p2g
    cell_separation: Stages


class GridCoupling(System):
    """A System coupling the velocity of the grid with another solver, between its update and its boundary."""

    slot = MPMSolver.grid_couplings
    grid_coupling: Stages


class CellVelocity(System):
    """A System adding the velocity of a cell it separates to the transfer from the grid to the particles."""

    slot = MPMSolver.g2p
    cell_velocity: Stages
