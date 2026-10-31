import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.state_change import StateChange, mutates
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.geom import SpatialHashGrid

from .base_solver import GravityMixin, Solver, TimeBasedMixin


VVERTS = Size("vverts")
VFACES = Size("vfaces")
# Sizes of the particle solvers: the particles, the slots of the hash grid and the support particles of a visual vertex
PARTICLES = Size("particles")
VVERT_SUPPORTS = Size("vvert_supports")


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


class ParticleSolver(GravityMixin, TimeBasedMixin, Solver):
    """Base class of the particle-based solvers, holding what they share: the accessors of the state of the particles.

    The accessors take global particle indices, given once for every environment or one row per environment (see
    '_sanitize_io_variables'), and the kernels of each solver read and write the state the particles hold between two
    steps. A solver whose particles extend 'Particles' and 'ParticlesParams' assigns them to
    '_particles' and '_particles_params' at build time, which the accessors then read and write.
    """

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        self._particles: Particles | None = None
        self._particles_params: ParticlesParams | None = None

    @gs.assert_built
    @mutates(StateChange.GEOMETRY)
    def set_particles_pos(self, poss, particles_idx=None, envs_idx=None):
        """
        Set the position of some particles, at rest.

        Parameters
        ----------
        poss : array_like, shape ([M,] [N,] 3)
            Target position of each particle.
        particles_idx : None | int | array_like, shape ([M,] N), optional
            Global index of the particles. If None, all particles will be considered. Defaults to None.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be considered. Defaults to None.
        """
        poss, particles_idx, envs_idx = self._sanitize_io_variables(
            poss, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_set_particles_pos(particles_idx, envs_idx, poss)
        # Overwriting the positions of the particles clears the errors their former state flagged
        self.clear_errno(envs_idx)

    def get_particles_pos(self, particles_idx=None, envs_idx=None):
        """Get the position of some particles (see 'set_particles_pos')."""
        poss, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_get_particles_pos(particles_idx, envs_idx, poss)
        return poss[0] if self._scene.n_envs == 0 else poss

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_particles_vel(self, vels, particles_idx=None, envs_idx=None):
        """Set the velocity of some particles (see 'set_particles_pos')."""
        vels, particles_idx, envs_idx = self._sanitize_io_variables(
            vels, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_set_particles_vel(particles_idx, envs_idx, vels)

    def get_particles_vel(self, particles_idx=None, envs_idx=None):
        """Get the velocity of some particles (see 'set_particles_pos')."""
        vels, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        self._kernel_get_particles_vel(particles_idx, envs_idx, vels)
        return vels[0] if self._scene.n_envs == 0 else vels

    @gs.assert_built
    @mutates(StateChange.GEOMETRY, differentiable=False)
    def set_particles_active(self, actives, particles_idx=None, envs_idx=None):
        """Set whether some particles take part in the simulation (see 'set_particles_pos')."""
        actives, particles_idx, envs_idx = self._sanitize_io_variables(
            actives, particles_idx, self.n_particles, "particles_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        self._kernel_set_particles_active(particles_idx, envs_idx, actives)

    def get_particles_active(self, particles_idx=None, envs_idx=None):
        """Get whether some particles take part in the simulation (see 'set_particles_pos')."""
        actives, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        self._kernel_get_particles_active(particles_idx, envs_idx, actives)
        return actives[0] if self._scene.n_envs == 0 else actives

    def get_particles_mass(self, particles_idx=None, envs_idx=None):
        """Get the mass of some particles (see 'set_particles_pos')."""
        masses, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True
        )
        self._kernel_get_particles_mass(particles_idx, envs_idx, masses)
        return masses[0] if self._scene.n_envs == 0 else masses

    # The hooks of the accessors, which a solver overrides when its particles carry one frame per substep (MPM) or when
    # a setter also resets a quantity of its own (SPH)
    def _kernel_set_particles_pos(self, particles_idx, envs_idx, poss):
        kernel_set_particles_pos(particles_idx, envs_idx, poss, self._particles)

    def _kernel_get_particles_pos(self, particles_idx, envs_idx, poss):
        kernel_get_particles_pos(particles_idx, envs_idx, poss, self._particles)

    def _kernel_set_particles_vel(self, particles_idx, envs_idx, vels):
        kernel_set_particles_vel(particles_idx, envs_idx, vels, self._particles)

    def _kernel_get_particles_vel(self, particles_idx, envs_idx, vels):
        kernel_get_particles_vel(particles_idx, envs_idx, vels, self._particles)

    def _kernel_set_particles_active(self, particles_idx, envs_idx, actives):
        kernel_set_particles_active(particles_idx, envs_idx, actives, self._particles)

    def _kernel_get_particles_active(self, particles_idx, envs_idx, actives):
        kernel_get_particles_active(particles_idx, envs_idx, actives, self._particles)

    def _kernel_get_particles_mass(self, particles_idx, envs_idx, masses):
        kernel_get_particles_mass(particles_idx, envs_idx, masses, self._particles_params)


@qd.kernel(fastcache=True)
def kernel_add_vverts(
    vvert_start: int,
    particle_start: int,
    vverts: qd.types.ndarray(element_dim=1),
    particles: qd.types.ndarray(element_dim=1),
    P_invs: qd.types.ndarray(element_dim=2),
    support_idxs_local: qd.types.ndarray(),
    vverts_support: VVertsSupport,
):
    """
    Bind the given vverts to their supporting particles, whose positions skin them for rendering.

    The weights of a vvert are its barycentric coordinates in the simplex of its support particles, `P_invs` mapping its
    offset from the last support to the others.
    """
    n_supports = support_idxs_local.shape[1]
    for i_vv_ in range(vverts.shape[0]):
        i_vv = i_vv_ + vvert_start
        v = vverts[i_vv_]
        P_inv = P_invs[i_vv_]
        i_p_last = support_idxs_local[i_vv_, n_supports - 1]
        b = P_inv @ (v - particles[i_p_last])
        for i_s in range(n_supports - 1):
            vverts_support.support_weights[i_vv][i_s] = b[i_s]
        vverts_support.support_weights[i_vv][n_supports - 1] = 1.0 - b.sum()
        for i_s in range(n_supports):
            vverts_support.support_idxs[i_vv][i_s] = support_idxs_local[i_vv_, i_s] + particle_start


@qd.kernel(fastcache=True)
def kernel_set_particles_pos(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    particles: Particles,
):
    """Set the position of some particles, at rest."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # The earlier occurrences of a repeated particle are -1 (see 'drop_repeated_index')
        if i_p >= 0:
            particles.pos[i_p, i_b] = qd.Vector([poss[i_b_, i_p_, 0], poss[i_b_, i_p_, 1], poss[i_b_, i_p_, 2]])
            particles.vel[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_get_particles_pos(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    poss: qd.types.ndarray(),
    particles: Particles,
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for k in qd.static(range(3)):
            poss[i_b_, i_p_, k] = particles.pos[i_p, i_b][k]


@qd.kernel(fastcache=True)
def kernel_set_particles_vel(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
    particles: Particles,
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_particles_pos'
        if i_p >= 0:
            particles.vel[i_p, i_b] = qd.Vector([vels[i_b_, i_p_, 0], vels[i_b_, i_p_, 1], vels[i_b_, i_p_, 2]])


@qd.kernel(fastcache=True)
def kernel_get_particles_vel(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    vels: qd.types.ndarray(),
    particles: Particles,
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        for k in qd.static(range(3)):
            vels[i_b_, i_p_, k] = particles.vel[i_p, i_b][k]


@qd.kernel(fastcache=True)
def kernel_set_particles_active(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),
    particles: Particles,
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_particles_pos'
        if i_p >= 0:
            particles.is_active[i_p, i_b] = actives[i_b_, i_p_]


@qd.kernel(fastcache=True)
def kernel_get_particles_active(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    actives: qd.types.ndarray(),
    particles: Particles,
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        actives[i_b_, i_p_] = particles.is_active[i_p, i_b]


@qd.kernel(fastcache=True)
def kernel_get_particles_mass(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    masses: qd.types.ndarray(),
    particles_params: ParticlesParams,
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        masses[i_b_, i_p_] = particles_params.mass[i_p]


@qd.kernel(fastcache=True)
def kernel_update_particles_render(particles: Particles, particles_render: ParticlesRender):
    """Update the particles as drawn from their state."""
    for i_p, i_b in qd.ndrange(particles.pos.shape[0], particles.pos.shape[1]):
        if particles.is_active[i_p, i_b]:
            particles_render.pos[i_p, i_b] = particles.pos[i_p, i_b]
            particles_render.vel[i_p, i_b] = particles.vel[i_p, i_b]
        else:
            particles_render.pos[i_p, i_b] = gu.qd_nowhere()
        particles_render.is_active[i_p, i_b] = particles.is_active[i_p, i_b]


# FIXME: quadrants#1036 - a func parameter accepts no subclass of its dataclass annotation, so the funcs shared by the
# solvers take the arrays of their particles one by one rather than their common base `Particles`.
@qd.func
def func_get_particles_mean_vel(
    i_b: int,
    pos: qd.types.vector(3),
    hash_grid: SpatialHashGrid,
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    half_size: float,
    n_cells: int,
):
    """
    Return the mean velocity of the sorted particles of a solver within a cube around a position, and their count.

    The cube is centered at `pos` with half side `half_size`. The particles are searched for among the cells of the hash
    grid of the solver, `n_cells` along each axis from the one holding the lowest corner of the cube.
    """
    base = gu.func_hash_pos_to_grid(pos - half_size, hash_grid)
    vel_sum = qd.Vector.zero(gs.qd_float, 3)
    n_particles = 0
    for offset in qd.grouped(qd.ndrange(n_cells, n_cells, n_cells)):
        i_s = gu.func_hash_grid_to_slot(base + offset, hash_grid)
        i_p_start = hash_grid.slot_start[i_s, i_b]
        for i_p in range(i_p_start, i_p_start + hash_grid.slot_size[i_s, i_b]):
            if qd.abs(pos - particles_pos[i_p, i_b]).max() < half_size:
                vel_sum += particles_vel[i_p, i_b]
                n_particles += 1
    return vel_sum / qd.max(n_particles, 1), n_particles


@ParticleSolver.api
@qd.func
def func_add_particles_momentum(
    i_b: int,
    pos: qd.types.vector(3),
    momentum: qd.types.vector(3),
    hash_grid: SpatialHashGrid,
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_mass: qd.Tensor,
    particles_is_moved: qd.Tensor,
    half_size: float,
    n_cells: int,
):
    """
    Add a momentum to each sorted particle of a solver within a cube around a position (see func_get_particles_mean_vel).

    Each particle that `particles_is_moved` selects gains the velocity `momentum` over its mass.
    """
    base = gu.func_hash_pos_to_grid(pos - half_size, hash_grid)
    for offset in qd.grouped(qd.ndrange(n_cells, n_cells, n_cells)):
        i_s = gu.func_hash_grid_to_slot(base + offset, hash_grid)
        i_p_start = hash_grid.slot_start[i_s, i_b]
        for i_p in range(i_p_start, i_p_start + hash_grid.slot_size[i_s, i_b]):
            if qd.abs(pos - particles_pos[i_p, i_b]).max() < half_size and particles_is_moved[i_p, i_b]:
                particles_vel[i_p, i_b] = particles_vel[i_p, i_b] + momentum / particles_mass[i_p, i_b]


@ParticleSolver.api
@qd.func
def func_set_particle_pos(i_p: int, i_b: int, pos: qd.types.vector(3), particles_pos: qd.Tensor):
    """Set the position of a sorted particle of a solver."""
    particles_pos[i_p, i_b] = pos


@ParticleSolver.api
@qd.func
def func_set_particle_vel(i_p: int, i_b: int, vel: qd.types.vector(3), particles_vel: qd.Tensor):
    """Set the velocity of a sorted particle of a solver."""
    particles_vel[i_p, i_b] = vel
