import numpy as np

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.particle_entity import PARTICLES
from genesis.engine.state_change import StateChange, mutates
from genesis.utils.data import ENVS, Data, DataKind, Size, field
from genesis.utils.geom import SpatialHashGrid

from .base_solver import VVERTS, GravityMixin, Solver, TimeBasedMixin, kernel_get_values, kernel_set_values


# The supports of a visual vertex
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


class ParticlesRender(Particles, kind=DataKind.SCRATCH):
    """The particles as drawn: their position and velocity, an inactive particle moved out of sight."""


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
    steps. A solver assigns the struct of its particles, holding their position, velocity and activity, and the one of
    their parameters, holding their mass, to '_particles' and '_particles_params' at build time, along with the
    particles and visual vertices as drawn, which the accessors and the update of the rendering then read and write.
    """

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        self._particle_size = options.particle_size
        self._upper_bound = np.array(options.upper_bound)
        self._lower_bound = np.array(options.lower_bound)

        self._n_vvert_supports = self.scene.options.vis.n_support_neighbors
        self._n_particles: int | None = None
        self._n_vverts: int | None = None
        self._n_vfaces: int | None = None
        # Held in frames for a solver whose state is (see 'System.has_frames')
        self._particles: Data | None = None
        self._particles_params: Data | None = None
        self._vverts_support: VVertsSupport | None = None
        self.particles_render: ParticlesRender | None = None
        self.vverts_render: VVertsRender | None = None

    def build(self):
        super().build()
        self._n_particles = self.n_particles
        self._n_vverts = self.n_vverts
        self._n_vfaces = self.n_vfaces

    def add_vverts(self, *args):
        """Bind the given vverts to their supporting particles (see 'kernel_add_vverts')."""
        kernel_add_vverts(*args, self._vverts_support)

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
        kernel_set_particles_pos(
            particles_idx, envs_idx, poss, self._particles.pos, self._particles.vel, has_frames=self.has_frames
        )
        self._reset_particles(particles_idx, envs_idx, pos_changed=True)
        # Overwriting the positions of the particles clears the errors their former state flagged
        self.clear_errno(envs_idx)

    def get_particles_pos(self, particles_idx=None, envs_idx=None):
        """Get the position of some particles (see 'set_particles_pos')."""
        poss, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_get_values(particles_idx, envs_idx, poss, self._particles.pos, has_frames=self.has_frames)
        return poss[0] if self._scene.n_envs == 0 else poss

    @gs.assert_built
    @mutates(StateChange.DYNAMICS)
    def set_particles_vel(self, vels, particles_idx=None, envs_idx=None):
        """Set the velocity of some particles (see 'set_particles_pos')."""
        vels, particles_idx, envs_idx = self._sanitize_io_variables(
            vels, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_set_values(particles_idx, envs_idx, vels, self._particles.vel, has_frames=self.has_frames)
        self._reset_particles(particles_idx, envs_idx, pos_changed=False)

    def get_particles_vel(self, particles_idx=None, envs_idx=None):
        """Get the velocity of some particles (see 'set_particles_pos')."""
        vels, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, (3,), batched_idx=True
        )
        kernel_get_values(particles_idx, envs_idx, vels, self._particles.vel, has_frames=self.has_frames)
        return vels[0] if self._scene.n_envs == 0 else vels

    @gs.assert_built
    @mutates(StateChange.GEOMETRY, differentiable=False)
    def set_particles_active(self, actives, particles_idx=None, envs_idx=None):
        """Set whether some particles take part in the simulation (see 'set_particles_pos')."""
        actives, particles_idx, envs_idx = self._sanitize_io_variables(
            actives, particles_idx, self.n_particles, "particles_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        kernel_set_values(particles_idx, envs_idx, actives, self._particles.is_active, has_frames=self.has_frames)

    def get_particles_active(self, particles_idx=None, envs_idx=None):
        """Get whether some particles take part in the simulation (see 'set_particles_pos')."""
        actives, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, dtype=gs.tc_bool, batched_idx=True
        )
        kernel_get_values(particles_idx, envs_idx, actives, self._particles.is_active, has_frames=self.has_frames)
        return actives[0] if self._scene.n_envs == 0 else actives

    def get_particles_mass(self, particles_idx=None, envs_idx=None):
        """Get the mass of some particles (see 'set_particles_pos')."""
        masses, particles_idx, envs_idx = self._sanitize_io_variables(
            None, particles_idx, self.n_particles, "particles_idx", envs_idx, batched_idx=True
        )
        kernel_get_particles_mass(particles_idx, envs_idx, masses, self._particles_params.mass)
        return masses[0] if self._scene.n_envs == 0 else masses

    def _reset_particles(self, particles_idx, envs_idx, pos_changed: bool):
        """
        Reset what the new velocity of some particles invalidates, or their new position at rest, nothing by default.

        `pos_changed` is whether their position was set, rather than their velocity alone.
        """

    def update_render_fields(self):
        """Update the particles and the visual vertices as drawn from the state of the particles."""
        f = self._sim.cur_substep_local
        particles = self._particles
        kernel_update_particles_render(
            f, particles.pos, particles.vel, particles.is_active, self.particles_render, has_frames=self.has_frames
        )
        if self.vverts_render is not None:
            kernel_update_vverts_render(
                f,
                particles.pos,
                self.particles_render,
                self.vverts_render,
                self._vverts_support,
                has_frames=self.has_frames,
            )

    @property
    def is_active(self):
        return self.n_particles > 0

    @property
    def n_particles(self):
        if self.is_built:
            return self._n_particles
        return sum([entity.n_particles for entity in self._entities])

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
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    has_frames: qd.template(),
):
    """Set the position of some particles, at rest."""
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_values' in base_solver.py
        if i_p >= 0:
            I_p = [0, i_p, i_b] if qd.static(has_frames) else [i_p, i_b]
            particles_pos[I_p] = qd.Vector([poss[i_b_, i_p_, 0], poss[i_b_, i_p_, 1], poss[i_b_, i_p_, 2]])
            particles_vel[I_p] = qd.Vector.zero(gs.qd_float, 3)


@qd.kernel(fastcache=True)
def kernel_get_particles_mass(
    particles_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    masses: qd.types.ndarray(),
    particles_mass: qd.Tensor,
):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        masses[i_b_, i_p_] = particles_mass[i_p]


@qd.kernel(fastcache=True)
def kernel_update_particles_render(
    f: int,
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_is_active: qd.Tensor,
    particles_render: ParticlesRender,
    has_frames: qd.template(),
):
    """Update the particles as drawn from their state, at the given frame for a solver holding its state in frames."""
    for i_p, i_b in qd.ndrange(particles_render.pos.shape[0], particles_render.pos.shape[1]):
        I_p = [f, i_p, i_b] if qd.static(has_frames) else [i_p, i_b]
        if particles_is_active[I_p]:
            particles_render.pos[i_p, i_b] = particles_pos[I_p]
            particles_render.vel[i_p, i_b] = particles_vel[I_p]
        else:
            particles_render.pos[i_p, i_b] = gu.qd_nowhere()
        particles_render.is_active[i_p, i_b] = particles_is_active[I_p]


@qd.kernel(fastcache=True)
def kernel_update_vverts_render(
    f: int,
    particles_pos: qd.Tensor,
    particles_render: ParticlesRender,
    vverts_render: VVertsRender,
    vverts_support: VVertsSupport,
    has_frames: qd.template(),
):
    """Update the visual vertices as drawn from the particles supporting them, once the particles are."""
    for i_vv, i_b in qd.ndrange(vverts_render.pos.shape[0], vverts_render.pos.shape[1]):
        support_idxs = vverts_support.support_idxs[i_vv]
        support_weights = vverts_support.support_weights[i_vv]
        vvert_pos = qd.Vector.zero(gs.qd_float, 3)
        for k in qd.static(range(support_idxs.n)):
            i_p = support_idxs[k]
            I_p = [f, i_p, i_b] if qd.static(has_frames) else [i_p, i_b]
            vvert_pos += particles_pos[I_p] * support_weights[k]
        vverts_render.pos[i_vv, i_b] = vvert_pos
        i_p_first = support_idxs[0]
        vverts_render.is_active[i_vv, i_b] = particles_render.is_active[i_p_first, i_b]


@qd.kernel(fastcache=True)
def kernel_set_particles_state(
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    flags: qd.types.ndarray(),
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_flags: qd.Tensor,
):
    """Set the position, the velocity and a flag of every particle, from arrays of the environment first."""
    for i_p, i_b in qd.ndrange(particles_pos.shape[0], particles_pos.shape[1]):
        particles_pos[i_p, i_b] = qd.Vector([pos[i_b, i_p, 0], pos[i_b, i_p, 1], pos[i_b, i_p, 2]])
        particles_vel[i_p, i_b] = qd.Vector([vel[i_b, i_p, 0], vel[i_b, i_p, 1], vel[i_b, i_p, 2]])
        particles_flags[i_p, i_b] = flags[i_b, i_p]


@qd.kernel(fastcache=True)
def kernel_get_particles_state(
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    flags: qd.types.ndarray(),
    particles_pos: qd.Tensor,
    particles_vel: qd.Tensor,
    particles_flags: qd.Tensor,
):
    """Get the position, the velocity and a flag of every particle, into arrays of the environment first."""
    for i_p, i_b in qd.ndrange(particles_pos.shape[0], particles_pos.shape[1]):
        for k in qd.static(range(3)):
            pos[i_b, i_p, k] = particles_pos[i_p, i_b][k]
            vel[i_b, i_p, k] = particles_vel[i_p, i_b][k]
        flags[i_b, i_p] = particles_flags[i_p, i_b]


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
def func_set_particle_vec3(i_p: int, i_b: int, value: qd.types.vector(3), particles_values: qd.Tensor):
    """Set a 3D vector of a sorted particle of a solver, such as its position or its velocity."""
    particles_values[i_p, i_b] = value
