import numpy as np

import quadrants as qd

import genesis as gs
from genesis.engine.boundaries import CubeBoundary, func_impose_cube_boundary
from genesis.engine.boundaries.boundaries import CubeBoundaryBox
from genesis.engine.entities import SPHEntity
from genesis.engine.materials import SPH
from genesis.engine.solvers.particle_solver import PARTICLES, Particles, ParticlesParams, ParticlesRender
from genesis.engine.states.solvers import SPHSolverState
from genesis.utils.data import ENVS, Data, DataKind, field
from genesis.utils.geom import (
    SpatialHashGrid,
    func_cubic_kernel,
    func_cubic_kernel_derivative,
    func_hash_compute_reordered_idx,
    func_hash_grid_to_slot,
    func_hash_pos_to_grid,
)
from genesis.utils.misc import qd_to_numpy

from .particle_solver import ParticleSolver, kernel_get_particles_state, kernel_set_particles_state


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


class SPHSolver(ParticleSolver):
    material_cls = SPH.Base
    sph_state: SPHState | None
    sph_info: SPHInfo | None

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._support_radius = options._support_radius
        self._pressure_solver = options.pressure_solver

        # Divergence-free smoothed particle hydrodynamics (DFSPH) parameters
        self._df_max_error_div = options.max_divergence_error
        self._df_max_error_den = options.max_density_error_percent
        self._df_max_div_iters = options.max_divergence_solver_iterations
        self._df_max_den_iters = options.max_density_solver_iterations

        self._particle_volume = 0.8 * self._particle_size**3  # 0.8 is an empirical value

        self._hash_grid_cell_size = options.hash_grid_cell_size
        self._hash_grid_res = options._hash_grid_res

        self.boundary = CubeBoundary(lower=self._lower_bound, upper=self._upper_bound)

        # Coefficient for computing stable timestep
        self._stable_dt_coef = 1.0

        # The rest density the incompressibility solves target, the one of the first particle
        # TODO: Support per-particle density
        self._density0: float | None = None

        self.sph_state = None
        self.sph_info = None

    def build(self):
        super().build()

        # See PBDSolver.build
        if self.is_active:
            # The density of the first particle, as the particles store it in the precision of the simulation
            self._density0 = float(gs.np_float(self.entities[0].material.rho))
        sizes = dict(envs=self._B, particles=self._n_particles)
        unused = {None: not self.is_active}
        self.sph_state = SPHState(
            sizes=dict(sizes, hash_slots=np.prod(self._hash_grid_res)),
            values=dict(hash_grid=dict(cell_size=self._hash_grid_cell_size, grid_res=self._hash_grid_res)),
            unused=unused,
        )
        self.sph_info = SPHInfo(
            sizes=sizes,
            values=dict(
                boundary=self.boundary.values,
                substep_dt=self.substep_dt,
                particle_size=self._particle_size,
                particle_volume=self._particle_volume,
                support_radius=self._support_radius,
                density0=self._density0,
            ),
            unused=unused,
        )
        self._particles = self.sph_state.particles
        self._particles_params = self.sph_info.particles
        self.particles_render = self.sph_state.particles_render

        if self.is_active:
            for entity in self.entities:
                entity._add_to_solver()

            # Compute stable timestep for the given particle properties (mu) and support radius and print warning
            # if the current timestep is larger than the stable timestep. This formula is derived from
            # [func_task_compute_non_pressure_forces] function, which computes the acceleration due to non-pressure
            # forces.
            mu_max = qd_to_numpy(self.sph_info.particles.mu, transpose=True).max()
            stable_dt = self._stable_dt_coef * (0.01 * self._support_radius**2) / mu_max
            if self.substep_dt > stable_dt:
                gs.logger.warning(
                    f"Current timestep ({self.substep_dt:.4g}) is larger than the stable timestep "
                    f"({stable_dt:.4g}) for SPH solver. "
                    "Please consider reducing the timestep, liquid viscosity (mu), or increasing the particle size."
                )

        self._build_gravity(self.sph_info.gravity)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name: str | None = None, desc=None
    ) -> "SPHEntity":
        entity = SPHEntity(
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

        self.entities.append(entity)
        return entity

    def add_particles(self, particle_start, active, mat_rho, mat_stiffness, mat_exponent, mat_mu, mat_gamma, pos):
        """Add the particles of an entity at rest, with the parameters of its material."""
        kernel_add_particles(
            particle_start,
            active,
            mat_rho,
            mat_stiffness,
            mat_exponent,
            mat_mu,
            mat_gamma,
            pos,
            self.sph_state,
            self.sph_info,
        )

    # ------------------------------------------------------------------------------------
    # ------------------------------------- DFSPH ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _dfsph_solve(self, solve_density: bool):
        """
        Run the divergence-free solve, or the density solve, of DFSPH until the mean density error meets its bound.

        The divergence-free solve cancels the rate of change of the density, and the density solve its deviation from
        the rest density at the end of the substep.
        """
        # TODO: warm start
        if solve_density:
            # Scale the factor of the density solve by the inverse of the squared time step
            kernel_compute_density_star(1.0 / self._substep_dt**2, self.sph_state, self.sph_info)
            max_iters, error_offset = self._df_max_den_iters, self._density0
            error_bound = self._df_max_error_den * 0.01 * self._density0
        else:
            kernel_compute_density_time_derivative(self.sph_state, self.sph_info)
            max_iters, error_offset = self._df_max_div_iters, 0.0
            # The SI unit of the divergence is s^-1, hence the bound of the density error divided by the time step
            error_bound = (1 / self._substep_dt) * self._df_max_error_div * 0.01 * self._density0

        iteration = 0
        avg_density_err = 0.0
        while iteration < max_iters:
            kernel_pressure_solve_iteration(self.sph_state, self.sph_info, solve_density=solve_density)
            if solve_density:
                kernel_compute_density_star(dfsph_factor_scale=1.0, sph_state=self.sph_state, sph_info=self.sph_info)
            else:
                kernel_compute_density_time_derivative(self.sph_state, self.sph_info)
            avg_density_err = (
                kernel_compute_density_error(error_offset, self.sph_state, self.sph_info) / self._n_particles
            )
            if avg_density_err <= error_bound:
                break
            iteration += 1

        solve_name = "density" if solve_density else "divergence"
        gs.logger.debug(f"DFSPH - {solve_name} solve: {iteration} iterations, mean density error {avg_density_err:.4g}")

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        if self.is_active:
            kernel_reorder_particles(self.sph_state, self.sph_info)
            if self._pressure_solver == "WCSPH":
                kernel_compute_rho(self.sph_state, self.sph_info)
                kernel_compute_non_pressure_forces(
                    self._sim.cur_t, self.sph_state, self.sph_info, self.force_fields_dispatcher
                )
                kernel_compute_pressure_forces(self.sph_state, self.sph_info)
                kernel_advect_velocity(self.sph_state, self.sph_info)
            elif self._pressure_solver == "DFSPH":
                kernel_compute_rho(self.sph_state, self.sph_info)
                kernel_compute_dfsph_factor(self.sph_state, self.sph_info)
                self._dfsph_solve(solve_density=False)
                kernel_advect_velocity(self.sph_state, self.sph_info)
                kernel_compute_non_pressure_forces(
                    self._sim.cur_t, self.sph_state, self.sph_info, self.force_fields_dispatcher
                )
                kernel_advect_velocity(self.sph_state, self.sph_info)
                self._dfsph_solve(solve_density=True)

    def substep_post_coupling(self, f):
        if self.is_active:
            kernel_advect_position(self.sph_state, self.sph_info)
            kernel_copy_from_reordered(self.sph_state)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            particles = self.sph_state.particles
            kernel_set_particles_state(
                state.pos, state.vel, state.active, particles.pos, particles.vel, particles.is_active
            )

    def get_state(self, f):
        if self.is_active:
            state = SPHSolverState(self.scene)
            particles = self.sph_state.particles
            kernel_get_particles_state(
                state.pos, state.vel, state.active, particles.pos, particles.vel, particles.is_active
            )
        else:
            state = None
        return state

    def _reset_particles(self, particles_idx, envs_idx, pos_changed: bool):
        # The acceleration of the particles carries over to the first velocity update of the next substep of
        # divergence-free smoothed particle hydrodynamics (DFSPH), so setting their position or velocity drops the
        # acceleration of their former state
        kernel_clear_particles_acc(particles_idx, envs_idx, self.sph_state)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def particle_volume(self):
        return self._particle_volume

    @property
    def support_radius(self):
        return self._support_radius

    @property
    def hash_grid_res(self):
        return self._hash_grid_res

    @property
    def hash_grid_cell_size(self):
        return self._hash_grid_cell_size


# ------------------------------------------------------------------------------------
# ------------------------------------- kernels --------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_for_all_neighbors(
    i_p: int,
    i_b: int,
    ret: qd.template(),
    sph_state: SPHState,
    sph_info: SPHInfo,
    task: qd.template(),
):
    """
    Run a task on every sorted particle within the support radius of a sorted particle, the particle itself excluded.

    `task` takes the indices of both particles, the environment, `ret`, the state and the info, in this order, and
    accumulates its result in `ret`.
    """
    base = func_hash_pos_to_grid(sph_state.particles_reordered.pos[i_p, i_b], sph_state.hash_grid)
    for offset in qd.grouped(qd.ndrange((-1, 2), (-1, 2), (-1, 2))):
        i_s = func_hash_grid_to_slot(base + offset, sph_state.hash_grid)
        slot_start = sph_state.hash_grid.slot_start[i_s, i_b]
        for j_p in range(slot_start, slot_start + sph_state.hash_grid.slot_size[i_s, i_b]):
            d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
            if i_p != j_p and d_ij.norm() < sph_info.support_radius[None]:
                task(i_p, j_p, i_b, ret, sph_state, sph_info)


@qd.kernel(fastcache=True)
def kernel_reorder_particles(sph_state: SPHState, sph_info: SPHInfo):
    """Sort the particles by the cell of the hash grid they lie in, into the copies the substep runs on."""
    func_hash_compute_reordered_idx(
        sph_state.particles.pos, sph_state.particles.is_active, sph_state.particles.reordered_idx, sph_state.hash_grid
    )

    for i_p, i_b in qd.ndrange(sph_state.particles.pos.shape[0], sph_state.particles.pos.shape[1]):
        sph_state.particles_reordered.is_active[i_p, i_b] = False
    for i_p, i_b in qd.ndrange(sph_state.particles.pos.shape[0], sph_state.particles.pos.shape[1]):
        if sph_state.particles.is_active[i_p, i_b]:
            i_r = sph_state.particles.reordered_idx[i_p, i_b]
            func_copy_particle(i_p, i_r, i_b, sph_state.particles, sph_state.particles_reordered)
            sph_state.particles_info_reordered.rho[i_r, i_b] = sph_info.particles.rho[i_p]
            sph_state.particles_info_reordered.mass[i_r, i_b] = sph_info.particles.mass[i_p]
            sph_state.particles_info_reordered.stiffness[i_r, i_b] = sph_info.particles.stiffness[i_p]
            sph_state.particles_info_reordered.exponent[i_r, i_b] = sph_info.particles.exponent[i_p]
            sph_state.particles_info_reordered.mu[i_r, i_b] = sph_info.particles.mu[i_p]
            sph_state.particles_info_reordered.gamma[i_r, i_b] = sph_info.particles.gamma[i_p]
            sph_state.particles_reordered.is_active[i_r, i_b] = True


@qd.func
def func_copy_particle(
    i_p_src: int,
    i_p_dst: int,
    i_b: int,
    particles_src: SPHParticles,
    particles_dst: SPHParticles,
):
    """Copy the dynamic state of a particle onto another slot of the same environment."""
    particles_dst.pos[i_p_dst, i_b] = particles_src.pos[i_p_src, i_b]
    particles_dst.vel[i_p_dst, i_b] = particles_src.vel[i_p_src, i_b]
    particles_dst.acc[i_p_dst, i_b] = particles_src.acc[i_p_src, i_b]
    particles_dst.rho[i_p_dst, i_b] = particles_src.rho[i_p_src, i_b]
    particles_dst.p[i_p_dst, i_b] = particles_src.p[i_p_src, i_b]
    particles_dst.dfsph_factor[i_p_dst, i_b] = particles_src.dfsph_factor[i_p_src, i_b]
    particles_dst.drho[i_p_dst, i_b] = particles_src.drho[i_p_src, i_b]


@qd.kernel(fastcache=True)
def kernel_copy_from_reordered(sph_state: SPHState):
    """Give the active particles back the state of their sorted copy."""
    for i_p, i_b in qd.ndrange(sph_state.particles.pos.shape[0], sph_state.particles.pos.shape[1]):
        if sph_state.particles.is_active[i_p, i_b]:
            i_r = sph_state.particles.reordered_idx[i_p, i_b]
            func_copy_particle(i_r, i_p, i_b, sph_state.particles_reordered, sph_state.particles)


@qd.func
def func_task_compute_rho(i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo):
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    ret += sph_info.particle_volume[None] * func_cubic_kernel(d_ij.norm(), sph_info.support_radius[None])


@qd.kernel(fastcache=True)
def kernel_compute_rho(sph_state: SPHState, sph_info: SPHInfo):
    """Compute the density of the sorted particles from their neighbours."""
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            # Base density using the kernel at distance 0
            sph_state.particles_reordered.rho[i_p, i_b] = sph_info.particle_volume[None] * func_cubic_kernel(
                0.0, sph_info.support_radius[None]
            )

            den = gs.qd_float(0.0)
            func_for_all_neighbors(i_p, i_b, den, sph_state, sph_info, func_task_compute_rho)
            sph_state.particles_reordered.rho[i_p, i_b] += den

            sph_state.particles_reordered.rho[i_p, i_b] *= sph_state.particles_info_reordered.rho[i_p, i_b]


@qd.func
def func_task_compute_non_pressure_forces(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    dist = d_ij.norm()

    gamma_i = sph_state.particles_info_reordered.gamma[i_p, i_b]
    mass_i = sph_state.particles_info_reordered.mass[i_p, i_b]
    mu_i = sph_state.particles_info_reordered.mu[i_p, i_b]

    mass_j = sph_state.particles_info_reordered.mass[j_p, i_b]

    # -----------------------------
    # Surface Tension term
    # -----------------------------
    # The distance is clamped to the particle size, which bounds the surface tension of overlapping particles
    particle_size = sph_info.particle_size[None]
    effective_dist = dist if dist > particle_size else particle_size

    ret -= gamma_i / mass_i * mass_j * d_ij * func_cubic_kernel(effective_dist, sph_info.support_radius[None])

    # -----------------------------
    # Viscosity Force
    # -----------------------------
    v_ij = (sph_state.particles_reordered.vel[i_p, i_b] - sph_state.particles_reordered.vel[j_p, i_b]).dot(d_ij)

    # Twice the dimension plus two, the normalization of the viscosity term in three dimensions
    d = 2 * (3 + 2)

    rho_j = sph_state.particles_reordered.rho[j_p, i_b]

    f_v = (
        d
        * mu_i
        * (mass_j / rho_j)
        * v_ij
        / (dist**2 + 0.01 * sph_info.support_radius[None] ** 2)
        * func_cubic_kernel_derivative(d_ij, sph_info.support_radius[None])
    )
    ret += f_v


@qd.kernel
def kernel_compute_non_pressure_forces(t: float, sph_state: SPHState, sph_info: SPHInfo, force_fields: qd.template()):
    """Compute the acceleration of the sorted particles from gravity, surface tension, viscosity and force fields."""
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            acc = sph_info.gravity[i_b]
            func_for_all_neighbors(i_p, i_b, acc, sph_state, sph_info, func_task_compute_non_pressure_forces)

            sph_state.particles_reordered.acc[i_p, i_b] = force_fields.add_acc(
                acc, sph_state.particles_reordered.pos[i_p, i_b], sph_state.particles_reordered.vel[i_p, i_b], t, i_p
            )


@qd.func
def func_task_compute_pressure_forces(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    dp_i = sph_state.particles_reordered.p[i_p, i_b] / sph_state.particles_reordered.rho[i_p, i_b] ** 2
    rho_j = (
        sph_state.particles_reordered.rho[j_p, i_b]
        * sph_state.particles_info_reordered.rho[j_p, i_b]
        / sph_state.particles_info_reordered.rho[j_p, i_b]
    )
    dp_j = sph_state.particles_reordered.p[j_p, i_b] / rho_j**2

    # Compute the pressure force contribution, Symmetric Formula
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    ret += (
        -sph_state.particles_info_reordered.rho[j_p, i_b]
        * sph_info.particle_volume[None]
        * (dp_i + dp_j)
        * func_cubic_kernel_derivative(d_ij, sph_info.support_radius[None])
    )


@qd.kernel(fastcache=True)
def kernel_compute_pressure_forces(sph_state: SPHState, sph_info: SPHInfo):
    """
    Add the acceleration of the pressure of the equation of state to the sorted particles.

    This is the pressure model of weakly compressible smoothed particle hydrodynamics (WCSPH).
    """
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            rho0 = sph_state.particles_info_reordered.rho[i_p, i_b]
            stiff = sph_state.particles_info_reordered.stiffness[i_p, i_b]
            expnt = sph_state.particles_info_reordered.exponent[i_p, i_b]

            sph_state.particles_reordered.rho[i_p, i_b] = qd.max(sph_state.particles_reordered.rho[i_p, i_b], rho0)

            sph_state.particles_reordered.p[i_p, i_b] = stiff * (
                qd.pow(sph_state.particles_reordered.rho[i_p, i_b] / rho0, expnt) - 1.0
            )

    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            acc = qd.Vector.zero(gs.qd_float, 3)
            func_for_all_neighbors(i_p, i_b, acc, sph_state, sph_info, func_task_compute_pressure_forces)
            sph_state.particles_reordered.acc[i_p, i_b] += acc


@qd.kernel(fastcache=True)
def kernel_advect_velocity(sph_state: SPHState, sph_info: SPHInfo):
    """Advance the velocity of the sorted particles by their acceleration."""
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            sph_state.particles_reordered.vel[i_p, i_b] += (
                sph_info.substep_dt[None] * sph_state.particles_reordered.acc[i_p, i_b]
            )


@qd.kernel(fastcache=True)
def kernel_advect_position(sph_state: SPHState, sph_info: SPHInfo):
    """Advance the position of the sorted particles by their velocity, inside the boundary."""
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            pos = sph_state.particles_reordered.pos[i_p, i_b]
            new_vel = sph_state.particles_reordered.vel[i_p, i_b]
            new_pos = pos + sph_info.substep_dt[None] * new_vel
            corrected_pos, corrected_vel = func_impose_cube_boundary(new_pos, new_vel, sph_info.boundary)
            sph_state.particles_reordered.vel[i_p, i_b] = corrected_vel
            sph_state.particles_reordered.pos[i_p, i_b] = corrected_pos


# ------------------------------------------------------------------------------------
# ------------------------------------- DFSPH ----------------------------------------
# ------------------------------------------------------------------------------------


@qd.func
def func_task_compute_dfsph_factor(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
    grad_j = -sph_info.particle_volume[None] * func_cubic_kernel_derivative(d_ij, sph_info.support_radius[None])
    # The last entry of `ret` accumulates sum_grad_p_k and the first three grad_p_i (see kernel_compute_dfsph_factor)
    ret[3] += grad_j.norm_sqr()
    for k in qd.static(range(3)):
        ret[k] -= grad_j[k]


@qd.kernel(fastcache=True)
def kernel_compute_dfsph_factor(sph_state: SPHState, sph_info: SPHInfo):
    """
    Compute the factor of the divergence-free and density solves of the sorted particles.

    These are the solves of divergence-free smoothed particle hydrodynamics (DFSPH).
    """
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = qd.Vector.zero(gs.qd_float, 4)
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_dfsph_factor)

            grad_p_i = qd.Vector([ret[0], ret[1], ret[2]], dt=gs.qd_float)
            sum_grad_p_k = ret[3] + grad_p_i.norm_sqr()

            # Compute pressure stiffness denominator
            factor = gs.qd_float(0.0)
            if sum_grad_p_k > 1e-6:
                factor = -1.0 / sum_grad_p_k
            sph_state.particles_reordered.dfsph_factor[i_p, i_b] = factor

            # Reset the implicit pressure accumulated by the density solve (see kernel_pressure_solve_iteration)
            sph_state.particles_reordered.p[i_p, i_b] = 0.0


@qd.func
def func_task_compute_density_time_derivative(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    v_i = sph_state.particles_reordered.vel[i_p, i_b]
    v_j = sph_state.particles_reordered.vel[j_p, i_b]

    x_i = sph_state.particles_reordered.pos[i_p, i_b]
    x_j = sph_state.particles_reordered.pos[j_p, i_b]

    ret.drho += sph_info.particle_volume[None] * (v_i - v_j).dot(
        func_cubic_kernel_derivative(x_i - x_j, sph_info.support_radius[None])
    )
    ret.num_neighbors += 1


@qd.kernel(fastcache=True)
def kernel_compute_density_time_derivative(sph_state: SPHState, sph_info: SPHInfo):
    """Compute the rate of change of the density of the sorted particles, its positive part alone."""
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            ret = qd.Struct(drho=0.0, num_neighbors=0)
            func_for_all_neighbors(i_p, i_b, ret, sph_state, sph_info, func_task_compute_density_time_derivative)

            # only correct positive divergence
            drho = qd.max(ret.drho, 0.0)
            num_neighbors = ret.num_neighbors

            # Do not perform divergence solve when particle deficiency happens
            if num_neighbors < 20:
                drho = 0.0

            sph_state.particles_reordered.drho[i_p, i_b] = drho


@qd.func
def func_task_compute_density_star(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    v_i = sph_state.particles_reordered.vel[i_p, i_b]
    v_j = sph_state.particles_reordered.vel[j_p, i_b]
    x_i = sph_state.particles_reordered.pos[i_p, i_b]
    x_j = sph_state.particles_reordered.pos[j_p, i_b]
    ret += sph_info.particle_volume[None] * (v_i - v_j).dot(
        func_cubic_kernel_derivative(x_i - x_j, sph_info.support_radius[None])
    )


@qd.kernel(fastcache=True)
def kernel_compute_density_star(dfsph_factor_scale: float, sph_state: SPHState, sph_info: SPHInfo):
    """
    Predict the relative density of the sorted particles at the end of the substep, at least one.

    The factor of the solves of divergence-free smoothed particle hydrodynamics (DFSPH) is multiplied by
    `dfsph_factor_scale` on the way.
    """
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            delta = gs.qd_float(0.0)
            func_for_all_neighbors(i_p, i_b, delta, sph_state, sph_info, func_task_compute_density_star)
            drho = (
                sph_state.particles_reordered.rho[i_p, i_b] / sph_info.density0[None]
                + sph_info.substep_dt[None] * delta
            )
            sph_state.particles_reordered.drho[i_p, i_b] = qd.max(drho, 1.0)
            sph_state.particles_reordered.dfsph_factor[i_p, i_b] *= dfsph_factor_scale


@qd.kernel(fastcache=True)
def kernel_compute_density_error(offset: float, sph_state: SPHState, sph_info: SPHInfo) -> float:
    """
    Return the sum of the density errors of the sorted particles.

    `offset` is subtracted from the density error of each particle.
    """
    density_error = gs.qd_float(0.0)
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            density_error += sph_info.density0[None] * sph_state.particles_reordered.drho[i_p, i_b] - offset
    return density_error


@qd.func
def func_task_divergence_solve_iteration(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    b_j = sph_state.particles_reordered.drho[j_p, i_b]
    k_j = b_j * sph_state.particles_reordered.dfsph_factor[j_p, i_b]
    # TODO: make the neighbor density different for multiphase fluid
    k_sum = ret.k_i + k_j
    if qd.abs(k_sum) > 1e-5:
        d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
        grad_p_j = -sph_info.particle_volume[None] * func_cubic_kernel_derivative(d_ij, sph_info.support_radius[None])
        # ki, kj already contain inverse density, i.e., density canceled if not mutiphase flow
        ret.dv -= k_sum * grad_p_j


# The task of the divergence-free solve, `drho` offset by one and the velocity change scaled by the time step. The offset
# stays a constant, which fast math folds into the arithmetic: a runtime offset would change the rounding.
@qd.func
def func_task_density_solve_iteration(
    i_p: int, j_p: int, i_b: int, ret: qd.template(), sph_state: SPHState, sph_info: SPHInfo
):
    b_j = sph_state.particles_reordered.drho[j_p, i_b] - 1.0
    k_j = b_j * sph_state.particles_reordered.dfsph_factor[j_p, i_b]
    # TODO: make the neighbor density0 different for multiphase fluid
    k_sum = ret.k_i + k_j
    if qd.abs(k_sum) > 1e-5:
        d_ij = sph_state.particles_reordered.pos[i_p, i_b] - sph_state.particles_reordered.pos[j_p, i_b]
        grad_p_j = -sph_info.particle_volume[None] * func_cubic_kernel_derivative(d_ij, sph_info.support_radius[None])
        # Directly update velocities instead of storing pressure accelerations. ki, kj already contain inverse density,
        # i.e., density canceled if not mutiphase flow.
        ret.dv -= sph_info.substep_dt[None] * k_sum * grad_p_j


@qd.kernel(fastcache=True)
def kernel_pressure_solve_iteration(sph_state: SPHState, sph_info: SPHInfo, solve_density: qd.template()):
    """
    Run one Jacobi iteration of the divergence-free solve or the density solve on the velocity of the sorted particles.

    These are the solves of divergence-free smoothed particle hydrodynamics (DFSPH). The divergence-free solve cancels
    the rate of change of the density of the particles, `drho`, and the density solve the deviation of their relative
    density, `drho` too, from one, which it accumulates into their pressure.
    """
    for i_p, i_b in qd.ndrange(sph_state.particles_reordered.pos.shape[0], sph_state.particles_reordered.pos.shape[1]):
        if sph_state.particles_reordered.is_active[i_p, i_b]:
            b_i = sph_state.particles_reordered.drho[i_p, i_b]
            if qd.static(solve_density):
                b_i = b_i - 1.0
            k_i = b_i * sph_state.particles_reordered.dfsph_factor[i_p, i_b]

            ret = qd.Struct(dv=qd.Vector.zero(gs.qd_float, 3), k_i=k_i)
            func_for_all_neighbors(
                i_p,
                i_b,
                ret,
                sph_state,
                sph_info,
                func_task_density_solve_iteration if qd.static(solve_density) else func_task_divergence_solve_iteration,
            )
            sph_state.particles_reordered.vel[i_p, i_b] = sph_state.particles_reordered.vel[i_p, i_b] + ret.dv

            # The stiffness k_i is the pressure this Jacobi iteration applies to cancel the current density error, up to
            # the -1/rho0 scaling carried by dfsph_factor, so summing -k_i * rho0 over the iterations of the density
            # solve yields the implicit pressure field that enforces incompressibility over the substep. This exposes
            # in the pressure the same physical quantity that the equation of state provides under weakly compressible
            # smoothed particle hydrodynamics (WCSPH). It is zeroed once per substep in kernel_compute_dfsph_factor.
            if qd.static(solve_density):
                rho0 = sph_state.particles_info_reordered.rho[i_p, i_b]
                sph_state.particles_reordered.p[i_p, i_b] = sph_state.particles_reordered.p[i_p, i_b] - k_i * rho0


@qd.kernel(fastcache=True)
def kernel_add_particles(
    particle_start: int,
    active: bool,
    mat_rho: float,
    mat_stiffness: float,
    mat_exponent: float,
    mat_mu: float,
    mat_gamma: float,
    pos: qd.types.ndarray(),
    sph_state: SPHState,
    sph_info: SPHInfo,
):
    for i_p_, i_b in qd.ndrange(pos.shape[0], sph_state.particles.pos.shape[1]):
        i_p = i_p_ + particle_start
        sph_state.particles.is_active[i_p, i_b] = active
        sph_state.particles.pos[i_p, i_b] = qd.Vector([pos[i_p_, 0], pos[i_p_, 1], pos[i_p_, 2]])
        sph_state.particles.vel[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
        sph_state.particles.p[i_p, i_b] = 0

    for i_p_ in range(pos.shape[0]):
        i_p = i_p_ + particle_start
        sph_info.particles.rho[i_p] = mat_rho
        sph_info.particles.stiffness[i_p] = mat_stiffness
        sph_info.particles.exponent[i_p] = mat_exponent
        sph_info.particles.mu[i_p] = mat_mu
        sph_info.particles.gamma[i_p] = mat_gamma
        sph_info.particles.mass[i_p] = sph_info.particle_volume[None] * mat_rho


@qd.kernel(fastcache=True)
def kernel_clear_particles_acc(particles_idx: qd.types.ndarray(), envs_idx: qd.types.ndarray(), sph_state: SPHState):
    for i_p_, i_b_ in qd.ndrange(particles_idx.shape[1], envs_idx.shape[0]):
        i_p = particles_idx[i_b_, i_p_]
        i_b = envs_idx[i_b_]
        # See 'kernel_set_particles_pos' in particle_solver.py
        if i_p >= 0:
            sph_state.particles.acc[i_p, i_b] = qd.Vector.zero(gs.qd_float, 3)
