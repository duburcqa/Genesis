import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, Size, field

from .base_solver import Solver, TimeBasedMixin


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


class SFSolver(TimeBasedMixin, Solver):
    """
    Stable Fluid solver for eulerian-based gaseous simulation.
    """

    sf_state: SFState | None = None
    sf_info: SFInfo | None = None

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        if options is None:
            return

        self.n_grid = options.res
        self.solver_iters = options.solver_iters
        self.decay = options.decay

        self.inlet_s = options.inlet_s

        self.jets = ()

    def build(self):
        super().build()

        if self.is_active:
            self.sf_state = SFState(sizes=dict(sf_grid=self.n_grid, jets=len(self.jets)))
            self.sf_info = SFInfo(values=dict(substep_dt=self.substep_dt, inlet_s=self.inlet_s, decay=self.decay))

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return bool(self.jets)

    def set_jets(self, jets):
        assert isinstance(jets, (list, tuple))
        self.jets = tuple(jets)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f, t):
        kernel_advect_and_impulse(t, self.sf_state, self.sf_info, self.jets)
        kernel_divergence(self.sf_state)
        for i_iter in range(self.solver_iters):
            kernel_pressure_jacobi(i_iter % 2, self.sf_state)
        kernel_subtract_gradient(self.solver_iters % 2, self.sf_state)

    def substep_post_coupling(self, f):
        return

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f):
        pass

    def set_state(self, f, state, envs_idx=None):
        pass


@qd.kernel
def kernel_advect_and_impulse(t: float, sf_state: SFState, sf_info: SFInfo, jets: qd.template()):
    """Advect the velocity and the smoke along the velocity, then let every jet inject its smoke and its momentum."""
    n_grid = sf_state.vel.shape[0]
    substep_dt = sf_info.substep_dt[None]
    for i, j, k in qd.ndrange(n_grid, n_grid, n_grid):
        p = qd.Vector([i, j, k], dt=gs.qd_float) + 0.5
        p = func_backtrace(sf_state.vel, p, substep_dt, n_grid)
        vel_advected = func_trilerp(sf_state.vel, p, n_grid)
        smoke_advected = func_trilerp(sf_state.smoke, p, n_grid)

        for q in qd.static(range(len(jets))):
            q_f = smoke_advected[q]

            imp_dir = jets[q].get_tan_dir(t)
            factor = jets[q].get_factor(i, j, k, 1.0 / n_grid, t)
            momentum = (imp_dir * sf_info.inlet_s[None] * factor) * substep_dt

            vel_advected += momentum

            sf_state.smoke[i, j, k][q] = (1 - factor) * q_f + factor
            sf_state.smoke[i, j, k][q] -= sf_info.decay[None] * substep_dt
            sf_state.smoke[i, j, k][q] = max(0.0, sf_state.smoke[i, j, k][q])

        sf_state.vel_advected[i, j, k] = vel_advected


@qd.kernel(fastcache=True)
def kernel_divergence(sf_state: SFState):
    """Compute the divergence of the advected velocity, mirrored at the walls of the grid."""
    n_grid = sf_state.vel.shape[0]
    for I in qd.grouped(qd.ndrange(n_grid, n_grid, n_grid)):
        # The rows of 'vel_lo' and 'vel_hi' are the velocities of the neighbours below and above the cell on each axis
        vel_lo = qd.Matrix.zero(gs.qd_float, 3, 3)
        vel_hi = qd.Matrix.zero(gs.qd_float, 3, 3)
        for d in qd.static(range(3)):
            offset = qd.Vector.unit(3, d, gs.qd_int)
            vel_lo[d, :] = sf_state.vel_advected[qd.math.clamp(I - offset, 0, n_grid - 1)]
            vel_hi[d, :] = sf_state.vel_advected[qd.math.clamp(I + offset, 0, n_grid - 1)]
        vel_c = sf_state.vel_advected[I]
        # The velocity across a wall of the grid is mirrored
        for d in qd.static(range(3)):
            offset = qd.Vector.unit(3, d, gs.qd_int)
            if not ((I - offset >= 0).all() and (I - offset < n_grid).all()):
                vel_lo[d, d] = -vel_c[d]
            if not ((I + offset >= 0).all() and (I + offset < n_grid).all()):
                vel_hi[d, d] = -vel_c[d]
        sf_state.div[I] = 0.5 * (
            vel_hi[0, 0] - vel_lo[0, 0] + vel_hi[1, 1] - vel_lo[1, 1] + vel_hi[2, 2] - vel_lo[2, 2]
        )


@qd.kernel(fastcache=True)
def kernel_pressure_jacobi(i_src: int, sf_state: SFState):
    """
    Run one Jacobi iteration of the pressure projection, from one pressure buffer to the other.

    `i_src` is the buffer the iteration reads, the other one receiving the result.
    """
    n_grid = sf_state.vel.shape[0]
    i_dst = 1 - i_src
    for I in qd.grouped(qd.ndrange(n_grid, n_grid, n_grid)):
        pressure_sum = gs.qd_float(0.0)
        for d in qd.static(range(3)):
            offset = qd.Vector.unit(3, d, gs.qd_int)
            pressure_sum += sf_state.pressure[i_src, qd.math.clamp(I - offset, 0, n_grid - 1)]
            pressure_sum += sf_state.pressure[i_src, qd.math.clamp(I + offset, 0, n_grid - 1)]
        sf_state.pressure[i_dst, I] = (pressure_sum - sf_state.div[I]) / 6.0


@qd.kernel(fastcache=True)
def kernel_subtract_gradient(i_src: int, sf_state: SFState):
    """
    Make the advected velocity divergence-free by subtracting the gradient of the pressure.

    `i_src` is the pressure buffer holding the result of the last Jacobi iteration, copied into the first buffer as the
    warm start of the next substep.
    """
    n_grid = sf_state.vel.shape[0]
    for I in qd.grouped(qd.ndrange(n_grid, n_grid, n_grid)):
        pressure_diff = qd.Vector.zero(gs.qd_float, 3)
        for d in qd.static(range(3)):
            offset = qd.Vector.unit(3, d, gs.qd_int)
            pressure_diff[d] = (
                sf_state.pressure[i_src, qd.math.clamp(I + offset, 0, n_grid - 1)]
                - sf_state.pressure[i_src, qd.math.clamp(I - offset, 0, n_grid - 1)]
            )
        sf_state.vel[I] = sf_state.vel_advected[I] - 0.5 * pressure_diff
        # Keep the result in the first buffer as the warm start of the next substep. No read races with this write: the
        # stencil above reads the second buffer whenever it holds the result, and the copy leaves the cell unchanged
        # otherwise.
        sf_state.pressure[0, I] = sf_state.pressure[i_src, I]


@qd.func
def func_trilerp(qf: qd.template(), p: qd.types.vector(3), n_grid: int):
    """
    Interpolate trilinearly a vector field of the grid at a position.

    The position is in cell units. The weights of the cells outside the grid are dropped and the others renormalized.
    """
    base_I = qd.floor(p - 0.5, gs.qd_int)
    p_I = p - 0.5

    q = qd.Vector.zero(gs.qd_float, qf[0, 0, 0].n)
    w_total = 0.0
    for offset in qd.static(qd.grouped(qd.ndrange(2, 2, 2))):
        grid_I = base_I + offset
        w_xyz = 1 - qd.abs(p_I - grid_I)
        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
        q += w * qf[qd.math.clamp(grid_I, 0, n_grid - 1)]
        w_total += w
    q /= w_total
    return q


@qd.func
def func_backtrace(vf: qd.template(), p: qd.types.vector(3), dt: float, n_grid: int):
    """Trace a position back along a velocity field over a timestep, by third-order Runge-Kutta."""
    v1 = func_trilerp(vf, p, n_grid)
    p1 = p - 0.5 * dt * v1
    v2 = func_trilerp(vf, p1, n_grid)
    p2 = p - 0.75 * dt * v2
    v3 = func_trilerp(vf, p2, n_grid)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p
