"""The couplings of the surface of the finite element method (FEM) entities with the other solvers, and its floor."""

import quadrants as qd

import genesis.utils.array_class as array_class
from genesis.engine.boundaries import func_impose_floor_boundary
from genesis.engine.pipeline import Inline, System, register_system
from genesis.engine.solvers.fem_solver import FEMSolver
from genesis.engine.step import Step
from genesis.options.solvers import LegacyCouplerOptions


@register_system
class FEMSurface(System):
    """
    Couple the surface of the finite element method (FEM) entities with the other solvers, then impose the floor on it.

    The couplings of the surface vertices are the funcs of `surface_couplings`, which the coupling Systems add (see
    func_fem_surface_couplings).
    """

    def __init__(self, fem_solver: FEMSolver, options: LegacyCouplerOptions) -> None:
        self.surface_couplings = Inline(kernel_fem_surface, func_fem_surface_couplings, run_without_funcs=False)

    def compose(self, step: Step, fem_solver: FEMSolver) -> None:
        self.surface_couplings.bind()
        step.couplings.add(self.surface_couplings, kernel_fem_surface_boundary)


@qd.func
def func_surface_triangle(i_s: int, i_b: int, f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """Return the vertices of a surface triangle, the mass of its element and its unit normal."""
    i_e = fem_info.surface.tri2el[i_s]
    mass = fem_info.elements.mass_scaled[i_e] / fem_info.vol_scale[None]
    tri2v = fem_info.surface.tri2v[i_s]
    p1 = fem_state.vertices.pos[f, tri2v[0], i_b]
    p2 = fem_state.vertices.pos[f, tri2v[1], i_b]
    p3 = fem_state.vertices.pos[f, tri2v[2], i_b]
    surface_normal = qd.math.cross(p2 - p1, p3 - p1)
    surface_normal = surface_normal / surface_normal.norm(fem_info.EPS[None])
    return tri2v, mass, surface_normal


@qd.func
def func_fem_surface_couplings(
    f: int,
    i_v: int,
    i_b: int,
    pos: qd.types.vector(3),
    vel: qd.types.vector(3),
    mass: float,
    surface_normal: qd.types.vector(3),
) -> qd.types.vector(3):
    """
    Return the velocity of a vertex of a surface triangle, once coupled with the other solvers.

    The vertex is at `pos` with velocity `vel`, `mass` is the mass of the element of the triangle and `surface_normal`
    its unit normal. The placeholder of the extension point of the couplings of the surface, which the coupling Systems
    add their funcs to (see 'FEMSurface.surface_couplings').
    """
    return vel


@qd.kernel
def kernel_fem_surface(f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """Couple the vertices of the surface triangles of the finite element method entities with the other solvers."""
    for i_s, i_b in qd.ndrange(fem_info.surface.tri2el.shape[0], fem_state.vertices.pos.shape[2]):
        tri2v, mass, surface_normal = func_surface_triangle(i_s, i_b, f, fem_state, fem_info)
        for j in range(3):
            i_v = tri2v[j]
            vel = fem_state.vertices.vel[f + 1, i_v, i_b]
            vel = func_fem_surface_couplings(
                f, i_v, i_b, fem_state.vertices.pos[f, i_v, i_b], vel, mass, surface_normal
            )
            fem_state.vertices.vel[f + 1, i_v, i_b] = vel


@qd.kernel(fastcache=True)
def kernel_fem_surface_boundary(f: int, fem_state: array_class.FEMState, fem_info: array_class.FEMInfo):
    """
    Impose the floor on the surface vertices of the finite element method entities.

    It runs once per surface vertex: the reverse pass of an overwrite of the velocity of a vertex by each of the surface
    triangles sharing it would race on its adjoint.
    """
    for i_sv, i_b in qd.ndrange(fem_info.surface.vertices_idx.shape[0], fem_state.vertices.pos.shape[2]):
        i_v = fem_info.surface.vertices_idx[i_sv]
        _, fem_state.vertices.vel[f + 1, i_v, i_b] = func_impose_floor_boundary(
            fem_state.vertices.pos[f, i_v, i_b], fem_state.vertices.vel[f + 1, i_v, i_b], fem_info.boundary
        )
