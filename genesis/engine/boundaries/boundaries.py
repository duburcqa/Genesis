import numpy as np

import quadrants as qd

import genesis as gs
from genesis.utils.data import Data, DataKind, field
from genesis.utils.repr import brief


class CubeBoundaryBox(Data, kind=DataKind.CONSTANT, shape=()):
    """An axis-aligned box confining the particles, which bounce off its walls with the given restitution."""

    lower: qd.Tensor = field((float, 3))
    upper: qd.Tensor = field((float, 3))
    restitution: qd.Tensor = field(float)


class FloorBoundaryPlane(Data, kind=DataKind.CONSTANT, shape=()):
    """The floor of the FEM solver, at a height and with a restitution (see FloorBoundary in boundaries.py)."""

    height: qd.Tensor = field(float)
    restitution: qd.Tensor = field(float)


class CubeBoundary:
    """An axis-aligned box confining the particles, which bounce off its walls with the given restitution."""

    def __init__(self, lower, upper, restitution=0.0):
        self.restitution = restitution

        self.upper = np.array(upper, dtype=gs.np_float)
        self.lower = np.array(lower, dtype=gs.np_float)
        assert (self.upper >= self.lower).all()

    @property
    def values(self) -> dict:
        """The constants of the box (see CubeBoundaryBox)."""
        return dict(lower=self.lower, upper=self.upper, restitution=self.restitution)

    def is_inside(self, pos):
        return np.all(pos < self.upper) and np.all(pos > self.lower)

    def __repr__(self):
        return (
            f"{brief(self)}\n"
            f"lower       : {brief(self.lower)}\n"
            f"upper       : {brief(self.upper)}\n"
            f"restitution : {brief(self.restitution)}"
        )


class FloorBoundary:
    """A horizontal floor below which nothing goes, bouncing off it with the given restitution."""

    def __init__(self, height, restitution=0.0):
        self.height = height
        self.restitution = restitution

    def __repr__(self):
        return f"{brief(self)}\nheight      : {brief(self.height)}\nrestitution : {brief(self.restitution)}"


@qd.func
def func_impose_cube_boundary(pos: qd.types.vector(3), vel: qd.types.vector(3), boundary: CubeBoundaryBox):
    """Clamp a position inside a cube boundary, the velocity bouncing off any wall it crosses or presses against."""
    lower = boundary.lower[None]
    upper = boundary.upper[None]
    for i in qd.static(range(3)):
        if pos[i] >= upper[i] and vel[i] >= 0:
            vel[i] *= -boundary.restitution[None]
        elif pos[i] <= lower[i] and vel[i] <= 0:
            vel[i] *= -boundary.restitution[None]

    pos = qd.max(qd.min(pos, upper), lower)

    return pos, vel


@qd.func
def func_impose_floor_boundary(pos: qd.types.vector(3), vel: qd.types.vector(3), boundary: FloorBoundaryPlane):
    """Clamp a position above a floor boundary, the velocity bouncing off it when it crosses or presses against it."""
    height = boundary.height[None]
    if pos[2] <= height and vel[2] <= 0:
        vel[2] *= -boundary.restitution[None]

    pos[2] = qd.max(pos[2], height)

    return pos, vel
