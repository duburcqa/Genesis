import re
import sys
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field, PrivateAttr
from pydantic_core import PydanticCustomError

import quadrants as qd
from quadrants.lang._quadrants_callable import QuadrantsCallable

import genesis as gs
from genesis.typing import PositiveFloat, StrictInt, ValidFloat

from ..base import Material


if TYPE_CHECKING:
    from genesis.engine.entities.mpm_entity import MPMEntity

_SAMPLER_PATTERN = re.compile(r"^pbs(-\d+)?$|^random$|^regular$")


def _validate_sampler(v: str) -> str:
    if not isinstance(v, str) or not _SAMPLER_PATTERN.match(v):
        raise PydanticCustomError(
            "invalid_sampler",
            "Input should be 'pbs', 'pbs-<res>', 'random', or 'regular'",
            {"value": v},
        )
    return v


SamplerType = Annotated[str, BeforeValidator(_validate_sampler)]


@qd.data_oriented
class Base(Material["MPMEntity"]):
    """
    The base class of MPM materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.
    lam : float, optional
        The first Lame's parameter. Default is None, computed from E and nu.
    mu : float, optional
        The second Lame's parameter. Default is None, computed from E and nu.
    sampler : str, optional
        Particle sampler. Defaults to 'pbs', which relaxes the particles to a uniform density inside the shape, at the
        cost of a slower first sampling of each shape, cached afterwards. 'regular' places them on a grid, which leaves
        the surface stair-stepped. 'random' draws them uniformly, which leaves local density fluctuations.
    """

    # The quadrants funcs a kernel dispatches over are methods of the material, which pydantic takes for fields otherwise
    model_config = ConfigDict(ignored_types=(QuadrantsCallable,))

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0
    lam: ValidFloat | None = None
    mu: ValidFloat | None = None
    sampler: SamplerType = "pbs"

    # Whether 'update_F_S_Jp' or 'update_stress' reads the singular value decomposition of the deformation gradient
    # (see MPMStaticConfig), set by the subclasses
    is_svd_needed: bool = Field(default=True, exclude=True, repr=False)

    # Auto-generated fields.
    idx: StrictInt | None = Field(default=None, exclude=True)

    # Internal solver state - does not belong in an options class, kept pending larger refactor.
    _default_Jp: float = PrivateAttr(default=1.0)

    def model_post_init(self, context: Any) -> None:
        # Resolve Lame parameters
        if self.mu is None:
            self.mu = self.E / (2.0 * (1.0 + self.nu))
        if self.lam is None:
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @qd.func
    def update_F_S_Jp(
        self,
        J: float,
        F_tmp: qd.types.matrix(3, 3),
        U: qd.types.matrix(3, 3),
        S: qd.types.matrix(3, 3),
        V: qd.types.matrix(3, 3),
        Jp: float,
    ):
        """
        Return the deformation gradient, its singular values and the plastic volume ratio once the plasticity applies.

        `F_tmp` is the deformation gradient advected over the substep, `U`, `S`, `V` its singular value decomposition, `J`
        its determinant and `Jp` the plastic volume ratio of the particle.
        """
        raise NotImplementedError

    @qd.func
    def update_stress(
        self,
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
        """
        Return the stress of a particle, as the corotated elastic model gives it by default.

        `F_new` is the deformation gradient once the plasticity applies, `actu` the actuation of the particle and `m_dir`
        the direction of its muscle fibers, the other arguments being those of 'update_F_S_Jp'.
        """
        stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress
