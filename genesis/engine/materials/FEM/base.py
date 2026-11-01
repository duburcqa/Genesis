from typing import TYPE_CHECKING, Annotated, Any

from pydantic import ConfigDict, Field, StrictBool

import quadrants as qd
from quadrants.lang._quadrants_callable import QuadrantsCallable

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, StrictInt, ValidFloat

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.fem_entity import FEMEntity


@qd.data_oriented
class Base(Material["FEMEntity"]):
    """
    The base class of FEM materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    E : float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio, describing the material's volume change under stress. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    hydroelastic_modulus : float, optional
        Hydroelastic modulus for hydroelastic contact. Default is 1e7.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override for this material. ``None`` means
        use the IPC coupler global default (``IPCCouplerOptions.contact_resistance``).
    hessian_invariant : bool, optional
        If True, Hessian is computed only once. Default is False.
    """

    # The quadrants funcs a kernel dispatches over are methods of the material, which pydantic takes for fields otherwise
    model_config = ConfigDict(ignored_types=(QuadrantsCallable,))

    E: PositiveFloat = 1e6
    nu: Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)] = 0.2
    rho: PositiveFloat = 1000.0
    hydroelastic_modulus: PositiveFloat = 1e7
    friction_mu: NonNegativeFloat = 0.1
    contact_resistance: PositiveFloat | None = None
    hessian_invariant: StrictBool = False

    # Whether 'pre_compute' returns the rotation of the deformation gradient of each element, which the energy functions
    # read (see FEMStaticConfig), set by the subclasses
    has_rotation: bool = Field(default=False, exclude=True, repr=False)

    # Auto-generated fields, computed in model_post_init rather than by the user.
    mu: ValidFloat = Field(default=0.0, exclude=True)
    lam: ValidFloat = Field(default=0.0, exclude=True)
    idx: StrictInt | None = Field(default=None, exclude=True)

    def model_post_init(self, context: Any) -> None:
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @qd.func
    def pre_compute(self, F: qd.types.matrix(3, 3)):
        """Return the rotation of the deformation gradient `F` the energy of the material linearizes around."""
        raise NotImplementedError

    @qd.func
    def update_stress(
        self, mu: float, lam: float, J: float, F: qd.types.matrix(3, 3), actu: float, m_dir: qd.types.vector(3)
    ):
        """
        Return the first Piola-Kirchhoff stress of an element, zero for a material without a constitutive model.

        `mu`, `lam` are the Lame parameters of the element, `J` and `F` the determinant of its deformation gradient and
        the gradient itself, `actu` its actuation and `m_dir` the direction of its muscle fibers. The explicit solver
        compiles the stress of every material of the scene, so a material whose physics another solver owns (Cloth,
        simulated by the IPC coupler) returns zero rather than raising.
        """
        return qd.Matrix.zero(gs.qd_float, 3, 3)

    @qd.func
    def compute_energy_gradient_hessian(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        R: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
        hessian: qd.template(),
    ):
        """
        Return the energy density of an element and its gradient, writing the blocks of its Hessian into `hessian`.

        `R` is the rotation of 'pre_compute' for a material that has one, the identity otherwise, the other arguments
        being those of 'update_stress'.
        """
        raise NotImplementedError

    @qd.func
    def compute_energy_gradient(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        R: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        """Return the energy density of an element and its gradient (see 'compute_energy_gradient_hessian')."""
        raise NotImplementedError

    @qd.func
    def compute_energy(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        R: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        """Return the energy density of an element (see 'compute_energy_gradient_hessian')."""
        raise NotImplementedError
