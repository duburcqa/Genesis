from typing import Any, Literal

import quadrants as qd

import genesis as gs
from genesis.typing import PositiveFloat

from .base import Base, func_corotated_stress


@qd.data_oriented
class Elastic(Base):
    """
    The elastic material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 3e5.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.
    model : str, optional
        Stress model ('corotation', 'neohooken'). Default is 'corotation'.
    """

    E: PositiveFloat = 3e5
    model: Literal["corotation", "neohooken"] = "corotation"

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        # The corotated stress reads U @ V^T, the neo-Hookean one F_tmp and J alone
        self.is_svd_needed = self.model == "corotation"

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
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

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
        return func_elastic_stress(U, V, F_tmp, F_new, J, self.mu, self.lam, corotated=self.model == "corotation")


@qd.func
def func_elastic_stress(
    U: qd.types.matrix(3, 3),
    V: qd.types.matrix(3, 3),
    F_tmp: qd.types.matrix(3, 3),
    F_new: qd.types.matrix(3, 3),
    J: float,
    mu: float,
    lam: float,
    corotated: qd.template(),
):
    """
    Return the stress of the corotated or the neo-Hookean elastic model, of the given Lame parameters.

    The arguments are those of 'Base.update_stress'.
    """
    stress = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(corotated):
        stress = func_corotated_stress(F_new, U, V, J, mu, lam)
    else:
        stress = mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (lam * qd.log(J) - mu)
    return stress
