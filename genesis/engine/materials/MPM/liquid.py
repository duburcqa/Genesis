from typing import Any

from pydantic import StrictBool

import quadrants as qd

import genesis as gs

from .base import Base


@qd.data_oriented
class Liquid(Base):
    """
    The liquid material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.
    viscous : bool, optional
        Whether the liquid is viscous. Simply sets mu to zero when non-viscous. Default is False.
    """

    viscous: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not self.viscous:
            self.mu = 0.0
        # The viscous term 2 * mu * (F_tmp - U @ V^T) @ F_tmp^T alone reads the singular value decomposition, and mu
        # is zero without viscosity
        self.is_svd_needed = self.viscous

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
        F_new = qd.Matrix.identity(gs.qd_float, 3) * qd.pow(J, 1.0 / 3.0)
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
        stress = 2 * self.mu * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress
