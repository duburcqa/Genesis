from typing import Any, Literal

from pydantic import model_validator

import quadrants as qd

import genesis as gs

from .base import Base


@qd.data_oriented
class Elastic(Base):
    """
    The elastic material class for FEM.

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
    model : str, optional
        Constitutive model to use for stress computation. Options are:
        - 'linear': Linear elasticity model
        - 'stable_neohookean': A numerically stable Neo-Hookean model
        - 'linear_corotated': Linear corotated elasticity model
        Default is 'linear'.
    contact_resistance : float | None, optional
        IPC contact resistance/stiffness override. ``None`` uses the coupler global
        default. Default is None.
    """

    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "linear"

    @model_validator(mode="before")
    @classmethod
    def _resolve_deprecated_model(cls, data: dict) -> dict:
        if data.get("model") == "stable_neohooken":
            gs.logger.warning("The 'stable_neohooken' model is deprecated. Use 'stable_neohookean' instead.")
            data["model"] = "stable_neohookean"
        return data

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        self.has_rotation = self.model == "linear_corotated"
        if self.model == "linear":
            self.hessian_invariant = True
        elif self.model == "stable_neohookean":
            self.hessian_invariant = False

    @qd.func
    def pre_compute(self, F: qd.types.matrix(3, 3)):
        U, S, V = qd.svd(F)
        return U @ V.transpose()

    @qd.func
    def update_stress(
        self, mu: float, lam: float, J: float, F: qd.types.matrix(3, 3), actu: float, m_dir: qd.types.vector(3)
    ):
        stress = qd.Matrix.zero(gs.qd_float, 3, 3)
        if qd.static(self.model == "linear"):
            I = qd.Matrix.identity(dt=gs.qd_float, n=3)
            stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I
        elif qd.static(self.model == "stable_neohookean"):
            IC = F.norm_sqr()
            dJdF0 = F[:, 1].cross(F[:, 2])
            dJdF1 = F[:, 2].cross(F[:, 0])
            dJdF2 = F[:, 0].cross(F[:, 1])
            dJdF = qd.Matrix.cols([dJdF0, dJdF1, dJdF2])
            alpha = 1 + 0.75 * mu / lam
            stress = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF
        else:
            gs.raise_exception("Linear corotated stress update is not implemented yet.")
        return stress

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
        if qd.static(self.model == "stable_neohookean"):
            raise NotImplementedError("Hessian computation is not implemented for stable_neohookean model.")
        energy, gradient = self.compute_energy_gradient(mu, lam, J, F, R, actu, m_dir)

        for i, j in qd.static(qd.ndrange(3, 3)):
            hessian[i, j, i_e, i_b] = qd.Matrix.zero(gs.qd_float, 3, 3)

        for i, k in qd.static(qd.ndrange(3, 3)):
            hessian[i, i, i_e, i_b][k, k] = mu

        for i, j, alpha, beta in qd.ndrange(3, 3, 3, 3):
            hessian[j, beta, i_e, i_b][i, alpha] += mu * R[i, beta] * R[alpha, j] + lam * R[alpha, beta] * R[i, j]

        return energy, gradient

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
        if qd.static(self.model == "stable_neohookean"):
            gs.raise_exception("Gradient computation is not implemented for stable_neohookean model.")
        # The linear model is the linear corotated one around the identity, which the solver passes as its rotation
        F_hat = R.transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * R @ eps + lam * trEps * R
        return energy, gradient

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
        energy = gs.qd_float(0.0)
        if qd.static(self.model == "stable_neohookean"):
            _lambda = lam + mu
            _alpha = 1.0 + mu / _lambda
            Ic = F.norm_sqr()
            Jminus1 = J - _alpha
            energy = 0.5 * (mu * (Ic - 3.0) + _lambda * Jminus1**2)
        else:
            energy, _ = self.compute_energy_gradient(mu, lam, J, F, R, actu, m_dir)
        return energy
