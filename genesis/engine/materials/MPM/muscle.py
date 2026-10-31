from typing import Any, Literal

from pydantic import Field

import quadrants as qd

from genesis.typing import PositiveFloat, PositiveInt, ValidFloat

from .elastic import Elastic


@qd.data_oriented
class Muscle(Elastic):
    """
    The muscle material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.
    model : str, optional
        Stress model ('corotation', 'neohooken'). Default is 'neohooken'.
    n_groups : int, optional
        Number of muscle groups. Default is 1.
    """

    E: PositiveFloat = 1e6
    model: Literal["corotation", "neohooken"] = "neohooken"
    n_groups: PositiveInt = 1

    # Auto-generated, equals E.
    stiffness: ValidFloat = Field(default=0.0, exclude=True)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.stiffness = self.E

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
        # Quadrants funcs reject 'super()' (see Elastic)
        stress = Elastic.update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir)

        AAt = m_dir.outer_product(m_dir)
        stress += self.stiffness * actu * F_tmp @ AAt @ F_tmp.transpose()

        return stress
