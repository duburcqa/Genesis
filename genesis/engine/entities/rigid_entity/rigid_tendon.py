import genesis as gs
from genesis.repr_base import RBC


class RigidTendon(RBC):
    """
    Fixed tendon for rigid body entities.

    A fixed tendon defines a scalar length as a linear combination of joint coordinates,
    length = sum_i coef_i * qpos_i. It can carry passive forces (stiffness, damping, frictionloss), length limits,
    and be driven by an actuator whose scalar force is distributed to the coupled joints through the tendon
    coefficients.
    """

    def __init__(
        self,
        entity,
        name,
        idx,
        dofs_idx,
        coefs,
        stiffness,
        damping,
        frictionloss,
        length0,
        spring_length,
        limited,
        length_range,
        sol_params,
        act_gain,
        act_bias,
        act_force_range,
    ):
        self._name = name
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx

        self._dofs_idx = dofs_idx
        self._coefs = coefs
        self._stiffness = stiffness
        self._damping = damping
        self._frictionloss = frictionloss
        self._length0 = length0
        self._spring_length = spring_length
        self._limited = limited
        self._length_range = length_range
        self._sol_params = sol_params
        self._act_gain = act_gain
        self._act_bias = act_bias
        self._act_force_range = act_force_range

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def control_force(self, force, envs_idx=None):
        """
        Apply a scalar actuator force to the tendon, distributed to the coupled joints by the tendon coefficients.
        """
        self._solver.control_tendons_force(force, tendons_idx=self._idx, envs_idx=envs_idx)

    @gs.assert_built
    def control_position(self, length, envs_idx=None):
        """
        Position-control the tendon length through its actuator (requires an actuator with affine bias).
        """
        self._solver.control_tendons_position(length, tendons_idx=self._idx, envs_idx=envs_idx)

    @gs.assert_built
    def control_velocity(self, velocity, envs_idx=None):
        """
        Velocity-control the tendon length rate through its actuator (requires an actuator with affine bias).
        """
        self._solver.control_tendons_velocity(velocity, tendons_idx=self._idx, envs_idx=envs_idx)

    @gs.assert_built
    def get_length(self, envs_idx=None):
        """
        Returns the current tendon length sum_i coef_i * qpos_i.
        """
        return self._solver.get_tendons_length(tendons_idx=self._idx, envs_idx=envs_idx)[..., 0]

    @gs.assert_built
    def get_velocity(self, envs_idx=None):
        """
        Returns the current tendon length rate sum_i coef_i * qvel_i.
        """
        return self._solver.get_tendons_velocity(tendons_idx=self._idx, envs_idx=envs_idx)[..., 0]

    def set_sol_params(self, sol_params):
        """
        Set the solver parameters of this tendon's limit and friction constraints.
        """
        if self._solver.is_built:
            self._solver.set_tendons_sol_params(sol_params, tendons_idx=self._idx)
        else:
            self._sol_params = sol_params

    @property
    def sol_params(self):
        """
        Returns the solver parameters of this tendon's limit and friction constraints.
        """
        if self._solver.is_built:
            return self._solver.get_tendons_sol_params(tendons_idx=self._idx)[..., 0, :]
        return self._sol_params

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """Returns the unique id of the tendon."""
        return self._uid

    @property
    def name(self):
        """Returns the name of the tendon."""
        return self._name

    @property
    def entity(self):
        """Returns the entity that the tendon belongs to."""
        return self._entity

    @property
    def solver(self):
        """The RigidSolver object that the tendon belongs to."""
        return self._solver

    @property
    def idx(self):
        """Returns the global index of the tendon in the rigid solver."""
        return self._idx

    @property
    def idx_local(self):
        """Returns the local index of the tendon in the entity."""
        return self._idx - self._entity._tendon_start

    @property
    def dofs_idx(self):
        """Returns the global indices of the DoFs coupled by the tendon."""
        return self._dofs_idx

    @property
    def coefs(self):
        """Returns the coefficients of the linear joint combination defining the tendon length."""
        return self._coefs

    @property
    def is_built(self):
        """Whether the rigid entity this tendon belongs to is built."""
        return self.entity.is_built
