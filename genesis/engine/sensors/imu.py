from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.options.sensors import IMU as IMUOptions
from genesis.options.sensors import CrossCouplingAxisType
from genesis.utils.misc import indices_to_mask, tensor_to_array

from .base_sensor import LinkAttachedSensorMixin, RigidSensorArrayMixin, SimpleSensor, SimpleSensorArray

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.vis.rasterizer_context import RasterizerContext


@qd.kernel(fastcache=True)
def _kernel_update_imu_raw_data(
    links_idx: qd.types.ndarray(),
    offsets_pos: qd.types.ndarray(),
    offsets_quat: qd.types.ndarray(),
    magnetic_field_vector: qd.types.ndarray(),
    raw_data: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    rigid_info: array_class.RigidInfo,
):
    """Compute the raw (linear acceleration, angular velocity, magnetic field) triplet of every IMU in its own frame."""
    for i_s, i_b in qd.ndrange(links_idx.shape[0], raw_data.shape[0]):
        i_l = links_idx[i_s]

        offset_pos = qd.Vector.zero(gs.qd_float, 3)
        mag = qd.Vector.zero(gs.qd_float, 3)
        for j in qd.static(range(3)):
            offset_pos[j] = offsets_pos[i_b, i_s, j]
            mag[j] = magnetic_field_vector[i_b, i_s, j]
        offset_quat = qd.Vector.zero(gs.qd_float, 4)
        for j in qd.static(range(4)):
            offset_quat[j] = offsets_quat[i_b, i_s, j]

        # Spatial velocity and acceleration are expressed at the root of the kinematic tree, hence the transport to the
        # link origin before converting the spatial acceleration to the classical one.
        cpos = dyn_state.links.pos[i_l, i_b] - dyn_state.links.root_COM[i_l, i_b]
        acc_ang = dyn_state.links.cacc_ang[i_l, i_b]
        ang = dyn_state.links.cd_ang[i_l, i_b]
        vel = dyn_state.links.cd_vel[i_l, i_b] + ang.cross(cpos)
        acc = dyn_state.links.cacc_lin[i_l, i_b] + acc_ang.cross(cpos) + ang.cross(vel)

        # Rigid-body transport from the link origin to the measurement point: a_imu = a + alpha x r + w x (w x r). A
        # zero mounting offset contributes nothing, so this stays unconditional.
        quat = dyn_state.links.quat[i_l, i_b]
        offset_pos_world = gu.qd_transform_by_quat(offset_pos, quat)
        acc = acc + acc_ang.cross(offset_pos_world) + ang.cross(ang.cross(offset_pos_world))

        # An accelerometer measures proper acceleration, so gravity is subtracted before rotating into the sensor frame
        sensor_quat = gu.qd_transform_quat_by_quat(quat, offset_quat)
        local_acc = gu.qd_inv_transform_by_quat(acc - rigid_info.gravity[i_b], sensor_quat)
        local_ang = gu.qd_inv_transform_by_quat(ang, sensor_quat)
        local_mag = gu.qd_inv_transform_by_quat(mag, sensor_quat)

        # Raw buffer layout: (B, n_imus * 9), one contiguous (acc, ang, mag) triplet per sensor
        i_cache_start = i_s * 9
        for j in qd.static(range(3)):
            raw_data[i_b, i_cache_start + j] = local_acc[j]
            raw_data[i_b, i_cache_start + 3 + j] = local_ang[j]
            raw_data[i_b, i_cache_start + 6 + j] = local_mag[j]


def _get_cross_axis_coupling_to_alignment_matrix(
    input: CrossCouplingAxisType, out: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Return the 3x3 alignment matrix of a cross-axis coupling, written into ``out`` when given and into a fresh tensor
    when None.

    A scalar fills every off-diagonal term. A 3-vector fills the off-diagonal terms of each column with its entry. Both
    keep a unit diagonal. A 3x3 array is taken as is.
    """
    if out is None:
        out = torch.eye(3, dtype=gs.tc_float, device=gs.device)

    if isinstance(input, float):
        # set off-diagonal elements to the scalar value
        torch.diagonal(out)[:] = input
        out.fill_diagonal_(1.0)
    elif isinstance(input, torch.Tensor):
        out.copy_(input)
    else:
        np_input = np.array(input)
        if np_input.shape == (3,):
            # set off-diagonal elements to the vector values
            out[1, 0] = np_input[0]
            out[2, 0] = np_input[0]
            out[0, 1] = np_input[1]
            out[2, 1] = np_input[1]
            out[0, 2] = np_input[2]
            out[1, 2] = np_input[2]
        elif np_input.shape == (3, 3):
            out.copy_(torch.tensor(np_input, dtype=gs.tc_float, device=gs.device))
    return out


class IMUReturnType(NamedTuple):
    lin_acc: torch.Tensor
    ang_vel: torch.Tensor
    mag: torch.Tensor


class IMUSensorArray(RigidSensorArrayMixin, SimpleSensorArray[IMUOptions, IMUReturnType]):
    """Array of every IMU of the scene."""

    def build(self):
        super().build()

        # The alignment of the accelerometer, gyroscope and magnetometer triplet of each sensor from its cross-axis
        # coupling, laid along the sensor axis in the order of the measured triplets
        alignment_rot_matrix = torch.stack(
            [
                torch.stack(
                    tuple(
                        map(
                            _get_cross_axis_coupling_to_alignment_matrix,
                            (
                                sensor.options.acc_cross_axis_coupling,
                                sensor.options.gyro_cross_axis_coupling,
                                sensor.options.mag_cross_axis_coupling,
                            ),
                        )
                    )
                )
                for sensor in self._sensors
            ]
        ).reshape(-1, 3, 3)
        self.alignment_rot_matrix = torch.stack([alignment_rot_matrix] * self._sim._B)
        magnetic_fields = [
            (0.0, 0.0, 0.5) if sensor.options.magnetic_field is None else sensor.options.magnetic_field
            for sensor in self._sensors
        ]
        # Per sensor, the magnetic field it measures
        magnetic_field_vector = torch.as_tensor(magnetic_fields, dtype=gs.tc_float, device=gs.device)
        self.magnetic_field_vector = torch.stack([magnetic_field_vector] * self._sim._B)
        self._debug_objects: list[list["Mesh"]] = [[] for _ in self._sensors]

    def set_cross_axis_coupling(
        self, i_s: int, i_triplet: int, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None
    ):
        """Set the cross-axis coupling of one measured triplet of sensor ``i_s`` (0 the accelerometer, 1 the gyroscope,
        2 the magnetometer), for the given environments."""
        rot_matrix = _get_cross_axis_coupling_to_alignment_matrix(cross_axis_coupling)
        self.alignment_rot_matrix[indices_to_mask(envs_idx, i_s * 3 + i_triplet, keepdim=False)] = rot_matrix

    def _get_return_format(self, options: IMUOptions) -> tuple[tuple[int, ...], ...]:
        return ((3,), (3,), (3,))

    def _get_cache_dtype(self) -> torch.dtype:
        return gs.tc_float

    def _update_raw_data(self, raw_data: torch.Tensor):
        _kernel_update_imu_raw_data(
            self.links_idx,
            self.offsets_pos,
            self.offsets_quat,
            self.magnetic_field_vector,
            raw_data,
            self.solver.dyn_state,
            self.solver.rigid_info,
        )

    def _apply_transform(self, data: torch.Tensor, timeline, *, is_measured: bool):
        # Apply alignment rotation to the (lin_acc, ang_vel, mag) triplet. View the flat cache as a stack of 3-vectors
        # and rotate them in place with the per-sensor `alignment_rot_matrix`. The alignment is stateless and identical
        # on both branches.
        data_xyz = data.view(data.shape[0], -1, 3)
        data_xyz.copy_(torch.matmul(self.alignment_rot_matrix, data_xyz.unsqueeze(-1)).squeeze(-1))

    def _draw_debug(self, i_s: int, context: "RasterizerContext"):
        """Draw the acceleration, angular velocity and magnetic field arrows of sensor ``i_s``, in the first rendered
        environment."""
        options = self._sensors[i_s].options
        env_idx = context.rendered_envs_idx[0] if self._sim.n_envs > 0 else None
        i_env = 0 if env_idx is None else env_idx

        link = self._links[i_s]
        quat = link.get_quat(env_idx, relative=False).reshape((4,))
        pos = link.get_pos(env_idx, relative=False).reshape((3,)) + gu.transform_by_quat(
            self.offsets_pos[i_env, i_s], quat
        )

        data = self.read(i_s, env_idx)
        acc_vec = data.lin_acc.reshape((3,)) * options.debug_acc_scale
        gyro_vec = data.ang_vel.reshape((3,)) * options.debug_gyro_scale
        mag_vec = data.mag.reshape((3,)) * options.debug_mag_scale

        # transform from local frame to world frame
        offset_quat = gu.transform_quat_by_quat(self.offsets_quat[i_env, i_s], quat)
        acc_vec = tensor_to_array(gu.transform_by_quat(acc_vec, offset_quat))
        gyro_vec = tensor_to_array(gu.transform_by_quat(gyro_vec, offset_quat))
        mag_vec = tensor_to_array(gu.transform_by_quat(mag_vec, offset_quat))

        debug_objects = self._debug_objects[i_s]
        for debug_object in debug_objects:
            context.clear_debug_object(debug_object)
        debug_objects.clear()

        debug_objects += filter(
            None,
            (
                context.draw_debug_arrow(pos=pos, vec=acc_vec, radius=0.006, color=options.debug_acc_color),
                context.draw_debug_arrow(pos=pos, vec=gyro_vec, radius=0.0055, color=options.debug_gyro_color),
                context.draw_debug_arrow(pos=pos, vec=mag_vec, radius=0.005, color=options.debug_mag_color),
            ),
        )


class IMUSensor(LinkAttachedSensorMixin, SimpleSensor[IMUOptions, IMUSensorArray]):
    def __init__(self, options: IMUOptions, array: IMUSensorArray):
        # FIXME: Resolution should be made private in mixin, so that it cannot be set by the user directly.
        options.resolution = options.acc_resolution + options.gyro_resolution + options.mag_resolution
        options.bias = options.acc_bias + options.gyro_bias + options.mag_bias
        options.random_walk = options.acc_random_walk + options.gyro_random_walk + options.mag_random_walk
        options.noise = options.acc_noise + options.gyro_noise + options.mag_noise

        super().__init__(options, array)

    @gs.assert_built
    def set_acc_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        self._array.set_cross_axis_coupling(
            self._idx, i_triplet=0, cross_axis_coupling=cross_axis_coupling, envs_idx=envs_idx
        )

    @gs.assert_built
    def set_gyro_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        self._array.set_cross_axis_coupling(
            self._idx, i_triplet=1, cross_axis_coupling=cross_axis_coupling, envs_idx=envs_idx
        )

    @gs.assert_built
    def set_mag_cross_axis_coupling(self, cross_axis_coupling: CrossCouplingAxisType, envs_idx=None):
        self._array.set_cross_axis_coupling(
            self._idx, i_triplet=2, cross_axis_coupling=cross_axis_coupling, envs_idx=envs_idx
        )
