import torch

from genesis.options.sensors import DepthCamera as DepthCameraOptions

from .base_sensor import Sensor
from .raycaster import RaycasterReturnType, RaycasterSensor, RaycasterSensorArray


class DepthCameraSensor(RaycasterSensor, Sensor[DepthCameraOptions, RaycasterSensorArray]):
    """Depth camera: a raycaster whose pattern is an image grid, read as a depth image."""

    @property
    def _image_shape(self) -> tuple[int, ...]:
        sim = self._array._sim
        batch_shape = (sim._B,) if sim.n_envs > 0 else ()
        return (*batch_shape, self._options.pattern.height, self._options.pattern.width)

    def read_image(self) -> torch.Tensor:
        """
        Read the depth image from the sensor.

        Returns
        -------
        torch.Tensor
            The depth image, of shape ([n_envs,] height, width).
        """
        return self.read().distances.reshape(*self._image_shape)
