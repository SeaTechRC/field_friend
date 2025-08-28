from typing import Any, Self

import numpy as np
import rosys
from rosys import persistence
from rosys.geometry import Pose3d
from rosys.vision import Image
from rosys.vision.image_processing import get_image_size_from_bytes, process_jpeg_image, process_ndarray_image

class CalibratableUsbCamera(rosys.vision.CalibratableCamera, rosys.vision.UsbCamera):

    def __init__(self,
                 *,
                 calibration: rosys.vision.Calibration | None = None,
                 focal_length: float | None = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.focal_length = focal_length
        self.calibration = calibration

    # TODO: Remove the need to overwrite this function?
    async def _handle_new_image_data(self, image_data: np.ndarray | bytes, timestamp: float) -> None:
        if not self.is_connected:
            return None

        assert self.device is not None

        pose = self.calibration.extrinsics.resolve()

        bytes_: bytes | None
        if isinstance(image_data, np.ndarray):
            bytes_ = await rosys.run.cpu_bound(process_ndarray_image, image_data, self.rotation, self.crop)
        else:
            bytes_ = image_data
            if self.crop or self.rotation != ImageRotation.NONE:
                bytes_ = await rosys.run.cpu_bound(process_jpeg_image, bytes_, self.rotation, self.crop)
        if bytes_ is None:
            return

        final_image_resolution = get_image_size_from_bytes(bytes_)

        image = Image(time=timestamp, camera_id=self.id, size=final_image_resolution, data=bytes_, pose=pose)
        self._add_image(image)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if 'translation' in (data.get('calibration') or {}).get('extrinsics', {}):
            data['calibration']['extrinsics']['x'] = data['calibration']['extrinsics']['translation'][0]
            data['calibration']['extrinsics']['y'] = data['calibration']['extrinsics']['translation'][1]
            data['calibration']['extrinsics']['z'] = data['calibration']['extrinsics']['translation'][2]
            data['calibration']['extrinsics'].pop('translation')
        if 'rotation' in (data.get('calibration') or {}).get('intrinsics', {}):
            data['calibration']['extrinsics']['rotation']['R'] = (
                rosys.geometry.Rotation(R=data['calibration']['extrinsics']['rotation']['R']) *
                rosys.geometry.Rotation(R=data['calibration']['intrinsics']['rotation']['R'])
            ).R
        return cls(**(data | {
            'calibration': persistence.from_dict(rosys.vision.Calibration, data['calibration']) if data.get('calibration') else None,
        }))

    def to_dict(self) -> dict:
        return super().to_dict() | {'focal_length': self.focal_length} | {name: param.value for name, param in self._parameters.items()}
