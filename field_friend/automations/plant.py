from collections import deque
from dataclasses import dataclass, field
import numpy as np
from uuid import uuid4

from rosys.geometry import Point3d, Pose, Pose3d
from rosys.vision import Image

@dataclass(slots=True, kw_only=True)
class Plant:
    id: str = field(default_factory=lambda: str(uuid4()))
    type: str
    positions: deque[Point3d] = field(default_factory=lambda: [])
    camera_poses: deque[Pose3d] = field(default_factory=lambda: [])
    detection_time: float
    confidences: deque[float] = field(default_factory=lambda: [])
    detection_image: Image | None = None

    @property
    def position_rem_parallax(self) -> Point3d:
        if len(self.positions) == 1:
            return self.positions[0]

        #https://stackoverflow.com/questions/48154210/3d-point-closest-to-multiple-lines-in-3d-space
        cps = [cam_pose.translation_vector for cam_pose in self.camera_poses] # Camera points
        pps = [plant_point.array for plant_point in self.positions] # Plant points

        # tuple(start, direction_unit_vector)
        lines = [(cp.reshape(1, 3), ((pp - cp) / np.linalg.norm(pp-cp)).reshape(1, 3)) for cp, pp in zip(cps, pps)]
        M = np.zeros((3, 3))
        b = np.zeros(3).reshape(1, 3)
        for a, d in lines:
            d2 = np.dot(d.reshape(3), d.reshape(3))
            da = np.dot(d.reshape(3), a.reshape(3))

            am = d.reshape(3, 1) @ d.reshape(1, 3)
            for i in range(3):
                am[i, i] -= d2
            M += am
            b += (d * da) - (a * d2)
        res = np.linalg.pinv(M) @ b.reshape(3)
        if abs(res[2]) > 0.2: # Pretty much impossible, return a different position
            total_x = sum(point3d.x for point3d in self.positions)
            total_y = sum(point3d.y for point3d in self.positions)
            total_z = sum(point3d.z for point3d in self.positions)

            middle_x = total_x / len(self.positions)
            middle_y = total_y / len(self.positions)
            middle_z = total_z / len(self.positions)

            return Point3d(x=middle_x, y=middle_y, z=middle_z)
        return Point3d.from_tuple(res)

    @property
    def position(self) -> Point3d:
        """Calculate the middle position of all points"""
        total_x = sum(point3d.x for point3d in self.positions)
        total_y = sum(point3d.y for point3d in self.positions)
        total_z = sum(point3d.z for point3d in self.positions)

        middle_x = total_x / len(self.positions)
        middle_y = total_y / len(self.positions)
        middle_z = total_z / len(self.positions)

        return Point3d(x=middle_x, y=middle_y, z=middle_z)

    @property
    def confidence(self) -> float:
        # TODO: maybe use weighted confidence
        # sum_confidence = sum(confidence**1.5 for confidence in self.confidences)
        sum_confidence = sum(self.confidences)
        return sum_confidence
