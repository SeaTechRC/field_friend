import logging
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Optional

import rosys
from rosys.geometry import Point
from rosys.vision import Image


@dataclass(slots=True, kw_only=True)
class Plant:
    id: str = ...
    type: str
    positions: deque[Point]
    detection_time: float
    confidences: deque[float]
    detection_image: Optional[Image] = None

    def __init__(self, type: str, position: Point, detection_time: float, id: str = ..., confidence: float = 0.0, max_positions: int = 20, detection_image: Optional[Image] = None) -> None:
        self.id = id
        self.type = type
        self.detection_time = detection_time
        self.detection_image = detection_image
        self.positions = deque([position], maxlen=max_positions)
        self.confidences = deque([confidence], maxlen=max_positions)

    def __post_init__(self) -> None:
        """Generate a unique ID if not already loaded from persistence"""
        if self.id == ...:
            self.id = str(uuid.uuid4())

    @property
    def position(self) -> Point:
        """Calculate the middle position of all points"""
        sum_confidence = sum(self.confidences)
        x = 0.0
        y = 0.0
        for position, confidence in zip(self.positions, self.confidences):
            confidence_weight = confidence / sum_confidence
            x += position.x * confidence_weight
            y += position.y * confidence_weight
        return Point(x=x, y=y)

    @property
    def confidence(self) -> float:
        # TODO: maybe use weighted confidence
        # sum_confidence = sum(confidence**1.5 for confidence in self.confidences)
        sum_confidence = sum(self.confidences)
        return sum_confidence


def check_if_plant_exists(plant: Plant, plants: list[Plant], distance: float) -> bool:
    for p in plants:
        if p.position.distance(plant.position) < distance and p.type == plant.type:
            # Update the confidence
            p.confidences.append(plant.confidence)
            # Add the new position to the positions list
            p.positions.append(plant.position)
            p.detection_image = plant.detection_image
            return True
    return False


class PlantProvider:
    def __init__(self, match_distance: float = 0.07, crop_spacing: float = 0.18, prediction_confidence: float = 0.3) -> None:
        self.log = logging.getLogger('field_friend.plant_provider')
        self.weeds: list[Plant] = []
        self.crops: list[Plant] = []

        self.match_distance = match_distance
        self.crop_spacing = crop_spacing
        self.prediction_confidence = prediction_confidence

        self.PLANTS_CHANGED = rosys.event.Event()
        """The collection of plants has changed."""

        self.ADDED_NEW_WEED = rosys.event.Event()
        """A new weed has been added."""

        self.ADDED_NEW_CROP = rosys.event.Event()
        """A new crop has been added."""

        rosys.on_repeat(self.prune, 10.0)

    def prune(self) -> None:
        weeds_max_age = 10.0
        crops_max_age = 60.0 * 300.0
        self.weeds[:] = [weed for weed in self.weeds if weed.detection_time > rosys.time() - weeds_max_age]
        self.crops[:] = [crop for crop in self.crops if crop.detection_time > rosys.time() - crops_max_age]
        self.PLANTS_CHANGED.emit()

    def get_plant_by_id(self, plant_id: str) -> Plant:
        for plant in self.crops + self.weeds:
            if plant.id == plant_id:
                return plant
        raise ValueError(f'Plant with ID {plant_id} not found')

    async def add_weed(self, weed: Plant) -> None:
        if check_if_plant_exists(weed, self.weeds, 0.04):
            return
        self.weeds.append(weed)
        self.PLANTS_CHANGED.emit()
        self.ADDED_NEW_WEED.emit()

    def remove_weed(self, weed_id: str) -> None:
        self.weeds[:] = [weed for weed in self.weeds if weed.id != weed_id]
        self.PLANTS_CHANGED.emit()

    def clear_weeds(self) -> None:
        self.weeds.clear()
        self.PLANTS_CHANGED.emit()

    async def add_crop(self, crop: Plant) -> None:
        if check_if_plant_exists(crop, self.crops, self.match_distance):
            return
        self._add_crop_prediction(crop)
        self.crops.append(crop)
        self.PLANTS_CHANGED.emit()
        self.ADDED_NEW_CROP.emit()

    def remove_crop(self, crop: Plant) -> None:
        self.crops[:] = [c for c in self.crops if c.id != crop.id]
        self.PLANTS_CHANGED.emit()

    def clear_crops(self) -> None:
        self.crops.clear()
        self.PLANTS_CHANGED.emit()

    def clear(self) -> None:
        self.clear_weeds()
        self.clear_crops()

    def _add_crop_prediction(self, plant: Plant) -> None:
        sorted_crops = sorted(self.crops, key=lambda crop: crop.position.distance(plant.position))
        if len(sorted_crops) < 2:
            return
        crop_1 = sorted_crops[0]
        crop_2 = sorted_crops[1]

        yaw = crop_2.position.direction(crop_1.position)
        prediction = crop_1.position.polar(self.crop_spacing, yaw)

        if plant.position.distance(prediction) > self.match_distance:
            return
        plant.positions.append(prediction)
        plant.confidences.append(self.prediction_confidence)
