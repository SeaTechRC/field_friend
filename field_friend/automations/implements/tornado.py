
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import rosys

from ..puncher import PuncherException
from .weeding_implement import ImplementException, WeedingImplement

if TYPE_CHECKING:
    from system import System


class Tornado(WeedingImplement):

    def __init__(self, system: 'System') -> None:
        super().__init__('Tornado', system)
        self.drill_with_open_tornado: bool = False
        self.drill_between_crops: bool = False
        self.with_punch_check: bool = False

    async def _perform_workflow(self) -> None:
        self.log.info('Performing Tornado Workflow..')
        try:
            closest_crop_id, closest_crop_position = list(self.crops_to_handle.items())[0]
            target_world_position = self.system.odometer.prediction.transform(closest_crop_position)
            self.log.info(f'Closest crop position: relative={closest_crop_position} world={target_world_position}')
            # fist check if the closest crop is in the working area
            if closest_crop_position.x < self.system.field_friend.WORK_X + self.WORKING_DISTANCE:
                self.log.info(f'target next crop at {closest_crop_position}')
                # do not steer while advancing on a crop

                if self.system.field_friend.can_reach(closest_crop_position) \
                        and not self._crops_in_drill_range(closest_crop_id, closest_crop_position, self.tornado_angle):
                    self.log.info('drilling crop')
                    open_drill = False
                    if self.drill_with_open_tornado and not self._crops_in_drill_range(closest_crop_id, closest_crop_position, 0):
                        open_drill = True
                    await self.system.puncher.drive_and_punch(plant_id=closest_crop_id,
                                                              x=closest_crop_position.x,
                                                              y=closest_crop_position.y,
                                                              angle=self.tornado_angle,
                                                              with_open_tornado=open_drill,
                                                              with_punch_check=self.with_punch_check)
                    # TODO remove weeds from plant_provider and increment kpis (like in Weeding Screw)
                    if isinstance(self.system.detector, rosys.vision.DetectorSimulation):
                        # remove the simulated weeds
                        inner_radius = 0.025  # TODO compute inner radius according to tornado angle
                        outer_radius = inner_radius + 0.05  # TODO compute outer radius according to inner radius and knife width
                        self.system.detector.simulated_objects = [obj for obj in self.system.detector.simulated_objects
                                                                  if not (inner_radius <= obj.position.projection().distance(target_world_position) <= outer_radius)]
                    # if self.drill_with_open_tornado and not self._crops_in_drill_range(closest_crop_id, closest_crop_position, 0):
                    #     self.log.info('drilling crop with open tornado')
                    #     await self.system.puncher.punch(plant_id=closest_crop_id, y=closest_crop_position.y, angle=0)
                self.log.info(f'crops to handle: {self.crops_to_handle}')
                if len(self.crops_to_handle) > 1 and self.drill_between_crops:
                    self.log.info('checking for second closest crop')
                    second_closest_crop_position = list(self.crops_to_handle.values())[1]
                    distance_to_next_crop = closest_crop_position.distance(second_closest_crop_position)
                    if distance_to_next_crop > 0.13:
                        # get the target of half the distance between the two crops
                        target = closest_crop_position.x + distance_to_next_crop / 2
                        self.log.info(f'driving to position between two crops: {target}')
                        await self.system.puncher.drive_and_punch(plant_id=closest_crop_id, x=target, y=0, angle=180)
            await self._driving_a_bit_forward()  # TODO is this necessary? It would be better to only let the navigation drive
            self.log.info('workflow completed')
        except PuncherException as e:
            self.log.error(f'Error while Tornado Workflow: {e}')
        except Exception as e:
            raise ImplementException(f'Error while tornado Workflow: {e}') from e

    def backup(self) -> dict:
        return super().backup() | {
            'drill_with_open_tornado': self.drill_with_open_tornado,
            'drill_between_crops': self.drill_between_crops,
            'with_punch_check': self.with_punch_check,
        }

    def restore(self, data: dict[str, Any]) -> None:
        super().restore(data)
        self.drill_with_open_tornado = data.get('drill_with_open_tornado', self.drill_with_open_tornado)
        self.drill_between_crops = data.get('drill_between_crops', self.drill_between_crops)
        self.with_punch_check = data.get('with_punch_check', self.with_punch_check)

    def _has_plants_to_handle(self) -> bool:
        super()._has_plants_to_handle()
        return any(self.crops_to_handle)
