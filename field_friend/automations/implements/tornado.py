
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import rosys
from nicegui import ui
from rosys.analysis import track
from rosys.geometry import Point, Point3d

from .weeding_implement import ImplementException, WeedingImplement

if TYPE_CHECKING:
    from ...system import System


class Tornado(WeedingImplement):
    def __init__(self, system: System) -> None:
        super().__init__('Tornado', system)
        self.tornado_angle: float = 30.0
        self.tornado_depth: float = 0
        self.tornado_turns: int = 2
        self.drill_with_open_tornado: bool = False
        self.drill_between_crops: bool = False
        self.skip_if_no_weeds: bool = False
        self.field_friend = system.field_friend

    async def start_workflow(self) -> None:
        await super().start_workflow()
        try:
            # TODO: do we need to set self.next_crop_id = '' on every return?
            punch_position = self.system.robot_locator.pose.transform(
                rosys.geometry.Point(x=self.system.field_friend.WORK_X, y=self.next_punch_y_position))
            self.last_punches.append(Point3d.from_point(punch_position))
            self.log.debug(f'Drilling crop at {punch_position} with angle {self.tornado_angle}°')
            open_drill = False
            if self.drill_with_open_tornado:
                open_drill = True
            await self.system.puncher.punch(y=self.next_punch_y_position, angle=self.tornado_angle, depth=self.tornado_depth, turns=self.tornado_turns, with_open_tornado=open_drill)
            # TODO remove weeds from plant_provider
            if isinstance(self.system.detector, rosys.vision.DetectorSimulation):
                # remove the simulated weeds
                inner_radius = 0.025  # TODO compute inner radius according to tornado angle
                outer_radius = inner_radius + 0.05  # TODO compute outer radius according to inner radius and knife width
                # inner_diameter, outer_diameter = self.system.field_friend.tornado_diameters(self.tornado_angle)
                # inner_radius = inner_diameter / 2
                # outer_radius = outer_diameter / 2
                self.system.detector.simulated_objects = [obj for obj in self.system.detector.simulated_objects
                                                          if not inner_radius <= obj.position.projection().distance(punch_position) <= outer_radius]
                self.log.debug(f'simulated_objects2: {len(self.system.detector.simulated_objects)}')
        except Exception as e:
            raise ImplementException('Error while tornado Workflow') from e

    def _has_plants_to_handle(self) -> bool:
        super()._has_plants_to_handle()
        if len(self.crops_to_handle) == 0:
            return False
        return True

    @track
    async def get_move_target(self) -> Point | None:
        """Return the target position to drive to."""
        self._has_plants_to_handle()
        if len(self.crops_to_handle) == 0:
            self.log.debug('No crops to handle')
            return None
        closest_crop_id, closest_crop_position = next(iter(self.crops_to_handle.items()))
        closest_crop_world_position = self.system.robot_locator.pose.transform3d(closest_crop_position)
        if any(p.distance(closest_crop_world_position) < self.field_friend.DRILL_RADIUS for p in self.last_punches):
            self.log.debug('Skipping crop because it was already punched')
            return None
        if not self.system.field_friend.can_reach(closest_crop_position.projection()):
            self.log.debug('Target crop is not in the working area')
            return None
        if self._crops_in_drill_range(closest_crop_id, closest_crop_position.projection(), self.tornado_angle):
            self.log.debug('Crops in drill range')
            return None
        tornado_outer_diameter = self.field_friend.tornado_diameters(self.tornado_angle)[1]
        if self.skip_if_no_weeds and \
            not any(closest_crop_position.distance(weed_position) < tornado_outer_diameter for weed_id, weed_position in self.weeds_to_handle.items()):
            
            self.log.info('Skipping crop because there are no weeds in range next to it.')
            return None

        relative_x = closest_crop_position.x - self.system.field_friend.WORK_X
        if relative_x < - self.system.field_friend.DRILL_RADIUS:
            self.log.debug(f'Skipping crop {closest_crop_id} because it is behind the robot')
            return None
        self.log.debug(f'Targeting crop {closest_crop_id} which is {relative_x} away at world: '
                           f'{closest_crop_world_position}, local: {closest_crop_position}')
        self.next_punch_y_position = closest_crop_position.y
        return closest_crop_world_position.projection()

    def _crops_in_drill_range(self, crop_id: str, crop_position: rosys.geometry.Point, angle: float) -> bool:
        inner_diameter, outer_diameter = self.system.field_friend.tornado_diameters(angle)
        crop_world_position = self.system.robot_locator.pose.transform(crop_position)
        for crop in self.system.plant_provider.crops:
            if crop.id != crop_id:
                distance = crop_world_position.distance(crop.position.projection())
                if inner_diameter/2 <= distance <= outer_diameter/2:
                    return True
        return False

    def backup_to_dict(self) -> dict[str, Any]:
        return super().backup_to_dict() | {
            'drill_with_open_tornado': self.drill_with_open_tornado,
            'drill_between_crops': self.drill_between_crops,
            'tornado_angle': self.tornado_angle,
            'tornado_depth': self.tornado_depth,
            'tornado_turns': self.tornado_turns,
            'skip_if_no_weeds': self.skip_if_no_weeds,
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        super().restore_from_dict(data)
        self.drill_with_open_tornado = data.get('drill_with_open_tornado', self.drill_with_open_tornado)
        self.drill_between_crops = data.get('drill_between_crops', self.drill_between_crops)
        self.tornado_angle = data.get('tornado_angle', self.tornado_angle)
        self.tornado_depth = data.get('tornado_depth', self.tornado_depth)
        self.tornado_turns = data.get('tornado_turns', self.tornado_turns)
        self.skip_if_no_weeds = data.get('skip_if_no_weeds', self.skip_if_no_weeds)

    def settings_ui(self):
        super().settings_ui()
        ui.number('Tornado angle', format='%.0f', step=1, min=0, max=180, on_change=self.request_backup) \
            .props('dense outlined suffix=°') \
            .classes('w-24') \
            .bind_value(self, 'tornado_angle') \
            .tooltip('Set the angle for the tornado drill')
        ui.number('Tornado depth offset', format='%.3f', step=0.001, min=0, max=0.02, on_change=self.request_backup) \
            .props('dense outlined suffix=m') \
            .classes('w-24') \
            .bind_value(self, 'tornado_depth') \
            .tooltip('Move the Tornado up by the given meters.')
        ui.number('Tornado turns', format='%.0f', step=1, min=1, max=10, on_change=self.request_backup) \
            .props('dense outlined suffix=turns') \
            .classes('w-24') \
            .bind_value(self, 'tornado_turns') \
            .tooltip('Amount of turns for the Tornado to do.')
        ui.label().bind_text_from(self, 'tornado_angle', lambda v: f'Tornado diameters: {self.field_friend.tornado_diameters(v)[0]*100:.1f} cm '
                                  f'- {self.field_friend.tornado_diameters(v)[1]*100:.1f} cm')
        ui.checkbox('Skip crop if no weeds are nearby') \
            .bind_value(self, 'skip_if_no_weeds') \
            .tooltip('Skip the crop if no weeds are in the working area of the Tornado.')

        # TODO test and reactivate these options
        # ui.checkbox('Drill 2x with open tornado') \
        #     .bind_value(self, 'drill_with_open_tornado') \
        #     .tooltip('Set the weeding automation to drill a second time with open tornado')
        # ui.checkbox('Drill between crops') \
        #     .bind_value(self, 'drill_between_crops') \
        #     .tooltip('Set the weeding automation to drill between crops')
