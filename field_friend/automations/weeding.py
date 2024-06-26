import logging
from copy import deepcopy
from random import randint
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import rosys
from rosys.driving import PathSegment
from rosys.geometry import Point, Pose, Spline
from rosys.helpers import eliminate_2pi

from ..hardware import ChainAxis
from . import Field, Row
from .plant import Plant
from .puncher import PuncherException
from .sequence import find_sequence

if TYPE_CHECKING:
    from system import System


class WorkflowException(Exception):
    pass


class Weeding(rosys.persistence.PersistentModule):
    WORKING_DISTANCE = 0.15
    DRIVE_DISTANCE = 0.03

    def __init__(self, system: 'System', persistence_key: str = 'weeding') -> None:
        super().__init__(persistence_key=f'field_friend.automations.{persistence_key}')
        self.PATH_PLANNED = rosys.event.Event()
        '''Event that is emitted when the path is planed. The event contains the path as a list of PathSegments.'''

        self.log = logging.getLogger('field_friend.weeding')
        self.system = system
        self.kpi_provider = system.kpi_provider

        # general settings
        self.continue_canceled_weeding: bool = False
        self.use_monitor_workflow: bool = False

        # field settings
        self.use_field_planning = True
        self.field: Optional[Field] = None
        self.start_row_id: Optional[str] = None
        self.end_row_id: Optional[str] = None
        self.minimum_turning_radius: float = 1.8
        self.turn_offset: float = 1.0
        self.drive_backwards_to_start: bool = True
        self.drive_to_start: bool = True

        # workflow settings
        self.only_monitoring: bool = False
        self.crop_confidence_threshold: float = 0.8
        self.weed_confidence_threshold: float = 0.8
        # tornado
        self.with_punch_check: bool = False
        self.drill_with_open_tornado: bool = False
        self.drill_between_crops: bool = False
        # dual mechanism
        self.with_drilling: bool = False
        self.with_chopping: bool = False
        self.chop_if_no_crops: bool = False

        # tool settings
        self.tornado_angle: float = 30.0
        self.weed_screw_depth: float = 0.13
        self.crop_safety_distance: float = 0.01

        # driver settings
        self.linear_speed_on_row: float = 0.04
        self.angular_speed_on_row: float = 0.3
        self.linear_speed_between_rows: float = 0.3
        self.angular_speed_between_rows: float = 0.8

        self.state: str = 'idle'
        self.start_time: Optional[float] = None
        self.last_pose: Optional[Pose] = None
        self.drived_distance: float = 0.0
        self.sorted_weeding_rows: list = []
        self.weeding_plan: list[list[PathSegment]] = []
        self.turn_paths: list[list[PathSegment]] = []
        self.current_row: Optional[Row] = None
        self.current_segment: Optional[PathSegment] = None
        self.row_segment_completed: bool = False
        self.crops_to_handle: dict[str, Point] = {}
        self.weeds_to_handle: dict[str, Point] = {}

        rosys.on_repeat(self._update_time_and_distance, 0.1)
        self.system.field_provider.FIELD_SELECTED.register(self.clear)

    def _update_time_and_distance(self):
        if self.state == 'idle':
            return
        if self.start_time is None:
            self.start_time = rosys.time()
        if self.last_pose is None:
            self.last_pose = self.system.odometer.prediction
            self.drived_distance = 0.0
        self.drived_distance += self.system.odometer.prediction.distance(self.last_pose)
        if self.drived_distance > 1:
            self.kpi_provider.increment_weeding_kpi('distance')
            self.drived_distance -= 1
        self.last_pose = self.system.odometer.prediction
        passed_time = rosys.time() - self.start_time
        if passed_time > 1:
            self.kpi_provider.increment_weeding_kpi('time')
            self.start_time = rosys.time()

    def backup(self) -> dict:
        data = {
            'use_field_planning': self.use_field_planning,
            'start_row_id': self.start_row_id,
            'end_row_id': self.end_row_id,
            'minimum_turning_radius': self.minimum_turning_radius,
            'turn_offset': self.turn_offset,
            'drive_backwards_to_start': self.drive_backwards_to_start,
            'drive_to_start': self.drive_to_start,
            'only_monitoring': self.only_monitoring,
            'crop_confidence_threshold': self.crop_confidence_threshold,
            'weed_confidence_threshold': self.weed_confidence_threshold,
            'with_punch_check': self.with_punch_check,
            'drill_with_open_tornado': self.drill_with_open_tornado,
            'drill_between_crops': self.drill_between_crops,
            'with_drilling': self.with_drilling,
            'with_chopping': self.with_chopping,
            'chop_if_no_crops': self.chop_if_no_crops,
            'tornado_angle': self.tornado_angle,
            'weed_screw_depth': self.weed_screw_depth,
            'crop_safety_distance': self.crop_safety_distance,
            'linear_speed_on_row': self.linear_speed_on_row,
            'angular_speed_on_row': self.angular_speed_on_row,
            'linear_speed_between_rows': self.linear_speed_between_rows,
            'angular_speed_between_rows': self.angular_speed_between_rows,
            'sorted_weeding_rows': [rosys.persistence.to_dict(row) for row in self.sorted_weeding_rows],
            'field': rosys.persistence.to_dict(self.field) if self.field else None,
            'weeding_plan': [[rosys.persistence.to_dict(segment) for segment in row] for row in self.weeding_plan],
            'turn_paths': [rosys.persistence.to_dict(segment) for segment in self.turn_paths],
            'current_row': rosys.persistence.to_dict(self.current_row) if self.current_row else None,
            'current_segment': rosys.persistence.to_dict(self.current_segment) if self.current_segment else None,
        }
        return data

    def restore(self, data: dict[str, Any]) -> None:
        self.use_field_planning = data.get('use_field_planning', self.use_field_planning)
        self.start_row_id = data.get('start_row_id', self.start_row_id)
        self.end_row_id = data.get('end_row_id', self.end_row_id)
        self.minimum_turning_radius = data.get('minimum_turning_radius', self.minimum_turning_radius)
        self.turn_offset = data.get('turn_offset', self.turn_offset)
        self.drive_backwards_to_start = data.get('drive_backwards_to_start', self.drive_backwards_to_start)
        self.drive_to_start = data.get('drive_to_start', self.drive_to_start)
        self.only_monitoring = data.get('only_monitoring', self.only_monitoring)
        self.crop_confidence_threshold = data.get('crop_confidence_threshold', self.crop_confidence_threshold)
        self.weed_confidence_threshold = data.get('weed_confidence_threshold', self.weed_confidence_threshold)
        self.with_punch_check = data.get('with_punch_check', self.with_punch_check)
        self.drill_with_open_tornado = data.get('drill_with_open_tornado', self.drill_with_open_tornado)
        self.drill_between_crops = data.get('drill_between_crops', self.drill_between_crops)
        self.with_drilling = data.get('with_drilling', self.with_drilling)
        self.with_chopping = data.get('with_chopping', self.with_chopping)
        self.chop_if_no_crops = data.get('chop_if_no_crops', self.chop_if_no_crops)
        self.tornado_angle = data.get('tornado_angle', self.tornado_angle)
        self.weed_screw_depth = data.get('weed_screw_depth', self.weed_screw_depth)
        self.crop_safety_distance = data.get('crop_safety_distance', self.crop_safety_distance)
        self.linear_speed_on_row = data.get('linear_speed_on_row', self.linear_speed_on_row)
        self.angular_speed_on_row = data.get('angular_speed_on_row', self.angular_speed_on_row)
        self.linear_speed_between_rows = data.get('linear_speed_between_rows', self.linear_speed_between_rows)
        self.angular_speed_between_rows = data.get('angular_speed_between_rows', self.angular_speed_between_rows)
        self.sorted_weeding_rows = [rosys.persistence.from_dict(Row, row_data)
                                    for row_data in data['sorted_weeding_rows']]
        self.field = rosys.persistence.from_dict(Field, data['field']) if data['field'] else None
        self.weeding_plan = [[rosys.persistence.from_dict(PathSegment, segment_data)
                              for segment_data in row_data] for row_data in data.get('weeding_plan', [])]
        self.turn_paths = [rosys.persistence.from_dict(PathSegment, segment_data)
                           for segment_data in data.get('turn_paths', [])]
        self.current_row = rosys.persistence.from_dict(Row, data['current_row']) if data['current_row'] else None
        self.current_segment = rosys.persistence.from_dict(PathSegment, data['current_segment']) \
            if data['current_segment'] else None

    def invalidate(self) -> None:
        self.log.info('backing up...')
        self.request_backup()

    def clear(self) -> None:
        self.field = None
        self.start_row_id = None
        self.end_row_id = None
        self.sorted_weeding_rows = []
        self.weeding_plan = []
        self.turn_paths = []
        self.current_row = None
        self.current_segment = None
        self.row_segment_completed = False
        self.crops_to_handle = {}
        self.weeds_to_handle = {}
        self.PATH_PLANNED.emit()

    async def start(self):
        self.log.info('starting weeding...')
        self.invalidate()
        if not await self._check_hardware_ready():
            return
        if not self.continue_canceled_weeding:
            if self.use_field_planning and not await self._field_planning():
                rosys.notify('Field planning failed', 'negative')
                return
        await self._weeding()

    async def _check_hardware_ready(self) -> bool:
        if self.system.field_friend.estop.active or self.system.field_friend.estop.is_soft_estop_active:
            rosys.notify('E-Stop is active, aborting', 'negative')
            self.log.error('E-Stop is active, aborting')
            return False
        camera = next((camera for camera in self.system.usb_camera_provider.cameras.values() if camera.is_connected), None)
        if not camera:
            rosys.notify('no camera connected')
            return False
        if camera.calibration is None:
            rosys.notify('camera has no calibration')
            return False
        if self.use_monitor_workflow:
            return True
        if self.system.field_friend.tool == 'none':
            rosys.notify('This field friend has no tool, only monitoring', 'info')
            self.log.info('This field friend has no tool, only monitoring')
            return True
        if self.system.field_friend.y_axis.alarm:
            rosys.notify('Y-Axis is in alarm, aborting', 'negative')
            self.log.error('Y-Axis is in alarm, aborting')
            return False
        if isinstance(self.system.field_friend.y_axis, ChainAxis):
            if not self.system.field_friend.y_axis.ref_t:
                rosys.notify('ChainAxis is not in top ref', 'negative')
                self.log.error('ChainAxis is not in top ref')
                return False
        if not await self.system.puncher.try_home():
            rosys.notify('Puncher homing failed, aborting', 'negative')
            self.log.error('Puncher homing failed, aborting')
            return False
        return True

    async def _field_planning(self) -> bool:
        if self.system.gnss.device is None:
            self.log.error('GNSS is not available')
            return False
        self.field = self.system.field_provider.active_field
        if self.field is None:
            self.log.error('Field is not available')
            rosys.notify('No field selected', 'negative')
            return False
        if not self.field.reference:
            self.log.error('Field reference is not available')
            return False
        self.system.gnss.reference = self.field.reference
        self.weeding_plan = self._make_plan()
        if not self.weeding_plan:
            self.log.error('No plan available')
            return False
        self.turn_paths = await self._generate_turn_paths()
        if not self.turn_paths:
            self.log.error('No turn paths available')
        self.PATH_PLANNED.emit()
        return True

    def _make_plan(self) -> list[list[rosys.driving.PathSegment]]:
        self.log.info('Making plan...')
        if self.field is None:
            self.log.warning('No field available')
            return []
        if not self.field.rows:
            self.log.warning('No rows available')
            return []
        if self.start_row_id is None:
            self.start_row_id = self.field.rows[0].id
        else:
            self.start_row_id = next((row.id for row in self.field.rows if row.id == self.start_row_id), None)
        if self.end_row_id is None:
            self.end_row_id = self.field.rows[-1].id
        else:
            self.end_row_id = next((row.id for row in self.field.rows if row.id == self.end_row_id), None)

        start_row = next((row for row in self.field.rows if row.id == self.start_row_id), None)
        end_row = next((row for row in self.field.rows if row.id == self.end_row_id), None)
        if start_row is None:
            start_row = self.field.rows[0]
        if end_row is None:
            end_row = self.field.rows[-1]
        self.start_row_id = start_row.id
        self.end_row_id = end_row.id
        reference = self.field.reference
        assert reference is not None
        rows_to_weed = self.field.rows[self.field.rows.index(
            start_row):self.field.rows.index(end_row) + 1]
        rows = [row for row in rows_to_weed if len(row.cartesian(reference)) > 1]
        robot_position = self.system.odometer.prediction.point
        distance_to_first_row = min([point.distance(robot_position) for point in rows[0].cartesian(reference)])
        distance_to_last_row = min([point.distance(robot_position) for point in rows[-1].cartesian(reference)])
        if distance_to_first_row > distance_to_last_row:
            rows = list(reversed(rows))
        minimum_row_distance = 1  # 1 = no row needs to be skipped when turning
        if len(rows) > 1:
            rows_distance = rows[0].cartesian(reference)[0].distance(rows[1].cartesian(reference)[0])
            self.log.info(f'Rows distance: {rows_distance}')
            self.log.info(f'Minimum turning radius: {self.minimum_turning_radius}')
            if self.minimum_turning_radius * 2 > rows_distance:
                self.log.info('Rows distance is smaller than minimum turning radius * 2')
                minimum_row_distance = int(
                    np.ceil(self.minimum_turning_radius * 2 / rows_distance))

        self.log.info(f'Minimum row distance: {minimum_row_distance} need to skip {minimum_row_distance - 1} rows')
        if minimum_row_distance > 1:
            sequence = find_sequence(len(rows), minimum_distance=minimum_row_distance)
            if not sequence:
                self.log.warning('No sequence found')
                sequence = list(range(len(rows)))
        else:
            sequence = list(range(len(rows)))
        self.log.info(f'Row sequence: {sequence}')

        paths = []
        switch_first_row = False
        for i, row_index in enumerate(sequence):
            splines = []
            row = rows[row_index]
            self.sorted_weeding_rows.append(row)
            row_points = row.cartesian(reference).copy()
            if i == 0:
                switch_first_row = robot_position.distance(row_points[0]) > robot_position.distance(row_points[-1])
                self.log.info(f'Switch first row: {switch_first_row}')
            if not switch_first_row:
                if i % 2 != 0:
                    row_points = list(reversed(row_points))
            else:
                if i % 2 == 0:
                    row_points = list(reversed(row_points))
            self.log.info(f'Row {row.name} has {row_points} points')
            # if row.crops:
            #     self.log.info(f'Row {row.name} has beets, creating {row.crops} points')
            #     # only take every tenth crop into account
            #     for i, beet in enumerate(row.crops):
            #         if i % 10 == 0 and i != 0:
            #             row_points.append(beet.position)
            #     row_points = sorted(row_points, key=lambda point: point.distance(row_points[0]))
            #     self.log.info(f'Row {row.name} has {len(row_points)} points')
            for j in range(len(row_points) - 1):
                splines.append(Spline.from_points(row_points[j], row_points[j + 1]))
            path = [PathSegment(spline=spline) for spline in splines]
            paths.append(path)
        return paths

    async def _generate_turn_paths(self) -> list[list[PathSegment]]:
        self.log.info('Generating turn paths...')
        if not self.weeding_plan or not self.field:
            self.log.error('No weeding plan or field available')
            return []
        turn_paths = []
        self.system.path_planner.obstacles.clear()
        assert self.field.reference is not None
        for obstacle in self.field.obstacles:
            self.system.path_planner.obstacles[obstacle.id] = \
                rosys.pathplanning.Obstacle(id=obstacle.id, outline=obstacle.cartesian(self.field.reference))
        for row in self.field.rows:
            row_points = row.cartesian(self.field.reference)
            # create a small polygon around the row to avoid the robot driving through the row
            row_polygon = [
                Point(x=row_points[0].x, y=row_points[0].y),
                Point(x=row_points[0].x - 0.01, y=row_points[0].y + 0.01),
                Point(x=row_points[-1].x, y=row_points[-1].y),
            ]
            self.system.path_planner.obstacles[f'row_{row.id}'] = \
                rosys.pathplanning.Obstacle(id=f'row_{row.id}', outline=row_polygon)

        area = rosys.pathplanning.Area(id=f'{self.field.id}', outline=self.field.outline)
        self.system.path_planner.areas = {area.id: area}
        for i in range(len(self.weeding_plan) - 1):
            # remove the current and the rows from obstacles to allow starting in it an insert it afterwards again
            start_row = self.sorted_weeding_rows[i]
            end_row = self.sorted_weeding_rows[i + 1]
            temp_removed_start_row = self.system.path_planner.obstacles.pop(f'row_{start_row.id}')
            temp_removed_end_row = self.system.path_planner.obstacles.pop(f'row_{end_row.id}')
            start_point = Point(x=self.weeding_plan[i][-1].spline.end.x,
                                y=self.weeding_plan[i][-1].spline.end.y)
            yaw = self.weeding_plan[i][-1].spline.start.direction(self.weeding_plan[i][-1].spline.end)
            offset_start_point = start_point.polar(self.turn_offset, yaw)

            start_pose = Pose(x=offset_start_point.x, y=offset_start_point.y, yaw=yaw)
            end_point = Point(x=self.weeding_plan[i + 1][0].spline.start.x,
                              y=self.weeding_plan[i + 1][0].spline.start.y)
            end_yaw = self.weeding_plan[i + 1][0].spline.start.direction(self.weeding_plan[i + 1][0].spline.end)
            offset_end_point = end_point.polar(0.5, yaw)
            end_pose = Pose(x=offset_end_point.x, y=offset_end_point.y, yaw=end_yaw)
            self.log.info(f'Searching path from row {i} to row {i + 1}...')
            turn_path = await self.system.path_planner.search(start=start_pose, goal=end_pose, timeout=120)
            if turn_path:
                turn_paths.append(turn_path)
            else:
                self.log.error(f'No turn path found from row {i} to row {i + 1}')
                return []
            self.system.path_planner.obstacles[f'row_{start_row.id}'] = temp_removed_start_row
            self.system.path_planner.obstacles[f'row_{end_row.id}'] = temp_removed_end_row
        # # clear all row obstacles
        # for row in self.field.rows:
        #     self.system.path_planner.obstacles.pop(f'row_{row.id}')
        return turn_paths

    async def _weeding(self):
        self.log.info('Starting driving...')
        await rosys.sleep(0.5)
        self.state = 'running'
        try:
            if self.weeding_plan and self.use_field_planning:
                await self._weed_with_plan()
                self.log.info('Weeding with plan completed')
            else:
                await self._weed_planless()
                self.log.info('Planless weeding completed')

        except WorkflowException as e:
            self.kpi_provider.increment_weeding_kpi('automation_stopped')
            self.log.error(f'WorkflowException: {e}')
        finally:
            self.kpi_provider.increment_weeding_kpi('weeding_completed')
            await self.system.field_friend.stop()
            self.system.plant_locator.pause()
            self.system.automation_watcher.stop_field_watch()
            self.system.automation_watcher.gnss_watch_active = False

    async def _drive_to_start(self):
        self.log.info('Driving to start...')
        start_pose = self.system.odometer.prediction
        end_pose = Pose(x=self.weeding_plan[0][0].spline.start.x, y=self.weeding_plan[0][0].spline.start.y,
                        yaw=self.weeding_plan[0][0].spline.start.direction(self.weeding_plan[0][0].spline.end))
        start_spline = Spline.from_poses(start_pose, end_pose)
        await self.system.driver.drive_spline(start_spline)

    async def _weed_with_plan(self):
        if self.continue_canceled_weeding:
            start_pose = self.system.odometer.prediction
            end_pose = Pose(x=self.current_segment.spline.start.x, y=self.current_segment.spline.start.y,
                            yaw=self.current_segment.spline.start.direction(self.current_segment.spline.end))
            start_spline = Spline.from_poses(start_pose, end_pose)
            await self.system.driver.drive_spline(start_spline)
        elif self.drive_to_start:
            await self._drive_to_start()
        self.system.automation_watcher.start_field_watch(self.field.outline)
        self.system.automation_watcher.gnss_watch_active = True
        for i, path in enumerate(self.weeding_plan):
            if self.continue_canceled_weeding and self.current_row != self.sorted_weeding_rows[i]:
                continue
            self.system.driver.parameters.can_drive_backwards = False
            self.system.driver.parameters.linear_speed_limit = self.linear_speed_on_row
            self.system.driver.parameters.angular_speed_limit = self.angular_speed_on_row
            self.current_row = self.sorted_weeding_rows[i]
            self.system.plant_locator.pause()
            self.system.plant_provider.clear()
            if self.system.field_friend.tool != 'none':
                await self.system.puncher.clear_view()
            await self.system.field_friend.flashlight.turn_on()
            await rosys.sleep(3)
            self.system.plant_locator.resume()
            await rosys.sleep(3)
            for j, segment in enumerate(path):
                if self.continue_canceled_weeding and self.current_segment != segment:
                    continue
                else:
                    self.continue_canceled_weeding = False
                self.current_segment = segment
                # self.invalidate()
                if not self.system.is_real:
                    self.system.detector.simulated_objects = []
                    await self._create_simulated_plants()
                self.log.info(f'Driving row {i + 1}/{len(self.weeding_plan)} and segment {j + 1}/{len(path)}...')
                self.row_segment_completed = False
                while not self.row_segment_completed:
                    self.log.info('while not row completed...')
                    await rosys.automation.parallelize(
                        self._check_for_plants(),
                        self._drive_segment(),
                        return_when_first_completed=True
                    )
                    await self.system.field_friend.stop()
                    await rosys.sleep(2)  # wait for robot to stand still
                    await self._get_upcoming_plants()
                    if self.crops_to_handle or self.weeds_to_handle:
                        self.log.info('Plants to handle...')
                        await self._handle_plants()
                        self.crops_to_handle = {}
                        self.weeds_to_handle = {}
                        if self.system.odometer.prediction.relative_point(self.current_segment.spline.end).x < 0.01:
                            self.row_segment_completed = True
                    await rosys.sleep(0.2)
            if self.drive_backwards_to_start:
                self.log.info('Low battery, driving backwards to start...')
                rosys.notify('Low battery, driving backwards to start', 'warning')
                self.system.driver.parameters.can_drive_backwards = True
                await self.system.driver.drive_to(Point(x=self.weeding_plan[0][0].spline.start.x, y=self.weeding_plan[0][0].spline.start.y), backward=True)
                self.system.driver.parameters.can_drive_backwards = False
                return

            await self.system.field_friend.flashlight.turn_off()
            self.system.plant_locator.pause()
            if i < len(self.weeding_plan) - 1:
                self.system.driver.parameters.can_drive_backwards = True
                self.system.driver.parameters.linear_speed_limit = self.linear_speed_between_rows
                self.system.driver.parameters.angular_speed_limit = self.angular_speed_between_rows
                self.log.info('Driving to next row...')
                turn_path = self.turn_paths[i]
                await self.system.driver.drive_path(turn_path)
                await rosys.sleep(1)
            self.kpi_provider.increment_weeding_kpi('rows_weeded')

        self.system.automation_watcher.stop_field_watch()
        self.system.automation_watcher.gnss_watch_active = False
        self.sorted_weeding_rows = []
        self.weeding_plan = []
        self.turn_paths = []
        self.current_row = None
        self.current_segment = None

    async def _weed_planless(self):
        already_explored_count = 0
        while True:
            self.system.plant_locator.pause()
            self.system.plant_provider.clear()
            if not self.system.is_real:
                self.system.detector.simulated_objects = []
                await self._create_simulated_plants()
            if self.system.field_friend.tool != 'none':
                await self.system.puncher.clear_view()
            await self.system.field_friend.flashlight.turn_on()
            await rosys.sleep(2)
            self.system.plant_locator.resume()
            await rosys.sleep(0.5)
            await self._get_upcoming_plants()
            while self.crops_to_handle or self.weeds_to_handle:
                await self._handle_plants()
                already_explored_count = 0
                await rosys.sleep(0.2)
                await self._get_upcoming_plants()
            if not self.crops_to_handle and already_explored_count != 5:
                self.log.info('No crops found, advancing a bit to ensure there are really no more crops')
                target = self.system.odometer.prediction.transform(Point(x=0.10, y=0))
                await self.system.driver.drive_to(target)
                already_explored_count += 1
            else:
                self.log.info('No more crops found')
                break

    async def _drive_segment(self):
        self.log.info('Driving segment...')
        await self.system.driver.drive_spline(self.current_segment.spline)
        self.row_segment_completed = True

    async def _check_for_plants(self):
        self.log.info('Checking for plants...')
        while True:
            await self._get_upcoming_plants()
            if self.system.field_friend.tool in ['tornado', 'none'] or self.use_monitor_workflow:
                if self.crops_to_handle:
                    return
            elif self.system.field_friend.tool == 'dual_mechanism':
                if (self.with_drilling and not self.only_monitoring) and self.weeds_to_handle:
                    return
                elif self.crops_to_handle:
                    return
            else:
                if self.weeds_to_handle or self.crops_to_handle:
                    return
            await rosys.sleep(0.2)

    async def _get_upcoming_plants(self):
        # TODO: confidence parameter
        relative_crop_positions = {
            c.id: self.system.odometer.prediction.relative_point(c.position)
            for c in self.system.plant_provider.crops if c.position.distance(self.system.odometer.prediction.point) < 0.5 and c.confidence > self.crop_confidence_threshold
        }
        # remove very distant crops (probably not row
        if self.current_segment:
            # Correctly filter to get upcoming crops based on their x position
            upcoming_crop_positions = {
                c: pos for c, pos in relative_crop_positions.items()
                if self.system.field_friend.WORK_X+self.system.field_friend.DRILL_RADIUS < pos.x <= self.system.odometer.prediction.relative_point(self.current_segment.spline.end).x
            }
        else:
            upcoming_crop_positions = {
                c: pos for c, pos in relative_crop_positions.items()
                if self.system.field_friend.WORK_X < pos.x < 0.3
            }

        # Sort the upcoming_crop_positions dictionary by the .x attribute of its values
        sorted_crops = dict(sorted(upcoming_crop_positions.items(), key=lambda item: item[1].x))

        self.crops_to_handle = sorted_crops

        relative_weed_positions = {
            w.id: self.system.odometer.prediction.relative_point(w.position)
            for w in self.system.plant_provider.weeds if w.position.distance(self.system.odometer.prediction.point) < 0.5 and w.confidence > self.weed_confidence_threshold
        }
        if self.current_segment:
            # Filter to get upcoming weeds based on their .x position
            upcoming_weed_positions = {
                w: pos for w, pos in relative_weed_positions.items()
                if self.system.field_friend.WORK_X+self.system.field_friend.DRILL_RADIUS < pos.x <= self.system.odometer.prediction.relative_point(self.current_segment.spline.end).x
            }
        else:
            upcoming_weed_positions = {
                w: pos for w, pos in relative_weed_positions.items()
                if self.system.field_friend.WORK_X < pos.x < 0.4
            }

        # Sort the upcoming_weed_positions dictionary by the .x attribute of its values
        sorted_weeds = dict(sorted(upcoming_weed_positions.items(), key=lambda item: item[1].x))

        self.weeds_to_handle = sorted_weeds

    async def _handle_plants(self) -> None:
        self.log.info('Handling plants...')
        # for crop_id in self.crops_to_handle:
        #     self._safe_crop_to_row(crop_id)
        if self.system.field_friend.tool == 'tornado' and not self.use_monitor_workflow and self.crops_to_handle:
            await self._tornado_workflow()
        elif self.system.field_friend.tool == 'weed_screw' and not self.use_monitor_workflow:
            await self._weed_screw_workflow()
        elif self.system.field_friend.tool == 'dual_mechanism' and not self.use_monitor_workflow:
            if self.with_chopping:
                await self._dual_mechanism_workflow()
            else:
                await self._weed_screw_workflow()
        elif self.system.field_friend.tool == 'none' or self.use_monitor_workflow:
            await self._monitor_workflow()

    async def _tornado_workflow(self) -> None:
        self.log.info('Starting Tornado Workflow..')
        try:
            try:
                closest_crop_id, closest_crop_position = list(self.crops_to_handle.items())[0]
            except IndexError:
                self.log.warning('No crops to handle')
                return
            self.log.info(f'Closest crop position: {closest_crop_position}')
            # fist check if the closest crop is in the working area
            if closest_crop_position.x < self.system.field_friend.WORK_X + self.WORKING_DISTANCE:
                self.log.info(f'target next crop at {closest_crop_position}')
                # do not steer while advancing on a crop

                if not self.only_monitoring and self.system.field_friend.can_reach(closest_crop_position) \
                        and not self._crops_in_drill_range(closest_crop_id, closest_crop_position, self.tornado_angle):
                    self.log.info('drilling crop')
                    open_drill = False
                    if self.drill_with_open_tornado and not self._crops_in_drill_range(closest_crop_id, closest_crop_position, 0):
                        open_drill = True
                    await self.system.puncher.drive_and_punch(
                        plant_id=closest_crop_id, x=closest_crop_position.x, y=closest_crop_position.y, angle=self.tornado_angle, with_open_tornado=open_drill, with_punch_check=self.with_punch_check)
                    # if self.drill_with_open_tornado and not self._crops_in_drill_range(closest_crop_id, closest_crop_position, 0):
                    #     self.log.info('drilling crop with open tornado')
                    #     await self.system.puncher.punch(plant_id=closest_crop_id, y=closest_crop_position.y, angle=0)
                else:
                    self.log.info('Cant reach crop')
                    await self._follow_line_of_crops()

                if len(self.crops_to_handle) > 1 and self.drill_between_crops:
                    self.log.info('checking for second closest crop')
                    second_closest_crop_position = list(self.crops_to_handle.values())[1]
                    distance_to_next_crop = closest_crop_position.distance(second_closest_crop_position)
                    if distance_to_next_crop > 0.13:
                        # get the target of half the distance between the two crops
                        target = closest_crop_position.x + distance_to_next_crop / 2
                        self.log.info(f'driving to position between two crops: {target}')
                        if not self.only_monitoring:
                            # punch in the middle position with closed knifes
                            await self.system.puncher.drive_and_punch(plant_id=closest_crop_id, x=target, y=0, angle=180)
                        else:
                            drive_distance = target - self.system.field_friend.WORK_X
                            target = self.system.odometer.prediction.transform(Point(x=drive_distance, y=0))
                            await self.system.driver.drive_to(target)

            else:
                await self._follow_line_of_crops()
            await rosys.sleep(0.2)
            self.log.info('workflow completed')
        except PuncherException as e:
            self.log.error(f'Error while Tornado Workflow: {e}')
        except Exception as e:
            raise WorkflowException(f'Error while tornado Workflow: {e}') from e

    async def _monitor_workflow(self) -> None:
        self.log.info('Starting Monitoring Workflow...')
        try:
            closest_crop_position = list(self.crops_to_handle.values())[0]
            self.log.info(f'Closest crop position: {closest_crop_position}')
            # fist check if the closest crop is in the working area
            if closest_crop_position.x < self.WORKING_DISTANCE:
                self.log.info(f'target next crop at {closest_crop_position}')
                # do not steer while advancing on a crop
                drive_distance = closest_crop_position.x - self.system.field_friend.WORK_X
                target = self.system.odometer.prediction.transform(Point(x=drive_distance, y=0))
                await self.system.driver.drive_to(target)
                self.system.plant_locator.resume()
            else:
                if self.crops_to_handle:
                    await self._follow_line_of_crops()
                else:
                    await self._driving_a_bit_forward()
            await rosys.sleep(0.2)
            self.log.info('workflow completed')
        except PuncherException as e:
            self.log.error(f'Error while Monitoring Workflow: {e}')
        except Exception as e:
            raise WorkflowException(f'Error while Monitoring Workflow: {e}') from e

    async def _weed_screw_workflow(self) -> None:
        self.log.info('Starting Weed Screw Workflow...')
        try:
            starting_position = deepcopy(self.system.odometer.prediction)
            self._keep_crops_safe()
            weeds_in_range = {weed_id: position for weed_id, position in self.weeds_to_handle.items(
            ) if position.x < self.system.field_friend.WORK_X + self.WORKING_DISTANCE and self.system.field_friend.can_reach(position)}
            if weeds_in_range:
                self.log.info(f'Weeds in range {len(weeds_in_range)}')
                while weeds_in_range:
                    next_weed_id, next_weed_position = list(self.crops_to_handle.items())[0]
                    weed_world_position = starting_position.transform(next_weed_position)
                    corrected_relative_weed_position = self.system.odometer.prediction.relative_point(
                        weed_world_position)
                    self.log.info(f'Targeting weed at {next_weed_position}')
                    if not self.only_monitoring:
                        await self.system.puncher.drive_and_punch(plant_id=next_weed_id,
                                                                  x=corrected_relative_weed_position.x, y=corrected_relative_weed_position.y, depth=self.weed_screw_depth, backwards_allowed=False)
                    punched_weeds = [weed_id for weed_id, position in weeds_in_range.items(
                    ) if position.distance(next_weed_position) <= self.system.field_friend.DRILL_RADIUS]
                    for weed_id in punched_weeds:
                        self.system.plant_provider.remove_weed(weed_id)
                        if weed_id in weeds_in_range:
                            del weeds_in_range[weed_id]
                        self.kpi_provider.increment_weeding_kpi('weeds_removed')

            elif self.crops_to_handle:
                await self._follow_line_of_crops()
            else:
                await self._driving_a_bit_forward()
            await rosys.sleep(0.2)
            self.log.info('Workflow completed')
        except Exception as e:
            raise WorkflowException(f'Error while Weed Screw Workflow: {e}') from e

    async def _dual_mechanism_workflow(self) -> None:
        self.log.info('Starting dual mechanism workflow...')
        try:
            moved = False
            starting_position = deepcopy(self.system.odometer.prediction)
            if self.crops_to_handle:
                next_crop_position = list(self.crops_to_handle.values())[0]
                # first: check if weeds near crop
                if self.with_drilling:
                    self.log.info(f'Drilling allowed: only drilling is {self.with_drilling}')
                    self._keep_crops_safe()
                    weeds_in_range = {weed_id: position for weed_id, position in self.weeds_to_handle.items() if next_crop_position.x - self.system.field_friend.DRILL_RADIUS*2
                                      < position.x < next_crop_position.x + self.system.field_friend.DRILL_RADIUS*2 and self.system.field_friend.can_reach(position)}
                    self.log.info(f'weed_position in range: {weeds_in_range.items()}')
                    if weeds_in_range:
                        self.log.info(f' {len(weeds_in_range)} Weeds in range for drilling')
                        while weeds_in_range:
                            next_weed_id, next_weed_position = list(self.crops_to_handle.items())[0]
                            self.log.info(f'Next weed position: {next_weed_position}')
                            weed_world_position = starting_position.transform(next_weed_position)
                            corrected_relative_weed_position = self.system.odometer.prediction.relative_point(
                                weed_world_position)
                            self.log.info(f'corrected relative weed position: {corrected_relative_weed_position}')
                            moved = True
                            if not self.only_monitoring:
                                await self.system.puncher.drive_and_punch(plant_id=next_weed_id,
                                                                          x=corrected_relative_weed_position.x, y=next_weed_position.y, depth=self.weed_screw_depth, backwards_allowed=False)
                            punched_weeds = [weed_id for weed_id, position in weeds_in_range.items(
                            ) if position.distance(next_weed_position) < self.system.field_friend.DRILL_RADIUS]
                            for weed_id in punched_weeds:
                                self.system.plant_provider.remove_weed(weed_id)
                                if weed_id in weeds_in_range:
                                    del weeds_in_range[weed_id]
                                self.kpi_provider.increment_weeding_kpi('weeds_removed')
                        await self.system.puncher.clear_view()
                # second: check if weed before crop to chop
                if self.with_chopping:
                    self.log.info(f'Chopping allowed: only chopping is {self.with_chopping}')
                    weeds_in_range = {weed_id: position for weed_id, position in self.weeds_to_handle.items() if position.x < next_crop_position.x - (
                        self.system.field_friend.DRILL_RADIUS) and self.system.field_friend.can_reach(position, second_tool=True)}
                    if weeds_in_range:
                        self.log.info('Weeds in range for chopping before crop')
                        crop_world_position = starting_position.transform(next_crop_position)
                        corrected_relative_crop_position = self.system.odometer.prediction.relative_point(
                            crop_world_position)
                        target_position = corrected_relative_crop_position.x - \
                            self.system.field_friend.DRILL_RADIUS - self.system.field_friend.CHOP_RADIUS
                        axis_distance = target_position - self.system.field_friend.WORK_X_CHOP
                        if axis_distance >= 0:
                            local_target = Point(x=axis_distance, y=0)
                            world_target = self.system.driver.prediction.transform(local_target)
                            moved = True
                            await self.system.driver.drive_to(world_target)
                            if not self.only_monitoring:
                                await self.system.puncher.chop()
                            choped_weeds = [weed_id for weed_id, position in self.weeds_to_handle.items(
                            ) if target_position - self.system.field_friend.CHOP_RADIUS < self.system.odometer.prediction.relative_point(starting_position.transform(position)).x < target_position + self.system.field_friend.CHOP_RADIUS]
                            for weed_id in choped_weeds:
                                self.system.plant_provider.remove_weed(weed_id)
                                self.kpi_provider.increment_weeding_kpi('weeds_removed')
                        else:
                            self.log.warning(f'Weed position {next_weed_position} is behind field friend')
                    if not moved:
                        await self._follow_line_of_crops()
                        moved = True
            elif self.weeds_to_handle and self.chop_if_no_crops:
                self.log.info('No crops in range, checking for weeds...')
                weeds_in_range = {weed_id: position for weed_id, position in self.weeds_to_handle.items() if self.system.field_friend.WORK_X_CHOP <
                                  position.x < self.system.field_friend.WORK_X + self.WORKING_DISTANCE and self.system.field_friend.can_reach(position)}
                if weeds_in_range:
                    next_weed_position = list(weeds_in_range.values())[0]
                    axis_distance = next_weed_position.x - self.system.field_friend.WORK_X_CHOP + self.system.field_friend.CHOP_RADIUS
                    if axis_distance >= 0:
                        local_target = Point(x=axis_distance, y=0)
                        self.log.info(f'Next weed position: {next_weed_position}')
                        self.log.info(f'Axis distance: {axis_distance}')
                        world_target = self.system.driver.prediction.transform(local_target)
                        moved = True
                        await self.system.driver.drive_to(world_target)
                        if not self.only_monitoring:
                            await self.system.puncher.chop()
                        choped_weeds = [weed_id for weed_id, position in self.weeds_to_handle.items(
                        ) if axis_distance - self.system.field_friend.CHOP_RADIUS < self.system.odometer.prediction.relative_point(starting_position.transform(position)).x < axis_distance + self.system.field_friend.CHOP_RADIUS]
                        for weed_id in choped_weeds:
                            self.system.plant_provider.remove_weed(weed_id)
                            self.kpi_provider.increment_weeding_kpi('weeds_removed')
                    else:
                        self.log.warning(f'Weed position {next_weed_position} is behind field friend')
            if not moved:
                await self._driving_a_bit_forward()
            await rosys.sleep(0.2)
            self.log.info('Workflow completed')
        except PuncherException as e:
            self.log.error(f'Error while Dual Mechanism Workflow: {e}')
        except Exception as e:
            raise WorkflowException(f'Error while double mechanism Workflow: {e}') from e

    async def _follow_line_of_crops(self):
        self.log.info('Following line of crops...')
        farthest_crop = list(self.crops_to_handle.values())[-1]
        self.log.info(f'Farthest crop: {farthest_crop}')
        upcoming_world_position = self.system.odometer.prediction.transform(farthest_crop)
        yaw = self.system.odometer.prediction.point.direction(upcoming_world_position)
        # only apply minimal yaw corrections to avoid oversteering
        target_yaw = self._weighted_angle_combine(self.system.odometer.prediction.yaw, 0.85, yaw, 0.15)
        # yaw = eliminate_2pi(self.system.odometer.prediction.yaw) * 0.9 + eliminate_2pi(yaw) * 0.1
        target = self.system.odometer.prediction.point.polar(self.DRIVE_DISTANCE, target_yaw)
        self.log.info(f'Current world position: {self.system.odometer.prediction} Target next crop at {target}')
        await self.system.driver.drive_to(target)

    def _weighted_angle_combine(self, angle1: float, weight1: float, angle2: float, weight2: float) -> float:
        # Normalize both angles
        angle1 = eliminate_2pi(angle1)
        angle2 = eliminate_2pi(angle2)

        # Combine angles with the weights
        x = np.cos(angle1) * weight1 + np.cos(angle2) * weight2
        y = np.sin(angle1) * weight1 + np.sin(angle2) * weight2

        # Compute the resultant angle
        combined_angle = np.arctan2(y, x)

        # Normalize the resultant angle
        return eliminate_2pi(combined_angle)

    async def _driving_a_bit_forward(self):
        self.log.info('No crops and no weeds in range, driving forward a bit...')
        target = self.system.odometer.prediction.point.polar(self.DRIVE_DISTANCE, self.system.odometer.prediction.yaw)
        self.log.info(f'Current world position: {self.system.odometer.prediction} Target: {target}')
        await self.system.driver.drive_to(target)

    def _keep_crops_safe(self) -> None:
        self.log.info('Keeping crops safe...')
        for crop in self.system.plant_provider.crops:
            crop_position = self.system.odometer.prediction.transform(crop.position)
            for weed, weed_position in self.weeds_to_handle.items():
                offset = self.system.field_friend.DRILL_RADIUS + \
                    self.crop_safety_distance - crop.position.distance(weed_position)
                if offset > 0:
                    safe_weed_position = weed_position.polar(offset, crop_position.direction(weed_position))
                    self.weeds_to_handle[weed] = safe_weed_position
                    self.log.info(
                        f'Moved weed {weed} from {weed_position} to {safe_weed_position} by {offset} to safe {crop.id} at {crop_position}')

    def _crops_in_drill_range(self, crop_id: str, crop_position: Point, angle: float) -> bool:
        inner_diameter, outer_diameter = self.system.field_friend.tornado_diameters(angle)
        for crop in self.system.plant_provider.crops:
            crop_world_position = self.system.odometer.prediction.transform(crop_position)
            if crop.id != crop_id:
                distance = crop_world_position.distance(crop.position)
                if distance >= inner_diameter/2 and distance <= outer_diameter/2:
                    self.log.info(f'other crop with distance {distance} in drill range, drilling not allowed')
                    return True
        return False

    # def _safe_crop_to_row(self, crop_id: str) -> None:
    #     if self.current_row is None:
    #         return
    #     self.log.info(f'Saving crop {crop_id} to row {self.current_row.name}...')
    #     crop = next((c for c in self.system.plant_provider.crops if c.id == crop_id), None)
    #     if crop is None:
    #         self.log.error(f'Error in crop saving: Crop with id {crop_id} not found')
    #         return
    #     for c in self.current_row.crops:
    #         if c.position.distance(crop.position) < 0.07:
    #             self.log.info('Crop already in row')
    #             self.current_row.crops.remove(c)
    #             self.current_row.crops.append(crop)
    #             return
    #     if crop.confidence >= 0.85 and len(crop.positions) >= 10:
    #         self.log.info('Adding new crop to row')
    #         self.current_row.crops.append(crop)

    async def _create_simulated_plants(self):
        self.log.info('Creating simulated plants...')
        if self.current_segment:
            self.log.info('Creating simulated plants for current segment')
            distance = self.current_segment.spline.start.distance(self.current_segment.spline.end)
            for i in range(1, int(distance/0.20)):
                for j in range(1, 4):
                    await self.system.plant_provider.add_crop(Plant(
                        id_=f'{i}_{j}',
                        type_='beet',
                        position=self.system.odometer.prediction.point.polar(
                            0.18*i+(randint(-2, 2)*0.01), self.system.odometer.prediction.yaw).polar(randint(-4, 4)*0.01, self.system.odometer.prediction.yaw+np.pi/2),
                        detection_time=rosys.time(),
                        confidence=0.9,
                    ))
                    # for j in range(1, 5):
                    #     await self.system.plant_provider.add_weed(Plant(
                    #         id=f'{i}_{j}',
                    #         type='weed',
                    #         positions=[self.system.odometer.prediction.point.polar(
                    #             0.20*i+randint(-5, 5)*0.01, self.system.odometer.prediction.yaw).polar(randint(-15, 15)*0.01, self.system.odometer.prediction.yaw + np.pi/2)],
                    #         detection_time=rosys.time(),
                    #         confidence=0.9,
                    #     ))
        else:
            self.log.info('Creating simulated plants for whole row')
            for i in range(0, 30):
                await self.system.plant_provider.add_crop(Plant(
                    id_=str(i),
                    type_='beet',
                    position=self.system.odometer.prediction.point.polar(0.20*i, self.system.odometer.prediction.yaw)
                    .polar(randint(-2, 2)*0.01, self.system.odometer.prediction.yaw+np.pi/2),
                    detection_time=rosys.time(),
                    confidence=0.9,
                ))
                for j in range(1, 7):
                    await self.system.plant_provider.add_weed(Plant(
                        id_=f'{i}_{j}',
                        type_='weed',
                        positions=[self.system.odometer.prediction.point.polar(0.20*i+randint(-5, 5)*0.01, self.system.odometer.prediction.yaw).polar(
                            randint(-15, 15)*0.01, self.system.odometer.prediction.yaw + np.pi/2)],
                        detection_time=rosys.time(),
                        confidence=0.9,
                    ))
