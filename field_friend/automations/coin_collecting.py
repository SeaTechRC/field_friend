
import logging
from typing import TYPE_CHECKING

import rosys
from rosys.geometry import Point
from rosys.helpers import eliminate_2pi

from .plant import Plant
from .puncher import PuncherException

if TYPE_CHECKING:
    from system import System

TORNADO_ANGLE = 110.0


class CoinCollecting():

    def __init__(self, system: 'System') -> None:
        super().__init__()
        self.log = logging.getLogger('field_friend.coin_collecting')
        self.system = system
        self.kpi_provider = system.kpi_provider
        self.work_x: float = 0.0
        self.front_x: float = 0.18

    async def start(self) -> None:
        self.log.info('starting coin collecting')

        if self.system.field_friend.estop.active or self.system.field_friend.estop.is_soft_estop_active:
            rosys.notify('E-Stop is active, aborting', 'negative')
            self.log.error('E-Stop is active, aborting')
            return
        if self.system.field_friend.tool != 'tornado':
            rosys.notify('Tool is not tornado, aborting', 'negative')
            self.log.error('Tool is not tornado, aborting')
            return
        if self.system.field_friend.y_axis.alarm:
            rosys.notify('Y-Axis is in alarm, aborting', 'negative')
            self.log.error('Y-Axis is in alarm, aborting')
            return
        if self.system.field_friend.z_axis.ref_knife_stop:
            rosys.notify('Tornado is in knife stop ref', 'negative')
            self.log.error('Tornado is in knife stop ref')
            return
        if not await self.system.puncher.try_home():
            rosys.notify('Puncher homing failed, aborting', 'negative')
            self.log.error('Puncher homing failed, aborting')
        self.work_x = self.system.field_friend.WORK_X
        self.system.odometer.reset()
        self.system.plant_provider.clear()
        if not self.system.field_friend.flashlight.is_active:
            await self.system.field_friend.flashlight.turn_on()
            await rosys.sleep(6)
        await self._weeding()

    async def _weeding(self) -> None:
        rosys.notify('Coin collecting started', 'positive')
        self.log.info('Coin collecting started')
        already_explored = False
        try:
            while True:
                self.system.plant_locator.pause()
                self.system.plant_provider.clear()
                self.log.info(f'cleared crops at start {self.system.plant_provider.crops}')
                if not self.system.is_real:
                    self.create_simulated_plants()
                self.system.plant_locator.resume()
                await rosys.sleep(2)
                while upcoming_crop_positions := self.get_upcoming_crops():
                    try:
                        already_explored = False
                        closest = upcoming_crop_positions[0]
                        self.log.info(f'all upcoming crops: {upcoming_crop_positions}')
                        if closest.x < 0.08:
                            # do not steer while advancing on a crop
                            target = self.system.odometer.prediction.transform(Point(x=closest.x, y=0))
                            self.log.info('target next crop')
                            await self.system.driver.drive_to(target)
                            await self._use_implement(closest)
                            # await self.system.driver.drive_to(self.system.odometer.prediction.transform(Point(x=0.03, y=0)))
                            target = self.system.odometer.prediction.transform(Point(x=0.03, y=0))
                            self.log.info(f'driving a bit forward after using implement: {target}')
                            await self.system.driver.drive_to(target)
                        else:
                            self.log.info('follow line of crops')
                            farthest = upcoming_crop_positions[-1]
                            upcoming_world_position = self.system.odometer.prediction.transform(farthest)
                            yaw = self.system.odometer.prediction.point.direction(upcoming_world_position)
                            # only apply minimal yaw corrections to avoid oversteering
                            yaw = eliminate_2pi(self.system.odometer.prediction.yaw) * 0.8 + eliminate_2pi(yaw) * 0.2
                            target = self.system.odometer.prediction.point.polar(0.03, yaw)
                            await self.system.driver.drive_to(target)
                        await rosys.sleep(2)  # ensure we have a super accurate detection
                    except PuncherException as e:
                        self.log.error(f'PuncherException: {e}')
                        continue
                    except Exception as e:
                        self.log.exception(f'error while advancing on crop: {e}')
                        self.kpi_provider.increment('automation_stopped')
                        break
                if self.system.odometer.prediction.point.distance(Point(x=0, y=0)) > 0.1:
                    self.log.info('returning to start position')
                    await self.system.driver.drive_to(Point(x=0, y=0), backward=True)
                if not self.get_upcoming_crops() and not already_explored:
                    self.log.info('no crops found, advancing a bit to ensure there are really no more crops')
                    target = self.system.odometer.prediction.transform(Point(x=0.05, y=0))
                    await self.system.driver.drive_to(target)
                    already_explored = True  # avoid infinite loop if there are no crops
        finally:
            self.kpi_provider.increment('coin_collecting_completed')
            await rosys.sleep(0.1)
            await self.system.field_friend.stop()

    def get_upcoming_crops(self):
        """Get upcoming crops in local coordinates sorted by distance to implement.

        param: backward: if True, crops behind the implement are returned
        """
        for crop in self.system.plant_provider.crops:
            self.log.info(f'crop: {crop.position} and {rosys.time() - crop.detection_time} seconds old')
        relative_crop_positions = [self.system.odometer.prediction.relative_point(c.position)
                                   for c in self.system.plant_provider.crops]
        upcoming = [c for c in relative_crop_positions if c.x >= self.work_x+0.05]
        upcoming.sort(key=lambda c: c.x)
        return upcoming

    def create_simulated_plants(self):
        for i in range(1, 8):
            self.system.plant_provider.add_crop(Plant(
                id_=str(i),
                type_='coin_with_hole',
                position=Point(x=0.1 * i, y=pow(i*0.1, 5)),
                detection_time=rosys.time(),
                confidence=0.9,
            ))

    async def _use_implement(self, center: Point) -> None:
        if self.system.field_friend.y_axis.min_position > center.y > self.system.field_friend.y_axis.max_position:
            self.log.info('crop is not reachable with y axis')
            return
        self.system.plant_locator.pause()
        self.log.info('Using implement')
        await self.system.puncher.punch(center.y, angle=TORNADO_ANGLE)
        self.system.plant_locator.resume()
