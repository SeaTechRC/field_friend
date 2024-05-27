import rosys

from field_friend.automations.implements.implement import Implement

from ..kpi_provider import KpiProvider
from .navigation import Navigation


class StraightLineNavigation(Navigation):

    def __init__(self,
                 driver: rosys.driving.Driver,
                 odometer: rosys.driving.Odometer,
                 kpi_provider: KpiProvider,
                 tool: Implement,
                 ) -> None:
        super().__init__(driver, odometer, kpi_provider, tool)
        self.length = 2.0
        self.start_position = self.odometer.prediction.point

    async def _start(self):
        self.start_position = self.odometer.prediction.point
        if not await self.implement.prepare():
            self.log.error('Tool-Preparation failed')
            return
        self.log.info('driving straight line forward...')
        await self.implement.activate()
        while True:
            await rosys.automation.parallelize(
                self.implement.observe(),
                self._drive_forward(),
                return_when_first_completed=True
            )
            await self.implement.on_focus()
            if await self._should_stop():
                break
        await self.implement.deactivate()

    async def _drive_forward(self):
        while not await self._should_stop():
            self.log.info('driving forward...')
            target = self.odometer.prediction.transform(rosys.geometry.Point(x=0.10, y=0))
            await self.driver.drive_to(target)

    async def _should_stop(self):
        distance = self.odometer.prediction.point.distance(self.start_position)
        return distance > self.length
