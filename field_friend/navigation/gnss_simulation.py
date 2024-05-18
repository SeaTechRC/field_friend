import rosys

from .geo_point import GeoPoint
from .gnss import Gnss, GNSSRecord


class GnssSimulation(Gnss):

    def __init__(self, odometer: rosys.driving.Odometer) -> None:
        super().__init__(odometer, 0.0)
        self.allow_connection = True
        self.gps_quality = 4

    async def update(self) -> None:
        pose = self.odometer.prediction
        reference = self.reference if self.reference else GeoPoint(lat=51.983159, long=7.434212)
        new_position = reference.shifted(pose.point)
        record = GNSSRecord()
        record.timestamp = pose.time
        record.latitude, record.longitude = new_position.tuple
        record.mode = "simulation"  # TODO check for possible values and replace "simulation"
        record.gps_qual = self.gps_quality
        self._update_record(record)
        await rosys.sleep(0.1)  # NOTE simulation does not be so fast and only eats a lot of cpu time

    async def try_connection(self) -> None:
        if self.allow_connection:
            self.device = 'simulation'

    def disconnect(self):
        """Simulate serial disconnection.

        The hardware implementation sets the device to None if it encounters a serial exception.
        Reconnect is not longer possible.
        """
        self.device = None
        self.allow_connection = False
