from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from field_friend.navigation.point_transformation import wgs84_to_cartesian
import numpy as np
import pynmea2
import rosys
import serial
from geographiclib.geodesic import Geodesic
from serial.tools import list_ports


@dataclass
class GNSSRecord:
    timestamp: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    mode: str = ''
    gps_qual: int = 0
    altitude: float = 0.0
    separation: float = 0.0
    heading: float = 0.0
    speed_kmh: float = 0.0


class Gnss(rosys.persistence.PersistentModule, ABC):

    def __init__(self) -> None:
        super().__init__()
        self.log = logging.getLogger('field_friend.gnss')

        self.ROBOT_LOCATED = rosys.event.Event()
        """the robot has been located (argument: pose)"""

        self.record = GNSSRecord()
        self.device = None
        self.ser = None
        self.reference_lat: Optional[float] = None
        self.reference_lon: Optional[float] = None

        self.needs_backup = False

        rosys.on_repeat(self.update, 1.0)
        rosys.on_repeat(self.try_connection, 3.0)

    def backup(self) -> dict[str, Any]:
        return {
            'reference_lat': self.reference_lat,
            'reference_lon': self.reference_lon,
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.reference_lat = data.get('reference_lat')
        self.reference_lon = data.get('reference_lon')

    @abstractmethod
    async def update(self) -> None:
        pass

    @abstractmethod
    async def try_connection(self) -> None:
        pass

    def clear_reference(self) -> None:
        self.reference_lat = None
        self.reference_lon = None
        self.request_backup()

    def set_reference(self, lat: float, lon: float) -> None:
        self.reference_lat = lat
        self.reference_lon = lon
        self.request_backup()

    def get_reference(self) -> Optional[tuple[float, float]]:
        return self.reference_lat, self.reference_lon

    def calculate_distance(self, lat: float, lon: float) -> Optional[float]:
        if self.reference_lat is None or self.reference_lon is None:
            return None
        geodetic_measurement = Geodesic.WGS84.Inverse(self.reference_lat, self.reference_lon, lat, lon)
        return geodetic_measurement['s12']


class GnssHardware(Gnss):
    PORT = '/dev/cu.usbmodem36307295'

    def __init__(self, odometer: rosys.driving.Odometer) -> None:
        super().__init__()
        self.odometer = odometer

    async def try_connection(self) -> None:
        await super().try_connection()
        if self.device is not None:
            return
        # self.log.info('Searching for GNSS device...')
        for port in list_ports.comports():
            self.log.info(f'Found port: {port.device} - {port.description}')
            if 'Septentrio' in port.description:
                self.log.info(f'Found GNSS device: {port.device}')
                self.device = port.device
                break
        else:
            self.device = None
            # self.log.error('No GNSS device found')
            return

        self.log.info(f'Connecting to GNSS device "{self.device}"')
        try:
            self.ser = serial.Serial(self.device, baudrate=115200, timeout=0.5)
        except serial.SerialException as e:
            self.log.error(f'Could not connect to GNSS device: {e}')
            self.device = None

    async def update(self) -> None:
        await super().update()
        if self.ser is None:
            return
        record = GNSSRecord()
        has_location = False
        has_heading = False
        try:
            lines = await rosys.run.io_bound(self.ser.readlines)
            if not lines:
                self.log.info('No data')
                return
            for line in lines:
                if not line:
                    self.log.info('No data')
                    continue
                try:
                    msg = await rosys.run.cpu_bound(pynmea2.parse, line.decode())
                    if not hasattr(msg, 'sentence_type'):
                        self.log.info(f'No sentence type: {msg}')
                        continue
                    if msg.sentence_type == 'GGA' and getattr(msg, 'gps_qual', 0) > 0:
                        # self.log.info(f'GGA: gps_qual: {msg.gps_qual}, lat:{msg.latitude} and long:{msg.longitude}')
                        record.gps_qual = msg.gps_qual
                        record.altitude = msg.altitude
                        record.separation = msg.geo_sep
                    if getattr(msg, 'spd_over_grnd_kmph', None) is not None:
                        record.speed_kmh = msg.spd_over_grnd_kmph
                    if msg.sentence_type == 'GNS' and getattr(msg, 'mode_indicator', None):
                        # self.log.info(f'GNS: mode: {msg.mode_indicator}, lat:{msg.latitude} and long:{msg.longitude}')
                        if isinstance(msg.timestamp, datetime.time):
                            dt = datetime.datetime.combine(datetime.date.today(), msg.timestamp)
                            dt.replace(tzinfo=datetime.timezone.utc)
                            record.timestamp = dt.timestamp()
                        else:
                            record.timestamp = msg.timestamp
                        record.latitude = msg.latitude
                        record.longitude = msg.longitude
                        record.mode = msg.mode_indicator
                        has_location = True
                    if msg.sentence_type == 'HDT' and getattr(msg, 'heading', None) is not None:
                        # self.log.info(f'HDT: Heading: {msg.heading}')
                        record.heading = msg.heading
                        has_heading = True
                except pynmea2.ParseError as e:
                    self.log.info(f'Parse error: {e}')
                    continue
        except serial.SerialException as e:
            self.log.info(f'Device error: {e}')
            self.device = None
            return
        self.record = record
        if has_location and record.gps_qual == 4:  # 4 = RTK fixed, 5 = RTK float
            if self.reference_lat is None or self.reference_lon is None:
                self.log.info(f'GNSS reference set to {record.latitude}, {record.longitude}')
                self.set_reference(record.latitude, record.longitude)
            else:
                cart_coords = wgs84_to_cartesian([self.reference_lat, self.reference_lon], [
                    record.latitude, record.longitude])
                if has_heading:
                    yaw = np.deg2rad(float(-record.heading))
                else:
                    yaw = self.odometer.get_pose(time=record.timestamp).yaw
                pose = rosys.geometry.Pose(
                    x=cart_coords[0],
                    y=cart_coords[1],
                    yaw=yaw,
                    time=record.timestamp,
                )
                distance = self.odometer.prediction.distance(pose)
                if distance > 1:
                    self.log.warning(f'GNSS distance to prediction to high: {distance:.2f}m!!')
                # TODO would it be helpful  to add an event that has the wgs84 coords as arg
                self.ROBOT_LOCATED.emit(pose)
        elif has_location and record.gps_qual == 5:
            print("5")


class PoseProvider(Protocol):

    @property
    def pose(self) -> rosys.geometry.Pose:
        ...


class GnssSimulation(Gnss):

    def __init__(self, pose_provider: PoseProvider) -> None:
        super().__init__()
        self.pose_provider = pose_provider

    async def update(self) -> None:
        if self.device is None:
            return
        pose = deepcopy(self.pose_provider.pose)
        pose.time = rosys.time()
        # TODO cart to wgs
        self.ROBOT_LOCATED.emit(pose)

    async def try_connection(self) -> None:
        self.device = 'simulation'
