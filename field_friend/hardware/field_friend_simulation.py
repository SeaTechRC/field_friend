import rosys

from .chain_axis import ChainAxisSimulation
from .configurations import fieldfriend_configurations
from .field_friend import FieldFriend
from .flashlight import FlashlightSimulation
from .flashlight_v2 import FlashlightSimulationV2
from .safety import SafetySimulation
from .y_axis import YAxisSimulation
from .z_axis import ZAxisSimulation
from .z_axis_v2 import ZAxisSimulationV2


class FieldFriendSimulation(FieldFriend, rosys.hardware.RobotSimulation):

    def __init__(self,  version: str) -> None:
        if version not in fieldfriend_configurations:
            raise ValueError(f'Unknown FieldFriend version: {version}')
        config = fieldfriend_configurations[version]
        wheels = rosys.hardware.WheelsSimulation()
        if config['y_axis']['version'] == 'chain_axis':
            y_axis = ChainAxisSimulation()
        elif config['y_axis']['version'] == 'y_axis':
            y_axis = YAxisSimulation()
        else:
            y_axis = None

        if config['z_axis']['version'] == 'z_axis':
            z_axis = ZAxisSimulation()
        elif config['z_axis']['version'] == 'z_axis_v2':
            z_axis = ZAxisSimulationV2()
        else:
            z_axis = None

        if config['flashlight']['version'] == 'flashlight':
            flashlight = FlashlightSimulation()
        elif config['flashlight']['version'] == 'flashlight_v2':
            flashlight = FlashlightSimulationV2()
        else:
            flashlight = None

        estop = rosys.hardware.EStopSimulation()
        bms = rosys.hardware.BmsSimulation()
        safety = SafetySimulation(wheels=wheels, estop=estop, y_axis=y_axis, z_axis=z_axis, flashlight=flashlight)
        modules = [wheels, y_axis, z_axis, flashlight, bms, estop, safety]
        active_modules = [module for module in modules if module is not None]
        super().__init__(version=version,
                         wheels=wheels,
                         flashlight=flashlight,
                         y_axis=y_axis,
                         z_axis=z_axis,
                         estop=estop,
                         bms=bms,
                         safety=safety,
                         modules=active_modules)