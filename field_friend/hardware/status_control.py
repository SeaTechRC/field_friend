from rosys.hardware.expander import ExpanderHardware
from rosys.hardware.module import ModuleHardware
from rosys.hardware.robot_brain import RobotBrain


class StatusControlHardware(ModuleHardware):

    def __init__(self, robot_brain: RobotBrain, *,
                 expander: ExpanderHardware,
                 rdyp_pin: int = 39,
                 vdp_pin: int = 39,
                 ) -> None:
        self.rdyp_status: bool = False
        self.vdp_status: bool = False
        self.heap: int = 0
        self.expander_last_message_age: int = 0
        lizard_code = f'rdyp_status = Input({rdyp_pin})\n'
        lizard_code += f'vdp_status = {expander.name + "."}Input({vdp_pin})\n'
        core_message_fields = [
            'rdyp_status.level',
            'vdp_status.level',
            'core.heap',
            f'{expander.name}.last_message_age'
        ]
        super().__init__(robot_brain=robot_brain, lizard_code=lizard_code, core_message_fields=core_message_fields)

    def handle_core_output(self, _: float, words: list[str]) -> None:
        self.rdyp_status = int(words.pop(0)) == 0
        self.vdp_status = int(words.pop(0)) == 0
        self.heap = int(words.pop(0))
        self.expander_last_message_age = int(words.pop(0))
