from typing import Optional

import rosys
from rosys.helpers import remove_indentation

from .z_axis import ZAxis


class ZAxisCanOpenHardware(ZAxis, rosys.hardware.ModuleHardware):
    """The z axis hardware module is a simple example for a representation of real robot hardware."""

    def __init__(self, robot_brain: rosys.hardware.RobotBrain, *,
                 name: str = 'zaxis',
                 can: rosys.hardware.CanHardware,
                 expander: Optional[rosys.hardware.ExpanderHardware],
                 can_address: int = 0x60,
                 max_speed: int = 2000,
                 reference_speed: int = 40,
                 min_position: float = -0.15,
                 max_position: float = 0.0,
                 axis_offset: float = 0.0,
                 steps_per_m: float = 1_481_481.48,  # 4000steps/turn motor; 1/20 gear; 0.054m/u
                 end_t_pin: int = 19,
                 end_b_pin: int = 21,
                 motor_on_expander: bool = False,
                 end_stops_on_expander: bool = True,
                 reversed_direction: bool = False,
                 ) -> None:
        self.name = name
        self.expander = expander
        lizard_code = remove_indentation(f'''
            {name}_motor = {expander.name + "." if motor_on_expander and expander else ""}CanOpenMotor({can.name}, {can_address})
            {name}_end_t = {expander.name + "." if end_stops_on_expander and expander else ""}Input({end_t_pin})
            {name}_end_b = {expander.name + "." if end_stops_on_expander and expander else ""}Input({end_b_pin})
            bool {name}_ends_enabled = true;
            bool {name}_is_referencing = false;
            when {name}_ends_enabled and ({name}_end_t.level == 0 or {name}_end_b.level == 0) then
                {name}_motor.set_ctrl_halt(true);
            end
            when !{name}_ends_enabled and {name}_is_referencing and {name}_end_t.level == 1 then
                {name}_motor.set_ctrl_halt(true);
            end
        ''')
        core_message_fields = [
            f'{name}_end_t.level',
            f'{name}_end_b.level',
            f'{name}_motor.actual_position',
            f'{name}_motor.status_target_reached',
            f'{name}_motor.status_fault',
        ]
        super().__init__(
            max_speed=max_speed,
            reference_speed=reference_speed,
            min_position=min_position,
            max_position=max_position,
            axis_offset=axis_offset,
            steps_per_m=steps_per_m,
            reversed_direction=reversed_direction,
            robot_brain=robot_brain,
            lizard_code=lizard_code,
            core_message_fields=core_message_fields)

    async def stop(self) -> None:
        await super().stop()
        await self.robot_brain.send(f'{self.name}_motor.set_ctrl_enable(false);')

    async def move_to(self, position: float, speed: int | None = None) -> None:
        if speed is None:
            speed = self.max_speed
        try:
            await super().move_to(position, speed)
        except RuntimeError as error:
            self.log.error(f'could not move zaxis to {position} because of {error}')
            raise Exception(f'could not move zaxis to {position} because of {error}') from error
        steps = self.compute_steps(position)
        self.log.info(f'moving to steps: {steps}')
        await self.enable_motor()
        await self.enter_pp_mode(speed)
        await rosys.sleep(0.1)
        await self.robot_brain.send(
            f'{self.name}_motor.set_target_position({steps});'
            f'{self.name}_motor.commit_target_position();'
        )
        # Give flags time to turn false first
        await rosys.sleep(0.2)
        while not self.idle and not self.alarm:
            await self.enter_pp_mode(speed)
            await self.robot_brain.send(
                f'{self.name}_motor.set_target_position({steps});'
                f'{self.name}_motor.commit_target_position();'
            )
            await rosys.sleep(0.2)
        if self.alarm:
            self.log.error(f'could not move zaxis to {position} because of fault')
            raise Exception(f'could not move zaxis to {position} because of fault')
        self.log.info(f'zaxis moved to {position}')
        await self.robot_brain.send(f'{self.name}_motor.set_ctrl_enable(false);')

    async def enable_motor(self) -> None:
        await self.robot_brain.send(f'{self.name}_motor.set_ctrl_enable(true);')

    async def disable_motor(self) -> None:
        await self.robot_brain.send(f'{self.name}_motor.set_ctrl_enable(false);')

    async def enter_pp_mode(self, velocity: int) -> None:
        await self.robot_brain.send(f'{self.name}_motor.enter_pp_mode({velocity});')

    async def reset_fault(self) -> None:
        await self.robot_brain.send(f'{self.name}_motor.reset_fault();')
        await rosys.sleep(1)
        if self.alarm:
            self.log.error('could not reset zaxis fault')
            raise Exception('could not reset zaxis fault')

    async def try_reference(self) -> bool:
        if not await super().try_reference():
            return False
        try:
            self.log.info("enabling h motors")
            await self.enable_motor()
            await self.robot_brain.send(
                f'{self.name}_is_referencing = true;'
                f'{self.name}_ends_enabled = true;'
                f'{self.name}_motor.position_offset = 0;'
            )
            await rosys.sleep(1)

            # if in end b stop, move out
            if self.end_b:
                self.log.info('already in end_b moving out of end_b stop')
                await self.robot_brain.send(f'{self.name}_ends_enabled = false;')
                await rosys.sleep(1)
                velocity = self.reference_speed * (-1 if self.reversed_direction else 1)
                await self.robot_brain.send(
                    f'{self.name}_motor.enter_pv_mode({velocity});'
                    f'{self.name}_motor.set_ctrl_halt(false);'
                )
                while self.end_b:
                    await rosys.sleep(0.2)
                await self.robot_brain.send(f'{self.name}_motor.set_ctrl_halt(true);')
            await rosys.sleep(0.5)

            # move to end t stop if not already there
            if not self.end_t:
                self.log.info('moving to end_t stop')
                await self.robot_brain.send(f'{self.name}_ends_enabled = true;')
                await rosys.sleep(1)
                velocity = self.reference_speed * (-1 if self.reversed_direction else 1)
                await self.robot_brain.send(
                    f'{self.name}_motor.enter_pv_mode({velocity});'
                    f'{self.name}_motor.set_ctrl_halt(false);'
                )
                while not self.end_t:
                    await rosys.sleep(0.2)
            await rosys.sleep(0.5)

            # move out of end t stop
            self.log.info('moving out of end_t stop')
            await self.robot_brain.send(f'{self.name}_ends_enabled = false;')
            await rosys.sleep(1)
            velocity = -self.reference_speed * (-1 if self.reversed_direction else 1)
            await self.robot_brain.send(
                f'{self.name}_motor.enter_pv_mode({velocity});'
                f'{self.name}_motor.set_ctrl_halt(false);'
            )
            while self.end_t:
                await rosys.sleep(0.2)
            await rosys.sleep(0.5)

            # move slowly to end t stop
            self.log.info('moving slowly to end_t stop')
            await self.robot_brain.send(f'{self.name}_ends_enabled = true;')
            await rosys.sleep(1)
            slow_velocity = 25 * (-1 if self.reversed_direction else 1)
            await self.robot_brain.send(
                f'{self.name}_motor.enter_pv_mode({slow_velocity});'
                f'{self.name}_motor.set_ctrl_halt(false);'
            )
            while not self.end_t:
                await rosys.sleep(0.2)
            await rosys.sleep(0.5)

            # move slowly out of end t stop
            self.log.info('moving slowly out of end_t stop')
            await self.robot_brain.send(f'{self.name}_ends_enabled = false;')
            await rosys.sleep(1)
            slow_velocity = -25 * (-1 if self.reversed_direction else 1)
            await self.robot_brain.send(
                f'{self.name}_motor.enter_pv_mode({slow_velocity});'
                f'{self.name}_motor.set_ctrl_halt(false);'
            )
            while self.end_t:
                await rosys.sleep(0.2)
            await rosys.sleep(0.5)

            # save position
            await self.robot_brain.send(f'{self.name}_motor.position_offset = {self.steps};')
            await rosys.sleep(0.2)
            await self.robot_brain.send(
                f'{self.name}_is_referencing = false;'
                f'{self.name}_ends_enabled = true;'
            )
            self.log.info('zaxis referenced')
            self.is_referenced = True
            self.log.info(f'actual position: {self.position}, and steps: {self.steps}')
            return True
        except Exception as error:
            self.log.error(f'could not reference zaxis because of {error}')
            return False
        finally:
            await self.stop()
            await self.robot_brain.send(
                f'{self.name}_is_referencing = false;'
                f'{self.name}_ends_enabled = true;'
            )

    def handle_core_output(self, time: float, words: list[str]) -> None:
        self.end_t = int(words.pop(0)) == 0
        self.end_b = int(words.pop(0)) == 0
        if self.end_t or self.end_b:
            self.is_referenced = False
        self.steps = int(words.pop(0))
        self.idle = words.pop(0) == 'true'
        self.alarm = words.pop(0) == 'true'
        if self.alarm:
            self.is_referenced = False
