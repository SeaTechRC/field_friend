from typing import TYPE_CHECKING

import rosys
from nicegui import ui

from ..hardware import FieldFriend, FieldFriendHardware
from ..navigation import Gnss


def status_drawer(robot: FieldFriend, gnss: Gnss, odometer: rosys.driving.Odometer) -> None:
    with ui.right_drawer().classes('bg-[#edf4fa]') as status_drawer, ui.column():
        ui.label('System status').classes('text-xl')
        ui.markdown('**Hardware:**')

        with ui.row().bind_visibility_from(robot.estop, 'active'):
            ui.icon('report').props('size=md').classes('text-red')
            ui.label('Emergency stop is pressed!').classes('text-red mt-1')

        with ui.row().bind_visibility_from(robot.estop, 'active', value=False):
            with ui.row().bind_visibility_from(robot.z_axis, 'ref_t', value=False):
                ui.icon('report').props('size=md').classes('text-red')
                ui.label('Z-axis not in top position!').classes('text-red mt-1')

            with ui.row().bind_visibility_from(robot.z_axis, 'end_b'):
                ui.icon('report').props('size=md').classes('text-red')
                ui.label('Z-axis in end bottom pisition, error!').classes('text-red mt-1')

            with ui.row().bind_visibility_from(robot.z_axis, 'alarm'):
                ui.icon('report').props('size=md').classes('text-yellow')
                ui.label('Z-axis in alarm, warning!').classes('text-orange mt-1')

            with ui.row().bind_visibility_from(robot.y_axis, 'alarm'):
                ui.icon('report').props('size=md').classes('text-yellow')
                ui.label('Y-axis in alarm, warning!').classes('text-orange mt-1')

        with ui.row():
            ui.markdown('**Robot:**').style('color: #6E93D6')
            if isinstance(robot, FieldFriendHardware):
                ui.label('real hardware')
            else:
                ui.label('simulated hardware')

        with ui.row():
            ui.markdown('**Battery:**').style('color: #6E93D6')
            bms_label = ui.label()

        with ui.row():
            ui.markdown('**Y-Axis:**').style('color: #6E93D6')
            y_axis_label = ui.label()

        with ui.row():
            ui.markdown('**Z-Axis:**').style('color: #6E93D6')
            z_axis_label = ui.label()

        ui.markdown('**Positioning:**')

        with ui.row():
            ui.markdown('**GNSS-Device:**').style('color: #6E93D6')
            gnss_device_label = ui.label()
        with ui.row():
            ui.markdown('**Reference position:**').style('color: #6E93D6')
            reference_position_label = ui.label()
        with ui.row():
            ui.markdown('**Position:**').style('color: #6E93D6')
            gnss_label = ui.label()
        # with ui.row():
        #     ui.markdown('**Heading:**').style('color: #6E93D6')
        #     heading_label = ui.label()
        with ui.row():
            ui.markdown('**RTK-Fix:**').style('color: #6E93D6')
            rtk_fix_label = ui.label()

        with ui.row():
            ui.markdown('**odometry:**').style('color: #6E93D6')
            odometry_label = ui.label()

        def update_status() -> None:
            bms_flags = [
                f'{robot.bms.state.short_string}',
                'charging' if robot.bms.state.is_charging else ''
            ]
            y_axis_flags = [
                'not referenced' if not robot.y_axis.is_referenced else '',
                'alarm' if robot.y_axis.alarm else '',
                'idle'if robot.y_axis.idle else 'moving',
                'ref l' if robot.y_axis.end_l else '',
                'ref r' if robot.y_axis.end_r else '',
                f'{robot.y_axis.steps:.0f}',
                f'{robot.y_axis.position:.2f}m' if robot.y_axis.is_referenced else ''
            ]
            z_axis_flags = [
                '' if robot.z_axis.is_referenced else 'not referenced',
                'alarm' if robot.z_axis.alarm else '',
                'idle' if robot.z_axis.idle else 'moving',
                'ref stop enabled' if robot.z_axis.is_ref_enabled else '',
                'end disabled' if not robot.z_axis.is_end_b_enabled else '',
                'ref_t active' if robot.z_axis.ref_t else '',
                'end_b active' if robot.z_axis.end_b else '',
                f'{robot.z_axis.steps}',
                f'{robot.z_axis.depth:.2f}m' if robot.z_axis.is_referenced else '',
            ]
            bms_label.text = ', '.join(flag for flag in bms_flags if flag)
            y_axis_label.text = ', '.join(flag for flag in y_axis_flags if flag)
            z_axis_label.text = ', '.join(flag for flag in z_axis_flags if flag)
            direction_flag = \
                'N' if gnss.record.heading <= 23 else \
                'NE' if gnss.record.heading <= 68 else \
                'E' if gnss.record.heading <= 113 else \
                'SE' if gnss.record.heading <= 158 else \
                'S' if gnss.record.heading <= 203 else \
                'SW' if gnss.record.heading <= 248 else \
                'W' if gnss.record.heading <= 293 else \
                'NW' if gnss.record.heading <= 338 else \
                'N'
            gnss_device_label.text = 'No connection' if gnss.device is None else 'Connected'
            reference_position_label.text = 'No reference' if gnss.reference_lat is None else 'Set'
            gnss_label.text = f'lat: {gnss.record.latitude:.6f}, lon: {gnss.record.longitude:.6f}'
            # heading_label.text = f'{gnss.record.heading:.2f}° ' + direction_flag
            rtk_fix_label.text = f'gps_qual: {gnss.record.gps_qual}, mode: {gnss.record.mode}'
            odometry_label.text = str(odometer.prediction)

        ui.timer(rosys.config.ui_update_interval, update_status)
    return status_drawer