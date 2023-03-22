import rosys
from nicegui import ui

from ..automations import Puncher
from ..hardware import FieldFriend


def axis_control(field_friend: FieldFriend, automator: rosys.automation.Automator, puncher: Puncher) -> None:
    with ui.card():
        with ui.column():
            ui.markdown('**Axis Settings**').classes('col-grow')
            with ui.row():
                with ui.menu() as developer_menu:
                    async def try_axis_home():
                        await puncher.home()
                    ui.menu_item('perform homing', on_click=lambda: automator.start(try_axis_home()))
                    ui.menu_item('Disable end stops', on_click=lambda: automator.start(
                        field_friend.y_axis.enable_end_stops(False)))
                    ui.menu_item('Enable end stops', on_click=lambda: automator.start(
                        field_friend.z_axis.enable_end_stops(True)))
                ui.button(on_click=developer_menu.open).props('dense fab-mini outline icon=more_vert')
                robot_status = ui.markdown()
        ui.timer(1, lambda: robot_status.set_content(
            f' YAXIS: Alarm: {field_friend.y_axis.alarm} | Idle: {field_friend.y_axis.idle} | Pos: {field_friend.y_axis.position} | Home: {field_friend.y_axis.home_position} | Ref:{field_friend.y_axis.is_referenced} | endL: {field_friend.y_axis.end_l} | endR: {field_friend.y_axis.end_r}<br>'
            f'ZAXIS: Alarm: {field_friend.z_axis.alarm} | Idle: {field_friend.z_axis.idle} | Pos: {field_friend.z_axis.position} | Home: {field_friend.z_axis.home_position} | Ref:{field_friend.z_axis.is_referenced} | endT: {field_friend.z_axis.end_t} | endB: {field_friend.z_axis.end_b}'
        ))