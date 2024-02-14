#!/usr/bin/env python3
from nicegui import app, ui

import field_friend.log_configuration as log_configuration
from field_friend import interface
from field_friend.system import System

logger = log_configuration.configure()
app.add_static_files('/assets', 'assets')


def startup() -> None:
    system = System()

    @ui.page('/')
    def page(dev: bool = False) -> None:
        ui.colors(primary='#6E93D6', secondary='#53B689', accent='#111B1E', positive='#53B689')
        status_drawer = interface.status_drawer(system.field_friend, system.gnss, system.odometer)
        interface.header_bar(system, status_drawer)
        interface.system_bar()
        with ui.row().style('max-height:calc(100vh - 125px); height:calc(100vh - 150px); width: calc(100vw - 2rem); flex-wrap: nowrap'):
            with ui.splitter(horizontal=False, reverse=False, value=35, limits=(10, 50)).classes('w-full h-full') as splitter:
                with splitter.before:
                    with ui.column().classes('h-full p-2').style('width: 100%;'):
                        leaflet_map_landing = interface.leaflet_map(system, False)
                        leaflet_map_landing.m.classes(
                            'h-full w-full')
                with splitter.after:
                    with ui.row().classes('w-full h-full ml-2 m-2'):
                        with ui.column().style('width: 55%; height: 100%; flex-wrap: nowrap'):
                            interface.operation(system, leaflet_map_landing)
                        with ui.column().classes('h-full').style('width: calc(45% - 2rem); flex-wrap: nowrap;'):
                            with ui.card().classes('w-full h-full p-0'):
                                with ui.scroll_area().classes('w-full h-full'):
                                    with ui.card().classes('w-full'):
                                        interface.camera(system.usb_camera_provider, system.automator, system.detector,
                                                         system.puncher, version=system.field_friend.version)
                with splitter.separator:
                    ui.button(icon='drag_indicator').props('round')
            if dev:
                with ui.row().classes('items-stretch justify-items-stretch'):
                    interface.development(system.field_friend)
                    interface.hardware_control(system.field_friend, system.automator, system.puncher)

    @ui.page('/dev')
    def dev_page():
        page(dev=True)

    @ui.page('/field')
    def field_page(dev: bool = True):
        ui.colors(primary='#6E93D6', secondary='#53B689', accent='#111B1E', positive='#53B689')
        status_drawer = interface.status_drawer(system.field_friend, system.gnss, system.odometer)
        interface.header_bar(system, status_drawer)
        interface.system_bar()
        with ui.column().classes('w-full items-stretch').style('max-height:calc(100vh - 125px); height:calc(100vh - 150px);'):
            with ui.row().classes('items-stretch justify-items-stretch').style('flex-wrap:nowrap; height:40%; max-height:40%;'):
                leaflet_map_field = interface.leaflet_map(system, True)
                leaflet_map_field.m.style('height: 100%; max-height:100%;')
            with ui.row().classes('items-stretch justify-items-stretch').style('flex-wrap:nowrap; height: 60%; max-height:60%;'):
                interface.field_planner(system.field_provider, system.odometer, system.gnss, leaflet_map_field)

    @ui.page('/path')
    def path_page(dev: bool = True):
        ui.colors(primary='#6E93D6', secondary='#53B689', accent='#111B1E', positive='#53B689')
        status_drawer = interface.status_drawer(system.field_friend, system.gnss, system.odometer)
        interface.header_bar(system, status_drawer)
        interface.system_bar()
        with ui.column().classes('w-full items-stretch'):
            with ui.row().classes('items-stretch justify-items-stretch').style('flex-wrap:nowrap'):
                interface.operation(system)
                interface.path_planner(system.path_provider, system.path_recorder, system.automator)
            if dev:
                with ui.row().classes('items-stretch justify-items-stretch'):
                    interface.development(system.field_friend)
                    interface.hardware_control(system.field_friend, system.automator, system.puncher)

    @ui.page('/test')
    def test_page():
        status_drawer = interface.status_drawer(system.field_friend, system.gnss, system.odometer)
        interface.header_bar(system, status_drawer)
        interface.system_bar()
        with ui.column().classes('w-full items-stretch'):
            with ui.row().classes('items-stretch justify-items-stretch').style('flex-wrap:nowrap'):
                interface.test(system.field_friend, system.steerer, system.odometer, system.automator, system.driver,
                               system.usb_camera_provider)

    @app.get('/status')
    def status():
        return {'status': 'ok'}


app.on_startup(startup)

ui.run(title='Field Friend', port=80, favicon='assets/favicon.ico')
