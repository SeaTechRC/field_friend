import colorsys
import logging
from typing import Optional

import numpy as np
import rosys
from nicegui import ui
from nicegui.events import MouseEventArguments, ValueChangeEventArguments
from rosys import background_tasks
from rosys.geometry import Point
from rosys.vision.detector import Autoupload

from ..automations import Puncher
from .calibration_dialog import calibration_dialog


class camera:

    def __init__(self,
                 camera_provider: rosys.vision.CameraProvider,
                 automator: rosys.automation.Automator,
                 detector: rosys.vision.Detector,
                 puncher: Optional[Puncher] = None,
                 *,
                 version: str,
                 shrink_factor: int = 1) -> None:
        self.log = logging.getLogger('field_friend.camera_card')
        self.camera: Optional[rosys.vision.CalibratableCamera] = None
        self.camera_provider = camera_provider
        self.automator = automator
        self.detector = detector
        self.capture_images = ui.timer(7, lambda: background_tasks.create(self.capture_image()), active=False)
        self.punching_enabled = False
        self.puncher = puncher
        self.shrink_factor = shrink_factor
        self.image_view: Optional[ui.interactive_image] = None
        self.calibration_dialog = calibration_dialog(camera_provider, version=version)
        self.camera_card = ui.card()
        with self.camera_card.tight().style('width:640px'):
            ui.label('no camera available').classes('text-center')
            ui.image('assets/field_friend.webp').classes('w-full')
        ui.timer(0.5, self.update_content)

    def use_camera(self, cam: rosys.vision.CalibratableCamera) -> None:
        self.camera = cam
        self.camera_card.clear()
        with self.camera_card:
            events = ['mousemove', 'mouseout', 'mouseup']
            self.image_view = ui.interactive_image(
                '',
                cross=True,
                on_mouse=self.on_mouse_move,
                events=events
            ).classes('w-full')
            with ui.row().classes('m-4 justify-end items-center'):
                ui.checkbox('Punching').bind_value(self, 'punching_enabled') \
                    .tooltip('Enable punching mode')
                # self.depth = ui.number('depth', value=0.02, format='%.2f',
                #                        step=0.01, min=0.01, max=0.18).classes('w-16').bind_visibility_from(self, 'punching_enabled')
                self.angle = ui.number('angle', value=180, format='%.0f', step=1, min=0, max=180)
                ui.checkbox('Capturing').bind_value_to(self.capture_images, 'active') \
                    .tooltip('Record new images for the Learning Loop')
                self.show_mapping_checkbox = ui.checkbox('Mapping', on_change=self.show_mapping) \
                    .tooltip('Show the mapping between camera and world coordinates')
                ui.button('calibrate', on_click=self.calibrate) \
                    .props('icon=straighten outline').tooltip('Calibrate camera')
            with ui.row():
                self.debug_position = ui.label()

    def update_content(self) -> None:
        cameras = list(self.camera_provider.cameras.values())
        if not cameras:
            self.camera = None
            self.camera_card.clear()
            with self.camera_card:
                ui.label('no camera available').classes('text-center')
                ui.image('assets/field_friend.webp').classes('w-full')
            return
        if self.camera is None or self.camera not in cameras:
            self.use_camera(cameras[0])
            assert self.camera is not None

        if self.shrink_factor > 1:
            url = f'{self.camera.get_latest_image_url()}?shrink={self.shrink_factor}'
        else:
            url = self.camera.get_latest_image_url()
        self.image_view.set_source(url)
        image = self.camera.latest_captured_image
        if image and image.detections:
            self.image_view.set_content(image.detections.to_svg())

    async def capture_image(self) -> None:
        if self.camera is None:
            return
        await self.detector.detect(self.camera.latest_captured_image, autoupload=Autoupload.ALL, tags=['capture'])

    def on_mouse_move(self, e: MouseEventArguments):
        if self.camera is None:
            return
        if e.type == 'mousemove':
            point2d = Point(x=e.image_x, y=e.image_y)
            if self.camera.calibration is None:
                self.debug_position.set_text(f'{point2d} no calibration')
                return
            point3d = self.camera.calibration.project_from_image(point2d)
            self.debug_position.set_text(f'{point2d} -> {point3d}')
        if e.type == 'mouseup':
            point2d = Point(x=e.image_x, y=e.image_y)
            if self.camera.calibration is None:
                self.debug_position.set_text(f'last punch: {point2d}')
                return
            point3d = self.camera.calibration.project_from_image(point2d)
            if point3d is not None:
                self.debug_position.set_text(f'last punch: {point2d} -> {point3d}')
                if self.puncher is not None and self.punching_enabled:
                    self.log.info(f'punching {point3d}')
                    # self.automator.start(self.puncher.drive_and_punch(point3d.x, point3d.y, self.depth.value))
                    self.automator.start(self.puncher.drive_and_punch(
                        point3d.x, point3d.y, depth=0.05, angle=self.angle.value))
        if e.type == 'mouseout':
            self.debug_position.set_text('')

    async def calibrate(self) -> None:
        result = await self.calibration_dialog.edit(self.camera)
        if result:
            self.show_mapping_checkbox.value = True

    def show_mapping(self, event: ValueChangeEventArguments) -> None:
        if self.camera is None or self.image_view is None:
            return
        if not event.value:
            self.image_view.content = ''
            return
        world_points = np.array([[x, y, 0] for x in np.linspace(0, 0.3, 15) for y in np.linspace(-0.2, 0.2, 20)])
        image_points = self.camera.calibration.project_array_to_image(world_points)
        colors_rgb = [colorsys.hsv_to_rgb(f, 1, 1) for f in np.linspace(0, 1, len(world_points))]
        colors_hex = [f'#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}' for rgb in colors_rgb]
        self.image_view.content = ''.join(f'<circle cx="{p[0]}" cy="{p[1]}" r="2" fill="{color}"/>'
                                          for p, color in zip(image_points, colors_hex))
