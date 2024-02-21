import colorsys
import logging
from typing import TYPE_CHECKING

import rosys
from nicegui import events, ui

from .field_friend_object import field_friend_object
from .field_object import field_object
from .plant_object import plant_objects
from .visualizer_object import visualizer_object

if TYPE_CHECKING:
    from field_friend.system import System


class robot_scene:

    def __init__(self, system: 'System'):
        self.log = logging.getLogger('field_friend.robot_scene')
        self.system = system
        self.scene_card = ui.card()
        self.scene_look = False

        with self.scene_card.tight().classes('w-full place-items-center').style('max-width: 100%; overflow: hidden;'):
            with ui.scene(200, 200, on_click=self.handle_click).classes('w-full') as self.scene:
                field_friend_object(self.system.odometer, self.system.usb_camera_provider, self.system.field_friend)
                rosys.driving.driver_object(self.system.driver)
                plant_objects(self.system.plant_provider, self.system.big_weed_category_names +
                              self.system.small_weed_category_names)
                visualizer_object(self.system.automator, self.system.path_provider,
                                  self.system.mowing, self.system.weeding)
                field_object(self.system.field_provider)
                self.scene.move_camera(-0.5, -1, 2)

    def handle_click(self, event: events.SceneClickEventArguments) -> None:
        if event.click_type == 'dblclick':
            position = self.system.odometer.prediction.point
            if self.scene_look:
                self.scene_look = False
                height = 10
                x = position.x-0.5
                y = position.y-0.5
            else:
                self.scene_look = True
                height = 2
                x = position.x + 0.8
                y = position.y - 0.8
            self.scene.move_camera(x=x, y=y, z=height,
                                   look_at_x=position.x, look_at_y=position.y)
            return