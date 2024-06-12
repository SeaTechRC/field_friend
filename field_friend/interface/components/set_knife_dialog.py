from typing import Optional

import rosys
from icecream import ic
from nicegui import ui
from rosys.driving import Odometer
from rosys.geometry import Point3d
from rosys.vision import Image

from ...automations import PlantLocator
from ...automations.plant import Plant


class SetKnifeDialog(ui.dialog):
    def __init__(self, field_friend) -> None:
        super().__init__()
        self.field_friend = field_friend
        with self, ui.card().style('max-width: 1400px'):
            with ui.row(wrap=False):
                ui.label('Click on turn to turn to the next knife.').classes('text-lg')
            with ui.row():
                ui.button('Turn', on_click=self.turn_knifes)
                ui.button('Ok', on_click=lambda: self.submit('Ok'))

    async def turn_knifes(self):
        self.field_friend.z_axis.turn_by(1/3.0)

    def open(self) -> None:
        assert self.field_friend is not None
        super().open()
