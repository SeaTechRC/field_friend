import contextlib
from collections.abc import Generator
from typing import Any

import rosys
from rosys.analysis import track
from rosys.geometry import Point


class Implement(rosys.persistence.Persistable):

    def __init__(self, name: str = 'None') -> None:
        super().__init__()
        self.name = name
        self.is_active = False
        self._is_blocked = False

    @contextlib.contextmanager
    def blocked(self) -> Generator[None, None, None]:
        """Context manager to temporarily block the implement from working.

        Usage:
            with implement.blocked():
                # do something where implement is not allowed
        """
        self._is_blocked = True
        try:
            yield
        finally:
            self._is_blocked = False

    @property
    def is_blocked(self) -> bool:
        """Returns whether the implement is currently blocked from working."""
        return self._is_blocked

    async def prepare(self) -> bool:
        """Prepare the implement once at the beginning (for reference points, etc.);

        return False if preparation failed."""
        return True

    @track
    async def finish(self) -> None:
        """Finish the implement once at the end"""
        return None

    @track
    async def activate(self):
        """Activate the implement (for example to start weeding in a new row)"""
        self.is_active = True

    @track
    async def deactivate(self):
        """Deactivate the implement (for example to stop weeding at the row's end)"""
        self.is_active = False

    @track
    async def get_move_target(self) -> Point | None:
        """Return the target position to drive to."""
        return None

    @track
    async def start_workflow(self) -> None:
        """Called after robot has stopped via observation to perform it's workflow on a specific point on the ground

        Returns True if the robot can drive forward, if the implement whishes to stay at the current location, return False
        """
        assert not self._is_blocked

    @track
    async def stop_workflow(self) -> None:
        """Called after workflow has been performed to stop the workflow"""
        return None

    def backup_to_dict(self) -> dict[str, Any]:
        return {}

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        pass

    def settings_ui(self) -> None:
        """Create UI for settings and configuration."""
        return None
