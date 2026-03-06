from __future__ import annotations
from typing import Any
from drones.dronecore.drone import DroneEnvironment, Pose


class RealDroneEnvironment(DroneEnvironment):
    def __init__(self, backend, camera=None) -> None:
        self.backend = backend
        self.camera = camera

    def move(self, delta_position, delta_orientation) -> Pose:
        # TODO: Send relative move command directly via the drone backend
        # TODO: Wait for the drone to settle (or time out), then return pose.
        raise NotImplementedError()

    def move_to(self, position, orientation) -> Pose:
        # TODO: Send an absolute move command (if supported) and return pose
        raise NotImplementedError()

    def get_pose(self) -> Pose:
        # TODO: Query the drone telemetry system and return current pose
        raise NotImplementedError()

    def render(self) -> Any:
        # TODO: Capture an image from the onboard camera or camera pipeline
        raise NotImplementedError()
