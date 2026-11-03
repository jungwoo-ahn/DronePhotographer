"""Local ground-station UI for the pilot process.

Runs on the laptop beside the control loop and serves a single page on localhost.
It is a *window onto the supervisor*, never a second way to command the drone: the
only write it offers is the e-stop, which can solely make the aircraft safer.
"""

from .state import UiState          # noqa: F401
from .server import UiServer        # noqa: F401
