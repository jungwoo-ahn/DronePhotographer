"""The panel's Take a photo: the camera's own frame, saved on the laptop with its note."""

import json
import urllib.error
import urllib.request

import numpy as np

from src.drone_control.camera import Frame
from src.drone_control.photos import PhotoTaker
from src.drone_control.ui import UiServer, UiState


class _Cam:
    def __init__(self, frame=True):
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[:, :960] = 200
        self.frame = Frame(image=img, seq=7) if frame else None

    def latest(self):
        return self.frame


def test_a_photo_is_the_full_frame_with_its_views_and_a_note(tmp_path):
    taker = PhotoTaker(_Cam(), tmp_path, views={"model": lambda im: im[:384, :512], "wide": None},
                       clock=lambda: 1_800_000_000.25)
    rec = taker.take({"shot": "eval-03"})
    assert rec["size"] == [1920, 1080] and rec["frame_seq"] == 7
    assert set(rec["files"]) == {"photo", "model"}               # a missing view is left out
    photo = tmp_path / rec["files"]["photo"]
    assert photo.read_bytes()[:2] == b"\xff\xd8"                  # a JPEG
    note = json.loads((tmp_path / f"{rec['name']}.json").read_text(encoding="utf-8"))
    assert note["shot"] == "eval-03" and note["files"] == rec["files"]
    assert rec["name"].endswith("-250")                           # milliseconds keep names apart
    assert taker.path(rec["name"]) == photo and taker.path("other") is None
    assert taker.summary()["count"] == 1


def test_no_frame_no_photo(tmp_path):
    assert PhotoTaker(_Cam(frame=False), tmp_path).take() is None
    assert list(tmp_path.iterdir()) == []


def test_the_panel_takes_a_photo_and_serves_only_its_own(tmp_path):
    ui = UiState(mode="DRY RUN")
    ui.attach_photos(PhotoTaker(_Cam(), tmp_path))

    class Sup:
        stats = {}

    server = UiServer(ui, Sup(), port=0).start()
    base = f"http://127.0.0.1:{server._srv.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/api/photo", method="POST")
        taken = json.loads(urllib.request.urlopen(req).read())["taken"]
        assert taken["files"]["photo"].endswith(".jpg")
        note = json.loads((tmp_path / f"{taken['name']}.json").read_text(encoding="utf-8"))
        assert note["mode"] == "DRY RUN"
        got = urllib.request.urlopen(base + "/api/photo.jpg?name=" + taken["name"]).read()
        assert got[:2] == b"\xff\xd8"
        for name in ("..%5C..%5Cwin.ini", "nope"):
            try:
                urllib.request.urlopen(base + "/api/photo.jpg?name=" + name)
                raise AssertionError("served a file it did not take")
            except urllib.error.HTTPError as e:
                assert e.code == 404
    finally:
        server.stop()
