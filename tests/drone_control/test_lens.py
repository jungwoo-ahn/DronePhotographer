"""The ultra-wide picture becomes a training-like one, on images whose geometry we made ourselves."""

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from src.drone_control.camera import Frame
from src.drone_control.lens import (
    TRAINING_HFOV_DEG,
    VIEW_SIZE,
    LensView,
    hfov_from_tilt,
    image_shift,
)
from src.drone_control.sources import ActionSequenceSource

W, H, HFOV = 1280, 720, 160.0
PER_RADIAN = W / math.radians(HFOV)


def spot_at(azimuth_deg, elevation_deg=0.0):
    """A wide frame with one bright spot, in the direction given, drawn by the lens model."""
    ray = np.array([math.tan(math.radians(azimuth_deg)), -math.tan(math.radians(elevation_deg)), 1.0])
    angle = math.atan2(math.hypot(ray[0], ray[1]), 1.0)
    r = PER_RADIAN * angle / max(math.hypot(ray[0], ray[1]), 1e-9)
    image = np.zeros((H, W, 3), np.uint8)
    # Pixel i covers i..i+1, so the pixel holding a point is the point less a half, rounded.
    cv2.circle(image, (round(W / 2 + ray[0] * r - 0.5), round(H / 2 + ray[1] * r - 0.5)), 3,
               (255, 255, 255), -1)
    return image


def found_at(view):
    ys, xs = np.nonzero(view[:, :, 0] > 100)
    return xs.mean() + 0.5, ys.mean() + 0.5


def narrow(source_hfov=HFOV):
    return LensView(source_hfov, size=VIEW_SIZE, view_hfov_deg=TRAINING_HFOV_DEG, k1=0.0)


def test_by_default_it_is_4_by_3_and_uses_the_full_height():
    assert LensView(HFOV, k1=0.0)(np.zeros((H, W, 3), np.uint8)).shape == (720, 960, 3)
    # A spot just inside the frame's top edge is still in the picture, near its top.
    edge = math.degrees(360.0 / PER_RADIAN)
    x, y = found_at(LensView(HFOV, k1=0.0)(spot_at(0.0, edge - 2.0)))
    assert x == pytest.approx(480, abs=2.0) and y < 40


def test_the_lens_curve_puts_the_frame_side_at_the_data_sheets_80_degrees():
    assert math.degrees(LensView().angle_at(640.0, 1280)) == pytest.approx(80.0, abs=0.5)
    assert math.degrees(LensView().angle_at(1.0, 1280)) == pytest.approx(140.0 / 1280, abs=0.001)


def test_the_full_view_has_no_empty_edges_on_the_real_lens():
    view = LensView()(np.full((H, W, 3), 255, np.uint8))
    assert view[1:-1, 1:-1].min() > 0


def test_by_default_a_straight_line_comes_out_straight():
    points = [(a, 15.0) for a in (-30.0, -15.0, 0.0, 15.0, 30.0)]    # one horizontal line in the scene
    ys = [found_at(LensView(HFOV, k1=0.0)(spot_at(a, e)))[1] for a, e in points]
    assert max(ys) - min(ys) < 2.0


def test_the_narrow_view_has_the_renders_field_of_view_and_shape():
    assert TRAINING_HFOV_DEG == pytest.approx(29.86, abs=0.01)
    assert VIEW_SIZE[0] / VIEW_SIZE[1] == pytest.approx(1024 / 768)
    assert narrow()(np.zeros((H, W, 3), np.uint8)).shape == (384, 512, 3)


@pytest.mark.parametrize("azimuth,elevation", [(0, 0), (10, 0), (-13, 0), (0, 8), (9, -7)])
def test_a_direction_lands_where_a_pinhole_render_would_put_it(azimuth, elevation):
    focal = VIEW_SIZE[0] / (2 * math.tan(math.radians(TRAINING_HFOV_DEG / 2)))
    x, y = found_at(narrow()(spot_at(azimuth, elevation)))
    assert x == pytest.approx(VIEW_SIZE[0] / 2 + focal * math.tan(math.radians(azimuth)), abs=2.0)
    assert y == pytest.approx(VIEW_SIZE[1] / 2 - focal * math.tan(math.radians(elevation)), abs=2.0)


def test_what_is_outside_the_view_is_gone():
    assert narrow()(spot_at(40.0)).max() == 0


def test_a_narrow_picture_refuses_rather_than_zooming_out():
    with pytest.raises(ValueError):
        narrow(20.0)


def test_a_new_resolution_gets_its_own_map():
    view = LensView(HFOV, k1=0.0)
    view(np.zeros((H, W, 3), np.uint8))
    assert view(np.zeros((1080, 1920, 3), np.uint8)).shape == (1080, 1440, 3)


def test_it_measures_how_far_the_picture_moved():
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 2)
    moved = np.roll(noise, 32, axis=0)
    dx, dy, confidence = image_shift(noise, moved)
    assert abs(dx) < 0.5 and dy == pytest.approx(32, abs=0.5) and confidence > 0.1


def test_a_lens_of_known_width_is_recovered_from_a_tilt():
    shift = PER_RADIAN * math.radians(4.0)
    assert hfov_from_tilt(-shift, 4.0, W) == pytest.approx(HFOV)


class FakeCamera:
    def __init__(self, image):
        self.image = image

    def fresh_frame(self):
        return Frame(image=self.image)


class FakeClient:
    def request_actions(self, jpeg, goal, timeout_s=None):
        self.sent = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return [[0.0] * 10]


def test_the_server_is_sent_the_corrected_view():
    client = FakeClient()
    source = ActionSequenceSource(FakeCamera(np.zeros((H, W, 3), np.uint8)), client, {}, move_s=1.0,
                                  view=narrow())
    assert source.check(1.0)
    assert client.sent.shape == (384, 512, 3)


def test_without_a_view_the_frame_goes_as_it_is():
    client = FakeClient()
    source = ActionSequenceSource(FakeCamera(np.zeros((H, W, 3), np.uint8)), client, {}, move_s=1.0)
    assert source.check(1.0)
    assert client.sent.shape == (H, W, 3)
