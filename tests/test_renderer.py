import numpy as np

from atlas.common.timing import FrameStats
from atlas.visualization.renderer import Renderer


def make_frame(height=540, width=780):
    return np.full((height, width, 3), 200, dtype=np.uint8)


def make_stats():
    return FrameStats(frame=44, latency_ms=3.6789, interval_ms=3.9,
                      fps=256.4, ema=259.449, stages={})


def test_lines_before_first_stats():
    renderer = Renderer("test", 29.97002997, 127)

    lines = renderer.build_lines(make_frame(), 1)

    assert lines == [
        "Frame: 1/127",
        "Input FPS: 30",
        "Processing FPS: --",
        "Latency: --",
        "Resolution: 780x540",
    ]


def test_lines_are_formatted():
    renderer = Renderer("test", 29.97002997, 127)

    lines = renderer.build_lines(make_frame(), 45, make_stats())

    assert lines[0] == "Frame: 45/127"
    assert lines[2] == "Processing FPS: 259"
    assert lines[3] == "Latency: 3.7 ms"


def test_resolution_is_width_by_height():
    renderer = Renderer("test", 30, 127)

    lines = renderer.build_lines(make_frame(height=180, width=320), 1)

    assert lines[4] == "Resolution: 320x180"


def test_unknown_frame_count_shows_only_frame_number():
    renderer = Renderer("test", 30, 0)

    lines = renderer.build_lines(make_frame(), 7)

    assert lines[0] == "Frame: 7"


def test_render_draws_on_frame_in_place():
    renderer = Renderer("test", 30, 127)
    frame = make_frame()
    original = frame.copy()

    result = renderer.render(frame, 1, make_stats())

    assert result is frame
    assert frame.shape == original.shape
    assert not np.array_equal(frame, original)
