from atlas.video.source import VideoSource
from configs.config import VIDEO_RES, VIDEO_SOURCE


def test_open_returns_false_for_bad_path():
    video_source = VideoSource("test.mp4", (250, 250))

    result = video_source.open()

    assert result is False
    video_source.close()


def test_read_returns_frame_with_requested_resolution():
    video_source = VideoSource(VIDEO_SOURCE, VIDEO_RES)

    video_source.open()
    frame = video_source.read()

    y = frame.shape[0]
    x = frame.shape[1]

    assert frame is not None
    assert (x, y) == VIDEO_RES

    video_source.close()
