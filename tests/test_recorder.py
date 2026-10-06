import os

import numpy as np

from atlas.video.recorder import Recorder
from configs.config import VIDEO_RES


def test_recorder_opens_and_creates_real_file(tmp_path):
    recorder = Recorder(30, VIDEO_RES, "mp4v")
    mp4_path = os.path.join(tmp_path, "output.mp4")

    open_result = recorder.open(mp4_path)
    recorder.write_frame(np.zeros((VIDEO_RES[1], VIDEO_RES[0], 3), dtype=np.uint8))
    recorder.close()

    file_check = os.path.exists(mp4_path)
    file_size = os.path.getsize(mp4_path)

    assert open_result is True
    assert file_check is True
    assert file_size > 0
