import os

import cv2
import numpy as np

from atlas.main import parse_args, save_snapshot


def test_save_snapshot_writes_png(tmp_path):
    frame = np.full((54, 78, 3), 120, dtype=np.uint8)

    path = save_snapshot(frame, 12, str(tmp_path))

    assert os.path.exists(path)
    assert "frame00012" in os.path.basename(path)
    assert cv2.imread(path).shape == frame.shape


def test_parse_args_defaults():
    args = parse_args([])

    assert args.no_display is False
    assert args.save is False
    assert args.warmup == 5
    assert args.run_dir is None


def test_parse_args_no_display_and_resolution():
    args = parse_args(["--no-display", "--res", "320", "180"])

    assert args.no_display is True
    assert args.res == [320, 180]
