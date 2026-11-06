import csv
import json
import os

import pytest

from atlas.common.run_log import RunLogger, collect_environment
from atlas.common.timing import FrameStats


def make_stats(frame, latency, interval=None, stages=None):
    if stages is None:
        stages = {"read": latency / 2, "render": latency / 2}
    if interval is None:
        fps = None
    else:
        fps = 1000.0 / interval
    return FrameStats(frame=frame, latency_ms=latency, interval_ms=interval,
                      fps=fps, ema=fps, stages=stages)


def read_rows(run_dir):
    with open(os.path.join(run_dir, "frames.csv"), newline="") as handle:
        return list(csv.DictReader(handle))


def test_logger_writes_one_row_per_frame(tmp_path):
    run_dir = str(tmp_path / "run")
    logger = RunLogger(run_dir, {"source": "test"}, warmup=0)
    for frame in range(1, 4):
        logger.log(make_stats(frame, 10.0, 10.0))
    logger.close()

    rows = read_rows(run_dir)

    assert len(rows) == 3
    assert rows[0]["frame"] == "1"
    assert rows[0]["read_ms"] == "5.0"


def test_missing_stage_is_empty_cell(tmp_path):
    run_dir = str(tmp_path / "run")
    logger = RunLogger(run_dir, {}, warmup=0)
    logger.log(make_stats(1, 10.0, 10.0))
    logger.close()

    rows = read_rows(run_dir)

    assert rows[0]["write_ms"] == ""
    assert rows[0]["display_ms"] == ""


def test_summary_skips_warmup_frames(tmp_path):
    logger = RunLogger(str(tmp_path / "run"), {}, warmup=2)
    logger.log(make_stats(1, 100.0))
    logger.log(make_stats(2, 100.0, 100.0))
    for frame in range(3, 6):
        logger.log(make_stats(frame, 10.0, 10.0))

    summary = logger.close()

    assert summary["frames"] == 5
    assert summary["measured_frames"] == 3
    assert summary["latency_mean_ms"] == pytest.approx(10.0)
    assert summary["fps_mean"] == pytest.approx(100.0)


def test_summary_reports_p95(tmp_path):
    logger = RunLogger(str(tmp_path / "run"), {}, warmup=0)
    for frame in range(1, 101):
        logger.log(make_stats(frame, float(frame)))

    summary = logger.close()

    assert summary["latency_median_ms"] == pytest.approx(50.5)
    assert summary["latency_p95_ms"] == pytest.approx(95.05)


def test_run_json_contains_metadata_and_summary(tmp_path):
    run_dir = str(tmp_path / "run")
    logger = RunLogger(run_dir, {"source": "test.mp4", "display": False}, warmup=0)
    logger.log(make_stats(1, 10.0, 10.0))
    logger.close()

    with open(os.path.join(run_dir, "run.json")) as handle:
        data = json.load(handle)

    assert data["source"] == "test.mp4"
    assert data["display"] is False
    assert data["summary"]["frames"] == 1


def test_environment_outside_git_repo(tmp_path):
    environment = collect_environment(str(tmp_path))

    assert environment["commit"] == "unknown"
    assert environment["dirty"] is None


def test_environment_has_versions(tmp_path):
    environment = collect_environment(str(tmp_path))

    for key in ("date", "python", "opencv", "numpy", "platform"):
        assert environment[key]
