import time

import pytest

from atlas.common.timing import FrameStats, Timing


def test_first_frame_returns():
    times = iter([10.0, 10.04, 10.06])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    results = test_frame.finish_frame()

    assert results.frame == 1
    assert results.latency_ms is None
    assert results.interval_ms is None
    assert results.fps is None
    assert results.ema is None
    assert test_frame.counter == 1


def test_second_frame_measures_interval_and_fps():
    times = iter([10.0, 10.04])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.finish_frame()
    results = test_frame.finish_frame()

    assert results.interval_ms == pytest.approx(40.0)
    assert results.fps == pytest.approx(25.0)
    assert results.ema == pytest.approx(25.0)


def test_counter_increments_every_frame():
    times = iter([0.0, 0.01, 0.02, 0.03])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    for _ in range(4):
        results = test_frame.finish_frame()

    assert results.frame == 4
    assert test_frame.counter == 4


def test_ema_reacts_slower_than_instant_fps():
    times = iter([0.0, 0.01, 0.02, 0.04])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.finish_frame()
    test_frame.finish_frame()
    test_frame.finish_frame()
    results = test_frame.finish_frame()

    assert results.fps == pytest.approx(50.0)
    assert results.ema == pytest.approx(90.0)
    assert results.fps < results.ema < 100.0


def test_same_timestamp_gives_no_fps():
    times = iter([5.0, 5.0])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.finish_frame()
    results = test_frame.finish_frame()

    assert results.interval_ms == 0.0
    assert results.fps is None
    assert results.ema is None
    assert test_frame.counter == 2


def test_ema_keeps_last_value_when_fps_missing():
    times = iter([0.0, 0.01, 0.01])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.finish_frame()
    test_frame.finish_frame()
    results = test_frame.finish_frame()

    assert results.fps is None
    assert results.ema == pytest.approx(100.0)


def test_default_clock_is_perf_counter():
    test_frame = Timing()

    assert test_frame.clock is time.perf_counter


def test_alpha_changes_ema_weight():
    times = iter([0.0, 0.01, 0.03])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock, alpha=0.5)
    test_frame.finish_frame()
    test_frame.finish_frame()
    results = test_frame.finish_frame()

    assert results.fps == pytest.approx(50.0)
    assert results.ema == pytest.approx(75.0)


def test_latency_measured_from_start_to_finish():
    times = iter([0.0, 0.015])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.start_frame()
    results = test_frame.finish_frame()

    assert results.latency_ms == pytest.approx(15.0)


def test_latency_is_not_interval():
    times = iter([0.0, 0.010, 0.030, 0.040])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.start_frame()
    test_frame.finish_frame()
    test_frame.start_frame()
    results = test_frame.finish_frame()

    assert results.latency_ms == pytest.approx(10.0)
    assert results.interval_ms == pytest.approx(30.0)
    assert results.fps == pytest.approx(1000.0 / 30.0)


def test_stages_are_measured_separately():
    times = iter([0.0, 0.0, 0.004, 0.004, 0.010, 0.011])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.start_frame()
    with test_frame.stage("read"):
        pass
    with test_frame.stage("render"):
        pass
    results = test_frame.finish_frame()

    assert results.stages == pytest.approx({"read": 4.0, "render": 6.0})
    assert results.latency_ms == pytest.approx(11.0)


def test_stages_reset_every_frame():
    times = iter([0.0, 0.0, 0.002, 0.003, 0.010, 0.012])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.start_frame()
    with test_frame.stage("read"):
        pass
    test_frame.finish_frame()
    test_frame.start_frame()
    results = test_frame.finish_frame()

    assert results.stages == {}


def test_previous_stats_not_changed_by_next_frame():
    times = iter([0.0, 0.0, 0.002, 0.003, 0.010, 0.010, 0.015, 0.016])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.start_frame()
    with test_frame.stage("read"):
        pass
    first = test_frame.finish_frame()
    test_frame.start_frame()
    with test_frame.stage("render"):
        pass
    second = test_frame.finish_frame()

    assert list(first.stages) == ["read"]
    assert list(second.stages) == ["render"]
    assert first.stages["read"] == pytest.approx(2.0)


def test_stage_recorded_even_on_error():
    times = iter([0.0, 0.0, 0.005, 0.006])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.start_frame()
    with pytest.raises(RuntimeError):
        with test_frame.stage("read"):
            raise RuntimeError("okuma hatasi")
    results = test_frame.finish_frame()

    assert results.stages == pytest.approx({"read": 5.0})


def test_stage_outside_frame_is_discarded():
    times = iter([0.0, 0.001, 0.002, 0.008, 0.010, 0.012])

    def fake_clock():
        return next(times)

    test_frame = Timing(fake_clock)
    test_frame.start_frame()
    test_frame.finish_frame()
    with test_frame.stage("wait"):
        pass
    test_frame.start_frame()
    results = test_frame.finish_frame()

    assert results.stages == {}


def test_finish_frame_returns_frame_stats():
    test_frame = Timing()

    results = test_frame.finish_frame()

    assert isinstance(results, FrameStats)
