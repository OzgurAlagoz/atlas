import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class FrameStats:
    frame: int
    latency_ms: float | None
    interval_ms: float | None
    fps: float | None
    ema: float | None
    stages: dict = field(default_factory=dict)


class Timing():
    def __init__(self, clock=None, alpha=0.2):
        if clock is None:
            self.clock = time.perf_counter
        else:
            self.clock = clock
        self.alpha = alpha
        self.counter = 0
        self.prev_time = None
        self.ema = None
        self.frame_start = None
        self.stages = {}

    def start_frame(self):
        self.frame_start = self.clock()
        self.stages = {}

    @contextmanager
    def stage(self, name):
        start = self.clock()
        try:
            yield
        finally:
            self.stages[name] = (self.clock() - start) * 1000

    def finish_frame(self):
        current_time = self.clock()

        if self.frame_start is None:
            latency_ms = None
        else:
            latency_ms = (current_time - self.frame_start) * 1000

        if self.prev_time is None:
            interval_ms = None
            fps = None
        else:
            elapsed = current_time - self.prev_time
            interval_ms = elapsed * 1000
            if elapsed > 0:
                fps = 1 / elapsed
            else:
                fps = None

        if fps is not None:
            if self.ema is None:
                self.ema = fps
            else:
                self.ema = self.alpha * fps + (1 - self.alpha) * self.ema

        self.prev_time = current_time
        self.counter += 1

        stats = FrameStats(
            frame=self.counter,
            latency_ms=latency_ms,
            interval_ms=interval_ms,
            fps=fps,
            ema=self.ema,
            stages=self.stages,
        )
        self.frame_start = None
        self.stages = {}
        return stats
