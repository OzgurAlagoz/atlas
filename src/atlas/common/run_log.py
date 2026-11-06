import csv
import datetime
import json
import os
import platform
import subprocess

import cv2
import numpy as np

STAGES = ("read", "render", "write", "display")
FIELDS = (["frame", "latency_ms", "interval_ms", "fps", "ema"]
          + [f"{name}_ms" for name in STAGES])


def rounded(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def git_output(args, repo_dir):
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_environment(repo_dir):
    commit = git_output(["rev-parse", "--short", "HEAD"], repo_dir)
    status = git_output(["status", "--porcelain"], repo_dir)

    if status is None:
        dirty = None
    else:
        dirty = status != ""

    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": commit or "unknown",
        "dirty": dirty,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


class RunLogger():
    def __init__(self, run_dir, metadata, warmup=5):
        self.run_dir = run_dir
        self.metadata = metadata
        self.warmup = warmup
        self.records = []

        os.makedirs(run_dir, exist_ok=True)
        self.csv_file = open(os.path.join(run_dir, "frames.csv"), "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=FIELDS)
        self.writer.writeheader()

    def log(self, stats):
        row = {
            "frame": stats.frame,
            "latency_ms": rounded(stats.latency_ms),
            "interval_ms": rounded(stats.interval_ms),
            "fps": rounded(stats.fps, 2),
            "ema": rounded(stats.ema, 2),
        }
        for name in STAGES:
            row[f"{name}_ms"] = rounded(stats.stages.get(name))
        self.writer.writerow(row)
        self.records.append(stats)

    def summarize(self):
        measured = self.records[self.warmup:]
        latencies = [s.latency_ms for s in measured if s.latency_ms is not None]
        intervals = [s.interval_ms for s in measured if s.interval_ms is not None]

        summary = {
            "frames": len(self.records),
            "warmup": self.warmup,
            "measured_frames": len(measured),
            "latency_mean_ms": None,
            "latency_median_ms": None,
            "latency_p95_ms": None,
            "fps_mean": None,
        }

        if latencies:
            summary["latency_mean_ms"] = rounded(np.mean(latencies))
            summary["latency_median_ms"] = rounded(np.median(latencies))
            summary["latency_p95_ms"] = rounded(np.percentile(latencies, 95))

        if intervals and np.mean(intervals) > 0:
            summary["fps_mean"] = rounded(1000.0 / np.mean(intervals), 2)

        for name in STAGES:
            values = [s.stages[name] for s in measured if name in s.stages]
            if values:
                summary[f"{name}_mean_ms"] = rounded(np.mean(values))
            else:
                summary[f"{name}_mean_ms"] = None

        return summary

    def close(self):
        self.csv_file.close()
        summary = self.summarize()
        with open(os.path.join(self.run_dir, "run.json"), "w") as handle:
            json.dump({**self.metadata, "summary": summary}, handle, indent=2)
        return summary
