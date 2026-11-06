import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

from configs.config import OUTPUT

PROJECT_ROOT = os.path.dirname(OUTPUT)

SCENARIOS = [
    {"name": "display_780x540", "res": (780, 540), "display": True},
    {"name": "headless_780x540", "res": (780, 540), "display": False},
    {"name": "headless_1920x1080", "res": (1920, 1080), "display": False},
    {"name": "headless_320x180", "res": (320, 180), "display": False},
]

METRICS = [
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "fps_mean",
    "read_mean_ms",
    "render_mean_ms",
    "display_mean_ms",
]


def run_once(scenario, run_dir, source):
    width, height = scenario["res"]
    command = [sys.executable, "-m", "atlas.main",
               "--res", str(width), str(height),
               "--run-dir", run_dir]
    if source is not None:
        command += ["--source", source]
    if not scenario["display"]:
        command.append("--no-display")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(PROJECT_ROOT, "src"), PROJECT_ROOT])

    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{scenario['name']} failed:\n{result.stderr[-800:]}")

    with open(os.path.join(run_dir, "run.json")) as handle:
        return json.load(handle)["summary"]


def average(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(statistics.mean(values), 3)


def print_table(rows):
    header = (f"{'scenario':<20}{'lat mean':>10}{'lat med':>9}{'lat p95':>9}"
              f"{'fps':>9}{'read':>8}{'render':>8}{'display':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = [row[metric] if row[metric] is not None else "--" for metric in METRICS]
        print(f"{row['scenario']:<20}{cells[0]:>10}{cells[1]:>9}{cells[2]:>9}"
              f"{cells[3]:>9}{cells[4]:>8}{cells[5]:>8}{cells[6]:>9}")


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline-benchmark",
        description="Run atlas.main under several scenarios and compare the timings.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source", default=None)
    parser.add_argument("--skip-display", action="store_true",
                        help="leave out scenarios that open a window")
    parser.add_argument("--out", default=os.path.join(OUTPUT, "day3"))
    args = parser.parse_args()

    scenarios = [s for s in SCENARIOS if not (args.skip_display and s["display"])]
    run_rows, summary_rows = [], []

    for scenario in scenarios:
        results = []
        for repeat in range(1, args.repeats + 1):
            run_dir = os.path.join(args.out, "runs", f"{scenario['name']}_r{repeat}")
            print(f"running {scenario['name']} ({repeat}/{args.repeats})")
            summary = run_once(scenario, run_dir, args.source)
            results.append(summary)
            run_rows.append({"scenario": scenario["name"], "repeat": repeat,
                             **{metric: summary.get(metric) for metric in METRICS}})

        summary_rows.append({"scenario": scenario["name"], "repeats": args.repeats,
                             **{metric: average(r.get(metric) for r in results)
                                for metric in METRICS}})

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "benchmark_runs.csv"), run_rows,
              ["scenario", "repeat"] + METRICS)
    write_csv(os.path.join(args.out, "benchmark.csv"), summary_rows,
              ["scenario", "repeats"] + METRICS)

    print(f"\naverage of {args.repeats} runs per scenario (ms unless noted)\n")
    print_table(summary_rows)
    print(f"\ncsv saved -> {os.path.join(args.out, 'benchmark.csv')}")


if __name__ == "__main__":
    main()
