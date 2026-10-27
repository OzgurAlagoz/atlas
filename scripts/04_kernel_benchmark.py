import argparse
import csv
import os
import statistics
import time

import cv2
import numpy as np

from atlas.common.convolution import convolve_2d, convolve_2d_fast
from configs.config import OUTPUT

KERNELS = {
    "identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]], dtype=np.float32),
    "box_blur": np.ones((3, 3), dtype=np.float32) / 9.0,
    "gaussian": np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]], dtype=np.float32) / 16.0,
    "sharpen": np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32),
    "sobel_x": np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]], dtype=np.float32),
}


def load_crop(path: str, size: tuple) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Image could not be read: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray[:size[0], :size[1]]


def median_ms(func, repeats: int) -> tuple:
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings), result


def compare(a: np.ndarray, b: np.ndarray) -> tuple:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return round(float(diff.mean()), 4), int(diff.max())


def benchmark(crop: np.ndarray, repeats: int) -> tuple:
    rows, outputs = [], {}
    for name, kernel in KERNELS.items():
        flipped = np.ascontiguousarray(kernel[::-1, ::-1])

        naive_ms, naive = median_ms(lambda: convolve_2d(crop, kernel), repeats)
        fast_ms, fast = median_ms(lambda: convolve_2d_fast(crop, kernel), repeats)
        cv2_ms, reference = median_ms(
            lambda: cv2.filter2D(crop, -1, flipped, borderType=cv2.BORDER_CONSTANT),
            repeats,
        )

        mae_impl, max_impl = compare(naive, fast)
        mae_cv2, max_cv2 = compare(fast, reference)

        rows.append({
            "kernel": name,
            "symmetric": bool(np.array_equal(kernel, flipped)),
            "kernel_sum": round(float(kernel.sum()), 3),
            "naive_ms": round(naive_ms, 3),
            "fast_ms": round(fast_ms, 3),
            "cv2_ms": round(cv2_ms, 4),
            "fast_vs_naive": round(naive_ms / fast_ms, 1),
            "cv2_vs_fast": round(fast_ms / cv2_ms, 1),
            "naive_vs_fast_mae": mae_impl,
            "mae_vs_cv2": mae_cv2,
            "max_vs_cv2": max_cv2,
        })
        outputs[name] = fast
    return rows, outputs


def print_table(rows: list) -> None:
    header = (f"{'kernel':<10}{'sym':>5}{'sum':>6}{'naive ms':>10}{'fast ms':>9}"
              f"{'cv2 ms':>9}{'fast/naive':>11}{'cv2/fast':>9}{'mae cv2':>9}{'max':>5}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['kernel']:<10}{str(row['symmetric'])[0]:>5}{row['kernel_sum']:>6}"
              f"{row['naive_ms']:>10}{row['fast_ms']:>9}{row['cv2_ms']:>9}"
              f"{row['fast_vs_naive']:>10}x{row['cv2_vs_fast']:>8}x"
              f"{row['mae_vs_cv2']:>9}{row['max_vs_cv2']:>5}")


def save_panel(crop: np.ndarray, outputs: dict, out_dir: str) -> str:
    tiles = [("original", crop)] + list(outputs.items())
    labelled = []
    for text, image in tiles:
        tile = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2),
                          interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(tile, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        labelled.append(tile)
    while len(labelled) % 4:
        labelled.append(np.zeros_like(labelled[0]))
    rows = [np.hstack(labelled[i:i + 4]) for i in range(0, len(labelled), 4)]
    panel_path = os.path.join(out_dir, "kernel_panel.png")
    if not cv2.imwrite(panel_path, np.vstack(rows)):
        raise RuntimeError(f"Panel could not be written to: {panel_path}")
    return panel_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kernel-benchmark",
        description="Time six kernels across the naive, vectorised and OpenCV paths.",
    )
    parser.add_argument("--image", default=os.path.join(OUTPUT, "day1", "frame.png"))
    parser.add_argument("--size", nargs=2, type=int, default=(120, 160),
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--repeats", type=int, default=5,
                        help="runs per measurement, the median is reported")
    parser.add_argument("--out", default=os.path.join(OUTPUT, "day2"))
    args = parser.parse_args()

    crop = load_crop(args.image, tuple(args.size))
    os.makedirs(args.out, exist_ok=True)
    print(f"crop {crop.shape}, {args.repeats} repeats per measurement (median)\n")

    rows, outputs = benchmark(crop, args.repeats)
    print_table(rows)

    csv_path = os.path.join(args.out, "kernel_benchmark.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    panel_path = save_panel(crop, outputs, args.out)
    print(f"\ncsv saved   -> {csv_path}")
    print(f"panel saved -> {panel_path}")


if __name__ == "__main__":
    main()
