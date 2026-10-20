import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from configs.config import OUTPUT, VIDEO_SOURCE

INTERPOLATIONS = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_CUBIC": cv2.INTER_CUBIC,
}


def extract_frame(source: str, out_dir: str) -> str:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Video source could not be opened: {source}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"No frame could be read from: {source}")

    os.makedirs(out_dir, exist_ok=True)
    frame_path = os.path.join(out_dir, "frame.png")
    if not cv2.imwrite(frame_path, frame):
        raise RuntimeError(f"Frame could not be written to: {frame_path}")

    print(f"frame saved -> {frame_path}")
    return frame_path


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Image could not be read: {path}")
    return img


def report_anatomy(img: np.ndarray) -> dict:
    height, width, channels = img.shape

    per_channel = {
        "min": img.min(axis=(0, 1)).tolist(),
        "max": img.max(axis=(0, 1)).tolist(),
        "mean": img.mean(axis=(0, 1)).round(2).tolist(),
        "std": img.std(axis=(0, 1)).round(2).tolist(),
    }

    print("\n--- anatomy ---")
    print(f"shape      : {img.shape}  (height={height}, width={width}, channels={channels})")
    print(f"ndim       : {img.ndim}")
    print(f"dtype      : {img.dtype}")
    print(f"size       : {img.size} values")
    print(f"nbytes     : {img.nbytes} bytes ({img.nbytes / 1024 / 1024:.2f} MB)")
    print(f"pixel[0,0] : {img[0, 0]}  (B, G, R order)")
    print(f"per channel (B, G, R):")
    for key, values in per_channel.items():
        print(f"  {key:5}: {values}")

    return {
        "shape": list(img.shape),
        "ndim": int(img.ndim),
        "dtype": str(img.dtype),
        "size": int(img.size),
        "nbytes": int(img.nbytes),
        "pixel_0_0": img[0, 0].tolist(),
        "per_channel": per_channel,
    }


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def peak_signal_noise_ratio(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def resize_benchmark(img: np.ndarray, out_dir: str) -> list:
    height, width = img.shape[:2]
    half = (width // 2, height // 2)
    full = (width, height)

    rows = []
    for name, flag in INTERPOLATIONS.items():
        start = time.perf_counter()
        small = cv2.resize(img, half, interpolation=flag)
        restored = cv2.resize(small, full, interpolation=flag)
        elapsed_ms = (time.perf_counter() - start) * 1000

        rows.append({
            "method": name,
            "mae": round(mean_absolute_error(img, restored), 4),
            "psnr_manual": round(peak_signal_noise_ratio(img, restored), 4),
            "psnr_cv2": round(float(cv2.PSNR(img, restored)), 4),
            "elapsed_ms": round(elapsed_ms, 3),
        })

    csv_path = os.path.join(out_dir, "resize_benchmark.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- resize round-trip (half then back) ---")
    print(f"{'method':<15}{'mae':>10}{'psnr(manual)':>15}{'psnr(cv2)':>12}{'ms':>9}")
    for row in rows:
        print(f"{row['method']:<15}{row['mae']:>10}{row['psnr_manual']:>15}"
              f"{row['psnr_cv2']:>12}{row['elapsed_ms']:>9}")
    print(f"csv saved -> {csv_path}")

    return rows


def normalize_report(img: np.ndarray) -> dict:
    as_float64 = (img - img.min()) / (img.max() - img.min())
    as_float32 = as_float64.astype(np.float32)

    report = {
        "uint8": {
            "dtype": str(img.dtype),
            "min": int(img.min()),
            "max": int(img.max()),
            "nbytes": int(img.nbytes),
        },
        "float64": {
            "dtype": str(as_float64.dtype),
            "min": float(as_float64.min()),
            "max": float(as_float64.max()),
            "nbytes": int(as_float64.nbytes),
        },
        "float32": {
            "dtype": str(as_float32.dtype),
            "min": float(as_float32.min()),
            "max": float(as_float32.max()),
            "nbytes": int(as_float32.nbytes),
        },
    }

    print("\n--- normalisation ---")
    print(f"{'variant':<10}{'dtype':<10}{'min':>8}{'max':>8}{'nbytes':>12}{'x uint8':>10}")
    base = report["uint8"]["nbytes"]
    for variant, values in report.items():
        print(f"{variant:<10}{values['dtype']:<10}{values['min']:>8.2f}"
              f"{values['max']:>8.2f}{values['nbytes']:>12}{values['nbytes'] / base:>10.1f}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="image-anatomy",
        description="Inspect a single video frame as a NumPy array.",
    )
    parser.add_argument("--source", default=VIDEO_SOURCE,
                        help="video file to grab the frame from")
    parser.add_argument("--out", default=os.path.join(OUTPUT, "day1"),
                        help="directory the artefacts are written to")
    parser.add_argument("--show", action="store_true",
                        help="display the frame in a window (needs a GUI)")
    args = parser.parse_args()

    frame_path = extract_frame(args.source, args.out)
    img = load_image(frame_path)

    metrics = {
        "source": args.source,
        "anatomy": report_anatomy(img),
        "resize_benchmark": resize_benchmark(img, args.out),
        "normalisation": normalize_report(img),
    }

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\nmetrics saved -> {metrics_path}")

    if args.show:
        cv2.imshow("frame", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
