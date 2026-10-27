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


GRAY_WEIGHTS = {"R": 0.299, "G": 0.587, "B": 0.114}


def report_colorspaces(img: np.ndarray) -> tuple:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blue, green, red = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    manual = (GRAY_WEIGHTS["R"] * red.astype(np.float32)
              + GRAY_WEIGHTS["G"] * green.astype(np.float32)
              + GRAY_WEIGHTS["B"] * blue.astype(np.float32))
    manual = np.clip(np.round(manual), 0, 255).astype(np.uint8)

    diff = np.abs(manual.astype(np.int16) - gray.astype(np.int16))
    pixel_bgr = img[0, 0].tolist()
    pixel_rgb = rgb[0, 0].tolist()

    print("\n--- colour spaces ---")
    print(f"{'variant':<10}{'shape':<20}{'nbytes':>12}")
    for name, array in (("BGR", img), ("RGB", rgb), ("GRAY", gray)):
        print(f"{name:<10}{str(array.shape):<20}{array.nbytes:>12}")
    print(f"pixel[0,0] BGR {pixel_bgr} -> RGB {pixel_rgb}  (first and last swap)")
    print(f"gray weights : {GRAY_WEIGHTS}")
    print(f"manual vs cvtColor: mae={diff.mean():.4f}  max={int(diff.max())}")

    report = {
        "shapes": {"bgr": list(img.shape), "rgb": list(rgb.shape),
                   "gray": list(gray.shape)},
        "nbytes": {"bgr": int(img.nbytes), "gray": int(gray.nbytes)},
        "pixel_0_0": {"bgr": pixel_bgr, "rgb": pixel_rgb},
        "gray_weights": GRAY_WEIGHTS,
        "manual_gray_mae": round(float(diff.mean()), 4),
        "manual_gray_max": int(diff.max()),
    }
    return report, gray


def report_overflow(img: np.ndarray, amount: int = 100) -> tuple:
    numpy_added = img + np.uint8(amount)
    opencv_added = cv2.add(img, np.full_like(img, amount))

    wrapped = (img.astype(np.int16) + amount) > 255
    ratio = float(wrapped.mean())

    brightness = img.astype(np.int32).sum(axis=2)
    y, x = np.unravel_index(int(brightness.argmax()), brightness.shape)

    print(f"\n--- uint8 overflow (+{amount}) ---")
    print(f"values that wrap around : {int(wrapped.sum())} of {img.size} ({ratio:.1%})")
    print(f"brightest pixel [{y},{x}]")
    print(f"  original      : {img[y, x].tolist()}")
    print(f"  numpy  img+{amount} : {numpy_added[y, x].tolist()}  (wraps past 255)")
    print(f"  cv2.add       : {opencv_added[y, x].tolist()}  (saturates at 255)")

    report = {
        "amount": amount,
        "wrapped_values": int(wrapped.sum()),
        "wrapped_ratio": round(ratio, 4),
        "brightest_pixel": [int(y), int(x)],
        "original": img[y, x].tolist(),
        "numpy_add": numpy_added[y, x].tolist(),
        "cv2_add": opencv_added[y, x].tolist(),
    }
    return report, numpy_added, opencv_added


def report_crop_view(img: np.ndarray) -> tuple:
    height, width = img.shape[:2]
    y0, x0 = height // 4, width // 4
    y1, x1 = y0 + height // 2, x0 + width // 2

    crop = img[y0:y1, x0:x1]

    sample = img.copy()
    view = sample[y0:y1, x0:x1]
    before = int(sample[y0, x0, 0])
    view[0, 0, 0] = 255 - before
    after_view = int(sample[y0, x0, 0])

    sample = img.copy()
    copied = sample[y0:y1, x0:x1].copy()
    copied[0, 0, 0] = 255 - before
    after_copy = int(sample[y0, x0, 0])

    print("\n--- crop: view or copy ---")
    print(f"crop region   : rows {y0}:{y1}, cols {x0}:{x1}  ->  shape {crop.shape}")
    print(f"shares memory : slice={np.shares_memory(crop, img)}  "
          f"copy={np.shares_memory(crop.copy(), img)}")
    print(f"edit through slice  : original {before} -> {after_view}  (changed)")
    print(f"edit through .copy(): original {before} -> {after_copy}  (untouched)")

    report = {
        "region": {"rows": [y0, y1], "cols": [x0, x1]},
        "crop_shape": list(crop.shape),
        "slice_shares_memory": bool(np.shares_memory(crop, img)),
        "copy_shares_memory": bool(np.shares_memory(crop.copy(), img)),
        "original_before": before,
        "after_slice_edit": after_view,
        "after_copy_edit": after_copy,
    }
    return report, crop


def build_panel(images: list, labels: list, out_dir: str,
                tile_height: int = 270, columns: int = 3) -> str:
    tiles = []
    for image, text in zip(images, labels):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        scale = tile_height / image.shape[0]
        tile_width = int(round(image.shape[1] * scale))
        tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile_width, 28), (0, 0, 0), -1)
        cv2.putText(tile, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    widest = max(tile.shape[1] for tile in tiles)
    padded = [cv2.copyMakeBorder(tile, 0, 0, 0, widest - tile.shape[1],
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
              for tile in tiles]
    while len(padded) % columns:
        padded.append(np.zeros_like(padded[0]))

    rows = [np.hstack(padded[i:i + columns]) for i in range(0, len(padded), columns)]
    panel = np.vstack(rows)

    panel_path = os.path.join(out_dir, "panel.png")
    if not cv2.imwrite(panel_path, panel):
        raise RuntimeError(f"Panel could not be written to: {panel_path}")
    print(f"\npanel saved -> {panel_path}")
    return panel_path


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

    colorspaces, gray = report_colorspaces(img)
    overflow, numpy_added, opencv_added = report_overflow(img)
    crop_view, crop = report_crop_view(img)

    height, width = img.shape[:2]
    roundtrip = cv2.resize(
        cv2.resize(img, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST),
        (width, height), interpolation=cv2.INTER_NEAREST,
    )

    metrics = {
        "source": args.source,
        "anatomy": report_anatomy(img),
        "colorspaces": colorspaces,
        "overflow": overflow,
        "crop_view": crop_view,
        "resize_benchmark": resize_benchmark(img, args.out),
        "normalisation": normalize_report(img),
    }

    build_panel(
        [img, gray, crop, numpy_added, opencv_added, roundtrip],
        ["original (BGR)", "grayscale", "center crop",
         "numpy +100 (wraps)", "cv2.add +100 (saturates)", "resize round-trip (nearest)"],
        args.out,
    )

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
