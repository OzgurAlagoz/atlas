import argparse
import csv
import os
import time

import cv2
import numpy as np

from atlas.common.convolution import convolve_2d, convolve_2d_fast
from configs.config import OUTPUT

BOX_BLUR = np.ones((3, 3), dtype=np.float32) / 9.0

GAUSSIAN = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]], dtype=np.float32) / 16.0

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

SYMMETRIC_KERNELS = {"box_blur": BOX_BLUR, "gaussian": GAUSSIAN}


def compare(a: np.ndarray, b: np.ndarray) -> tuple:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return round(float(diff.mean()), 4), int(diff.max())


def load_crop(path: str, size: tuple) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Image could not be read: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray[:size[0], :size[1]]


def test_implementations_agree(crop: np.ndarray, rows: list) -> None:
    print("\n--- 1. naive vs fast (same code path, different execution) ---")
    for name, kernel in SYMMETRIC_KERNELS.items():
        start = time.perf_counter()
        naive = convolve_2d(crop, kernel)
        naive_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        fast = convolve_2d_fast(crop, kernel)
        fast_ms = (time.perf_counter() - start) * 1000

        mae, max_diff = compare(naive, fast)
        print(f"{name:<12} mae={mae:<8} max={max_diff:<4} "
              f"naive={naive_ms:7.1f}ms fast={fast_ms:6.2f}ms "
              f"speedup={naive_ms / fast_ms:5.1f}x")
        rows.append({"test": "naive_vs_fast", "kernel": name, "padding": "zero",
                     "mae": mae, "max_diff": max_diff})


def test_symmetric_against_opencv(crop: np.ndarray, rows: list) -> None:
    print("\n--- 2. symmetric kernels vs cv2 (border matched) ---")
    for name, kernel in SYMMETRIC_KERNELS.items():
        mine = convolve_2d_fast(crop, kernel, padding="zero")

        matched = cv2.filter2D(crop, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        default = cv2.filter2D(crop, -1, kernel)

        mae_matched, max_matched = compare(mine, matched)
        mae_default, max_default = compare(mine, default)

        print(f"{name:<12} border matched : mae={mae_matched:<8} max={max_matched}")
        print(f"{'':<12} border default : mae={mae_default:<8} max={max_default}"
              "   <- unfair comparison")
        rows.append({"test": "vs_cv2_border_matched", "kernel": name,
                     "padding": "zero", "mae": mae_matched, "max_diff": max_matched})
        rows.append({"test": "vs_cv2_border_default", "kernel": name,
                     "padding": "zero/reflect", "mae": mae_default,
                     "max_diff": max_default})


def test_convolution_vs_correlation(crop: np.ndarray, rows: list, out_dir: str) -> None:
    print("\n--- 3. asymmetric kernel: convolution vs correlation ---")
    opencv = cv2.filter2D(crop, -1, SOBEL_X, borderType=cv2.BORDER_CONSTANT)

    with_flip = convolve_2d_fast(crop, SOBEL_X, padding="zero")
    without_flip = convolve_2d_fast(crop, SOBEL_X[::-1, ::-1], padding="zero")

    mae_flip, max_flip = compare(with_flip, opencv)
    mae_noflip, max_noflip = compare(without_flip, opencv)

    print(f"convolution (kernel flipped)   vs cv2: mae={mae_flip:<8} max={max_flip}")
    print(f"correlation (flip cancelled)   vs cv2: mae={mae_noflip:<8} max={max_noflip}")
    print("cv2.filter2D performs correlation, so only the second one matches.")

    rows.append({"test": "sobel_convolution_vs_cv2", "kernel": "sobel_x",
                 "padding": "zero", "mae": mae_flip, "max_diff": max_flip})
    rows.append({"test": "sobel_correlation_vs_cv2", "kernel": "sobel_x",
                 "padding": "zero", "mae": mae_noflip, "max_diff": max_noflip})

    panel = np.hstack([crop, with_flip, without_flip, opencv])
    cv2.imwrite(os.path.join(out_dir, "sobel_comparison.png"), panel)

    negatives = (cv2.filter2D(crop.astype(np.float32), -1, SOBEL_X,
                              borderType=cv2.BORDER_CONSTANT) < 0).mean()
    print(f"note: {negatives:.1%} of the raw Sobel response is negative and gets "
          "clipped to 0 by both sides - half the edge information is lost.")


def test_padding_scope(crop: np.ndarray, rows: list) -> None:
    print("\n--- 4. padding affects edges only ---")
    margin = 5
    zero = convolve_2d_fast(crop, BOX_BLUR, padding="zero")
    replicate = convolve_2d_fast(crop, BOX_BLUR, padding="replicate")

    mae_all, max_all = compare(zero, replicate)
    mae_inner, max_inner = compare(zero[margin:-margin, margin:-margin],
                                   replicate[margin:-margin, margin:-margin])

    top_zero = zero[:1, :]
    top_replicate = replicate[:1, :]
    mae_edge, max_edge = compare(top_zero, top_replicate)

    print(f"whole image : mae={mae_all:<8} max={max_all}")
    print(f"interior    : mae={mae_inner:<8} max={max_inner}   <- must be zero")
    print(f"top row     : mae={mae_edge:<8} max={max_edge}")

    rows.append({"test": "padding_whole", "kernel": "box_blur",
                 "padding": "zero_vs_replicate", "mae": mae_all, "max_diff": max_all})
    rows.append({"test": "padding_interior", "kernel": "box_blur",
                 "padding": "zero_vs_replicate", "mae": mae_inner,
                 "max_diff": max_inner})
    rows.append({"test": "padding_top_row", "kernel": "box_blur",
                 "padding": "zero_vs_replicate", "mae": mae_edge, "max_diff": max_edge})


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convolution-validation",
        description="Measure the hand-written convolution against OpenCV.",
    )
    parser.add_argument("--image", default=os.path.join(OUTPUT, "day1", "frame.png"))
    parser.add_argument("--size", nargs=2, type=int, default=(120, 160),
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--out", default=os.path.join(OUTPUT, "day2"))
    args = parser.parse_args()

    crop = load_crop(args.image, tuple(args.size))
    os.makedirs(args.out, exist_ok=True)
    print(f"crop        : {crop.shape}")

    rows = []
    test_implementations_agree(crop, rows)
    test_symmetric_against_opencv(crop, rows)
    test_convolution_vs_correlation(crop, rows, args.out)
    test_padding_scope(crop, rows)

    csv_path = os.path.join(args.out, "validation.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test", "kernel", "padding",
                                                    "mae", "max_diff"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nresults saved -> {csv_path}")


if __name__ == "__main__":
    main()
