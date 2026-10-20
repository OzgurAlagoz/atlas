"""Apply hand-written convolution to a frame - week 2, day 2.

Runs the naive and the vectorised implementation on a grayscale crop, compares
both padding strategies against OpenCV, and saves a side-by-side panel.

    python scripts/02_image_convolution_and_padding.py
    python scripts/02_image_convolution_and_padding.py --size 240 320 --show
"""

import argparse
import os

import cv2
import numpy as np

from atlas.common.convolution import convolve_2d, convolve_2d_fast
from configs.config import OUTPUT

# Box blur: every weight equal, divided by the count so the sum is 1 and the
# overall brightness of the image is preserved.
BOX_BLUR = np.ones((3, 3), dtype=np.float32) / 9.0


def load_grayscale(path: str, size: tuple) -> np.ndarray:
    """Read the frame, drop the colour channels, keep a small corner of it."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Image could not be read: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = size
    crop = gray[:height, :width]

    print(f"source      : {path}")
    print(f"colour shape: {img.shape}  ->  grayscale shape: {gray.shape}")
    print(f"working crop: {crop.shape}  (naive loop is slow on full frames)")
    return crop


def label(image: np.ndarray, text: str) -> np.ndarray:
    """Write a caption on a copy of the image, leaving the original untouched."""
    tagged = image.copy()
    cv2.putText(tagged, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,), 1, cv2.LINE_AA)
    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convolution-demo",
        description="Run the hand-written convolution on a video frame.",
    )
    parser.add_argument("--image", default=os.path.join(OUTPUT, "day1", "frame.png"),
                        help="frame produced by 01_image_anatomy.py")
    parser.add_argument("--size", nargs=2, type=int, default=(120, 160),
                        metavar=("HEIGHT", "WIDTH"),
                        help="crop size the convolution runs on")
    parser.add_argument("--out", default=os.path.join(OUTPUT, "day2"),
                        help="directory the panel is written to")
    parser.add_argument("--show", action="store_true",
                        help="display the panel in a window (needs a GUI)")
    args = parser.parse_args()

    crop = load_grayscale(args.image, tuple(args.size))

    # Every call returns a new image - keep what comes back.
    naive_zero = convolve_2d(crop, BOX_BLUR, padding="zero")
    naive_replicate = convolve_2d(crop, BOX_BLUR, padding="replicate")
    fast_zero = convolve_2d_fast(crop, BOX_BLUR, padding="zero")
    opencv_zero = cv2.filter2D(crop, -1, BOX_BLUR, borderType=cv2.BORDER_CONSTANT)

    print(f"\nkernel sum  : {BOX_BLUR.sum():.1f}  (1.0 keeps the brightness)")
    print(f"mean before : {crop.mean():.2f}")
    print(f"mean after  : {naive_zero.mean():.2f}")

    os.makedirs(args.out, exist_ok=True)
    panel = np.hstack([
        label(crop, "original"),
        label(naive_zero, "naive zero"),
        label(naive_replicate, "naive replicate"),
        label(fast_zero, "fast zero"),
        label(opencv_zero, "cv2 zero"),
    ])
    panel_path = os.path.join(args.out, "convolution_panel.png")
    if not cv2.imwrite(panel_path, panel):
        raise RuntimeError(f"Panel could not be written to: {panel_path}")
    print(f"\npanel saved -> {panel_path}")

    if args.show:
        cv2.imshow("convolution", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
