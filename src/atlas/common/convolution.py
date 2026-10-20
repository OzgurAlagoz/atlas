"""Hand-written 2D convolution for ATLAS - week 2, day 2.

Two implementations with the same contract:

- `convolve_2d`       naive, nested Python loops, easy to read
- `convolve_2d_fast`  vectorised with sliding windows

Both take a 2D (grayscale) uint8 image and an odd-sized 2D kernel, flip the
kernel (true convolution rather than correlation) and return a uint8 image of
the same shape as the input.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PADDING_MODES = {
    "zero": {"mode": "constant", "constant_values": 0},
    "replicate": {"mode": "edge"},
}


def _validate(image: np.ndarray, kernel: np.ndarray) -> None:
    """Reject inputs the implementations cannot handle."""
    if image.ndim != 2:
        raise ValueError(
            f"Image must be 2D grayscale, got shape {image.shape}. "
            "Convert it with cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) first."
        )
    if kernel.ndim != 2:
        raise ValueError(f"Kernel must be 2D, got shape {kernel.shape}.")

    kernel_height, kernel_width = kernel.shape
    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError(
            f"Kernel sides must be odd, got {kernel.shape}. "
            "An even kernel has no single centre pixel."
        )


def _pad(image: np.ndarray, pad_y: int, pad_x: int, padding: str) -> np.ndarray:
    """Grow the image border so the kernel never falls off the edge."""
    if padding not in PADDING_MODES:
        raise ValueError(
            f"Unknown padding '{padding}'. Expected one of {sorted(PADDING_MODES)}."
        )
    return np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), **PADDING_MODES[padding])


def convolve_2d(image: np.ndarray, kernel: np.ndarray,
                padding: str = "zero") -> np.ndarray:
    """Convolve with explicit loops. Slow, but shows every step."""
    _validate(image, kernel)

    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_y, pad_x = kernel_height // 2, kernel_width // 2

    image = image.astype(np.float32)
    kernel = kernel[::-1, ::-1].astype(np.float32)   # flip: convolution, not correlation
    padded = _pad(image, pad_y, pad_x, padding)

    output = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            total = 0.0
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    total += padded[y + ky, x + kx] * kernel[ky, kx]
            output[y, x] = total

    return np.clip(output, 0, 255).astype(np.uint8)


def convolve_2d_fast(image: np.ndarray, kernel: np.ndarray,
                     padding: str = "zero") -> np.ndarray:
    """Same result as `convolve_2d`, computed without Python loops."""
    _validate(image, kernel)

    kernel_height, kernel_width = kernel.shape
    pad_y, pad_x = kernel_height // 2, kernel_width // 2

    image = image.astype(np.float32)
    kernel = kernel[::-1, ::-1].astype(np.float32)
    padded = _pad(image, pad_y, pad_x, padding)

    # One window per output pixel, all extracted at once.
    windows = sliding_window_view(padded, (kernel_height, kernel_width))
    output = np.sum(windows * kernel, axis=(-2, -1))

    return np.clip(output, 0, 255).astype(np.uint8)
