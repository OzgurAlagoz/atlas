import numpy as np
import pytest

from atlas.common.convolution import convolve_2d, convolve_2d_fast

IDENTITY = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.float32)

BOX_BLUR = np.ones((3, 3), dtype=np.float32) / 9.0


def make_image(height=10, width=12):
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


def test_identity_kernel_keeps_image_unchanged():
    image = make_image()

    result = convolve_2d(image, IDENTITY)

    assert np.array_equal(result, image)


def test_naive_and_fast_give_identical_output():
    image = make_image()

    naive = convolve_2d(image, BOX_BLUR)
    fast = convolve_2d_fast(image, BOX_BLUR)

    assert np.array_equal(naive, fast)


@pytest.mark.parametrize("padding", ["zero", "replicate"])
def test_output_shape_matches_input(padding):
    image = make_image()

    result = convolve_2d(image, BOX_BLUR, padding=padding)

    assert result.shape == image.shape


def test_even_kernel_raises():
    image = make_image()
    even_kernel = np.ones((4, 4), dtype=np.float32) / 16.0

    with pytest.raises(ValueError, match="odd"):
        convolve_2d(image, even_kernel)


def test_color_image_raises():
    color_image = np.zeros((10, 12, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="grayscale"):
        convolve_2d(color_image, BOX_BLUR)
