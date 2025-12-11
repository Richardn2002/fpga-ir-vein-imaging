from PIL import Image
import os
import sys
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("QtAgg")

import numpy as np


# parameters


INPUT_X = 128
INPUT_Y = 128

CLAHE_CLIP_LIMIT = 10
CLAHE_PATCH_X_NUM = 4
CLAHE_PATCH_Y_NUM = 4
CLAHE_PATCH_X = 32
CLAHE_PATCH_Y = 32

HESSIAN_INPUT_X = (CLAHE_PATCH_X_NUM - 1) * CLAHE_PATCH_X
HESSIAN_INPUT_Y = (CLAHE_PATCH_Y_NUM - 1) * CLAHE_PATCH_Y
HESSIAN_SIGMA = 3
HESSIAN_RADIUS = 3
HESSIAN_OUTPUT_X = HESSIAN_INPUT_X - 2 * HESSIAN_RADIUS
HESSIAN_OUTPUT_Y = HESSIAN_INPUT_Y - 2 * HESSIAN_RADIUS

HESSIAN_KERNEL_SCALE = 4
"""
def gaussian_kernel(sigma, radius):
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)
HESSIAN_KERNEL = gaussian_kernel(HESSIAN_SIGMA, HESSIAN_RADIUS)
HESSIAN_KERNEL = np.round(HESSIAN_KERNEL * HESSIAN_KERNEL_SCALE).astype(np.int16)
"""
HESSIAN_KERNEL = np.array([0, 1, 1, 1, 1, 1, 0], dtype=np.uint8)

"""
SQRT_BINS = [round((i + 0.5) ** 2) for i in range(16)]
def sqrt_quant(v):
    return np.digitize(v, SQRT_BINS, True)
"""
SQRT_BINS = np.array(
    [0, 2, 6, 12, 20, 30, 42, 56, 72, 90, 110, 132, 156, 182, 210, 240], dtype=np.uint8
)


# Modules


def CLAHE_mappings(img_in: np.ndarray, hist_mapping_inout: np.ndarray):
    """
    calculate CLAHE mapping for one patch of input image
    """
    assert img_in.shape == (CLAHE_PATCH_Y, CLAHE_PATCH_X) and img_in.dtype == np.uint8
    assert hist_mapping_inout.shape == (256,) and hist_mapping_inout.dtype == np.uint8

    # calculate hist

    excess = np.uint16(0)
    # instead of such initialization, can also make 256 flags of "has this i been accessed already"
    for i in range(256):
        hist_mapping_inout[i] = np.uint8(0)

    # for pixel in patch:
    for y in range(CLAHE_PATCH_Y):
        for x in range(CLAHE_PATCH_X):
            pixel = img_in[y, x]
            if hist_mapping_inout[pixel] == CLAHE_CLIP_LIMIT - 2:
                excess += 1
            else:
                hist_mapping_inout[pixel] += 1

    redist = excess // 256
    for i in range(256):
        hist_mapping_inout[i] += redist

    # calculate mapping

    s = np.uint32(0)
    for i in range(256):
        s += hist_mapping_inout[i]
        s_normed = np.uint8(np.clip(s * 256 // CLAHE_PATCH_X // CLAHE_PATCH_Y, 0, 255))
        hist_mapping_inout[i] = s_normed


def CLAHE_output(img_in: np.ndarray, mapping_in: np.ndarray, clahe_out: np.ndarray):
    """
    create CLAHE output from input image and CLAHE mappings
    """
    assert img_in.shape == (INPUT_Y, INPUT_X) and img_in.dtype == np.uint8
    assert (
        mapping_in.shape == (CLAHE_PATCH_Y_NUM, CLAHE_PATCH_X_NUM, 256)
        and mapping_in.dtype == np.uint8
    )
    assert (
        clahe_out.shape == (INPUT_Y - CLAHE_PATCH_Y, INPUT_X - CLAHE_PATCH_X)
        and clahe_out.dtype == np.uint8
    )

    for y in range(INPUT_Y - CLAHE_PATCH_Y):
        for x in range(INPUT_X - CLAHE_PATCH_X):
            cur_row = y // CLAHE_PATCH_Y
            cur_col = x // CLAHE_PATCH_X

            pixel = img_in[y + CLAHE_PATCH_Y // 2, x + CLAHE_PATCH_X // 2]
            tl = mapping_in[cur_row, cur_col][pixel]
            tr = mapping_in[cur_row, cur_col + 1][pixel]
            bl = mapping_in[cur_row + 1, cur_col][pixel]
            br = mapping_in[cur_row + 1, cur_col + 1][pixel]

            dx = np.uint32(x % CLAHE_PATCH_X)
            dx_n = np.uint32(CLAHE_PATCH_X - 1 - dx)
            dy = np.uint32(y % CLAHE_PATCH_Y)
            dy_n = np.uint32(CLAHE_PATCH_Y - 1 - dy)
            interp = (
                (tl * dx_n * dy_n + tr * dx * dy_n + bl * dx_n * dy + br * dx * dy)
                // CLAHE_PATCH_X
                // CLAHE_PATCH_Y
            )

            clahe_out[y, x] = np.uint8(interp)


def hessian_conv_r(clahe_in: np.ndarray, conv_r_out: np.ndarray):
    """
    convolve CLAHE output along rows

    NON-parametrizable code! dependent on the fact that the kernel is [0, 1, 1, 1, 1, 1, 0]
    """
    assert (
        clahe_in.shape == (HESSIAN_INPUT_Y, HESSIAN_INPUT_X)
        and clahe_in.dtype == np.uint8
    )
    assert (
        conv_r_out.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and conv_r_out.dtype == np.int16
    )

    window = np.empty(5, dtype=np.int16)
    for row in range(HESSIAN_RADIUS, HESSIAN_INPUT_Y - HESSIAN_RADIUS):
        window[0:5] = clahe_in[row][1:6]
        for col in range(HESSIAN_RADIUS, HESSIAN_INPUT_X - HESSIAN_RADIUS):
            conv_r_out[row - HESSIAN_RADIUS][col - HESSIAN_RADIUS] = np.sum(window)
            # shift
            window[0:4] = window[1:5]
            window[4] = clahe_in[row][col + 3]


def hessian_conv_c(conv_r_in: np.ndarray, conv_out: np.ndarray):
    """
    convolve row convolution result along columns

    NON-parametrizable code! dependent on the fact that the kernel is [0, 1, 1, 1, 1, 1, 0]
    """
    assert (
        conv_r_in.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and conv_r_in.dtype == np.int16
    )
    assert (
        conv_out.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and conv_out.dtype == np.int16
    )

    window = np.empty(5, dtype=np.int16)
    for col in range(HESSIAN_OUTPUT_X):
        # for ram buffer to fit in 2x 32 kb blocks, top rows and bottom rows of conv_x output are missing
        # pad by border
        window[0:5] = np.array(
            [
                conv_r_in[0][0],
                conv_r_in[0][0],
                conv_r_in[0][0],
                conv_r_in[1][0],
                conv_r_in[2][0],
            ]
        )
        for row in range(HESSIAN_OUTPUT_Y):
            conv_out[row][col] = np.sum(window)
            # shift
            window[0:4] = window[1:5]
            if row + 3 >= HESSIAN_OUTPUT_Y:
                window[4] = window[3]
            else:
                window[4] = conv_r_in[row + 3][col]


def hessian_grad_r(
    conv_in_0: np.ndarray,
    conv_in_1: np.ndarray,
    gr_out: np.ndarray,
):
    """
    compute gradient along rows from convolution result
    """
    assert (
        conv_in_0.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and conv_in_0.dtype == np.int16
    )
    assert (
        conv_in_1.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and conv_in_1.dtype == np.int16
    )
    assert (
        gr_out.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and gr_out.dtype == np.int16
    )

    for y in range(HESSIAN_OUTPUT_Y):
        for x in range(HESSIAN_OUTPUT_X):
            gr = 0
            if y == 0:
                gr = (conv_in_1[1][x] - conv_in_0[0][x]) * 2
            elif y == HESSIAN_OUTPUT_Y - 1:
                gr = (
                    conv_in_1[HESSIAN_OUTPUT_Y - 1][x]
                    - conv_in_0[HESSIAN_OUTPUT_Y - 2][x]
                ) * 2
            else:
                gr = conv_in_1[y + 1][x] - conv_in_0[y - 1][x]

            gr_out[y][x] = gr


def hessian_grad_c(conv_in_0: np.ndarray, conv_in_1: np.ndarray, gc_out: np.ndarray):
    """
    compute gradient along columns from convolution result
    """
    assert (
        conv_in_0.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and conv_in_0.dtype == np.int16
    )
    assert (
        conv_in_1.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and conv_in_1.dtype == np.int16
    )
    assert (
        gc_out.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and gc_out.dtype == np.int16
    )

    for y in range(HESSIAN_OUTPUT_Y):
        for x in range(HESSIAN_OUTPUT_X):
            gc = 0
            if x == 0:
                gc = (conv_in_1[y][1] - conv_in_0[y][0]) * 2
            elif x == HESSIAN_OUTPUT_X - 1:
                gc = (
                    conv_in_1[y][HESSIAN_OUTPUT_X - 1]
                    - conv_in_0[y][HESSIAN_OUTPUT_X - 2]
                ) * 2
            else:
                gc = conv_in_1[y][x + 1] - conv_in_0[y][x - 1]

            gc_out[y][x] = gc


def hessian_grad_rr_cc(
    gr_in_0: np.ndarray,
    gr_in_1: np.ndarray,
    gc_in_0: np.ndarray,
    gc_in_1: np.ndarray,
    rr_p_cc_term_out: np.ndarray,
    rr_m_cc_term_out: np.ndarray,
):
    """
    compute (Hrr + Hcc) / 2 / 4 and (Hrr - Hcc) ** 2 / 4 / 4 from gr and gc
    """
    assert (
        gr_in_0.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and gr_in_0.dtype == np.int16
    )
    assert (
        gr_in_1.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and gr_in_1.dtype == np.int16
    )
    assert (
        gc_in_0.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and gc_in_0.dtype == np.int16
    )
    assert (
        gc_in_1.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and gc_in_1.dtype == np.int16
    )
    assert (
        rr_p_cc_term_out.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and rr_p_cc_term_out.dtype == np.int16
    )
    assert (
        rr_m_cc_term_out.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and rr_m_cc_term_out.dtype == np.int16
    )

    for y in range(HESSIAN_OUTPUT_Y):
        for x in range(HESSIAN_OUTPUT_X):
            hrr = 0
            if y == 0:
                hrr = (gr_in_1[1][x] - gr_in_0[0][x]) * 2
            elif y == HESSIAN_OUTPUT_Y - 1:
                hrr = (
                    gr_in_1[HESSIAN_OUTPUT_Y - 1][x] - gr_in_0[HESSIAN_OUTPUT_Y - 2][x]
                ) * 2
            else:
                hrr = gr_in_1[y + 1][x] - gr_in_0[y - 1][x]

            hcc = 0
            if x == 0:
                hcc = (gc_in_1[y][1] - gc_in_0[y][0]) * 2
            elif x == HESSIAN_OUTPUT_X - 1:
                hcc = (
                    gc_in_1[y][HESSIAN_OUTPUT_X - 1] - gc_in_0[y][HESSIAN_OUTPUT_X - 2]
                ) * 2
            else:
                hcc = gc_in_1[y][x + 1] - gc_in_0[y][x - 1]

            rr_p_cc_term_out[y][x] = (hrr + hcc) // 2 // 4
            rr_m_cc_term_out[y][x] = np.int16((np.int32(hrr - hcc) ** 2) // 4 // 4)


def sqrt_quant(v: np.int16) -> np.int16:
    """
    quantized sqrt, output [0, 16]
    """

    if v <= SQRT_BINS[0]:
        return np.int16(0)
    elif v <= SQRT_BINS[1]:
        return np.int16(1)
    elif v <= SQRT_BINS[2]:
        return np.int16(2)
    elif v <= SQRT_BINS[3]:
        return np.int16(3)
    elif v <= SQRT_BINS[4]:
        return np.int16(4)
    elif v <= SQRT_BINS[5]:
        return np.int16(5)
    elif v <= SQRT_BINS[6]:
        return np.int16(6)
    elif v <= SQRT_BINS[7]:
        return np.int16(7)
    elif v <= SQRT_BINS[8]:
        return np.int16(8)
    elif v <= SQRT_BINS[9]:
        return np.int16(9)
    elif v <= SQRT_BINS[10]:
        return np.int16(10)
    elif v <= SQRT_BINS[11]:
        return np.int16(11)
    elif v <= SQRT_BINS[12]:
        return np.int16(12)
    elif v <= SQRT_BINS[13]:
        return np.int16(13)
    elif v <= SQRT_BINS[14]:
        return np.int16(14)
    elif v <= SQRT_BINS[15]:
        return np.int16(15)
    else:
        return np.int16(16)


def hessian_output(
    rr_p_cc_term_in: np.ndarray,
    rr_m_cc_term_in: np.ndarray,
    rc_in: np.ndarray,
    hessian_out: np.ndarray,
):
    """
    calculate abs(
    (Hrr + Hcc) / 2 / 4 + sqrt_quant(Hrc ** 2 / 4 + (Hrr - Hcc) ** 2 / 4 / 4 + 1)
    ) * scaling (default * 64 // HESSIAN_KERNEL_SCALE ** 2)

    given ((Hrr + Hcc) / 2 / 4), ((Hrr - Hcc) ** 2 / 4 / 4) and Hrc
    """
    assert (
        rr_p_cc_term_in.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and rr_p_cc_term_in.dtype == np.int16
    )
    assert (
        rr_m_cc_term_in.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and rr_m_cc_term_in.dtype == np.int16
    )
    assert (
        rc_in.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X) and rc_in.dtype == np.int16
    )
    assert (
        hessian_out.shape == (HESSIAN_OUTPUT_Y, HESSIAN_OUTPUT_X)
        and hessian_out.dtype == np.int16
    )

    for y in range(HESSIAN_OUTPUT_Y):
        for x in range(HESSIAN_OUTPUT_X):
            out = (
                (
                    rr_p_cc_term_in[y][x]
                    + sqrt_quant(
                        np.int16(np.int32(rc_in[y][x]) ** 2 // 4)
                        + rr_m_cc_term_in[y][x]
                        + 1
                    )
                )
                * 64
                // (HESSIAN_KERNEL_SCALE**2)
            )
            hessian_out[y][x] = out if out > 0 else -out


# Testbench


def load_test_input() -> np.ndarray:
    img = Image.open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../data/inputs/20. testing_1.PNG",
        )
    ).convert("L")
    y_offset = 250
    x_offset = 320
    img = np.array(img)[y_offset : y_offset + 256, x_offset : x_offset + 256]
    # resize to 128 * 128
    img_resized = np.zeros((128, 128), dtype=np.uint8)
    for y in range(128):
        for x in range(128):
            img_resized[y, x] = np.uint8(
                np.mean(img[2 * y : 2 * y + 2, 2 * x : 2 * x + 2])
            )

    return img_resized


if __name__ == "__main__":
    # Block RAM usages

    # 128 * 128 * 8 = 131072 b = 4 blocks
    ram_input = np.empty((INPUT_Y, INPUT_X), dtype=np.uint8)

    # use for both histogram and cdf
    # 4 * 4 * 256 * 8 = 32768 b = 1 block
    ram_hist_mapping = np.empty(
        (CLAHE_PATCH_Y_NUM, CLAHE_PATCH_X_NUM, 256), dtype=np.uint8
    )

    # 96 * 96 * 8 = 173728 b = 2.25 blocks
    ram_clahe_output = np.empty(
        (INPUT_Y - CLAHE_PATCH_Y, INPUT_X - CLAHE_PATCH_X), dtype=np.uint8
    )

    # 4 buffers for hessian to use
    # each 90 * 90 * 16 = 129600 b = 3.96 blocks
    ram_hessian_0 = np.empty(
        (
            HESSIAN_OUTPUT_Y,
            HESSIAN_OUTPUT_X,
        ),
        dtype=np.int16,
    )
    ram_hessian_1 = np.empty(
        (
            HESSIAN_OUTPUT_Y,
            HESSIAN_OUTPUT_X,
        ),
        dtype=np.int16,
    )
    ram_hessian_2 = np.empty(
        (
            HESSIAN_OUTPUT_Y,
            HESSIAN_OUTPUT_X,
        ),
        dtype=np.int16,
    )
    ram_hessian_3 = np.empty(
        (
            HESSIAN_OUTPUT_Y,
            HESSIAN_OUTPUT_X,
        ),
        dtype=np.int16,
    )

    # module instantiations and visual checks

    if len(sys.argv) == 1:
        ram_input = load_test_input()
    else:
        img = np.array(Image.open(sys.argv[1]).convert("L"))
        img_resized = np.zeros((128, 128), dtype=np.uint8)
        for y in range(128):
            for x in range(128):
                img_resized[y, x] = np.uint8(
                    np.mean(img[4 * y : 4 * y + 4, 4 * x : 4 * x + 4])
                )
        ram_input = img_resized

    for row in range(CLAHE_PATCH_Y_NUM):
        for col in range(CLAHE_PATCH_X_NUM):
            CLAHE_mappings(
                ram_input[
                    row * CLAHE_PATCH_Y : (row + 1) * CLAHE_PATCH_Y,
                    col * CLAHE_PATCH_X : (col + 1) * CLAHE_PATCH_X,
                ],
                ram_hist_mapping[row][col],
            )
    CLAHE_output(ram_input, ram_hist_mapping, ram_clahe_output)
    hessian_conv_r(ram_clahe_output, ram_hessian_0)
    hessian_conv_c(ram_hessian_0, ram_hessian_1)

    plt.subplot(3, 3, 1)
    plt.title("Input")
    plt.imshow(ram_input, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.title("After CLAHE")
    plt.imshow(ram_clahe_output, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.title("After Hessian (Conv)")
    plt.imshow(ram_hessian_1, cmap="gray")
    plt.axis("off")

    hessian_grad_r(ram_hessian_1, ram_hessian_1, ram_hessian_0)
    hessian_grad_c(ram_hessian_1, ram_hessian_1, ram_hessian_2)

    plt.subplot(3, 3, 4)
    plt.title("gr")
    plt.imshow(ram_hessian_0, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.title("gc")
    plt.imshow(ram_hessian_2, cmap="gray")
    plt.axis("off")

    hessian_grad_rr_cc(
        ram_hessian_0,
        ram_hessian_0,
        ram_hessian_2,
        ram_hessian_2,
        ram_hessian_1,
        ram_hessian_3,
    )

    plt.subplot(3, 3, 6)
    plt.title("(Hrr + Hcc) / 2 / 4")
    plt.imshow(ram_hessian_1, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 7)
    plt.title("(Hrr - Hcc) ** 2 / 4 / 4")
    plt.imshow(ram_hessian_3, cmap="gray")
    plt.axis("off")

    hessian_grad_c(ram_hessian_0, ram_hessian_0, ram_hessian_2)

    plt.subplot(3, 3, 8)
    plt.title("Hrc")
    plt.imshow(ram_hessian_2, cmap="gray")
    plt.axis("off")

    hessian_output(ram_hessian_1, ram_hessian_3, ram_hessian_2, ram_hessian_0)

    plt.subplot(3, 3, 9)
    plt.title("After Hessian")
    plt.imshow(ram_hessian_0, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
