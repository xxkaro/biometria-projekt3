import cv2 as cv
import numpy as np
import scipy.ndimage

def apply_gabor_filters(image, orientation, frequency_map, mask,
                                       block_size=16, default_frequency=None,
                                       kx=0.65, ky=0.65, angle_inc=5):
    """
    Apply Gabor filtering to an image based on orientation and frequency maps,
    using precomputed rotated Gabor filters (strategy analogiczna do poprzedniego kodu).
    """

    im = np.double(image)
    rows, cols = im.shape
    filtered = np.zeros_like(im)

    if default_frequency is None:
        non_zero_freqs = frequency_map[frequency_map > 0]
        default_frequency = np.median(non_zero_freqs) if non_zero_freqs.size else 0.1

    freq = default_frequency

    sigma_x = 1 / freq * kx
    sigma_y = 1 / freq * ky
    max_sigma = max(sigma_x, sigma_y)
    filt_size = int(np.round(3 * max_sigma))
    x = np.linspace(-filt_size, filt_size, 2 * filt_size + 1)
    y = np.linspace(-filt_size, filt_size, 2 * filt_size + 1)
    x, y = np.meshgrid(x, y)

    base_filter = np.exp(-((x ** 2) / (sigma_x ** 2) + (y ** 2) / (sigma_y ** 2))) * np.cos(2 * np.pi * freq * x)
    filt_rows, filt_cols = base_filter.shape

    n_angles = 180 // angle_inc
    gabor_filters = np.zeros((n_angles, filt_rows, filt_cols))
    for i in range(n_angles):
        angle_deg = -(i * angle_inc + 90)
        gabor_filters[i] = scipy.ndimage.rotate(base_filter, angle_deg, reshape=False)

    max_orient_index = n_angles

    orient_indices = np.round(orientation / np.pi * 180 / angle_inc).astype(int)
    orient_indices[orient_indices < 0] += max_orient_index
    orient_indices[orient_indices >= max_orient_index] -= max_orient_index

    for r in range(0, rows - block_size + 1, block_size):
        for c in range(0, cols - block_size + 1, block_size):
            if not mask[r + block_size // 2, c + block_size // 2]:
                continue

            block = im[r:r + block_size, c:c + block_size]
            freq_block = np.mean(frequency_map[r:r + block_size, c:c + block_size])
            if freq_block == 0 or np.isnan(freq_block):
                freq_block = default_frequency

            orient_idx = orient_indices[r // block_size, c // block_size]

            gabor_kernel = gabor_filters[orient_idx]
            center = filt_size
            half_block = block_size // 2
            kernel_block = gabor_kernel[center - half_block:center + half_block + 1,
                                        center - half_block:center + half_block + 1]
            if kernel_block.shape != (block_size, block_size):
                kernel_block = kernel_block[:block_size, :block_size]

            filtered_block = cv.filter2D(block, -1, kernel_block)
            filtered[r:r + block_size, c:c + block_size] = filtered_block

    filtered_norm = cv.normalize(filtered, None, 0, 255, norm_type=cv.NORM_MINMAX)
    return filtered_norm.astype(np.uint8)
