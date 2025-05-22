"""
Creating a directional and frequency maps for the fingerprint image.
"""

# DIRECTION MAP 
import math
import numpy as np
import cv2 as cv
import scipy.ndimage


def compute_orientations(image, block_size=16, smooth=False):
    """
    Estimate ridge orientations for each block of the image using Sobel gradients.

    Args:
        image (ndarray): Input grayscale image.
        block_size (int): Size of the square block for local orientation estimation.
        smooth (bool): Whether to apply angle smoothing.

    Returns:
        ndarray: 2D array of local orientation angles in radians.
    """
    height, width = image.shape
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = sobel_y.T

    gx = cv.filter2D(image.astype(np.float32), -1, sobel_x)
    gy = cv.filter2D(image.astype(np.float32), -1, sobel_y)

    angles = []

    for y in range(0, height, block_size):
        row_angles = []
        for x in range(0, width, block_size):
            end_y = min(y + block_size, height)
            end_x = min(x + block_size, width)

            block_gx = gx[y:end_y, x:end_x]
            block_gy = gy[y:end_y, x:end_x]

            j1 = 2 * np.sum(block_gx * block_gy)
            j2 = np.sum(block_gx**2 - block_gy**2)

            if j1 == 0 and j2 == 0:
                angle = 0
            else:
                angle = 0.5 * math.atan2(j1, j2) + math.pi / 2

            row_angles.append(angle)
        angles.append(row_angles)

    angles = np.array(angles)

    if smooth:
        angles = smooth_orientation_field(angles)

    return angles


def gaussian_kernel(size=5, sigma=1.0):
    """
    Generate a 2D Gaussian kernel.

    Args:
        size (int): Kernel size.
        sigma (float): Standard deviation of Gaussian.

    Returns:
        ndarray: Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def smooth_orientation_field(angles):
    """
    Apply low-pass filtering to smooth orientation field using vector averaging.

    Args:
        angles (ndarray): Orientation angles.

    Returns:
        ndarray: Smoothed angles.
    """
    cos2theta = np.cos(2 * angles)
    sin2theta = np.sin(2 * angles)
    kernel = gaussian_kernel(5)

    smooth_cos = cv.filter2D(cos2theta, -1, kernel)
    smooth_sin = cv.filter2D(sin2theta, -1, kernel)

    return 0.5 * np.arctan2(smooth_sin, smooth_cos)


def get_line_coords(cx, cy, length, angle):
    """
    Get line endpoints centered at (cx, cy) with given length and angle.
    Returns:
        (tuple, tuple): (start_point, end_point)
    """
    dx = int(length * math.cos(angle) / 2)
    dy = int(length * math.sin(angle) / 2)
    return (cx - dx, cy - dy), (cx + dx, cy + dy)


def draw_orientations(image, mask, angles, block_size=16):
    """
    Draw orientation lines on a blank RGB canvas using the orientation field.

    Args:
        image (ndarray): Grayscale input image.
        mask (ndarray): Binary mask defining valid fingerprint region.
        angles (ndarray): Block orientation angles.
        block_size (int): Block size used for orientation estimation.

    Returns:
        ndarray: RGB image with orientation lines.
    """
    height, width = image.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            if np.sum(mask[y:y+block_size, x:x+block_size]) >= ((block_size-1)**2):
                angle = angles[y // block_size][x // block_size]
                center = (x + block_size // 2, y + block_size // 2)
                start, end = get_line_coords(*center, block_size, angle)
                cv.line(overlay, start, end, color=(150, 150, 150), thickness=1)

    return overlay


# FREQUENCY MAP - needs fix

def estimate_frequency_block(block, angle, kernel_size=5, min_wavelength=3, max_wavelength=20):
    """
    Estimate ridge frequency in a block of a fingerprint image by projecting intensities
    along the orientation and measuring spacing between peaks.
    """
    rows, cols = block.shape

    theta = angle
    rotated = scipy.ndimage.rotate(block, theta * 180 / np.pi + 90, reshape=False, order=3, mode='nearest')

    crop = int(rows / np.sqrt(2))
    offset = (rows - crop) // 2
    cropped = rotated[offset:offset+crop, offset:offset+crop]

    projection = np.sum(cropped, axis=0)
    dilated = scipy.ndimage.grey_dilation(projection, size=(kernel_size,))
  
    threshold = 6
    peak_mask = (np.abs(dilated - projection) < threshold) & (projection > np.mean(projection))
    peaks = np.where(peak_mask)[0]

    if len(peaks) < 2 or peaks[-1] == peaks[0]:
        return np.zeros(block.shape)

    wavelength = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
    if min_wavelength <= wavelength <= max_wavelength:
        return np.full(block.shape, 1.0 / wavelength)
    
    return np.zeros(block.shape)



def estimate_ridge_frequency(image, mask, orientation, block_size=16, kernel_size=5,
                             min_wavelength=3, max_wavelength=20):
    """
    Estimate ridge frequency for the entire fingerprint image based on block-wise analysis.
    """
    rows, cols = image.shape
    frequency_map = np.zeros_like(image, dtype=np.float32)

    for r in range(0, rows - block_size + 1, block_size):
        for c in range(0, cols - block_size + 1, block_size):
            block = image[r:r+block_size, c:c+block_size]
            angle = orientation[r // block_size][c // block_size]  # Adjust if needed

            if mask[r + block_size//2, c + block_size//2]:  # Use mask to validate block
                freq_block = estimate_frequency_block(block, angle, kernel_size,
                                                      min_wavelength, max_wavelength)
                frequency_map[r:r+block_size, c:c+block_size] = freq_block

    frequency_map *= mask

    non_zero_freqs = frequency_map[frequency_map > 0]
    median_frequency = np.median(non_zero_freqs) if non_zero_freqs.size else 0

    return frequency_map, median_frequency