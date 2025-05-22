"""
Changing image to grayscale and normalizing image to a specific range to improve contrast.
"""

import numpy as np
from math import sqrt
from PIL import Image

def adjust_pixel_intensity(pixel, desired_mean, desired_var, global_mean, global_var):

    """
    Adjust a pixel's intensity based on desired and global statistics.
    """
    diff = pixel - global_mean
    factor = sqrt((desired_var * (diff ** 2)) / global_var)
    return desired_mean + factor if diff > 0 else desired_mean - factor

def normalize_image(image, desired_mean, desired_var):
    """
    Normalize the image to have a desired mean and variance.
    Each pixel is transformed individually.
    """

    # Convert to grayscale if input is a PIL Image
    if isinstance(image, Image.Image):
        image = image.convert('L')
        img_array = np.array(image).astype(float)
    else:
        # Assume image is a NumPy array; convert to grayscale if it has 3 channels
        img_array = image.astype(float)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Simple luminance grayscale conversion
            img_array = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]

    global_mean = np.mean(img_array)
    global_var = np.var(img_array)

    normalized = np.empty_like(img_array)
    height, width = img_array.shape

    for row in range(height):
        for col in range(width):
            normalized[row, col] = adjust_pixel_intensity(
                img_array[row, col],
                desired_mean,
                desired_var,
                global_mean,
                global_var
            )

    return normalized
