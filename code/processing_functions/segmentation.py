"""
Fingerprint Image Segmentation and Normalization

This script identifies the region of interest (ROI) in a fingerprint image by dividing it into blocks 
and computing local variance. Blocks with variance below a threshold are considered background and excluded. 
The image is then normalized to have zero mean and unit variance, enhancing ridge detail for further processing.

Returns the segmented image, normalized image, and a binary ROI mask.
"""

import numpy as np
import cv2 as cv

def normalize_image(img):
    """Normalize image to zero mean and unit variance."""
    return (img - np.mean(img)) / np.std(img)

def perform_segmentation(img, block_size=16, threshold_ratio=0.2):
    """
    Segments the image into regions of interest based on local variance,
    and returns the segmented image, normalized image, and the ROI mask.
    """
    height, width = img.shape
    std_threshold = np.std(img) * threshold_ratio

    variance_map = np.zeros_like(img)
    roi_mask = np.ones_like(img, dtype=np.uint8)

    # Calculate standard deviation for each block
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = img[y:y_end, x:x_end]
            block_std = np.std(block)
            variance_map[y:y_end, x:x_end] = block_std

    # Generate mask using threshold
    roi_mask[variance_map < std_threshold] = 0

    # Refine mask using morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (block_size * 2, block_size * 2))
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_OPEN, kernel)
    roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, kernel)

    # Apply mask to input image
    segmented = img * roi_mask

    # Normalize only the region of interest
    normalized = normalize_image(img)
    bg_mean = np.mean(normalized[roi_mask == 0])
    bg_std = np.std(normalized[roi_mask == 0])
    normalized = (normalized - bg_mean) / bg_std

    return segmented, normalized, roi_mask
