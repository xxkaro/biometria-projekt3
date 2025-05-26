import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def morphological_skeleton(image, mask=None, element_shape=cv2.MORPH_CROSS):
    """
    Perform morphological skeletonization on a binary image, optionally within a masked region.
    """
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = (image > 0).astype(np.uint8)

    if mask is not None:
        image = cv2.bitwise_and(image, mask.astype(np.uint8))

    element = cv2.getStructuringElement(element_shape, (3, 3))
    skeleton = np.zeros_like(image)
    k = 0

    while True:
        eroded = image.copy()
        for _ in range(k):
            eroded = cv2.erode(eroded, element)

        if not np.any(eroded):
            break

        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        sk = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, sk)
        k += 1

    
    skeleton = skeleton * 255 
    skeleton = cv2.bitwise_not(skeleton)
    return skeleton



def thinning_with_masks(image, max_iter=1000):
    """
    Perform thinning of a binary image using hit-or-miss transforms with predefined masks.
    """
    
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.bitwise_not(image)
    image = (image > 0).astype(np.uint8)

    masks = {
        "mask0": np.array([[ 0,  0,  0],
                           [-1,  1, -1],
                           [ 1,  1,  1]], dtype=np.int8),

        "mask1": np.array([[-1,  0,  0],
                           [ 1,  1,  0],
                           [ 1,  1, -1]], dtype=np.int8),

        "mask2": np.array([[ 1, -1,  0],
                           [ 1,  1,  0],
                           [ 1, -1,  0]], dtype=np.int8),

        "mask3": np.array([[ 1,  1, -1],
                           [ 1,  1,  0],
                           [-1,  0,  0]], dtype=np.int8),

        "mask4": np.array([[ 1,  1,  1],
                           [-1,  1, -1],
                           [ 0,  0,  0]], dtype=np.int8),

        "mask5": np.array([[-1,  1,  1],
                           [ 0,  1,  1],
                           [ 0,  0, -1]], dtype=np.int8),

        "mask6": np.array([[ 0, -1,  1],
                           [ 0,  1,  1],
                           [ 0, -1,  1]], dtype=np.int8),

        "mask7": np.array([[ 0,  0, -1],
                           [ 0,  1,  1],
                           [-1,  1,  1]], dtype=np.int8)
    }

    changed = True
    iter_count = 0

    while changed and iter_count < max_iter:
        changed = False
        for mask in masks.values():
            hitmiss = cv2.morphologyEx(image, cv2.MORPH_HITMISS, mask)
            if np.any(hitmiss):
                image = cv2.bitwise_and(image, cv2.bitwise_not(hitmiss))
                changed = True
        iter_count += 1

    return cv2.bitwise_not(image)