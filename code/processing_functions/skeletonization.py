import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_skeleton(image, element_shape=cv2.MORPH_CROSS):
    """
    Perform morphological skeletonization on a binary image.
    """
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image = (image > 0).astype(np.uint8)
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

    return skeleton
