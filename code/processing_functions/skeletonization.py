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

    return image

import numpy as np
import cv2

# Helper: Get 8-neighbour coordinates
def get_neighbours(x, y, shape):
    h, w = shape
    neighbours = [
        (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
        (x,     y - 1),           (x,     y + 1),
        (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
    ]
    return [(i, j) for i, j in neighbours if 0 <= i < h and 0 <= j < w]

# Helper: Get stick neighbours (1-value neighbours)
def get_stick_neighbours(img, x, y):
    return sum(img[nx, ny] != 0 for nx, ny in get_neighbours(x, y, img.shape))

# Helper: Compute weight using matrix ((128,1,2),(64,x,4),(32,16,8))
def get_weight(img, x, y):
    weight_matrix = [
        [128,   1,   2],
        [ 64,   0,   4],
        [ 32,  16,   8]
    ]
    weight = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                if img[nx, ny] != 0:
                    weight += weight_matrix[dx + 1][dy + 1]
    return weight

# Set of weights to remove
removal_weights = set([
    3, 5, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 48, 52, 53, 54,
    55, 56, 60, 61, 62, 63, 65, 67, 69, 71, 77, 79, 80, 81, 83, 84, 85, 86, 
    87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101, 103, 109, 111, 112, 113, 
    115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127, 131, 133, 
    135, 141, 143, 149, 151, 157, 159, 181, 183, 189, 191, 192, 193, 195, 
    197, 199, 205, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 219, 
    220, 221, 222, 223, 224, 225, 227, 229, 231, 237, 239, 240, 241, 243, 
    244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255
])

def kmm_skeletonize(bitmap):
    img = bitmap.copy()

    # Step 1-2: Normalize input so that foreground = 1, background = 0
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if np.max(img) > 1:  # likely 0-255 range
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)

    # Step 2: Mark all 1s (already done if input is binary with foreground 1)
    img[img != 0] = 1

    # Step 3 & 4: Edge and corner detection
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 1:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= x + dx < img.shape[0] and 0 <= y + dy < img.shape[1]:
                        if img[x + dx, y + dy] == 0:
                            img[x, y] = 2
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    if 0 <= x + dx < img.shape[0] and 0 <= y + dy < img.shape[1]:
                        if img[x + dx, y + dy] == 0:
                            if img[x, y] != 2:
                                img[x, y] = 3

    # Step 5: Mark 2/3 with 2-4 neighbours as 4
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] in [2, 3]:
                sticks = get_stick_neighbours(img, x, y)
                if 2 <= sticks <= 4:
                    img[x, y] = 4

    # Step 6: Remove all 4s
    img[img == 4] = 0

    # Step 7: Initialize
    N = 2
    i_max = img.size

    # Step 8+: Main iterative thinning
    while True:
        changed = False
        flat_img = img.flatten()
        for i in range(i_max):
            x, y = divmod(i, img.shape[1])
            if flat_img[i] == N:
                weight = get_weight(img, x, y)
                if weight in removal_weights:
                    flat_img[i] = 0
                    changed = True
                else:
                    flat_img[i] = 1
        img = flat_img.reshape(img.shape)
        if N == 3 or not changed:
            break
        N = 3

    return img
