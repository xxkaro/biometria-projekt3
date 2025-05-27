import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from collections import deque


def morphological_skeleton(image, mask=None, element_shape=cv2.MORPH_CROSS):
    _, img_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if mask is not None:
        mask_bin = (mask > 0).astype(np.uint8) * 255
        img_bin = cv2.bitwise_and(img_bin, mask_bin)
    
    element = cv2.getStructuringElement(element_shape, (3, 3))
    skeleton = np.zeros_like(img_bin)

    eroded = img_bin.copy()
    
    while True:
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        eroded = cv2.erode(eroded, element)
        
        if not np.any(eroded):
            break

    element_close = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    skeleton_smooth = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, element_close)

    skeleton_final = skeleton - skeleton_smooth

    skeleton_final = cv2.bitwise_not(skeleton_final)

    return skeleton_final


# Returns 4-connected neighbors
def get_4_neighbours(x, y, shape):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
            yield (nx, ny)

# Returns 8-connected neighbors
def get_8_neighbours(x, y, shape):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                yield (nx, ny)

# Check if all active neighbors are 4-connected (stick to each other)
def are_neighbors_4_connected(img, neighbors):
    if not neighbors:
        return False
    visited = set()
    queue = deque([neighbors[0]])
    visited.add(neighbors[0])

    while queue:
        cx, cy = queue.popleft()
        for nx, ny in get_4_neighbours(cx, cy, img.shape):
            if (nx, ny) in neighbors and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return visited == set(neighbors)


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

def run_single_pass(bitmap, visualize=True):
    img = bitmap.copy()

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if np.max(img) > 1:  # likely 0-255 range
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)

    img[img != 0] = 1
    if visualize:
        visualize_step(img, "Step 2: Normalized Binary Image")

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] == 1:
                is_edge_neighbor = any(
                    0 <= x + dx < img.shape[0] and 0 <= y + dy < img.shape[1] and img[x + dx, y + dy] == 0
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if is_edge_neighbor:
                    img[x, y] = 2
                else:
                    is_corner_neighbor = any(
                        0 <= x + dx < img.shape[0] and 0 <= y + dy < img.shape[1] and img[x + dx, y + dy] == 0
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    )
                    if is_corner_neighbor:
                        img[x, y] = 3
    if visualize:
        visualize_step(img, "Step 4: After Edge and Corner Detection")

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] in [2, 3]:
                neighbors = [(nx, ny) for nx, ny in get_8_neighbours(x, y, img.shape) if img[nx, ny] != 0]
                if 2 <= len(neighbors) <= 4 and are_neighbors_4_connected(img, neighbors):
                    img[x, y] = 4
    if visualize:
        visualize_step(img, "Step 5: Marked for Removal (Label 4)")

    img[img == 4] = 0
    if visualize:
        visualize_step(img, "Step 6: Removed Pixels Marked as 4")

    N = 2
    iter_count = 0

    while True:
        changed = False

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x, y] == N:
                    weight = get_weight(img, x, y)
                    if weight in removal_weights:
                        img[x, y] = 0
                        changed = True
                    else:
                        img[x, y] = 1

        if visualize:
            visualize_step(img, f"Step 8+: Iteration {iter_count}, Label {N}")
        iter_count += 1

        if N == 3 or not changed:
            break
        N = 3

    return img



def visualize_step(img, title):
    color_map = {
        0: [255, 255, 255],  # white
        1: [0, 0, 0],        # black
        2: [255, 255, 0],    # yellow
        3: [255, 0, 0],      # red
        4: [0, 255, 0],      # green
    }

    color_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    for val, color in color_map.items():
        color_img[img == val] = color

    plt.figure(figsize=(10, 10))
    plt.imshow(color_img)
    plt.axis('off')
    plt.grid(False)
    plt.show()
    

def kmm_skeletonize(bitmap):
    current_img = bitmap.copy()
    last_changed_img = current_img.copy()
    first_iteration = True

    while True:
        next_img = run_single_pass(current_img, visualize=first_iteration)
        first_iteration = False  # Only visualize the first iteration

        if np.array_equal(current_img, next_img):
            break

        last_changed_img = current_img
        current_img = next_img

    return last_changed_img

