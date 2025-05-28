import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_minutiae(skeleton):
    """
    Detects ridge endings and bifurcations in a skeletonized binary image.
    Input image should have 1-pixel width ridges, with values 0 (black) and 1 (white).
    """
    endings = []
    bifurcations = []

    skeleton = (skeleton == 255).astype(np.uint8)

    rows, cols = skeleton.shape

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if skeleton[y, x] == 1:
                neighborhood = skeleton[y-1:y+2, x-1:x+2]
                total = np.sum(neighborhood) - 1  

                if total == 1:
                    endings.append((x, y))
                elif total >= 3:
                    bifurcations.append((x, y))

    return endings, bifurcations


def visualize_minutiae(skeleton, endings, bifurcations):
    skeleton = cv2.bitwise_not(skeleton)  # odwrócenie kolorów
    vis_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    for (x, y) in endings:
        cv2.circle(vis_image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

    for (x, y) in bifurcations:
        cv2.circle(vis_image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    # Konwersja BGR -> RGB do poprawnego wyświetlenia w matplotlib
    vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(vis_rgb)
    plt.axis('off')
    plt.title("Minutiae Detection")
    plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_direction(neighborhood, center=(0, 0)):
    """
    Estimate direction by computing the vector sum of neighboring ridge pixels.
    `neighborhood` is a binary 2D array centered on the minutia.
    `center` is the relative coordinate of the center pixel in the neighborhood.
    """
    directions = []

    rows, cols = neighborhood.shape
    cy, cx = center

    for y in range(rows):
        for x in range(cols):
            if (y == cy and x == cx) or neighborhood[y, x] == 0:
                continue
            dy = y - cy
            dx = x - cx
            directions.append((dx, dy))

    if not directions:
        return (0, 0)

    dx = sum(d[0] for d in directions) / len(directions)
    dy = sum(d[1] for d in directions) / len(directions)
    length = math.hypot(dx, dy)
    if length == 0:
        return (0, 0)

    return (dx / length, dy / length)


def detect_minutiae_with_directions(skeleton):
    endings = []
    bifurcations = []

    skeleton = (skeleton == 255).astype(np.uint8)
    rows, cols = skeleton.shape

    for y in range(2, rows - 2):
        for x in range(2, cols - 2):
            if skeleton[y, x] == 1:
                neighborhood_3x3 = skeleton[y - 1:y + 2, x - 1:x + 2]
                total = np.sum(neighborhood_3x3) - 1  # exclude center pixel

                # Use a larger 5x5 neighborhood for direction
                neighborhood_5x5 = skeleton[y - 2:y + 3, x - 2:x + 3]
                direction = get_direction(neighborhood_5x5, center=(2, 2))

                if total == 1:
                    endings.append(((x, y), direction))
                elif total >= 3:
                    bifurcations.append(((x, y), direction))

    return endings, bifurcations

def visualize_minutiae_with_directions(skeleton, endings, bifurcations):
    skeleton = cv2.bitwise_not(skeleton)  # invert colors for visibility
    vis_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    scale = 10  # arrow length

    for (x, y), (dx, dy) in endings:
        # Dot (ending): red
        #cv2.circle(vis_image, (x, y), radius=3, color=(255, 0, 0), thickness=1)
        # Arrow: orange
        tip_x = int(x + dx * scale)
        tip_y = int(y + dy * scale)
        cv2.arrowedLine(vis_image, (x, y), (tip_x, tip_y), color=(255, 165, 0), thickness=1, tipLength=0.4)

    for (x, y), (dx, dy) in bifurcations:
        # Dot (bifurcation): green
        #cv2.circle(vis_image, (x, y), radius=3, color=(0, 200, 0), thickness=1)
        # Arrow: cyan
        tip_x = int(x + dx * scale)
        tip_y = int(y + dy * scale)
        cv2.arrowedLine(vis_image, (x, y), (tip_x, tip_y), color=(0, 255, 255), thickness=1, tipLength=0.4)

    vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 12))
    plt.imshow(vis_rgb)
    plt.axis('off')
    plt.title("Minutiae Detection with Directions", fontsize=18)
    plt.tight_layout()
    plt.show()


