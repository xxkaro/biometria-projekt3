import cv2
import numpy as np

def detect_minutiae(skeleton):
    """
    Detects ridge endings and bifurcations in a skeletonized binary image.
    Input image should have 1-pixel width ridges, with values 0 (black) and 1 (white).
    """
    endings = []
    bifurcations = []

    skeleton = (skeleton == 0).astype(np.uint8)

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
    vis_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    for (x, y) in endings:
        cv2.circle(vis_image, (x, y), radius=1, color=(0, 0, 255), thickness=-1) 

    for (x, y) in bifurcations:
        cv2.circle(vis_image, (x, y), radius=1, color=(0, 255, 0), thickness=-1) 
    

    return vis_image