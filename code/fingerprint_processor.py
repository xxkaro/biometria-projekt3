from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from processing_functions.normalization import normalize_image
from processing_functions.segmentation import perform_segmentation
from processing_functions.mapping import compute_orientations, draw_orientations, estimate_ridge_frequency



class FingerprintProcessor:
    def __init__(self):
        self.raw_image = None             # Original loaded image
        self.normalized_image = None      # Normalized image (illumination, contrast adjusted)
        self.segmented_image = None       # Image after segmentation (only fingerprint region)
        self.roi_mask = None              # Mask indicating fingerprint ROI
        self.orientations = None          # Orientation map (angles)
        self.freq_map = None              # Frequency map (ridge frequencies)
        self.median_freq = None           # Median frequency value
        self.directional_map = None       # Directional map visualization
        self.filtered_image = None        # Final Gabor filtered image or enhanced image

    def load_image(self, filepath):
        from PIL import Image
        self.raw_image = np.array(Image.open(filepath).convert('L'))
        print(f"Loaded image {filepath} with shape {self.raw_image.shape}")
        # Reset dependent images
        self.normalized_image = None
        self.segmented_image = None
        self.roi_mask = None
        self.orientations = None
        self.freq_map = None
        self.median_freq = None
        self.directional_map = None
        self.filtered_image = None

    def display_image(self, image=None, title="Image"):
        import matplotlib.pyplot as plt
        if image is None:
            # Default display raw_image if nothing passed
            if self.raw_image is not None:
                image = self.raw_image
            else:
                print("No image to display")
                return
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def save_image(self, image, filepath):
        if image is None:
            print("No image to save")
            return
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(filepath)
        print(f"Image saved to {filepath}")
    
    # PROCESSING FINGERPRINT IMAGES
    # separate methods for each processing step

    # 1 - grayscale transformation
    # 2 - image normalization (1 i 2 w jednym pliku normalization.py)

    # 3 - segmentation (segmentation.py)

    # 4 - directional map 
    # 5 - frequency map (4 i 5 w jednym pliku mapping.py)

    # 6 - Gabor filter (filtering.py)

    # 7 - skeletonization (thinning, two types in skeletonization.py)

    # 8 - minutiae detection (minutiae.py)

    # 9 - singularity detection (singularity.py)

    def normalize_fingerprint(self, target_mean=100, target_variance=100):
        # Normalize self.raw_image -> self.normalized_image
        if self.raw_image is None:
            print("Load image first")
            return
        self.normalized_image = normalize_image(self.raw_image, target_mean, target_variance)
        print("Fingerprint normalized")

    def fingerprint_segmentation(self):
        if self.normalized_image is None:
            print("Normalize image first")
            return
        seg_img, norm_img, mask = perform_segmentation(self.normalized_image)
        self.segmented_image = seg_img
        self.normalized_image = norm_img     # Use normalized image after segmentation as base for next steps
        self.roi_mask = mask
        print("Fingerprint segmented")

    def create_directional_map(self):
        if self.segmented_image is None or self.roi_mask is None:
            print("Segment image first")
            return
        self.orientations = compute_orientations(self.segmented_image, block_size=16, smooth=True)
        dir_map = draw_orientations(self.segmented_image, self.roi_mask, self.orientations)
        self.directional_map = dir_map
        print("Directional map created")
        self.display_image(dir_map, title="Directional Map")

    def frequency_map(self):
        if self.segmented_image is None or self.roi_mask is None or self.orientations is None:
            print("Complete previous steps first")
            return
        freq_map, median_freq = estimate_ridge_frequency(
            self.segmented_image,
            self.roi_mask,
            self.orientations,
            block_size=16,
            kernel_size=5,
            min_wavelength=5,
            max_wavelength=15
        )
        self.freq_map = freq_map
        self.median_freq = median_freq
        print("Frequency map created")
        print(f"Median frequency: {median_freq}")
        self.display_image(freq_map, title="Frequency Map")

    
