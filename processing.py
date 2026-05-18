"""
processing.py
-------------
Cell segmentation pipeline using thresholding and watershed algorithm.
Supports Otsu, Adaptive, and Manual thresholding methods.
"""

import cv2
import numpy as np
from skimage import morphology, segmentation, feature
from skimage.measure import label, regionprops
import streamlit as st


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale and apply CLAHE for contrast enhancement.
    CLAHE helps reveal cell boundaries in low-contrast regions.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def apply_threshold(gray: np.ndarray, method: str, manual_value: int = 127) -> np.ndarray:
    """
    Apply binary thresholding to separate cells from background.

    Parameters
    ----------
    gray         : Grayscale input image
    method       : One of 'Otsu', 'Adaptive', 'Manual'
    manual_value : Threshold value used only when method='Manual'

    Returns
    -------
    binary : Binary mask where cells are white (255)
    """
    if method == "Otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "Adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=5
        )

    elif method == "Manual":
        _, binary = cv2.threshold(gray, manual_value, 255, cv2.THRESH_BINARY)

    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return binary


def clean_binary_mask(binary: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove small noise objects and fill holes inside cells.

    Parameters
    ----------
    binary   : Binary mask from thresholding
    min_area : Minimum pixel area to keep an object

    Returns
    -------
    cleaned : Cleaned binary mask
    """
    # Remove small noise objects
    cleaned = morphology.remove_small_objects(
        binary.astype(bool), max_size=min_area
    )
    # Fill holes inside cell bodies
    cleaned = morphology.remove_small_holes(cleaned, max_size=min_area // 2)
    return (cleaned * 255).astype(np.uint8)


@st.cache_data
def segment_cells(
    image_bytes: bytes,
    threshold_method: str,
    manual_threshold: int,
    min_area: int,
    max_area: int,
    watershed_sensitivity: float
) -> tuple:
    """
    Full segmentation pipeline: preprocess → threshold → clean → watershed → measure.
    Results are cached so re-running filters does not re-run the slow watershed step.

    Parameters
    ----------
    image_bytes           : Raw image bytes (used as cache key)
    threshold_method      : 'Otsu', 'Adaptive', or 'Manual'
    manual_threshold      : Manual threshold value (0-255)
    min_area              : Minimum cell area in pixels
    max_area              : Maximum cell area in pixels
    watershed_sensitivity : Controls peak detection distance (0.1 - 1.0)

    Returns
    -------
    labeled_image   : Integer array where each cell has a unique label
    cell_properties : List of regionprops objects with shape metrics
    binary_mask     : Final cleaned binary mask
    """
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1: Preprocess
    gray = preprocess_image(image)

    # Step 2: Threshold
    binary = apply_threshold(gray, threshold_method, manual_threshold)

    # Step 3: Clean mask using min_area only (max_area applied after labeling)
    binary_clean = clean_binary_mask(binary, min_area)

    # Step 4: Watershed to separate touching cells
    # Distance transform: each pixel's value = distance to nearest background
    distance = cv2.distanceTransform(binary_clean, cv2.DIST_L2, 5)

    # Find local maxima (cell centers) with sensitivity control
    min_distance = max(5, int(watershed_sensitivity * 30))
    local_max = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary_clean
    )

    # Create markers for watershed
    markers = np.zeros(distance.shape, dtype=np.int32)
    for i, (r, c) in enumerate(local_max):
        markers[r, c] = i + 1

    # Apply watershed
    labeled_image = segmentation.watershed(-distance, markers, mask=binary_clean)

    # Step 5: Measure region properties for each cell
    props = regionprops(labeled_image)

    # Step 6: Filter by max area (min_area already handled in clean step)
    props = [p for p in props if p.area <= max_area]

    return labeled_image, props, binary_clean
