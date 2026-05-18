"""
visualization.py
----------------
Draw colored overlays on the original image to highlight
selected (green) and non-selected (red/dim) cells.
"""

import cv2
import numpy as np
from typing import List, Dict


# Color constants (RGB)
COLOR_SELECTED     = (60, 220, 100)   # Green  — passes filters
COLOR_NOT_SELECTED = (200, 60, 60)    # Red    — fails filters
COLOR_LABEL_TEXT   = (255, 255, 255)  # White  — cell index label
OVERLAY_ALPHA      = 0.45             # Transparency of the colored fill


def draw_cell_overlays(
    image: np.ndarray,
    labeled_image: np.ndarray,
    cell_metrics: List[Dict],
    show_labels: bool = True
) -> np.ndarray:
    """
    Draw a semi-transparent colored fill over each cell.
    Green = passes current filters, Red = fails filters.
    Optionally draw the cell index at its centroid.

    Parameters
    ----------
    image         : Original RGB image
    labeled_image : Integer label map from watershed
    cell_metrics  : Output of apply_filters() — must contain 'selected' key
    show_labels   : Whether to draw cell index numbers

    Returns
    -------
    output : Annotated RGB image
    """
    output = image.copy().astype(np.float32)
    overlay = image.copy().astype(np.float32)

    # Build a lookup: label → metric dict
    label_to_metric = {c["label"]: c for c in cell_metrics}

    for cell in cell_metrics:
        lbl = cell["label"]
        mask = (labeled_image == lbl)

        color = COLOR_SELECTED if cell["selected"] else COLOR_NOT_SELECTED

        # Fill cell region with color on the overlay layer
        overlay[mask] = color

    # Blend overlay with original image
    output = cv2.addWeighted(overlay, OVERLAY_ALPHA, output, 1 - OVERLAY_ALPHA, 0)
    output = np.clip(output, 0, 255).astype(np.uint8)

    # Draw cell boundary contours for clarity
    for cell in cell_metrics:
        lbl = cell["label"]
        mask = (labeled_image == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = COLOR_SELECTED if cell["selected"] else COLOR_NOT_SELECTED
        cv2.drawContours(output, contours, -1, color, 1)

    # Draw cell index labels at centroids
    if show_labels:
        for cell in cell_metrics:
            if not cell["selected"]:
                continue  # Only label selected cells to reduce clutter
            r, c = cell["centroid_r"], cell["centroid_c"]
            cv2.putText(
                output,
                str(cell["label"]),
                (c - 5, r + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                COLOR_LABEL_TEXT,
                1,
                cv2.LINE_AA
            )

    return output


def draw_binary_mask(binary: np.ndarray) -> np.ndarray:
    """
    Convert a binary mask to a 3-channel RGB image for display.
    White = cell foreground, Black = background.
    """
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return rgb


def draw_distance_map(labeled_image: np.ndarray) -> np.ndarray:
    """
    Create a false-color visualization of the labeled regions.
    Each cell gets a unique random color for easy distinction.
    """
    h, w = labeled_image.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels > 0]  # Skip background (0)

    rng = np.random.default_rng(seed=42)  # Fixed seed for consistent colors
    for lbl in unique_labels:
        color = rng.integers(80, 255, size=3).tolist()
        color_map[labeled_image == lbl] = color

    return color_map
