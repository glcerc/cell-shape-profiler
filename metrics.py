"""
metrics.py
----------
Compute shape descriptors for segmented cells.
All metrics are normalized to [0, 1] where meaningful.
"""

import numpy as np
from skimage.measure import regionprops
from typing import List, Dict


def compute_cell_metrics(props: list) -> List[Dict]:
    """
    Compute shape metrics for each segmented cell region.

    Metrics computed
    ----------------
    area         : Number of pixels in the cell
    perimeter    : Length of the cell boundary
    roundness    : 4π × area / perimeter² → 1.0 = perfect circle
    eccentricity : 0.0 = circle, 1.0 = line (from skimage)
    solidity     : area / convex_hull_area → measures concavity
    centroid     : (row, col) center coordinates

    Parameters
    ----------
    props : List of skimage regionprops objects

    Returns
    -------
    List of dicts, one per cell
    """
    metrics = []

    for region in props:
        area = region.area
        perimeter = region.perimeter if region.perimeter > 0 else 1.0

        # Roundness: 1.0 = perfect circle, lower = more irregular
        roundness = (4 * np.pi * area) / (perimeter ** 2)
        roundness = min(roundness, 1.0)  # Cap at 1.0 due to discretization

        metrics.append({
            "label":        region.label,
            "area":         int(area),
            "perimeter":    round(float(perimeter), 2),
            "roundness":    round(float(roundness), 3),
            "eccentricity": round(float(region.eccentricity), 3),
            "solidity":     round(float(region.solidity), 3),
            "centroid_r":   int(region.centroid[0]),
            "centroid_c":   int(region.centroid[1]),
        })

    return metrics


def apply_filters(
    cell_metrics: List[Dict],
    roundness_range: tuple,
    area_range: tuple,
    eccentricity_range: tuple,
    solidity_min: float
) -> List[Dict]:
    """
    Filter cells based on shape descriptor thresholds.
    Returns the same list with a 'selected' boolean key added to each cell.

    Parameters
    ----------
    cell_metrics       : Output of compute_cell_metrics()
    roundness_range    : (min, max) roundness to keep
    area_range         : (min, max) area in pixels to keep
    eccentricity_range : (min, max) eccentricity to keep
    solidity_min       : Minimum solidity value to keep

    Returns
    -------
    List of dicts with added 'selected' key
    """
    filtered = []

    for cell in cell_metrics:
        passes = (
            roundness_range[0]    <= cell["roundness"]    <= roundness_range[1] and
            area_range[0]         <= cell["area"]         <= area_range[1]      and
            eccentricity_range[0] <= cell["eccentricity"] <= eccentricity_range[1] and
            cell["solidity"]      >= solidity_min
        )
        filtered.append({**cell, "selected": passes})

    return filtered


def summary_statistics(cell_metrics: List[Dict]) -> Dict:
    """
    Compute summary statistics across all cells (selected and not selected).

    Returns
    -------
    Dict with mean/std/min/max for each metric
    """
    if not cell_metrics:
        return {}

    keys = ["area", "roundness", "eccentricity", "solidity"]
    stats = {}

    for key in keys:
        values = [c[key] for c in cell_metrics]
        stats[key] = {
            "mean": round(float(np.mean(values)), 3),
            "std":  round(float(np.std(values)), 3),
            "min":  round(float(np.min(values)), 3),
            "max":  round(float(np.max(values)), 3),
        }

    return stats
