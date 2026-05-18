# Design Choices

## Segmentation Pipeline

### Why CLAHE before thresholding?
Fluorescence microscopy images often have uneven illumination.
CLAHE (Contrast Limited Adaptive Histogram Equalization) corrects this locally,
making thresholding more consistent across the image.

### Why three threshold methods?
- **Otsu**: Works well when cells and background have clearly separated intensity peaks.
- **Adaptive**: Better for images with spatially varying illumination.
- **Manual**: Gives the user full control for unusual images.

### Why watershed?
Simple thresholding cannot separate touching cells.
Watershed treats the distance transform of the binary mask as a topographic surface
and finds cell boundaries at the "ridges" between local maxima (cell centers).
The sensitivity slider controls the minimum distance between detected cell centers,
allowing the user to trade off over-splitting vs. merging.

## Caching Strategy

`@st.cache_data` is applied to `segment_cells()`.
The cache key includes all segmentation parameters (method, min/max area, sensitivity).
This means:
- Changing a **filter slider** → no re-segmentation, instant response
- Changing a **segmentation parameter** → cache miss, watershed re-runs (~1–2 s)

This separation makes the Filter & Explore tab feel responsive even on large images.

## Shape Metrics

| Metric | Formula | Range | Interpretation |
|---|---|---|---|
| Roundness | 4π·area / perimeter² | 0–1 | 1 = perfect circle |
| Eccentricity | from ellipse fit | 0–1 | 0 = circle, 1 = line |
| Solidity | area / convex hull area | 0–1 | 1 = convex shape |
| Area | pixel count | px | absolute size |

Roundness is capped at 1.0 because pixel discretization can produce perimeters
slightly shorter than the theoretical minimum, pushing the value marginally above 1.

## Visualization

Selected cells (green) and excluded cells (red) are drawn as semi-transparent
filled regions blended over the original image (alpha=0.45).
Contours are drawn at full opacity to keep cell boundaries visible.
Cell index labels are shown only for selected cells to reduce visual clutter.
