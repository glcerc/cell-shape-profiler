"""
generate_samples.py
-------------------
Generate synthetic fluorescence microscopy sample images
for demonstration purposes (no real data dependency).
Run once to populate the sample_images/ directory.
"""

import numpy as np
import cv2
import os


def generate_cell_image(
    n_cells: int = 40,
    image_size: int = 512,
    cell_radius_range: tuple = (15, 35),
    noise_level: float = 25.0,
    seed: int = 0
) -> np.ndarray:
    """
    Generate a synthetic grayscale fluorescence cell image.
    Cells are modeled as Gaussian blobs on a dark background.

    Parameters
    ----------
    n_cells           : Number of cells to place
    image_size        : Square image dimension in pixels
    cell_radius_range : (min, max) radius of cells in pixels
    noise_level       : Standard deviation of background Gaussian noise
    seed              : Random seed for reproducibility

    Returns
    -------
    image : uint8 grayscale image
    """
    rng = np.random.default_rng(seed)
    image = np.zeros((image_size, image_size), dtype=np.float32)

    for _ in range(n_cells):
        cx = rng.integers(40, image_size - 40)
        cy = rng.integers(40, image_size - 40)
        r  = rng.integers(*cell_radius_range)
        intensity = rng.uniform(150, 255)

        # Draw a Gaussian blob to simulate fluorescence signal
        for dx in range(-r * 2, r * 2):
            for dy in range(-r * 2, r * 2):
                px, py = cx + dx, cy + dy
                if 0 <= px < image_size and 0 <= py < image_size:
                    dist = np.sqrt(dx**2 + dy**2)
                    val  = intensity * np.exp(-(dist**2) / (2 * (r * 0.6)**2))
                    image[py, px] = min(255, image[py, px] + val)

    # Add Gaussian noise for realism
    noise = rng.normal(0, noise_level, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return image


def save_sample_images(output_dir: str = "sample_images"):
    """
    Generate and save two sample images:
    - normal_cells.png   : Well-separated round cells
    - crowded_cells.png  : Densely packed, touching cells (harder to segment)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample 1: Normal — well-separated cells
    img1 = generate_cell_image(n_cells=30, cell_radius_range=(15, 28), noise_level=20, seed=1)
    cv2.imwrite(os.path.join(output_dir, "normal_cells.png"), img1)

    # Sample 2: Crowded — many touching cells, harder watershed case
    img2 = generate_cell_image(n_cells=60, cell_radius_range=(12, 22), noise_level=30, seed=2)
    cv2.imwrite(os.path.join(output_dir, "crowded_cells.png"), img2)

    print(f"Sample images saved to '{output_dir}/'")


if __name__ == "__main__":
    save_sample_images()
