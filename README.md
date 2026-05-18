# 🔬 Cell Shape Profiler

An interactive Streamlit app for segmenting and exploring cell morphology in fluorescence microscopy images.

Built for the **Image Analysis** course assignment.

---

## What it does

1. **Segments cells** using thresholding (Otsu / Adaptive / Manual) and watershed
2. **Computes shape metrics** for each cell: roundness, eccentricity, solidity, area
3. **Filters cells live** — sliders update the overlay on the image instantly
4. **Visualizes distributions** — histograms and scatter plots for all metrics

---

## Local Setup

```bash
git clone <your-repo-url>
cd cell-shape-profiler

pip install -r requirements.txt

# Generate built-in sample images
python generate_samples.py

# Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Hugging Face Space

🔗 **Live demo:** `https://huggingface.co/spaces/glcerc/cell-shape-profiler`

---

## Project Structure

```
cell-shape-profiler/
├── app.py               # Main Streamlit application
├── processing.py        # Segmentation pipeline (threshold + watershed)
├── metrics.py           # Shape metric computation and filtering
├── visualization.py     # Colored overlay drawing
├── generate_samples.py  # Synthetic sample image generator
├── requirements.txt
├── sample_images/
│   ├── normal_cells.png
│   └── crowded_cells.png
└── docs/
    └── design_choices.md
```

---

## Known Limitations

- Watershed may over-split or under-split densely packed cells depending on sensitivity
- Very dark or low-contrast images may require Manual threshold tuning
- Large images (>2000×2000 px) may be slow; consider downsampling before upload
- Synthetic sample images do not perfectly represent real fluorescence data

---

## Screenshots

