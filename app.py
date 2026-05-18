"""
app.py
------
Cell Shape Profiler — Interactive Cell Segmentation & Filtering Tool
Built with Streamlit for the Image Analysis course assignment.

Tabs
----
1. Segmentation  : Tune threshold and watershed parameters, see labeled cells
2. Filter & Explore : Apply shape-based filters, cells update live on the image
3. Metric Distributions : Histograms and scatter plot of all cell shape metrics
"""

import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from processing import segment_cells
from metrics import compute_cell_metrics, apply_filters, summary_statistics
from visualization import draw_cell_overlays, draw_binary_mask, draw_distance_map


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cell Shape Profiler",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Cell Shape Profiler")
st.caption(
    "Segment cells in fluorescence microscopy images, "
    "explore shape metrics, and filter cells by morphological properties."
)


# ── Sidebar: Image input ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Image Input")

    input_mode = st.radio(
        "Choose input source",
        ["Upload your own image", "Use sample image"],
        index=1
    )

    image_bytes = None

    if input_mode == "Upload your own image":
        uploaded = st.file_uploader(
            "Upload a grayscale or RGB microscopy image",
            type=["png", "jpg", "jpeg", "tif", "tiff"]
        )
        if uploaded is not None:
            image_bytes = uploaded.read()

    else:
        sample_choice = st.selectbox(
            "Select sample image",
            ["normal_cells.png", "crowded_cells.png"]
        )
        sample_path = f"sample_images/{sample_choice}"
        try:
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            st.warning(
                "Sample images not found. "
                "Run `python generate_samples.py` first."
            )

    st.divider()
    st.header("⚙️ Segmentation Parameters")
    st.caption("Changes here re-run the watershed algorithm.")

    threshold_method = st.selectbox(
        "Threshold method",
        ["Otsu", "Adaptive", "Manual"],
        help=(
            "Otsu: automatic global threshold. "
            "Adaptive: local threshold per region. "
            "Manual: set the value yourself."
        )
    )

    manual_threshold = 127
    if threshold_method == "Manual":
        manual_threshold = st.slider(
            "Manual threshold value", 0, 255, 127,
            help="Pixels above this value are classified as cells."
        )

    min_area = st.slider(
        "Minimum cell area (px)", 50, 500, 100,
        help="Objects smaller than this are removed as noise."
    )

    max_area = st.slider(
        "Maximum cell area (px)", 500, 5000, 2000,
        help="Objects larger than this are excluded (e.g. clumps)."
    )

    watershed_sensitivity = st.slider(
        "Watershed sensitivity", 0.1, 1.0, 0.5, step=0.05,
        help=(
            "Higher = more cell centers detected (may over-split). "
            "Lower = fewer splits (may merge touching cells)."
        )
    )

    show_cell_labels = st.checkbox("Show cell index labels", value=True)


# ── Guard: no image loaded ──────────────────────────────────────────────────────
if image_bytes is None:
    st.info("👈 Please upload an image or select a sample from the sidebar to begin.")
    st.stop()


# ── Decode image for display ────────────────────────────────────────────────────
nparr   = np.frombuffer(image_bytes, np.uint8)
img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

if img_bgr is None:
    st.error("Could not read the image. Please upload a valid PNG/JPG/TIF file.")
    st.stop()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ── Run segmentation (cached on parameters) ─────────────────────────────────────
with st.spinner("Segmenting cells…"):
    try:
        labeled_image, props, binary_mask = segment_cells(
            image_bytes        = image_bytes,
            threshold_method   = threshold_method,
            manual_threshold   = manual_threshold,
            min_area           = min_area,
            max_area           = max_area,
            watershed_sensitivity = watershed_sensitivity
        )
    except Exception as e:
        st.error(f"Segmentation failed: {e}")
        st.stop()

# Compute metrics for all segmented cells
all_metrics = compute_cell_metrics(props)

if len(all_metrics) == 0:
    st.warning(
        "No cells detected with current parameters. "
        "Try lowering the minimum area or adjusting the threshold."
    )
    st.stop()


# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🧫 Segmentation",
    "🔎 Filter & Explore",
    "📊 Metric Distributions"
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Segmentation
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Segmentation Result")
    st.caption(
        "Adjust segmentation parameters in the sidebar. "
        "Each cell is colored uniquely. The binary mask shows the "
        "threshold decision before watershed."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Image**")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.markdown("**Binary Mask** (after threshold + cleaning)")
        st.image(draw_binary_mask(binary_mask), use_container_width=True)

    with col3:
        st.markdown("**Segmented Cells** (unique color per cell)")
        st.image(draw_distance_map(labeled_image), use_container_width=True)

    # Summary metrics row
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    stats = summary_statistics(all_metrics)

    m1.metric("Total cells detected", len(all_metrics))
    m2.metric(
        "Mean roundness",
        f"{stats['roundness']['mean']:.3f}",
        help="1.0 = perfect circle"
    )
    m3.metric(
        "Mean area (px)",
        f"{stats['area']['mean']:.0f}"
    )
    m4.metric(
        "Mean eccentricity",
        f"{stats['eccentricity']['mean']:.3f}",
        help="0 = circle, 1 = line"
    )

    # Interpretation box
    with st.expander("ℹ️ How to interpret these results"):
        st.markdown("""
        - **Binary mask**: white pixels are classified as cells.
          If too much background is white, lower the manual threshold or switch to Otsu.
        - **Segmented cells**: if two touching cells appear as one color,
          increase the watershed sensitivity to split them.
        - **Minimum area**: increase this to remove noise dots;
          decrease it if small cells are being missed.
        """)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Filter & Explore
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Filter Cells by Shape")
    st.caption(
        "Move the sliders below to select cells by morphological criteria. "
        "**Green** = passes all filters. **Red** = fails at least one filter. "
        "The image updates instantly without re-running segmentation."
    )

    # ── Filter controls ───────────────────────────────────────────────
    fc1, fc2 = st.columns(2)

    with fc1:
        roundness_range = st.slider(
            "Roundness range",
            0.0, 1.0, (0.3, 1.0), step=0.01,
            help="1.0 = perfect circle. Lower values = more elongated or irregular cells."
        )
        area_range = st.slider(
            "Area range (px)",
            int(min_area), int(max_area),
            (int(min_area), int(max_area)),
            help="Keep only cells within this pixel area range."
        )

    with fc2:
        eccentricity_range = st.slider(
            "Eccentricity range",
            0.0, 1.0, (0.0, 0.9), step=0.01,
            help="0 = circle, 1 = line. Filter out highly elongated cells."
        )
        solidity_min = st.slider(
            "Minimum solidity",
            0.0, 1.0, 0.7, step=0.01,
            help="Solidity = area / convex hull area. Low values indicate concave or irregular shapes."
        )

    # ── Apply filters (fast — no re-segmentation) ─────────────────────
    filtered_metrics = apply_filters(
        all_metrics,
        roundness_range    = roundness_range,
        area_range         = area_range,
        eccentricity_range = eccentricity_range,
        solidity_min       = solidity_min
    )

    n_selected = sum(1 for c in filtered_metrics if c["selected"])
    n_total    = len(filtered_metrics)

    # ── Selection summary ─────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    s1.metric("Selected cells", n_selected)
    s2.metric("Total cells",    n_total)
    s3.metric("Selection rate", f"{100 * n_selected / max(n_total, 1):.1f}%")

    # ── Annotated image ───────────────────────────────────────────────
    annotated = draw_cell_overlays(
        image         = img_rgb,
        labeled_image = labeled_image,
        cell_metrics  = filtered_metrics,
        show_labels   = show_cell_labels
    )

    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.markdown("**Filtered overlay** — green: selected / red: excluded")
        st.image(annotated, use_container_width=True)

    with col_info:
        st.markdown("**Selected cell details**")
        selected_df = pd.DataFrame([
            {
                "ID":           c["label"],
                "Area":         c["area"],
                "Roundness":    c["roundness"],
                "Eccentricity": c["eccentricity"],
                "Solidity":     c["solidity"],
            }
            for c in filtered_metrics if c["selected"]
        ])
        if not selected_df.empty:
            st.dataframe(selected_df, use_container_width=True, height=350)
        else:
            st.info("No cells pass the current filters.")

    # Interpretation
    with st.expander("ℹ️ Filter interpretation guide"):
        st.markdown("""
        | Metric | Biological meaning |
        |---|---|
        | **Roundness** | Healthy cells tend to be rounder. Stressed or mitotic cells become more irregular. |
        | **Eccentricity** | High values indicate elongated cells, e.g. migrating or spindle-shaped cells. |
        | **Solidity** | Low solidity suggests blebbing or irregular membranes. |
        | **Area** | Very large objects may be cell clumps; very small may be debris. |
        """)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Metric Distributions
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Shape Metric Distributions")
    st.caption(
        "Histograms and scatter plot for all detected cells. "
        "Green bars/points = currently selected by filters. "
        "Red = excluded."
    )

    # Re-use filtered_metrics from Tab 2 (same session state)
    df_all = pd.DataFrame(filtered_metrics)
    df_all["Status"] = df_all["selected"].map({True: "Selected", False: "Excluded"})

    color_map = {"Selected": "#3CDC64", "Excluded": "#C83C3C"}

    # ── Histograms ────────────────────────────────────────────────────
    h1, h2 = st.columns(2)

    with h1:
        fig_round = px.histogram(
            df_all, x="roundness", color="Status",
            color_discrete_map=color_map,
            nbins=30, barmode="overlay", opacity=0.75,
            title="Roundness Distribution",
            labels={"roundness": "Roundness (0–1)"}
        )
        fig_round.update_layout(legend_title_text="Filter status")
        st.plotly_chart(fig_round, use_container_width=True)

    with h2:
        fig_area = px.histogram(
            df_all, x="area", color="Status",
            color_discrete_map=color_map,
            nbins=30, barmode="overlay", opacity=0.75,
            title="Area Distribution",
            labels={"area": "Area (pixels)"}
        )
        fig_area.update_layout(legend_title_text="Filter status")
        st.plotly_chart(fig_area, use_container_width=True)

    h3, h4 = st.columns(2)

    with h3:
        fig_ecc = px.histogram(
            df_all, x="eccentricity", color="Status",
            color_discrete_map=color_map,
            nbins=30, barmode="overlay", opacity=0.75,
            title="Eccentricity Distribution",
            labels={"eccentricity": "Eccentricity (0=circle, 1=line)"}
        )
        st.plotly_chart(fig_ecc, use_container_width=True)

    with h4:
        fig_sol = px.histogram(
            df_all, x="solidity", color="Status",
            color_discrete_map=color_map,
            nbins=30, barmode="overlay", opacity=0.75,
            title="Solidity Distribution",
            labels={"solidity": "Solidity (0–1)"}
        )
        st.plotly_chart(fig_sol, use_container_width=True)

    # ── Scatter plot: Roundness vs. Area ──────────────────────────────
    st.divider()
    fig_scatter = px.scatter(
        df_all,
        x="area", y="roundness",
        color="Status",
        color_discrete_map=color_map,
        hover_data=["label", "eccentricity", "solidity"],
        title="Roundness vs. Area — each point is one cell",
        labels={"area": "Area (px)", "roundness": "Roundness"},
        opacity=0.8
    )
    fig_scatter.update_traces(marker=dict(size=8))
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("ℹ️ Reading the scatter plot"):
        st.markdown("""
        - Points in the **top-right** are large, round cells — likely healthy and well-isolated.
        - Points in the **bottom-left** are small and irregular — possibly debris or dividing cells.
        - A cluster of green points separated from red shows your filters are capturing a distinct subpopulation.
        - Hover over any point to see its cell ID and other metrics.
        """)

    # ── Full data table ───────────────────────────────────────────────
    with st.expander("📋 Show full cell data table"):
        display_df = df_all[["label","area","roundness","eccentricity","solidity","Status"]].copy()
        display_df.columns = ["Cell ID","Area","Roundness","Eccentricity","Solidity","Status"]
        st.dataframe(display_df.sort_values("Cell ID"), use_container_width=True)
