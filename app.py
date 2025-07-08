import streamlit as st
import numpy as np
import rasterio
import joblib
import imageio
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Forest Fire Simulation",
    page_icon="🔥",
    layout="wide"
)

# --- Load Pre-Processed Data and Model (Cached for performance) ---
@st.cache_data
def load_data():
    print("Loading data and model for the first time...")
    with rasterio.open('aligned_slope.tif') as dataset:
        slope_data = dataset.read(1)
    with rasterio.open('aligned_aspect.tif') as dataset:
        aspect_data = dataset.read(1)
    with rasterio.open('aligned_fuel.tif') as dataset:
        fuel_data = dataset.read(1)
        map_height, map_width = dataset.height, dataset.width
    
    model = joblib.load('random_forest_fire_model.joblib')
    print("Data and model loaded successfully.")
    return slope_data, aspect_data, fuel_data, model, map_height, map_width

# --- Main Simulation Function ---
def run_simulation(slope_data, aspect_data, fuel_data, model, map_height, map_width, steps, threshold):
    fire_map = fuel_data.copy().astype(np.uint8)
    ignition_row, ignition_col = 1500, 1500
    fire_map[ignition_row, ignition_col] = 40  # Burning state
    
    frames = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(steps):
        status_text.text(f"Simulating step {step + 1}/{steps}...")
        burning_cells = np.argwhere(fire_map == 40)
        newly_ignited = []

        for r, c in burning_cells:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    
                    if 0 <= nr < map_height and 0 <= nc < map_width and fire_map[nr, nc] in [10, 20, 30]:
                        features = [[slope_data[nr, nc], aspect_data[nr, nc], fuel_data[nr, nc]]]
                        prediction_prob = model.predict_proba(features)[0][1]
                        
                        if prediction_prob > threshold:
                            newly_ignited.append((nr, nc))
        
        for r_new, c_new in newly_ignited:
            fire_map[r_new, c_new] = 40
        for r_old, c_old in burning_cells:
            fire_map[r_old, c_old] = 50 # Burned out
            
        # Create a visual frame
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(fire_map, cmap='gist_heat_r')
        ax.set_title(f"Fire Spread Simulation - Step {step + 1}", fontsize=20)
        ax.axis('off')
        
        frame_path = f'frame_{step:02d}.png'
        fig.savefig(frame_path, dpi=90, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))
        progress_bar.progress((step + 1) / steps)

    gif_path = 'fire_simulation_app.gif'
    imageio.mimsave(gif_path, frames, fps=3)
    status_text.success("Simulation Complete!")
    return gif_path

# --- Streamlit App Interface ---
st.title("🔥 Forest Fire Spread Simulation using AI/ML")
st.write("""
This application uses a trained Random Forest model to simulate the spread of a forest fire in Uttarakhand, India.
The model considers terrain **slope**, **aspect**, and **fuel type** to predict the probability of a fire spreading to new areas.
""")

# Load data
try:
    slope, aspect, fuel, model, h, w = load_data()
    st.sidebar.success("Data and AI Model loaded successfully!")

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Simulation Parameters")
    sim_steps = st.sidebar.slider("Number of Simulation Steps", min_value=5, max_value=50, value=20, step=1)
    prob_threshold = st.sidebar.slider("Ignition Probability Threshold", min_value=0.1, max_value=0.9, value=0.6, step=0.05)

    # --- Main App Body ---
    if st.button("🚀 Start Simulation"):
        with st.spinner('Running AI simulation... This may take a few minutes.'):
            gif_path = run_simulation(slope, aspect, fuel, model, h, w, sim_steps, prob_threshold)
            st.image(gif_path, caption="Final Fire Spread Simulation")
            
            with open(gif_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Simulation GIF",
                    data=file,
                    file_name="fire_simulation.gif",
                    mime="image/gif"
                )
except FileNotFoundError:
    st.error("""
    **Required data files not found!**
    
    Please make sure the following files are in the same folder as `app.py`:
    - `aligned_slope.tif`
    - `aligned_aspect.tif`
    - `aligned_fuel.tif`
    - `random_forest_fire_model.joblib`
    """)