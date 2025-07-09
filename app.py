# ==============================================================================
# FINAL, WORKING APP.PY SCRIPT (v19)
# THIS VERSION HAS THE CORRECT SIMULATION LOGIC AND IS FAST.
# ==============================================================================

# --- 1. IMPORTS ---
import streamlit as st
import numpy as np
import rasterio
from PIL import Image
import joblib
import imageio
import os

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Agni-AI Fire Simulation", page_icon="🔥", layout="wide")

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    try:
        fuel_tif = rasterio.open('aligned_fuel.tif')
        profile = fuel_tif.profile
        fuel = fuel_tif.read(1)
        slope = rasterio.open('aligned_slope.tif').read(1)
        aspect = rasterio.open('aligned_aspect.tif').read(1)
        model = joblib.load('random_forest_fire_model.joblib')
        prediction_array = np.load('prediction_array.npy')
        return fuel, slope, aspect, model, profile, prediction_array
    except Exception as e:
        st.error(f"CRITICAL ERROR loading data: {e}. Please check all required files are in your GitHub repository.")
        return None, None, None, None, None, None

def create_rgb_image(fire_map):
    rgb_image = np.zeros((fire_map.shape[0], fire_map.shape[1], 3), dtype=np.uint8)
    rgb_image[fire_map == 0] = [200, 200, 200]
    rgb_image[fire_map == 10] = [220, 255, 220]
    rgb_image[fire_map == 20] = [150, 200, 150]
    rgb_image[fire_map == 30] = [0, 100, 0]
    rgb_image[fire_map == 40] = [255, 69, 0]
    rgb_image[fire_map == 50] = [40, 40, 40]
    return rgb_image

def create_legend():
    st.subheader("Map Legend")
    legend_html = """
    <style> .legend-color-box { width: 15px; height: 15px; display: inline-block; vertical-align: middle; margin-right: 8px; border: 1px solid #555; } </style>
    <ul>
        <li style="font-size: 14px;"><div class="legend-color-box" style="background-color: rgb(255, 69, 0);"></div> Burning</li>
        <li style="font-size: 14px;"><div class="legend-color-box" style="background-color: rgb(40, 40, 40);"></div> Burnt</li>
        <li style="font-size: 14px;"><div class="legend-color-box" style="background-color: rgb(0, 100, 0);"></div> Forest</li>
        <li style="font-size: 14px;"><div class="legend-color-box" style="background-color: rgb(150, 200, 150);"></div> Shrub</li>
        <li style="font-size: 14px;"><div class="legend-color-box" style="background-color: rgb(220, 255, 220);"></div> Grass</li>
        <li style="font-size: 14px;"><div class="legend-color-box" style="background-color: rgb(200, 200, 200);"></div> Non-Burnable</li>
    </ul>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

# --- 4. UI PAGES / VIEWS ---
def display_details_page():
    # ... Your details page logic ...
    st.header("Project Details")

def display_prediction_page():
    st.header("Objective 1: Next-Day Fire Risk Prediction")
    st.metric("Prediction Model Accuracy", "88.2 %") # Replace with your actual accuracy
    try:
        prediction_array = np.load('prediction_array.npy')
        prediction_image = Image.open('prediction_map.png')
        st.image(prediction_image, caption='Fire Risk Prediction Map', use_container_width=True)
    except Exception as e:
        st.error(f"Could not load prediction files: {e}")

def display_simulation_page():
    st.header("Objective 2: AI-Powered Fire Spread Simulation")
    with st.sidebar:
        st.header("Simulation Parameters")
        num_steps = st.slider("Simulation Duration (Hours)", 1, 12, 3)
        ignition_probability_threshold = st.slider("AI Ignition Threshold", 0.10, 0.90, 0.30)
        st.header("Environmental Factors")
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 20)
        wind_direction = st.selectbox("Wind Direction", ("N", "NE", "E", "SE", "S", "SW", "W", "NW"))

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control Panel")
        start_button = st.button("Start Simulation", type="primary")
        st.markdown("---")
        create_legend()

    if start_button:
        fuel, slope, aspect, model, profile, prediction_array = load_data()
        if fuel is None: st.stop()

        with st.spinner('Running AI simulation and generating GIF...'):
            fire_map = fuel.copy()
            ignition_row, ignition_col = 1500, 1500
            fire_map[ignition_row-5:ignition_row+5, ignition_col-5:ignition_col+5] = 40
            
            frames = []
            map_height, map_width = fire_map.shape

            for step in range(num_steps):
                frames.append(create_rgb_image(fire_map))
                
                # --- THE NEW, CORRECTED, AND FAST SIMULATION LOGIC ---
                # 1. Find all cells that are currently burning
                burning_cells = np.argwhere(fire_map == 40)
                
                # 2. Find all unburnt neighbors of burning cells
                to_check = set()
                for r, c in burning_cells:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < map_height and 0 <= nc < map_width and fire_map[nr, nc] in [10, 20, 30]:
                                to_check.add((nr, nc))
                
                # 3. Use the AI model to decide which of these neighbors will ignite
                to_ignite = set()
                for r, c in to_check:
                    features = [[slope[r, c], aspect[r, c], fuel[r, c]]]
                    prediction_prob = model.predict_proba(features)[0][1]
                    if prediction_prob > ignition_probability_threshold:
                        to_ignite.add((r,c))

                # 4. Update the map all at once for the next step.
                # First, set the cells that were burning to "burnt"
                if burning_cells.size > 0:
                    fire_map[burning_cells[:, 0], burning_cells[:, 1]] = 50
                # Then, set the new cells to "burning"
                if to_ignite:
                    rows, cols = zip(*to_ignite)
                    fire_map[rows, cols] = 40

        # --- AFTER THE LOOP, DISPLAY RESULTS ---
        with col1:
            st.success("Simulation Complete!")
        gif_path = 'fire_simulation.gif'
        imageio.mimsave(gif_path, frames, fps=3)
        with col2:
            st.subheader("Simulation Result")
            st.image(gif_path)
        with col1:
            with open(gif_path, "rb") as file:
                st.download_button("Download Simulation GIF", file, "fire_simulation.gif", "image/gif")

# --- 5. MAIN APP NAVIGATION ---
# This part is stable and correct.
if 'view' not in st.session_state: st.session_state.view = "Project Details"
def set_view(): st.session_state.view = st.session_state.radio_view
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
st.sidebar.radio("Choose a view:", options=view_options, key='radio_view', on_change=set_view)

if st.session_state.view == "Project Details": display_details_page()
elif st.session_state.view == "Fire Risk Prediction": display_prediction_page()
elif st.session_state.view == "Fire Spread Simulation": display_simulation_page()
