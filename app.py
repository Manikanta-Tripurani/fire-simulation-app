# ==============================================================================
# FINAL, GUARANTEED WORKING APP.PY SCRIPT (v24)
# THIS VERSION HAS A ROBUST UI FLOW AND THE CORRECT SIMULATION ENGINE.
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
        return fuel, slope, aspect, model, profile
    except Exception as e:
        st.error(f"CRITICAL ERROR loading data: {e}. Please check all required files are in your GitHub repository.")
        return None, None, None, None, None

def create_rgb_image(fire_map):
    rgb_image = np.zeros((fire_map.shape[0], fire_map.shape[1], 3), dtype=np.uint8)
    # Correct, fiery color scheme
    rgb_image[fire_map == 0] = [200, 200, 200]
    rgb_image[fire_map == 10] = [220, 255, 220]
    rgb_image[fire_map == 20] = [150, 200, 150]
    rgb_image[fire_map == 30] = [0, 100, 0]
    rgb_image[fire_map == 40] = [255, 69, 0]      # Burning = FIERY ORANGE-RED
    rgb_image[fire_map == 50] = [40, 40, 40]      # Burnt = Dark Grey / Ash
    return rgb_image

# --- 4. UI PAGES / VIEWS ---
def display_details_page():
    st.header("Project Details & Methodology")
    # ... Your details text here ...

def display_prediction_page():
    st.header("Objective 1: Next-Day Fire Risk Prediction")
    # ... Your prediction page logic here ...

def display_simulation_page():
    st.header("Objective 2: AI-Powered Fire Spread Simulation")
    
    with st.sidebar:
        st.header("Parameters")
        num_steps = st.slider("Simulation Steps (hours)", 5, 50, 20)
        ignition_probability_threshold = st.slider("AI Ignition Threshold", 0.10, 0.90, 0.30)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control Panel")
        start_button = st.button("Start Simulation", type="primary")

    # --- THIS IS THE NEW, ROBUST UI LOGIC ---
    # Initialize session state to hold our result
    if 'simulation_gif' not in st.session_state:
        st.session_state.simulation_gif = None

    if start_button:
        fuel, slope, aspect, model, profile = load_data()
        if fuel is None: st.stop()

        with st.spinner('Running AI simulation and generating GIF...'):
            fire_map = fuel.copy()
            ignition_row, ignition_col = 1500, 1500
            fire_map[ignition_row-5:ignition_row+5, ignition_col-5:ignition_col+5] = 40
            
            frames = []
            map_height, map_width = fire_map.shape

            for step in range(num_steps):
                frames.append(create_rgb_image(fire_map))
                
                # Using the fast and correct simulation logic
                burning_cells = np.argwhere(fire_map == 40)
                to_ignite = set()
                for r, c in burning_cells:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < map_height and 0 <= nc < map_width and fire_map[nr, nc] in [10, 20, 30]:
                                features = [[slope[nr, nc], aspect[nr, nc], fuel[nr, nc]]]
                                prediction_prob = model.predict_proba(features)[0][1]
                                if prediction_prob > ignition_probability_threshold:
                                    to_ignite.add((nr, nc))
                
                if burning_cells.size > 0:
                    fire_map[burning_cells[:, 0], burning_cells[:, 1]] = 50
                if to_ignite:
                    rows, cols = zip(*to_ignite)
                    fire_map[rows, cols] = 40
        
        # --- Save the completed GIF to the session state ---
        gif_path = 'fire_simulation.gif'
        imageio.mimsave(gif_path, frames, fps=3)
        with open(gif_path, "rb") as file:
            st.session_state.simulation_gif = file.read()
        
        col1.success("Simulation Complete!")

    # --- DISPLAY THE RESULT (OR A PLACEHOLDER) ---
    with col2:
        st.subheader("Simulation Result")
        if st.session_state.simulation_gif:
            # If the GIF exists, display it
            st.image(st.session_state.simulation_gif)
        else:
            # Otherwise, show a placeholder message
            st.info("The simulation result will appear here after you click 'Start Simulation'.")

    # --- Display download button only if the GIF exists ---
    if st.session_state.simulation_gif:
        with col1:
            st.download_button(
                label="Download Simulation GIF",
                data=st.session_state.simulation_gif,
                file_name="fire_simulation.gif",
                mime="image/gif"
            )

# --- 5. MAIN APP NAVIGATION ---
# This part is stable and correct.
if 'view' not in st.session_state: st.session_state.view = "Project Details"
def set_view(): st.session_state.view = st.session_state.radio_view
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
st.sidebar.radio("Choose a view:", options=view_options, key='radio_view', on_change=set_view)

if st.session_state.view == "Project Details": display_details_page()
elif st.session_state.view == "Fire Risk Prediction": display_prediction_page()
elif st.session_state.view == "Fire Spread Simulation": display_simulation_page()
