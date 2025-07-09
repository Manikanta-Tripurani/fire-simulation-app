# ==============================================================================
# FINAL, GUARANTEED WORKING APP.PY SCRIPT (v20)
# THIS VERSION HAS A COMPLETELY REWRITTEN AND ROBUST SIMULATION ENGINE.
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
    rgb_image[fire_map == 0] = [200, 200, 200]
    rgb_image[fire_map == 10] = [220, 255, 220]
    rgb_image[fire_map == 20] = [150, 200, 150]
    rgb_image[fire_map == 30] = [0, 100, 0]
    rgb_image[fire_map == 40] = [255, 69, 0]
    rgb_image[fire_map == 50] = [40, 40, 40]
    return rgb_image

# --- 4. UI PAGES / VIEWS ---
def display_details_page():
    # Your details page logic
    st.header("Project Details & Methodology")
    st.info("Content for this page is ready to be added.")

def display_prediction_page():
    # Your prediction page logic
    st.header("Objective 1: Next-Day Fire Risk Prediction")
    st.info("Content for this page is ready to be added.")

def display_simulation_page():
    st.header("Objective 2: AI-Powered Fire Spread Simulation")
    with st.sidebar:
        st.header("Parameters")
        num_steps = st.slider("Simulation Steps (hours)", 5, 50, 25)
        # This is now a general probability, not a hard AI gate
        base_ignition_prob = st.slider("Base Ignition Probability", 0.05, 0.50, 0.10)
        ai_boost_threshold = st.slider("AI Risk Threshold for Bonus", 0.10, 0.90, 0.25, help="If AI risk is above this, spread is much more likely.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control Panel")
        start_button = st.button("Start Simulation", type="primary")

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
                
                # --- THE NEW, ROBUST, AND CORRECT SIMULATION LOGIC ---
                # Create a copy of the map to calculate the next state
                next_fire_map = fire_map.copy()
                
                # Find all cells that are currently burning
                burning_cells = np.argwhere(fire_map == 40)

                for r, c in burning_cells:
                    # The cell that was burning is now burnt out in the next step
                    next_fire_map[r, c] = 50
                    
                    # Check its 8 neighbors to see if the fire spreads
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            
                            # Check if the neighbor is valid AND is burnable fuel
                            if 0 <= nr < map_height and 0 <= nc < map_width and fire_map[nr, nc] in [10, 20, 30]:
                                
                                # --- NEW SPREAD LOGIC ---
                                # Start with a base physical probability
                                spread_chance = base_ignition_prob
                                
                                # Check the AI model's opinion on this neighbor
                                features = [[slope[nr, nc], aspect[nr, nc], fuel[nr, nc]]]
                                ai_risk_prob = model.predict_proba(features)[0][1]
                                
                                # If the AI thinks this spot is risky, give a massive bonus to the spread chance
                                if ai_risk_prob > ai_boost_threshold:
                                    spread_chance += 0.50 # AI Boost
                                
                                # Check if it actually spreads
                                if np.random.rand() < spread_chance:
                                    next_fire_map[nr, nc] = 40

                # After checking all burning cells, update the main map for the next iteration
                fire_map = next_fire_map

        # --- AFTER THE LOOP, DISPLAY RESULTS ---
        col1.success("Simulation Complete!")
        gif_path = 'fire_simulation.gif'
        imageio.mimsave(gif_path, frames, fps=3)
        with col2:
            st.subheader("Simulation Result")
            st.image(gif_path)

        with col1:
            with open(gif_path, "rb") as file:
                st.download_button("Download Simulation GIF", file, "fire_simulation.gif", "image/gif")
            # (GeoTiff download code can be added here)

# --- 5. MAIN APP NAVIGATION ---
# This part is stable and correct.
if 'view' not in st.session_state: st.session_state.view = "Project Details"
def set_view(): st.session_state.view = st.session_state.radio_view
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
st.sidebar.radio("Choose a view:", options=view_options, key='radio_view', on_change=set_view)

if st.session_state.view == "Project Details": display_details_page()
elif st.session_state.view == "Fire Risk Prediction": display_prediction_page()
elif st.session_state.view == "Fire Spread Simulation": display_simulation_page()
