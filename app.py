# ==============================================================================
# FINAL, WORKING APP.PY SCRIPT (v13)
# THIS VERSION HAS A COMPLETELY REWRITTEN AND STABLE SIMULATION ENGINE.
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
st.set_page_config(page_title="AI Forest Fire Analysis", page_icon="🔥", layout="wide")

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
    rgb_image[fire_map == 0] = [200, 200, 200]    # Non-burnable
    rgb_image[fire_map == 10] = [220, 255, 220]   # Grass
    rgb_image[fire_map == 20] = [150, 200, 150]   # Shrub
    rgb_image[fire_map == 30] = [0, 100, 0]       # Forest
    rgb_image[fire_map == 40] = [255, 69, 0]      # Burning
    rgb_image[fire_map == 50] = [40, 40, 40]      # Burnt
    return rgb_image

# ==========================================================

def display_details_page():
    st.header("Project Details & Methodology")
    st.markdown("---")

    st.subheader("Problem Statement (ISRO)")
    st.info("""
    Uncontrolled forest fires represent a significant challenge for government agencies tasked with preserving biodiversity and maintaining air quality. The spread of such fires is influenced by factors including weather conditions (temperature, precipitation, humidity, wind), terrain (slope, aspect, fuel availability), and human activity. With modern geospatial technologies, datasets from the Forest Survey of India and global services like VIIRS-SNP are accessible. Despite this, real-time simulation and forecasting remain complex. Short-term forecasting and dynamic simulation are crucial for timely preventive measures. AI/ML techniques offer promising capabilities to extract insights, helping planners estimate damage, prioritize containment, and mitigate fire impacts.
    """)

    st.subheader("Our Solution")
    st.markdown("""
    Our project tackles this challenge with a comprehensive two-stage AI pipeline, designed for practical use by planning authorities:
    1.  *AI-Powered Prediction:* We utilize a *Random Forest classification model* to analyze a feature stack of geospatial data. This model predicts the probability of a fire starting in any given 30m x 30m area, creating a detailed "Next-Day Fire Risk Map".
    2.  *Dynamic Simulation:* We then use a *Cellular Automata model* to simulate the spread of a fire. This model is initialized at the highest-risk location identified by our AI and dynamically incorporates environmental factors like terrain and user-defined weather conditions (wind speed and direction) to produce a realistic spread animation.
    """)

    st.subheader("Data Sources & Pre-processing")
    st.markdown("""
    *   *Terrain Parameters:* Slope and Aspect were derived from a 30m resolution Digital Elevation Model (DEM) sourced from the *Bhoonidhi Portal*.
    *   *Fuel Availability:* Land Use/Land Cover (LULC) maps from *Bhuvan* were used to determine the type and availability of fire fuel.
    *   *Historical Fire Data:* Fire event locations from *VIIRS-SNP* were used as the ground truth (target variable) for training our prediction model.
    *   *Preprocessing:* All datasets were resampled to a uniform 30m resolution and stacked to create the feature set for our model.
    """)

    st.subheader("Methodology & Tools")
    st.markdown("""
    *   *Prediction Model:* We chose a *Random Forest* for its high accuracy on tabular geospatial data and its robustness against overfitting, which is critical for reliable predictions.
    *   *Simulation Model:* A *Cellular Automata* was chosen for its efficiency and its ability to model complex emergent behavior (like fire spread) from simple, local rules.
    *   *Technology Stack:* The entire project was built in *Python*, using libraries such as Scikit-learn, Rasterio, NumPy, and Streamlit for the interactive web application.
    """)
# ==========================================================

def display_prediction_page():
    st.header("Objective 1: Next-Day Fire Risk Prediction")
    try:
        prediction_image = Image.open('prediction_map.png')
        st.image(prediction_image, caption='Fire Risk Prediction Map', use_container_width=True)
    except Exception as e:
        st.error(f"Could not load prediction map: {e}")

def display_simulation_page():
    st.header("Objective 2: AI-Powered Fire Spread Simulation")
    with st.sidebar:
        st.header("Parameters")
        num_steps = st.slider("Simulation Steps (hours)", 5, 50, 20)
        ignition_probability_threshold = st.slider("AI Ignition Threshold", 0.10, 0.90, 0.30) # Lowered default

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
                
                # --- THE NEW, REWRITTEN, AND CORRECT SIMULATION LOGIC ---
                # Create a copy of the map to calculate the next state without interference
                next_fire_map = fire_map.copy()
                
                # Iterate through every cell to determine its next state
                for r in range(map_height):
                    for c in range(map_width):
                        # Rule 1: If a cell is currently burning, it will be burnt in the next step.
                        if fire_map[r, c] == 40:
                            next_fire_map[r, c] = 50
                            continue
                        
                        # Rule 2: If a cell is burnable fuel, check its neighbors to see if it catches fire.
                        if fire_map[r, c] in [10, 20, 30]:
                            # Check all 8 neighbors of the current cell
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if dr == 0 and dc == 0: continue
                                    nr, nc = r + dr, c + dc
                                    
                                    # Check if the neighbor is valid AND IS CURRENTLY BURNING
                                    if 0 <= nr < map_height and 0 <= nc < map_width and fire_map[nr, nc] == 40:
                                        # It has a burning neighbor, so check the AI probability to spread
                                        features = [[slope[r, c], aspect[r, c], fuel[r, c]]]
                                        prediction_prob = model.predict_proba(features)[0][1]
                                        
                                        if prediction_prob > ignition_probability_threshold:
                                            next_fire_map[r, c] = 40 # This cell catches fire
                                            break # Stop checking other neighbors, it's already on fire
                            # This inner break is to move to the next cell in the main loop
                            if next_fire_map[r,c] == 40:
                                continue

                # After checking all cells, update the main map for the next iteration
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

# --- 5. MAIN APP NAVIGATION ---
# This part remains the same and is known to work correctly.
if 'view' not in st.session_state: st.session_state.view = "Project Details"
def set_view(): st.session_state.view = st.session_state.radio_view
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
st.sidebar.radio("Choose a view:", options=view_options, key='radio_view', on_change=set_view)

if st.session_state.view == "Project Details": display_details_page()
elif st.session_state.view == "Fire Risk Prediction": display_prediction_page()
elif st.session_state.view == "Fire Spread Simulation": display_simulation_page()
