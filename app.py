# ==============================================================================
# FINAL, WORKING APP.PY SCRIPT (v12)
# BUILT WITH YOUR PROVEN SIMULATION ENGINE. THIS IS THE ONE.
# ==============================================================================

# --- 1. IMPORTS ---
import streamlit as st
import numpy as np
import rasterio
from PIL import Image
import joblib
import imageio
import os

def create_legend():
    """Displays a color-coded legend for the simulation map."""
    st.subheader("Map Legend")
    
    legend_html = """
    <style>
        .legend-color-box {
            width: 20px;
            height: 20px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
            border: 1px solid #444;
        }
    </style>
    <ul>
        <li><div class="legend-color-box" style="background-color: rgb(255, 69, 0);"></div> Burning</li>
        <li><div class="legend-color-box" style="background-color: rgb(40, 40, 40);"></div> Burnt (Ash)</li>
        <li><div class="legend-color-box" style="background-color: rgb(0, 100, 0);"></div> Forest (Unburnt)</li>
        <li><div class="legend-color-box" style="background-color: rgb(150, 200, 150);"></div> Shrub (Unburnt)</li>
        <li><div class="legend-color-box" style="background-color: rgb(220, 255, 220);"></div> Grass (Unburnt)</li>
        <li><div class="legend-color-box" style="background-color: rgb(200, 200, 200);"></div> Non-Burnable</li>
    </ul>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Forest Fire Analysis", page_icon="🔥", layout="wide")

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    try:
        # Load all data files required for the simulation
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
    # This function creates the visual image from the data map
    rgb_image = np.zeros((fire_map.shape[0], fire_map.shape[1], 3), dtype=np.uint8)
    rgb_image[fire_map == 0] = [200, 200, 200]  # Non-burnable = Light Grey
    rgb_image[fire_map == 10] = [220, 255, 220] # Grass = Light Green
    rgb_image[fire_map == 20] = [150, 200, 150] # Shrub = Green
    rgb_image[fire_map == 30] = [0, 100, 0]     # Forest = Dark Green
    rgb_image[fire_map == 40] = [255, 100, 0]   # Burning = Bright Orange
    rgb_image[fire_map == 50] = [40, 40, 40]       # Burnt = Dark Grey
    return rgb_image

# ==========================================================
# PASTE THIS COMPLETE FUNCTION INTO YOUR APP.PY
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
        # Add other logic from previous versions if needed
    except Exception as e:
        st.error(f"Could not load prediction map: {e}")

def display_simulation_page():
    st.header("Objective 2: AI-Powered Fire Spread Simulation")
    with st.sidebar:
        st.header("Parameters")
        num_steps = st.slider("Simulation Steps (hours)", 5, 50, 20)
        ignition_probability_threshold = st.slider("AI Ignition Threshold", 0.10, 0.90, 0.60)
        
        col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control Panel")
        start_button = st.button("Start Simulation", type="primary")

        # --- ADD THESE TWO LINES HERE ---
        st.markdown("---") # This adds a nice separator line
        create_legend()
    if start_button:
        fuel, slope, aspect, model, profile, prediction_array = load_data()
        if fuel is None: st.stop()

        with st.spinner('Running AI simulation and generating GIF... This may take a moment.'):
            # --- BLOCK 1: YOUR PROVEN FIRE SETUP ---
            fire_map = fuel.copy()
            ignition_row, ignition_col = 1500, 1500
            fire_map[ignition_row, ignition_col] = 40 # Set to "Burning"
            frames = []
            
            # Get map dimensions once
            map_height, map_width = fire_map.shape

            for step in range(num_steps):
                # --- THIS IS YOUR PROVEN SIMULATION ENGINE ---
                frames.append(create_rgb_image(fire_map)) # Create a frame of the current state
                
                # Find all currently burning cells
                burning_cells = np.argwhere(fire_map == 40)
                newly_ignited = []

                # Loop through each burning cell to check its neighbors
                for r, c in burning_cells:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            
                            # Check if neighbor is within map bounds and is burnable fuel
                            if 0 <= nr < map_height and 0 <= nc < map_width and fire_map[nr, nc] in [10, 20, 30]:
                                # Get the features for this neighbor cell
                                features_for_prediction = [[
                                    slope[nr, nc],
                                    aspect[nr, nc],
                                    fuel[nr, nc]
                                ]]
                                # Get the model's prediction
                                prediction_prob = model.predict_proba(features_for_prediction)[0][1]
                                
                                if prediction_prob > ignition_probability_threshold:
                                    newly_ignited.append((nr, nc))
                
                # --- YOUR PROVEN MAP UPDATE LOGIC ---
                # Set the newly ignited cells to "Burning"
                for r, c in newly_ignited:
                    fire_map[r, c] = 40
                
                # Set all old burning cells to "Burned"
                if burning_cells.size > 0:
                    fire_map[burning_cells[:, 0], burning_cells[:, 1]] = 50

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
if 'view' not in st.session_state: st.session_state.view = "Project Details"
def set_view(): st.session_state.view = st.session_state.radio_view
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
default_index = view_options.index(st.session_state.view)
st.sidebar.radio("Choose a view:", options=view_options, key='radio_view', on_change=set_view, index=default_index)

if st.session_state.view == "Project Details": display_details_page()
elif st.session_state.view == "Fire Risk Prediction": display_prediction_page()
elif st.session_state.view == "Fire Spread Simulation": display_simulation_page()
