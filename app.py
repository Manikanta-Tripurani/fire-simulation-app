# ==============================================================================
# FINAL, PUBLIC-READY APP.PY SCRIPT (v17)
# THIS VERSION IS MEMORY-EFFICIENT AND INCLUDES THE MAP LEGEND.
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

# --- 3. DATA LOADING FUNCTIONS (MEMORY EFFICIENT) ---
@st.cache_data
def load_prediction_data():
    try:
        prediction_array = np.load('prediction_array.npy')
        return prediction_array
    except Exception as e:
        st.error(f"Error loading prediction data: {e}")
        return None

@st.cache_data
def load_simulation_data():
    try:
        fuel_tif = rasterio.open('aligned_fuel.tif')
        profile = fuel_tif.profile
        fuel = fuel_tif.read(1)
        slope = rasterio.open('aligned_slope.tif').read(1)
        aspect = rasterio.open('aligned_aspect.tif').read(1)
        model = joblib.load('random_forest_fire_model.joblib')
        return fuel, slope, aspect, model, profile
    except Exception as e:
        st.error(f"Error loading simulation data files: {e}")
        return None, None, None, None, None

# --- 4. HELPER FUNCTIONS ---
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
    """Displays a color-coded legend for the simulation map."""
    st.subheader("Map Legend")
    legend_html = """
    <style>
        .legend-color-box { width: 20px; height: 20px; display: inline-block; vertical-align: middle; margin-right: 10px; border: 1px solid #444; }
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

# ==========================================================
# PASTE THIS COMPLETE FUNCTION INTO YOUR APP.PY
# ==========================================================

# ==========================================================
# PASTE THIS NEW, COMPLETE FUNCTION INTO YOUR APP.PY
# ==========================================================
def display_details_page():
    st.header("Project Details & Methodology")
    st.markdown("---")
    st.subheader("Problem Statement (ISRO)")
    st.info("""
    Uncontrolled forest fires represent a significant challenge... [and the rest of the problem statement]
    """)

    st.subheader("Our Solution: The Agni-AI Pipeline")
    st.markdown("""
    Our project is an end-to-end decision support system that moves fire management from a reactive to a **proactive** stance. It consists of a two-stage AI pipeline:
    1.  **AI-Powered Prediction:** We use a **Random Forest model** to analyze a feature stack of geospatial data from ISRO portals. This model predicts the probability of a fire starting, creating a detailed "Next-Day Fire Risk Map".
    2.  **Dynamic Simulation:** A **Cellular Automata model** then simulates the fire's spread from the AI-identified hotspots, allowing authorities to visualize scenarios based on environmental factors.
    """)

    st.subheader("Methodology & Technology Stack")
    st.markdown("""
    *   **Prediction Model:** We chose a **Random Forest** for its proven high accuracy and efficiency on tabular geospatial data. This allowed for rapid training and iteration, which is critical in a hackathon environment, while still providing robust and explainable results.
    *   **Simulation Model:** A **Cellular Automata** was chosen for its ability to model complex emergent behavior (like fire spread) from simple, computationally efficient rules.
    *   **Technology Stack:** The project is built entirely in **Python**, leveraging Scikit-learn, Rasterio, NumPy, and deployed as an interactive web application using **Streamlit**.
    """)
def display_prediction_page():
    st.header("Objective 1: Next-Day Fire Risk Prediction")
    
    # --- ADD THIS LINE HERE ---
    st.metric("Prediction Model Accuracy", "88.2 %") # IMPORTANT: Replace 88.2 % with YOUR model's actual accuracy

    try:
        prediction_image = Image.open('prediction_map.png')
        # ... rest of the code ...
        if prediction_array is None: st.stop()
        prediction_image = Image.open('prediction_map.png')
        st.image(prediction_image, caption='Fire Risk Prediction Map', use_container_width=True)
        st.markdown("---")
        hotspot_coords = np.unravel_index(np.argmax(prediction_array), prediction_array.shape)
        st.info(f"AI has identified the highest fire risk at coordinates: **{hotspot_coords}**")
        if st.button("Simulate Fire from Highest Risk Zone", type="primary"):
            st.session_state.ignition_point = hotspot_coords
            st.session_state.view = "Fire Spread Simulation"
            st.rerun()
    except Exception as e:
        st.error(f"Could not load prediction files: {e}")

def display_simulation_page():
    st.header("Objective 2: AI-Powered Fire Spread Simulation")
    with st.sidebar:
        st.header("Parameters")
       num_steps = st.slider("Simulation Duration (Hours)", 1, 12, 3, help="Set the total time for the fire spread simulation (1 to 12 hours).")
        ignition_probability_threshold = st.slider("AI Ignition Threshold", 0.10, 0.90, 0.30)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control Panel")
        start_button = st.button("Start Simulation", type="primary")
        st.markdown("---")
        create_legend() # <-- THE LEGEND IS NOW CORRECTLY PLACED HERE

    if start_button:
        fuel, slope, aspect, model, profile = load_simulation_data()
        if fuel is None: st.stop()

        with st.spinner('Running AI simulation and generating GIF...'):
            fire_map = fuel.copy()
            if 'ignition_point' in st.session_state and st.session_state.ignition_point is not None:
                start_coords = st.session_state.ignition_point
                st.session_state.ignition_point = None
            else:
                start_coords = (1500, 1500)
            
            fire_map[start_coords[0]-5:start_coords[0]+5, start_coords[1]-5:start_coords[1]+5] = 40
            
            frames = []
            map_height, map_width = fire_map.shape

            for step in range(num_steps):
                frames.append(create_rgb_image(fire_map))
                next_fire_map = fire_map.copy()
                burning_cells = np.argwhere(fire_map == 40)
                
                for r, c in burning_cells:
                    next_fire_map[r, c] = 50
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < map_height and 0 <= nc < map_width and fire_map[nr, nc] in [10, 20, 30]:
                                features = [[slope[nr, nc], aspect[nr, nc], fuel[nr, nc]]]
                                prediction_prob = model.predict_proba(features)[0][1]
                                if prediction_prob > ignition_probability_threshold:
                                    next_fire_map[nr, nc] = 40
                fire_map = next_fire_map

        # --- Display Results ---
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

# --- 6. MAIN APP NAVIGATION ---
if 'view' not in st.session_state: st.session_state.view = "Project Details"
def set_view(): st.session_state.view = st.session_state.radio_view
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
default_index = view_options.index(st.session_state.view)
st.sidebar.radio("Choose a view:", options=view_options, key='radio_view', on_change=set_view, index=default_index)

if st.session_state.view == "Project Details": display_details_page()
elif st.session_state.view == "Fire Risk Prediction": display_prediction_page()
elif st.session_state.view == "Fire Spread Simulation": display_simulation_page()
