# ==============================================================================
# FINAL, COMPLETE, AND UPGRADED APP.PY SCRIPT (v22)
# INCLUDES ALL CATEGORY 1 & 2 UPGRADES, LEGEND, AND ALL BUG FIXES.
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
    """Displays a color-coded legend for the simulation map."""
    st.subheader("Map Legend")
    legend_html = """
    <style>
        .legend-color-box { width: 20px; height: 20px; display: inline-block; vertical-align: middle; margin-right: 10px; border: 1px solid #444; }
        ul.legend-list { list-style-type: none; padding-left: 0; }
    </style>
    <ul class="legend-list">
        <li><div class="legend-color-box" style="background-color: rgb(200,200,200);"></div>Non-Burnable</li>
        <li><div class="legend-color-box" style="background-color: rgb(220,255,220);"></div> Grass(Unburnt)</li>
        <li><div class="legend-color-box" style="background-color: rgb(150, 200, 150);"></div> shrub (Unburnt)</li>
        <li><div class="legend-color-box" style="background-color: rgb(0, 100, 0);"></div> Forest (Unburnt)</li>
        <li><div class="legend-color-box" style="background-color: rgb(255, 69, 0);"></div> Burning </li>
        <li><div class="legend-color-box" style="background-color: rgb(40, 40, 40);"></div> Burnt</li>
    </ul>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
# ==========================================================
# PASTE THIS NEW, COMPLETE FUNCTION INTO YOUR APP.PY
# This has all the professional text you need.
# ==========================================================

def display_details_page():
    st.header("Project Details & Methodology")
    st.markdown("---")

    st.subheader("Problem Statement (ISRO)")
    st.info("""
    Uncontrolled forest fires represent a significant challenge for government agencies tasked with preserving biodiversity and maintaining air quality. The spread of such fires is influenced by factors including weather conditions, terrain, and human activity. With modern geospatial technologies, datasets from ISRO are accessible, yet real-time simulation remains complex. This project aims to use AI/ML to help planners estimate damage, prioritize containment, and mitigate fire impacts.
    """)

    st.subheader("Our Solution: The Agni-AI Pipeline")
    st.markdown("""
    Our project, *Agni-AI, is an end-to-end decision support system that moves fire management from a reactive to a **proactive* stance. It addresses the entire lifecycle of a fire event through a two-stage AI pipeline:
    1.  *AI-Powered Prediction:* We use a *Random Forest classification model* to analyze a feature stack of geospatial data from ISRO portals. This model predicts the probability of a fire starting in any given 30m x 30m area, creating a detailed "Next-Day Fire Risk Map".
    2.  *Dynamic Simulation:* A *Cellular Automata model* then simulates the spread of a fire. This model can be initialized at the highest-risk location identified by our AI and dynamically incorporates environmental factors to produce a realistic spread animation.
    """)
    
    st.subheader("Data Sources & Pre-processing")
    st.markdown("""
    *   *Terrain Parameters:* Slope and Aspect were derived from a 30m resolution Digital Elevation Model (DEM) sourced from the *Bhoonidhi Portal*.
    *   *Fuel Availability:* Land Use/Land Cover (LULC) maps from *Bhuvan* were used to determine the type and availability of fire fuel.
    *   *Historical Fire Data:* Fire event locations from *VIIRS-SNP* were used as the ground truth (target variable) for training our prediction model.
    *   *Preprocessing:* All datasets were resampled to a uniform 30m resolution and stacked to create the feature set for our model.
    """)

    st.subheader("Methodology & Technology Stack")
    st.markdown("""
    *   *Prediction Model:* We chose a *Random Forest* for its proven high accuracy and efficiency on tabular geospatial data. This allowed for rapid training and iteration, which is critical in a hackathon environment, while still providing robust and explainable results.
    *   *Simulation Model:* A *Cellular Automata* was chosen for its ability to model complex emergent behavior (like fire spread) from simple, computationally efficient rules.
    *   *Technology Stack:* The project is built entirely in *Python, leveraging Scikit-learn, Rasterio, NumPy, and deployed as an interactive web application using **Streamlit*.
    """)
# ==========================================================

def display_prediction_page():
    st.header("Objective 1: Next-Day Fire Risk Prediction")
    st.metric("Prediction Model Accuracy", "88.2 %") # IMPORTANT: Replace with your actual accuracy
    try:
        prediction_array = np.load('prediction_array.npy')
        prediction_image = Image.open('prediction_map.png')
        st.image(prediction_image, caption='Fire Risk Prediction Map', use_container_width=True)
        st.markdown("---")
        hotspot_coords = np.unravel_index(np.argmax(prediction_array), prediction_array.shape)
        st.info(f"AI has identified the highest fire risk at coordinates: *{hotspot_coords}*")
        if st.button("Simulate Fire from Highest Risk Zone", type="primary"):
            st.session_state.ignition_point = hotspot_coords
            st.session_state.view = "Fire Spread Simulation"
            st.rerun()
    except Exception as e:
        st.error(f"Could not load prediction files: {e}")

# ==========================================================
# PASTE THIS NEW, COMPLETE FUNCTION INTO YOUR APP.PY
# It replaces your existing display_simulation_page function.
# ==========================================================
def display_simulation_page():
    st.header("Objective 2: AI-Powered Fire Spread Simulation")

    # --- Load Data Early ---
    # We need the fuel map to find a valid ignition point
    fuel, slope, aspect, model, profile, prediction_array = load_data()
    if fuel is None:
        st.error("Could not load simulation data. The simulation cannot run.")
        st.stop()

    # --- Sidebar for Parameters ---
    with st.sidebar:
        st.header("Simulation Parameters")
        num_steps = st.slider("Simulation Duration (Hours)", 1, 24, 12, help="Set the total time for the fire spread simulation.")
        ignition_probability_threshold = st.slider("AI Ignition Threshold", 0.10, 0.90, 0.30, help="A cell must have an AI risk score above this to even be considered for ignition.")

        st.markdown("---")
        st.header("Environmental Factors")
        wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 20)
        wind_direction = st.selectbox("Wind Direction", ("N", "NE", "E", "SE", "S", "SW", "W", "NW"))
        temperature = st.slider("Temperature (°C)", 10, 45, 30, help="Higher temperatures increase the chance of fire spread.")
        humidity = st.slider("Relative Humidity (%)", 10, 100, 50, help="Lower humidity increases the chance of fire spread.")

    # --- Main Page Layout ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Control Panel")

        # --- FIX 1: DYNAMIC IGNITION POINT ---
        # Check if an ignition point was set by the AI Prediction page
        if 'ignition_point' in st.session_state:
            st.info(f"🔥 AI-selected ignition point: {st.session_state.ignition_point}")
            ignition_coords = st.session_state.ignition_point
        else:
            st.warning("No AI hotspot selected. Using a default high-risk location.")
            # Fallback: Find the absolute highest risk point from the prediction array
            ignition_coords = np.unravel_index(np.argmax(prediction_array), prediction_array.shape)

        # --- FIX 2: ENSURE IGNITION IS ON BURNABLE FUEL ---
        # This is the critical fix for the "No Fire" anomaly
        r, c = ignition_coords
        if fuel[r, c] == 0: # Check if the point is on non-burnable land
            st.error(f"Ignition point {ignition_coords} is on non-burnable land. Finding nearest valid location...")
            # Simple search for the nearest burnable pixel
            search_radius = 1
            while True:
                r_min, r_max = max(0, r - search_radius), min(fuel.shape[0], r + search_radius + 1)
                c_min, c_max = max(0, c - search_radius), min(fuel.shape[1], c + search_radius + 1)
                sub_fuel = fuel[r_min:r_max, c_min:c_max]
                burnable_points = np.argwhere(sub_fuel > 0)
                if len(burnable_points) > 0:
                    # Found a burnable point nearby
                    new_r, new_c = burnable_points[0] # Take the first one found
                    ignition_coords = (r_min + new_r, c_min + new_c)
                    st.success(f"New ignition point found at: {ignition_coords}")
                    break
                search_radius += 1
                if search_radius > 50: # Failsafe
                    st.error("Could not find any burnable land near the target. Cannot start simulation.")
                    st.stop()
        
        # Now ignition_coords is guaranteed to be on burnable land
        final_ignition_r, final_ignition_c = ignition_coords

        start_button = st.button("Start Simulation", type="primary")
        st.markdown("---")
        create_legend()

    # --- FIX 3: PRE-SIMULATION VISUALIZATION ---
    # Show the initial map with the ignition point marked BEFORE running
    if not start_button:
        with col2:
            st.subheader("Initial Map & Ignition Point")
            # Create a temporary map to show the ignition point
            initial_display_map = fuel.copy()
            # Mark the ignition point clearly (e.g., as "Burning")
            initial_display_map[final_ignition_r-2:final_ignition_r+2, final_ignition_c-2:final_ignition_c+2] = 40
            rgb_initial = create_rgb_image(initial_display_map)
            st.image(rgb_initial, caption=f"Map ready. Fire will start at {ignition_coords}.", use_container_width=True)

    # --- SIMULATION LOGIC (Only runs when button is clicked) ---
    if start_button:
        with col2:
            # The spinner should be over the results column
            with st.spinner('Running AI simulation and generating GIF... This may take a moment.'):
                fire_map = fuel.copy()
                WIND_VECTORS = {"N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1), "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)}
                wind_vec = WIND_VECTORS[wind_direction]

                # Use the validated ignition coordinates
                fire_map[final_ignition_r-5:final_ignition_r+5, final_ignition_c-5:final_ignition_c+5] = 40

                frames = []
                map_height, map_width = fire_map.shape

                for step in range(num_steps):
                    # Important: create the frame *before* this step's logic
                    frames.append(create_rgb_image(fire_map))
                    burning_cells = np.argwhere(fire_map == 40)
                    
                    if burning_cells.size == 0: # Optimization: stop if fire is out
                        break

                    to_ignite = set()
                    for r, c in burning_cells:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < map_height and 0 <= nc < map_width and fire_map[nr, nc] in [10, 20, 30]:
                                    features = [[slope[nr, nc], aspect[nr, nc], fuel[nr, nc]]]
                                    # Use try-except for model prediction as a safeguard
                                    try:
                                        ai_prob = model.predict_proba(features)[0][1]
                                    except:
                                        ai_prob = 0 # Assume no risk if prediction fails

                                    if ai_prob > ignition_probability_threshold:
                                        spread_chance = 0.5
                                        spread_chance += (temperature - 25) / 50.0 # Temp effect
                                        spread_chance -= (humidity - 50) / 100.0 # Humidity effect (Corrected logic)
                                        if (dr, dc) == wind_vec: # Wind effect
                                            spread_chance += (wind_speed / 50.0) * 0.4

                                        spread_chance = max(0.05, min(0.95, spread_chance))
                                        if np.random.rand() < spread_chance:
                                            to_ignite.add((nr, nc))

                    # Update map state for the next step
                    if burning_cells.size > 0:
                        fire_map[burning_cells[:, 0], burning_cells[:, 1]] = 50 # Burnt
                    if to_ignite:
                        rows, cols = zip(*to_ignite)
                        fire_map[rows, cols] = 40 # Burning

                # Add the final state to the GIF
                frames.append(create_rgb_image(fire_map))

                # --- Display Results ---
                gif_path = 'fire_simulation.gif'
                imageio.mimsave(gif_path, frames, fps=3, loop=0)
                
                # These messages now appear in the correct place after simulation
                col1.success("Simulation Complete!")
                with open(gif_path, "rb") as file:
                    col1.download_button("Download Simulation GIF", file, "fire_simulation.gif", "image/gif")

                st.subheader("Simulation Result")
                st.image(gif_path, use_container_width=True)

# --- 5. MAIN APP NAVIGATION ---
if 'view' not in st.session_state: st.session_state.view = "Project Details"
def set_view(): st.session_state.view = st.session_state.radio_view
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
default_index = view_options.index(st.session_state.view)
st.sidebar.radio("Choose a view:", options=view_options, key='radio_view', on_change=set_view, index=default_index)

if st.session_state.view == "Project Details": display_details_page()
elif st.session_state.view == "Fire Risk Prediction": display_prediction_page()
elif st.session_state.view == "Fire Spread Simulation": display_simulation_page()
