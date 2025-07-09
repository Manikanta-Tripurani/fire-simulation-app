# ==============================================================================
# FINAL, CORRECTED APP.PY SCRIPT (v5 - With BULLETPROOF Visualization Fix)
# FOR ISRO HACKATHON
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
st.set_page_config(
    page_title="AI Forest Fire Analysis",
    page_icon="🔥",
    layout="wide"
)

# --- 3. HELPER FUNCTIONS ---

@st.cache_data
def load_data():
    """Loads all necessary data files from the repository."""
    try:
        fuel_tif = rasterio.open('aligned_fuel.tif')
        profile = fuel_tif.profile
        fuel = fuel_tif.read(1)
        model = joblib.load('random_forest_fire_model.joblib')
        prediction_array = np.load('prediction_array.npy')
        return fuel, model, profile, prediction_array
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None, None, None

def save_as_geotiff(final_grid, profile, output_path):
    """Saves the final simulation grid as a GeoTIFF file."""
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_grid.astype(rasterio.float32), 1)

# --- NEW HELPER FUNCTION FOR VISUALIZATION ---
def create_rgb_image(fire_map):
    """Converts the fire map data array into a displayable RGB image array."""
    # Create an empty RGB image (height, width, 3 channels)
    rgb_image = np.zeros((fire_map.shape[0], fire_map.shape[1], 3), dtype=np.uint8)
    # Define colors
    unburnt_color = [255, 255, 255] # White
    burning_color = [255, 100, 0]   # Bright Orange/Red
    burnt_color = [40, 40, 40]      # Dark Grey/Black
    # Apply colors based on the fire map values
    rgb_image[fire_map == 0] = unburnt_color
    rgb_image[fire_map == 50] = burning_color
    rgb_image[fire_map == 100] = burnt_color
    return rgb_image

# --- 4. VIEWS / PAGES ---

def display_details_view():
    st.header("Project Details & Methodology")
    st.markdown("---")
    st.subheader("Problem Statement (ISRO)")
    st.info("""...""") # Add your text here
    st.subheader("Our Solution")
    st.markdown("...") # Add your text here

def display_prediction_view():
    st.header("Objective 1: Next-Day Fire Risk Prediction")
    # st.metric("Prediction Model Accuracy", "88.2 %")
    try:
        prediction_array = np.load('prediction_array.npy')
        prediction_image = Image.open('prediction_map.png')
        st.image(prediction_image, caption='Fire Risk Prediction Map', use_container_width=True)
        st.markdown("---")
        st.subheader("Automated Simulation")
        hotspot_coords = np.unravel_index(np.argmax(prediction_array), prediction_array.shape)
        st.info(f"AI has identified the highest fire risk at coordinates: **{hotspot_coords}**")
        if st.button("Simulate Fire from Highest Risk Zone", type="primary"):
            st.session_state.ignition_point = hotspot_coords
            st.session_state.view = "Fire Spread Simulation"
            st.rerun()
    except FileNotFoundError:
        st.error("Error: `prediction_map.png` or `prediction_array.npy` not found.")

def display_simulation_view():
    st.header("Objective 2: Fire Spread Simulation")
    
    st.sidebar.header("Simulation Parameters")
    num_steps = st.sidebar.slider("Number of Simulation Steps (e.g., hours)", 5, 50, 10, key="sim_steps")
    ignition_prob = st.sidebar.slider("Base Ignition Probability", 0.10, 0.90, 0.40, key="ign_prob")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Weather Parameters")
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 15, key="wind_speed")
    wind_direction = st.sidebar.selectbox("Wind Direction", ("N", "NE", "E", "SE", "S", "SW", "W", "NW"), key="wind_dir")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Control Panel")
        st.markdown("Adjust parameters and click 'Start Simulation'.")
        if st.button("Start Simulation", type="primary", key="start_sim"):
            st.session_state.simulation_run = True
    
    with col2:
        st.subheader("Simulation Visual")
        image_placeholder = st.empty()

    if 'simulation_run' in st.session_state and st.session_state.simulation_run:
        fuel, model, profile, prediction_array = load_data()
        
        if fuel is not None:
            with st.spinner('Running simulation...'):
                fire_map = np.zeros_like(fuel, dtype=np.int8)
                WIND_VECTORS = {"N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1), "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)}
                wind_vec = WIND_VECTORS[wind_direction]

                if 'ignition_point' in st.session_state and st.session_state.ignition_point is not None:
                    start_coords = st.session_state.ignition_point
                    col1.info(f"Starting from high-risk point: {start_coords}")
                    st.session_state.ignition_point = None
                else:
                    start_coords = (fire_map.shape[0] // 2, fire_map.shape[1] // 2)

                fire_map[start_coords[0]-2:start_coords[0]+2, start_coords[1]-2:start_coords[1]+2] = 50
                
                frames = []
                progress_bar = col1.progress(0)

                for step in range(num_steps):
                    # --- SIMULATION LOGIC (REMAINS THE SAME) ---
                    newly_ignited = []
                    burning_cells = np.argwhere(fire_map == 50)
                    for r, c in burning_cells:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < fire_map.shape[0] and 0 <= nc < fire_map.shape[1] and fire_map[nr, nc] == 0 and fuel[nr, nc] > 0:
                                    spread_chance = ignition_prob
                                    is_downwind = (dr, dc) == wind_vec
                                    if is_downwind:
                                        spread_chance += (wind_speed / 50.0) * 0.4
                                    if np.random.rand() < spread_chance:
                                        newly_ignited.append((nr, nc))
                    if newly_ignited:
                        rows, cols = zip(*newly_ignited)
                        fire_map[rows, cols] = 50
                    fire_map[burning_cells[:, 0], burning_cells[:, 1]] = 100
                    
                    # --- THE NEW, BULLETPROOF VISUALIZATION ---
                    # 1. Create the RGB image from the data
                    rgb_frame = create_rgb_image(fire_map)
                    # 2. Display it directly using st.image
                    image_placeholder.image(rgb_frame, caption=f"Simulation Step {step + 1}/{num_steps}", use_column_width=True)
                    # 3. Add the raw RGB frame to our list for the GIF
                    frames.append(rgb_frame)
                    
                    progress_bar.progress((step + 1) / num_steps)

                col1.success("Simulation Complete!")
                
                with col1:
                    gif_path = 'fire_simulation.gif'
                    imageio.mimsave(gif_path, frames, plugin="pillow", fps=2)
                    with open(gif_path, "rb") as file:
                        st.download_button("Download Simulation GIF", file, "fire_simulation.gif", "image/gif")
                    
                    geotiff_path = "final_fire_spread.tif"
                    save_as_geotiff(fire_map, profile, geotiff_path)
                    with open(geotiff_path, "rb") as file:
                         st.download_button("Download Final Map (.tif)", file, "final_fire_spread.tif", "image/tiff")
                
                st.session_state.simulation_run = False

# --- 5. MAIN APP NAVIGATION ---
# (This section remains unchanged)
st.sidebar.title("Project Navigation")
st.sidebar.markdown("---")
if 'view' not in st.session_state:
    st.session_state.view = "Project Details"
def set_view():
    st.session_state.view = st.session_state.radio_view
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
st.sidebar.radio("Choose a view:", options=view_options, key='radio_view', on_change=set_view, index=view_options.index(st.session_state.view))
if st.session_state.view == "Project Details":
    display_details_view()
elif st.session_state.view == "Fire Risk Prediction":
    display_prediction_view()
elif st.session_state.view == "Fire Spread Simulation":
    display_simulation_view()
