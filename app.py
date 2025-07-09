# ==============================================================================
# FINAL APP.PY SCRIPT
# ==============================================================================

# --- 1. IMPORTS ---
import streamlit as st
import numpy as np
import rasterio
from PIL import Image
import joblib
import matplotlib.pyplot as plt
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
        slope_tif = rasterio.open('aligned_slope.tif')
        aspect_tif = rasterio.open('aligned_aspect.tif')
        
        fuel = fuel_tif.read(1)
        slope = slope_tif.read(1)
        aspect = aspect_tif.read(1)
        
        # We need the metadata (profile) for saving later
        profile = fuel_tif.profile
        
        model = joblib.load('random_forest_model.joblib')
        
        return fuel, slope, aspect, model, profile
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.info("Please ensure 'aligned_fuel.tif', 'aligned_slope.tif', 'aligned_aspect.tif', and 'random_forest_model.joblib' are in your GitHub repository.")
        return None, None, None, None, None

def save_as_geotiff(final_grid, profile, output_path):
    """Saves the final simulation grid as a GeoTIFF file."""
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_grid.astype(rasterio.float32), 1)

# --- 4. VIEWS / PAGES ---

def display_prediction_view():
    st.header("Objective 1: Next-Day Fire Risk Prediction")
    st.markdown("""
    This map shows the predicted probability of a forest fire occurring in the next 24 hours. The prediction is generated using an AI model trained on terrain characteristics (slope, aspect, fuel type) from the Bhoonidhi portal.
    """)
    try:
        prediction_image = Image.open('prediction_map.png')
        st.image(prediction_image, caption='Fire Risk Prediction Map', use_column_width=True)
    except FileNotFoundError:
        st.error("Error: `prediction_map.png` not found. Please complete Step 1 of the instructions.")

def display_simulation_view():
    st.header("Objective 2: Fire Spread Simulation")
    
    st.sidebar.header("Simulation Parameters")
    num_steps = st.sidebar.slider("Number of Simulation Steps (e.g., hours)", 5, 50, 20)
    ignition_prob = st.sidebar.slider("Ignition Probability Threshold", 0.10, 0.90, 0.60)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Control Panel")
        st.markdown("""
        Adjust the parameters in the sidebar and click 'Start Simulation' to see how a fire might spread. The simulation uses a Cellular Automata model influenced by the terrain.
        """)
        if st.button("Start Simulation", type="primary"):
            st.session_state.simulation_run = True # Flag to know we've run it
    
    with col2:
        st.subheader("Simulation Visual")
        # Placeholder for the dynamic simulation image
        image_placeholder = st.empty()
        
    # --- Main Simulation Logic ---
    if 'simulation_run' in st.session_state and st.session_state.simulation_run:
        fuel, slope, aspect, model, profile = load_data()
        
        if fuel is not None:
            with st.spinner('Running simulation... this may take a moment.'):
                fire_map = np.zeros_like(fuel, dtype=np.int8)
                # Set initial ignition point (e.g., center)
                center_x, center_y = fire_map.shape[0] // 2, fire_map.shape[1] // 2
                fire_map[center_x-5:center_x+5, center_y-5:center_y+5] = 50 # Burning state
                
                frames = []
                progress_bar = st.progress(0)

                for step in range(num_steps):
                    burning_cells = np.argwhere(fire_map == 50)
                    
                    # YOUR FIRE SPREAD LOGIC GOES HERE. This is a simple placeholder.
                    # In a real model, you'd check neighbors and use fuel/slope to decide spread.
                    for r, c in burning_cells:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r + dr, c + dc
                                # Check bounds and if the neighbor cell is fuel
                                if 0 <= nr < fire_map.shape[0] and 0 <= nc < fire_map.shape[1] and fire_map[nr, nc] == 0 and fuel[nr, nc] > 0:
                                    if np.random.rand() < ignition_prob:
                                        fire_map[nr, nc] = 50 # It catches fire
                    
                    fire_map[burning_cells[:, 0], burning_cells[:, 1]] = 100 # Burnt out

                    # --- Visualization ---
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(fire_map, cmap='gist_heat_r', vmin=0, vmax=100)
                    ax.set_title(f"Fire Spread Simulation - Step {step + 1}/{num_steps}")
                    ax.axis('off')
                    
                    # Update the image placeholder in the app
                    image_placeholder.pyplot(fig)
                    
                    # Save frame for GIF
                    frame_path = f"frame_{step:02d}.png"
                    fig.savefig(frame_path)
                    plt.close(fig)
                    frames.append(imageio.imread(frame_path))
                    os.remove(frame_path) # Clean up temporary file
                    
                    progress_bar.progress((step + 1) / num_steps)

                st.success("Simulation Complete!")
                
                # --- Create and provide downloads ---
                gif_path = 'fire_simulation.gif'
                imageio.mimsave(gif_path, frames, plugin="pillow", fps=2)
                
                with open(gif_path, "rb") as file:
                    st.download_button(
                        label="Download Simulation GIF",
                        data=file,
                        file_name="fire_simulation.gif",
                        mime="image/gif",
                    )
                
                # Save and provide GeoTiff download
                geotiff_path = "final_fire_spread.tif"
                save_as_geotiff(fire_map, profile, geotiff_path)

                with open(geotiff_path, "rb") as file:
                     st.download_button(
                        label="Download Final Map (.tif)",
                        data=file,
                        file_name="final_fire_spread.tif",
                        mime="image/tiff"
                    )
                
                # Reset the flag
                st.session_state.simulation_run = False


# --- 5. MAIN APP NAVIGATION ---
st.sidebar.title("Project Navigation")
st.sidebar.markdown("---")
view = st.sidebar.radio("Choose a view:", ("Fire Risk Prediction", "Fire Spread Simulation"))

if view == "Fire Risk Prediction":
    display_prediction_view()
elif view == "Fire Spread Simulation":
    display_simulation_view()
