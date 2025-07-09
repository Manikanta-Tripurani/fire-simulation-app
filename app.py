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
        
        # --- ADD THIS LINE ---
        prediction_array = np.load('prediction_array.npy')
        
        # --- AND MODIFY THIS LINE ---
        return fuel, slope, aspect, model, profile, prediction_array
    
    except Exception as e:
        # ... your error handling
        # --- AND MODIFY THIS LINE ---
        return None, None, None, None, None, None
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
    This map shows the predicted probability of a forest fire occurring in the next 24 hours...
    """)
    
    # Load just the prediction array for this view
    try:
        prediction_array = np.load('prediction_array.npy')
        prediction_image = Image.open('prediction_map.png')
        
        st.image(prediction_image, caption='Fire Risk Prediction Map', use_container_width=True)

        # --- NEW LOGIC: FIND HOTSPOT AND ADD BUTTON ---
        st.markdown("---")
        st.subheader("Automated Simulation")

        # Find the coordinates of the highest risk pixel
        hotspot_coords = np.unravel_index(np.argmax(prediction_array), prediction_array.shape)
        
        st.info(f"AI has identified the highest fire risk at coordinates: **{hotspot_coords}**")

        if st.button("Simulate Fire from Highest Risk Zone", type="primary"):
            # Save the coordinates for the simulation page to use
            st.session_state.ignition_point = hotspot_coords
            # Set a flag to automatically switch the view
            st.session_state.view = "Fire Spread Simulation"
            # Rerun the script to force the view change
            st.rerun()

    except FileNotFoundError:
        st.error("Error: `prediction_map.png` or `prediction_array.npy` not found. Please upload them.")

def display_simulation_view():
    st.header("Objective 2: Fire Spread Simulation")
    
    st.sidebar.header("Simulation Parameters")
    num_steps = st.sidebar.slider("Number of Simulation Steps (e.g., hours)", 5, 50, 20)
    ignition_prob = st.sidebar.slider("Base Ignition Probability", 0.10, 0.90, 0.40)
    
    # --- NEW WEATHER SLIDERS ---
    st.sidebar.markdown("---")
    st.sidebar.header("Weather Parameters")
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 15)
    wind_direction = st.sidebar.selectbox(
        "Wind Direction",
        ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    )
    # ... (rest of your existing display_simulation_view code)
    
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
     # ==========================================================
# PASTE THIS ENTIRE BLOCK (The new B2 code)
# ==========================================================

# --- Load data first ---
fuel, slope, aspect, model, profile, prediction_array = load_data()

if fuel is not None:
    with st.spinner('Running simulation... this may take a moment.'):
        fire_map = np.zeros_like(fuel, dtype=np.int8) # Start with a fresh map

        # --- Define wind vectors for each direction ---
        WIND_VECTORS = {
            "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
            "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1)
        }
        wind_vec = WIND_VECTORS[wind_direction]

        # --- Determine starting point (uses the hotspot if available) ---
        if 'ignition_point' in st.session_state and st.session_state.ignition_point is not None:
            start_coords = st.session_state.ignition_point
            # Use st.info inside the main column, not sidebar
            col2.info(f"Starting from high-risk point: {start_coords}")
            # Clear the ignition point so it doesn't get used again unless clicked
            st.session_state.ignition_point = None
        else:
            # Default to center if no hotspot is provided
            start_coords = (fire_map.shape[0] // 2, fire_map.shape[1] // 2)

        # Set initial fire at the starting coordinates
        fire_map[start_coords[0]-2:start_coords[0]+2, start_coords[1]-2:start_coords[1]+2] = 50 # Status: Burning
        
        frames = []
        progress_bar = st.progress(0)

        # --- THE NEW SIMULATION LOOP WITH WEATHER ---
        for step in range(num_steps):
            newly_ignited = []
            # Find all cells that are currently burning
            burning_cells = np.argwhere(fire_map == 50)
            
            for r, c in burning_cells:
                # Check all 8 neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        
                        nr, nc = r + dr, c + dc
                        
                        # Check if neighbor is within map bounds and is fuel
                        if 0 <= nr < fire_map.shape[0] and 0 <= nc < fire_map.shape[1] and fire_map[nr, nc] == 0 and fuel[nr, nc] > 0:
                            
                            # Calculate the chance of spread
                            spread_chance = ignition_prob # Start with base probability
                            
                            # Add wind bonus if the neighbor is downwind
                            is_downwind = (dr, dc) == wind_vec
                            if is_downwind:
                                wind_bonus = (wind_speed / 50.0) * 0.4 # Max 40% bonus from wind
                                spread_chance += wind_bonus
                                
                            if np.random.rand() < spread_chance:
                                newly_ignited.append((nr, nc))

            # Update the map for the next step
            if newly_ignited:
                rows, cols = zip(*newly_ignited)
                fire_map[rows, cols] = 50 # Set new cells to Burning
                
            fire_map[burning_cells[:, 0], burning_cells[:, 1]] = 100 # Set old burning cells to Burnt
            
            # --- Visualization and GIF Frame Creation ---
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(fire_map, cmap='gist_heat_r', vmin=0, vmax=100)
            ax.set_title(f"Fire Spread Simulation - Step {step + 1}/{num_steps}")
            ax.axis('off')
            
            # Update the image placeholder in the app
            image_placeholder.pyplot(fig)
            
            # Save frame for GIF
            frame_path = f"frame_{step:02d}.png"
            fig.savefig(frame_path)
            plt.close(fig) # IMPORTANT to close the figure to save memory
            frames.append(imageio.imread(frame_path))
            os.remove(frame_path) # Clean up the temporary frame file
            
            progress_bar.progress((step + 1) / num_steps)

        st.success("Simulation Complete!")
        
        # --- Create and provide download buttons ---
        with col1: # Put download buttons in the first column
            gif_path = 'fire_simulation.gif'
            imageio.mimsave(gif_path, frames, plugin="pillow", fps=2)
            
            with open(gif_path, "rb") as file:
                st.download_button(
                    label="Download Simulation GIF",
                    data=file,
                    file_name="fire_simulation.gif",
                    mime="image/gif",
                )
            
            geotiff_path = "final_fire_spread.tif"
            save_as_geotiff(fire_map, profile, geotiff_path)

            with open(geotiff_path, "rb") as file:
                 st.download_button(
                    label="Download Final Map (.tif)",
                    data=file,
                    file_name="final_fire_spread.tif",
                    mime="image/tiff"
                )
        
        # Reset the flag so the simulation doesn't run again on its own
        st.session_state.simulation_run = False
# ==========================================================
# END OF THE BLOCK TO PASTE
# ==========================================================
                    
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

# Use st.session_state to control the view
if 'view' not in st.session_state:
    st.session_state.view = "Project Details" # Default view

# Function to update the view state
def set_view():
    st.session_state.view = st.session_state.radio_view

# The radio button now updates the session state
view_options = ("Project Details", "Fire Risk Prediction", "Fire Spread Simulation")
st.sidebar.radio(
    "Choose a view:",
    options=view_options,
    key='radio_view', # Give it a key
    on_change=set_view # Call the function when it changes
)

# Display the view based on the session state
if st.session_state.view == "Project Details":
    # display_details_view() # You will need to create this page
    st.header("Project Details")
    st.info("Add your project details here as described in the success plan.")
elif st.session_state.view == "Fire Risk Prediction":
    display_prediction_view()
elif st.session_state.view == "Fire Spread Simulation":
    display_simulation_view()
