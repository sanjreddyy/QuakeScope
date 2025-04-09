import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import plotly.express as px

# =============================================================================
# 2D Heat Map - Existing functionality (replace placeholder with your code)
# =============================================================================
def app_heatmap():
    st.header("2D Heat Map")
    st.write("**2D Heat Map visualization code goes here.**")
    # ---------------------------------
    # Your existing code for 2D heat map visualization
    # ---------------------------------


# =============================================================================
# 3D Globe - Existing functionality (replace placeholder with your code)
# =============================================================================
def app_3d_globe():
    st.header("3D Globe")
    st.write("**3D Globe visualization code goes here.**")
    # ---------------------------------
    # Your existing code for 3D globe visualization
    # ---------------------------------

# =============================================================================
# Animated Time-Lapse - New Tab for Animated Earthquake Map
# =============================================================================
def app_animated_map():
    st.header("Animated Earthquake Map")
    
    @st.cache_data
    def load_data():
        # Load dataset; ensure the CSV file is in the same folder as quakescope.py
        df = pd.read_csv("Sorted_Earthquake_1995_2025.csv")
        # Convert the 'time' column to datetime format (handles mixed formats)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        # Drop rows where conversion failed or where critical data is missing
        df = df.dropna(subset=['time', 'latitude', 'longitude', 'mag'])
        # Extract year for animation
        df['year'] = df['time'].dt.year
        return df

    df = load_data()

    # Sidebar filter for selecting a range of years
    st.subheader("Filter by Year")
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.slider("Select Year Range:", min_year, max_year, (min_year, max_year), step=1)
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    st.write(f"Showing earthquakes from **{year_range[0]}** to **{year_range[1]}**")

    # Create animated scatter geo map using Plotly Express
    fig = px.scatter_geo(
        df_filtered,
        lat='latitude',
        lon='longitude',
        color='mag',
        size='mag',
        size_max=8,
        animation_frame='year',
        projection='natural earth',
        title='Time-lapse of Global Earthquakes (1995–2025)',
        color_continuous_scale='Turbo'
    )

    # Update marker properties (area mode and opacity)
    fig.update_traces(marker=dict(sizemode='area', opacity=0.6))
    
    # Display the figure in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Main App Navigation
# =============================================================================
def main():
    st.set_page_config(page_title="QuakeScope", layout="wide")
    st.title("QuakeScope")

    # Sidebar navigation for selecting which visualization to display
    viz_option = st.sidebar.radio("Select Visualization", ("2D Heat Map", "3D Globe", "Animated Time-Lapse"))

    if viz_option == "2D Heat Map":
        app_heatmap()
    elif viz_option == "3D Globe":
        app_3d_globe()
    elif viz_option == "Animated Time-Lapse":
        app_animated_map()

if __name__ == "__main__":
    main()

