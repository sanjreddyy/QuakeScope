import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import io
from scipy.ndimage import gaussian_filter
import base64
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="QuakeScope 🌍", layout="wide")
st.markdown("""
    <style>
        /* Hide top-right GitHub & menu */
        #MainMenu, header, .stDeployButton, .st-emotion-cache-6qob1r {
            visibility: hidden;
        }

        /* Hide bottom right watermark & user info */
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Custom styles
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        h1 { text-align: center; font-size: 3rem; color: #4b4b4b; }
        h3 { font-size: 1.5rem; color: #444; }
        .sidebar .sidebar-content { background-color: #f0f2f6; }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("QuakeScope/Earthquake_1995_2025.csv")

    df['year'] = pd.to_datetime(df['time'], errors='coerce', utc=True).dt.year
    return df.dropna(subset=['latitude', 'longitude', 'depth', 'mag', 'year'])

df = load_data()

# App header
st.markdown("<h1>QuakeScope 🌍</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📘 About", "📊 Visualizations"])

# 📘 ABOUT TAB
with tab1:
    st.markdown("""
    ### Welcome to QuakeScope  
    This interactive tool helps visualize global earthquake data from **1995 to 2025**.
    
    **What You Can Explore:**
    - 🗺️ 2D Heatmap of Earthquake Density
    - 🌐 3D Interactive Globe Visualization
    
    **Fields Used**: Latitude, Longitude, Magnitude, Depth, Year  
    **Powered by**: Streamlit, Cartopy, Plotly, Pandas, Matplotlib
    """)
    st.markdown("---")

# 📊 VISUALIZATION TAB
with tab2:
    vis_option = st.radio("Choose Visualization", ["🗺️ 2D Heatmap", "🌐 3D Globe"], horizontal=True)

    # === 2D HEATMAP ===
    if vis_option == "🗺️ 2D Heatmap":
        st.markdown("### 2D Earthquake Density Map")

        with st.sidebar:
            st.header("🗺️ Heatmap Filters")
            year_range = st.slider("Year Range", 1995, 2025, (1995, 2025), key="heatmap_year")
            depth_range = st.slider("Depth Range (km)", int(df['depth'].min()), int(df['depth'].max()), (0, 700), key="heatmap_depth")
            mag_range = st.slider("Magnitude Range", float(df['mag'].min()), float(df['mag'].max()), (5.0, 9.5), step=0.1, key="heatmap_mag")
            colormap = st.selectbox("Color Gradient", ['YlOrRd', 'OrRd', 'YlGnBu', 'PuRd', 'Reds', 'Blues', 'YlGn'], key="heatmap_colormap")

        @st.cache_data
        def get_filtered_heatmap(df, year_range, depth_range, mag_range):
            return df[
                (df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
                (df['depth'] >= depth_range[0]) & (df['depth'] <= depth_range[1]) &
                (df['mag'] >= mag_range[0]) & (df['mag'] <= mag_range[1])
            ]

        filtered = get_filtered_heatmap(df, year_range, depth_range, mag_range)

        lon_bins = np.linspace(-180, 180, 360)
        lat_bins = np.linspace(-90, 90, 180)
        heatmap, _, _ = np.histogram2d(filtered['latitude'], filtered['longitude'], bins=[lat_bins, lon_bins])
        smoothed = gaussian_filter(heatmap, sigma=2.5)
        log_smoothed = np.log1p(smoothed)
        lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)

        fig = plt.figure(figsize=(16, 8))  # Slightly smaller for faster render
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor='white')
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f0ff')

        mesh = ax.pcolormesh(lon_grid, lat_grid, log_smoothed, cmap=colormap, shading='auto',
                             transform=ccrs.PlateCarree(), alpha=0.9)
        cb = fig.colorbar(mesh, orientation='horizontal', pad=0.05, shrink=0.8)
        cb.set_label("Smoothed Earthquake Frequency (log scale)", fontsize=12)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        st.markdown(
            f"<div style='text-align: center;'>"
            f"<img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}' width='950'/>"
            f"</div>", unsafe_allow_html=True)

    # === 3D GLOBE ===
    elif vis_option == "🌐 3D Globe":
        st.markdown("### Interactive 3D Earthquake Globe")

        with st.sidebar:
            st.header("🌐 Globe Filters")
            year_filter_type = st.radio("Filter by:", ["Year Range", "Single Year"], key="globe_filter_type")
            if year_filter_type == "Year Range":
                year_range = st.slider("Select Year Range", 1995, 2025, (1995, 2025), key="globe_year_range")
                year_mask = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
            else:
                selected_year = st.selectbox("Select Year", sorted(df['year'].unique()), key="globe_year_select")
                year_mask = (df['year'] == selected_year)

            depth_range = st.slider("Depth Range (km)", int(df['depth'].min()), int(df['depth'].max()), (0, 700), key="globe_depth")
            mag_range = st.slider("Magnitude Range", float(df['mag'].min()), float(df['mag'].max()), (5.0, 9.5), key="globe_mag")
            color_by = st.radio("Color Points By", ["Depth", "Magnitude"], key="globe_color_by")

        @st.cache_data
        def get_filtered_globe(df, mask, depth_range, mag_range):
            return df[
                mask &
                (df['depth'] >= depth_range[0]) & (df['depth'] <= depth_range[1]) &
                (df['mag'] >= mag_range[0]) & (df['mag'] <= mag_range[1])
            ]

        filtered_df = get_filtered_globe(df, year_mask, depth_range, mag_range)

        color_field = 'depth' if color_by == 'Depth' else 'mag'
        colorscale = 'Viridis' if color_by == 'Depth' else 'Turbo'
        color_title = "Depth (km)" if color_by == 'Depth' else "Magnitude"

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=filtered_df['longitude'],
            lat=filtered_df['latitude'],
            text="Mag: " + filtered_df['mag'].astype(str) + "<br>Depth: " + filtered_df['depth'].astype(str) + " km",
            mode='markers',
            marker=dict(
                size=filtered_df['mag'] ** 1.5,
                color=filtered_df[color_field],
                colorscale=colorscale,
                colorbar=dict(title=color_title, ticks="outside"),
                opacity=0.85,
                line_color='white',
                line_width=0.5,
                cmin=df[color_field].min(),
                cmax=df[color_field].max()
            )
        ))

        fig.update_geos(
            projection_type="orthographic",
            showcoastlines=True,
            coastlinecolor="black",
            showland=True,
            landcolor="rgb(204, 255, 204)",
            showocean=True,
            oceancolor="rgb(173, 216, 230)",
            showcountries=True,
            countrycolor="gray",
            showframe=False,
            resolution=50
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            geo=dict(bgcolor='rgba(0,0,0,0)')
        )

        st.plotly_chart(fig, use_container_width=True)
