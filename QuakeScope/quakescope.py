# Imports
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
import plotly.express as px

# Page setup
st.set_page_config(page_title="QuakeScope 🌍", layout="wide")

# Custom styles
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        h1 { text-align: center; font-size: 3rem; color: #4b4b4b; }
        h3 { font-size: 1.5rem; color: #444; }
        .sidebar .sidebar-content { background-color: #f0f2f6; }
        footer, #MainMenu, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("QuakeScope/Earthquake_1995_2025.csv")
    df['year'] = pd.to_datetime(df['time'], errors='coerce', utc=True).dt.year
    return df.dropna(subset=['latitude', 'longitude', 'depth', 'mag', 'year'])

df = load_data()

@st.cache_data
def load_sorted_data():
    df_sorted = pd.read_csv("QuakeScope/Sorted_Earthquake_1995_2025.csv")
    df_sorted['time'] = pd.to_datetime(df_sorted['time'], errors='coerce')
    df_sorted = df_sorted.dropna(subset=['time', 'latitude', 'longitude', 'mag'])
    df_sorted['year'] = df_sorted['time'].dt.year
    return df_sorted

df_sorted = load_sorted_data()

# App header
st.markdown("<h1>QuakeScope 🌍</h1>", unsafe_allow_html=True)

# Tabs
about_tab, vis_tab = st.tabs(["\ud83d\udcd8 About", "\ud83d\udcca Visualizations"])

# === ABOUT TAB ===
with about_tab:
    st.markdown("""
    ### Welcome to QuakeScope
    This interactive tool helps visualize global earthquake data from **1995 to 2025**.

    **What You Can Explore:**
    - \ud83d\udcfd\ufe0f 2D Heatmap of Earthquake Density
    - \ud83c\udf10 3D Interactive Globe Visualization
    - \ud83c\udf07 Earthquake Time-lapse Animation

    **Fields Used**: Latitude, Longitude, Magnitude, Depth, Year  
    **Powered by**: Streamlit, Cartopy, Plotly, Pandas, Matplotlib
    """)

# === VISUALIZATIONS TAB ===
with vis_tab:
    vis_option = st.radio("Choose Visualization", ["\ud83d\udcfd\ufe0f 2D Heatmap", "\ud83c\udf10 3D Globe", "\ud83c\udf07 Time-lapse Animation"], horizontal=True)

    if vis_option == "\ud83d\udcfd\ufe0f 2D Heatmap":
        with st.sidebar:
            st.header("\ud83d\udcfd\ufe0f Heatmap Filters")
            year_range = st.slider("Year Range", 1995, 2025, (1995, 2025), key="heatmap_year")
            depth_range = st.slider("Depth Range (km)", int(df['depth'].min()), int(df['depth'].max()), (0, 700), key="heatmap_depth")
            mag_range = st.slider("Magnitude Range", float(df['mag'].min()), float(df['mag'].max()), (5.0, 9.5), step=0.1, key="heatmap_mag")
            colormap = st.selectbox("Color Gradient", ['YlOrRd', 'OrRd', 'YlGnBu', 'PuRd', 'Reds', 'Blues', 'YlGn'], key="heatmap_colormap")

        filtered = df[
            (df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
            (df['depth'] >= depth_range[0]) & (df['depth'] <= depth_range[1]) &
            (df['mag'] >= mag_range[0]) & (df['mag'] <= mag_range[1])
        ]

        lon_bins = np.linspace(-180, 180, 360)
        lat_bins = np.linspace(-90, 90, 180)
        heatmap, _, _ = np.histogram2d(filtered['latitude'], filtered['longitude'], bins=[lat_bins, lon_bins])
        smoothed = gaussian_filter(heatmap, sigma=2.5)
        log_smoothed = np.log1p(smoothed)
        lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)

        fig = plt.figure(figsize=(16, 8))
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

        st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}' width='950'/></div>", unsafe_allow_html=True)

    elif vis_option == "\ud83c\udf10 3D Globe":
        with st.sidebar:
            st.header("\ud83c\udf10 Globe Filters")
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

        filtered_df = df[
            year_mask &
            (df['depth'] >= depth_range[0]) & (df['depth'] <= depth_range[1]) &
            (df['mag'] >= mag_range[0]) & (df['mag'] <= mag_range[1])
        ]

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

    elif vis_option == "\ud83c\udf07 Time-lapse Animation":
        with st.sidebar:
            st.header("\ud83c\udf07 Animation Filters")
            min_year = int(df_sorted['year'].min())
            max_year = int(df_sorted['year'].max())
            year_range = st.slider("Select Year Range:", min_year, max_year, (min_year, max_year), step=1)

        df_anim = df_sorted[(df_sorted['year'] >= year_range[0]) & (df_sorted['year'] <= year_range[1])]
        st.write(f"Showing earthquakes from {year_range[0]} to {year_range[1]}")

        fig = px.scatter_geo(
            df_anim,
            lat="latitude",
            lon="longitude",
            color="mag",
            size="mag",
            size_max=8,
            animation_frame="year",
            projection="natural earth",
            title="Time-lapse of Global Earthquakes (1995–2025)",
            color_continuous_scale="Turbo"
        )

        fig.update_traces(marker=dict(sizemode='area', opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)
