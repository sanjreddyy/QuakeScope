
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
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="QuakeScope 🌍", layout="wide")

@st.cache_data
def load_main_data():
    df = pd.read_csv("QuakeScope/Earthquake_1995_2025.csv")
    df['year'] = pd.to_datetime(df['time'], errors='coerce', utc=True).dt.year
    return df.dropna(subset=['latitude', 'longitude', 'depth', 'mag', 'year'])

@st.cache_data
def load_sorted_data():
    df = pd.read_csv("QuakeScope/Earthquake_1995_2025.csv", parse_dates=['time'], infer_datetime_format=True)
    return df

df_main = load_main_data()
df_sorted = load_sorted_data()

st.markdown("<h1 style='text-align: center;'>QuakeScope 🌍</h1>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📘 About", "🗺️ 2D Heatmap", "🌐 3D Globe", "🎞️ Animated Map", "⚠️ Risk Visualization"])

with tab1:
    st.markdown("""
    ### Welcome to QuakeScope  
    This interactive tool helps visualize global earthquake data from **1995 to 2025**.

    **What You Can Explore:**
    - 🗺️ 2D Heatmap of Earthquake Density  
    - 🌐 3D Interactive Globe  
    - 🎞️ Animated Earthquake Timeline + Yearly Bar Chart  
    - ⚠️ Seismic Risk Visualization + Forecasting  
    """)

with tab2:
    st.markdown("### 2D Earthquake Density Map")
    with st.sidebar:
        st.header("🗺️ Heatmap Filters")
        year_range = st.slider("Year Range", 1995, 2025, (1995, 2025), key="heatmap_year")
        depth_range = st.slider("Depth Range (km)", int(df_main['depth'].min()), int(df_main['depth'].max()), (0, 700), key="heatmap_depth")
        mag_range = st.slider("Magnitude Range", float(df_main['mag'].min()), float(df_main['mag'].max()), (5.0, 9.5), step=0.1, key="heatmap_mag")
        colormap = st.selectbox("Color Gradient", ['YlOrRd', 'OrRd', 'YlGnBu', 'PuRd', 'Reds', 'Blues', 'YlGn'], key="heatmap_colormap")

    df_heat = df_main[
        (df_main['year'] >= year_range[0]) & (df_main['year'] <= year_range[1]) &
        (df_main['depth'] >= depth_range[0]) & (df_main['depth'] <= depth_range[1]) &
        (df_main['mag'] >= mag_range[0]) & (df_main['mag'] <= mag_range[1])
    ]

    lon_bins = np.linspace(-180, 180, 360)
    lat_bins = np.linspace(-90, 90, 180)
    heatmap, _, _ = np.histogram2d(df_heat['latitude'], df_heat['longitude'], bins=[lat_bins, lon_bins])
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
    mesh = ax.pcolormesh(lon_grid, lat_grid, log_smoothed, cmap=colormap, shading='auto', transform=ccrs.PlateCarree(), alpha=0.9)
    cb = fig.colorbar(mesh, orientation='horizontal', pad=0.05, shrink=0.8)
    cb.set_label("Smoothed Earthquake Frequency (log scale)", fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}' width='950'/></div>", unsafe_allow_html=True)

with tab3:
    st.markdown("### Interactive 3D Earthquake Globe")
    year_filter_type = st.radio("Filter by:", ["Year Range", "Single Year"], key="globe_filter_type")
    if year_filter_type == "Year Range":
        year_range = st.slider("Select Year Range", 1995, 2025, (1995, 2025), key="globe_year_range")
        year_mask = (df_main['year'] >= year_range[0]) & (df_main['year'] <= year_range[1])
    else:
        selected_year = st.selectbox("Select Year", sorted(df_main['year'].unique()), key="globe_year_select")
        year_mask = (df_main['year'] == selected_year)

    depth_range = st.slider("Depth Range (km)", int(df_main['depth'].min()), int(df_main['depth'].max()), (0, 700), key="globe_depth")
    mag_range = st.slider("Magnitude Range", float(df_main['mag'].min()), float(df_main['mag'].max()), (5.0, 9.5), key="globe_mag")
    color_by = st.radio("Color Points By", ["Depth", "Magnitude"], key="globe_color_by")
    color_field = 'depth' if color_by == 'Depth' else 'mag'
    colorscale = 'Viridis' if color_by == 'Depth' else 'Turbo'
    color_title = "Depth (km)" if color_by == 'Depth' else "Magnitude"

    filtered_df = df_main[
        year_mask &
        (df_main['depth'] >= depth_range[0]) & (df_main['depth'] <= depth_range[1]) &
        (df_main['mag'] >= mag_range[0]) & (df_main['mag'] <= mag_range[1])
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=filtered_df['longitude'],
        lat=filtered_df['latitude'],
        text="Mag: " + filtered_df['mag'].astype(str) + "<br>Depth: " + filtered_df['depth'].astype(str) + " km",
        mode='markers',
        marker=dict(size=filtered_df['mag'] ** 1.5, color=filtered_df[color_field], colorscale=colorscale,
                    colorbar=dict(title=color_title, ticks="outside"), opacity=0.85,
                    line_color='white', line_width=0.5,
                    cmin=df_main[color_field].min(), cmax=df_main[color_field].max())
    ))

    fig.update_geos(projection_type="orthographic", showcoastlines=True, coastlinecolor="black",
                    showland=True, landcolor="rgb(204, 255, 204)", showocean=True, oceancolor="rgb(173, 216, 230)",
                    showcountries=True, countrycolor="gray", showframe=False, resolution=50)

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, geo=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 🎞️ Animated Earthquake Timeline")
    df = df_main.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'latitude', 'longitude', 'mag'])
    df['year'] = df['time'].dt.year
    year_range = st.slider("Select Year Range:", int(df['year'].min()), int(df['year'].max()), (1995, 2025), step=1)
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

    fig = px.scatter_geo(df_filtered, lat='latitude', lon='longitude', color='mag', size='mag', size_max=8,
                         animation_frame='year', projection='natural earth', title='Global Earthquakes (Animated)',
                         color_continuous_scale='Turbo')
    fig.update_traces(marker=dict(sizemode='area', opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Bar Chart: Events by Year & Magnitude")
    def get_mag_category(mag):
        return "Moderate" if mag < 6 else "Strong" if mag < 7 else "Major"
    df_filtered['mag_category'] = df_filtered['mag'].apply(get_mag_category)
    bar_data = df_filtered.groupby(['year', 'mag_category']).size().reset_index(name='count')
    bar_data_pivot = bar_data.pivot(index='year', columns='mag_category', values='count').fillna(0).sort_index()

    fig_bar = go.Figure()
    for category in ["Moderate", "Strong", "Major"]:
        y_values = bar_data_pivot[category] if category in bar_data_pivot.columns else [0]*len(bar_data_pivot)
        fig_bar.add_trace(go.Bar(x=bar_data_pivot.index.astype(str), y=y_values, name=category))
    fig_bar.update_layout(barmode='group', title="Events by Magnitude per Year",
                          xaxis_title="Year", yaxis_title="Count")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab5:
    st.markdown("### ⚠️ Seismic Risk Analysis")
    exec(open("QuakeScope/risk_visualization.py").read())
