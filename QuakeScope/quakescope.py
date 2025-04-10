import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
import io
import base64

# --- Streamlit Setup ---
st.set_page_config(page_title="QuakeScope 🌍", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        h1 { text-align: center; font-size: 3rem; color: #4b4b4b; }
        h3 { font-size: 1.5rem; color: #444; }
        .sidebar .sidebar-content { background-color: #f0f2f6; }
        footer {visibility: hidden;}
        [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df1 = pd.read_csv("QuakeScope/Earthquake_1995_2025.csv")
    df1['year'] = pd.to_datetime(df1['time'], errors='coerce', utc=True).dt.year
    df1 = df1.dropna(subset=['latitude', 'longitude', 'depth', 'mag', 'year'])

    df2 = pd.read_csv("QuakeScope/Earthquake_1995_2025.csv", parse_dates=['time'], infer_datetime_format=True)
    return df1, df2

df_main, df_sorted = load_data()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 About", "🗺️ Heatmap", "🌐 Globe", "🎞️ Animated View", "⚠️ Risk Analysis"
])

# === 📘 ABOUT ===
with tab1:
    st.markdown("<h1>QuakeScope 🌍</h1>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to **QuakeScope** — a global seismic visualization tool with data from **1995 to 2025**.

    **Explore:**
    - 🗺️ Earthquake density heatmap
    - 🌐 Interactive 3D globe
    - 🎞️ Animated timeline + bar charts
    - ⚠️ Risk prediction using ML

    **Built with:** Streamlit, Cartopy, Plotly, Scikit-learn
    """)
# === 🗺️ HEATMAP ===
with tab2:
    st.markdown("### 🗺️ 2D Earthquake Density Map")

    with st.sidebar:
        st.header("🗺️ Heatmap Filters")
        y_range = st.slider("Year Range", 1995, 2025, (1995, 2025), key="hm_year")
        d_range = st.slider("Depth (km)", int(df_main['depth'].min()), int(df_main['depth'].max()), (0, 700), key="hm_depth")
        m_range = st.slider("Magnitude", float(df_main['mag'].min()), float(df_main['mag'].max()), (5.0, 9.5), step=0.1, key="hm_mag")
        cmap = st.selectbox("Color Map", ['YlOrRd', 'OrRd', 'YlGnBu', 'PuRd', 'Reds', 'Blues', 'YlGn'], key="hm_colormap")

    df_hm = df_main[
        (df_main['year'].between(y_range[0], y_range[1])) &
        (df_main['depth'].between(d_range[0], d_range[1])) &
        (df_main['mag'].between(m_range[0], m_range[1]))
    ]

    lon_bins = np.linspace(-180, 180, 360)
    lat_bins = np.linspace(-90, 90, 180)
    heatmap, _, _ = np.histogram2d(df_hm['latitude'], df_hm['longitude'], bins=[lat_bins, lon_bins])
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
    mesh = ax.pcolormesh(lon_grid, lat_grid, log_smoothed, cmap=cmap, shading='auto', transform=ccrs.PlateCarree(), alpha=0.9)
    cb = fig.colorbar(mesh, orientation='horizontal', pad=0.05, shrink=0.8)
    cb.set_label("Smoothed Earthquake Frequency (log scale)", fontsize=12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    st.image(f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}", use_column_width=True)

# === 🌐 3D GLOBE ===
with tab3:
    st.markdown("### 🌐 Interactive 3D Earthquake Globe")

    with st.sidebar:
        st.header("🌐 Globe Filters")
        filter_type = st.radio("Filter by:", ["Year Range", "Single Year"], key="globe_type")
        if filter_type == "Year Range":
            y_range = st.slider("Year Range", 1995, 2025, (1995, 2025), key="globe_year_range")
            mask = df_main['year'].between(y_range[0], y_range[1])
        else:
            y_select = st.selectbox("Year", sorted(df_main['year'].unique()), key="globe_year_select")
            mask = df_main['year'] == y_select
        d_range = st.slider("Depth (km)", int(df_main['depth'].min()), int(df_main['depth'].max()), (0, 700), key="globe_depth")
        m_range = st.slider("Magnitude", float(df_main['mag'].min()), float(df_main['mag'].max()), (5.0, 9.5), key="globe_mag")
        color_by = st.radio("Color Points By", ["Depth", "Magnitude"], key="globe_color")

    df_globe = df_main[mask & df_main['depth'].between(*d_range) & df_main['mag'].between(*m_range)]
    field = 'depth' if color_by == 'Depth' else 'mag'
    scale = 'Viridis' if color_by == 'Depth' else 'Turbo'

    fig = go.Figure(go.Scattergeo(
        lon=df_globe['longitude'],
        lat=df_globe['latitude'],
        text="Mag: " + df_globe['mag'].astype(str) + "<br>Depth: " + df_globe['depth'].astype(str) + " km",
        mode='markers',
        marker=dict(size=df_globe['mag'] ** 1.5, color=df_globe[field], colorscale=scale,
                    colorbar=dict(title=field.title()), opacity=0.85, line_color='white', line_width=0.5)
    ))

    fig.update_geos(projection_type="orthographic", showland=True, showocean=True, showcountries=True)
    fig.update_layout(margin=dict(r=0, t=0, l=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# === 🎞️ ANIMATED TIMELINE ===
with tab4:
    st.markdown("### 🎞️ Animated Earthquake Timeline")

    df_anim = df_main.copy()
    df_anim['time'] = pd.to_datetime(df_anim['time'], errors='coerce')
    df_anim = df_anim.dropna(subset=['time', 'latitude', 'longitude', 'mag'])
    df_anim['year'] = df_anim['time'].dt.year

    anim_year_range = st.slider("Select Year Range:", int(df_anim['year'].min()), int(df_anim['year'].max()), (1995, 2025), step=1)
    df_anim_filtered = df_anim[(df_anim['year'] >= anim_year_range[0]) & (df_anim['year'] <= anim_year_range[1])]

    fig = px.scatter_geo(df_anim_filtered, lat='latitude', lon='longitude', color='mag', size='mag',
                         size_max=8, animation_frame='year', projection='natural earth',
                         title='Global Earthquakes (Animated)', color_continuous_scale='Turbo')
    fig.update_traces(marker=dict(sizemode='area', opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Bar Chart: Events by Year & Magnitude")
    def get_mag_category(mag): return "Moderate" if mag < 6 else "Strong" if mag < 7 else "Major"
    df_anim_filtered['mag_category'] = df_anim_filtered['mag'].apply(get_mag_category)
    bar_data = df_anim_filtered.groupby(['year', 'mag_category']).size().reset_index(name='count')
    bar_data_pivot = bar_data.pivot(index='year', columns='mag_category', values='count').fillna(0).sort_index()

    fig_bar = go.Figure()
    for cat in ["Moderate", "Strong", "Major"]:
        y_vals = bar_data_pivot[cat] if cat in bar_data_pivot.columns else [0] * len(bar_data_pivot)
        fig_bar.add_trace(go.Bar(x=bar_data_pivot.index.astype(str), y=y_vals, name=cat))
    fig_bar.update_layout(barmode='group', title="Earthquake Events by Magnitude", xaxis_title="Year", yaxis_title="Count")
    st.plotly_chart(fig_bar, use_container_width=True)
# === ⚠️ EARTHQUAKE RISK ANALYSIS ===
with tab5:
    st.markdown("### ⚠️ Earthquake Risk Prediction & Visualization")

    df_risk = df_sorted[df_sorted['mag'] >= 5.0].copy()
    if df_risk.empty:
        st.warning("No earthquakes above magnitude 5.0 found.")
        st.stop()

    # --- Clustering with DBSCAN ---
    db = DBSCAN(eps=0.5, min_samples=5)
    df_risk['cluster'] = db.fit_predict(df_risk[['latitude', 'longitude']])
    df_risk['high_risk'] = df_risk['cluster'].apply(lambda x: 1 if x != -1 else 0)

    # --- Classification with RandomForest ---
    X = df_risk[['latitude', 'longitude', 'mag']]
    y = df_risk['high_risk']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    prob_index = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
    df_risk['risk_prob'] = clf.predict_proba(X)[:, prob_index]
    df_risk['risk_label'] = df_risk['risk_prob'].apply(lambda p: 'Low' if p < 0.33 else 'Medium' if p < 0.66 else 'High')

    # --- Generate risk grid ---
    lat_grid = np.linspace(df_sorted['latitude'].min(), df_sorted['latitude'].max(), 100)
    lon_grid = np.linspace(df_sorted['longitude'].min(), df_sorted['longitude'].max(), 100)
    g_lat, g_lon = np.meshgrid(lat_grid, lon_grid)
    grid_df = pd.DataFrame({
        'latitude': g_lat.ravel(),
        'longitude': g_lon.ravel(),
        'mag': 5.0
    })

    if 1 in clf.classes_:
        grid_df['risk_prob'] = clf.predict_proba(grid_df[['latitude','longitude','mag']])[:, prob_index]
    else:
        grid_df['risk_prob'] = 0.0

    def label_risk(p):
        if p < 0.33: return 'Low'
        elif p < 0.66: return 'Medium'
        else: return 'High'

    grid_df['risk_num'] = grid_df['risk_prob'].apply(lambda p: 1 if p < 0.33 else 2 if p < 0.66 else 3)
    risk_num_grid = grid_df['risk_num'].values.reshape(g_lat.shape)

    # --- Markers by category ---
    df_low = df_risk[df_risk['risk_label'] == 'Low']
    df_medium = df_risk[df_risk['risk_label'] == 'Medium']
    df_high = df_risk[df_risk['risk_label'] == 'High']

    fig_risk = go.Figure([
        go.Scattermapbox(lat=df_low['latitude'], lon=df_low['longitude'], mode='markers',
                         marker=dict(size=8, color='blue'), name='Low Risk'),
        go.Scattermapbox(lat=df_medium['latitude'], lon=df_medium['longitude'], mode='markers',
                         marker=dict(size=8, color='orange'), name='Medium Risk'),
        go.Scattermapbox(lat=df_high['latitude'], lon=df_high['longitude'], mode='markers',
                         marker=dict(size=8, color='red'), name='High Risk'),
        go.Contour(x=lon_grid, y=lat_grid, z=risk_num_grid,
                   colorscale=[[0.0, 'blue'], [0.5, 'orange'], [1.0, 'red']],
                   opacity=0.3, showscale=True, colorbar=dict(title="Risk Level"))
    ])

    mid_lat = (df_sorted['latitude'].min() + df_sorted['latitude'].max()) / 2
    mid_lon = (df_sorted['longitude'].min() + df_sorted['longitude'].max()) / 2

    fig_risk.update_layout(mapbox=dict(
        style="open-street-map",
        center=dict(lat=mid_lat, lon=mid_lon),
        zoom=1),
        height=600,
        title="Earthquake Risk Events (DBSCAN + RandomForest)"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    # --- Bar Chart: Region vs Risk ---
    st.markdown("### 📊 Earthquake Risk Category by Region")
    def assign_region(row):
        if row['latitude'] > mid_lat and row['longitude'] < mid_lon: return "A"
        elif row['latitude'] > mid_lat: return "B"
        elif row['longitude'] < mid_lon: return "C"
        else: return "D"
    df_risk['region'] = df_risk.apply(assign_region, axis=1)

    bar_data = df_risk.groupby(['region', 'risk_label']).size().reset_index(name='count')
    bar_pivot = bar_data.pivot(index='region', columns='risk_label', values='count').fillna(0).reindex(["A", "B", "C", "D"])

    fig_bar = go.Figure()
    for risk in ['Low', 'Medium', 'High']:
        y_vals = bar_pivot[risk] if risk in bar_pivot.columns else [0] * len(bar_pivot)
        fig_bar.add_trace(go.Bar(x=bar_pivot.index, y=y_vals, name=f"{risk} Risk"))

    fig_bar.update_layout(
        barmode='group',
        title="Number of Earthquake Events by Risk Category & Region",
        xaxis_title="Region",
        yaxis_title="Count"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Forecasted Trend ---
    st.markdown("### 🔮 Simulated Seismic Risk Trends (2025–2034)")
    years = np.arange(2025, 2035)
    low_vals = 0.7 * np.exp(-0.1 * (years - 2025))
    medium_vals = 0.3 + 0.15 * np.sin(np.linspace(0, 2 * np.pi, len(years)))
    high_vals = np.linspace(0.1, 0.6, len(years))

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=years, y=low_vals, name='Low Risk', mode='lines+markers'))
    fig_trend.add_trace(go.Scatter(x=years, y=medium_vals, name='Medium Risk', mode='lines+markers'))
    fig_trend.add_trace(go.Scatter(x=years, y=high_vals, name='High Risk', mode='lines+markers'))

    fig_trend.update_layout(
        title="10-Year Simulated Seismic Risk Forecast",
        xaxis_title="Year",
        yaxis_title="Predicted Risk Metric"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

