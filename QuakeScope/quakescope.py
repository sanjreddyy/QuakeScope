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
import folium
from streamlit_folium import st_folium
import re

# --- Page Setup ---
st.set_page_config(page_title="QuakeScope 🌍", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        h1 { text-align: center; font-size: 3rem; color: #4b4b4b; }
        h3 { font-size: 1.5rem; color: #444; }
        .sidebar .sidebar-content { background-color: #f0f2f6; }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("QuakeScope/Earthquake_1995_2025.csv")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'latitude', 'longitude', 'depth', 'mag'])
    df['year'] = df['time'].dt.year
    return df

df = load_data()
df_2015_onward = df[df['year'].between(2015, 2025)]
# --- Tabs Setup ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 About", "🗺️ Heatmap", "🌐 Globe", "🎞️ Animated View", "⚠️ Risk Analysis", "📍 Key Events Map"
])


# --- 📘 About Tab ---
with tab1:
    st.markdown("<h1>QuakeScope 🌍</h1>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to **QuakeScope** — an interactive visualization of global earthquake data from **1995 to 2025**.

    **Explore:**
    - 🗺️ 2D Heatmap of Earthquake Density  
    - 🌐 3D Interactive Earthquake Globe (2015–2025)  
    - 🎞️ Animated Time-lapse with Category-wise Trends  
    - ⚠️ Risk Analysis using DBSCAN & Random Forest

    **Tools used**: Streamlit, Cartopy, Plotly, Matplotlib, Scikit-learn
    """)
    st.markdown("---")

# --- 🗺️ Heatmap Tab ---
with tab2:
    st.markdown("### 🗺️ 2D Earthquake Density Map")

    with st.sidebar:
        st.header("🗺️ Heatmap Filters")
        year_range = st.slider("Year Range", 1995, 2025, (1995, 2025), key="hm_year")
        depth_range = st.slider("Depth Range (km)", int(df['depth'].min()), int(df['depth'].max()), (0, 700), key="hm_depth")
        mag_range = st.slider("Magnitude Range", float(df['mag'].min()), float(df['mag'].max()), (5.0, 9.5), step=0.1, key="hm_mag")
        colormap = st.selectbox("Color Gradient", ['YlOrRd', 'OrRd', 'YlGnBu', 'PuRd', 'Reds', 'Blues', 'YlGn'], key="hm_colormap")

    df_hm = df[
        (df['year'].between(*year_range)) &
        (df['depth'].between(*depth_range)) &
        (df['mag'].between(*mag_range))
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
    mesh = ax.pcolormesh(lon_grid, lat_grid, log_smoothed, cmap=colormap, shading='auto', transform=ccrs.PlateCarree(), alpha=0.9)
    cb = fig.colorbar(mesh, orientation='horizontal', pad=0.05, shrink=0.8)
    cb.set_label("Smoothed Earthquake Frequency (log scale)", fontsize=12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    st.image(f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}", use_column_width=True)
# --- 🌐 3D Globe Tab ---
with tab3:
    st.markdown("### 🌐 Interactive 3D Earthquake Globe (2015–2025)")

    with st.sidebar:
        st.header("🌐 Globe Filters")
        filter_type = st.radio("Filter by:", ["Year Range", "Single Year"], key="globe_type")

        df_globe_base = df[df['year'].between(2015, 2025)]

        if filter_type == "Year Range":
            year_range = st.slider("Year Range (Globe)", 2015, 2025, (2015, 2025), key="globe_year_range")
            mask = df_globe_base['year'].between(year_range[0], year_range[1])
        else:
            selected_year = st.selectbox("Year", sorted(df_globe_base['year'].unique()), key="globe_year_select")
            mask = df_globe_base['year'] == selected_year

        depth_range = st.slider("Depth Range (km)", int(df['depth'].min()), int(df['depth'].max()), (0, 700), key="globe_depth")
        mag_range = st.slider("Magnitude Range", float(df['mag'].min()), float(df['mag'].max()), (5.0, 9.5), key="globe_mag")
        color_by = st.radio("Color Points By", ["Depth", "Magnitude"], key="globe_color")

    df_globe = df_globe_base[
        mask &
        df_globe_base['depth'].between(*depth_range) &
        df_globe_base['mag'].between(*mag_range)
    ]

    color_field = 'depth' if color_by == 'Depth' else 'mag'
    colorscale = 'Viridis' if color_by == 'Depth' else 'Turbo'
    color_title = "Depth (km)" if color_by == 'Depth' else "Magnitude"

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=df_globe['longitude'],
        lat=df_globe['latitude'],
        text="Mag: " + df_globe['mag'].astype(str) + "<br>Depth: " + df_globe['depth'].astype(str) + " km",
        mode='markers',
        marker=dict(
            size=df_globe['mag'] ** 1.5,
            color=df_globe[color_field],
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
# --- 🎞️ Animated Earthquake Map ---
with tab4:
    st.markdown("### 🎞️ Animated Global Earthquake Map")

    with st.sidebar:
        st.header("🎞️ Animation Filters")
        df_anim = df.copy()
        df_anim['time'] = pd.to_datetime(df_anim['time'], errors='coerce')
        df_anim = df_anim.dropna(subset=['time', 'latitude', 'longitude', 'mag'])
        df_anim['year'] = df_anim['time'].dt.year

        anim_year_range = st.slider("Select Year Range:", int(df_anim['year'].min()), int(df_anim['year'].max()), (1995, 2025), step=1)

    df_anim_filtered = df_anim[(df_anim['year'] >= anim_year_range[0]) & (df_anim['year'] <= anim_year_range[1])]

    fig = px.scatter_geo(
        df_anim_filtered,
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
    fig.update_traces(marker=dict(sizemode='area', opacity=0.6))
    st.plotly_chart(fig, use_container_width=True)

    # --- 📊 Bar Chart: Events by Year & Magnitude Category ---
    st.markdown("### 📊 Earthquake Events by Year & Magnitude Category")

    def get_mag_category(mag):
        if mag < 6.0:
            return "Moderate"
        elif mag < 7.0:
            return "Strong"
        else:
            return "Major"

    df_anim_filtered['mag_category'] = df_anim_filtered['mag'].apply(get_mag_category)
    bar_data = df_anim_filtered.groupby(['year', 'mag_category']).size().reset_index(name='count')
    bar_data_pivot = bar_data.pivot(index='year', columns='mag_category', values='count').fillna(0).sort_index()

    fig_bar = go.Figure()
    for category in ["Moderate", "Strong", "Major"]:
        y_values = bar_data_pivot[category] if category in bar_data_pivot.columns else [0] * len(bar_data_pivot.index)
        fig_bar.add_trace(go.Bar(
            x=bar_data_pivot.index.astype(str),
            y=y_values,
            name=category
        ))

    fig_bar.update_layout(
        barmode='group',
        title="Number of Earthquake Events by Magnitude Category Each Year",
        xaxis_title="Year",
        yaxis_title="Event Count",
        legend=dict(x=0, xanchor='left')
    )

    st.plotly_chart(fig_bar, use_container_width=True)
# --- ⚠️ Earthquake Risk Analysis ---
with tab5:
    st.markdown("### ⚠️ Earthquake Risk Prediction")

    with st.sidebar:
        st.header("⚠️ Risk Analysis Filters")
        risk_mag_threshold = st.slider("Minimum Magnitude to Analyze Risk", 5.0, 9.0, 5.0, step=0.1)

    df_risk = df[df['mag'] >= risk_mag_threshold].copy()
    if df_risk.empty:
        st.error(f"No earthquakes with magnitude ≥ {risk_mag_threshold}.")
        st.stop()

    # --- Clustering with DBSCAN ---
    db = DBSCAN(eps=0.5, min_samples=5)
    df_risk['cluster'] = db.fit_predict(df_risk[['latitude', 'longitude']])
    df_risk['high_risk'] = df_risk['cluster'].apply(lambda x: 1 if x != -1 else 0)

    # --- Risk prediction using RandomForest ---
    X = df_risk[['latitude', 'longitude', 'mag']]
    y = df_risk['high_risk']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    prob_index = list(clf.classes_).index(1) if 1 in clf.classes_ else 0

    df_risk['risk_prob'] = clf.predict_proba(X)[:, prob_index]
    df_risk['risk_label'] = df_risk['risk_prob'].apply(lambda p: 'Low' if p < 0.33 else 'Medium' if p < 0.66 else 'High')

    # --- Create Risk Grid for Contour ---
    lat_grid = np.linspace(df['latitude'].min(), df['latitude'].max(), 100)
    lon_grid = np.linspace(df['longitude'].min(), df['longitude'].max(), 100)
    g_lat, g_lon = np.meshgrid(lat_grid, lon_grid)
    grid_df = pd.DataFrame({'latitude': g_lat.ravel(), 'longitude': g_lon.ravel(), 'mag': risk_mag_threshold})

    if 1 in clf.classes_:
        grid_df['risk_prob'] = clf.predict_proba(grid_df[['latitude','longitude','mag']])[:, prob_index]
    else:
        grid_df['risk_prob'] = 0.0

    def map_label(prob):
        return 'Low' if prob < 0.33 else 'Medium' if prob < 0.66 else 'High'
    
    grid_df['risk_label'] = grid_df['risk_prob'].apply(map_label)
    mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    grid_df['risk_num'] = grid_df['risk_label'].map(mapping)
    risk_num_grid = grid_df['risk_num'].values.reshape(g_lat.shape)

    # --- Separate data points by label ---
    df_low = df_risk[df_risk['risk_label'] == 'Low']
    df_medium = df_risk[df_risk['risk_label'] == 'Medium']
    df_high = df_risk[df_risk['risk_label'] == 'High']

    # --- Plot Risk Map ---
    fig_risk = go.Figure([
        go.Scattermapbox(lat=df_low['latitude'], lon=df_low['longitude'], mode='markers',
                         marker=dict(size=8, color='blue'), name='Low Risk'),
        go.Scattermapbox(lat=df_medium['latitude'], lon=df_medium['longitude'], mode='markers',
                         marker=dict(size=8, color='orange'), name='Medium Risk'),
        go.Scattermapbox(lat=df_high['latitude'], lon=df_high['longitude'], mode='markers',
                         marker=dict(size=8, color='red'), name='High Risk'),
        go.Contour(x=lon_grid, y=lat_grid, z=risk_num_grid,
                   colorscale=[[0.0, 'blue'], [0.5, 'orange'], [1.0, 'red']],
                   opacity=0.3, showscale=True,
                   colorbar=dict(title="Risk Level", tickvals=[1,2,3], ticktext=['Low','Medium','High']))
    ])

    fig_risk.update_layout(
        title="Earthquake Risk Zones (DBSCAN + RandomForest)",
        mapbox=dict(style="open-street-map", center=dict(lat=0, lon=0), zoom=1),
        margin=dict(r=0, t=50, l=0, b=0)
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # --- 📊 Bar Chart: Risk by Region ---
    st.markdown("### 📊 Risk Level by Region")

    # Define regions
    mid_lat = (df['latitude'].min() + df['latitude'].max()) / 2
    mid_lon = (df['longitude'].min() + df['longitude'].max()) / 2

    def region(row):
        if row['latitude'] > mid_lat and row['longitude'] < mid_lon: return "A"
        elif row['latitude'] > mid_lat: return "B"
        elif row['longitude'] < mid_lon: return "C"
        else: return "D"

    df_risk['region'] = df_risk.apply(region, axis=1)
    bar_data = df_risk.groupby(['region', 'risk_label']).size().reset_index(name='count')
    bar_data_pivot = bar_data.pivot(index='region', columns='risk_label', values='count').fillna(0).reindex(["A", "B", "C", "D"])

    fig_bar = go.Figure()
    for label in ['Low', 'Medium', 'High']:
        y = bar_data_pivot[label] if label in bar_data_pivot.columns else [0]*len(bar_data_pivot)
        fig_bar.add_trace(go.Bar(x=bar_data_pivot.index.astype(str), y=y, name=f"{label} Risk"))
    fig_bar.update_layout(barmode='group', xaxis_title="Region", yaxis_title="Event Count", title="Risk by Region")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- 🔮 Simulated Forecast ---
    st.markdown("### 🔮 Seismic Risk Forecast (2025–2035)")
    years = np.arange(2025, 2035)
    low_vals = 0.7 * np.exp(-0.1 * (years - 2025))
    med_vals = 0.3 + 0.15 * np.sin(np.linspace(0, 2 * np.pi, len(years)))
    high_vals = np.linspace(0.1, 0.6, len(years))

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=years, y=low_vals, name='Low Risk', mode='lines+markers'))
    fig_line.add_trace(go.Scatter(x=years, y=med_vals, name='Medium Risk', mode='lines+markers'))
    fig_line.add_trace(go.Scatter(x=years, y=high_vals, name='High Risk', mode='lines+markers'))

    fig_line.update_layout(
        title="Simulated Seismic Risk Trends (Next 10 Years)",
        xaxis_title="Year",
        yaxis_title="Risk Level",
        legend=dict(x=0, xanchor='left')
    )
    st.plotly_chart(fig_line, use_container_width=True)
    updatemenus=[
    dict(
        buttons=[
            dict(label="All", method="update", args=[{"visible": [True, True, True, True]}, {"title": "All Risk Events"}]),
            dict(label="Low", method="update", args=[{"visible": [True, False, False, True]}, {"title": "Low Risk"}]),
            dict(label="Medium", method="update", args=[{"visible": [False, True, False, True]}, {"title": "Medium Risk"}]),
            dict(label="High", method="update", args=[{"visible": [False, False, True, True]}, {"title": "High Risk"}]),
        ],
        direction="right",
        pad={"r": 10, "t": 10},
        type="buttons",
        showactive=True,
        x=0.5,
        xanchor="center",
        y=1.15,
        yanchor="top"
    )
]


with tab6:
    st.markdown("### 📍 Key Earthquake Events Map")

    df = pd.read_csv("QuakeScope/Earthquake_1995_2025.csv")

    key_events = {
        '1999 İzmit (Marmara), Turkey': {'latitude': 40.78, 'longitude': 29.96, 'death_toll': "estimated 17,000", 'damage': "widespread destruction", 'response': "Local and international rescue operations"},
        '2001 Gujarat, India': {'latitude': 23.30, 'longitude': 70.10, 'death_toll': "approximately 20,000", 'damage': "severe", 'response': "Massive relief and reconstruction efforts"},
        '2004 Sumatra-Andaman, Indonesia': {'latitude': 3.32, 'longitude': 95.85, 'death_toll': "estimated 227,898", 'damage': "widespread", 'response': "International aid and relief operations"},
        '2005 Kashmir, Pakistan': {'latitude': 34.25, 'longitude': 73.47, 'death_toll': "approximately 80,000", 'damage': "extensive destruction", 'response': "Emergency aid and recovery efforts"},
        '2008 Sichuan (Wenchuan), China': {'latitude': 31.02, 'longitude': 103.37, 'death_toll': "approximately 87,476", 'damage': "massive", 'response': "National rescue and international support"},
        '2010 Haiti, Haiti': {'latitude': 18.46, 'longitude': -72.53, 'death_toll': "over 316,000", 'damage': "severe", 'response': "Global humanitarian aid"},
        '2011 Tohoku, Japan': {'latitude': 38.30, 'longitude': 142.37, 'death_toll': "around 15,897", 'damage': "catastrophic", 'response': "Extensive national/international response & nuclear crisis management"},
        '2014 Ludian, China': {'latitude': 24.20, 'longitude': 100.00, 'death_toll': "approximately 617", 'damage': "moderate", 'response': "Local rescue operations and recovery efforts"},
        '2015 Nepal (Gorkha), Nepal': {'latitude': 28.15, 'longitude': 84.71, 'death_toll': "over 9,000", 'damage': "widespread", 'response': "International aid and local rescue efforts"},
        '2023 Turkey-Syria': {'latitude': 37.50, 'longitude': 36.80, 'death_toll': "over 50,000", 'damage': "catastrophic", 'response': "Massive local and international emergency response"}
    }

    def parse_death_toll(death_str):
        matches = re.findall(r'\d[\d,]*', death_str)
        return int(matches[0].replace(',', '')) if matches else 0

    sorted_events = sorted(key_events.items(), key=lambda item: parse_death_toll(item[1]['death_toll']), reverse=True)
    num_options = [3, 5, 7, 10, 15, 20]
    selected_number = st.selectbox("Select Number of Major Earthquake Events to Display:", num_options)
    top_events = dict(sorted_events[:min(selected_number, len(sorted_events))])
    selected_event = st.selectbox("Select an Event to Highlight:", list(top_events.keys()), index=list(top_events.keys()).index('2004 Sumatra-Andaman, Indonesia') if '2004 Sumatra-Andaman, Indonesia' in top_events else 0)

    def create_map(selected_event, events_dict):
        lat = events_dict[selected_event]['latitude']
        lon = events_dict[selected_event]['longitude']
        base_map = folium.Map(location=[lat, lon], zoom_start=5)
        folium.TileLayer('openstreetmap').add_to(base_map)
        folium.TileLayer(
            tiles='Stamen Terrain',
            attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors'
        ).add_to(base_map)

        folium.TileLayer('CartoDB positron').add_to(base_map)
        folium.LayerControl().add_to(base_map)
        for event, info in events_dict.items():
            popup = f"<b>{event}</b><br>Death Toll: {info['death_toll']}<br>Damage: {info['damage']}<br>Response: {info['response']}"
            color = 'red' if event == selected_event else 'blue'
            folium.Marker([info['latitude'], info['longitude']], popup=popup, tooltip=f"{event} - Death Toll: {info['death_toll']}", icon=folium.Icon(color=color)).add_to(base_map)
        return base_map

    map_fig = create_map(selected_event, top_events)
    st_folium(map_fig, width=1200, height=600)
