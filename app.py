import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="EV Database Finder, Recommender & Admin Pro",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# 2. POSTGRESQL CONNECTION & INITIALIZATION
# ---------------------------------------------------------
@st.cache_resource
def get_db_engine():
    try:
        if "postgres" in st.secrets:
            db_config = st.secrets["postgres"]
            
            # secrets.toml ထဲမှာ 'url' တိုက်ရိုက်ပါရင် url ကို အရင်သုံးမည်
            if "url" in db_config:
                database_url = db_config["url"]
            else:
                user = db_config.get('user', 'postgres')
                password = db_config.get('password', '')
                host = db_config.get('host', 'localhost')
                port = db_config.get('port', 5432)
                dbname = db_config.get('dbname', 'ev_db')
                database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        else:
            database_url = "sqlite:///ev_database.db"
            
        # Supabase Connection ကို ငြိမ်အောင် pool_pre_ping=True ထည့်သွင်းထားသည်
        engine = create_engine(
            database_url,
            pool_pre_ping=True,      # Connection ကျမသွားအောင် စစ်ဆေးပေးသည်
            pool_recycle=300         # ၅ မိနစ်တိုင်း Connection အသစ်ပြန်ဖွင့်ပေးသည်
        )
        return engine
    except Exception as e:
        st.error(f"❌ Database Connection Error: {e}")
        return None

engine = get_db_engine()

def init_db():
    if engine is None:
        return
    create_table_query = """
    CREATE TABLE IF NOT EXISTS ev_vehicles (
        id SERIAL PRIMARY KEY,
        image_url TEXT,
        brand VARCHAR(50),
        model VARCHAR(50),
        seats INT,
        price_usd NUMERIC,
        range_km INT,
        battery_kwh NUMERIC,
        weight_kg INT,
        fastcharge_kw INT,
        acceleration_0_100 NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_query))
    except Exception:
        pass

init_db()

def seed_default_data_if_empty():
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM ev_vehicles")).scalar()
            if result == 0:
                default_data = [
                    ('https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=600&h=400&fit=crop', 'Dacia', 'Spring', 4, 18500, 165, 26.8, 970, 30, 13.7),
                    ('https://images.unsplash.com/photo-1593941707882-a5bba14938c7?w=600&h=400&fit=crop', 'Nissan', 'Leaf', 5, 30500, 270, 40.0, 1580, 50, 7.9),
                    ('https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=600&h=400&fit=crop', 'MG', 'MG4 EV', 5, 34000, 350, 64.0, 1685, 88, 7.7),
                    ('https://images.unsplash.com/photo-1502877338535-766e1452684a?w=600&h=400&fit=crop', 'Fiat', '500e', 4, 31500, 260, 42.0, 1365, 85, 9.0),
                    ('https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=600&h=400&fit=crop', 'Hyundai', 'Ioniq 5 LR', 5, 52000, 481, 77.4, 2020, 233, 5.1),
                    ('https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&h=400&fit=crop', 'Tesla', 'Model 3 LR', 5, 55000, 533, 75.0, 1830, 250, 4.4),
                    ('https://images.unsplash.com/photo-1619767886558-efdc259cde1a?w=600&h=400&fit=crop', 'Tesla', 'Model Y LR', 5, 56500, 533, 75.0, 1980, 250, 5.0),
                    ('https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=600&h=400&fit=crop', 'Volvo', 'EX30', 5, 41000, 476, 69.0, 1830, 153, 5.3),
                    ('https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=600&h=400&fit=crop', 'Kia', 'EV9 AWD', 7, 78000, 505, 99.8, 2565, 210, 6.0),
                    ('https://images.unsplash.com/photo-1555215695-3004980ad54e?w=600&h=400&fit=crop', 'BMW', 'i4 eDrive40', 5, 64500, 590, 80.7, 2125, 205, 5.7),
                    ('https://images.unsplash.com/photo-1614162692292-7ac56d7f7f1e?w=600&h=400&fit=crop', 'Porsche', 'Taycan', 4, 110000, 470, 93.4, 2295, 270, 3.7),
                    ('https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&h=400&fit=crop', 'Lucid', 'Air GT', 5, 140000, 685, 118.0, 2360, 300, 3.0),
                    ('https://images.unsplash.com/photo-1606152421802-db97b9c7a11b?w=600&h=400&fit=crop', 'Audi', 'Q4 e-tron', 5, 57500, 420, 82.0, 2125, 175, 6.2),
                    ('https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=600&h=400&fit=crop', 'Mercedes', 'EQE 350', 5, 75000, 550, 89.0, 2355, 170, 6.4)
                ]
                insert_query = text("""
                    INSERT INTO ev_vehicles (image_url, brand, model, seats, price_usd, range_km, battery_kwh, weight_kg, fastcharge_kw, acceleration_0_100)
                    VALUES (:image_url, :brand, :model, :seats, :price_usd, :range_km, :battery_kwh, :weight_kg, :fastcharge_kw, :acceleration_0_100)
                """)
                for row in default_data:
                    conn.execute(insert_query, {
                        'image_url': row[0], 'brand': row[1], 'model': row[2], 'seats': row[3],
                        'price_usd': row[4], 'range_km': row[5], 'battery_kwh': row[6],
                        'weight_kg': row[7], 'fastcharge_kw': row[8], 'acceleration_0_100': row[9]
                    })
    except Exception:
        pass

seed_default_data_if_empty()

# ---------------------------------------------------------
# 3. DATA PROCESSING & ML PIPELINE
# ---------------------------------------------------------
def load_data_from_db():
    if engine is None:
        return pd.DataFrame()
    try:
        query = "SELECT * FROM ev_vehicles ORDER BY id ASC;"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return pd.DataFrame()

        df.rename(columns={
            'image_url': 'Image', 'brand': 'Brand', 'model': 'Model', 'seats': 'Seats',
            'price_usd': 'Price_USD', 'range_km': 'Range_KM', 'battery_kwh': 'Battery_kWh',
            'weight_kg': 'Weight_KG', 'fastcharge_kw': 'FastCharge_KW', 'acceleration_0_100': 'Acceleration_0_100'
        }, inplace=True)
        
        numeric_cols = ['Seats', 'Price_USD', 'Range_KM', 'Battery_kWh', 'Weight_KG', 'FastCharge_KW', 'Acceleration_0_100']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['Full_Name'] = df['Brand'].astype(str) + ' ' + df['Model'].astype(str)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def process_ml_pipeline():
    df = load_data_from_db()
    
    if df.empty or len(df) < 2:
        return df, None, None, [], None, None

    if 'Full_Name' not in df.columns:
        df['Full_Name'] = df['Brand'].astype(str) + ' ' + df['Model'].astype(str)

    df['FastCharge_Min_10_80'] = df.apply(
        lambda row: max(int((row['Battery_kWh'] * 0.7 / row['FastCharge_KW']) * 60), 15) if row['FastCharge_KW'] > 0 else 30, axis=1
    )
    
    features = ['Price_USD', 'Range_KM', 'Battery_kWh', 'FastCharge_KW', 'Acceleration_0_100']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = min(3, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=features)
    centroids_df['Cluster'] = range(n_clusters)
    
    cluster_means = df.groupby('Cluster')['Price_USD'].mean().sort_values()
    tier_labels = ['Low (Budget)', 'Medium (Mid-Range)', 'High (Luxury/Performance)']
    tier_mapping = {cluster_means.index[i]: tier_labels[i] for i in range(len(cluster_means))}
    
    df['Price_Tier'] = df['Cluster'].map(tier_mapping)
    centroids_df['Price_Tier'] = centroids_df['Cluster'].map(tier_mapping)
    centroids_df = centroids_df.sort_values(by='Price_USD').reset_index(drop=True)

    n_components = min(2, len(df))
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_coords[:, 0]
    df['PCA2'] = pca_coords[:, 1] if n_components > 1 else 0
    
    n_neighbors = min(4, len(df))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nn_model.fit(X_scaled)

    return df, centroids_df, scaler, features, X_scaled, nn_model

df, centroids_df, scaler, feature_cols, X_scaled, nn_model = process_ml_pipeline()

# Guarantee Full_Name column exists even if df is empty or minimal
if not df.empty and 'Full_Name' not in df.columns:
    df['Full_Name'] = df['Brand'].astype(str) + ' ' + df['Model'].astype(str)

if df.empty or len(df) < 2:
    st.warning("⚠️ Machine Learning Analytics များ လုပ်ဆောင်နိုင်ရန် Database ထဲတွင် EV Data အနည်းဆုံး (၂) စီး ရှိရပါမည်။ Admin Panel မှ Data ထပ်မံထည့်သွင်းပေးပါ။")

# ---------------------------------------------------------
# 4. SIDEBAR: CURRENCY & DYNAMIC FILTERS
# ---------------------------------------------------------
st.sidebar.header("💱 Currency Exchange Rates")
usd_to_mmk = st.sidebar.number_input("USD to MMK (မြန်မာကျပ်):", value=4500, step=50)
usd_to_cny = st.sidebar.number_input("USD to CNY (တရုတ်ယွမ်):", value=7.20, step=0.05, format="%.2f")

if not df.empty:
    df['Price_MMK'] = df['Price_USD'] * usd_to_mmk
    df['Price_MMK_Lakhs'] = df['Price_MMK'] / 100000
    df['Price_CNY'] = df['Price_USD'] * usd_to_cny

    if centroids_df is not None:
        centroids_df['Price_MMK'] = centroids_df['Price_USD'] * usd_to_mmk
        centroids_df['Price_MMK_Lakhs'] = centroids_df['Price_MMK'] / 100000
        centroids_df['Price_CNY'] = centroids_df['Price_USD'] * usd_to_cny

st.sidebar.markdown("---")
st.sidebar.header("🔍 Dynamic Filters")

currency_mode = st.sidebar.radio("Filter Price By:", ["USD ($)", "MMK (သိန်း)"])

if not df.empty:
    min_price_val = int(df['Price_USD'].min())
    max_price_val = int(df['Price_USD'].max())
    if min_price_val == max_price_val:
        max_price_val = min_price_val + 1000

    min_lakhs_val = int(df['Price_MMK_Lakhs'].min())
    max_lakhs_val = int(df['Price_MMK_Lakhs'].max())
    if min_lakhs_val == max_lakhs_val:
        max_lakhs_val = min_lakhs_val + 100

    if currency_mode == "USD ($)":
        min_price, max_price = st.sidebar.slider("Price Range ($ USD):", min_value=min_price_val, max_value=max_price_val, value=(min_price_val, max_price_val), step=1000)
        filter_mask = (df['Price_USD'] >= min_price) & (df['Price_USD'] <= max_price)
    else:
        min_lakhs, max_lakhs = st.sidebar.slider("Price Range (ကျပ် သိန်းပေါင်း):", min_value=min_lakhs_val, max_value=max_lakhs_val, value=(min_lakhs_val, max_lakhs_val), step=100)
        filter_mask = (df['Price_MMK_Lakhs'] >= min_lakhs) & (df['Price_MMK_Lakhs'] <= max_lakhs)

    tier_option = st.sidebar.radio("Price Tier:", options=["All Tiers", "Low (Budget)", "Medium (Mid-Range)", "High (Luxury/Performance)"])
    seat_options = ["All Seats"] + sorted(df['Seats'].unique().tolist())
    selected_seat = st.sidebar.selectbox("Seats (ထိုင်ခုံ အရေအတွက်):", options=seat_options)

    min_r_val = int(df['Range_KM'].min())
    max_r_val = int(df['Range_KM'].max())
    if min_r_val == max_r_val:
        max_r_val = min_r_val + 50
    min_range = st.sidebar.slider("Minimum Range (KM):", min_value=min_r_val, max_value=max_r_val, value=min_r_val, step=25)

    min_b_val = int(df['Battery_kWh'].min())
    max_b_val = int(df['Battery_kWh'].max())
    if min_b_val == max_b_val:
        max_b_val = min_b_val + 10
    min_battery = st.sidebar.slider("Minimum Battery (kWh):", min_value=min_b_val, max_value=max_b_val, value=min_b_val, step=5)

    filtered_df = df[filter_mask & (df['Range_KM'] >= min_range) & (df['Battery_kWh'] >= min_battery)]
    if tier_option != "All Tiers" and 'Price_Tier' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Price_Tier'] == tier_option]
    if selected_seat != "All Seats":
        filtered_df = filtered_df[filtered_df['Seats'] == selected_seat]
else:
    filtered_df = pd.DataFrame()

# ---------------------------------------------------------
# 5. CAR DETAILS DIALOG MODAL
# ---------------------------------------------------------
@st.dialog("⚡ EV Full Profile & Benchmark Analytics", width="large")
def show_car_details(car_row):
    st.markdown(f"## 🚘 {car_row['Brand']} {car_row['Model']}")
    st.markdown(f"🏷️ Class Category: **{car_row.get('Price_Tier', 'N/A')}** | 🪑 **{car_row['Seats']} Seats**")
    
    col_img, col_quick = st.columns([1.2, 1])
    with col_img:
        st.image(car_row['Image'], use_container_width=True)
    
    with col_quick:
        st.markdown("#### 🌟 Quick Specs Summary")
        if centroids_df is not None and 'Price_Tier' in car_row and car_row['Price_Tier'] in centroids_df['Price_Tier'].values:
            cluster_centroid = centroids_df[centroids_df['Price_Tier'] == car_row['Price_Tier']].iloc[0]
            st.metric("Starting Price (USD)", f"${car_row['Price_USD']:,}", delta=f"${car_row['Price_USD'] - cluster_centroid['Price_USD']:,.0f} vs {car_row['Price_Tier']} Avg", delta_color="inverse")
            st.metric("Max Driving Range", f"{car_row['Range_KM']} km", delta=f"{car_row['Range_KM'] - cluster_centroid['Range_KM']:.0f} km vs Tier Avg")
        else:
            st.metric("Starting Price (USD)", f"${car_row['Price_USD']:,}")
            st.metric("Max Driving Range", f"{car_row['Range_KM']} km")
            
        st.metric("Price (သိန်းပေါင်း)", f"{car_row['Price_MMK_Lakhs']:,.1f} သိန်း")

    st.divider()
    m_tab1, m_tab2 = st.tabs(["💳 Currency Breakdown", "⚙️ Battery & Dynamics"])
    with m_tab1:
        p1, p2, p3 = st.columns(3)
        p1.metric("USD Price ($)", f"${car_row['Price_USD']:,}")
        p2.metric("MMK Price (သိန်း)", f"{car_row['Price_MMK_Lakhs']:,.1f} သိန်း")
        p3.metric("CNY Price (တရုတ်ယွမ်)", f"¥{car_row['Price_CNY']:,.2f}")
    
    with m_tab2:
        c1, c2 = st.columns(2)
        c1.write(f"🔋 **Battery Capacity:** {car_row['Battery_kWh']} kWh")
        c1.write(f"🛣️ **Driving Range:** {car_row['Range_KM']} KM")
        c2.write(f"⚡ **DC Fast Charge:** {car_row['FastCharge_KW']} kW")
        c2.write(f"🚀 **0-100 km/h:** {car_row['Acceleration_0_100']}s")

# ---------------------------------------------------------
# 6. MAIN NAVIGATION TABS
# ---------------------------------------------------------
st.title("⚡ Advanced EV Analytics & Database Recommender Portal")

tab1, tab2, tab3, tab_trip, tab_loan, tab4, tab_admin = st.tabs([
    "🚗 EV Finder & Comparison", 
    "🧠 Cosine Recommender", 
    "💡 TCO Analysis", 
    "🛣️ Trip & Route Calculator",
    "🏦 EMI Loan Calculator",
    "📈 3D & PCA Analytics",
    "🛠️ Admin Management"
])

# ---------------------------------------------------------
# TAB 1: EV SEARCH & COMPARISON
# ---------------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vehicles Found", len(filtered_df))
    col2.metric("Avg Price ($)", f"${filtered_df['Price_USD'].mean():,.0f}" if not filtered_df.empty else "N/A")
    col3.metric("Avg Price (သိန်း)", f"{filtered_df['Price_MMK_Lakhs'].mean():,.1f} သိန်း" if not filtered_df.empty else "N/A")
    col4.metric("Fastest Fast Charge", f"{filtered_df['FastCharge_KW'].max()} kW" if not filtered_df.empty else "N/A")

    st.markdown("---")
    view_option = st.radio("View Mode:", ["📋 Data Table", "📄 Clean List View", "📊 Side-by-Side Comparison Matrix", "🕸️ Radar Comparison Mode"], horizontal=True)

    if not filtered_df.empty:
        if view_option == "📋 Data Table":
            display_cols = ['Image', 'Brand', 'Model', 'Seats', 'Price_USD', 'Price_MMK_Lakhs', 'Price_CNY', 'Range_KM', 'Battery_kWh', 'FastCharge_KW']
            if 'Price_Tier' in filtered_df.columns:
                display_cols.append('Price_Tier')

            st.dataframe(
                filtered_df[display_cols],
                column_config={
                    "Image": st.column_config.ImageColumn("Photo"),
                    "Price_USD": st.column_config.NumberColumn("Price ($)", format="$%d"),
                    "Price_MMK_Lakhs": st.column_config.NumberColumn("Price (သိန်း)", format="%.1f သိန်း"),
                    "Price_CNY": st.column_config.NumberColumn("Price (CNY)", format="¥%.2f"),
                    "Range_KM": st.column_config.NumberColumn("Range", format="%d km"),
                },
                hide_index=True,
                use_container_width=True
            )
        elif view_option == "📄 Clean List View":
            for idx, row in filtered_df.reset_index().iterrows():
                st.subheader(f"🚘 {row['Brand']} {row['Model']}")
                img_col, info_col = st.columns([1, 2])
                with img_col:
                    st.image(row['Image'], use_container_width=True)
                with info_col:
                    st.markdown(f"**Price:** `${row['Price_USD']:,}` | **{row['Price_MMK_Lakhs']:,.1f} သိန်း** | `¥{row['Price_CNY']:,.2f}`")
                    st.markdown(f"**Range:** `{row['Range_KM']} km` | **Battery:** `{row['Battery_kWh']} kWh` | **0-100:** `{row['Acceleration_0_100']}s`")
                    
                    full_name = row.get('Full_Name', f"{row['Brand']} {row['Model']}")
                    if st.button(f"🔍 View Details - {full_name}", key=f"btn_{row['Model']}_{idx}"):
                        show_car_details(row)
                st.divider()

        elif view_option == "📊 Side-by-Side Comparison Matrix":
            st.subheader("📊 Detailed Side-by-Side Feature Comparison")
            all_names = df['Full_Name'].tolist() if ('Full_Name' in df.columns and not df.empty) else []
            selected_cars = st.multiselect("ယှဉ်ပြိုင်ကြည့်ရှုလိုသည့် EV ကားများကို ရွေးချယ်ပါ (Max 4):", options=all_names, default=all_names[:3] if len(all_names)>=3 else all_names, max_selections=4)
            
            if selected_cars:
                matrix_df = df[df['Full_Name'].isin(selected_cars)].copy()
                
                matrix_df['Price (USD)'] = matrix_df['Price_USD'].apply(lambda x: f"${x:,}")
                matrix_df['Price (သိန်း)'] = matrix_df['Price_MMK_Lakhs'].apply(lambda x: f"{x:,.1f} သိန်း")
                matrix_df['Driving Range'] = matrix_df['Range_KM'].apply(lambda x: f"{x} km")
                matrix_df['Battery Capacity'] = matrix_df['Battery_kWh'].apply(lambda x: f"{x} kWh")
                matrix_df['Fast Charge Speed'] = matrix_df['FastCharge_KW'].apply(lambda x: f"{x} kW")
                matrix_df['0-100 Acceleration'] = matrix_df['Acceleration_0_100'].apply(lambda x: f"{x} s")
                matrix_df['Seating Capacity'] = matrix_df['Seats'].apply(lambda x: f"{x} Seats")
                
                display_features = ['Price (USD)', 'Price (သိန်း)', 'Driving Range', 'Battery Capacity', 'Fast Charge Speed', '0-100 Acceleration', 'Seating Capacity']
                
                transposed_matrix = matrix_df.set_index('Full_Name')[display_features].T
                st.dataframe(transposed_matrix, use_container_width=True)
            else:
                st.info("အနည်းဆုံး ကား ၁ စီး စာရင်းထဲတွင် ရွေးချယ်ပေးပါ၊")

        elif view_option == "🕸️ Radar Comparison Mode":
            all_names = df['Full_Name'].tolist() if ('Full_Name' in df.columns and not df.empty) else []
            selected_cars = st.multiselect("Select EVs to Compare (Max 3 Recommended):", options=all_names, default=all_names[:2] if len(all_names)>=2 else all_names)
            
            if len(selected_cars) >= 2:
                radar_df = df.copy()
                radar_df['Price_Affordability'] = radar_df['Price_USD'].max() - radar_df['Price_USD']
                radar_df['Acceleration_Perf'] = radar_df['Acceleration_0_100'].max() - radar_df['Acceleration_0_100']
                
                norm_scaler = MinMaxScaler(feature_range=(10, 100))
                radar_df[['Range_KM_N', 'Battery_kWh_N', 'FastCharge_KW_N', 'Price_N', 'Accel_N']] = norm_scaler.fit_transform(
                    radar_df[['Range_KM', 'Battery_kWh', 'FastCharge_KW', 'Price_Affordability', 'Acceleration_Perf']]
                )
                
                categories = ['Range (Distance)', 'Battery Size', 'Fast Charging', 'Affordability', 'Acceleration']
                fig_radar = go.Figure()
                
                for car_name in selected_cars:
                    car_data = radar_df[radar_df['Full_Name'] == car_name].iloc[0]
                    values = [
                        car_data['Range_KM_N'], car_data['Battery_kWh_N'], 
                        car_data['FastCharge_KW_N'], car_data['Price_N'], car_data['Accel_N']
                    ]
                    values.append(values[0])
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=car_name
                    ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title="🕸️ EV Specification Radar Comparison (Normalized Benchmarking)"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("No vehicles match your search criteria.")

# ---------------------------------------------------------
# TAB 2: COSINE SIMILARITY RECOMMENDER
# ---------------------------------------------------------
with tab2:
    st.subheader("🧠 Machine Learning Nearest-Neighbors Recommender")
    st.write("သင်ကြိုက်နှစ်သက်သည့် EV ကားကို ရွေးချယ်ပါ - ML Algorithm မှ Vector Space တွက်ချက်မှုအပေါ် မူတည်၍ **အလားတူဆုံး ကား (၃) စီး** ကို ရှာဖွေပြသပေးပါမည်။")

    if not df.empty and nn_model is not None and 'Full_Name' in df.columns:
        target_car = st.selectbox("🎯 Target Vehicle ရွေးချယ်ပါ:", options=df['Full_Name'].tolist())
        
        if st.button("🔍 Find Similar EVs"):
            target_idx = df[df['Full_Name'] == target_car].index[0]
            target_scaled = X_scaled[target_idx].reshape(1, -1)
            
            distances, indices = nn_model.kneighbors(target_scaled)
            similar_indices = indices[0][1:]
            similar_cars = df.iloc[similar_indices]
            
            st.success(f"🎉 **{target_car}** နှင့် Vector Spec အလားတူဆုံး ကားများမှာ:")
            
            cols = st.columns(len(similar_cars))
            for idx, (_, rec_car) in enumerate(similar_cars.iterrows()):
                with cols[idx]:
                    st.image(rec_car['Image'], use_container_width=True)
                    st.markdown(f"### {rec_car['Full_Name']}")
                    st.write(f"🏷️ Tier: **{rec_car.get('Price_Tier', 'N/A')}**")
                    st.write(f"💰 Price: **${rec_car['Price_USD']:,}** ({rec_car['Price_MMK_Lakhs']:,.1f} သိန်း)")
                    st.write(f"🔋 Range: **{rec_car['Range_KM']} km**")
                    st.write(f"🚀 Acceleration: **{rec_car['Acceleration_0_100']}s**")
    else:
        st.info("Database ထဲတွင် Data မရှိသေးပါ သို့မဟုတ် မလုံလောက်သေးပါ။")

# ---------------------------------------------------------
# TAB 3: TCO & BREAKEVEN ANALYSIS
# ---------------------------------------------------------
with tab3:
    st.subheader("💡 Total Cost of Ownership (TCO) & Breakeven Analysis")
    
    col_tco1, col_tco2 = st.columns(2)
    with col_tco1:
        st.markdown("##### ⛽ Gas vs EV Parameters")
        daily_km = st.number_input("တစ်ရက်မောင်းနှင်မည့် အကွာအဝေး (KM):", value=60)
        gas_price = st.number_input("ဓာတ်ဆီ ၁ လီတာ ဈေးနှုန်း (MMK):", value=3000)
        gas_kml = st.number_input("ဓာတ်ဆီကား ၁ လီတာ မောင်းနိုင်သည့် အကွာအဝေး (KM):", value=10.0)
        gas_maint_year = st.number_input("ဓာတ်ဆီကား ၁ နှစ်စာ ထိန်းသိမ်းစရိတ် (MMK):", value=1200000)

    with col_tco2:
        st.markdown("##### ⚡ EV Parameters & Upfront Investment")
        ev_kwh_price = st.number_input("EV လျှပ်စစ် ၁ ယူနစ် (kWh) ဈေးနှုန်း (MMK):", value=500)
        ev_kmkwh = st.number_input("EV ၁ ယူနစ် မောင်းနိုင်သည့် အကွာအဝေး (KM):", value=6.0)
        ev_maint_year = st.number_input("EV ကား ၁ နှစ်စာ ထိန်းသိမ်းစရိတ် (MMK):", value=400000)
        ev_extra_upfront = st.number_input("EV ဝယ်ယူရန် ပိုကုန်ကျမည့် စရိတ် - Initial Premium (MMK):", value=20000000)

    annual_km = daily_km * 365
    annual_gas_fuel = (annual_km / gas_kml) * gas_price
    annual_gas_total = annual_gas_fuel + gas_maint_year

    annual_ev_fuel = (annual_km / ev_kmkwh) * ev_kwh_price
    annual_ev_total = annual_ev_fuel + ev_maint_year

    annual_saving = annual_gas_total - annual_ev_total
    breakeven_years = ev_extra_upfront / annual_saving if annual_saving > 0 else 0

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("ဓာတ်ဆီကား နှစ်စဉ် စရိတ်ပေါင်း", f"{annual_gas_total:,.0f} MMK")
    m2.metric("EV ကား နှစ်စဉ် စရိတ်ပေါင်း", f"{annual_ev_total:,.0f} MMK")
    m3.metric("🎯 အရင်းကျေမည့် ကြာချိန် (Breakeven)", f"{breakeven_years:.1f} နှစ် (Years)")

    years = np.arange(0, 11)
    gas_cumulative = years * annual_gas_total
    ev_cumulative = ev_extra_upfront + (years * annual_ev_total)

    fig_break = go.Figure()
    fig_break.add_trace(go.Scatter(x=years, y=gas_cumulative, mode='lines+markers', name='Gas Vehicle Cumulative Cost'))
    fig_break.add_trace(go.Scatter(x=years, y=ev_cumulative, mode='lines+markers', name='EV Vehicle Cumulative Cost (With Premium)'))
    fig_break.update_layout(
        title="📈 Cumulative Cost Comparison over 10 Years (Breakeven Plot)",
        xaxis_title="Years of Ownership",
        yaxis_title="Total Cost (MMK)"
    )
    st.plotly_chart(fig_break, use_container_width=True)

# ---------------------------------------------------------
# TAB TRIP & ROUTE RANGE CALCULATOR
# ---------------------------------------------------------
with tab_trip:
    st.subheader("🛣️ EV Trip & Route Range Calculator")
    st.write("ခရီးစဉ်အကွာအဝေးကို ရိုက်ထည့်၍ မိမိရွေးချယ်ထားသော EV ကားဖြင့် လမ်းခရီး အားသွင်းစရိတ်နှင့် ကြာချိန်ကို တွက်ချက်ပါ။")

    if not df.empty and 'Full_Name' in df.columns:
        trip_col1, trip_col2 = st.columns(2)
        with trip_col1:
            selected_trip_car = st.selectbox("🚘 မောင်းနှင်မည့် EV ကားကို ရွေးချယ်ပါ:", options=df['Full_Name'].tolist(), key="trip_car")
            trip_distance = st.number_input("📍 သွားရောက်မည့် ခရီးစဉ်အကွာအဝေး Total Distance (KM):", value=450, step=10)
            traffic_factor = st.slider("🚦 လမ်းခရီး/ရာသီဥတု သက်ရောက်မှု (Buffer Efficiency %):", min_value=70, max_value=100, value=85, help="၈၅% ထားပါက လေအေးပေးစက်နှင့် လမ်းပိတ်မှုကြောင့် Range ၁၅% လျော့ကျမည်ဟု တွက်ချက်ပါသည်")
            
        car_data = df[df['Full_Name'] == selected_trip_car].iloc[0]
        effective_range = car_data['Range_KM'] * (traffic_factor / 100.0)
        
        needed_charges = int(np.ceil(trip_distance / effective_range)) - 1
        needed_charges = max(0, needed_charges)
        
        kwh_per_km = car_data['Battery_kWh'] / car_data['Range_KM'] if car_data['Range_KM'] > 0 else 0.15
        total_kwh_needed = trip_distance * kwh_per_km
        estimated_trip_cost = total_kwh_needed * 500
        
        with trip_col2:
            st.markdown(f"#### 📊 Trip Summary: {car_data['Full_Name']}")
            st.write(f"🔋 **Effective Single-Charge Range:** `{effective_range:.0f} KM` (Max Spec: {car_data['Range_KM']} KM)")
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("⚡ လမ်းခရီး အားသွင်းရမည့် အကြိမ်အရေအတွက်", f"{needed_charges} ကြိမ်")
            res_col2.metric("💰 ခန့်မှန်း အားသွင်းစရိတ်", f"{estimated_trip_cost:,.0f} MMK")

            if needed_charges > 0:
                st.info(f"💡 ခရီးစဉ်မပြီးဆုံးမီ လမ်းခရီးတွင် အနည်းဆုံး **{needed_charges} ကြိမ်** DC Fast Charger ၌ အားသွင်းရန် လိုအပ်ပါမည်။")
            else:
                st.success("🎉 အားတစ်ကြိမ် အပြည့်သွင်းရုံဖြင့် ခရီးစဉ်ဆုံးသည်ထိ အေးဆေးမောင်းနှင်နိုင်ပါသည်။")
    else:
        st.info("Database ထဲတွင် Data မရှိသေးပါ။")

# ---------------------------------------------------------
# TAB LOAN & FINANCING EMI CALCULATOR
# ---------------------------------------------------------
with tab_loan:
    st.subheader("🏦 EV Loan & Financing EMI Calculator")
    st.write("EV ကားများအား ဘဏ် / Finance ဖြင့် အရစ်ကျ ဝယ်ယူပါက လစဉ် ပေးသွင်းရမည့် အမြတ်+အရင်း (EMI) ကို တွက်ချက်ပါ။")

    loan_col1, loan_col2 = st.columns(2)
    with loan_col1:
        if not df.empty and 'Full_Name' in df.columns:
            loan_car = st.selectbox("🚘 ဝယ်ယူလိုသော EV ကား ရွေးပါ:", options=df['Full_Name'].tolist(), key="loan_car")
            car_price_mmk = df[df['Full_Name'] == loan_car]['Price_MMK'].values[0]
            car_price_lakhs = car_price_mmk / 100000
            st.info(f"💵 **{loan_car}** ဈေးနှုန်း: **{car_price_lakhs:,.1f} သိန်း** ({car_price_mmk:,.0f} MMK)")
        else:
            car_price_mmk = 100000000

        down_payment_pct = st.slider("💰 Down Payment (%) ကနဦး ပေးသွင်းငွေ:", min_value=10, max_value=50, value=30, step=5)
        interest_rate_ann = st.number_input("📈 နှစ်စဉ် တိုးတက်အတိုးနှုန်း Annual Interest Rate (%):", value=13.0, step=0.5)
        loan_years = st.selectbox("📅 ချေးငွေ သက်တမ်း (Years):", options=[1, 2, 3, 4, 5], index=2)

    down_payment_amount = car_price_mmk * (down_payment_pct / 100.0)
    loan_amount = car_price_mmk - down_payment_amount
    
    monthly_rate = (interest_rate_ann / 100) / 12
    months = loan_years * 12
    
    if monthly_rate > 0:
        emi = loan_amount * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    else:
        emi = loan_amount / months

    total_payment = down_payment_amount + (emi * months)
    total_interest = (emi * months) - loan_amount

    with loan_col2:
        st.markdown("#### 💳 Payment Breakdown")
        st.metric("ကနဦး ပေးသွင်းရမည့်ငွေ (Down Payment)", f"{down_payment_amount/100000:,.1f} သိန်း ({down_payment_amount:,.0f} MMK)")
        st.metric("လစဉ် ပေးသွင်းရမည့်ငွေ (Monthly EMI)", f"{emi/100000:,.2f} သိန်း ({emi:,.0f} MMK)")
        st.metric("စုစုပေါင်း အတိုးစရိတ် (Total Interest)", f"{total_interest/100000:,.1f} သိန်း")

        fig_pie = px.pie(
            names=['Down Payment', 'Principal Loan Amount', 'Total Interest Cost'],
            values=[down_payment_amount, loan_amount, total_interest],
            title="📊 Total Outflow Breakdown Structure",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: 3D & PCA VISUAL ANALYTICS
# ---------------------------------------------------------
with tab4:
    st.subheader("📈 Advanced 3D & PCA Dimensionality Visual Analytics")

    if not df.empty and 'Price_Tier' in df.columns:
        ch_a, ch_b = st.columns(2)
        with ch_a:
            st.write("#### 🧊 3D Scatter Plot (Price vs Range vs FastCharge)")
            fig_3d = px.scatter_3d(
                df, x="Price_USD", y="Range_KM", z="FastCharge_KW",
                color="Price_Tier", hover_name="Full_Name" if "Full_Name" in df.columns else "Brand", size="Battery_kWh",
                labels={"Price_USD": "Price ($)", "Range_KM": "Range (KM)", "FastCharge_KW": "FastCharge (kW)"}
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        with ch_b:
            st.write("#### 🧬 PCA 2D Clustering Projection")
            fig_pca = px.scatter(
                df, x="PCA1", y="PCA2", color="Price_Tier",
                hover_name="Full_Name" if "Full_Name" in df.columns else "Brand", text="Brand",
                labels={"PCA1": "Principal Component 1", "PCA2": "Principal Component 2"}
            )
            fig_pca.update_traces(textposition="top center")
            st.plotly_chart(fig_pca, use_container_width=True)
# ---------------------------------------------------------
# TAB 5: POSTGRESQL ADMIN MANAGEMENT PANEL
# ---------------------------------------------------------
with tab_admin:
    st.subheader("🛠️ PostgreSQL Admin Management Panel")
    
    admin_pass = st.text_input("🔐 Admin Password ထည့်သွင်းပါ:", type="password", key="admin_pass_input")
    expected_pass = st.secrets.get("admin", {}).get("password", "admin123")
    
    if admin_pass == expected_pass:
        st.success("✅ Admin Authorization Approved!")
        
        admin_sub_tab1, admin_sub_tab2, admin_sub_tab3 = st.tabs([
            "📥 Bulk CSV/Excel Upload", 
            "➕ Add Single Vehicle", 
            "🗄️ Manage PostgreSQL Records"
        ])
        
        # --- SUB TAB 1: BULK CSV UPLOAD ---
        with admin_sub_tab1:
            st.markdown("#### CSV သို့မဟုတ် Excel File မှတစ်ဆင့် Data များ Bulk Upload တင်ရန်")
            uploaded_file = st.file_uploader("Upload CSV / XLSX File", type=['csv', 'xlsx'])
            
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    upload_df = pd.read_csv(uploaded_file)
                else:
                    upload_df = pd.read_excel(uploaded_file)
                
                st.write("📋 **Preview Upload Data:**")
                st.dataframe(upload_df.head(5), use_container_width=True)
                
                if st.button("🚀 PostgreSQL သို့ Data များ တိုက်ရိုက် Upload တင်မည်"):
                    try:
                        db_col_map = {
                            'Image': 'image_url', 'Brand': 'brand', 'Model': 'model', 'Seats': 'seats',
                            'Price_USD': 'price_usd', 'Range_KM': 'range_km', 'Battery_kWh': 'battery_kwh',
                            'Weight_KG': 'weight_kg', 'FastCharge_KW': 'fastcharge_kw', 'Acceleration_0_100': 'acceleration_0_100'
                        }
                        upload_df.rename(columns=db_col_map, inplace=True)
                        upload_df.to_sql('ev_vehicles', con=engine, if_exists='append', index=False)
                        
                        # Session State ထဲသို့ Noti Message မှတ်ထားခြင်း
                        st.session_state['noti_msg'] = f"📥 Bulk Data ({len(upload_df)} စီး) ကို Database ထဲသို့ အောင်မြင်စွာ Upload တင်ပြီးပါပြီ!"
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Upload Error: {e}")

        # --- SUB TAB 2: ADD SINGLE VEHICLE WITH NOTIFICATION ---
        with admin_sub_tab2:
            st.markdown("#### EV Vehicle Data အသစ်တစ်ခုချင်းစီ ထည့်သွင်းရန်")
            with st.form("add_single_ev_form", clear_on_submit=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    brand = st.text_input("Brand (ဥပမာ - BYD)")
                    model = st.text_input("Model (ဥပမာ - Atto 3)")
                    seats = st.number_input("Seats (ထိုင်ခုံ)", min_value=1, max_value=10, value=5)
                    price_usd = st.number_input("Price ($ USD)", min_value=0.0, step=500.0, value=38000.0)
                    range_km = st.number_input("Range (KM)", min_value=0, step=10, value=420)
                
                with col_b:
                    battery_kwh = st.number_input("Battery (kWh)", min_value=0.0, step=1.0, value=60.4)
                    weight_kg = st.number_input("Weight (KG)", min_value=0, step=50, value=1750)
                    fastcharge_kw = st.number_input("Fast Charge (kW)", min_value=0, step=5, value=88)
                    acceleration = st.number_input("0-100 km/h (s)", min_value=0.0, step=0.1, value=7.3)
                    image_url = st.text_input("Image URL Link", value="https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&h=400&fit=crop")
                
                submit_ev = st.form_submit_button("💾 Save to Database")
                
                if submit_ev:
                    if not brand or not model:
                        st.warning("⚠️ Brand နှင့် Model နာမည် ဖြည့်သွင်းပေးရန် လိုအပ်ပါသည်။")
                    else:
                        insert_query = text("""
                            INSERT INTO ev_vehicles (image_url, brand, model, seats, price_usd, range_km, battery_kwh, weight_kg, fastcharge_kw, acceleration_0_100)
                            VALUES (:image_url, :brand, :model, :seats, :price_usd, :range_km, :battery_kwh, :weight_kg, :fastcharge_kw, :acceleration_0_100)
                        """)
                        try:
                            with engine.begin() as conn:
                                conn.execute(insert_query, {
                                    'image_url': image_url, 'brand': brand, 'model': model, 'seats': seats,
                                    'price_usd': price_usd, 'range_km': range_km, 'battery_kwh': battery_kwh,
                                    'weight_kg': weight_kg, 'fastcharge_kw': fastcharge_kw, 'acceleration_0_100': acceleration
                                })
                            
                            # Session State တွင် Noti စာသား သိမ်းဆည်းခြင်း
                            st.session_state['noti_msg'] = f"🎉 🚘 {brand} {model} ကို Database ထဲသို့ အောင်မြင်စွာ ထည့်သွင်းပြီးပါပြီ!"
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

        # --- SUB TAB 3: DELETE & MANAGE RECORDS WITH NOTIFICATION ---
        with admin_sub_tab3:
            st.markdown("#### 🗄️ Database Record များကို စစ်ဆေးခြင်းနှင့် ဖျက်ထုတ်ခြင်း")
            if st.button("🔄 Refresh Table"):
                st.cache_data.clear()
                st.rerun()

            current_db_df = load_data_from_db()
            st.dataframe(current_db_df, use_container_width=True)

            st.divider()
            st.markdown("##### 🗑️ Record ဖျက်ရန်")
            delete_id = st.number_input("ဖျက်လိုသော ID ရိုက်ထည့်ပါ:", min_value=1, step=1)
            if st.button("❌ Record ကို ဖျက်မည်"):
                try:
                    with engine.begin() as conn:
                        conn.execute(text("DELETE FROM ev_vehicles WHERE id = :id"), {'id': delete_id})
                    
                    st.session_state['noti_msg'] = f"🗑️ Record ID ({delete_id}) ကို Database ထဲမှ ဖျက်ပြီးပါပြီ!"
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Delete Error: {e}")

# ---------------------------------------------------------
# 7. GLOBAL NOTIFICATION TOAST DISPATCHER
# ---------------------------------------------------------
# Page Refresh ဖြစ်ပြီးတိုင်း ညာဘက်အောက်ခြေတွင် Pop-up Toast Notification ပြပေးခြင်း
if 'noti_msg' in st.session_state and st.session_state['noti_msg']:
    st.toast(st.session_state['noti_msg'], icon="⚡")
    # ပြပြီးပါက Notification ကို ပြန်လည် Clear လုပ်ခြင်း
    st.session_state['noti_msg'] = None