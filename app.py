import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sqlalchemy import create_engine, text

import streamlit as st

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & SESSION STATE INITIALIZATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="EV Database Finder, Recommender & Admin Pro",
    page_icon="⚡",
    layout="wide",
)

# Compare လုပ်ရန် ကားများ မှတ်ထားမည့် Session State
if "selected_compare" not in st.session_state:
    st.session_state["selected_compare"] = []

# Custom CSS for Responsive Cards & Styling
st.markdown(
    """
<style>
    .ev-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .ev-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        border-color: #007bff;
    }
    .ev-card img {
        border-radius: 8px;
        object-fit: cover;
        width: 100%;
        height: 180px;
    }
    .ev-card-title {
        font-size: 20px;
        font-weight: 700;
        color: #1f2937;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .ev-badge {
        background-color: #eef2ff;
        color: #4f46e5;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 10px;
    }
    .ev-price {
        font-size: 18px;
        font-weight: 700;
        color: #059669;
        margin-bottom: 8px;
    }
    .ev-spec-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        font-size: 13px;
        color: #4b5563;
    }
    .rating-badge {
        color: #f59e0b;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. POSTGRESQL CONNECTION & INITIALIZATION
# ---------------------------------------------------------
@st.cache_resource
def get_db_engine():
  try:
    if "postgres" in st.secrets:
      db_config = st.secrets["postgres"]
      if "url" in db_config:
        database_url = db_config["url"]
      else:
        user = db_config.get("user", "postgres")
        password = db_config.get("password", "")
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)
        dbname = db_config.get("dbname", "ev_db")
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    else:
      database_url = "sqlite:///ev_database.db"

    engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300)
    return engine
  except Exception as e:
    st.error(f"❌ Database Connection Error: {e}")
    return None


engine = get_db_engine()


def init_db():
  if engine is None:
    return
  create_vehicles_table = """
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
        fastcharge_min_10_80 INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
  create_reviews_table = """
    CREATE TABLE IF NOT EXISTS ev_reviews (
        id SERIAL PRIMARY KEY,
        vehicle_id INT NOT NULL,
        user_name VARCHAR(100) NOT NULL,
        rating INT CHECK (rating >= 1 AND rating <= 5),
        review_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
  try:
    with engine.begin() as conn:
      conn.execute(text(create_vehicles_table))
      conn.execute(text(create_reviews_table))
      conn.execute(
          text(
              "ALTER TABLE ev_vehicles ADD COLUMN IF NOT EXISTS"
              " fastcharge_min_10_80 INT DEFAULT 0;"
          )
      )
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
            (
                "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=600&h=400&fit=crop",
                "Dacia",
                "Spring",
                4,
                18500,
                165,
                26.8,
                970,
                30,
                13.7,
                37,
            ),
            (
                "https://images.unsplash.com/photo-1593941707882-a5bba14938c7?w=600&h=400&fit=crop",
                "Nissan",
                "Leaf",
                5,
                30500,
                270,
                40.0,
                1580,
                50,
                7.9,
                33,
            ),
            (
                "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=600&h=400&fit=crop",
                "MG",
                "MG4 EV",
                5,
                34000,
                350,
                64.0,
                1685,
                88,
                7.7,
                30,
            ),
            (
                "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=600&h=400&fit=crop",
                "Fiat",
                "500e",
                4,
                31500,
                260,
                42.0,
                1365,
                85,
                9.0,
                24,
            ),
            (
                "https://images.unsplash.com/photo-1617814076367-b759c7d7e738?w=600&h=400&fit=crop",
                "Hyundai",
                "Ioniq 5 LR",
                5,
                52000,
                481,
                77.4,
                2020,
                233,
                5.1,
                18,
            ),
            (
                "https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&h=400&fit=crop",
                "Tesla",
                "Model 3 LR",
                5,
                55000,
                533,
                75.0,
                1830,
                250,
                4.4,
                27,
            ),
            (
                "https://images.unsplash.com/photo-1619767886558-efdc259cde1a?w=600&h=400&fit=crop",
                "Tesla",
                "Model Y LR",
                5,
                56500,
                533,
                75.0,
                1980,
                250,
                5.0,
                27,
            ),
            (
                "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=600&h=400&fit=crop",
                "Volvo",
                "EX30",
                5,
                41000,
                476,
                69.0,
                1830,
                153,
                5.3,
                26,
            ),
            (
                "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=600&h=400&fit=crop",
                "Kia",
                "EV9 AWD",
                7,
                78000,
                505,
                99.8,
                2565,
                210,
                6.0,
                24,
            ),
            (
                "https://images.unsplash.com/photo-1555215695-3004980ad54e?w=600&h=400&fit=crop",
                "BMW",
                "i4 eDrive40",
                5,
                64500,
                590,
                80.7,
                2125,
                205,
                5.7,
                31,
            ),
            (
                "https://images.unsplash.com/photo-1614162692292-7ac56d7f7f1e?w=600&h=400&fit=crop",
                "Porsche",
                "Taycan",
                4,
                110000,
                470,
                93.4,
                2295,
                270,
                3.7,
                22,
            ),
            (
                "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&h=400&fit=crop",
                "Lucid",
                "Air GT",
                5,
                140000,
                685,
                118.0,
                2360,
                300,
                3.0,
                20,
            ),
        ]
        insert_query = text("""
                    INSERT INTO ev_vehicles (image_url, brand, model, seats, price_usd, range_km, battery_kwh, weight_kg, fastcharge_kw, acceleration_0_100, fastcharge_min_10_80)
                    VALUES (:image_url, :brand, :model, :seats, :price_usd, :range_km, :battery_kwh, :weight_kg, :fastcharge_kw, :acceleration_0_100, :fastcharge_min_10_80)
                """)
        for row in default_data:
          conn.execute(
              insert_query,
              {
                  "image_url": row[0],
                  "brand": row[1],
                  "model": row[2],
                  "seats": row[3],
                  "price_usd": row[4],
                  "range_km": row[5],
                  "battery_kwh": row[6],
                  "weight_kg": row[7],
                  "fastcharge_kw": row[8],
                  "acceleration_0_100": row[9],
                  "fastcharge_min_10_80": row[10],
              },
          )

        sample_reviews = [
            (
                6,
                "Aung Aung",
                5,
                "Tesla Model 3 က Range လည်းတော်တော်ရတယ်၊ အဆွဲအရုန်းလည်းရှယ်ပဲ!",
            ),
            (
                6,
                "Kyaw Kyaw",
                4,
                "မောင်းရတာ အဆင်ပြေပါတယ်။ Charging Speed လည်း မြန်တယ်။",
            ),
            (
                5,
                "Min Thant",
                5,
                "Ioniq 5 ရဲ့ 800V Fast charging က တကယ်မြန်လွန်းတယ်။ Design"
                " လည်း မိုက်တယ်။",
            ),
        ]
        rev_query = text(
            "INSERT INTO ev_reviews (vehicle_id, user_name, rating, review_text)"
            " VALUES (:v_id, :u_name, :rating, :r_text)"
        )
        for rev in sample_reviews:
          conn.execute(
              rev_query,
              {
                  "v_id": rev[0],
                  "u_name": rev[1],
                  "rating": rev[2],
                  "r_text": rev[3],
              },
          )
  except Exception:
    pass


seed_default_data_if_empty()


def fetch_vehicle_reviews(vehicle_id):
  if engine is None:
    return pd.DataFrame()
  try:
    query = text(
        "SELECT user_name, rating, review_text, created_at FROM ev_reviews"
        " WHERE vehicle_id = :v_id ORDER BY id DESC"
    )
    return pd.read_sql(query, engine, params={"v_id": vehicle_id})
  except Exception:
    return pd.DataFrame()


def insert_vehicle_review(vehicle_id, user_name, rating, review_text):
  if engine is None:
    return False
  try:
    query = text(
        "INSERT INTO ev_reviews (vehicle_id, user_name, rating, review_text)"
        " VALUES (:v_id, :u_name, :rating, :r_text)"
    )
    with engine.begin() as conn:
      conn.execute(
          query,
          {
              "v_id": vehicle_id,
              "u_name": user_name,
              "rating": rating,
              "r_text": review_text,
          },
      )
    return True
  except Exception:
    return False


# ---------------------------------------------------------
# 3. DATA PROCESSING & ML PIPELINE
# ---------------------------------------------------------
def load_data_from_db():
  if engine is None:
    return pd.DataFrame()
  try:
    query = """
        SELECT v.*, 
               COALESCE(AVG(r.rating), 0) as avg_rating,
               COUNT(r.id) as review_count
        FROM ev_vehicles v
        LEFT JOIN ev_reviews r ON v.id = r.vehicle_id
        GROUP BY v.id
        ORDER BY v.id ASC;
        """
    df = pd.read_sql(query, engine)

    if df.empty:
      return pd.DataFrame()

    df.rename(
        columns={
            "image_url": "Image",
            "brand": "Brand",
            "model": "Model",
            "seats": "Seats",
            "price_usd": "Price_USD",
            "range_km": "Range_KM",
            "battery_kwh": "Battery_kWh",
            "weight_kg": "Weight_KG",
            "fastcharge_kw": "FastCharge_KW",
            "acceleration_0_100": "Acceleration_0_100",
            "fastcharge_min_10_80": "FastCharge_Min_10_80",
            "avg_rating": "Avg_Rating",
            "review_count": "Review_Count",
        },
        inplace=True,
    )

    numeric_cols = [
        "Seats",
        "Price_USD",
        "Range_KM",
        "Battery_kWh",
        "Weight_KG",
        "FastCharge_KW",
        "Acceleration_0_100",
        "FastCharge_Min_10_80",
        "Avg_Rating",
        "Review_Count",
    ]
    for col in numeric_cols:
      if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Full_Name"] = df["Brand"].astype(str) + " " + df["Model"].astype(str)
    return df
  except Exception:
    return pd.DataFrame()


@st.cache_data(ttl=30)
def process_ml_pipeline():
  df = load_data_from_db()

  if df.empty or len(df) < 2:
    return df, None, None, [], None, None

  if "Full_Name" not in df.columns:
    df["Full_Name"] = df["Brand"].astype(str) + " " + df["Model"].astype(str)

  if "FastCharge_Min_10_80" not in df.columns:
    df["FastCharge_Min_10_80"] = 0

  df["FastCharge_Min_10_80"] = df.apply(
      lambda row: row["FastCharge_Min_10_80"]
      if row["FastCharge_Min_10_80"] > 0
      else (
          max(int((row["Battery_kWh"] * 0.7 / row["FastCharge_KW"]) * 60), 15)
          if row["FastCharge_KW"] > 0
          else 30
      ),
      axis=1,
  )

  features = [
      "Price_USD",
      "Range_KM",
      "Battery_kWh",
      "FastCharge_KW",
      "Acceleration_0_100",
  ]
  X = df[features]

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  n_clusters = min(3, len(df))
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  df["Cluster"] = kmeans.fit_predict(X_scaled)

  centroids_scaled = kmeans.cluster_centers_
  centroids_original = scaler.inverse_transform(centroids_scaled)
  centroids_df = pd.DataFrame(centroids_original, columns=features)
  centroids_df["Cluster"] = range(n_clusters)

  cluster_means = df.groupby("Cluster")["Price_USD"].mean().sort_values()
  tier_labels = [
      "Low (Budget)",
      "Medium (Mid-Range)",
      "High (Luxury/Performance)",
  ]
  tier_mapping = {
      cluster_means.index[i]: tier_labels[i] for i in range(len(cluster_means))
  }

  df["Price_Tier"] = df["Cluster"].map(tier_mapping)
  centroids_df["Price_Tier"] = centroids_df["Cluster"].map(tier_mapping)
  centroids_df = centroids_df.sort_values(by="Price_USD").reset_index(drop=True)

  n_neighbors = min(4, len(df))
  nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
  nn_model.fit(X_scaled)

  return df, centroids_df, scaler, features, X_scaled, nn_model


df, centroids_df, scaler, feature_cols, X_scaled, nn_model = (
    process_ml_pipeline()
)

# ---------------------------------------------------------
# 4. SIDEBAR: SEARCH, SORT & DYNAMIC FILTERS
# ---------------------------------------------------------
st.sidebar.header("💱 Currency Exchange Rates")
usd_to_mmk = st.sidebar.number_input(
    "USD to MMK (မြန်မာကျပ်):", value=4500, step=50
)

if not df.empty:
  df["Price_MMK"] = df["Price_USD"] * usd_to_mmk
  df["Price_MMK_Lakhs"] = df["Price_MMK"] / 100000

  if centroids_df is not None:
    centroids_df["Price_MMK"] = centroids_df["Price_USD"] * usd_to_mmk
    centroids_df["Price_MMK_Lakhs"] = centroids_df["Price_MMK"] / 100000

st.sidebar.markdown("---")
st.sidebar.header("🔍 Search & Dynamic Filters")

search_query = st.sidebar.text_input(
    "🔎 Search Model/Brand (အမည်ဖြင့်ရှာရန်):",
    placeholder="e.g. Tesla, BYD, Leaf",
    key="search_query",
)

sort_option = st.sidebar.selectbox(
    "🔃 Sort By (စီစဉ်ရန်):",
    options=[
        "Default (ID)",
        "Price: Low to High",
        "Price: High to Low",
        "Range: High to Low",
        "User Rating: Highest First",
        "Fast Charge Speed: Fastest",
        "Acceleration: Fastest (0-100)",
    ],
    key="sort_option",
)

if not df.empty:
  brand_list = sorted(df["Brand"].unique().tolist())
  selected_brands = st.sidebar.multiselect(
      "🏷️ Select Brand(s):",
      options=brand_list,
      default=[],
      key="filter_brands",
  )

currency_mode = st.sidebar.radio(
    "Filter Price By:", ["USD ($)", "MMK (သိန်း)"], key="filter_curr_mode"
)

if not df.empty:
  min_price_val = int(df["Price_USD"].min())
  max_price_val = int(df["Price_USD"].max())
  if min_price_val == max_price_val:
    max_price_val = min_price_val + 1000

  min_lakhs_val = int(df["Price_MMK_Lakhs"].min())
  max_lakhs_val = int(df["Price_MMK_Lakhs"].max())
  if min_lakhs_val == max_lakhs_val:
    max_lakhs_val = min_lakhs_val + 100

  if currency_mode == "USD ($)":
    min_price, max_price = st.sidebar.slider(
        "Price Range ($ USD):",
        min_value=min_price_val,
        max_value=max_price_val,
        value=(min_price_val, max_price_val),
        step=1000,
        key="filter_price_usd",
    )
    filter_mask = (df["Price_USD"] >= min_price) & (df["Price_USD"] <= max_price)
  else:
    min_lakhs, max_lakhs = st.sidebar.slider(
        "Price Range (ကျပ် သိန်းပေါင်း):",
        min_value=min_lakhs_val,
        max_value=max_lakhs_val,
        value=(min_lakhs_val, max_lakhs_val),
        step=100,
        key="filter_price_mmk",
    )
    filter_mask = (df["Price_MMK_Lakhs"] >= min_lakhs) & (
        df["Price_MMK_Lakhs"] <= max_lakhs
    )

  tier_option = st.sidebar.radio(
      "Price Tier:",
      options=[
          "All Tiers",
          "Low (Budget)",
          "Medium (Mid-Range)",
          "High (Luxury/Performance)",
      ],
      key="filter_tier",
  )
  seat_options = ["All Seats"] + sorted(df["Seats"].unique().tolist())
  selected_seat = st.sidebar.selectbox(
      "Seats (ထိုင်ခုံ အရေအတွက်):", options=seat_options, key="filter_seats"
  )

  min_r_val = int(df["Range_KM"].min())
  max_r_val = int(df["Range_KM"].max())
  if min_r_val == max_r_val:
    max_r_val = min_r_val + 50
  min_range = st.sidebar.slider(
      "Minimum Range (KM):",
      min_value=min_r_val,
      max_value=max_r_val,
      value=min_r_val,
      step=25,
      key="filter_range",
  )

  min_b_val = int(df["Battery_kWh"].min())
  max_b_val = int(df["Battery_kWh"].max())
  if min_b_val == max_b_val:
    max_b_val = min_b_val + 10
  min_battery = st.sidebar.slider(
      "Minimum Battery (kWh):",
      min_value=min_b_val,
      max_value=max_b_val,
      value=min_b_val,
      step=5,
      key="filter_battery",
  )

  st.sidebar.markdown("---")
  st.sidebar.subheader("⚡ Charging & Performance")

  min_fc_val = int(df["FastCharge_KW"].min())
  max_fc_val = int(df["FastCharge_KW"].max())
  if min_fc_val == max_fc_val:
    max_fc_val = min_fc_val + 10
  min_fastcharge = st.sidebar.slider(
      "Minimum Fast Charge Speed (kW):",
      min_value=min_fc_val,
      max_value=max_fc_val,
      value=min_fc_val,
      step=10,
      key="filter_fc_kw",
  )

  if "FastCharge_Min_10_80" in df.columns:
    min_time_val = int(df["FastCharge_Min_10_80"].min())
    max_time_val = int(df["FastCharge_Min_10_80"].max())
    if min_time_val == max_time_val:
      max_time_val = min_time_val + 5
    max_charge_time = st.sidebar.slider(
        "Max Charging Time 10-80% (Mins):",
        min_value=min_time_val,
        max_value=max_time_val,
        value=max_time_val,
        step=5,
        key="filter_fc_time",
    )
  else:
    max_charge_time = None

  min_acc_val = float(df["Acceleration_0_100"].min())
  max_acc_val = float(df["Acceleration_0_100"].max())
  if min_acc_val == max_acc_val:
    max_acc_val = min_acc_val + 1.0
  max_acceleration = st.sidebar.slider(
      "Max Accel 0-100 km/h (Secs):",
      min_value=min_acc_val,
      max_value=max_acc_val,
      value=max_acc_val,
      step=0.5,
      key="filter_accel",
  )

  filtered_df = df[
      filter_mask
      & (df["Range_KM"] >= min_range)
      & (df["Battery_kWh"] >= min_battery)
      & (df["FastCharge_KW"] >= min_fastcharge)
      & (df["Acceleration_0_100"] <= max_acceleration)
  ]

  if search_query.strip():
    q = search_query.strip().lower()
    filtered_df = filtered_df[
        filtered_df["Brand"].astype(str).str.lower().str.contains(q)
        | filtered_df["Model"].astype(str).str.lower().str.contains(q)
        | filtered_df["Full_Name"].astype(str).str.lower().str.contains(q)
    ]

  if selected_brands:
    filtered_df = filtered_df[filtered_df["Brand"].isin(selected_brands)]

  if (
      max_charge_time is not None
      and "FastCharge_Min_10_80" in filtered_df.columns
  ):
    filtered_df = filtered_df[
        filtered_df["FastCharge_Min_10_80"] <= max_charge_time
    ]

  if tier_option != "All Tiers" and "Price_Tier" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Price_Tier"] == tier_option]
  if selected_seat != "All Seats":
    filtered_df = filtered_df[filtered_df["Seats"] == selected_seat]

  # SORTING
  if sort_option == "Price: Low to High":
    filtered_df = filtered_df.sort_values(by="Price_USD", ascending=True)
  elif sort_option == "Price: High to Low":
    filtered_df = filtered_df.sort_values(by="Price_USD", ascending=False)
  elif sort_option == "Range: High to Low":
    filtered_df = filtered_df.sort_values(by="Range_KM", ascending=False)
  elif sort_option == "User Rating: Highest First":
    filtered_df = filtered_df.sort_values(by="Avg_Rating", ascending=False)
  elif sort_option == "Fast Charge Speed: Fastest":
    filtered_df = filtered_df.sort_values(by="FastCharge_KW", ascending=False)
  elif sort_option == "Acceleration: Fastest (0-100)":
    filtered_df = filtered_df.sort_values(
        by="Acceleration_0_100", ascending=True
    )

else:
  filtered_df = pd.DataFrame()


# ---------------------------------------------------------
# 5. DIALOG MODALS (CAR DETAILS & COMPARE POPUP)
# ---------------------------------------------------------
@st.dialog("⚡ EV Full Profile & User Reviews", width="large")
def show_car_details(car_row):
  vehicle_id = car_row["id"]
  st.markdown(f"## 🚘 {car_row['Brand']} {car_row['Model']}")
  st.markdown(
      f"🏷️ Class Category: **{car_row.get('Price_Tier', 'N/A')}** |"
      f" **{car_row['Seats']} Seats**"
  )

  col_img, col_quick = st.columns([1.2, 1])
  with col_img:
    st.image(car_row["Image"], use_container_width=True)

  with col_quick:
    st.markdown("#### 🌟 Quick Specs Summary")
    if (
        centroids_df is not None
        and "Price_Tier" in car_row
        and car_row["Price_Tier"] in centroids_df["Price_Tier"].values
    ):
      cluster_centroid = centroids_df[
          centroids_df["Price_Tier"] == car_row["Price_Tier"]
      ].iloc[0]
      st.metric(
          "Starting Price (USD)",
          f"${car_row['Price_USD']:,}",
          delta=(
              f"${car_row['Price_USD'] - cluster_centroid['Price_USD']:,.0f} vs"
              f" {car_row['Price_Tier']} Avg"
          ),
          delta_color="inverse",
      )
      st.metric(
          "Max Driving Range",
          f"{car_row['Range_KM']} km",
          delta=(
              f"{car_row['Range_KM'] - cluster_centroid['Range_KM']:.0f} km vs"
              " Tier Avg"
          ),
      )
    else:
      st.metric("Starting Price (USD)", f"${car_row['Price_USD']:,}")
      st.metric("Max Driving Range", f"{car_row['Range_KM']} km")

    st.metric("Price (သိန်းပေါင်း)", f"{car_row['Price_MMK_Lakhs']:,.1f} သိန်း")

  st.divider()

  m_tab1, m_tab2, m_tab3 = st.tabs([
      "💳 Currency Breakdown",
      "⚙️ Battery & Dynamics",
      "💬 User Reviews & Rating",
  ])

  with m_tab1:
    p1, p2 = st.columns(2)
    p1.metric("USD Price ($)", f"${car_row['Price_USD']:,}")
    p2.metric("MMK Price (သိန်း)", f"{car_row['Price_MMK_Lakhs']:,.1f} သိန်း")

  with m_tab2:
    c1, c2 = st.columns(2)
    c1.write(f"🔋 **Battery Capacity:** {car_row['Battery_kWh']} kWh")
    c1.write(f"🛣️ **Driving Range:** {car_row['Range_KM']} KM")
    c2.write(f"⚡ **DC Fast Charge:** {car_row['FastCharge_KW']} kW")
    c2.write(
        "⏱️ **10-80% Charging Time:**"
        f" {car_row.get('FastCharge_Min_10_80', 'N/A')} mins"
    )
    c2.write(f"🚀 **0-100 km/h:** {car_row['Acceleration_0_100']}s")

  with m_tab3:
    reviews_df = fetch_vehicle_reviews(vehicle_id)
    r_col1, r_col2 = st.columns([1, 1.2])

    with r_col1:
      st.markdown("#### 📊 Customer Reviews Summary")
      if not reviews_df.empty:
        avg_r = reviews_df["rating"].mean()
        total_r = len(reviews_df)
        stars_str = "⭐" * int(round(avg_r))
        st.markdown(f"### {avg_r:.1f} / 5.0  {stars_str}")
        st.caption(f"Based on **{total_r}** verified review(s)")
      else:
        st.info(
            "ဤကားအတွက် Review များ မရှိသေးပါ။ ပထမဆုံး Review ရေးသားသူ အဖြစ်"
            " ပါဝင်လိုက်ပါ!"
        )

      st.markdown("---")
      st.markdown("✍️ **Write a Review (Review အသစ် ရေးရန်)**")
      with st.form(f"review_form_{vehicle_id}"):
        u_name = st.text_input(
            "သင့်အမည် (Your Name):", placeholder="e.g. Mg Mg"
        )
        u_rating = st.select_slider(
            "Rating အမှတ်ပေးရန် (1 to 5 Stars):",
            options=[1, 2, 3, 4, 5],
            value=5,
        )
        u_text = st.text_area(
            "Review သုံးသပ်ချက် မှတ်ချက်:",
            placeholder="ကား၏ အားသာချက်/အားနည်းချက်များကို ရေးသားပေးပါ...",
        )
        submit_review = st.form_submit_button("📤 Submit Review")

        if submit_review:
          if not u_name.strip():
            st.error("⚠️ အမည် ဖြည့်သွင်းရန် လိုအပ်ပါသည်။")
          elif not u_text.strip():
            st.error("⚠️ Review မှတ်ချက် ရေးသားပေးပါ။")
          else:
            success = insert_vehicle_review(
                vehicle_id, u_name.strip(), u_rating, u_text.strip()
            )
            if success:
              st.session_state["noti_msg"] = (
                  "🎉 သင့် Review ကို အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!"
              )
              st.cache_data.clear()
              st.rerun()
            else:
              st.error(
                  "❌ Review သိမ်းဆည်းရာတွင် အမှားအယွင်း ရှိနေပါသည်။"
              )

    with r_col2:
      st.markdown("#### 💬 Recent Reviews")
      if not reviews_df.empty:
        for _, r_row in reviews_df.iterrows():
          star_display = "⭐" * int(r_row["rating"])
          st.markdown(
              f"""
                    <div style="background-color: #f9fafb; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #3b82f6;">
                        <div style="font-weight: bold; font-size: 14px;">👤 {r_row['user_name']} <span style="float:right;">{star_display} ({r_row['rating']}/5)</span></div>
                        <div style="font-size: 13px; color: #374151; margin-top: 5px;">"{r_row['review_text']}"</div>
                    </div>
                    """,
              unsafe_allow_html=True,
          )
      else:
        st.write("Review များ မရှိသေးပါ။")


# E-COMMERCE STYLE POP-UP MODAL DIALOG FOR COMPARISON
@st.dialog("📊 EV Comparison Matrix", width="large")
def show_compare_modal(df_data):
  selected_names = st.session_state.get("selected_compare", [])

  if not selected_names:
    st.info("ယှဉ်ပြိုင်ရန် ကားများ ရွေးချယ်ထားခြင်း မရှိသေးပါ။")
    return

  comp_df = df_data[df_data["Full_Name"].isin(selected_names)].copy()

  cols = st.columns(len(comp_df))
  for idx, (_, row) in enumerate(comp_df.iterrows()):
    with cols[idx]:
      with st.container(border=True):
        if "Image" in row and pd.notna(row["Image"]):
          st.image(row["Image"], use_container_width=True)
        st.subheader(row["Full_Name"])
        st.markdown(f"**💰 Price:** ${row.get('Price_USD', 0):,}")
        st.markdown(
            f"**💵 Price (MMK):** {row.get('Price_MMK_Lakhs', 0):,.1f} သိန်း"
        )
        st.markdown(f"**🛣️ Range:** {row.get('Range_KM', 0)} KM")
        st.markdown(f"**🔋 Battery:** {row.get('Battery_kWh', 0)} kWh")
        st.markdown(f"**⚡ Fast Charge:** {row.get('FastCharge_KW', 0)} kW")
        st.markdown(
            "**⏱️ 10-80% Time:**"
            f" {row.get('FastCharge_Min_10_80', 'N/A')} mins"
        )
        st.markdown(
            f"**🚀 0-100 km/h:** {row.get('Acceleration_0_100', 0)}s"
        )
        st.markdown(f"**⭐ Rating:** {row.get('Avg_Rating', 0):.1f}/5")

  st.markdown("---")
  if st.button("🗑️ Clear Selected Cars", type="secondary"):
    st.session_state["selected_compare"] = []
    st.rerun()


def validate_url(url):
  regex = re.compile(
      r"^(?:http|ftp)s?://"
      r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
      r"localhost|"
      r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
      r"(?::\d+)?"
      r"(?:/?|[/?]\S+)$",
      re.IGNORECASE,
  )
  return re.match(regex, url) is not None


# ---------------------------------------------------------
# 6. MAIN NAVIGATION TABS
# ---------------------------------------------------------
st.title("⚡ EV Finder & Recommendation Portal")

tab1, tab2, tab3, tab_trip, tab_loan, tab_admin = st.tabs([
    "🚗 EV Finder & Catalog",
    "🧠 Smart Recommender",
    "💡 TCO Analysis",
    "🛣️ Trip & Route Calculator",
    "🏦 EMI Loan Calculator",
    "🛠️ Admin Management",
])

# ---------------------------------------------------------
# TAB 1: EV SEARCH & CATALOG WITH BOTTOM DRAWER COMPARE
# ---------------------------------------------------------
with tab1:
  st.markdown(
      """
        <style>
        [data-testid="stMetricLabel"] { font-size: 15px !important; font-weight: 600 !important; }
        [data-testid="stMetricValue"] { font-size: 18px !important; font-weight: 700 !important; }
        </style>
        """,
      unsafe_allow_html=True,
  )

  col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
  col1.metric("Vehicles Found", len(filtered_df))
  col2.metric(
      "Avg Price ($)",
      f"${filtered_df['Price_USD'].mean():,.0f}"
      if not filtered_df.empty
      else "N/A",
  )
  col3.metric(
      "Fastest Fast Charge",
      f"{filtered_df['FastCharge_KW'].max()} kW"
      if not filtered_df.empty
      else "N/A",
  )

  with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    if not filtered_df.empty:
      csv_data = filtered_df.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Download CSV Report",
          data=csv_data,
          file_name="EV_Filtered_Data.csv",
          mime="text/csv",
          use_container_width=True,
      )

  st.markdown("---")

  if not filtered_df.empty:
    items_per_page = 6
    total_items = len(filtered_df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

    col_p1, col_p2, _ = st.columns([1.5, 2, 4])
    with col_p1:
      page_num = st.number_input(
          "📌 Page Select:",
          min_value=1,
          max_value=total_pages,
          value=1,
          step=1,
      )
    with col_p2:
      st.markdown(
          f"<br><b>Page {page_num} of {total_pages}</b> (Total Vehicles:"
          f" {total_items})",
          unsafe_allow_html=True,
      )

    start_idx = (page_num - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_df = filtered_df.iloc[start_idx:end_idx].reset_index(drop=True)

    st.markdown("---")

    card_cols = st.columns(3)
    for idx, row in page_df.iterrows():
      col_target = card_cols[idx % 3]

      avg_rating = row.get("Avg_Rating", 0)
      review_cnt = int(row.get("Review_Count", 0))
      stars = (
          "⭐" * int(round(avg_rating)) if avg_rating > 0 else "⭐ No Ratings"
      )
      rating_label = f"{stars} ({avg_rating:.1f})" if avg_rating > 0 else stars

      with col_target:
        st.markdown(
            f"""
                <div class="ev-card">
                    <img src="{row['Image']}" alt="{row['Brand']}">
                    <div class="ev-card-title">{row['Brand']} {row['Model']}</div>
                    <div class="rating-badge">{rating_label} • <span style="font-size:12px; color:#6b7280;">({review_cnt} reviews)</span></div>
                    <span class="ev-badge">{row.get('Price_Tier', 'N/A')} • {row['Seats']} Seats</span>
                    <div class="ev-price">${row['Price_USD']:,} <span style="font-size:14px; color:#6b7280;">({row['Price_MMK_Lakhs']:,.1f} သိန်း)</span></div>
                    <div class="ev-spec-grid">
                        <div>🛣️ Range: <b>{row['Range_KM']} km</b></div>
                        <div>🔋 Battery: <b>{row['Battery_kWh']} kWh</b></div>
                        <div>⚡ Fast Charge: <b>{row['FastCharge_KW']} kW</b></div>
                        <div>⏱️ 10-80%: <b>{row.get('FastCharge_Min_10_80', 'N/A')} mins</b></div>
                    </div>
                </div>
                """,
            unsafe_allow_html=True,
        )

        btn_c1, btn_c2 = st.columns([1, 1])
        is_selected = row["Full_Name"] in st.session_state["selected_compare"]

        with btn_c1:
          if is_selected:
            if st.button(
                "❌ Remove",
                key=f"btn_rem_{row['id']}_{idx}",
                type="secondary",
                use_container_width=True,
            ):
              st.session_state["selected_compare"].remove(row["Full_Name"])
              st.rerun()
          else:
            if st.button(
                "➕ Compare",
                key=f"btn_add_{row['id']}_{idx}",
                use_container_width=True,
            ):
              if len(st.session_state["selected_compare"]) >= 4:
                st.toast(
                    "⚠️ အများဆုံး ၄ စီးအထိသာ ယှဉ်ပြိုင်နိုင်ပါသည်။", icon="⚠️"
                )
              else:
                st.session_state["selected_compare"].append(row["Full_Name"])
                st.rerun()

        with btn_c2:
          if st.button(
              "🔍 Specs",
              key=f"card_btn_{row['id']}_{idx}",
              use_container_width=True,
          ):
            show_car_details(row)

    # ---------------------------------------------------------
    # E-COMMERCE STYLE BOTTOM FLOATING DRAWER BAR
    # ---------------------------------------------------------
    if st.session_state["selected_compare"]:
      st.markdown("---")
      with st.container(border=True):
        d_col1, d_col2, d_col3 = st.columns([5, 2, 1])

        with d_col1:
          st.markdown(
              f"**📌 ရွေးချယ်ထားသော ကားများ ({len(st.session_state['selected_compare'])}/4):**"
          )
          badges = [
              f"`🚘 {name}`" for name in st.session_state["selected_compare"]
          ]
          st.markdown(" ".join(badges))

        with d_col2:
          if st.button(
              "📊 Now Compare", type="primary", use_container_width=True
          ):
            show_compare_modal(df)

        with d_col3:
          if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["selected_compare"] = []
            st.rerun()

  else:
    st.warning("No vehicles match your search criteria.")

# ---------------------------------------------------------
# TAB 2: SMART RECOMMENDER
# ---------------------------------------------------------
with tab2:
  st.subheader("🧠 Machine Learning Smart Recommender")
  st.write(
      "သင်ကြိုက်နှစ်သက်သည့် EV ကားကို ရွေးချယ်ပါ - Specifications အတူဆုံး ကား"
      " (၃) စီး** ကို ရှာဖွေပြသပေးပါမည်။"
  )

  if not df.empty and nn_model is not None and "Full_Name" in df.columns:
    target_car = st.selectbox(
        "🎯 Target Vehicle ရွေးချယ်ပါ:", options=df["Full_Name"].tolist()
    )

    if st.button("🔍 Find Similar EVs"):
      target_idx = df[df["Full_Name"] == target_car].index[0]
      target_scaled = X_scaled[target_idx].reshape(1, -1)

      distances, indices = nn_model.kneighbors(target_scaled)
      similar_indices = indices[0][1:]
      similar_cars = df.iloc[similar_indices]

      st.success(f"🎉 **{target_car}** နှင့် Vector Spec အလားတူဆုံး ကားများမှာ:")

      cols = st.columns(len(similar_cars))
      for idx, (_, rec_car) in enumerate(similar_cars.iterrows()):
        with cols[idx]:
          st.image(rec_car["Image"], use_container_width=True)
          st.markdown(f"### {rec_car['Full_Name']}")
          st.write(f"⭐ Rating: **{rec_car['Avg_Rating']:.1f} / 5**")
          st.write(f"🏷️ Tier: **{rec_car.get('Price_Tier', 'N/A')}**")
          st.write(
              f"💰 Price: **${rec_car['Price_USD']:,}**"
              f" ({rec_car['Price_MMK_Lakhs']:,.1f} သိန်း)"
          )
          st.write(f"🔋 Range: **{rec_car['Range_KM']} km**")
          st.write(
              "⏱️ Charging Time:"
              f" **{rec_car.get('FastCharge_Min_10_80', 'N/A')} mins**"
          )
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
    daily_km = st.number_input(
        "တစ်ရက်မောင်းနှင်မည့် အကွာအဝေး (KM):", value=60
    )
    gas_price = st.number_input("ဓာတ်ဆီ ၁ လီတာ ဈေးနှုန်း (MMK):", value=3000)
    gas_kml = st.number_input(
        "ဓာတ်ဆီကား ၁ လီတာ မောင်းနိုင်သည့် အကွာအဝေး (KM):", value=10.0
    )
    gas_maint_year = st.number_input(
        "ဓာတ်ဆီကား ၁ နှစ်စာ ထိန်းသိမ်းစရိတ် (MMK):", value=1200000
    )

  with col_tco2:
    st.markdown("##### ⚡ EV Parameters & Upfront Investment")
    ev_kwh_price = st.number_input(
        "EV လျှပ်စစ် ၁ ယူနစ် (kWh) ဈေးနှုန်း (MMK):", value=500
    )
    ev_kmkwh = st.number_input(
        "EV ၁ ယူနစ် မောင်းနိုင်သည့် အကွာအဝေး (KM):", value=6.0
    )
    ev_maint_year = st.number_input(
        "EV ကား ၁ နှစ်စာ ထိန်းသိမ်းစရိတ် (MMK):", value=400000
    )
    ev_extra_upfront = st.number_input(
        "EV ဝယ်ယူရန် ပိုကုန်ကျမည့် စရိတ် - Initial Premium (MMK):",
        value=20000000,
    )

  annual_km = daily_km * 365
  annual_gas_fuel = (annual_km / gas_kml) * gas_price
  annual_gas_total = annual_gas_fuel + gas_maint_year

  annual_ev_fuel = (annual_km / ev_kmkwh) * ev_kwh_price
  annual_ev_total = annual_ev_fuel + ev_maint_year

  annual_saving = annual_gas_total - annual_ev_total
  breakeven_years = (
      ev_extra_upfront / annual_saving if annual_saving > 0 else 0
  )

  st.markdown("---")
  m1, m2, m3 = st.columns(3)
  m1.metric("ဓာတ်ဆီကား နှစ်စဉ် စရိတ်ပေါင်း", f"{annual_gas_total:,.0f} MMK")
  m2.metric("EV ကား နှစ်စဉ် စရိတ်ပေါင်း", f"{annual_ev_total:,.0f} MMK")
  m3.metric(
      "🎯 အရင်းကျေမည့် ကြာချိန် (Breakeven)", f"{breakeven_years:.1f} နှစ် (Years)"
  )

  years = np.arange(0, 11)
  gas_cumulative = years * annual_gas_total
  ev_cumulative = ev_extra_upfront + (years * annual_ev_total)

  fig_break = go.Figure()
  fig_break.add_trace(
      go.Scatter(
          x=years,
          y=gas_cumulative,
          mode="lines+markers",
          name="Gas Vehicle Cumulative Cost",
      )
  )
  fig_break.add_trace(
      go.Scatter(
          x=years,
          y=ev_cumulative,
          mode="lines+markers",
          name="EV Vehicle Cumulative Cost",
      )
  )
  fig_break.update_layout(
      title="📈 Cumulative Cost Comparison over 10 Years",
      xaxis_title="Years of Ownership",
      yaxis_title="Total Cost (MMK)",
  )
  st.plotly_chart(fig_break, use_container_width=True)

# ---------------------------------------------------------
# TAB TRIP & ROUTE RANGE CALCULATOR
# ---------------------------------------------------------
with tab_trip:
  st.subheader("🛣️ EV Trip & Route Range Calculator")
  if not df.empty and "Full_Name" in df.columns:
    trip_col1, trip_col2 = st.columns(2)
    with trip_col1:
      selected_trip_car = st.selectbox(
          "🚘 မောင်းနှင်မည့် EV ကားကို ရွေးချယ်ပါ:",
          options=df["Full_Name"].tolist(),
          key="trip_car",
      )
      trip_distance = st.number_input(
          "📍 သွားရောက်မည့် ခရီးစဉ်အကွာအဝေး Total Distance (KM):",
          value=450,
          step=10,
      )
      traffic_factor = st.slider(
          "🚦 လမ်းခရီး/ရာသီဥတု သက်ရောက်မှု (Buffer Efficiency %):",
          min_value=70,
          max_value=100,
          value=85,
      )

    car_data = df[df["Full_Name"] == selected_trip_car].iloc[0]
    effective_range = car_data["Range_KM"] * (traffic_factor / 100.0)
    needed_charges = max(0, int(np.ceil(trip_distance / effective_range)) - 1)

    kwh_per_km = (
        car_data["Battery_kWh"] / car_data["Range_KM"]
        if car_data["Range_KM"] > 0
        else 0.15
    )
    total_kwh_needed = trip_distance * kwh_per_km
    estimated_trip_cost = total_kwh_needed * 500

    with trip_col2:
      st.markdown(f"#### 📊 Trip Summary: {car_data['Full_Name']}")
      st.write(
          f"🔋 **Effective Range:** `{effective_range:.0f} KM` (Max:"
          f" {car_data['Range_KM']} KM)"
      )

      res_col1, res_col2 = st.columns(2)
      res_col1.metric("⚡ လမ်းခရီး အားသွင်းရမည့် အကြိမ်", f"{needed_charges} ကြိမ်")
      res_col2.metric("💰 ခန့်မှန်း အားသွင်းစရိတ်", f"{estimated_trip_cost:,.0f} MMK")

      if needed_charges > 0:
        st.info(
            f"💡 ခရီးစဉ်မပြီးဆုံးမီ လမ်းခရီးတွင် အနည်းဆုံး **{needed_charges}"
            " ကြိမ်** DC Fast Charger ၌ အားသွင်းရန် လိုအပ်ပါမည်။"
        )
      else:
        st.success(
            "🎉 အားတစ်ကြိမ် အပြည့်သွင်းရုံဖြင့် ခရီးစဉ်ဆုံးသည်ထိ"
            " မောင်းနှင်နိုင်ပါသည်။"
        )

# ---------------------------------------------------------
# TAB LOAN & FINANCING EMI CALCULATOR
# ---------------------------------------------------------
with tab_loan:
  st.subheader("🏦 EV Loan & Financing EMI Calculator")
  loan_col1, loan_col2 = st.columns(2)
  with loan_col1:
    if not df.empty and "Full_Name" in df.columns:
      loan_car = st.selectbox(
          "🚘 ဝယ်ယူလိုသော EV ကား ရွေးပါ:",
          options=df["Full_Name"].tolist(),
          key="loan_car",
      )
      car_price_mmk = df[df["Full_Name"] == loan_car]["Price_MMK"].values[0]
      st.info(
          f"💵 ဈေးနှုန်း: **{car_price_mmk/100000:,.1f} သိန်း**"
          f" ({car_price_mmk:,.0f} MMK)"
      )
    else:
      car_price_mmk = 100000000

    down_payment_pct = st.slider(
        "💰 Down Payment (%):", min_value=10, max_value=50, value=30, step=5
    )
    interest_rate_ann = st.number_input(
        "📈 နှစ်စဉ် အတိုးနှုန်း Annual Interest Rate (%):", value=13.0, step=0.5
    )
    loan_years = st.selectbox(
        "📅 ချေးငွေ သက်တမ်း (Years):", options=[1, 2, 3, 4, 5], index=2
    )

  down_payment_amount = car_price_mmk * (down_payment_pct / 100.0)
  loan_amount = car_price_mmk - down_payment_amount
  monthly_rate = (interest_rate_ann / 100) / 12
  months = loan_years * 12

  emi = (
      loan_amount
      * (monthly_rate * (1 + monthly_rate) ** months)
      / ((1 + monthly_rate) ** months - 1)
      if monthly_rate > 0
      else loan_amount / months
  )
  total_interest = (emi * months) - loan_amount

  with loan_col2:
    st.markdown("#### 💳 Payment Breakdown")
    st.metric(
        "ကနဦး ပေးသွင်းရမည့်ငွေ (Down Payment)",
        f"{down_payment_amount/100000:,.1f} သိန်း",
    )
    st.metric(
        "လစဉ် ပေးသွင်းရမည့်ငွေ (Monthly EMI)",
        f"{emi/100000:,.2f} သိန်း ({emi:,.0f} MMK)",
    )
    st.metric(
        "စုစုပေါင်း အတိုးစရိတ် (Total Interest)",
        f"{total_interest/100000:,.1f} သိန်း",
    )

# ---------------------------------------------------------
# TAB 5: POSTGRESQL ADMIN MANAGEMENT PANEL
# ---------------------------------------------------------
with tab_admin:
  st.subheader("🛠️ PostgreSQL Admin Management Panel")
  admin_pass = st.text_input(
      "🔐 Admin Password ထည့်သွင်းပါ:", type="password", key="admin_pass_input"
  )
  expected_pass = st.secrets.get("admin", {}).get("password", "admin123")

  if admin_pass == expected_pass:
    st.success("✅ Admin Authorization Approved!")

    admin_sub_tab1, admin_sub_tab2, admin_sub_tab3, admin_sub_tab4 = st.tabs([
        "📥 Bulk CSV Upload",
        "➕ Add Vehicle",
        "✏️ Edit Vehicle Record",
        "🗄️ Manage Records",
    ])

    with admin_sub_tab1:
      uploaded_file = st.file_uploader(
          "Upload CSV / XLSX File", type=["csv", "xlsx"]
      )
      if uploaded_file is not None:
        upload_df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )
        st.dataframe(upload_df.head(5), use_container_width=True)
        if st.button("🚀 PostgreSQL သို့ Data များ တိုက်ရိုက် Upload တင်မည်"):
          try:
            db_col_map = {
                "Image": "image_url",
                "Brand": "brand",
                "Model": "model",
                "Seats": "seats",
                "Price_USD": "price_usd",
                "Range_KM": "range_km",
                "Battery_kWh": "battery_kwh",
                "Weight_KG": "weight_kg",
                "FastCharge_KW": "fastcharge_kw",
                "Acceleration_0_100": "acceleration_0_100",
                "FastCharge_Min_10_80": "fastcharge_min_10_80",
            }
            upload_df.rename(columns=db_col_map, inplace=True)
            upload_df.to_sql(
                "ev_vehicles", con=engine, if_exists="append", index=False
            )
            st.session_state["noti_msg"] = (
                f"📥 Bulk Data ({len(upload_df)} စီး) ကို Upload တင်ပြီးပါပြီ!"
            )
            st.cache_data.clear()
            st.rerun()
          except Exception as e:
            st.error(f"❌ Upload Error: {e}")

    with admin_sub_tab2:
      st.markdown("#### EV Vehicle Data အသစ် ထည့်သွင်းရန် (Validation ပါဝင်သည်)")
      with st.form("add_single_ev_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)
        with col_a:
          brand = st.text_input("Brand (ဥပမာ - BYD)")
          model = st.text_input("Model (ဥပမာ - Atto 3)")
          seats = st.number_input("Seats", min_value=1, max_value=10, value=5)
          price_usd = st.number_input(
              "Price ($ USD)", min_value=0.0, step=500.0, value=38000.0
          )
          range_km = st.number_input("Range (KM)", min_value=0, step=10, value=420)

        with col_b:
          battery_kwh = st.number_input(
              "Battery (kWh)", min_value=0.0, step=1.0, value=60.4
          )
          weight_kg = st.number_input(
              "Weight (KG)", min_value=0, step=50, value=1750
          )
          fastcharge_kw = st.number_input(
              "Fast Charge Speed (kW)", min_value=0, step=5, value=88
          )
          fastcharge_min_10_80 = st.number_input(
              "Charging Time 10-80% (Mins)", min_value=0, step=1, value=30
          )
          acceleration = st.number_input(
              "0-100 km/h (s)", min_value=0.0, step=0.1, value=7.3
          )
          image_url = st.text_input(
              "Image URL Link",
              value="https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=600&h=400&fit=crop",
          )

        submit_ev = st.form_submit_button("💾 Save to Database")

        if submit_ev:
          if not brand.strip() or not model.strip():
            st.error("⚠️ Brand နှင့် Model အမည် ဖြည့်သွင်းရန် လိုအပ်ပါသည်။")
          elif not validate_url(image_url):
            st.error(
                "⚠️ မှန်ကန်သော Image URL Link (http/https) ထည့်သွင်းပေးပါ။"
            )
          elif price_usd <= 0 or range_km <= 0 or battery_kwh <= 0:
            st.error(
                "⚠️ Price, Range နှင့် Battery Value များသည် 0 ထက် ကြီးရပါမည်။"
            )
          else:
            insert_query = text("""
                            INSERT INTO ev_vehicles (image_url, brand, model, seats, price_usd, range_km, battery_kwh, weight_kg, fastcharge_kw, acceleration_0_100, fastcharge_min_10_80)
                            VALUES (:image_url, :brand, :model, :seats, :price_usd, :range_km, :battery_kwh, :weight_kg, :fastcharge_kw, :acceleration, :fastcharge_min_10_80)
                        """)
            try:
              with engine.begin() as conn:
                conn.execute(
                    insert_query,
                    {
                        "image_url": image_url,
                        "brand": brand,
                        "model": model,
                        "seats": seats,
                        "price_usd": price_usd,
                        "range_km": range_km,
                        "battery_kwh": battery_kwh,
                        "weight_kg": weight_kg,
                        "fastcharge_kw": fastcharge_kw,
                        "acceleration": acceleration,
                        "fastcharge_min_10_80": fastcharge_min_10_80,
                    },
                )
              st.session_state["noti_msg"] = (
                  f"🎉 🚘 {brand} {model} ကို အောင်မြင်စွာ ထည့်သွင်းပြီးပါပြီ!"
              )
              st.cache_data.clear()
              st.rerun()
            except Exception as e:
              st.error(f"❌ Database Error: {e}")

    with admin_sub_tab3:
      st.markdown("#### ✏️ Existing Vehicle Data ပြင်ဆင်ရန် (Update Record)")
      current_db_df = load_data_from_db()

      if not current_db_df.empty:
        car_options = {
            f"ID {row['id']}: {row['Brand']} {row['Model']}": row["id"]
            for _, row in current_db_df.iterrows()
        }
        selected_car_str = st.selectbox(
            "✏️ ပြင်ဆင်လိုသည့် EV ကားကို ရွေးချယ်ပါ:",
            options=list(car_options.keys()),
        )
        selected_id = car_options[selected_car_str]

        edit_row = current_db_df[current_db_df["id"] == selected_id].iloc[0]

        with st.form("edit_ev_form"):
          col_e1, col_e2 = st.columns(2)
          with col_e1:
            e_brand = st.text_input("Brand", value=str(edit_row["Brand"]))
            e_model = st.text_input("Model", value=str(edit_row["Model"]))
            e_seats = st.number_input(
                "Seats", min_value=1, max_value=10, value=int(edit_row["Seats"])
            )
            e_price = st.number_input(
                "Price ($ USD)",
                min_value=0.0,
                value=float(edit_row["Price_USD"]),
            )
            e_range = st.number_input(
                "Range (KM)", min_value=0, value=int(edit_row["Range_KM"])
            )

          with col_e2:
            e_battery = st.number_input(
                "Battery (kWh)",
                min_value=0.0,
                value=float(edit_row["Battery_kWh"]),
            )
            e_weight = st.number_input(
                "Weight (KG)", min_value=0, value=int(edit_row["Weight_KG"])
            )
            e_fc_kw = st.number_input(
                "Fast Charge (kW)",
                min_value=0,
                value=int(edit_row["FastCharge_KW"]),
            )
            e_fc_min = st.number_input(
                "Charge Time 10-80% (Mins)",
                min_value=0,
                value=int(edit_row.get("FastCharge_Min_10_80", 30)),
            )
            e_accel = st.number_input(
                "0-100 km/h (s)",
                min_value=0.0,
                value=float(edit_row["Acceleration_0_100"]),
            )
            e_img = st.text_input("Image URL Link", value=str(edit_row["Image"]))

          btn_update = st.form_submit_button("🔄 Update Data")

          if btn_update:
            if not e_brand.strip() or not e_model.strip():
              st.error("⚠️ Brand နှင့် Model အမည် ဖြည့်သွင်းရန် လိုအပ်ပါသည်။")
            elif not validate_url(e_img):
              st.error("⚠️ မှန်ကန်သော Image URL Link ထည့်သွင်းပေးပါ။")
            else:
              update_query = text("""
                                UPDATE ev_vehicles 
                                SET brand=:brand, model=:model, seats=:seats, price_usd=:price, 
                                    range_km=:range, battery_kwh=:battery, weight_kg=:weight, 
                                    fastcharge_kw=:fc_kw, fastcharge_min_10_80=:fc_min, 
                                    acceleration_0_100=:accel, image_url=:img
                                WHERE id=:id
                            """)
              try:
                with engine.begin() as conn:
                  conn.execute(
                      update_query,
                      {
                          "brand": e_brand,
                          "model": e_model,
                          "seats": e_seats,
                          "price": e_price,
                          "range": e_range,
                          "battery": e_battery,
                          "weight": e_weight,
                          "fc_kw": e_fc_kw,
                          "fc_min": e_fc_min,
                          "accel": e_accel,
                          "img": e_img,
                          "id": selected_id,
                      },
                  )
                st.session_state["noti_msg"] = (
                    f"🎉 Record ID ({selected_id}) ကို အောင်မြင်စွာ Update"
                    " ပြုလုပ်ပြီးပါပြီ!"
                )
                st.cache_data.clear()
                st.rerun()
              except Exception as e:
                st.error(f"❌ Update Error: {e}")

    with admin_sub_tab4:
      st.markdown("#### 🗄️ Database Record ဖျက်ထုတ်ခြင်း")
      current_db_df = load_data_from_db()
      st.dataframe(current_db_df, use_container_width=True)

      st.divider()
      delete_id = st.number_input(
          "ဖျက်လိုသော ID ရိုက်ထည့်ပါ:", min_value=1, step=1
      )
      if st.button("❌ Record ကို ဖျက်မည်"):
        try:
          with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM ev_vehicles WHERE id = :id"),
                {"id": delete_id},
            )
            conn.execute(
                text("DELETE FROM ev_reviews WHERE vehicle_id = :id"),
                {"id": delete_id},
            )
          st.session_state["noti_msg"] = (
              f"🗑️ Record ID ({delete_id}) နှင့် သက်ဆိုင်သော Review များကို"
              " ဖျက်ပြီးပါပြီ!"
          )
          st.cache_data.clear()
          st.rerun()
        except Exception as e:
          st.error(f"❌ Delete Error: {e}")

# ---------------------------------------------------------
# 7. GLOBAL NOTIFICATION TOAST DISPATCHER
# ---------------------------------------------------------
if "noti_msg" in st.session_state and st.session_state["noti_msg"]:
  st.toast(st.session_state["noti_msg"], icon="⚡")
  st.session_state["noti_msg"] = None
