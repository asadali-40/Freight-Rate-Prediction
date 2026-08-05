import streamlit as st
import pandas as pd
from datetime import date
from catboost import CatBoostRegressor

st.set_page_config(page_title="Freight Rate Estimator", page_icon="🚛", layout="centered")

MODEL_PATH = "catboost_model.cbm"

# Same order/names of features the model was trained on in the notebook
FEATURE_ORDER = [
    "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment",
    "weight", "market_index", "quote_signal",
    "month", "day", "day_of_week", "week_of_year", "is_weekend",
]


@st.cache_resource
def load_model(path: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(path)
    return model


st.title("🚛 Freight Rate Estimator")
st.caption("Predicts the posted rate for a shipment using the CatBoost model trained in your notebook.")

# --- Load the bundled model (ships with the app, nothing for users to upload) ---
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        f"Couldn't load the bundled model file `{MODEL_PATH}`. "
        f"Make sure it's in the same folder as app.py when you deploy. ({e})"
    )
    st.stop()

st.divider()
st.subheader("Shipment details")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Pickup**")
    pickup = st.text_input("Pickup city", "Richmond")
    pickup_lat = st.number_input("Pickup latitude", value=38.09122, format="%.5f")
    pickup_lon = st.number_input("Pickup longitude", value=-76.78906, format="%.5f")
with col2:
    st.markdown("**Delivery**")
    delivery = st.text_input("Delivery city", "Baltimore")
    delivery_lat = st.number_input("Delivery latitude", value=38.16908, format="%.5f")
    delivery_lon = st.number_input("Delivery longitude", value=-76.74564, format="%.5f")

distance = st.number_input("Distance (miles)", min_value=0.0, value=274.3, step=1.0)
equipment = st.selectbox("Equipment type", ["Dry Van", "Reefer", "Flatbed"])
weight = st.number_input("Weight (lbs)", min_value=0.0, value=30000.0, step=100.0)

col3, col4 = st.columns(2)
with col3:
    market_index = st.number_input("Market index", value=1.0, format="%.5f")
with col4:
    quote_signal = st.number_input("Quote signal", value=2.0, format="%.5f")

ship_date = st.date_input("Shipment date", value=date.today())

if st.button("Predict rate", type="primary"):
    month = ship_date.month
    day = ship_date.day
    day_of_week = ship_date.weekday()
    week_of_year = ship_date.isocalendar()[1]
    is_weekend = int(day_of_week >= 5)

    input_row = {
        "pickup": pickup,
        "delivery": delivery,
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "delivery_lat": delivery_lat,
        "delivery_lon": delivery_lon,
        "distance": distance,
        "equipment": equipment,
        "weight": weight,
        "market_index": market_index,
        "quote_signal": quote_signal,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "week_of_year": week_of_year,
        "is_weekend": is_weekend,
    }
    input_df = pd.DataFrame([input_row])[FEATURE_ORDER]

    prediction = model.predict(input_df)[0]
    st.success(f"### Predicted posted rate: **${prediction:,.2f}**")

    with st.expander("See the exact input sent to the model"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "Model: CatBoostRegressor (iterations=164, learning_rate=0.05, depth=8). "
    "Validation performance from your notebook — MAE: 97.51, RMSE: 513.56, R²: 0.8767."
)
