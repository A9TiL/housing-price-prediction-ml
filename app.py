import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

if 'app_started' not in st.session_state:
    st.session_state.app_started = False

def start_app():
    st.session_state.app_started = True

# -----------------------------------------------------------------------------
# 2. Critical Dependency: Redefine Custom Transformer
# -----------------------------------------------------------------------------
def add_extra_features(X):
    rooms_per_household = X[:, 3] / X[:, 6]
    bedrooms_per_room = X[:, 4] / X[:, 3]
    population_per_household = X[:, 5] / X[:, 6]
    return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]

# -----------------------------------------------------------------------------
# 3. Load the Models (Cached so it only runs once)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_deployment_package():
    return joblib.load('artifacts/deployment_package.joblib')

try:
    deployment_package = load_deployment_package()
except FileNotFoundError:
    st.error("Error: deployment_package.joblib not found. Please ensure it is in the 'artifacts' folder.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. User Interface: Ad-Style Welcome & Onboarding
# -----------------------------------------------------------------------------
if not st.session_state.app_started:
    st.markdown("<div style='margin-top: 10vh;'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #f0f2f6 0%, #e1eaf5 100%); border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
    <h1 style="color: #1f77b4; font-size: 3.5rem; margin-bottom: 0.5rem; font-weight: 800;">🏡 Find the <span style="color: #ff7f0e;">Right</span> Value.</h1>
    <h3 style="color: #444; font-weight: 400; margin-bottom: 1.5rem; font-size: 1.5rem;">The smartest way to navigate the Golden State's real estate market.</h3>
    <p style="font-size: 1.2rem; color: #555; max-width: 800px; margin: 1rem auto; line-height: 1.6;">
        Every person needs a house, and knowing its true, fair market value is the most critical step of the journey. 
        Whether you already live in California and want to evaluate your property, or you're planning to shift to the West Coast to start a new chapter—<b>we got you.</b>
    </p>
    <p style="font-size: 1.2rem; color: #555; max-width: 800px; margin: 0 auto; line-height: 1.6;">
        We are your intelligent AI real estate companion. Our advanced machine learning engines analyze decades of geographic, demographic, and economic data in milliseconds. Skip the guesswork. Just tell us about the neighborhood, and let our algorithms give you the right value, right now.
    </p>
</div>
""", unsafe_allow_html=True)


if not st.session_state.app_started:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("Start Property Valuation ➔", type="primary", use_container_width=True, on_click=start_app)
    
    
    st.stop()

st.divider()

# -----------------------------------------------------------------------------
# 5. User Interface: Model Selection
# -----------------------------------------------------------------------------
st.subheader("⚙️ Configure Prediction Engine")


selected_model_name = st.selectbox(
    "Select the underlying machine learning model:", 
    options=list(deployment_package.keys())
)


active_pipeline = deployment_package[selected_model_name]["pipeline"]
active_metrics = deployment_package[selected_model_name]["metrics"]


st.info(f"**Current Engine Performance:** Average Error (RMSE): ${active_metrics['RMSE']:,.0f} | Average Error (MAE): ${active_metrics['MAE']:,.0f}")

st.divider()

# -----------------------------------------------------------------------------
# 6. User Interface: Input Form
# -----------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("📍 Property Characteristics")
    st.markdown("Adjust the sliders and inputs below to define the property and its surrounding neighborhood.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        longitude = st.number_input("Longitude", value=-122.23, format="%.2f", help="Geographic coordinate (West/East)")
        latitude = st.number_input("Latitude", value=37.88, format="%.2f", help="Geographic coordinate (North/South)")
        ocean_proximity = st.selectbox("Ocean Proximity", options=['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])
        
    with col2:
        housing_median_age = st.slider("Housing Median Age (Years)", min_value=1.0, max_value=100.0, value=41.0)
        total_rooms = st.number_input("Total Rooms (Block)", min_value=1.0, value=880.0)
       
        total_bedrooms = st.number_input("Total Bedrooms (Optional)", min_value=1.0, value=129.0, placeholder="Leave blank to impute")
        
    with col3:
        population = st.number_input("Population (Block)", min_value=1.0, value=322.0)
        households = st.number_input("Households (Block)", min_value=1.0, value=126.0)
        median_income = st.slider("Median Income (Tens of Thousands $)", min_value=0.1, max_value=15.0, value=8.32, format="%.2f")

    submit_button = st.form_submit_button(label="Calculate Property Value", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 7. Prediction Execution
# -----------------------------------------------------------------------------
if submit_button:
   
    user_data = pd.DataFrame([{
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms if pd.notna(total_bedrooms) else np.nan,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }])

    
    


    with st.spinner("Analyzing demographic data, evaluating location, and generating prediction..."):
        
        start_time = time.time()
        prediction = active_pipeline.predict(user_data)[0]
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        st.write(f"Latency: {latency_ms:.2f} ms")
        
    st.success("Analysis Complete!")
    
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background-color: #d4edda; border-radius: 10px; border: 2px solid #c3e6cb; margin-top: 1rem;">
        <h2 style="color: #155724; margin-bottom: 0;">Estimated Market Value</h2>
        <h1 style="color: #28a745; font-size: 4rem; margin-top: 0.5rem;">${prediction:,.2f}</h1>
        <p style="color: #155724; font-size: 1.2rem;">Generated by the <b>{selected_model_name}</b> engine.</p>
    </div>
    """, unsafe_allow_html=True)