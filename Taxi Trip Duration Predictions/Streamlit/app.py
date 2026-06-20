import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# ==================================
# Page Configuration
# ==================================

st.set_page_config(
    page_title="NYC Taxi Trip Duration Analytics Dashboard",
    layout="wide"
)

# ==================================
# Load Model
# ==================================

model = joblib.load("Models/xgboost_model.pkl")

# ==================================
# Mapping Dictionaries
# ==================================

vendor_mapping = {
    "Creative Mobile Technologies": 1,
    "VeriFone": 2
}

fare_mapping = {
    "Standard Rate": 1,
    "JFK Airport": 2,
    "Newark Airport": 3,
    "Nassau / Westchester": 4,
    "Negotiated Fare": 5,
    "Group Ride": 6
}

day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

hour_options = [
    f"{i:02d}:00"
    for i in range(24)
]

st.markdown("""
<style>

/* =======================
Main Layout
======================= */

.main{
    background-color:#F8FAFC;
}

.block-container{
    max-width:1300px;
    padding-top:2rem;
}

/* =======================
Titles
======================= */

h1{
    color:#0F172A;
    font-size:3rem;
    font-weight:800;
}

h2,h3{
    color:#1E293B;
    font-weight:700;
}

/* =======================
Metric Cards
======================= */

div[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:18px;
    padding:22px;
    box-shadow:
    0 4px 10px rgba(0,0,0,0.05);
    transition:0.3s ease;
}

div[data-testid="stMetric"]:hover{
    transform:translateY(-4px);
    box-shadow:
    0 10px 20px rgba(0,0,0,0.10);
}

/* =======================
Inputs
======================= */

.stSelectbox div[data-baseweb="select"]{
    border-radius:12px;
}

.stNumberInput input{
    border-radius:12px;
}

/* =======================
Button
======================= */

.stButton>button{
    background:#1E3A8A;
    color:white;
    border:none;
    border-radius:14px;
    height:60px;
    font-size:18px;
    font-weight:700;
    transition:0.3s;
}

.stButton>button:hover{
    background:#0F172A;
    transform:translateY(-2px);
}

/* =======================
Prediction Cards
======================= */

.pred-card{
    background:white;
    padding:25px;
    border-radius:18px;
    border:1px solid #E2E8F0;
    box-shadow:
    0 4px 10px rgba(0,0,0,0.05);
}

/* =======================
Traffic Badge
======================= */

.badge{
    padding:18px;
    border-radius:12px;
    font-size:20px;
    font-weight:700;
}

/* =======================
Business Insight
======================= */

.insight{
    background:#FFFFFF;
    padding:25px;
    border-radius:18px;
    border-left:6px solid #1E3A8A;
    box-shadow:
    0 4px 10px rgba(0,0,0,0.05);
}

/* =======================
Footer
======================= */

.footer{

    text-align:center;
    color:#64748B;
    padding:20px;
    line-height:2;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# Header
# ==================================

st.title("NYC Taxi Trip Duration Analytics Dashboard")

st.markdown("""
### Transportation Decision Support System

This dashboard predicts New York City taxi trip duration using an XGBoost machine learning model to support fleet management, route optimization, and transportation operational efficiency.
""")
st.divider()

# ==================================
# Layout
# ==================================

left_col, right_col = st.columns([2,1])

with left_col:

    st.markdown("## Trip Information")

    st.caption(
        "Enter trip characteristics to estimate travel duration."
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_vendor = st.selectbox(
            "Taxi Provider",
            list(vendor_mapping.keys())
        )

        VendorID = vendor_mapping[selected_vendor]

        passenger_count = st.number_input(
            "Number of Passengers",
            min_value=1,
            max_value=6,
            value=2
        )

        trip_distance = st.number_input(
            "Trip Distance (Miles)",
            min_value=0.1,
            value=3.5
        )

        selected_fare = st.selectbox(
            "Fare Type",
            list(fare_mapping.keys())
        )

        RatecodeID = fare_mapping[selected_fare]

    with col2:

        PULocationID = st.number_input(
            "Pickup Zone ID (NYC TLC)",
            value=161,
            help="NYC Taxi Zone Identifier"
        )

        DOLocationID = st.number_input(
            "Dropoff Zone ID (NYC TLC)",
            value=236,
            help="NYC Taxi Zone Identifier"
        )

        selected_hour = st.text_input(
            "Pickup Time (HH:MM)",
            value="08:00"
        )

        try:

            hour, minute = map(int, selected_hour.split(":"))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                pickup_hour = hour

            else:

                st.error("Enter a valid time (00:00 - 23:59).")
                st.stop()

        except:

            st.error("Use HH:MM format (e.g., 08:00).")
            st.stop()

        selected_day = st.selectbox(
            "Day of Week",
            list(day_mapping.keys()),
            index=2
        )

        day_of_week = day_mapping[selected_day]

        # Weekend otomatis mengikuti hari

        if selected_day in ["Saturday", "Sunday"]:
            weekend_option = "Yes"
            is_weekend = 1

        else:
            weekend_option = "No"
            is_weekend = 0

        st.selectbox(
            "Weekend Trip",
            [weekend_option],
            disabled=True
        )

# ==================================
# Sidebar Summary
# ==================================

with right_col:

    st.subheader("Trip Summary")

    st.metric(
        label="Trip Distance",
        value=f"{trip_distance:.1f} miles"
    )

    st.metric(
        label="Passengers",
        value=passenger_count
    )

    st.metric(
        label="Pickup Time",
        value=selected_hour
    )

    st.metric(
        label="Day",
        value=selected_day
    )

    st.metric(
        label="Weekend Trip",
        value=weekend_option
    )

# ==================================
# Prediction Data
# ==================================

input_data = pd.DataFrame({

    'VendorID':[VendorID],
    'passenger_count':[passenger_count],
    'trip_distance':[trip_distance],
    'RatecodeID':[RatecodeID],
    'PULocationID':[PULocationID],
    'DOLocationID':[DOLocationID],
    'pickup_hour':[pickup_hour],
    'day_of_week':[day_of_week],
    'is_weekend':[is_weekend]

})
# ==================================
# Predict
# ==================================
st.divider()
if st.button("Predict Duration", use_container_width=True):

    prediction = model.predict(input_data)
    duration = prediction[0]

    # Calculate estimated arrival time

    departure_time = datetime.strptime(
        selected_hour,
        "%H:%M"
    )

    arrival_time = departure_time + timedelta(
        minutes=float(duration)
    )

    arrival_time = arrival_time.strftime("%H:%M")
    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Estimated Duration",
            f"{duration:.2f} minutes"
        )

    with result_col2:
        st.metric(
            "Estimated Arrival Time",
            arrival_time
        )

    with result_col3:
        if duration < 15:
            st.success(
                "Traffic Condition: Low Traffic"
            )

        elif duration < 30:
            st.warning(
                "Traffic Condition: Moderate Traffic"
            )

        else:
            st.error(
                "Traffic Condition: Heavy Traffic"
            )
    # =====================
    # Dynamic Business Insight
    # =====================

    st.divider()
    st.subheader("Business Insight & Recommendation")

    if duration < 15:

        st.markdown("""

    ### Efficient Operations

    Traffic flow is efficient with minimal congestion.

    **Recommendations**

    - Increase ride promotions
    - Maximize fleet utilization
    - Maintain current routing strategies

    """)

    elif duration < 30:

        st.markdown("""

    ### Moderate Traffic Activity

    Traffic conditions are balanced with moderate congestion.

    **Recommendations**

    - Monitor demand fluctuations
    - Optimize route assignments
    - Allocate additional drivers during peak periods

    """)

    else:

        st.markdown("""

    ### High Traffic Congestion

    Traffic density is high and may reduce operational efficiency.

    **Recommendations**

    - Increase driver allocation
    - Suggest alternative routes
    - Improve dispatch planning
    - Implement dynamic pricing strategies

    """)

# ==================================
# Footer
# ==================================

st.divider()

st.markdown(
"""

<div class='footer'>
<b>NYC Taxi Trip Duration Analytics Dashboard</b>
<br>
Machine Learning Model: XGBoost
<br>
Purpose: Support transportation operational efficiency through predictive analytics
<br>
Author: Retno Lintang
</div>
""",

unsafe_allow_html=True
)