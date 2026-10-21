
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Cancellation Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "Customer_Cancellations_Training.csv"
API_URL = "https://customer-cancellation-balanced-2.onrender.com/predict"

# ---------------- PROFESSIONAL THEME ----------------
st.markdown("""
<style>
.stApp {
    background: #f3f6fb !important;
    color: #172033 !important;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #10213e !important;
    border-right: 1px solid #263d5d;
}
[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #10213e !important;
}
.stApp h1, .stApp h2, .stApp h3 {
    color: #14243c !important;
}
.stApp label,
.stApp [data-testid="stWidgetLabel"] p {
    color: #253750 !important;
    font-weight: 600 !important;
}
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="base-input"] {
    background: #ffffff !important;
    color: #172033 !important;
    border-color: #cbd5e1 !important;
}
.stApp [data-baseweb="select"] *,
.stApp input {
    color: #172033 !important;
}
.hero {
    background: linear-gradient(120deg, #10213e, #1769aa);
    padding: 30px 32px;
    border-radius: 18px;
    color: white !important;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(16,33,62,.12);
}
.hero h1 {
    color: white !important;
    font-size: 32px;
    margin-bottom: 8px;
}
.hero p {
    color: #e4efff !important;
    font-size: 15px;
}
.metric-card {
    background: #ffffff;
    padding: 22px;
    border: 1px solid #dbe5f0;
    border-radius: 16px;
    min-height: 115px;
    box-shadow: 0 4px 14px rgba(20,40,70,.06);
}
.metric-label {
    color: #52647b !important;
    font-size: 13px;
    font-weight: 700;
}
.metric-value {
    color: #14243c !important;
    font-size: 26px;
    font-weight: 750;
    margin-top: 8px;
}
div.stButton > button,
div.stFormSubmitButton > button {
    background: #1769aa !important;
    color: white !important;
    border: 0 !important;
    border-radius: 10px !important;
    min-height: 48px;
    font-weight: 700;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #105486 !important;
    color: white !important;
}
.stApp [data-testid="stMarkdownContainer"] p {
    color: #334155;
}
.small-note {
    color: #64748b !important;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD TRAINING DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


if not os.path.exists(DATA_PATH):
    st.error(f"Training dataset not found: {DATA_PATH}")
    st.info("Ensure the CSV file is present in the GitHub repository.")
    st.stop()

try:
    data = load_data()
except Exception as error:
    st.error(f"Could not load training dataset: {error}")
    st.stop()


# ---------------- INPUT OPTIONS ----------------
required_data_columns = [
    "Territory", "DeviceType", "PlanType", "StartMonth", "StartWeek",
    "CustomerTenure", "BaseFee", "AdditionalFeatureSpend",
    "CurrentPeriodUsage", "PriorPeriodUsage", "NumberOfPenalties",
    "ServiceTickets", "ServiceRating", "Promotion"
]

missing_columns = [
    col for col in required_data_columns if col not in data.columns
]

if missing_columns:
    st.error(f"Dataset is missing required columns: {missing_columns}")
    st.stop()

territories = sorted(data["Territory"].dropna().unique().tolist())
device_types = sorted(data["DeviceType"].dropna().unique().tolist())
plan_types = sorted(data["PlanType"].dropna().unique().tolist())
promotion_options = sorted(data["Promotion"].dropna().unique().tolist())

months = sorted(
    data["StartMonth"].dropna().unique().tolist(),
    key=lambda x: pd.to_datetime(str(x), format="%b").month
)
weeks = sorted(
    data["StartWeek"].dropna().unique().tolist(),
    key=lambda x: int(str(x)[1:])
)


def median_value(column):
    return float(data[column].median())


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 📊 Cancellation Intelligence")
    st.caption("Customer Analytics | ML Prediction")
    st.divider()
    st.markdown("### About this tool")
    st.write(
        "Estimate customer cancellation probability using a trained "
        "machine learning model served through the FastAPI backend."
    )
    st.divider()
    st.markdown("**Frontend status**")
    st.success("Streamlit application loaded")
    st.markdown("**Prediction backend**")
    st.caption("FastAPI deployed on Render")
    st.markdown("**Input source**")
    st.caption("Categories and default values use the training dataset.")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Customer Cancellation Intelligence</h1>
    <p>Customer retention analytics • Machine learning predictions •
    Decision support dashboard</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Enter the customer's details below to estimate cancellation "
    "probability. Predictions are generated by the deployed Render API."
)


# ---------------- INPUT FORM ----------------
with st.form("customer_form"):
    st.markdown("### 01 · Customer & Subscription")

    c1, c2, c3 = st.columns(3)

    with c1:
        territory = st.selectbox("Territory", territories)
        device = st.selectbox("Device Type", device_types)
        plan = st.selectbox("Plan Type", plan_types)

    with c2:
        promotion = st.selectbox("Promotion", promotion_options)
        month = st.selectbox("Start Month", months)
        week = st.selectbox("Start Week", weeks)

    with c3:
        tenure = st.number_input(
            "Customer Tenure",
            min_value=0,
            value=max(0, int(median_value("CustomerTenure"))),
            step=1
        )
        base_fee = st.number_input(
            "Base Fee",
            min_value=0.0,
            value=median_value("BaseFee"),
            step=1.0
        )
        extra_spend = st.number_input(
            "Additional Feature Spend",
            min_value=0.0,
            value=median_value("AdditionalFeatureSpend"),
            step=1.0
        )

    st.markdown("### 02 · Usage & Service Experience")

    a1, a2, a3 = st.columns(3)

    with a1:
        current_usage = st.number_input(
            "Current Period Usage",
            min_value=0.0,
            value=median_value("CurrentPeriodUsage"),
            step=1.0
        )
        prior_usage = st.number_input(
            "Prior Period Usage",
            min_value=0.0,
            value=median_value("PriorPeriodUsage"),
            step=1.0
        )

    with a2:
        penalties = st.number_input(
            "Number of Penalties",
            min_value=0,
            value=max(0, int(median_value("NumberOfPenalties"))),
            step=1
        )
        tickets = st.number_input(
            "Service Tickets",
            min_value=0,
            value=max(0, int(median_value("ServiceTickets"))),
            step=1
        )

    with a3:
        rating = st.number_input(
            "Service Rating",
            min_value=0.0,
            value=median_value("ServiceRating"),
            step=0.5
        )

    submitted = st.form_submit_button(
        "🔍  ANALYZE CUSTOMER CANCELLATION",
        use_container_width=True
    )


# ---------------- PREDICTION THROUGH RENDER API ----------------
if submitted:
    try:
        month_number = int(
            pd.to_datetime(str(month), format="%b").month
        )
        week_number = int(str(week)[1:])

        # Feature engineering for the fields expected by the API
        usage_difference = current_usage - prior_usage

        additional_spend_ratio = (
            extra_spend / base_fee if base_fee != 0 else 0.0
        )

        total_spend = base_fee + extra_spend

        # Prepare API request.
        # Engineered categorical/interaction features are created
        # inside the deployed backend.
        payload = {
            "Territory": str(territory),
            "DeviceType": str(device),
            "Promotion": str(promotion),
            "PlanType": str(plan),
            "BaseFee": float(base_fee),
            "AdditionalFeatureSpend": float(extra_spend),
            "NumberOfPenalties": int(penalties),
            "CurrentPeriodUsage": float(current_usage),
            "PriorPeriodUsage": float(prior_usage),
            "ServiceRating": float(rating),
            "ServiceTickets": int(tickets),
            "CustomerTenure": int(tenure),
            "UsageDifference": float(usage_difference),
            "AdditionalSpendRatio": float(additional_spend_ratio),
            "TotalSpend": float(total_spend),
            "StartMonthNumber": month_number,
            "StartWeekNumber": week_number
        }

        with st.spinner("Connecting to Render backend..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=90
            )

        if response.status_code != 200:
            st.error(f"Backend returned HTTP {response.status_code}")
            st.code(response.text)
            st.stop()

        result = response.json()

        # Read results returned by FastAPI
        prediction_label = result["cancellation_prediction"]
        probability = float(result["cancellation_probability"])
        prediction = 1 if prediction_label == "Yes" else 0

        # ---------------- RESULTS ----------------
        st.markdown("---")
        st.markdown("## Prediction Results")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PREDICTED OUTCOME</div>
                <div class="metric-value">
                    {"Cancellation" if prediction == 1 else "No Cancellation"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CANCELLATION PROBABILITY</div>
                <div class="metric-value">{probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CUSTOMER TENURE</div>
                <div class="metric-value">{tenure} units</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Cancellation Probability")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            "This probability is the model's estimate, not a guarantee "
            "of the customer's future behaviour."
        )

        if prediction == 1:
            st.warning(
                "The model predicts cancellation for this input. "
                "Review the customer's experience and retention options."
            )
        else:
            st.success(
                "The model predicts no cancellation for this input. "
                "Continue normal customer monitoring and support."
            )

        # ---------------- CUSTOMER SUMMARY ----------------
        st.markdown("### Customer Summary")

        summary = pd.DataFrame({
            "Input": [
                "Territory",
                "Device Type",
                "Plan Type",
                "Promotion",
                "Base Fee",
                "Additional Feature Spend",
                "Current Period Usage",
                "Prior Period Usage",
                "Usage Difference",
                "Total Spend",
                "Number of Penalties",
                "Service Tickets",
                "Service Rating",
                "Customer Tenure",
                "Start Month",
                "Start Week"
            ],
            "Value": [
                territory,
                device,
                plan,
                str(promotion),
                f"{base_fee:.2f}",
                f"{extra_spend:.2f}",
                f"{current_usage:.2f}",
                f"{prior_usage:.2f}",
                f"{usage_difference:.2f}",
                f"{total_spend:.2f}",
                penalties,
                tickets,
                rating,
                tenure,
                month,
                week
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # ---------------- DOWNLOAD REPORT ----------------
        report = pd.DataFrame([{
            "Prediction": (
                "Cancellation" if prediction == 1 else "No Cancellation"
            ),
            "Cancellation Probability (%)": round(probability * 100, 2),
            "Territory": territory,
            "DeviceType": device,
            "PlanType": plan,
            "Promotion": promotion,
            "CustomerTenure": tenure,
            "BaseFee": base_fee,
            "AdditionalFeatureSpend": extra_spend
        }])

        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="customer_cancellation_prediction.csv",
            mime="text/csv"
        )

    except requests.exceptions.Timeout:
        st.error(
            "Render backend timed out. The service may be waking up. "
            "Please retry in a little while."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Could not connect to Render backend: {error}")

    except (KeyError, ValueError, TypeError) as error:
        st.error(f"Unexpected backend response or data error: {error}")

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    '<p class="small-note">Customer Cancellation Intelligence · '
    'Machine Learning Assessment Project · Predictions are decision '
    'support estimates, not guarantees.</p>',
    unsafe_allow_html=True
)
