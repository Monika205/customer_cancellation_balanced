#Step 39

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "customer_churn_model.joblib"
DATA_PATH = "Customer_Cancellations_Training.csv"

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


# ---------------- LOAD MODEL AND TRAINING DATA ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Run the model-saving notebook cell first."
    )
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(
        f"Training dataset not found: {DATA_PATH}. "
        "Place the CSV in the app's working directory."
    )
    st.stop()

try:
    model = load_model()
    data = load_data()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()


# ---------------- INPUT OPTIONS ----------------
required_data_columns = [
    "Territory", "DeviceType", "PlanType", "StartMonth", "StartWeek",
    "CustomerTenure", "BaseFee", "AdditionalFeatureSpend",
    "CurrentPeriodUsage", "PriorPeriodUsage", "NumberOfPenalties",
    "ServiceTickets", "ServiceRating"
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
    st.markdown("## 📊 Churn Intelligence")
    st.caption("Customer Analytics | ML Prediction")
    st.divider()
    st.markdown("### About this tool")
    st.write(
        "Estimate customer churn using the trained machine learning "
        "pipeline and customer service, usage and plan information."
    )
    st.divider()
    st.markdown("**Model status**")
    st.success("Saved model loaded")
    st.markdown("**Input source**")
    st.caption("Categories are taken from the training dataset.")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Customer Churn Intelligence</h1>
    <p>Customer retention analytics • Machine learning predictions •
    Decision support dashboard</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Enter the customer's details below to estimate churn probability. "
    "Fields and categories follow the project training data."
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
        promotion_options = sorted(
            data["Promotion"].dropna().unique().tolist()
        ) if "Promotion" in data.columns else ["no", "yes"]

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
        "🔍  ANALYZE CUSTOMER CHURN",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:
    try:
        month_number = int(
            pd.to_datetime(str(month), format="%b").month
        )
        week_number = int(str(week)[1:])

        # Step 20: Feature engineering
        usage_difference = current_usage - prior_usage

        additional_spend_ratio = (
            extra_spend / base_fee if base_fee != 0 else np.nan
        )

        total_spend = base_fee + extra_spend

        # Step 21C: Binning features
        penalty_level = (
            "Low (0-3)" if penalties <= 3 else "High (4+)"
        )

        if 0 < base_fee <= 20:
            fee_band = "<=20"
        elif 20 < base_fee <= 30:
            fee_band = "20-30"
        elif 30 < base_fee <= 38:
            fee_band = "30-38"
        elif 38 < base_fee <= 100:
            fee_band = ">38"
        else:
            fee_band = np.nan

        if tenure <= 60:
            tenure_group = "New"
        elif tenure <= 90:
            tenure_group = "Mid"
        elif tenure <= 120:
            tenure_group = "Established"
        else:
            tenure_group = "Long-term"

        # Step 21C: Interaction features
        red_plan_x_penalties = (
            int(plan == "Red Plan") * penalties
        )
        penalties_per_tenure = penalties / (tenure + 1)
        tickets_per_tenure = tickets / (tenure + 1)

        # Create all 23 model input features
        customer = pd.DataFrame([{
            "Territory": territory,
            "DeviceType": device,
            "Promotion": promotion,
            "PlanType": plan,
            "BaseFee": base_fee,
            "AdditionalFeatureSpend": extra_spend,
            "NumberOfPenalties": penalties,
            "CurrentPeriodUsage": current_usage,
            "PriorPeriodUsage": prior_usage,
            "ServiceRating": rating,
            "ServiceTickets": tickets,
            "CustomerTenure": tenure,
            "UsageDifference": usage_difference,
            "AdditionalSpendRatio": additional_spend_ratio,
            "TotalSpend": total_spend,
            "StartMonthNumber": month_number,
            "StartWeekNumber": week_number,
            "PenaltyLevel": penalty_level,
            "FeeBand": fee_band,
            "TenureGroup": tenure_group,
            "RedPlan_x_Penalties": red_plan_x_penalties,
            "Penalties_per_Tenure": penalties_per_tenure,
            "Tickets_per_Tenure": tickets_per_tenure
        }])

        # Match the saved model's expected feature order
        expected_features = [
            "Territory",
            "DeviceType",
            "Promotion",
            "PlanType",
            "BaseFee",
            "AdditionalFeatureSpend",
            "NumberOfPenalties",
            "CurrentPeriodUsage",
            "PriorPeriodUsage",
            "ServiceRating",
            "ServiceTickets",
            "CustomerTenure",
            "UsageDifference",
            "AdditionalSpendRatio",
            "TotalSpend",
            "StartMonthNumber",
            "StartWeekNumber",
            "PenaltyLevel",
            "FeeBand",
            "TenureGroup",
            "RedPlan_x_Penalties",
            "Penalties_per_Tenure",
            "Tickets_per_Tenure"
        ]

        customer = customer[expected_features]

        with st.spinner("Analyzing customer data..."):
            prediction = int(model.predict(customer)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(customer)[0]
                classes = list(model.classes_)

                if 1 not in classes:
                    raise ValueError(
                        "The saved model does not contain class 1 "
                        "for churn prediction."
                    )

                positive_class_index = classes.index(1)
                probability = float(
                    probabilities[positive_class_index]
                )
            else:
                raise ValueError(
                    "The saved model does not support predict_proba()."
                )

        # ---------------- RESULTS ----------------
        st.markdown("---")
        st.markdown("## Prediction Results")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PREDICTED OUTCOME</div>
                <div class="metric-value">
                    {"Churn" if prediction == 1 else "No Churn"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MODEL CHURN PROBABILITY</div>
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

        st.markdown("### Churn Probability")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            "Probability is the model's estimated likelihood of churn; "
            "it is not a guarantee of the customer's future behaviour."
        )

        if prediction == 1:
            st.warning(
                "The model predicts churn for this input. "
                "Consider reviewing the customer's experience and "
                "retention options before taking action."
            )
        else:
            st.success(
                "The model predicts no churn for this input. "
                "Continue normal customer monitoring and service support."
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
                "Start Week",
                "Penalty Level",
                "Fee Band",
                "Tenure Group"
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
                week,
                penalty_level,
                str(fee_band),
                tenure_group
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # ---------------- DOWNLOAD REPORT ----------------
        result_text = "Churn" if prediction == 1 else "No Churn"

        report = pd.DataFrame([{
            "Prediction": result_text,
            "Churn Probability (%)": round(probability * 100, 2),
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
            file_name="customer_churn_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
        st.info(
            "Check that the saved model was trained using the same "
            "23 features, feature names, categories and preprocessing."
        )


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    '<p class="small-note">Customer Churn Intelligence · '
    'Machine Learning Assessment Project · Predictions are decision '
    'support estimates, not guarantees.</p>',
    unsafe_allow_html=True
)
