import streamlit as st
import pandas as pd
import numpy as np
import joblib


# PAGE CONFIG (MUST BE FIRST)

st.set_page_config(
    page_title="Cart Conversion AI",
    page_icon="🛒",
    layout="wide"
)


# LOAD MODEL
model = joblib.load("rf_model.pkl")
features = joblib.load("features.pkl")


# HEADER

st.markdown("""
    <h1 style='text-align:center; color:pink;'>🛒 Cart Conversion Prediction</h1>
    <p style='text-align:center; color:gray;'>Predict customer purchase behavior using Machine Learning</p>
""", unsafe_allow_html=True)

st.divider()

# MODEL COMPARISON

col1, col2, col3, col4 = st.columns(4)

col1.metric("Random Forest + stratify", "72%")
col2.metric("Decision Tree", "65%")
col3.metric("XGBoost", "79%")
col4.metric("Best Recall", "82%")

st.subheader("🏆 Model Performance Comparison")

results_df = pd.DataFrame({
    "Model": [
        "Random Forest + stratify",
        "Random Forest + SMOTE",
        "XGBoost",
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        0.72,
        0.71,
        0.79,
        0.70,
        0.65
    ],
    "Precision": [
        0.36,
        0.35,
        0.39,
        0.33,
        0.32
    ],
    "Recall": [
        0.82,
        0.79,
        0.42,
        0.77,
        0.96
    ],
    "F1 Score": [
        0.50,
        0.48,
        0.41,
        0.46,
        0.49
    ]
})

st.dataframe(results_df, use_container_width=True)
st.info("👉 Random Forest + stratify selected due to best balanced Precision + Recall")


#st.divider()

# SIDEBAR INPUT

st.sidebar.header("👤 Customer Profile")

device = st.sidebar.selectbox("Device Type", ["Android", "iOS", "Desktop"])
traffic = st.sidebar.selectbox("Traffic Source", ["Organic", "Paid Search", "Referral", "Social"])
loyalty = st.sidebar.selectbox("Loyalty Tier", ["Bronze", "Silver", "Gold", "Platinum"])

pages = st.sidebar.slider("Pages Viewed", 1, 25, 7)
session = st.sidebar.slider("Session Duration", 0, 60, 10)
cart = st.sidebar.selectbox("Added To Cart", [0, 1])

tenure = st.sidebar.slider("Tenure Months", 0, 120, 24)
recency = st.sidebar.slider("Recency Days", 0, 200, 30)
orders = st.sidebar.slider("Total Orders", 0, 100, 5)
avg_order = st.sidebar.number_input("Avg Order Value", 0.0, 100000.0, 5000.0)


# INPUT DATA FRAME

input_dict = {
    "Pages_Viewed": pages,
    "Session_Duration": session,
    "Added_To_Cart": cart,
    "Tenure_Months": tenure,
    "Recency_Days": recency,
    "Total_Orders": orders,
    "Avg_Order_Value": avg_order
}

input_df = pd.DataFrame([input_dict])

# fill missing features
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0


# ENCODING

if device == "Desktop":
    input_df["Device_Type_Desktop"] = 1
elif device == "iOS":
    input_df["Device_Type_iOS"] = 1

if traffic == "Organic":
    input_df["Traffic_Source_Organic"] = 1
elif traffic == "Paid Search":
    input_df["Traffic_Source_Paid Search"] = 1
elif traffic == "Referral":
    input_df["Traffic_Source_Referral"] = 1
elif traffic == "Social":
    input_df["Traffic_Source_Social"] = 1

if loyalty == "Gold":
    input_df["Loyalty_Tier_Gold"] = 1
elif loyalty == "Platinum":
    input_df["Loyalty_Tier_Platinum"] = 1
elif loyalty == "Silver":
    input_df["Loyalty_Tier_Silver"] = 1

input_df = input_df.reindex(columns=features, fill_value=0)

# PREDICTION

proba = model.predict_proba(input_df)[:, 1][0]

threshold = 0.50
pred = 1 if proba >= threshold else 0

# OUTPUT (CLEAN STRUCTURE)

st.subheader("🎯 Prediction Result")

st.write(f"📊 Purchase Probability: **{round(proba, 3)}**")

if pred == 1:
    st.success("Likely to Purchase")
else:
    st.error("Likely to Abandon Cart")


# BUSINESS INSIGHT (CLEAN SINGLE LOGIC)

st.subheader("Business Insight")

if proba >= 0.70:
    st.success("🔥 High Intent → Offer discount / urgency push")
    st.write("🔥 High intent → Give discount / retarget ads")

elif proba >= 0.55:
    st.success("🎯 Medium Intent → Retarget ads + reminders")
    st.write("⚡ Medium intent → Send reminders / email campaign")

elif proba >= 0.40:
    st.warning("⚡ Warm Lead → Email + ads")
    st.write("⚡ Medium intent → Send reminders / email campaign")

else:
    st.error("❌ Low Intent → No marketing spend")
    st.write("📉 Low intent → No marketing spend recommended")

# FOOTER

st.markdown("---")
st.markdown("<p style='text-align:center; color:pink;'>Built with ❤️ ML + Streamlit</p>", unsafe_allow_html=True)