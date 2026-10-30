
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================================
# 1. PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "customer_churn_cleaned.csv"
MODEL_PATH = ROOT / "models" / "tuned_random_forest.joblib"
OUTPUT_PATH = ROOT / "outputs"

# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# 3. LOAD DATASET
# =====================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    return pd.read_csv(DATA_PATH)


df = load_data()

# =====================================================
# 4. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Saved model not found: {MODEL_PATH}")
        st.stop()

    return joblib.load(MODEL_PATH)


saved_model = load_model()

model = saved_model["model"]
feature_columns = saved_model["feature_columns"]

# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🏦 Bank Churn System")
st.sidebar.caption("Customer Churn Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Churn Prediction",
        "Model Evaluation",
        "Business Insights"
    ]
)

# =====================================================
# 6. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🏦 Bank Customer Churn Prediction")
    st.write("Data Mining and Business Intelligence – CA2")

    st.markdown("---")

    total = len(df)
    churned = int(df["Exited"].sum())
    stayed = total - churned
    churn_rate = (churned / total) * 100 if total else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Customers Retained", f"{stayed:,}")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Churn Distribution")

        churn_data = pd.DataFrame({
            "Customer Status": ["Stayed", "Churned"],
            "Customers": [stayed, churned]
        })

        fig = px.pie(
            churn_data,
            names="Customer Status",
            values="Customers",
            hole=0.45,
            title="Customer Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Dataset Overview")

        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")

        st.write("**Target Variable:** Exited")

        st.write("0 = Stayed")
        st.write("1 = Churned")

# =====================================================
# 7. CHURN PREDICTION
# =====================================================

elif page == "Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer details to predict whether the customer "
        "is likely to leave the bank."
    )

    st.info("Prediction is generated using the saved Random Forest model.")

    with st.form("prediction_form"):

        st.subheader("Customer Information")

        col1, col2 = st.columns(2)

        with col1:

            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=650
            )

            geography = st.selectbox(
                "Geography",
                ["France", "Germany", "Spain"]
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )

            tenure = st.number_input(
                "Tenure (Years)",
                min_value=0,
                max_value=10,
                value=5
            )

        with col2:

            salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                max_value=1000000.0,
                value=50000.0
            )

            balance = st.number_input(
                "Balance",
                min_value=0.0,
                max_value=300000.0,
                value=50000.0
            )

            products = st.selectbox(
                "Number of Products",
                [1, 2, 3, 4]
            )

            card = st.selectbox(
                "Has Credit Card?",
                ["Yes", "No"]
            )

            active = st.selectbox(
                "Is Active Member?",
                ["Yes", "No"]
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True
        )

    if submitted:

        # Create customer input
        customer = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "EstimatedSalary": salary,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": 1 if card == "Yes" else 0,
            "IsActiveMember": 1 if active == "Yes" else 0
        }])

        # Apply the same encoding used during model training
        customer = pd.get_dummies(
            customer,
            columns=["Geography", "Gender"],
            drop_first=True,
            dtype=int
        )

        # Match the exact training feature columns
        customer = customer.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Generate prediction
        prediction = model.predict(customer)[0]

        probability = model.predict_proba(customer)[0][1] * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.error("⚠️ Customer is predicted to CHURN.")

            else:

                st.success("✅ Customer is predicted to STAY.")

        with result_col2:

            st.metric(
                "Predicted Churn Probability",
                f"{probability:.2f}%"
            )

        st.progress(int(round(probability)))

        st.caption(
            "Probability is the model's estimated likelihood of churn, "
            "not a guarantee of the customer's future behaviour."
        )

# =====================================================
# 8. MODEL EVALUATION
# =====================================================

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation")

    st.write("Performance comparison of the machine learning models.")

    metrics_path = OUTPUT_PATH / "ca2_evaluation_metrics.csv"

    if metrics_path.exists():

        metrics_df = pd.read_csv(metrics_path)

        st.subheader("Model Performance Metrics")

        st.dataframe(
            metrics_df,
            use_container_width=True
        )

    else:

        st.warning("Evaluation metrics CSV was not found.")

    st.markdown("---")

    st.subheader("Existing Model Evaluation Charts")

    chart_files = [
        ("Decision Tree Confusion Matrix",
         "decision_tree_confusion_matrix.png"),

        ("Random Forest Confusion Matrix",
         "ca2_random_forest_confusion_matrix.png"),

        ("Random Forest Feature Importance",
         "ca2_random_forest_feature_importance.png"),

        ("Model Comparison",
         "model_comparison.png")
    ]

    for title, filename in chart_files:

        chart_path = OUTPUT_PATH / filename

        if chart_path.exists():

            st.subheader(title)

            st.image(
                str(chart_path),
                use_container_width=True
            )

# =====================================================
# 9. BUSINESS INSIGHTS
# =====================================================

elif page == "Business Insights":

    st.title("💡 Business Insights")

    st.write(
        "Explore customer characteristics and patterns associated "
        "with bank customer churn."
    )

    st.subheader("Churn by Geography")

    geography_churn = (
        df.groupby("Geography")["Exited"]
        .mean()
        .mul(100)
        .reset_index(name="Churn Rate")
    )

    fig1 = px.bar(
        geography_churn,
        x="Geography",
        y="Churn Rate",
        title="Churn Rate by Geography",
        text_auto=".2f"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Churn by Gender")

    gender_churn = (
        df.groupby("Gender")["Exited"]
        .mean()
        .mul(100)
        .reset_index(name="Churn Rate")
    )

    fig2 = px.bar(
        gender_churn,
        x="Gender",
        y="Churn Rate",
        title="Churn Rate by Gender",
        text_auto=".2f"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Churn by Age")

    fig3 = px.histogram(
        df,
        x="Age",
        color="Exited",
        barmode="overlay",
        title="Customer Age Distribution by Churn Status",
        labels={
            "Exited": "Churn Status",
            "Age": "Customer Age"
        }
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Feature Importance")

    importance_path = OUTPUT_PATH / "ca2_feature_importance.csv"

    if importance_path.exists():

        importance_df = pd.read_csv(importance_path)

        st.dataframe(
            importance_df,
            use_container_width=True
        )

    else:

        st.warning("Feature importance CSV was not found.")

    st.caption(
        "These patterns show associations in the dataset. "
        "They do not establish that a customer characteristic causes churn."
    )

# =====================================================
# END OF APPLICATION
# =====================================================