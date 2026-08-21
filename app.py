import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    churn_model = joblib.load(
        "models/churn_model.pkl"
    )

    churn_threshold = joblib.load(
        "models/churn_threshold.pkl"
    )

    kmeans_model = joblib.load(
        "models/kmeans_model.pkl"
    )

    cluster_scaler = joblib.load(
        "models/cluster_scaler.pkl"
    )

    cluster_features = joblib.load(
        "models/cluster_features.pkl"
    )

    return (
        churn_model,
        churn_threshold,
        kmeans_model,
        cluster_scaler,
        cluster_features
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )


# ============================================================
# LOAD CLUSTER PROFILE
# ============================================================

@st.cache_data
def load_cluster_profile():

    profile = pd.read_csv(
        "models/cluster_profile.csv"
    )

    if "Cluster" in profile.columns:

        profile["Cluster"] = (
            profile["Cluster"].astype(int)
        )

    else:

        profile = profile.rename(
            columns={
                profile.columns[0]: "Cluster"
            }
        )

        profile["Cluster"] = (
            profile["Cluster"].astype(int)
        )

    return profile


# ============================================================
# INITIALIZE
# ============================================================

(
    churn_model,
    churn_threshold,
    kmeans_model,
    cluster_scaler,
    cluster_features
) = load_models()

df = load_data()

cluster_profile = load_cluster_profile()


# ============================================================
# GLOBAL CLUSTER STATISTICS
# ============================================================

# These MUST be calculated when the app loads because they
# are also required for displaying the cluster table.

median_tenure = cluster_profile[
    "tenure"
].median()

median_charges = cluster_profile[
    "MonthlyCharges"
].median()


# ============================================================
# FUNCTION — BUSINESS SEGMENT NAME
# ============================================================

def get_segment_name(row):

    tenure = row["tenure"]

    charges = row["MonthlyCharges"]

    if (
        tenure >= median_tenure
        and charges >= median_charges
    ):

        return "High-Value Loyal"

    elif (
        tenure >= median_tenure
        and charges < median_charges
    ):

        return "Loyal Low-Cost"

    elif (
        tenure < median_tenure
        and charges >= median_charges
    ):

        return "High-Cost New"

    else:

        return "Low-Value / New"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Customer Churn Intelligence"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🔮 Churn Prediction",
        "👥 Customer Segmentation"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.title(
        "📊 Customer Churn Intelligence"
    )

    st.write(
        "An end-to-end machine learning system for "
        "customer churn prediction and customer segmentation."
    )

    st.divider()

    # --------------------------------------------------------
    # KEY METRICS
    # --------------------------------------------------------

    total_customers = len(df)

    churned_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100

    avg_monthly_charges = (
        df["MonthlyCharges"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.1f}%"
    )

    col4.metric(
        "Avg Monthly Charges",
        f"${avg_monthly_charges:.2f}"
    )

    st.divider()

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Churn Distribution"
    )

    churn_counts = (
        df["Churn"].value_counts()
    )

    st.bar_chart(
        churn_counts
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.title(
        "🔮 Customer Churn Prediction"
    )

    st.write(
        "Enter customer information to estimate "
        "the probability of churn."
    )

    st.divider()

    customer = {}

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Customer Information"
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        customer["gender"] = st.selectbox(
            "Gender",
            df["gender"].dropna().unique()
        )

        customer["SeniorCitizen"] = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        customer["Partner"] = st.selectbox(
            "Partner",
            df["Partner"].dropna().unique()
        )

        customer["Dependents"] = st.selectbox(
            "Dependents",
            df["Dependents"].dropna().unique()
        )

    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        customer["tenure"] = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        # Automatically create TenureGroup

        if customer["tenure"] <= 12:

            customer["TenureGroup"] = "0-12"

        elif customer["tenure"] <= 24:

            customer["TenureGroup"] = "13-24"

        elif customer["tenure"] <= 36:

            customer["TenureGroup"] = "25-36"

        elif customer["tenure"] <= 48:

            customer["TenureGroup"] = "37-48"

        elif customer["tenure"] <= 60:

            customer["TenureGroup"] = "49-60"

        else:

            customer["TenureGroup"] = "61-72"

        st.caption(
            f"Tenure Group: "
            f"{customer['TenureGroup']}"
        )

        customer["MonthlyCharges"] = (
            st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )
        )

        customer["TotalCharges"] = (
            st.number_input(
                "Total Charges",
                min_value=0.0,
                value=800.0
            )
        )

    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        customer["NumberOfServices"] = (
            st.number_input(
                "Number of Services",
                min_value=0,
                max_value=10,
                value=3
            )
        )

        customer["HasInternet"] = (
            st.selectbox(
                "Has Internet",
                [0, 1]
            )
        )

        customer["AvgMonthlySpend"] = (
            st.number_input(
                "Average Monthly Spend",
                min_value=0.0,
                value=70.0
            )
        )

    st.divider()

    # --------------------------------------------------------
    # SERVICES & CONTRACT
    # --------------------------------------------------------

    st.subheader(
        "Services & Contract"
    )

    categorical_features = [

        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"

    ]

    cols = st.columns(3)

    for i, feature in enumerate(
        categorical_features
    ):

        with cols[i % 3]:

            customer[feature] = (
                st.selectbox(
                    feature,
                    df[feature]
                    .dropna()
                    .unique()
                )
            )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    ):

        customer_df = pd.DataFrame(
            [customer]
        )

        probability = (
            churn_model
            .predict_proba(
                customer_df
            )[0, 1]
        )

        prediction = int(
            probability >=
            churn_threshold
        )

        st.subheader(
            "Prediction Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Churn Probability",
                f"{probability:.1%}"
            )

        with col2:

            st.metric(
                "Decision Threshold",
                f"{churn_threshold:.2f}"
            )

        if prediction == 1:

            st.error(
                "⚠️ High Risk — Customer is likely to churn."
            )

        else:

            st.success(
                "✅ Low Risk — Customer is unlikely to churn."
            )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "👥 Customer Segmentation":

    st.title(
        "👥 Customer Segmentation"
    )

    st.write(
        "Use K-Means clustering to identify customer segments "
        "based on behavioral and service characteristics."
    )

    st.divider()

    # --------------------------------------------------------
    # CUSTOMER INPUT
    # --------------------------------------------------------

    st.subheader(
        "Customer Characteristics"
    )

    cluster_input = {}

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    with col1:

        cluster_input["SeniorCitizen"] = (
            st.selectbox(
                "Senior Citizen",
                [0, 1]
            )
        )

        cluster_input["tenure"] = (
            st.number_input(
                "Tenure (months)",
                min_value=0.0,
                max_value=72.0,
                value=24.0
            )
        )

        cluster_input["MonthlyCharges"] = (
            st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )
        )

        cluster_input["TotalCharges"] = (
            st.number_input(
                "Total Charges",
                min_value=0.0,
                value=1600.0
            )
        )

    # --------------------------------------------------------
    # SERVICE INFORMATION
    # --------------------------------------------------------

    with col2:

        cluster_input["NumberOfServices"] = (
            st.number_input(
                "Number of Services",
                min_value=0.0,
                max_value=10.0,
                value=3.0
            )
        )

        cluster_input["HasInternet"] = (
            st.selectbox(
                "Has Internet",
                [0, 1]
            )
        )

        cluster_input["HasTechSupport"] = (
            st.selectbox(
                "Has Tech Support",
                [0, 1]
            )
        )

        cluster_input["HasOnlineSecurity"] = (
            st.selectbox(
                "Has Online Security",
                [0, 1]
            )
        )

    st.divider()

    # --------------------------------------------------------
    # IDENTIFY CUSTOMER SEGMENT
    # --------------------------------------------------------

    if st.button(
        "Identify Customer Segment",
        type="primary",
        use_container_width=True
    ):

        cluster_df = pd.DataFrame(
            [cluster_input]
        )

        # Exact feature order used during training

        cluster_df = cluster_df[
            [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "NumberOfServices",
                "HasInternet",
                "HasTechSupport",
                "HasOnlineSecurity"
            ]
        ]

        # Scale input

        scaled_input = (
            cluster_scaler.transform(
                cluster_df
            )
        )

        # Predict cluster

        cluster = int(
            kmeans_model.predict(
                scaled_input
            )[0]
        )

        # ----------------------------------------------------
        # FIND CLUSTER PROFILE
        # ----------------------------------------------------

        profile_row = (
            cluster_profile[
                cluster_profile["Cluster"]
                == cluster
            ]
        )

        cluster_tenure = float(
            profile_row[
                "tenure"
            ].iloc[0]
        )

        cluster_charges = float(
            profile_row[
                "MonthlyCharges"
            ].iloc[0]
        )

        avg_total = float(
            profile_row[
                "TotalCharges"
            ].iloc[0]
        )

        avg_services = float(
            profile_row[
                "NumberOfServices"
            ].iloc[0]
        )

        # ----------------------------------------------------
        # BUSINESS SEGMENT
        # ----------------------------------------------------

        if (
            cluster_tenure >= median_tenure
            and
            cluster_charges >= median_charges
        ):

            segment_name = (
                "High-Value Loyal Customers"
            )

            segment_icon = "💎"

        elif (
            cluster_tenure >= median_tenure
            and
            cluster_charges < median_charges
        ):

            segment_name = (
                "Loyal Low-Cost Customers"
            )

            segment_icon = "🟢"

        elif (
            cluster_tenure < median_tenure
            and
            cluster_charges >= median_charges
        ):

            segment_name = (
                "High-Cost New Customers"
            )

            segment_icon = "🟠"

        else:

            segment_name = (
                "Low-Value / New Customers"
            )

            segment_icon = "🔵"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Segmentation Result"
        )

        st.success(
            f"{segment_icon} Cluster {cluster} — "
            f"**{segment_name}**"
        )

        # ----------------------------------------------------
        # CLUSTER METRICS
        # ----------------------------------------------------

        st.subheader(
            "Cluster Profile"
        )

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        with metric1:

            st.metric(
                "Average Tenure",
                f"{cluster_tenure:.1f} months"
            )

        with metric2:

            st.metric(
                "Monthly Charges",
                f"${cluster_charges:.2f}"
            )

        with metric3:

            st.metric(
                "Average Total Charges",
                f"${avg_total:.2f}"
            )

        with metric4:

            st.metric(
                "Average Services",
                f"{avg_services:.2f}"
            )

        # ----------------------------------------------------
        # SERVICE CHARACTERISTICS
        # ----------------------------------------------------

        st.subheader(
            "Service Characteristics"
        )

        service1, service2, service3 = (
            st.columns(3)
        )

        with service1:

            internet = float(
                profile_row[
                    "HasInternet"
                ].iloc[0]
            )

            st.metric(
                "Internet Adoption",
                f"{internet:.1%}"
            )

        with service2:

            security = float(
                profile_row[
                    "HasOnlineSecurity"
                ].iloc[0]
            )

            st.metric(
                "Online Security",
                f"{security:.1%}"
            )

        with service3:

            support = float(
                profile_row[
                    "HasTechSupport"
                ].iloc[0]
            )

            st.metric(
                "Tech Support",
                f"{support:.1%}"
            )

        # ----------------------------------------------------
        # BUSINESS INTERPRETATION
        # ----------------------------------------------------

        st.subheader(
            "💡 Business Interpretation"
        )

        if (
            segment_name
            == "High-Value Loyal Customers"
        ):

            st.write(
                "These customers have relatively high tenure "
                "and higher monthly spending. They represent "
                "an important high-value customer base and "
                "should be prioritized for retention and "
                "loyalty programs."
            )

        elif (
            segment_name
            == "Loyal Low-Cost Customers"
        ):

            st.write(
                "These customers have relatively high tenure "
                "but lower monthly spending. They appear "
                "stable and could be targeted with relevant "
                "upselling or cross-selling opportunities."
            )

        elif (
            segment_name
            == "High-Cost New Customers"
        ):

            st.write(
                "These customers have relatively low tenure "
                "but higher monthly spending. They may "
                "represent valuable new customers who should "
                "receive strong onboarding and engagement."
            )

        else:

            st.write(
                "These customers have relatively lower tenure "
                "and lower monthly spending. They may "
                "represent newer or lower-value customers "
                "and could benefit from targeted engagement "
                "campaigns."
            )

        # ----------------------------------------------------
        # CUSTOMER VS CLUSTER
        # ----------------------------------------------------

        st.subheader(
            "Customer vs Cluster"
        )

        comparison_data = {

            "Feature": [
                "Tenure",
                "Monthly Charges",
                "Total Charges",
                "Number of Services"
            ],

            "Customer": [
                cluster_input["tenure"],
                cluster_input["MonthlyCharges"],
                cluster_input["TotalCharges"],
                cluster_input["NumberOfServices"]
            ],

            "Cluster Average": [
                cluster_tenure,
                cluster_charges,
                avg_total,
                avg_services
            ]
        }

        comparison_df = pd.DataFrame(
            comparison_data
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # ALL CUSTOMER SEGMENTS
    # ========================================================

    st.divider()

    st.subheader(
        "📋 All Customer Segments"
    )

    display_profile = (
        cluster_profile.copy()
    )

    # Add business segment name
    # This now works immediately when the page loads.

    display_profile["Segment"] = (
        display_profile.apply(
            get_segment_name,
            axis=1
        )
    )

    st.dataframe(
        display_profile,
        use_container_width=True,
        hide_index=True
    )