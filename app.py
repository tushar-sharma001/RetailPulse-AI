import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="RetailPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Custom Styling
# ----------------------------------------------------

st.markdown("""
<style>

/* Main app */
.main{
    background-color:#F8FAFC;
}

/* Remove Streamlit padding */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#FFFFFF;
    border-right:1px solid #E5E7EB;
}

/* Headers */
h1{
    color:#0F172A;
    font-weight:700;
}

h2{
    color:#1E293B;
}

h3{
    color:#334155;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:white;
    border:1px solid #E2E8F0;
    padding:18px;
    border-radius:14px;
    box-shadow:0 1px 6px rgba(0,0,0,.05);
}

/* Dataframes */
div[data-testid="stDataFrame"]{
    border-radius:12px;
}

/* Buttons */
.stButton>button{
    border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------

@st.cache_data
def load_data():

    sales = pd.read_csv(
        "data/train.csv",
        parse_dates=[
            "Order Date",
            "Ship Date"
        ],
        dayfirst=True
    )

    forecast = pd.read_csv(
        "outputs/future_forecast.csv"
    )

    forecast["Forecast Month"] = pd.to_datetime(
        forecast["Forecast Month"]
    )

    forecast.rename(
        columns={
            "Forecast Month":"Date",
            "Predicted Sales":"Forecast"
        },
        inplace=True
    )

    segment_forecasts = pd.read_csv(
        "outputs/segment_forecasts.csv",
        parse_dates=["Date"]
    )

    anomalies = pd.read_csv(
        "outputs/weekly_anomalies.csv",
        parse_dates=["Order Date"]
    )

    clusters = pd.read_csv(
        "outputs/product_clusters.csv"
    )

    comparison = pd.read_csv(
        "outputs/model_comparison.csv"
    )

    return (
        sales,
        forecast,
        segment_forecasts,
        anomalies,
        clusters,
        comparison
    )


(
    sales,
    forecast,
    segment_forecasts,
    anomalies,
    clusters,
    comparison
) = load_data()

# ----------------------------------------------------
# Load ML Models
# ----------------------------------------------------

@st.cache_resource
def load_models():

    best_model = joblib.load(
        "models/best_xgboost.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    kmeans = joblib.load(
        "models/kmeans.pkl"
    )

    return best_model, scaler, kmeans


best_model, scaler, kmeans = load_models()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

with st.sidebar:

    st.markdown("# RetailPulse AI")

    st.caption(
        "Sales Forecasting & Demand Intelligence"
    )

    st.divider()

    page = st.radio(

        "Navigation",

        [

            "Home",

            "Executive Dashboard",

            "Forecast Explorer",

            "Anomaly Detection",

            "Demand Segmentation",

            "Model Performance"

        ]

    )

    st.divider()

    st.markdown(
        """
**Dataset**

• 9,994 Orders

• 3 Categories

• 4 Regions

• 2014–2017
"""
    )

    st.divider()

    st.caption("Version 1.0")


# ----------------------------------------------------
# Home
# ----------------------------------------------------

if page == "Home":

    st.title("RetailPulse AI")

    st.markdown(
        """
### AI-Powered Sales Forecasting & Demand Intelligence Platform

RetailPulse AI is an end-to-end analytics solution that helps businesses forecast sales,
identify unusual demand patterns, segment products based on purchasing behaviour,
and support inventory planning using machine learning.
"""
    )

    st.divider()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Orders",
        f"{sales['Order ID'].nunique():,}"
    )

    c2.metric(
        "Customers",
        f"{sales['Customer ID'].nunique():,}"
    )

    c3.metric(
        "Products",
        f"{sales['Product ID'].nunique():,}"
    )

    c4.metric(
        "Total Sales",
        f"${sales['Sales'].sum():,.0f}"
    )

    st.divider()

    left,right = st.columns([2,1])

    with left:

        st.subheader("Platform Capabilities")

        st.markdown("""

- Sales Trend Analysis

- Time Series Forecasting

- Demand Forecasting

- Anomaly Detection

- Product Demand Segmentation

- Executive Business Dashboard

- Interactive Visual Analytics

        """)

    with right:

        st.subheader("Technology Stack")

        st.markdown("""

        Python
        
        Streamlit
        
        Pandas
        
        Plotly
        
        XGBoost
        
        Prophet
        
        SARIMA
        
        Scikit-learn

        """)

# ----------------------------------------------------
# Executive Dashboard
# ----------------------------------------------------

elif page == "Executive Dashboard":

    st.title("Executive Dashboard")

    st.caption(
        "Monitor sales performance, customer activity and demand trends."
    )

    st.divider()


    # -----------------------------
    # Filters
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:

        selected_region = st.selectbox(

            "Region",

            ["All"] +
            sorted(sales["Region"].unique())

        )

    with col2:

        selected_category = st.selectbox(

            "Category",

            ["All"] +
            sorted(sales["Category"].unique())

        )

    filtered_sales = sales.copy()

    if selected_region != "All":

        filtered_sales = filtered_sales[
            filtered_sales["Region"] == selected_region
        ]

    if selected_category != "All":

        filtered_sales = filtered_sales[
            filtered_sales["Category"] == selected_category
        ]


    # -----------------------------
    # KPI Calculations
    # -----------------------------

    total_sales = filtered_sales["Sales"].sum()

    total_orders = filtered_sales["Order ID"].nunique()

    total_customers = filtered_sales["Customer ID"].nunique()

    total_products = filtered_sales["Product ID"].nunique()

    average_order = filtered_sales["Sales"].mean()

    average_shipping = (

        filtered_sales["Ship Date"]

        -

        filtered_sales["Order Date"]

    ).dt.days.mean()

    yearly_sales = (

        filtered_sales

        .groupby(

            filtered_sales["Order Date"].dt.year

        )["Sales"]

        .sum()

    )

    if len(yearly_sales) >= 2:

        growth = (

            yearly_sales.iloc[-1]

            -

            yearly_sales.iloc[-2]

        ) / yearly_sales.iloc[-2] * 100

    else:

        growth = 0


    # -----------------------------
    # KPI Cards
    # -----------------------------

    c1, c2, c3 = st.columns(3)

    c4, c5, c6 = st.columns(3)

    c1.metric(

        "Total Sales",

        f"${total_sales:,.0f}"

    )

    c2.metric(

        "Orders",

        f"{total_orders:,}"

    )

    c3.metric(

        "Customers",

        f"{total_customers:,}"

    )

    c4.metric(

        "Products",

        f"{total_products:,}"

    )

    c5.metric(

        "Average Order Value",

        f"${average_order:.2f}"

    )

    c6.metric(

        "Sales Growth",

        f"{growth:.2f}%"

    )

    st.divider()

    # -----------------------------
    # Monthly Sales Trend
    # -----------------------------

    monthly_sales = (

        filtered_sales

        .groupby(

            pd.Grouper(

                key="Order Date",

                freq="ME"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        monthly_sales,

        x="Order Date",

        y="Sales",

        markers=True,

        title="Monthly Sales Trend"

    )

    fig.update_layout(

        template="plotly_white",

        title_x=0.5,

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    left, right = st.columns(2)

    with left:

        region_sales = (

            filtered_sales

            .groupby("Region")["Sales"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            region_sales,

            x="Region",

            y="Sales",

            color="Region",

            title="Sales by Region"

        )

        fig.update_layout(

            template="plotly_white",

            showlegend=False

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        category_sales = (

            filtered_sales

            .groupby("Category")["Sales"]

            .sum()

            .reset_index()

        )

        fig = px.pie(

            category_sales,

            names="Category",

            values="Sales",

            hole=0.45,

            title="Sales by Category"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.subheader("Top Performing Products")

    top_products = (

        filtered_sales

        .groupby("Product Name")["Sales"]

        .sum()

        .sort_values(

            ascending=False

        )

        .head(10)

        .reset_index()

    )

    fig = px.bar(

        top_products,

        x="Sales",

        y="Product Name",

        orientation="h",

        color="Sales"

    )

    fig.update_layout(

        template="plotly_white",

        yaxis=dict(

            categoryorder="total ascending"

        ),

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.subheader("Executive Insights")

    best_region = (

        filtered_sales

        .groupby("Region")["Sales"]

        .sum()

        .idxmax()

    )

    best_category = (

        filtered_sales

        .groupby("Category")["Sales"]

        .sum()

        .idxmax()

    )

    shipping_days = round(

        average_shipping,

        1

    )

    st.info(

f"""

### Business Summary

**Top Region**

{best_region}

**Top Category**

{best_category}

**Average Shipping Time**

{shipping_days} Days

**Recommendation**

Increase inventory allocation for high-performing categories while maintaining additional stock ahead of seasonal demand periods.

"""

    )


# ----------------------------------------------------
# Forecast Explorer
# ----------------------------------------------------

elif page == "Forecast Explorer":

    st.title("Forecast Explorer")

    st.caption(
        "Analyze future sales forecasts and compare model performance."
    )

    st.divider()

    # -----------------------------
    # Forecast Controls
    # -----------------------------

    left, right = st.columns(2)

    with left:

        forecast_type = st.selectbox(

            "Forecast Level",

            [

                "Overall Sales",

                "Category",

                "Region"

            ]

        )

    with right:

        horizon = st.slider(

            "Forecast Horizon",

            min_value=1,

            max_value=3,

            value=3

        )

    # -----------------------------
    # Overall Forecast
    # -----------------------------

    if forecast_type == "Overall Sales":

        forecast_df = forecast.head(horizon)

        fig = px.line(

            forecast_df,

            x="Date",

            y="Forecast",

            markers=True,

            title="Sales Forecast"

        )

        fig.update_layout(

            template="plotly_white",

            title_x=0.5,

            height=450

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    elif forecast_type == "Category":

        category = st.selectbox(

            "Category",

            sorted(

                sales["Category"].unique()

            )

        )

        forecast_df = (

            segment_forecasts[

                segment_forecasts["Segment"] == category

            ]

            .head(horizon)

        )

        fig = px.line(

            forecast_df,

            x="Date",

            y="Forecast",

            markers=True,

            title=f"{category} Forecast"

        )

        fig.update_layout(

            template="plotly_white",

            title_x=0.5,

            height=450

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    else:

        region = st.selectbox(

            "Region",

            sorted(

                sales["Region"].unique()

            )

        )

        forecast_df = (

            segment_forecasts[

                segment_forecasts["Segment"] == region

            ]

            .head(horizon)

        )

        fig = px.line(

            forecast_df,

            x="Date",

            y="Forecast",

            markers=True,

            title=f"{region} Forecast"

        )

        fig.update_layout(

            template="plotly_white",

            title_x=0.5,

            height=450

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Month 1",

        f"${forecast_df.iloc[0]['Forecast']:,.0f}"

    )

    c2.metric(

        "Month 2",

        f"${forecast_df.iloc[1]['Forecast']:,.0f}"

    )

    c3.metric(

        "Month 3",

        f"${forecast_df.iloc[2]['Forecast']:,.0f}"

    )

    st.subheader("Forecast Results")

    st.dataframe(

        forecast_df,

        use_container_width=True,

        hide_index=True

    )

    st.subheader("Model Performance")

    st.dataframe(

        comparison,

        use_container_width=True,

        hide_index=True

    )

    best_model = comparison.loc[

        comparison["RMSE"].idxmin()

    ]

    st.info(

f"""

### Recommended Model

**{best_model['Model']}**

MAE : {best_model['MAE']:.2f}

RMSE : {best_model['RMSE']:.2f}

MAPE : {best_model['MAPE']:.2f}%

This model achieved the lowest forecasting error and is recommended for production deployment.

"""

    )

    st.download_button(

        label="Download Forecast",

        data=forecast_df.to_csv(index=False),

        file_name="forecast.csv",

        mime="text/csv"

    )

# ----------------------------------------------------
# Anomaly Detection
# ----------------------------------------------------

elif page == "Anomaly Detection":

    st.title("Anomaly Detection")

    st.caption(
        "Identify unusual sales spikes and drops using machine learning."
    )

    st.divider()

    total_anomalies = len(anomalies)

    highest_sale = anomalies["Sales"].max()

    lowest_sale = anomalies["Sales"].min()

    average_sale = anomalies["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Detected Anomalies",
        total_anomalies
    )

    c2.metric(
        "Highest Spike",
        f"${highest_sale:,.0f}"
    )

    c3.metric(
        "Lowest Sale",
        f"${lowest_sale:,.0f}"
    )

    c4.metric(
        "Average Anomaly",
        f"${average_sale:,.0f}"
    )

    st.divider()

    weekly_sales = (

        sales

        .groupby(

            pd.Grouper(

                key="Order Date",

                freq="W"

            )

        )["Sales"]

        .sum()

        .reset_index()

    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=weekly_sales["Order Date"],

            y=weekly_sales["Sales"],

            mode="lines",

            name="Weekly Sales",

            line=dict(width=3)

        )

    )

    fig.add_trace(

        go.Scatter(

            x=anomalies["Order Date"],

            y=anomalies["Sales"],

            mode="markers",

            marker=dict(

                color="#DC2626",

                size=10

            ),

            name="Anomalies"

        )

    )

    fig.update_layout(

        template="plotly_white",

        title="Weekly Sales with Detected Anomalies",

        title_x=0.5,

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.subheader("Detected Anomalies")

    display = anomalies.copy()

    display["Order Date"] = display["Order Date"].dt.strftime(
        "%d %b %Y"
    )

    st.dataframe(

        display,

        use_container_width=True,

        hide_index=True

    )

    st.download_button(

        "Download Anomaly Report",

        anomalies.to_csv(index=False),

        "weekly_anomalies.csv",

        "text/csv"

    )

    st.subheader("Business Interpretation")

    st.info("""

### Key Observations

• Significant sales spikes generally align with promotional or seasonal events.

• Sudden sales drops may indicate inventory shortages or reduced customer demand.

• Monitoring these anomalies helps improve inventory planning and forecasting.

• Weekly anomaly tracking allows early identification of unexpected business events.

""")

    left, right = st.columns(2)

    with left:

        st.subheader("Sales Distribution")

        fig = px.histogram(

            weekly_sales,

            x="Sales",

            nbins=25,

            title="Distribution of Weekly Sales"

        )

        fig.update_layout(

            template="plotly_white"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        st.subheader("Monthly Anomaly Count")

        anomaly_month = anomalies.copy()

        anomaly_month["Month"] = anomaly_month["Order Date"].dt.month_name()

        monthly_count = (

            anomaly_month

            .groupby("Month")

            .size()

            .reset_index(name="Count")

        )

        fig = px.bar(

            monthly_count,

            x="Month",

            y="Count",

            color="Count",

            title="Detected Anomalies by Month"

        )

        fig.update_layout(

            template="plotly_white"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

# ----------------------------------------------------
# Demand Segmentation
# ----------------------------------------------------

elif page == "Demand Segmentation":

    st.title("Demand Segmentation")

    st.caption(
        "Understand product demand patterns and identify inventory strategies."
    )

    st.divider()

    total_subcategories = clusters["Sub-Category"].nunique()

    total_segments = clusters["Demand_Segment"].nunique()

    total_sales = clusters["Total_Sales"].sum()

    avg_order = clusters["Avg_Order_Value"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Sub Categories",
        total_subcategories
    )

    c2.metric(
        "Demand Segments",
        total_segments
    )

    c3.metric(
        "Total Sales",
        f"${total_sales:,.0f}"
    )

    c4.metric(
        "Average Order Value",
        f"${avg_order:.2f}"
    )

    st.divider()

    st.subheader("Product Demand Map")

    fig = px.scatter(

        clusters,

        x="PC1",

        y="PC2",

        color="Demand_Segment",

        size="Total_Sales",

        hover_name="Sub-Category",

        hover_data=[

            "Growth_Rate",

            "Avg_Order_Value"

        ]

    )

    fig.update_layout(

        template="plotly_white",

        title="Demand Segment Distribution",

        title_x=0.5,

        xaxis_title="Cluster Projection",

        yaxis_title="Cluster Projection",

        height=550

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    left, right = st.columns(2)

    with left:

        segment_count = (

            clusters["Demand_Segment"]

            .value_counts()

            .reset_index()

        )

        segment_count.columns = [

            "Segment",

            "Products"

        ]

        fig = px.bar(

            segment_count,

            x="Segment",

            y="Products",

            color="Segment",

            title="Products by Demand Segment"

        )

        fig.update_layout(

            template="plotly_white",

            showlegend=False

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        fig = px.pie(

            segment_count,

            names="Segment",

            values="Products",

            hole=0.45,

            title="Demand Segment Share"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.subheader("Product Classification")

    display = clusters.copy()

    display = display.sort_values(

        "Demand_Segment"

    )

    st.dataframe(

        display,

        use_container_width=True,

        hide_index=True

    )

    st.subheader("Inventory Recommendations")

    st.markdown("""

| Demand Segment | Recommended Strategy |
|----------------|----------------------|
| High Volume Stable | Maintain high inventory and prioritize replenishment. |
| Growing Demand | Increase stock gradually to support future demand. |
| Seasonal Products | Plan inventory before peak demand periods. |
| Low Demand | Maintain lean inventory and review pricing strategies. |

""")

    st.download_button(

        "Download Segmentation Report",

        clusters.to_csv(index=False),

        file_name="product_segments.csv",

        mime="text/csv"

    )

    st.info("""

### Executive Summary

Demand segmentation helps identify which products require aggressive stocking,
which products should be monitored closely, and which products may require
inventory optimization.

Using clustering allows planners to allocate warehouse space and procurement
budgets more efficiently.

""")

# ----------------------------------------------------
# Model Performance
# ----------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance")

    st.caption(
        "Compare forecasting models and identify the best model for deployment."
    )

    st.divider()

    best_model = comparison.loc[
        comparison["RMSE"].idxmin()
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Selected Model",
        best_model["Model"]
    )

    c2.metric(
        "MAE",
        f"{best_model['MAE']:.2f}"
    )

    c3.metric(
        "RMSE",
        f"{best_model['RMSE']:.2f}"
    )

    c4.metric(
        "MAPE",
        f"{best_model['MAPE']:.2f}%"
    )

    st.divider()

    st.subheader("Forecast Accuracy Comparison")

    ranking = comparison.sort_values(
        "RMSE"
    )

    st.dataframe(

        ranking,

        use_container_width=True,

        hide_index=True

    )

    left, right = st.columns(2)

    with left:

        fig = px.bar(

            ranking,

            x="Model",

            y="RMSE",

            color="Model",

            title="RMSE Comparison"

        )

        fig.update_layout(

            template="plotly_white",

            showlegend=False

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        fig = px.bar(

            ranking,

            x="Model",

            y="MAE",

            color="Model",

            title="MAE Comparison"

        )

        fig.update_layout(

            template="plotly_white",

            showlegend=False

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    fig = px.line(

        ranking,

        x="Model",

        y="MAPE",

        markers=True,

        title="MAPE Comparison"

    )

    fig.update_layout(

        template="plotly_white",

        title_x=0.5,

        height=420

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.subheader("Model Ranking")

    leaderboard = ranking.copy()

    leaderboard.insert(

        0,

        "Rank",

        range(

            1,

            len(leaderboard)+1

        )

    )

    st.dataframe(

        leaderboard,

        use_container_width=True,

        hide_index=True

    )

    st.subheader("Deployment Recommendation")

    st.success(

f"""

### Recommended Production Model

**{best_model['Model']}**

This model achieved the lowest prediction error across the evaluation metrics.

It provides the best balance between:

- Forecast Accuracy
- Model Stability
- Generalization
- Business Reliability

This makes it the preferred model for retail demand forecasting
and inventory planning.

"""

    )

    with st.expander("Model Overview"):

        st.markdown("""

### SARIMA

A statistical forecasting model that captures trend and seasonality.
Best suited for stable historical time series.

---

### Prophet

A forecasting framework developed for business time series with
strong seasonal behaviour and holiday effects.

---

### XGBoost

A machine learning model capable of learning complex nonlinear
relationships using engineered time-series features.

""")

    st.download_button(

        "Download Model Evaluation",

        ranking.to_csv(index=False),

        file_name="model_performance.csv",

        mime="text/csv"

    )

st.divider()

st.caption(
    "RetailPulse AI • AI-Powered Sales Forecasting & Demand Intelligence • Built with Python, Streamlit, XGBoost, Prophet, SARIMA and Plotly"
)




