import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="RetailPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():

    sales = pd.read_csv(
        "data/train.csv",
        parse_dates=["Order Date", "Ship Date"],
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
        "Forecast Month": "Date",
        "Predicted Sales": "Forecast"
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


sales, forecast, segment_forecasts, anomalies, clusters, comparison = load_data()

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

with st.sidebar:

    st.title("📈 RetailPulse AI")

    st.caption(
        "Sales Forecasting & Demand Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📈 Forecast Explorer",
            "🚨 Anomaly Report",
            "📦 Demand Segments",
            "🤖 Model Performance"
        ]
    )

    st.divider()

    st.caption("Version 1.0")
    # ==========================================================
# Dashboard
# ==========================================================

if page == "🏠 Dashboard":

    # ---------------- Hero Section ---------------- #

    st.title("📈 RetailPulse AI")

    st.markdown(
        """
        ### AI-Powered Sales Forecasting & Demand Intelligence Platform

        Monitor retail sales, forecast future demand, detect anomalies, and
        analyze product performance through an interactive dashboard.
        """
    )

    st.divider()

    # ---------------- Apply Filters ---------------- #

    filtered_sales = sales.copy()

    filter1, filter2 = st.columns(2)

    with filter1:

        selected_region = st.selectbox(
            "Region",
            ["All"] + sorted(sales["Region"].unique())
        )

    with filter2:

        selected_category = st.selectbox(
            "Category",
            ["All"] + sorted(sales["Category"].unique())
        )

    if selected_region != "All":
        filtered_sales = filtered_sales[
            filtered_sales["Region"] == selected_region
        ]

    if selected_category != "All":
        filtered_sales = filtered_sales[
            filtered_sales["Category"] == selected_category
        ]

    # ---------------- KPI Calculations ---------------- #

    total_sales = filtered_sales["Sales"].sum()

    total_profit = filtered_sales["Profit"].sum()

    total_orders = filtered_sales["Order ID"].nunique()

    total_customers = filtered_sales["Customer ID"].nunique()

    average_order = filtered_sales["Sales"].mean()

    average_shipping = (
        (
            filtered_sales["Ship Date"]
            - filtered_sales["Order Date"]
        )
        .dt.days
        .mean()
    )

    yearly_sales = (
        filtered_sales
        .groupby(filtered_sales["Order Date"].dt.year)["Sales"]
        .sum()
    )

    if len(yearly_sales) >= 2:

        sales_growth = (
            (
                yearly_sales.iloc[-1]
                - yearly_sales.iloc[-2]
            )
            / yearly_sales.iloc[-2]
        ) * 100

    else:

        sales_growth = 0

    # ---------------- KPI Cards ---------------- #

    st.subheader("Business Overview")

    c1, c2, c3, c4 = st.columns(4)

    c5, c6, c7 = st.columns(3)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "💵 Total Profit",
        f"${total_profit:,.0f}"
    )

    c3.metric(
        "🛒 Orders",
        total_orders
    )

    c4.metric(
        "👥 Customers",
        total_customers
    )

    c5.metric(
        "📦 Avg Order Value",
        f"${average_order:.2f}"
    )

    c6.metric(
        "🚚 Avg Shipping Time",
        f"{average_shipping:.1f} Days"
    )

    c7.metric(
        "📈 Sales Growth",
        f"{sales_growth:.2f}%"
    )

    st.divider()

    # ---------------- Monthly Trend ---------------- #

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
        title_x=0.5
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- Region & Category ---------------- #

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
            title="Revenue by Region"
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
            values="Sales",
            names="Category",
            hole=0.45,
            title="Revenue by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- Top Products ---------------- #

    st.subheader("Top Selling Products")

    top_products = (
        filtered_sales
        .groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        color="Sales",
        title="Top 10 Products by Revenue"
    )

    fig.update_layout(
        template="plotly_white",
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- Summary ---------------- #

    st.subheader("Business Summary")

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

    st.success(
        f"""
        **Highlights**
        • Highest Revenue Region: **{best_region}**
        • Best Performing Category: **{best_category}**
        • Total Revenue: **${total_sales:,.0f}**
        • Sales Growth: **{sales_growth:.2f}%**
        • Average Shipping Time: **{average_shipping:.1f} Days**
        """
    )
    # ==========================================================
    # Forecast Explorer
    # =========================================================

elif page == "📈 Forecast Explorer":

    st.title("📈 Forecast Explorer")

    st.markdown(
        """
        Explore future sales forecasts and review the performance of the
        forecasting model used in this project.
        """
    )

    st.divider()

    # ---------------- Forecast Options ---------------- #

    left, right = st.columns(2)

    with left:

        forecast_type = st.selectbox(
            "Forecast Type",
            [
                "Overall Sales",
                "Category",
                "Region"
            ]
        )

    with right:

        horizon = st.slider(
            "Forecast Horizon (Months)",
            min_value=1,
            max_value=3,
            value=3
        )

    st.divider()

    # ---------------- Overall Forecast ---------------- #

    if forecast_type == "Overall Sales":

        display_forecast = forecast.head(horizon)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=display_forecast["Date"],
                y=display_forecast["Forecast"],
                mode="lines+markers",
                name="Forecast"
            )
        )

        fig.update_layout(
            template="plotly_white",
            title="Overall Sales Forecast",
            title_x=0.5,
            xaxis_title="Month",
            yaxis_title="Predicted Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            display_forecast,
            use_container_width=True
        )

    # ---------------- Category Forecast ---------------- #

    elif forecast_type == "Category":

        category = st.selectbox(
            "Select Category",
            sorted(
                sales["Category"].unique()
            )
        )

        category_forecast = (
            segment_forecasts[
                segment_forecasts["Segment"] == category
            ]
            .head(horizon)
        )

        fig = px.line(
            category_forecast,
            x="Date",
            y="Forecast",
            markers=True,
            title=f"{category} Forecast"
        )

        fig.update_layout(
            template="plotly_white",
            title_x=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            category_forecast,
            use_container_width=True
        )

    # ---------------- Region Forecast ---------------- #

    else:

        region = st.selectbox(
            "Select Region",
            sorted(
                sales["Region"].unique()
            )
        )

        region_forecast = (
            segment_forecasts[
                segment_forecasts["Segment"] == region
            ]
            .head(horizon)
        )

        fig = px.line(
            region_forecast,
            x="Date",
            y="Forecast",
            markers=True,
            title=f"{region} Forecast"
        )

        fig.update_layout(
            template="plotly_white",
            title_x=0.5
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            region_forecast,
            use_container_width=True
        )

    st.divider()

    # ---------------- Model Performance ---------------- #

    st.subheader("Forecast Model Performance")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    best_model = comparison.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{best_model['MAE']:.2f}"
    )

    col2.metric(
        "RMSE",
        f"{best_model['RMSE']:.2f}"
    )

    col3.metric(
        "MAPE",
        f"{best_model['MAPE']:.2f}%"
    )

    st.success(
        f"""
        **Recommended Model:** {best_model['Model']}
        This model achieved the lowest forecasting error and is recommended for 
        future demand prediction and inventory planning.
        """
    )

    st.download_button(
        label="📥 Download Forecast",
        data=display_forecast.to_csv(index=False),
        file_name="sales_forecast.csv",
        mime="text/csv"
    )
    # ==========================================================
# Anomaly Report
# ==========================================================

elif page == "🚨 Anomaly Report":

    st.title("🚨 Sales Anomaly Detection")

    st.markdown(
        """
        Detect unusual sales spikes and sudden drops using machine learning
        to identify potential business events and operational risks.
        """
    )

    st.divider()

    # ---------------- Overview ---------------- #

    total_anomalies = len(anomalies)

    highest_sale = anomalies["Sales"].max()

    lowest_sale = anomalies["Sales"].min()

    avg_sale = anomalies["Sales"].mean()

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
        f"${avg_sale:,.0f}"
    )

    st.divider()

    # ---------------- Weekly Sales ---------------- #

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
            name="Weekly Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomalies["Order Date"],
            y=anomalies["Sales"],
            mode="markers",
            marker=dict(
                size=11,
                color="red",
                symbol="circle"
            ),
            name="Detected Anomalies"
        )
    )

    fig.update_layout(

        template="plotly_white",

        title="Weekly Sales with Detected Anomalies",

        title_x=0.5,

        xaxis_title="Week",

        yaxis_title="Sales"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ---------------- Anomaly Table ---------------- #

    st.subheader("Detected Anomaly Events")

    display = anomalies.copy()

    display["Order Date"] = display["Order Date"].dt.strftime(
        "%d-%b-%Y"
    )

    st.dataframe(
        display,
        use_container_width=True
    )

    st.divider()

    # ---------------- Download ---------------- #

    st.download_button(

        "📥 Download Anomaly Report",

        anomalies.to_csv(index=False),

        file_name="weekly_anomalies.csv",

        mime="text/csv"

    )

    st.divider()

    # ---------------- Business Insights ---------------- #

    st.subheader("Business Interpretation")

    st.success(
        """
        ### Key Findings
        • Isolation Forest detected unusual sales behaviour throughout the dataset.

        • Most anomalies correspond to sharp demand spikes which may represent
        seasonal campaigns, festive sales or promotional events.

        • A few anomalies indicate unusually weak sales periods that may be linked
        to inventory shortages or lower customer demand.

        • Monitoring these events helps businesses improve inventory planning,
        marketing strategy and demand forecasting.
        """
    )
    # ==========================================================
# Demand Segmentation
# ==========================================================

elif page == "📦 Demand Segments":

    st.title("📦 Product Demand Segmentation")

    st.markdown("""
    Explore how product sub-categories are grouped based on
    sales behaviour, demand growth, volatility and order value.
    """)

    st.divider()

    # ---------------- KPIs ---------------- #

    total_products = clusters["Sub-Category"].nunique()

    total_clusters = clusters["Demand_Segment"].nunique()

    highest_sales = clusters["Total_Sales"].max()

    avg_growth = clusters["Growth_Rate"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Sub Categories",
        total_products
    )

    c2.metric(
        "Demand Segments",
        total_clusters
    )

    c3.metric(
        "Highest Sales",
        f"${highest_sales:,.0f}"
    )

    c4.metric(
        "Average Growth",
        f"{avg_growth:.1f}%"
    )

    st.divider()

    # ---------------- PCA Scatter ---------------- #

    st.subheader("Demand Cluster Visualization")

    fig = px.scatter(

        clusters,

        x="PC1",

        y="PC2",

        color="Demand_Segment",

        text="Sub-Category",

        size="Total_Sales",

        hover_data=[
            "Growth_Rate",
            "Avg_Order_Value"
        ]

    )

    fig.update_traces(
        textposition="top center"
    )

    fig.update_layout(

        template="plotly_white",

        title="K-Means Product Segmentation",

        title_x=0.5

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ---------------- Cluster Distribution ---------------- #

    left, right = st.columns(2)

    with left:

        segment_count = (

            clusters["Demand_Segment"]

            .value_counts()

            .reset_index()

        )

        segment_count.columns = [

            "Demand Segment",

            "Products"

        ]

        fig = px.bar(

            segment_count,

            x="Demand Segment",

            y="Products",

            color="Demand Segment",

            title="Products in Each Segment"

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

            names="Demand Segment",

            values="Products",

            hole=0.45,

            title="Demand Segment Share"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ---------------- Product Table ---------------- #

    st.subheader("Product Classification")

    st.dataframe(

        clusters.sort_values(

            "Demand_Segment"

        ),

        use_container_width=True

    )

    st.divider()

    # ---------------- Download ---------------- #

    st.download_button(

        "📥 Download Product Segments",

        clusters.to_csv(index=False),

        file_name="product_segments.csv",

        mime="text/csv"

    )

    st.divider()

    # ---------------- Business Strategy ---------------- #

    st.subheader("Recommended Stocking Strategy")

    st.info("""

    ### Core Revenue Drivers
    Maintain high inventory levels and prioritize replenishment.

    ### Fast Growing Products
    Increase procurement gradually to support rising demand.

    ### Premium High-Value Products
    Maintain controlled inventory while closely monitoring demand.

    ### Low Demand Products
    Adopt lean inventory practices and review pricing or promotions.

    """
    )
    # ==========================================================
# Model Performance
# ==========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Forecast Model Performance")

    st.markdown("""
    Compare the forecasting models used in this project and
    understand why the final production model was selected.
    """)

    st.divider()

    # ---------------- Best Model ---------------- #

    best_model_row = comparison.loc[
        comparison["RMSE"].idxmin()
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🏆 Best Model",
        best_model_row["Model"]
    )

    c2.metric(
        "MAE",
        f"{best_model_row['MAE']:.2f}"
    )

    c3.metric(
        "RMSE",
        f"{best_model_row['RMSE']:.2f}"
    )

    c4.metric(
        "MAPE",
        f"{best_model_row['MAPE']:.2f}%"
    )

    st.divider()

    # ---------------- Comparison Table ---------------- #

    st.subheader("Model Comparison")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.divider()

    # ---------------- Charts ---------------- #

    left, right = st.columns(2)

    with left:

        fig = px.bar(

            comparison,

            x="Model",

            y="MAE",

            color="Model",

            title="Mean Absolute Error"

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

            comparison,

            x="Model",

            y="RMSE",

            color="Model",

            title="Root Mean Squared Error"

        )

        fig.update_layout(

            template="plotly_white",

            showlegend=False

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ---------------- MAPE ---------------- #

    fig = px.line(

        comparison,

        x="Model",

        y="MAPE",

        markers=True,

        title="Mean Absolute Percentage Error"

    )

    fig.update_layout(

        template="plotly_white",

        title_x=0.5

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ---------------- Recommendation ---------------- #

    st.subheader("Production Recommendation")

    st.success(

        f"""
        ### Recommended Forecasting Model

        **{best_model_row['Model']}**

        This model achieved the lowest forecasting error and demonstrated
        the strongest balance between prediction accuracy and generalization.

        It is recommended for:

        • Sales Forecasting

        • Inventory Planning

        • Demand Prediction

        • Supply Chain Decision Support

        • Production Deployment

        """
    )

    st.divider()

    # ---------------- Download ---------------- #

    st.download_button(

        "📥 Download Model Results",

        comparison.to_csv(index=False),

        file_name="model_performance.csv",

        mime="text/csv"

    )
    # ==========================================================
# Model Performance
# ==========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Forecast Model Performance")

    st.markdown("""
    Compare the forecasting models used in this project and
    understand why the final production model was selected.
    """)

    st.divider()

    # ---------------- Best Model ---------------- #

    best_model_row = comparison.loc[
        comparison["RMSE"].idxmin()
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🏆 Best Model",
        best_model_row["Model"]
    )

    c2.metric(
        "MAE",
        f"{best_model_row['MAE']:.2f}"
    )

    c3.metric(
        "RMSE",
        f"{best_model_row['RMSE']:.2f}"
    )

    c4.metric(
        "MAPE",
        f"{best_model_row['MAPE']:.2f}%"
    )

    st.divider()

    # ---------------- Comparison Table ---------------- #

    st.subheader("Model Comparison")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.divider()

    # ---------------- Charts ---------------- #

    left, right = st.columns(2)

    with left:

        fig = px.bar(

            comparison,

            x="Model",

            y="MAE",

            color="Model",

            title="Mean Absolute Error"

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

            comparison,

            x="Model",

            y="RMSE",

            color="Model",

            title="Root Mean Squared Error"

        )

        fig.update_layout(

            template="plotly_white",

            showlegend=False

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ---------------- MAPE ---------------- #

    fig = px.line(

        comparison,

        x="Model",

        y="MAPE",

        markers=True,

        title="Mean Absolute Percentage Error"

    )

    fig.update_layout(

        template="plotly_white",

        title_x=0.5

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ---------------- Recommendation ---------------- #

    st.subheader("Production Recommendation")

    st.success(

        f"""
        ### Recommended Forecasting Model

        **{best_model_row['Model']}**

        This model achieved the lowest forecasting error and demonstrated
        the strongest balance between prediction accuracy and generalization.

        It is recommended for:

        • Sales Forecasting

        • Inventory Planning

        • Demand Prediction

        • Supply Chain Decision Support

        • Production Deployment

        """
    )

    st.divider()

    # ---------------- Download ---------------- #

    st.download_button(

        "📥 Download Model Results",

        comparison.to_csv(index=False),

        file_name="model_performance.csv",

        mime="text/csv"

    )
    # ==========================================================
    # Footer
    # ==========================================================

    st.divider()

    st.caption(
        "RetailPulse AI • Built using Python, Streamlit, XGBoost, Prophet, SARIMA, Scikit-learn and Plotly"
    )   
