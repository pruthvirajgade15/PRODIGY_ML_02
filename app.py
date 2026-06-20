import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.Pipeline.predict_pipeline import CustomData, PredictPipeline

# Page configuration with enhanced styling
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .header-container {
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_customer_data():
    data_path = os.path.join("artifacts", "data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


@st.cache_resource(show_spinner=False)
def get_predict_pipeline():
    return PredictPipeline()


def get_cluster_info(cluster_id):
    """Get cluster characteristics based on cluster ID"""
    cluster_info = {
        0: {
            "name": "High Value Customers",
            "emoji": "💎",
            "color": "#FFD700",
            "description": "High income, high spending. VIP customers requiring premium service.",
        },
        1: {
            "name": "Budget Conscious",
            "emoji": "🏷️",
            "color": "#4CAF50",
            "description": "High income, low spending. Price-sensitive customers.",
        },
        2: {
            "name": "At-Risk Customers",
            "emoji": "⚠️",
            "color": "#FF6B6B",
            "description": "Low income, low spending. Retention focus needed.",
        },
        3: {
            "name": "Standard Customers",
            "emoji": "👥",
            "color": "#2196F3",
            "description": "Average income and spending patterns.",
        },
    }
    return cluster_info.get(
        cluster_id,
        {
            "name": "Unknown",
            "emoji": "❓",
            "color": "#9C27B0",
            "description": "Cluster characteristics",
        },
    )


def predict_cluster(form_data):
    custom_data = CustomData(
        age=form_data["age"],
        annual_income=form_data["annual_income"],
        spending_score=form_data["spending_score"],
    )
    prediction_df = custom_data.get_data_as_data_frame()
    result = get_predict_pipeline().predict(prediction_df)
    return int(result[0])


def create_cluster_visualization(data):
    """Create an enhanced cluster visualization"""
    if data is not None and "Annual Income (k$)" in data.columns:
        income_col = "Annual Income (k$)"
        spending_col = "Spending Score (1-100)"
    elif data is not None:
        income_col = "Annual_Income"
        spending_col = "Spending_Score"
    else:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        data[income_col],
        data[spending_col],
        alpha=0.6,
        s=100,
        c=data.get("Cluster", range(len(data))),
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Annual Income (k$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spending Score", fontsize=12, fontweight="bold")
    ax.set_title("Customer Cluster Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    return fig


def main():
    # Header Section
    st.markdown(
        """
        <div class='header-container'>
            <h1>🛍️ Mall Customer Segmentation</h1>
            <p>AI-powered customer clustering for targeted marketing strategies</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🎯 Predict Cluster", "📊 Data Analysis", "ℹ️ About"])

    data = load_customer_data()

    with tab1:
        st.subheader("Customer Prediction Form")
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender_select")
            age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)

        with col2:
            st.markdown("#### 💰 Financial Profile")
            annual_income = st.slider(
                "Annual Income (k$)",
                min_value=15,
                max_value=140,
                value=50,
                step=5,
            )

        with col3:
            st.markdown("#### 🛒 Shopping Behavior")
            spending_score = st.slider(
                "Spending Score",
                min_value=1,
                max_value=100,
                value=50,
                step=1,
            )

        # Prediction button with custom styling
        st.divider()
        col1_pred, col2_pred, col3_pred = st.columns(3)
        with col2_pred:
            predict_clicked = st.button(
                "🔮 Predict Cluster",
                type="primary",
                use_container_width=True,
                key="predict_button",
            )

        if predict_clicked:
            form_data = {
                "age": age,
                "annual_income": annual_income,
                "spending_score": spending_score,
            }

            try:
                with st.spinner("✨ Analyzing customer profile..."):
                    cluster = predict_cluster(form_data)

                cluster_info = get_cluster_info(cluster)

                # Display result in an enhanced card
                st.divider()
                st.markdown("### 🎯 Prediction Result")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.markdown(
                        f"""
                        <div class='info-card'>
                            <h3>{cluster_info['emoji']} {cluster_info['name']}</h3>
                            <p><strong>Cluster ID:</strong> {cluster}</p>
                            <p>{cluster_info['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with result_col2:
                    # Display customer metrics
                    st.metric("Customer Age", f"{age} years")
                    st.metric("Annual Income", f"${annual_income}k")
                    st.metric("Spending Score", f"{spending_score}/100")

                st.success("✅ Prediction completed successfully!")

                # Display customer summary
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                with col_summary1:
                    st.text(f"Gender: {gender}")
                with col_summary2:
                    st.text(f"Income Category: {'High' if annual_income > 70 else 'Medium' if annual_income > 40 else 'Low'}")
                with col_summary3:
                    st.text(f"Spending Type: {'High Spender' if spending_score > 60 else 'Low Spender'}")

            except Exception as error:
                st.error(f"❌ Prediction failed: {error}")
        else:
            st.info("👆 Fill in the customer details and click 'Predict Cluster' to get started!")

    with tab2:
        st.subheader("📊 Data Analytics Dashboard")

        if data is not None:
            # Dataset statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("Total Customers", len(data))

            with col_stat2:
                if "Annual Income (k$)" in data.columns:
                    avg_income = data["Annual Income (k$)"].mean()
                else:
                    avg_income = data["Annual_Income"].mean()
                st.metric("Avg Income", f"${avg_income:.0f}k")

            with col_stat3:
                if "Spending Score (1-100)" in data.columns:
                    avg_spending = data["Spending Score (1-100)"].mean()
                else:
                    avg_spending = data["Spending_Score"].mean()
                st.metric("Avg Spending Score", f"{avg_spending:.1f}")

            with col_stat4:
                if "Age" in data.columns:
                    avg_age = data["Age"].mean()
                else:
                    avg_age = 0
                st.metric("Avg Age", f"{avg_age:.0f} years")

            st.divider()

            # Visualization tabs
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(
                ["📈 Cluster Visualization", "📊 Distribution Charts", "🔍 Data Table"]
            )

            with viz_tab1:
                st.markdown("#### Customer Cluster Distribution")
                fig = create_cluster_visualization(data)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("Unable to create visualization")

            with viz_tab2:
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    # Income distribution
                    if "Annual Income (k$)" in data.columns:
                        fig_income, ax_income = plt.subplots(figsize=(8, 5))
                        ax_income.hist(
                            data["Annual Income (k$)"],
                            bins=20,
                            color="#667eea",
                            edgecolor="black",
                            alpha=0.7,
                        )
                        ax_income.set_xlabel("Annual Income (k$)", fontweight="bold")
                        ax_income.set_ylabel("Frequency", fontweight="bold")
                        ax_income.set_title("Income Distribution", fontweight="bold")
                        ax_income.grid(True, alpha=0.3)
                        st.pyplot(fig_income)

                with col_chart2:
                    # Spending score distribution
                    if "Spending Score (1-100)" in data.columns:
                        fig_spending, ax_spending = plt.subplots(figsize=(8, 5))
                        ax_spending.hist(
                            data["Spending Score (1-100)"],
                            bins=20,
                            color="#764ba2",
                            edgecolor="black",
                            alpha=0.7,
                        )
                        ax_spending.set_xlabel("Spending Score", fontweight="bold")
                        ax_spending.set_ylabel("Frequency", fontweight="bold")
                        ax_spending.set_title("Spending Score Distribution", fontweight="bold")
                        ax_spending.grid(True, alpha=0.3)
                        st.pyplot(fig_spending)

            with viz_tab3:
                st.markdown("#### Dataset Preview")
                st.dataframe(data.head(20), use_container_width=True)

                # Add download button
                csv = data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name="customer_data.csv",
                    mime="text/csv",
                )

        else:
            st.warning("⚠️ No data available. Please check the data file.")

    with tab3:
        st.markdown(
            """
            ### About This Application
            
            **Mall Customer Segmentation** is an AI-powered tool that uses machine learning 
            to classify shopping mall customers into distinct groups based on their:
            
            - 👤 **Demographics**: Age and gender
            - 💰 **Financial Profile**: Annual income
            - 🛒 **Shopping Behavior**: Spending score
            
            #### How It Works:
            
            1. **Data Collection**: Customer data is collected and preprocessed
            2. **Feature Engineering**: Relevant features are extracted and normalized
            3. **Clustering**: K-Means algorithm groups customers into segments
            4. **Prediction**: New customers are classified into existing clusters
            
            #### Use Cases:
            
            - 🎯 **Targeted Marketing**: Create tailored campaigns for each segment
            - 💡 **Customer Insights**: Understand customer behaviors and preferences
            - 📊 **Business Strategy**: Optimize inventory and service offerings
            - 💼 **CRM Optimization**: Better customer relationship management
            
            #### Model Information:
            
            - **Algorithm**: K-Means Clustering
            - **Number of Clusters**: 4 distinct customer segments
            - **Features**: Age, Annual Income, Spending Score
            - **Data Source**: Mall customer purchase history
            
            ---
            
            *Powered by ML & Streamlit* 🚀
            """
        )

        # Add footer with some stats
        st.divider()
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>Made with ❤️ | Data-Driven Customer Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

