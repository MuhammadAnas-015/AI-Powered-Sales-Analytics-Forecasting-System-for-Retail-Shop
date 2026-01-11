"""
Streamlit GUI Application
AI-Powered Sales Analytics & Forecasting System
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import io

# Import all modules
from data_generator import SalesDataGenerator
from eda_module import SalesEDA
from visualization_module import SalesVisualizer
from ml_forecasting import SalesForecaster
from product_demand_prediction import ProductDemandPredictor
from ai_insights import AIBusinessInsights
from ai_recommendations import SmartRecommendations

# Page configuration
st.set_page_config(
    page_title="AI Sales Analytics System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        padding: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<p class="main-header">🚀 AI-Powered Sales Analytics & Forecasting System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete Data Science & AI Solution for Retail Shop</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Choose an option:",
        ["🏠 Home", "📤 Upload Data", "📊 Data Analysis", "🤖 ML Forecasting", "💡 AI Insights", "📈 Visualizations"]
    )
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'eda_results' not in st.session_state:
        st.session_state.eda_results = None
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'demand_results' not in st.session_state:
        st.session_state.demand_results = None
    
    # Home Page
    if page == "🏠 Home":
        st.markdown("## Welcome to AI Sales Analytics System")
        st.markdown("""
        This comprehensive system provides:
        
        - **📊 Exploratory Data Analysis (EDA)**: Deep insights into your sales data
        - **📈 Professional Visualizations**: Beautiful charts and graphs
        - **🤖 Machine Learning Forecasting**: Predict future sales using ML models
        - **🛍️ Product Demand Prediction**: Forecast product-level demand
        - **💡 AI Business Insights**: Get answers to business questions
        - **🎯 Smart Recommendations**: Data-driven business recommendations
        
        ### Getting Started:
        1. **Upload Data**: Go to "Upload Data" tab and upload your CSV file or generate sample data
        2. **View Analysis**: Navigate through different tabs to explore insights
        3. **Get Forecasts**: Use ML models to predict future sales
        4. **Ask Questions**: Use AI Insights to get answers about your business
        """)
        
        # Quick stats if data exists
        if st.session_state.df is not None:
            st.success("✅ Data loaded successfully!")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(st.session_state.df):,}")
            with col2:
                st.metric("Total Revenue", f"${st.session_state.df['Revenue'].sum():,.2f}")
            with col3:
                st.metric("Unique Products", st.session_state.df['Product'].nunique())
            with col4:
                st.metric("Date Range", f"{st.session_state.df['Date'].min().date()} to {st.session_state.df['Date'].max().date()}")
        else:
            st.info("👆 Please upload your sales data or generate sample data to get started!")
    
    # Upload Data Page
    elif page == "📤 Upload Data":
        st.header("📤 Upload Sales Data")
        
        option = st.radio(
            "Choose an option:",
            ["Upload CSV File", "Generate Sample Data"]
        )
        
        if option == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your sales data CSV file"
            )
            
            if uploaded_file is not None:
                try:
                    # Read CSV
                    df = pd.read_csv(uploaded_file)
                    
                    # Convert Date column if exists
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Show preview
                    st.success("✅ File uploaded successfully!")
                    st.subheader("📋 Data Preview (First 10 rows)")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show data info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Data Shape:**", df.shape)
                        st.write("**Columns:**", list(df.columns))
                    with col2:
                        st.write("**Missing Values:**")
                        st.write(df.isnull().sum())
                    
                    # Save to session state
                    if st.button("✅ Use This Data", type="primary"):
                        st.session_state.df = df
                        st.success("✅ Data loaded! Navigate to other tabs to analyze.")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    st.info("Please ensure your CSV has the required columns: Date, Product, Category, Quantity, Unit_Price, Revenue, etc.")
        
        else:  # Generate Sample Data
            st.subheader("Generate Sample Sales Data")
            st.info("This will generate realistic sample retail sales data for demonstration.")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
            with col2:
                end_date = st.date_input("End Date", value=datetime(2024, 12, 31))
            
            if st.button("🔄 Generate Sample Data", type="primary"):
                with st.spinner("Generating sample data... This may take a moment."):
                    try:
                        generator = SalesDataGenerator(
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d')
                        )
                        df = generator.generate_sales_data()
                        
                        st.success("✅ Sample data generated successfully!")
                        st.subheader("📋 Generated Data Preview (First 10 rows)")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Show summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Records", f"{len(df):,}")
                        with col2:
                            st.metric("Total Revenue", f"${df['Revenue'].sum():,.2f}")
                        with col3:
                            st.metric("Unique Products", df['Product'].nunique())
                        with col4:
                            st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
                        
                        # Save to session state
                        if st.button("✅ Use Generated Data", type="primary"):
                            st.session_state.df = df
                            st.success("✅ Data loaded! Navigate to other tabs to analyze.")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error generating data: {str(e)}")
    
    # Data Analysis Page
    elif page == "📊 Data Analysis":
        if st.session_state.df is None:
            st.warning("⚠️ Please upload or generate data first!")
            return
        
        st.header("📊 Exploratory Data Analysis")
        
        if st.button("🔍 Run Complete EDA Analysis", type="primary"):
            with st.spinner("Analyzing data... This may take a moment."):
                eda = SalesEDA(st.session_state.df)
                st.session_state.eda_results = eda.run_full_analysis()
                st.success("✅ Analysis complete!")
        
        if st.session_state.eda_results:
            st.subheader("📈 Key Statistics")
            stats = st.session_state.eda_results['stats']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Revenue", f"${stats['total_revenue']:,.2f}")
            with col2:
                st.metric("Total Transactions", f"{stats['total_transactions']:,.0f}")
            with col3:
                st.metric("Avg Transaction", f"${stats['avg_transaction_value']:.2f}")
            with col4:
                st.metric("Products Sold", f"{stats['total_products_sold']:,.0f}")
            
            # Top Products
            st.subheader("🏆 Top 10 Products by Revenue")
            top_products = st.session_state.eda_results['top_products'].head(10)
            st.dataframe(top_products, use_container_width=True)
            
            # Category Analysis
            st.subheader("📂 Category Performance")
            categories = st.session_state.eda_results['categories']
            st.dataframe(categories, use_container_width=True)
            
            # Peak Hours
            st.subheader("⏰ Peak Sales Hours")
            peak_hours = st.session_state.eda_results['peak_hours'].head(10)
            st.dataframe(peak_hours, use_container_width=True)
    
    # ML Forecasting Page
    elif page == "🤖 ML Forecasting":
        if st.session_state.df is None:
            st.warning("⚠️ Please upload or generate data first!")
            return
        
        st.header("🤖 Machine Learning Sales Forecasting")
        
        if st.button("🚀 Train ML Models & Generate Forecasts", type="primary"):
            with st.spinner("Training ML models... This may take a few minutes."):
                try:
                    forecaster = SalesForecaster(st.session_state.df)
                    forecaster.train_models()  # This trains models and stores metrics in forecaster object
                    
                    # Generate forecasts
                    next_week = forecaster.predict_next_week()
                    next_month = forecaster.predict_next_month()
                    
                    st.session_state.forecaster = forecaster
                    st.session_state.next_week = next_week
                    st.session_state.next_month = next_month
                    
                    st.success("✅ ML models trained and forecasts generated!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        if 'forecaster' in st.session_state:
            # Model Performance
            st.subheader("📊 Model Performance Comparison")
            if hasattr(st.session_state.forecaster, 'metrics') and st.session_state.forecaster.metrics:
                metrics_df = pd.DataFrame(st.session_state.forecaster.metrics).T
                st.dataframe(metrics_df, use_container_width=True)
                
                # Find best model
                best_model = min(st.session_state.forecaster.metrics.items(), key=lambda x: x[1]['RMSE'])
                st.success(f"🏆 Best Model: **{best_model[0]}** (RMSE: ${best_model[1]['RMSE']:,.2f})")
            else:
                st.warning("Metrics not available. Please retrain the models.")
            
            # Next Week Forecast
            st.subheader("📅 Next Week Sales Forecast")
            st.dataframe(st.session_state.next_week, use_container_width=True)
            total_week = st.session_state.next_week['Predicted_Revenue'].sum()
            st.metric("Total Predicted Revenue (Next Week)", f"${total_week:,.2f}")
            
            # Next Month Forecast
            st.subheader("📅 Next Month Sales Forecast")
            st.dataframe(st.session_state.next_month.head(15), use_container_width=True)
            total_month = st.session_state.next_month['Predicted_Revenue'].sum()
            st.metric("Total Predicted Revenue (Next Month)", f"${total_month:,.2f}")
    
    # AI Insights Page
    elif page == "💡 AI Insights":
        if st.session_state.df is None:
            st.warning("⚠️ Please upload or generate data first!")
            return
        
        st.header("💡 AI Business Insights & Q&A")
        
        # Initialize AI insights if needed
        if st.session_state.eda_results is None:
            with st.spinner("Preparing AI insights..."):
                eda = SalesEDA(st.session_state.df)
                st.session_state.eda_results = eda.run_full_analysis()
        
        if 'demand_results' not in st.session_state or st.session_state.demand_results is None:
            try:
                with st.spinner("Preparing demand predictions..."):
                    demand_predictor = ProductDemandPredictor(st.session_state.df)
                    demand_predictor.train_product_models()
                    st.session_state.demand_results = demand_predictor.get_demand_insights(silent=True)
            except ValueError as e:
                st.warning(f"⚠️ {str(e)}")
                st.info("💡 Tip: Ensure your data has sufficient historical records (at least 30+ days) for each product.")
                st.session_state.demand_results = None
            except Exception as e:
                st.error(f"❌ Error preparing demand predictions: {str(e)}")
                st.session_state.demand_results = None
        
        # Q&A Section
        st.subheader("🤖 Ask Business Questions")
        
        # Pre-defined questions
        questions = [
            "Why did sales drop in March?",
            "Which product should I promote next?",
            "What is the reason for low revenue this month?",
            "Show general business insights"
        ]
        
        selected_question = st.selectbox("Select a question or type your own:", ["-- Select --"] + questions)
        
        custom_question = st.text_input("Or type your own question:")
        
        if st.button("🔍 Get Answer", type="primary"):
            question = custom_question if custom_question else selected_question
            
            if question and question != "-- Select --":
                with st.spinner("Analyzing..."):
                    # Get ML results if available
                    ml_results = st.session_state.forecaster if 'forecaster' in st.session_state else None
                    ai_insights = AIBusinessInsights(
                        st.session_state.df,
                        st.session_state.eda_results,
                        ml_results,
                        st.session_state.demand_results
                    )
                    answer = ai_insights.answer_question(question)
                    st.markdown(f"### 🤖 AI Response:")
                    st.markdown(answer)
        
        # Recommendations
        st.subheader("🎯 Smart AI Recommendations")
        if st.button("📋 Generate Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                if 'forecaster' not in st.session_state:
                    forecaster = SalesForecaster(st.session_state.df)
                    forecaster.train_models()
                    st.session_state.forecaster = forecaster
                
                try:
                    demand_predictor = ProductDemandPredictor(st.session_state.df)
                    demand_predictor.train_product_models()
                    
                    recommendations = SmartRecommendations(
                        st.session_state.df,
                        st.session_state.forecaster,
                        demand_predictor
                    )
                    recs = recommendations.generate_all_recommendations()
                    
                    st.session_state.recommendations = recs
                except ValueError as e:
                    st.warning(f"⚠️ {str(e)}")
                    st.info("💡 Tip: Product demand prediction requires sufficient historical data. Using basic recommendations instead.")
                    # Generate basic recommendations without demand prediction
                    recommendations = SmartRecommendations(
                        st.session_state.df,
                        st.session_state.forecaster,
                        None
                    )
                    recs = recommendations.generate_all_recommendations()
                    st.session_state.recommendations = recs
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")
        
        if 'recommendations' in st.session_state:
            recs_df = pd.DataFrame(st.session_state.recommendations)
            st.dataframe(recs_df, use_container_width=True)
    
    # Visualizations Page
    elif page == "📈 Visualizations":
        if st.session_state.df is None:
            st.warning("⚠️ Please upload or generate data first!")
            return
        
        st.header("📈 Data Visualizations")
        
        if st.button("🎨 Generate All Visualizations", type="primary"):
            with st.spinner("Generating visualizations..."):
                visualizer = SalesVisualizer(st.session_state.df)
                visualizer.generate_all_visualizations()
                st.success("✅ Visualizations generated! Check the 'visualizations' folder.")
        
        # Display visualizations if they exist
        viz_folder = "visualizations"
        if os.path.exists(viz_folder):
            viz_files = [f for f in os.listdir(viz_folder) if f.endswith('.png')]
            
            if viz_files:
                st.subheader("📊 Generated Charts")
                for viz_file in viz_files:
                    st.image(os.path.join(viz_folder, viz_file), use_container_width=True)
            else:
                st.info("Click 'Generate All Visualizations' to create charts.")

if __name__ == "__main__":
    main()
