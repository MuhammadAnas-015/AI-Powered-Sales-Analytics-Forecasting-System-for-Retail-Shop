# 🚀 How to Run the GUI Application

## Quick Start

1. **Install Streamlit** (if not already installed):
   ```bash
   pip install streamlit
   ```

2. **Run the GUI Application**:
   ```bash
   streamlit run app.py
   ```

3. **The application will open in your browser automatically!**

## Features

### 📤 Upload Data Tab
- **Upload CSV File**: Browse and upload your sales data CSV file
- **Generate Sample Data**: Create realistic sample data for testing
- **Data Preview**: See a preview of your data before analysis

### 📊 Data Analysis Tab
- Complete EDA analysis
- Key statistics and metrics
- Top products analysis
- Category performance
- Peak hours analysis

### 🤖 ML Forecasting Tab
- Train ML models (Linear Regression, Random Forest, XGBoost)
- View model performance comparison
- Next week sales forecast
- Next month sales forecast

### 💡 AI Insights Tab
- Ask business questions
- Get AI-powered answers
- Generate smart recommendations
- View actionable business advice

### 📈 Visualizations Tab
- Generate all visualizations
- View charts and graphs
- Download visualizations

## CSV File Format

Your CSV file should have these columns:
- `Date` (datetime format)
- `Product` (string)
- `Category` (string)
- `Quantity` (numeric)
- `Unit_Price` (numeric)
- `Revenue` (numeric)
- `Hour` (0-23)
- `Day_of_Week` (string)
- `Month` (string)
- `Year` (numeric)
- `Quarter` (string)

## Troubleshooting

**Issue**: Streamlit not found
- **Solution**: `pip install streamlit`

**Issue**: Port already in use
- **Solution**: `streamlit run app.py --server.port 8502`

**Issue**: Browser doesn't open
- **Solution**: Copy the URL from terminal (usually http://localhost:8501)
