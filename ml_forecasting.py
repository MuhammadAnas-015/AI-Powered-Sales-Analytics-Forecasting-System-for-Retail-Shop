"""
Machine Learning Sales Forecasting Module
Predicts future sales using Linear Regression, Random Forest, and XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SalesForecaster:
    """ML models for sales forecasting"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def prepare_features(self):
        """Prepare features for ML models"""
        # Aggregate to daily level
        daily_data = self.df.groupby(self.df['Date'].dt.date).agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Transactions'})
        daily_data.index = pd.to_datetime(daily_data.index)
        daily_data = daily_data.sort_index()
        
        # Create time-based features
        daily_data['Day'] = daily_data.index.day
        daily_data['Month'] = daily_data.index.month
        daily_data['Year'] = daily_data.index.year
        daily_data['DayOfWeek'] = daily_data.index.dayofweek
        daily_data['DayOfYear'] = daily_data.index.dayofyear
        daily_data['Week'] = daily_data.index.isocalendar().week
        daily_data['Quarter'] = daily_data.index.quarter
        
        # Lag features (previous day, week, month)
        daily_data['Revenue_Lag1'] = daily_data['Revenue'].shift(1)
        daily_data['Revenue_Lag7'] = daily_data['Revenue'].shift(7)
        daily_data['Revenue_Lag30'] = daily_data['Revenue'].shift(30)
        
        # Rolling statistics
        daily_data['Revenue_MA7'] = daily_data['Revenue'].rolling(window=7).mean()
        daily_data['Revenue_MA30'] = daily_data['Revenue'].rolling(window=30).mean()
        daily_data['Revenue_STD7'] = daily_data['Revenue'].rolling(window=7).std()
        
        # Trend features
        daily_data['Days_Since_Start'] = (daily_data.index - daily_data.index.min()).days
        
        # Remove rows with NaN (from lag features)
        daily_data = daily_data.dropna()
        
        return daily_data
    
    def train_models(self, test_size=0.2):
        """Train all ML models"""
        print("\n" + "=" * 80)
        print("🤖 TRAINING MACHINE LEARNING MODELS")
        print("=" * 80)
        
        daily_data = self.prepare_features()
        
        # Feature selection
        feature_columns = [
            'Day', 'Month', 'Year', 'DayOfWeek', 'DayOfYear', 'Week', 'Quarter',
            'Revenue_Lag1', 'Revenue_Lag7', 'Revenue_Lag30',
            'Revenue_MA7', 'Revenue_MA30', 'Revenue_STD7',
            'Days_Since_Start', 'Transactions', 'Quantity'
        ]
        
        X = daily_data[feature_columns]
        y = daily_data['Revenue']
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n📊 Training Data: {len(X_train)} days")
        print(f"📊 Test Data: {len(X_test)} days")
        print(f"📊 Features: {len(feature_columns)}")
        
        # Model 1: Linear Regression
        print("\n1️⃣ Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        self.models['Linear Regression'] = lr_model
        self.predictions['Linear Regression'] = lr_pred
        self.metrics['Linear Regression'] = self._calculate_metrics(y_test, lr_pred)
        
        # Model 2: Random Forest
        print("2️⃣ Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        self.models['Random Forest'] = rf_model
        self.predictions['Random Forest'] = rf_pred
        self.metrics['Random Forest'] = self._calculate_metrics(y_test, rf_pred)
        
        # Model 3: XGBoost
        print("3️⃣ Training XGBoost Regressor...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        
        self.models['XGBoost'] = xgb_model
        self.predictions['XGBoost'] = xgb_pred
        self.metrics['XGBoost'] = self._calculate_metrics(y_test, xgb_pred)
        
        # Store feature importance
        self.feature_importance = {
            'Random Forest': dict(zip(feature_columns, rf_model.feature_importances_)),
            'XGBoost': dict(zip(feature_columns, xgb_model.feature_importances_))
        }
        
        # Store data for future predictions
        self.daily_data = daily_data
        self.feature_columns = feature_columns
        self.X_test = X_test
        self.y_test = y_test
        
        self._print_model_comparison()
        
        return self.models
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def _print_model_comparison(self):
        """Print model comparison"""
        print("\n" + "=" * 80)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("=" * 80)
        
        comparison_df = pd.DataFrame(self.metrics).T
        print("\n" + comparison_df.to_string())
        
        # Find best model
        best_model = min(self.metrics.items(), key=lambda x: x[1]['RMSE'])
        print(f"\n🏆 Best Model: {best_model[0]} (RMSE: ${best_model[1]['RMSE']:,.2f})")
    
    def predict_next_week(self, model_name='XGBoost'):
        """Predict sales for next week"""
        if model_name not in self.models:
            model_name = list(self.models.keys())[-1]  # Use best model
        
        model = self.models[model_name]
        daily_data = self.daily_data.copy()
        
        # Get last date
        last_date = daily_data.index[-1]
        
        # Predict next 7 days
        predictions = []
        dates = []
        
        for i in range(1, 8):
            next_date = last_date + pd.Timedelta(days=i)
            dates.append(next_date)
            
            # Prepare features for this date
            features = pd.DataFrame({
                'Day': [next_date.day],
                'Month': [next_date.month],
                'Year': [next_date.year],
                'DayOfWeek': [next_date.dayofweek],
                'DayOfYear': [next_date.dayofyear],
                'Week': [next_date.isocalendar().week],
                'Quarter': [next_date.quarter],
                'Days_Since_Start': [(next_date - daily_data.index.min()).days],
                'Revenue_Lag1': [daily_data['Revenue'].iloc[-1] if i == 1 else predictions[-1]],
                'Revenue_Lag7': [daily_data['Revenue'].iloc[-7] if len(daily_data) >= 7 else daily_data['Revenue'].mean()],
                'Revenue_Lag30': [daily_data['Revenue'].iloc[-30] if len(daily_data) >= 30 else daily_data['Revenue'].mean()],
                'Revenue_MA7': [daily_data['Revenue'].tail(7).mean()],
                'Revenue_MA30': [daily_data['Revenue'].tail(30).mean()],
                'Revenue_STD7': [daily_data['Revenue'].tail(7).std()],
                'Transactions': [daily_data['Transactions'].mean()],
                'Quantity': [daily_data['Quantity'].mean()]
            })
            
            # Ensure all features are in correct order
            features = features[self.feature_columns]
            
            pred = model.predict(features)[0]
            predictions.append(max(0, pred))  # Ensure non-negative
        
        next_week_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Revenue': predictions
        })
        
        total_predicted = sum(predictions)
        
        print("\n" + "=" * 80)
        print(f"📅 NEXT WEEK SALES FORECAST ({model_name})")
        print("=" * 80)
        print(next_week_df.to_string(index=False))
        print(f"\n💰 Total Predicted Revenue (Next Week): ${total_predicted:,.2f}")
        
        return next_week_df
    
    def predict_next_month(self, model_name='XGBoost'):
        """Predict sales for next month"""
        if model_name not in self.models:
            model_name = list(self.models.keys())[-1]
        
        model = self.models[model_name]
        daily_data = self.daily_data.copy()
        
        last_date = daily_data.index[-1]
        
        # Predict next 30 days
        predictions = []
        dates = []
        
        for i in range(1, 31):
            next_date = last_date + pd.Timedelta(days=i)
            dates.append(next_date)
            
            features = pd.DataFrame({
                'Day': [next_date.day],
                'Month': [next_date.month],
                'Year': [next_date.year],
                'DayOfWeek': [next_date.dayofweek],
                'DayOfYear': [next_date.dayofyear],
                'Week': [next_date.isocalendar().week],
                'Quarter': [next_date.quarter],
                'Days_Since_Start': [(next_date - daily_data.index.min()).days],
                'Revenue_Lag1': [daily_data['Revenue'].iloc[-1] if i == 1 else predictions[-1]],
                'Revenue_Lag7': [daily_data['Revenue'].iloc[-7] if len(daily_data) >= 7 else daily_data['Revenue'].mean()],
                'Revenue_Lag30': [daily_data['Revenue'].iloc[-30] if len(daily_data) >= 30 else daily_data['Revenue'].mean()],
                'Revenue_MA7': [daily_data['Revenue'].tail(7).mean()],
                'Revenue_MA30': [daily_data['Revenue'].tail(30).mean()],
                'Revenue_STD7': [daily_data['Revenue'].tail(7).std()],
                'Transactions': [daily_data['Transactions'].mean()],
                'Quantity': [daily_data['Quantity'].mean()]
            })
            
            features = features[self.feature_columns]
            pred = model.predict(features)[0]
            predictions.append(max(0, pred))
        
        next_month_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Revenue': predictions
        })
        
        # Aggregate by week
        next_month_df['Week'] = next_month_df['Date'].dt.isocalendar().week
        weekly_summary = next_month_df.groupby('Week')['Predicted_Revenue'].sum()
        
        total_predicted = sum(predictions)
        
        print("\n" + "=" * 80)
        print(f"📅 NEXT MONTH SALES FORECAST ({model_name})")
        print("=" * 80)
        print("\n📊 Weekly Summary:")
        print(weekly_summary.to_string())
        print(f"\n💰 Total Predicted Revenue (Next Month): ${total_predicted:,.2f}")
        print(f"📈 Average Daily Revenue: ${total_predicted/30:,.2f}")
        
        return next_month_df
    
    def get_feature_importance(self, model_name='XGBoost', top_n=10):
        """Get feature importance for tree-based models"""
        if model_name not in self.feature_importance:
            print(f"Feature importance not available for {model_name}")
            return None
        
        importance = self.feature_importance[model_name]
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Top {top_n} Most Important Features ({model_name}):")
        for i, (feature, score) in enumerate(sorted_importance[:top_n], 1):
            print(f"{i}. {feature}: {score:.4f}")
        
        return sorted_importance
