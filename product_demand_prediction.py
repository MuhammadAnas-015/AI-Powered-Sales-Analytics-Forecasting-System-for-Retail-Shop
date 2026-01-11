"""
Product Demand Prediction Module
Predicts product-level demand for inventory management
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ProductDemandPredictor:
    """Predict product-level demand"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.models = {}
        self.predictions = {}
    
    def prepare_product_features(self):
        """Prepare features for product-level prediction"""
        # Aggregate to product-date level
        product_daily = self.df.groupby(['Product', 'Date']).agg({
            'Quantity': 'sum',
            'Revenue': 'sum',
            'Unit_Price': 'mean'
        }).reset_index()
        
        product_daily['Date'] = pd.to_datetime(product_daily['Date'])
        product_daily = product_daily.sort_values(['Product', 'Date'])
        
        # Create time features
        product_daily['Day'] = product_daily['Date'].dt.day
        product_daily['Month'] = product_daily['Date'].dt.month
        product_daily['Year'] = product_daily['Date'].dt.year
        product_daily['DayOfWeek'] = product_daily['Date'].dt.dayofweek
        product_daily['Quarter'] = product_daily['Date'].dt.quarter
        
        # Product-level lag features (with forward fill for missing values)
        product_daily['Quantity_Lag1'] = product_daily.groupby('Product')['Quantity'].shift(1)
        product_daily['Quantity_Lag7'] = product_daily.groupby('Product')['Quantity'].shift(7)
        product_daily['Quantity_Lag30'] = product_daily.groupby('Product')['Quantity'].shift(30)
        
        # Fill NaN lag features with product mean or overall mean
        for lag_col in ['Quantity_Lag1', 'Quantity_Lag7', 'Quantity_Lag30']:
            # Fill with product mean first, then overall mean
            product_daily[lag_col] = product_daily.groupby('Product')[lag_col].transform(
                lambda x: x.fillna(x.mean() if not x.isna().all() else product_daily['Quantity'].mean())
            )
            # If still NaN, fill with overall mean
            product_daily[lag_col] = product_daily[lag_col].fillna(product_daily['Quantity'].mean())
        
        # Rolling statistics per product (with min_periods to handle short histories)
        product_daily['Quantity_MA7'] = product_daily.groupby('Product')['Quantity'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        product_daily['Quantity_MA30'] = product_daily.groupby('Product')['Quantity'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        # Encode product names
        le = LabelEncoder()
        product_daily['Product_Encoded'] = le.fit_transform(product_daily['Product'])
        self.product_encoder = le
        
        # Get category mapping
        category_map = self.df[['Product', 'Category']].drop_duplicates().set_index('Product')['Category'].to_dict()
        product_daily['Category'] = product_daily['Product'].map(category_map)
        le_cat = LabelEncoder()
        product_daily['Category_Encoded'] = le_cat.fit_transform(product_daily['Category'])
        self.category_encoder = le_cat
        
        # Ensure no NaN values remain (use forward fill then backward fill)
        product_daily = product_daily.ffill().bfill()
        
        # Final check - remove any remaining NaN rows
        product_daily = product_daily.dropna()
        
        # If still empty after all processing, raise informative error
        if len(product_daily) == 0:
            raise ValueError("❌ Insufficient historical data. Products need at least some sales history for demand prediction.")
        
        return product_daily
    
    def train_product_models(self):
        """Train models for product demand prediction"""
        print("\n" + "=" * 80)
        print("🛍️  TRAINING PRODUCT DEMAND PREDICTION MODELS")
        print("=" * 80)
        
        product_data = self.prepare_product_features()
        
        # Validate we have enough data
        if len(product_data) == 0:
            raise ValueError("❌ Insufficient data for product demand prediction. Need at least some products with historical sales.")
        
        if len(product_data) < 10:
            raise ValueError(f"❌ Insufficient data: Only {len(product_data)} records available. Need at least 10 records for training.")
        
        feature_columns = [
            'Product_Encoded', 'Category_Encoded',
            'Day', 'Month', 'Year', 'DayOfWeek', 'Quarter',
            'Quantity_Lag1', 'Quantity_Lag7', 'Quantity_Lag30',
            'Quantity_MA7', 'Quantity_MA30',
            'Unit_Price'
        ]
        
        # Ensure all feature columns exist and have no NaN
        for col in feature_columns:
            if col not in product_data.columns:
                raise ValueError(f"❌ Missing required column: {col}")
            if product_data[col].isna().any():
                product_data[col] = product_data[col].fillna(product_data[col].mean())
        
        X = product_data[feature_columns]
        y = product_data['Quantity']
        
        # Validate X and y
        if len(X) == 0 or len(y) == 0:
            raise ValueError("❌ Empty dataset after feature preparation.")
        
        # Split data (80-20)
        split_idx = int(len(X) * 0.8)
        
        # Ensure we have at least some training data
        if split_idx < 1:
            split_idx = max(1, len(X) - 1)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Final validation
        if len(X_train) == 0:
            raise ValueError("❌ No training data available. Need more historical data.")
        
        print(f"\n📊 Training Data: {len(X_train)} records")
        print(f"📊 Test Data: {len(X_test)} records")
        
        # Random Forest Model
        print("\n1️⃣ Training Random Forest for Product Demand...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        self.models['Random Forest'] = rf_model
        self.feature_columns = feature_columns
        
        # XGBoost Model
        print("2️⃣ Training XGBoost for Product Demand...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        self.models['XGBoost'] = xgb_model
        
        # Evaluate
        rf_pred = rf_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)
        
        rf_mae = np.mean(np.abs(y_test - rf_pred))
        xgb_mae = np.mean(np.abs(y_test - xgb_pred))
        
        print(f"\n📊 Random Forest MAE: {rf_mae:.2f} units")
        print(f"📊 XGBoost MAE: {xgb_mae:.2f} units")
        
        self.product_data = product_data
        self.X_test = X_test
        self.y_test = y_test
        
        return self.models
    
    def predict_product_demand_next_week(self, model_name='XGBoost'):
        """Predict demand for each product in next week"""
        if not hasattr(self, 'models') or len(self.models) == 0:
            self.train_product_models()
        
        if model_name not in self.models:
            model_name = list(self.models.keys())[0]
        
        model = self.models[model_name]
        product_data = self.product_data.copy()
        
        # Get unique products
        products = product_data['Product'].unique()
        
        # Get last date
        last_date = product_data['Date'].max()
        
        predictions = []
        
        for product in products:
            product_history = product_data[product_data['Product'] == product].sort_values('Date')
            
            if len(product_history) == 0:
                continue
            
            # Predict for next 7 days
            for day_offset in range(1, 8):
                next_date = last_date + pd.Timedelta(days=day_offset)
                
                # Get recent history
                recent = product_history.tail(30)
                
                if len(recent) == 0:
                    continue
                
                # Prepare features
                features = pd.DataFrame({
                    'Product_Encoded': [self.product_encoder.transform([product])[0]],
                    'Category_Encoded': [self.category_encoder.transform([product_history.iloc[-1]['Category']])[0]],
                    'Day': [next_date.day],
                    'Month': [next_date.month],
                    'Year': [next_date.year],
                    'DayOfWeek': [next_date.dayofweek],
                    'Quarter': [next_date.quarter],
                    'Quantity_Lag1': [recent['Quantity'].iloc[-1] if len(recent) > 0 else recent['Quantity'].mean()],
                    'Quantity_Lag7': [recent['Quantity'].iloc[-7] if len(recent) >= 7 else recent['Quantity'].mean()],
                    'Quantity_Lag30': [recent['Quantity'].iloc[-30] if len(recent) >= 30 else recent['Quantity'].mean()],
                    'Quantity_MA7': [recent['Quantity'].tail(7).mean()],
                    'Quantity_MA30': [recent['Quantity'].mean()],
                    'Unit_Price': [recent['Unit_Price'].iloc[-1]]
                })
                
                features = features[self.feature_columns]
                pred_quantity = max(0, model.predict(features)[0])
                
                predictions.append({
                    'Product': product,
                    'Date': next_date,
                    'Predicted_Demand': pred_quantity
                })
        
        pred_df = pd.DataFrame(predictions)
        
        # Aggregate by product
        product_summary = pred_df.groupby('Product')['Predicted_Demand'].sum().sort_values(ascending=False)
        
        return pred_df, product_summary
    
    def get_demand_insights(self, silent=False):
        """Generate demand insights and recommendations"""
        if not silent:
            print("\n" + "=" * 80)
            print("📊 PRODUCT DEMAND PREDICTIONS & INSIGHTS")
            print("=" * 80)
        
        pred_df, product_summary = self.predict_product_demand_next_week()
        
        if not silent:
            print("\n🔼 Products with INCREASING Demand (Top 10):")
            print(product_summary.head(10).to_string())
            
            print("\n🔽 Products with DECREASING Demand (Bottom 10):")
            print(product_summary.tail(10).to_string())
            
            # Calculate average daily demand
            avg_daily_demand = pred_df.groupby('Product')['Predicted_Demand'].mean().sort_values(ascending=False)
            
            print("\n📈 Average Daily Demand (Next Week):")
            print(avg_daily_demand.head(10).to_string())
        
        # Identify products needing restock
        current_stock_estimate = self.df.groupby('Product')['Quantity'].sum() * 0.1  # Assume 10% of total sales is current stock
        weekly_demand = product_summary
        
        restock_needed = []
        for product in weekly_demand.index:
            if product in current_stock_estimate.index:
                current = current_stock_estimate[product]
                predicted = weekly_demand[product]
                if predicted > current * 0.5:  # If predicted demand > 50% of current stock
                    restock_needed.append({
                        'Product': product,
                        'Current_Stock_Estimate': current,
                        'Predicted_Weekly_Demand': predicted,
                        'Urgency': 'HIGH' if predicted > current else 'MEDIUM'
                    })
        
        restock_df = pd.DataFrame(restock_needed).sort_values('Predicted_Weekly_Demand', ascending=False)
        
        if not silent:
            if len(restock_df) > 0:
                print("\n⚠️  PRODUCTS REQUIRING RESTOCK:")
                print(restock_df.to_string(index=False))
            else:
                print("\n✅ No immediate restocking needed based on predictions")
        
        # Store results for reuse
        self.last_insights = {
            'predictions': pred_df,
            'product_summary': product_summary,
            'restock_needed': restock_df
        }
        
        return self.last_insights
