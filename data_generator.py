"""
Data Generator Module
Generates realistic retail sales data for analysis and ML modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SalesDataGenerator:
    """Generate realistic retail sales data with trends and patterns"""
    
    def __init__(self, start_date='2022-01-01', end_date='2024-12-31'):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Product categories and products
        self.categories = {
            'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Hat'],
            'Food & Beverages': ['Coffee', 'Snacks', 'Beverages', 'Candy', 'Bread'],
            'Home & Garden': ['Lamp', 'Plant', 'Tool', 'Furniture', 'Decor'],
            'Sports': ['Basketball', 'Tennis Racket', 'Yoga Mat', 'Dumbbells', 'Running Shoes']
        }
        
        # Base prices for products
        self.base_prices = {
            'Laptop': 800, 'Smartphone': 500, 'Tablet': 300, 'Headphones': 100, 'Camera': 400,
            'T-Shirt': 25, 'Jeans': 60, 'Jacket': 80, 'Shoes': 70, 'Hat': 20,
            'Coffee': 5, 'Snacks': 3, 'Beverages': 2, 'Candy': 1.5, 'Bread': 2.5,
            'Lamp': 50, 'Plant': 15, 'Tool': 30, 'Furniture': 200, 'Decor': 25,
            'Basketball': 30, 'Tennis Racket': 80, 'Yoga Mat': 25, 'Dumbbells': 50, 'Running Shoes': 90
        }
    
    def generate_sales_data(self, n_records=10000):
        """Generate comprehensive sales data"""
        records = []
        current_date = self.start_date
        
        # Generate date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Seasonal multipliers (higher sales in certain months)
        seasonal_multipliers = {
            1: 0.9, 2: 0.85, 3: 1.0, 4: 1.1, 5: 1.15, 6: 1.2,
            7: 1.1, 8: 1.0, 9: 1.05, 10: 1.15, 11: 1.3, 12: 1.4  # Dec and Nov peak
        }
        
        # Day of week multipliers (weekends higher)
        day_multipliers = {
            0: 0.8, 1: 0.9, 2: 0.95, 3: 1.0, 4: 1.05, 5: 1.3, 6: 1.4
        }
        
        # Generate records
        for date in date_range:
            # Base number of transactions per day
            n_transactions = int(np.random.normal(50, 15))
            n_transactions = max(10, min(100, n_transactions))  # Clamp between 10-100
            
            # Apply day of week multiplier
            day_mult = day_multipliers[date.weekday()]
            n_transactions = int(n_transactions * day_mult)
            
            # Apply seasonal multiplier
            month_mult = seasonal_multipliers[date.month]
            n_transactions = int(n_transactions * month_mult)
            
            # Generate transactions for this day
            for _ in range(n_transactions):
                # Select random category and product
                category = random.choice(list(self.categories.keys()))
                product = random.choice(self.categories[category])
                
                # Base price with some variation
                base_price = self.base_prices[product]
                price = base_price * np.random.uniform(0.85, 1.15)
                
                # Quantity (some products sell in multiples)
                if category == 'Food & Beverages':
                    quantity = np.random.randint(1, 5)
                else:
                    quantity = np.random.randint(1, 3)
                
                # Hour of day (peak hours 10-14, 17-20)
                hour_probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08,
                              0.1, 0.1, 0.1, 0.08, 0.06, 0.05, 0.04, 0.08, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01]
                # Normalize probabilities to sum to 1
                hour_probs = np.array(hour_probs)
                hour_probs = hour_probs / hour_probs.sum()
                hour = np.random.choice(list(range(24)), p=hour_probs)
                
                # Create transaction timestamp
                timestamp = date.replace(hour=hour, minute=np.random.randint(0, 60))
                
                # Calculate revenue
                revenue = price * quantity
                
                # Add some trend (gradual growth over time)
                days_from_start = (date - self.start_date).days
                trend_factor = 1 + (days_from_start / 1000) * 0.1  # 10% growth over period
                revenue *= trend_factor
                
                records.append({
                    'Date': timestamp,
                    'Product': product,
                    'Category': category,
                    'Quantity': quantity,
                    'Unit_Price': round(price, 2),
                    'Revenue': round(revenue, 2),
                    'Hour': hour,
                    'Day_of_Week': date.strftime('%A'),
                    'Month': date.strftime('%B'),
                    'Year': date.year,
                    'Quarter': f"Q{(date.month-1)//3 + 1}"
                })
        
        df = pd.DataFrame(records)
        return df
    
    def save_data(self, df, filename='sales_data.csv'):
        """Save generated data to CSV"""
        df.to_csv(filename, index=False)
        print(f"✅ Data saved to {filename}")
        print(f"📊 Total records: {len(df):,}")
        print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        return filename

if __name__ == "__main__":
    generator = SalesDataGenerator()
    df = generator.generate_sales_data()
    generator.save_data(df, 'sales_data.csv')
