"""
Exploratory Data Analysis (EDA) Module
Comprehensive data analysis for retail sales
"""

import pandas as pd
import numpy as np
from datetime import datetime

class SalesEDA:
    """Perform comprehensive exploratory data analysis on sales data"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.insights = []
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        print("=" * 80)
        print("📊 BASIC STATISTICAL SUMMARY")
        print("=" * 80)
        
        total_revenue = self.df['Revenue'].sum()
        total_transactions = len(self.df)
        avg_transaction_value = self.df['Revenue'].mean()
        total_products_sold = self.df['Quantity'].sum()
        
        print(f"\n💰 Total Revenue: ${total_revenue:,.2f}")
        print(f"📦 Total Transactions: {total_transactions:,}")
        print(f"💵 Average Transaction Value: ${avg_transaction_value:.2f}")
        print(f"🛍️  Total Products Sold: {total_products_sold:,}")
        print(f"📅 Date Range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        print(f"🏷️  Unique Products: {self.df['Product'].nunique()}")
        print(f"📂 Categories: {self.df['Category'].nunique()}")
        
        self.insights.append(f"Total revenue generated: ${total_revenue:,.2f}")
        self.insights.append(f"Average transaction value: ${avg_transaction_value:.2f}")
        
        return {
            'total_revenue': total_revenue,
            'total_transactions': total_transactions,
            'avg_transaction_value': avg_transaction_value,
            'total_products_sold': total_products_sold
        }
    
    def daily_sales_analysis(self):
        """Analyze daily sales patterns"""
        print("\n" + "=" * 80)
        print("📅 DAILY SALES ANALYSIS")
        print("=" * 80)
        
        daily_sales = self.df.groupby(self.df['Date'].dt.date).agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Transactions'})
        
        print(f"\n📈 Average Daily Revenue: ${daily_sales['Revenue'].mean():,.2f}")
        print(f"📊 Average Daily Transactions: {daily_sales['Transactions'].mean():.1f}")
        print(f"📉 Best Day Revenue: ${daily_sales['Revenue'].max():,.2f} on {daily_sales['Revenue'].idxmax()}")
        print(f"📉 Worst Day Revenue: ${daily_sales['Revenue'].min():,.2f} on {daily_sales['Revenue'].idxmin()}")
        
        return daily_sales
    
    def monthly_sales_analysis(self):
        """Analyze monthly sales trends"""
        print("\n" + "=" * 80)
        print("📆 MONTHLY SALES ANALYSIS")
        print("=" * 80)
        
        monthly_sales = self.df.groupby(['Year', 'Month']).agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Transactions'})
        
        # Calculate month-over-month growth
        monthly_sales['MoM_Growth'] = monthly_sales['Revenue'].pct_change() * 100
        
        print("\n📊 Monthly Revenue Summary:")
        print(monthly_sales[['Revenue', 'Transactions', 'MoM_Growth']].to_string())
        
        best_month = monthly_sales['Revenue'].idxmax()
        worst_month = monthly_sales['Revenue'].idxmin()
        
        print(f"\n🏆 Best Month: {best_month[1]} {best_month[0]} - ${monthly_sales.loc[best_month, 'Revenue']:,.2f}")
        print(f"📉 Worst Month: {worst_month[1]} {worst_month[0]} - ${monthly_sales.loc[worst_month, 'Revenue']:,.2f}")
        
        self.insights.append(f"Best performing month: {best_month[1]} {best_month[0]}")
        self.insights.append(f"Worst performing month: {worst_month[1]} {worst_month[0]}")
        
        return monthly_sales
    
    def yearly_sales_analysis(self):
        """Analyze yearly sales trends"""
        print("\n" + "=" * 80)
        print("📅 YEARLY SALES ANALYSIS")
        print("=" * 80)
        
        yearly_sales = self.df.groupby('Year').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Transactions'})
        
        yearly_sales['YoY_Growth'] = yearly_sales['Revenue'].pct_change() * 100
        
        print("\n📊 Yearly Revenue Summary:")
        print(yearly_sales.to_string())
        
        if len(yearly_sales) > 1:
            print(f"\n📈 Year-over-Year Growth: {yearly_sales['YoY_Growth'].iloc[-1]:.2f}%")
            self.insights.append(f"Year-over-year growth: {yearly_sales['YoY_Growth'].iloc[-1]:.2f}%")
        
        return yearly_sales
    
    def top_products_analysis(self, top_n=10):
        """Identify top and least selling products"""
        print("\n" + "=" * 80)
        print(f"🏆 TOP {top_n} PRODUCTS ANALYSIS")
        print("=" * 80)
        
        product_stats = self.df.groupby('Product').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Transactions'})
        
        product_stats = product_stats.sort_values('Revenue', ascending=False)
        
        print("\n📊 Top Products by Revenue:")
        print(product_stats.head(top_n).to_string())
        
        print("\n📉 Bottom Products by Revenue:")
        print(product_stats.tail(top_n).to_string())
        
        top_product = product_stats.index[0]
        bottom_product = product_stats.index[-1]
        
        self.insights.append(f"Top-selling product: {top_product} (${product_stats.loc[top_product, 'Revenue']:,.2f})")
        self.insights.append(f"Least-selling product: {bottom_product} (${product_stats.loc[bottom_product, 'Revenue']:,.2f})")
        
        return product_stats
    
    def category_analysis(self):
        """Analyze sales by category"""
        print("\n" + "=" * 80)
        print("📂 CATEGORY-WISE SALES ANALYSIS")
        print("=" * 80)
        
        category_stats = self.df.groupby('Category').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Transactions'})
        
        category_stats['Revenue_Share'] = (category_stats['Revenue'] / category_stats['Revenue'].sum() * 100).round(2)
        category_stats = category_stats.sort_values('Revenue', ascending=False)
        
        print("\n📊 Category Performance:")
        print(category_stats.to_string())
        
        top_category = category_stats.index[0]
        self.insights.append(f"Top category: {top_category} ({category_stats.loc[top_category, 'Revenue_Share']}% of revenue)")
        
        return category_stats
    
    def peak_hours_analysis(self):
        """Identify peak sales hours"""
        print("\n" + "=" * 80)
        print("⏰ PEAK HOURS ANALYSIS")
        print("=" * 80)
        
        hourly_sales = self.df.groupby('Hour').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Transactions'})
        
        hourly_sales = hourly_sales.sort_values('Revenue', ascending=False)
        
        print("\n📊 Top 5 Peak Hours:")
        print(hourly_sales.head(5).to_string())
        
        peak_hour = hourly_sales.index[0]
        self.insights.append(f"Peak sales hour: {peak_hour}:00 (${hourly_sales.loc[peak_hour, 'Revenue']:,.2f})")
        
        return hourly_sales
    
    def peak_days_analysis(self):
        """Identify peak sales days of week"""
        print("\n" + "=" * 80)
        print("📅 PEAK DAYS OF WEEK ANALYSIS")
        print("=" * 80)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_sales = self.df.groupby('Day_of_Week').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Transactions'})
        
        day_sales = day_sales.reindex(day_order)
        
        print("\n📊 Sales by Day of Week:")
        print(day_sales.to_string())
        
        peak_day = day_sales['Revenue'].idxmax()
        self.insights.append(f"Peak sales day: {peak_day} (${day_sales.loc[peak_day, 'Revenue']:,.2f})")
        
        return day_sales
    
    def seasonal_analysis(self):
        """Analyze seasonal patterns"""
        print("\n" + "=" * 80)
        print("🌍 SEASONAL PATTERNS ANALYSIS")
        print("=" * 80)
        
        # By month
        monthly_pattern = self.df.groupby('Month').agg({
            'Revenue': 'sum'
        })
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_pattern = monthly_pattern.reindex(month_order)
        
        print("\n📊 Revenue by Month:")
        print(monthly_pattern.to_string())
        
        # By quarter
        quarterly_pattern = self.df.groupby('Quarter').agg({
            'Revenue': 'sum'
        })
        
        print("\n📊 Revenue by Quarter:")
        print(quarterly_pattern.to_string())
        
        best_quarter = quarterly_pattern['Revenue'].idxmax()
        self.insights.append(f"Best quarter: {best_quarter} (${quarterly_pattern.loc[best_quarter, 'Revenue']:,.2f})")
        
        return monthly_pattern, quarterly_pattern
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "=" * 80)
        print("💡 KEY INSIGHTS SUMMARY")
        print("=" * 80)
        
        for i, insight in enumerate(self.insights, 1):
            print(f"{i}. {insight}")
        
        return self.insights
    
    def run_full_analysis(self):
        """Run complete EDA analysis"""
        print("\n" + "🔍" * 40)
        print("EXPLORATORY DATA ANALYSIS - COMPLETE REPORT")
        print("🔍" * 40)
        
        stats = self.basic_statistics()
        daily = self.daily_sales_analysis()
        monthly = self.monthly_sales_analysis()
        yearly = self.yearly_sales_analysis()
        top_products = self.top_products_analysis()
        categories = self.category_analysis()
        peak_hours = self.peak_hours_analysis()
        peak_days = self.peak_days_analysis()
        seasonal = self.seasonal_analysis()
        insights = self.generate_insights_report()
        
        return {
            'stats': stats,
            'daily': daily,
            'monthly': monthly,
            'yearly': yearly,
            'top_products': top_products,
            'categories': categories,
            'peak_hours': peak_hours,
            'peak_days': peak_days,
            'seasonal': seasonal,
            'insights': insights
        }
