"""
Data Visualization Module
Professional-quality visualizations for retail sales analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Set style for professional visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

class SalesVisualizer:
    """Create professional data visualizations"""
    
    def __init__(self, df, save_path='visualizations/'):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.save_path = save_path
        import os
        os.makedirs(save_path, exist_ok=True)
    
    def plot_sales_trend_over_time(self, figsize=(14, 6)):
        """Line chart: Sales trends over time"""
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Daily sales trend
        daily_sales = self.df.groupby(self.df['Date'].dt.date)['Revenue'].sum()
        axes[0].plot(daily_sales.index, daily_sales.values, linewidth=2, color='#2E86AB')
        axes[0].set_title('Daily Sales Revenue Trend', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Revenue ($)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Monthly sales trend
        monthly_sales = self.df.groupby([self.df['Date'].dt.to_period('M')])['Revenue'].sum()
        monthly_sales.index = monthly_sales.index.astype(str)
        axes[1].plot(range(len(monthly_sales)), monthly_sales.values, 
                     marker='o', linewidth=2, markersize=6, color='#A23B72')
        axes[1].set_title('Monthly Sales Revenue Trend', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Month', fontsize=12)
        axes[1].set_ylabel('Revenue ($)', fontsize=12)
        axes[1].set_xticks(range(len(monthly_sales)))
        axes[1].set_xticklabels(monthly_sales.index, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}sales_trend_over_time.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: sales_trend_over_time.png")
        plt.close()
    
    def plot_top_products(self, top_n=10, figsize=(12, 8)):
        """Bar chart: Top selling products"""
        product_revenue = self.df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Top products by revenue
        colors = plt.cm.viridis(np.linspace(0, 1, len(product_revenue)))
        axes[0].barh(range(len(product_revenue)), product_revenue.values, color=colors)
        axes[0].set_yticks(range(len(product_revenue)))
        axes[0].set_yticklabels(product_revenue.index)
        axes[0].set_xlabel('Revenue ($)', fontsize=12)
        axes[0].set_title(f'Top {top_n} Products by Revenue', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(product_revenue.values):
            axes[0].text(v, i, f'${v:,.0f}', va='center', fontsize=9)
        
        # Top products by quantity
        product_quantity = self.df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(top_n)
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(product_quantity)))
        axes[1].barh(range(len(product_quantity)), product_quantity.values, color=colors2)
        axes[1].set_yticks(range(len(product_quantity)))
        axes[1].set_yticklabels(product_quantity.index)
        axes[1].set_xlabel('Quantity Sold', fontsize=12)
        axes[1].set_title(f'Top {top_n} Products by Quantity', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(product_quantity.values):
            axes[1].text(v, i, f'{int(v)}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}top_products.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: top_products.png")
        plt.close()
    
    def plot_category_contribution(self, figsize=(12, 6)):
        """Pie chart: Category-wise revenue contribution"""
        category_revenue = self.df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_revenue)))
        wedges, texts, autotexts = axes[0].pie(category_revenue.values, 
                                                labels=category_revenue.index,
                                                autopct='%1.1f%%',
                                                colors=colors,
                                                startangle=90)
        axes[0].set_title('Category-wise Revenue Contribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        axes[1].bar(range(len(category_revenue)), category_revenue.values, color=colors)
        axes[1].set_xticks(range(len(category_revenue)))
        axes[1].set_xticklabels(category_revenue.index, rotation=45, ha='right')
        axes[1].set_ylabel('Revenue ($)', fontsize=12)
        axes[1].set_title('Category Revenue Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(category_revenue.values):
            axes[1].text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}category_contribution.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: category_contribution.png")
        plt.close()
    
    def plot_sales_heatmap(self, figsize=(14, 8)):
        """Heatmap: Sales vs days of week / hours"""
        # Create pivot table: Day of Week vs Hour
        self.df['Day_of_Week_Num'] = pd.to_datetime(self.df['Date']).dt.dayofweek
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        self.df['Day_Name'] = self.df['Day_of_Week_Num'].map(dict(zip(range(7), day_names)))
        
        heatmap_data = self.df.pivot_table(
            values='Revenue',
            index='Day_Name',
            columns='Hour',
            aggfunc='sum'
        )
        
        # Reorder days
        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Heatmap 1: Revenue by Day and Hour
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Revenue ($)'}, ax=axes[0])
        axes[0].set_title('Sales Heatmap: Revenue by Day of Week & Hour', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Hour of Day', fontsize=12)
        axes[0].set_ylabel('Day of Week', fontsize=12)
        
        # Heatmap 2: Transaction count
        heatmap_data2 = self.df.pivot_table(
            values='Product',
            index='Day_Name',
            columns='Hour',
            aggfunc='count'
        )
        heatmap_data2 = heatmap_data2.reindex(day_order)
        
        sns.heatmap(heatmap_data2, annot=True, fmt='.0f', cmap='Blues',
                   cbar_kws={'label': 'Transaction Count'}, ax=axes[1])
        axes[1].set_title('Sales Heatmap: Transactions by Day of Week & Hour',
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Hour of Day', fontsize=12)
        axes[1].set_ylabel('Day of Week', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}sales_heatmap.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: sales_heatmap.png")
        plt.close()
    
    def plot_monthly_comparison(self, figsize=(14, 6)):
        """Compare sales across months"""
        monthly_data = self.df.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
        monthly_data['Year_Month'] = monthly_data['Year'].astype(str) + '-' + monthly_data['Month']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(monthly_data)))
        bars = ax.bar(range(len(monthly_data)), monthly_data['Revenue'].values, color=colors)
        
        ax.set_xticks(range(len(monthly_data)))
        ax.set_xticklabels(monthly_data['Year_Month'].values, rotation=45, ha='right')
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Monthly Revenue Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(monthly_data['Revenue'].values):
            ax.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}monthly_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: monthly_comparison.png")
        plt.close()
    
    def plot_revenue_growth_trend(self, figsize=(14, 6)):
        """Plot revenue growth trends"""
        monthly_sales = self.df.groupby([self.df['Date'].dt.to_period('M')])['Revenue'].sum()
        monthly_sales.index = monthly_sales.index.astype(str)
        
        # Calculate growth rate
        growth_rate = monthly_sales.pct_change() * 100
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Revenue trend
        axes[0].plot(range(len(monthly_sales)), monthly_sales.values, 
                    marker='o', linewidth=2, markersize=6, color='#2E86AB', label='Revenue')
        axes[0].set_ylabel('Revenue ($)', fontsize=12)
        axes[0].set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Growth rate
        colors = ['green' if x > 0 else 'red' for x in growth_rate.values[1:]]
        axes[1].bar(range(1, len(growth_rate)), growth_rate.values[1:], color=colors, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].set_ylabel('Growth Rate (%)', fontsize=12)
        axes[1].set_xlabel('Month', fontsize=12)
        axes[1].set_title('Month-over-Month Growth Rate', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(monthly_sales)))
        axes[1].set_xticklabels(monthly_sales.index, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}revenue_growth_trend.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: revenue_growth_trend.png")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "=" * 80)
        print("📊 GENERATING PROFESSIONAL VISUALIZATIONS")
        print("=" * 80)
        
        self.plot_sales_trend_over_time()
        self.plot_top_products()
        self.plot_category_contribution()
        self.plot_sales_heatmap()
        self.plot_monthly_comparison()
        self.plot_revenue_growth_trend()
        
        print("\n✅ All visualizations generated successfully!")
        print(f"📁 Saved in: {self.save_path}")
