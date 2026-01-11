"""
Smart AI Recommendations System
Generates automatic, data-driven business recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SmartRecommendations:
    """Generate AI-powered business recommendations"""
    
    def __init__(self, df, ml_forecaster, demand_predictor):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.ml_forecaster = ml_forecaster
        self.demand_predictor = demand_predictor
        self.recommendations = []
    
    def generate_all_recommendations(self):
        """Generate comprehensive recommendations"""
        print("\n" + "=" * 80)
        print("🤖 GENERATING AI-POWERED BUSINESS RECOMMENDATIONS")
        print("=" * 80)
        
        self.recommendations = []
        
        # 1. Inventory recommendations
        self._inventory_recommendations()
        
        # 2. Pricing recommendations
        self._pricing_recommendations()
        
        # 3. Staffing recommendations
        self._staffing_recommendations()
        
        # 4. Marketing recommendations
        self._marketing_recommendations()
        
        # 5. Product mix recommendations
        self._product_mix_recommendations()
        
        return self.recommendations
    
    def _inventory_recommendations(self):
        """Generate inventory management recommendations"""
        # Get demand predictions (silent mode to avoid duplicate output)
        demand_insights = {}
        if self.demand_predictor is not None:
            if hasattr(self.demand_predictor, 'last_insights'):
                demand_insights = self.demand_predictor.last_insights
            elif hasattr(self.demand_predictor, 'get_demand_insights'):
                try:
                    demand_insights = self.demand_predictor.get_demand_insights(silent=True)
                except:
                    demand_insights = {}
        
        if 'restock_needed' in demand_insights and len(demand_insights['restock_needed']) > 0:
                restock_df = demand_insights['restock_needed']
                
                for _, row in restock_df.head(5).iterrows():
                    self.recommendations.append({
                        'Category': 'Inventory Management',
                        'Priority': row['Urgency'],
                        'Recommendation': f"Increase stock of {row['Product']} due to rising demand",
                        'Justification': f"Predicted weekly demand ({row['Predicted_Weekly_Demand']:.0f} units) exceeds current stock estimate ({row['Current_Stock_Estimate']:.0f} units)",
                        'Action': f"Order additional {row['Product']} inventory immediately"
                    })
        
        # Identify slow-moving products
        product_sales = self.df.groupby('Product')['Quantity'].sum().sort_values()
        slow_moving = product_sales.head(5)
        
        for product, qty in slow_moving.items():
            self.recommendations.append({
                'Category': 'Inventory Management',
                'Priority': 'LOW',
                'Recommendation': f"Reduce inventory for {product} due to low sales",
                'Justification': f"Only {qty} units sold in the entire period, indicating low demand",
                'Action': f"Consider discontinuing or reducing stock levels for {product}"
            })
    
    def _pricing_recommendations(self):
        """Generate pricing strategy recommendations"""
        # Analyze products with declining sales
        recent_date = self.df['Date'].max() - timedelta(days=30)
        recent_sales = self.df[self.df['Date'] >= recent_date].groupby('Product')['Revenue'].sum()
        historical_sales = self.df.groupby('Product')['Revenue'].sum()
        
        for product in recent_sales.index:
            if product in historical_sales.index:
                recent_share = recent_sales[product] / historical_sales[product]
                if recent_share < 0.7 and historical_sales[product] > historical_sales.quantile(0.5):
                    # Good product with declining recent sales
                    self.recommendations.append({
                        'Category': 'Pricing Strategy',
                        'Priority': 'MEDIUM',
                        'Recommendation': f"Offer discounts on {product} due to declining sales",
                        'Justification': f"Recent sales ({recent_share*100:.1f}% of historical) show declining trend despite good historical performance",
                        'Action': f"Implement 10-15% discount promotion for {product}"
                    })
                    break  # Limit to one pricing recommendation
    
    def _staffing_recommendations(self):
        """Generate staffing recommendations based on peak hours/days"""
        # Peak hours analysis
        hourly_transactions = self.df.groupby('Hour')['Product'].count().sort_values(ascending=False)
        peak_hours = hourly_transactions.head(3)
        
        # Peak days analysis
        day_transactions = self.df.groupby('Day_of_Week')['Product'].count().sort_values(ascending=False)
        peak_day = day_transactions.idxmax()
        
        peak_hour_str = ", ".join([f"{int(h)}:00" for h in peak_hours.index])
        
        self.recommendations.append({
            'Category': 'Staffing',
            'Priority': 'HIGH',
            'Recommendation': f"Increase staff during peak hours ({peak_hour_str}) and on {peak_day}s",
            'Justification': f"Peak sales occur at {peak_hour_str} with {int(peak_hours.iloc[0])} transactions. {peak_day} shows highest transaction volume.",
            'Action': f"Schedule additional staff during {peak_hour_str} and ensure full staffing on {peak_day}s"
        })
    
    def _marketing_recommendations(self):
        """Generate marketing recommendations"""
        # Top products by revenue
        top_products = self.df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(3)
        
        # Top categories
        top_categories = self.df.groupby('Category')['Revenue'].sum().sort_values(ascending=False).head(2)
        
        top_product = top_products.index[0]
        top_category = top_categories.index[0]
        
        self.recommendations.append({
            'Category': 'Marketing',
            'Priority': 'HIGH',
            'Recommendation': f"Promote {top_product} and {top_category} category in marketing campaigns",
            'Justification': f"{top_product} generates ${top_products.iloc[0]:,.2f} revenue. {top_category} is the top-performing category.",
            'Action': f"Create marketing campaigns highlighting {top_product} and expand {top_category} product line"
        })
    
    def _product_mix_recommendations(self):
        """Generate product mix optimization recommendations"""
        # Category performance
        category_revenue = self.df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
        category_share = (category_revenue / category_revenue.sum() * 100).round(2)
        
        # Identify underperforming categories
        avg_share = category_share.mean()
        underperforming = category_share[category_share < avg_share * 0.7]
        
        if len(underperforming) > 0:
            worst_category = underperforming.index[0]
            self.recommendations.append({
                'Category': 'Product Mix',
                'Priority': 'MEDIUM',
                'Recommendation': f"Review and optimize {worst_category} category offerings",
                'Justification': f"{worst_category} contributes only {category_share[worst_category]}% of revenue, below average",
                'Action': f"Analyze customer preferences and consider adding/removing products in {worst_category} category"
            })
    
    def display_recommendations(self):
        """Display all recommendations in a formatted way"""
        if not self.recommendations:
            self.generate_all_recommendations()
        
        print("\n" + "=" * 80)
        print("📋 AI RECOMMENDATIONS SUMMARY")
        print("=" * 80)
        
        # Group by category
        by_category = {}
        for rec in self.recommendations:
            cat = rec['Category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(rec)
        
        for category, recs in by_category.items():
            print(f"\n📂 {category}:")
            print("-" * 80)
            
            for i, rec in enumerate(recs, 1):
                priority_emoji = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }
                emoji = priority_emoji.get(rec['Priority'], '⚪')
                
                print(f"\n{emoji} Recommendation {i}: {rec['Recommendation']}")
                print(f"   💡 Justification: {rec['Justification']}")
                print(f"   ✅ Action: {rec['Action']}")
        
        return self.recommendations
    
    def get_recommendations_by_priority(self, priority='HIGH'):
        """Get recommendations filtered by priority"""
        filtered = [r for r in self.recommendations if r['Priority'] == priority]
        return filtered
    
    def export_recommendations(self, filename='ai_recommendations.csv'):
        """Export recommendations to CSV"""
        if not self.recommendations:
            self.generate_all_recommendations()
        
        rec_df = pd.DataFrame(self.recommendations)
        rec_df.to_csv(filename, index=False)
        print(f"\n✅ Recommendations exported to {filename}")
        return filename
