"""
AI-Based Business Insights & Q&A System
Generates human-like business insights and answers questions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class AIBusinessInsights:
    """AI-powered business insights and Q&A system"""
    
    def __init__(self, df, eda_results, ml_results, demand_results):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.eda_results = eda_results
        self.ml_results = ml_results
        self.demand_results = demand_results
        self.insights_cache = {}
    
    def analyze_sales_drop(self, month_name=None):
        """Analyze why sales dropped in a specific month"""
        monthly_sales = self.df.groupby(['Year', 'Month'])['Revenue'].sum()
        
        if month_name:
            # Find the month
            month_num = self._month_name_to_num(month_name)
            if month_num:
                month_data = monthly_sales[monthly_sales.index.get_level_values(1) == month_num]
                if len(month_data) > 0:
                    target_month = month_data.index[0]
                    target_revenue = month_data.iloc[0]
                    
                    # Compare with previous month
                    prev_month_revenue = self._get_previous_month_revenue(target_month, monthly_sales)
                    if prev_month_revenue:
                        drop_pct = ((prev_month_revenue - target_revenue) / prev_month_revenue) * 100
                        
                        # Analyze reasons
                        reasons = []
                        if drop_pct > 10:
                            reasons.append(f"Significant revenue drop of {drop_pct:.1f}% compared to previous month")
                        
                        # Check category performance
                        month_df = self.df[
                            (self.df['Date'].dt.year == target_month[0]) &
                            (self.df['Date'].dt.month == target_month[1])
                        ]
                        category_perf = month_df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
                        
                        if len(category_perf) > 0:
                            worst_category = category_perf.index[-1]
                            reasons.append(f"Lowest performing category: {worst_category} (${category_perf[worst_category]:,.2f})")
                        
                        # Check day of week pattern
                        day_perf = month_df.groupby('Day_of_Week')['Revenue'].sum()
                        if len(day_perf) > 0:
                            worst_day = day_perf.idxmin()
                            reasons.append(f"Weakest sales day: {worst_day}")
                        
                        # Check transaction count
                        prev_month_trans = self._get_previous_month_transactions(target_month)
                        current_trans = len(month_df)
                        if prev_month_trans and current_trans < prev_month_trans * 0.9:
                            reasons.append(f"Lower transaction volume: {current_trans} vs {prev_month_trans} in previous month")
                        
                        explanation = f"""
🔍 ANALYSIS: Sales Drop in {month_name} {target_month[0]}

📉 Revenue Performance:
   • Current Month: ${target_revenue:,.2f}
   • Previous Month: ${prev_month_revenue:,.2f}
   • Drop: {drop_pct:.1f}%

💡 Key Reasons Identified:
{chr(10).join(f'   {i+1}. {reason}' for i, reason in enumerate(reasons))}

🎯 Recommendations:
   1. Review marketing campaigns for underperforming categories
   2. Increase promotional activities on weak sales days
   3. Analyze customer behavior patterns during this period
   4. Consider seasonal factors affecting demand
                        """
                        return explanation
        
        return "Could not analyze sales drop. Please specify a valid month name."
    
    def recommend_promotion_product(self):
        """Recommend which product to promote next"""
        # Analyze product performance
        product_stats = self.df.groupby('Product').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Date': 'count'
        }).rename(columns={'Date': 'Transactions'})
        
        # Calculate metrics
        product_stats['Avg_Price'] = product_stats['Revenue'] / product_stats['Quantity']
        product_stats['Revenue_Per_Transaction'] = product_stats['Revenue'] / product_stats['Transactions']
        
        # Recent performance (last 30 days)
        recent_date = self.df['Date'].max() - timedelta(days=30)
        recent_df = self.df[self.df['Date'] >= recent_date]
        recent_product = recent_df.groupby('Product')['Revenue'].sum()
        
        # Products with declining trend but good historical performance
        product_stats['Recent_Revenue'] = product_stats.index.map(
            lambda x: recent_product.get(x, 0)
        )
        product_stats['Recent_Share'] = product_stats['Recent_Revenue'] / product_stats['Revenue']
        
        # Find products with good potential but declining recent performance
        candidates = product_stats[
            (product_stats['Revenue'] > product_stats['Revenue'].quantile(0.5)) &
            (product_stats['Recent_Share'] < 0.8) &
            (product_stats['Quantity'] > product_stats['Quantity'].quantile(0.3))
        ].sort_values('Revenue', ascending=False)
        
        if len(candidates) > 0:
            top_candidate = candidates.index[0]
            candidate_data = candidates.loc[top_candidate]
            
            explanation = f"""
🎯 PRODUCT PROMOTION RECOMMENDATION

📦 Recommended Product: {top_candidate}

📊 Performance Metrics:
   • Total Revenue: ${candidate_data['Revenue']:,.2f}
   • Total Quantity Sold: {int(candidate_data['Quantity']):,}
   • Average Price: ${candidate_data['Avg_Price']:.2f}
   • Recent Performance: {candidate_data['Recent_Share']*100:.1f}% of historical average

💡 Why Promote This Product:
   1. Strong historical performance indicates market demand
   2. Recent decline suggests promotion could boost sales
   3. Good price point for promotional campaigns
   4. Established customer base and recognition

🎨 Promotion Strategy:
   • Offer 15-20% discount to stimulate demand
   • Create bundle deals with complementary products
   • Highlight in-store displays and marketing materials
   • Run social media campaigns featuring this product
            """
            return explanation
        
        # Fallback: recommend top product
        top_product = product_stats.sort_values('Revenue', ascending=False).index[0]
        return f"Recommended product for promotion: {top_product} (Top revenue generator)"
    
    def analyze_low_revenue_month(self, month_name=None, year=None):
        """Analyze reasons for low revenue in a month"""
        monthly_sales = self.df.groupby(['Year', 'Month'])['Revenue'].sum().sort_values()
        
        if month_name and year:
            month_num = self._month_name_to_num(month_name)
            if month_num:
                target_revenue = monthly_sales.get((year, month_num), None)
                if target_revenue:
                    avg_revenue = monthly_sales.mean()
                    pct_below_avg = ((avg_revenue - target_revenue) / avg_revenue) * 100
                    
                    # Analyze the month
                    month_df = self.df[
                        (self.df['Date'].dt.year == year) &
                        (self.df['Date'].dt.month == month_num)
                    ]
                    
                    reasons = []
                    if pct_below_avg > 15:
                        reasons.append(f"Revenue {pct_below_avg:.1f}% below average")
                    
                    # Check product diversity
                    unique_products = month_df['Product'].nunique()
                    avg_unique = self.df.groupby([self.df['Date'].dt.year, self.df['Date'].dt.month])['Product'].nunique().mean()
                    if unique_products < avg_unique * 0.9:
                        reasons.append(f"Lower product diversity: {unique_products} vs {avg_unique:.0f} average")
                    
                    # Check top products performance
                    top_products = month_df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(3)
                    if len(top_products) > 0:
                        reasons.append(f"Top product: {top_products.index[0]} (${top_products.iloc[0]:,.2f})")
                    
                    explanation = f"""
🔍 LOW REVENUE ANALYSIS: {month_name} {year}

📊 Revenue Performance:
   • Month Revenue: ${target_revenue:,.2f}
   • Average Monthly Revenue: ${avg_revenue:,.2f}
   • Below Average: {pct_below_avg:.1f}%

💡 Identified Factors:
{chr(10).join(f'   {i+1}. {reason}' for i, reason in enumerate(reasons))}

🎯 Action Items:
   1. Review and optimize product mix
   2. Increase marketing efforts for top-performing products
   3. Analyze external factors (holidays, events, competition)
   4. Consider promotional campaigns to boost sales
                    """
                    return explanation
        
        # Analyze lowest month overall
        lowest_month = monthly_sales.index[0]
        return self.analyze_low_revenue_month(
            self._num_to_month_name(lowest_month[1]),
            lowest_month[0]
        )
    
    def answer_question(self, question):
        """Answer business questions using AI logic"""
        question_lower = question.lower()
        
        # Sales drop analysis
        if 'sales drop' in question_lower or 'sales decline' in question_lower:
            month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', question_lower)
            if month_match:
                return self.analyze_sales_drop(month_match.group(1).capitalize())
            return self.analyze_sales_drop('March')  # Default
        
        # Product promotion
        if 'promote' in question_lower or 'promotion' in question_lower or 'which product' in question_lower:
            return self.recommend_promotion_product()
        
        # Low revenue
        if 'low revenue' in question_lower or 'revenue low' in question_lower:
            # Try to extract month and year
            month_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', question_lower)
            year_match = re.search(r'\b(202[2-4])\b', question_lower)
            
            month = month_match.group(1).capitalize() if month_match else None
            year = int(year_match.group(1)) if year_match else None
            
            if month and year:
                return self.analyze_low_revenue_month(month, year)
            return self.analyze_low_revenue_month()
        
        # General insights
        if 'insight' in question_lower or 'summary' in question_lower or 'overview' in question_lower:
            return self.generate_general_insights()
        
        # Default response
        return """
🤖 AI Business Assistant Response:

I can help you with:
• Analyzing sales drops in specific months
• Recommending products for promotion
• Explaining low revenue periods
• General business insights

Please rephrase your question or ask about one of these topics.
        """
    
    def generate_general_insights(self):
        """Generate general business insights"""
        total_revenue = self.df['Revenue'].sum()
        avg_daily = self.df.groupby(self.df['Date'].dt.date)['Revenue'].sum().mean()
        top_product = self.df.groupby('Product')['Revenue'].sum().idxmax()
        top_category = self.df.groupby('Category')['Revenue'].sum().idxmax()
        peak_day = self.df.groupby('Day_of_Week')['Revenue'].sum().idxmax()
        
        return f"""
📊 GENERAL BUSINESS INSIGHTS

💰 Financial Overview:
   • Total Revenue: ${total_revenue:,.2f}
   • Average Daily Revenue: ${avg_daily:,.2f}

🏆 Top Performers:
   • Best Product: {top_product}
   • Best Category: {top_category}
   • Peak Sales Day: {peak_day}

💡 Key Observations:
   • Business shows consistent performance patterns
   • Seasonal trends impact sales volume
   • Product mix diversity supports revenue stability

🎯 Strategic Recommendations:
   1. Focus marketing on top-performing products
   2. Optimize inventory for peak sales days
   3. Expand successful category offerings
   4. Leverage seasonal patterns for planning
        """
    
    def _month_name_to_num(self, month_name):
        """Convert month name to number"""
        months = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        return months.get(month_name, None)
    
    def _num_to_month_name(self, month_num):
        """Convert month number to name"""
        months = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        return months.get(month_num, 'Unknown')
    
    def _get_previous_month_revenue(self, current_month, monthly_sales):
        """Get previous month revenue"""
        year, month = current_month
        if month == 1:
            prev_month = (year - 1, 12)
        else:
            prev_month = (year, month - 1)
        
        return monthly_sales.get(prev_month, None)
    
    def _get_previous_month_transactions(self, current_month):
        """Get previous month transaction count"""
        year, month = current_month
        if month == 1:
            prev_month = (year - 1, 12)
        else:
            prev_month = (year, month - 1)
        
        month_df = self.df[
            (self.df['Date'].dt.year == prev_month[0]) &
            (self.df['Date'].dt.month == prev_month[1])
        ]
        return len(month_df)
