"""
Main Execution Script
AI-Powered Sales Analytics & Forecasting System for Retail Shop
Complete end-to-end execution of all modules
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import all modules
from data_generator import SalesDataGenerator
from eda_module import SalesEDA
from visualization_module import SalesVisualizer
from ml_forecasting import SalesForecaster
from product_demand_prediction import ProductDemandPredictor
from ai_insights import AIBusinessInsights
from ai_recommendations import SmartRecommendations

class SalesAnalyticsSystem:
    """Main system orchestrator"""
    
    def __init__(self, data_file='sales_data.csv', generate_data=True):
        self.data_file = data_file
        self.df = None
        self.eda_results = None
        self.ml_results = None
        self.demand_results = None
        
        # Generate or load data
        if generate_data or not os.path.exists(data_file):
            print("📊 Generating sample sales data...")
            generator = SalesDataGenerator()
            self.df = generator.generate_sales_data()
            generator.save_data(self.df, data_file)
        else:
            print(f"📂 Loading data from {data_file}...")
            self.df = pd.read_csv(data_file)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
    
    def run_complete_analysis(self):
        """Run complete end-to-end analysis"""
        print("\n" + "🚀" * 40)
        print("AI-POWERED SALES ANALYTICS & FORECASTING SYSTEM")
        print("🚀" * 40)
        print("\nStarting comprehensive analysis...\n")
        
        # Step 1: Exploratory Data Analysis
        print("\n" + "=" * 80)
        print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 80)
        eda = SalesEDA(self.df)
        self.eda_results = eda.run_full_analysis()
        
        # Step 2: Data Visualization
        print("\n" + "=" * 80)
        print("STEP 2: DATA VISUALIZATION")
        print("=" * 80)
        visualizer = SalesVisualizer(self.df)
        visualizer.generate_all_visualizations()
        
        # Step 3: Machine Learning Forecasting
        print("\n" + "=" * 80)
        print("STEP 3: MACHINE LEARNING SALES FORECASTING")
        print("=" * 80)
        forecaster = SalesForecaster(self.df)
        self.ml_results = forecaster.train_models()
        
        # Next week forecast
        next_week = forecaster.predict_next_week()
        
        # Next month forecast
        next_month = forecaster.predict_next_month()
        
        # Feature importance
        forecaster.get_feature_importance()
        
        # Step 4: Product Demand Prediction
        print("\n" + "=" * 80)
        print("STEP 4: PRODUCT DEMAND PREDICTION")
        print("=" * 80)
        demand_predictor = ProductDemandPredictor(self.df)
        demand_predictor.train_product_models()
        self.demand_results = demand_predictor.get_demand_insights()
        
        # Step 5: AI Business Insights
        print("\n" + "=" * 80)
        print("STEP 5: AI-BASED BUSINESS INSIGHTS")
        print("=" * 80)
        ai_insights = AIBusinessInsights(
            self.df,
            self.eda_results,
            self.ml_results,
            self.demand_results
        )
        
        # Demonstrate Q&A
        print("\n🤖 AI Q&A System Demo:")
        print("-" * 80)
        
        questions = [
            "Why did sales drop in March?",
            "Which product should I promote next?",
            "What is the reason for low revenue this month?"
        ]
        
        for q in questions:
            print(f"\n❓ Question: {q}")
            answer = ai_insights.answer_question(q)
            print(answer)
            print("-" * 80)
        
        # Step 6: Smart AI Recommendations
        print("\n" + "=" * 80)
        print("STEP 6: SMART AI RECOMMENDATIONS")
        print("=" * 80)
        recommendations = SmartRecommendations(self.df, forecaster, demand_predictor)
        recommendations.generate_all_recommendations()
        recommendations.display_recommendations()
        recommendations.export_recommendations()
        
        # Final Summary
        self._generate_final_summary(forecaster, next_week, next_month)
        
        return {
            'eda': self.eda_results,
            'ml': self.ml_results,
            'demand': self.demand_results,
            'forecasts': {
                'next_week': next_week,
                'next_month': next_month
            }
        }
    
    def _generate_final_summary(self, forecaster, next_week, next_month):
        """Generate final business summary"""
        print("\n" + "=" * 80)
        print("📊 FINAL BUSINESS SUMMARY & CONCLUSION")
        print("=" * 80)
        
        total_revenue = self.df['Revenue'].sum()
        avg_daily = self.df.groupby(self.df['Date'].dt.date)['Revenue'].sum().mean()
        total_predicted_week = next_week['Predicted_Revenue'].sum()
        total_predicted_month = next_month['Predicted_Revenue'].sum()
        
        print(f"""
💰 FINANCIAL OVERVIEW:
   • Historical Total Revenue: ${total_revenue:,.2f}
   • Average Daily Revenue: ${avg_daily:,.2f}
   • Predicted Next Week Revenue: ${total_predicted_week:,.2f}
   • Predicted Next Month Revenue: ${total_predicted_month:,.2f}

📈 GROWTH PROJECTIONS:
   • Weekly Growth Rate: {((total_predicted_week / (avg_daily * 7)) - 1) * 100:.1f}%
   • Monthly Growth Rate: {((total_predicted_month / (avg_daily * 30)) - 1) * 100:.1f}%

🤖 AI INSIGHTS HIGHLIGHTS:
   • Best performing products identified
   • Peak sales hours and days mapped
   • Seasonal patterns analyzed
   • Demand forecasts generated
   • Actionable recommendations provided

🎯 KEY RECOMMENDATIONS:
   1. Optimize inventory based on ML demand predictions
   2. Increase staffing during identified peak hours
   3. Focus marketing on top-performing products
   4. Leverage seasonal trends for planning
   5. Monitor slow-moving products for discontinuation

✅ PROJECT DELIVERABLES:
   ✓ Comprehensive EDA analysis
   ✓ Professional data visualizations
   ✓ ML forecasting models (Linear Regression, Random Forest, XGBoost)
   ✓ Product-level demand predictions
   ✓ AI-powered business insights
   ✓ Smart recommendation system
   ✓ Business-oriented final report

📁 OUTPUT FILES:
   • sales_data.csv - Complete sales dataset
   • visualizations/ - All generated charts and graphs
   • ai_recommendations.csv - AI-generated recommendations
        """)
        
        print("\n" + "✅" * 40)
        print("ANALYSIS COMPLETE - ALL MODULES EXECUTED SUCCESSFULLY")
        print("✅" * 40 + "\n")

def interactive_mode():
    """Interactive mode for asking questions"""
    print("\n" + "=" * 80)
    print("🤖 INTERACTIVE AI ASSISTANT MODE")
    print("=" * 80)
    print("\nYou can ask questions like:")
    print("  • 'Why did sales drop in March?'")
    print("  • 'Which product should I promote next?'")
    print("  • 'What is the reason for low revenue this month?'")
    print("  • Type 'exit' to quit\n")
    
    # Load data and initialize
    system = SalesAnalyticsSystem(generate_data=False)
    eda = SalesEDA(system.df)
    eda_results = eda.run_full_analysis()
    
    forecaster = SalesForecaster(system.df)
    forecaster.train_models()
    
    demand_predictor = ProductDemandPredictor(system.df)
    demand_predictor.train_product_models()
    demand_results = demand_predictor.get_demand_insights()
    
    ai_insights = AIBusinessInsights(system.df, eda_results, None, demand_results)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        if question:
            answer = ai_insights.answer_question(question)
            print(f"\n🤖 AI Response:\n{answer}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        # Run complete analysis
        system = SalesAnalyticsSystem()
        results = system.run_complete_analysis()
        
        print("\n💡 Tip: Run with --interactive flag for Q&A mode:")
        print("   python main.py --interactive")
