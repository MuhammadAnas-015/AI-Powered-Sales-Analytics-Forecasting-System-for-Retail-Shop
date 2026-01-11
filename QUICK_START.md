# 🚀 Quick Start Guide

## Installation (One-Time Setup)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python -c "import pandas, numpy, sklearn, xgboost; print('✅ Ready!')"
```

## Running the Complete Analysis

```bash
python main.py
```

This single command will:
- ✅ Generate sample sales data
- ✅ Perform comprehensive EDA
- ✅ Create professional visualizations
- ✅ Train ML forecasting models
- ✅ Predict product demand
- ✅ Generate AI insights
- ✅ Provide business recommendations

**Expected Runtime:** 2-5 minutes (depending on your system)

## Interactive Q&A Mode

Ask business questions:

```bash
python main.py --interactive
```

Example questions:
- "Why did sales drop in March?"
- "Which product should I promote next?"
- "What is the reason for low revenue this month?"

## Output Files

After running, you'll find:

1. **sales_data.csv** - Complete sales dataset
2. **visualizations/** - 6 professional charts
3. **ai_recommendations.csv** - AI-generated recommendations

## Using Your Own Data

1. Prepare CSV with columns: `Date`, `Product`, `Category`, `Quantity`, `Unit_Price`, `Revenue`, `Hour`, `Day_of_Week`, `Month`, `Year`, `Quarter`
2. Save as `sales_data.csv`
3. Run: `python main.py`

## Troubleshooting

**Issue:** Import errors
- **Solution:** `pip install -r requirements.txt`

**Issue:** Visualization errors
- **Solution:** Ensure matplotlib and seaborn are installed

**Issue:** XGBoost installation fails
- **Solution:** `pip install xgboost --upgrade`

---

**That's it! You're ready to analyze your retail sales data! 🎉**
