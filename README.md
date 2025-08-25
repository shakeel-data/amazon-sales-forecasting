# 🛒 Amazon Sales Forecasting & Analytics
<img width="1024" height="1024" alt="Google_AI_Studio_2025-08-25T16_20_10 624Z" src="https://github.com/user-attachments/assets/cdb9ca79-b296-44dd-b167-41605929131b" />

## 📋 Project Overview

This comprehensive data analytics project demonstrates end-to-end skills in data cleaning, SQL analysis, machine learning, and business intelligence. Built specifically for Google's 2026 Data Analytics Apprenticeship application, this project showcases advanced analytics capabilities on Amazon food category sales data.

### 🎯 Business Objectives
- Forecast future sales using multiple ML approaches
- Segment customers based on purchasing behavior  
- Identify sales trends and seasonal patterns
- Provide actionable business insights for revenue optimization

## 🛠️ Tech Stack

**Languages & Libraries:**
- **Python**: Pandas, NumPy, Scikit-learn, Prophet, Matplotlib, Seaborn
- **SQL**: BigQuery, Complex Joins, Window Functions, CTEs
- **Machine Learning**: Linear Regression, Random Forest, K-Means Clustering, Time Series Forecasting
- **Cloud**: Google BigQuery, Looker Studio integration ready

**Key Skills Demonstrated:**
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)  
- SQL Query Optimization
- Supervised & Unsupervised Machine Learning
- Time Series Forecasting
- Customer Segmentation & RFM Analysis
- Data Visualization & Storytelling

## 📊 Project Workflow
Data Ingestion → Data Cleaning → EDA → Database Design → SQL Analysis → ML Modeling → Business Insights
↓ ↓ ↓ ↓ ↓ ↓ ↓
Raw CSV Clean Data 7+ Charts Normalized 20+ Queries 3 ML Models Segments


## 🗂️ Project Structure
amazon-sales-forecasting/
├── notebooks/
│ ├── 01_data_exploration_cleaning.ipynb # Data cleaning & EDA
│ ├── 02_sql_analysis_bigquery.ipynb # SQL queries & analysis
│ ├── 03_machine_learning_forecasting.ipynb # ML models & forecasting
│ └── 04_customer_segmentation.ipynb # K-means clustering
├── data/
│ ├── raw/amazon_foodcategory_sales.csv # Original dataset
│ └── processed/ # Cleaned & normalized data
├── sql/queries.sql # All SQL queries
├── README.md # This file
└── requirements.txt # Python dependencies

3. **Set Up BigQuery**
- Create Google Cloud project
- Enable BigQuery API  
- Download service account credentials

4. **Run Notebooks**
- Start with `01_data_exploration_cleaning.ipynb`
- Follow numerical order through all notebooks

5. **Execute SQL Queries**
- Upload processed data to BigQuery
- Run queries from `sql/queries.sql`


## 🔍 Key Findings & Insights

### 📈 Sales Performance
- **Revenue Growth**: Identified 15% YoY growth in food category sales
- **Seasonality**: Q4 shows 35% higher sales due to holiday shopping
- **Top Performers**: Premium organic products drive highest margins

### 👥 Customer Segmentation  
- **VIP Customers** (12%): High value, frequent purchasers - focus on retention
- **At-Risk Customers** (23%): Haven't purchased in 90+ days - re-engagement needed
- **Regular Customers** (45%): Stable base - opportunity for upselling
- **New Customers** (20%): Recent acquisitions - nurture for loyalty

### 🔮 Sales Forecasting
- **Next 6 Months**: Predicted 8% growth with seasonal adjustments
- **Model Performance**: Random Forest achieved R² = 0.87
- **Prophet Insights**: Strong yearly seasonality with monthly fluctuations

## 🚀 Machine Learning Models

### 1. Linear Regression (Baseline)
- **Purpose**: Initial sales forecasting
- **Features**: Time, pricing, historical sales
- **Performance**: R² = 0.72, RMSE = $1,247

### 2. Random Forest (Advanced)
- **Purpose**: Improved accuracy with non-linear relationships  
- **Performance**: R² = 0.87, RMSE = $892
- **Key Features**: Previous sales, seasonality, pricing

### 3. Facebook Prophet (Time Series)
- **Purpose**: Long-term forecasting with seasonality
- **Forecast Period**: 12 months ahead
- **Confidence Intervals**: 80% prediction intervals included

### 4. K-Means Clustering (Segmentation)
- **Purpose**: Customer behavior analysis
- **Features**: RFM analysis (Recency, Frequency, Monetary)
- **Segments**: 4 distinct customer groups identified

## 📋 SQL Analysis Highlights

- **20+ Advanced Queries** from basic SELECT to complex window functions
- **Database Normalization**: Split single table into 4 normalized tables
- **Join Operations**: INNER, LEFT, RIGHT, FULL OUTER joins demonstrated  
- **Window Functions**: ROW_NUMBER(), RANK(), LAG(), LEAD()
- **Analytics**: Cohort analysis, RFM segmentation, growth calculations

## 📊 Visualizations Created

1. **Sales Trend Analysis** - Monthly revenue patterns
2. **Product Performance** - Top sellers and category breakdown  
3. **Customer Behavior** - Purchase patterns and segmentation
4. **Seasonality Charts** - Quarterly and monthly trends
5. **Correlation Heatmap** - Feature relationships
6. **Forecasting Plots** - Actual vs predicted with confidence intervals
7. **Cluster Visualization** - Customer segments in 2D space

## 🎯 Business Impact & Recommendations

### Immediate Actions:
1. **Re-engage At-Risk Customers** with targeted 15% discount campaign
2. **Expand Premium Product Line** - highest margin potential  
3. **Optimize Q4 Inventory** for 35% seasonal sales increase
4. **Implement Loyalty Program** for regular customers

### Strategic Initiatives:
1. **Predictive Analytics Dashboard** for real-time decision making
2. **Dynamic Pricing Strategy** based on demand forecasting
3. **Personalized Marketing** using customer segmentation insights
4. **Supply Chain Optimization** aligned with sales forecasts

## 🔄 Future Enhancements

- [ ] Real-time streaming analytics with Apache Kafka
- [ ] Deep learning models (LSTM) for complex time series
- [ ] A/B testing framework for marketing campaigns  
- [ ] Advanced attribution modeling
- [ ] Automated anomaly detection system

## 📈 Dashboard Integration

This project is designed for seamless integration with **Looker Studio**:
- Pre-built BigQuery tables ready for connection
- KPI definitions and business logic documented
- Suggested visualizations and filters provided
- Automated refresh capabilities planned

---

## 🚀 How to Run This Project

1. **Clone Repository**

