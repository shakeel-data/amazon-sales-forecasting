# 🛒 Amazon Sales Forecasting & Analytics
<img width="1024" height="1024" alt="Google_AI_Studio_2025-08-25T16_20_10 624Z" src="https://github.com/user-attachments/assets/cdb9ca79-b296-44dd-b167-41605929131b" />

## 📋 Project Overview

This project provides a full-stack analytics solution, beginning with raw data ingestion and concluding with actionable business strategies. It showcases a robust workflow that includes data cleaning, database normalization, advanced SQL querying, and the implementation of multiple machine learning models for both supervised and unsupervised tasks. The primary goal is to unlock data-driven insights to forecast future sales, understand customer behavior, and guide strategic decision-making.

### 🎯 Business Objectives
- Forecast Future Sales: Predict revenue trends using multiple robust ML models.
- Segment Customers: Identify distinct customer personas based on purchasing behavior to enable targeted marketing.
- Analyze Performance: Uncover sales patterns, seasonal impacts, and key growth drivers.
- Generate Strategic Insights: Translate complex data into clear, actionable recommendations for business growth.

## 📊 Project Workflow
The project follows a structured, multi-stage workflow designed to transform raw data into high-value business intelligence.
1. Data Ingestion & Cleaning: Load the raw CSV, handle missing values, correct data types, and perform initial quality assessment using Python.

2. Database Normalization: Deconstruct the flat file into a relational schema with four distinct tables (customers, products, orders, sales) to ensure data integrity and query efficiency.

3. BigQuery Integration: Upload the normalized tables to Google BigQuery to serve as the single source of truth for all subsequent analysis.

4. Advanced SQL Analysis: Execute over 20 complex SQL queries to perform cohort analysis, calculate RFM scores, and uncover deep business trends.

5. Machine Learning Modeling:
- Forecasting: Develop three models (Linear Regression, Random Forest, Prophet) to predict future sales.
- Segmentation: Apply KMeans clustering to segment customers based on their purchasing behavior.
- Insight Generation & Strategy: Synthesize analytical findings into strategic business recommendations.

🚀 How to Run This Project
**Prerequisites**
- Python 3.8+
- Access to a Google Cloud Platform (GCP) project with BigQuery enabled.
- A GCP service account key (.json file) with BigQuery User & Data Editor roles.
**Installation**
1. Clone the repository:
```
git clone https://github.com/shakeel-data/amazon-sales-forecasting.git
cd amazon-sales-forecasting
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Set up GCP credentials:
Place your service account key .json file in the root directory and update the code to reference its path.

**Execution**
The project is organized into modular Jupyter notebooks. Run them in the following order:

- 01_data_exploration_cleaning.ipynb: To clean the raw data and perform EDA.
- 02_sql_analysis_bigquery.ipynb: To normalize the data, upload it to BigQuery, and run SQL queries.
- 03_machine_learning_forecasting.ipynb: To build and evaluate sales forecasting models.
- 04_customer_segmentation.ipynb: To perform customer segmentation and generate strategic insights.

## 🗄️ Database Schema
The initial flat CSV was normalized into a relational star schema to improve query performance and maintain data integrity.

- customers (Dimension): Stores unique customer information.
  - customer_id (Primary Key), customer_name, email, registration_date

- products (Dimension): Stores unique product details.
  - product_id (Primary Key), product_name, category, list_price

- orders (Dimension): Captures order-level information.
  - order_id (Primary Key), customer_id (Foreign Key), order_date, sales_rep

- sales (Fact): Contains transactional line-item data.
  - invoice_number (Primary Key), order_id (Foreign Key), product_id (Foreign Key), quantity, sales_amount, margin_amount

##⚙️ SQL Analysis Showcase
This project features over 20 advanced SQL queries. Below are a few highlights demonstrating key analytical techniques.




🤖 Machine Learning Models
Sales Forecasting
Three different models were developed to forecast sales, each providing a unique perspective.
1. Linear Regression (Baseline): A simple model to establish a performance baseline.
   - Result: R² = 0.72, RMSE = $1,247. Served as a good starting point.

2. Random Forest (Advanced): A more complex ensemble model that captures non-linear relationships.
   - Result: R² = 0.87, RMSE = $892. Significantly outperformed the baseline, making it the preferred model.

3. Facebook Prophet (Time-Series): A specialized model for forecasting time-series data with strong seasonal effects.
   - Result: Produced a 12-month forecast identifying strong yearly and quarterly seasonality, aligning with business expectations.

**Customer Segmentation**
KMeans Clustering was used to segment customers into distinct groups based on their purchasing behavior.

- Features Used: total_spent, order_frequency, avg_order_value, days_since_last_purchase.
- Outcome: Identified 4 key customer personas:
  - VIP Customers: High-value, frequent buyers. The most valuable segment.
  - Loyal Customers: Consistent, regular purchasers. Form the stable core of the business.
  - At-Risk Customers: Previously active but have not purchased recently. High churn risk.
  - New/Budget Customers: Recent or infrequent buyers with low spending.

##💡 Key Insights & Business Impact
| **Area**              | **Insight**                                                               | **Strategic Recommendation**                                                                    |
| --------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Sales Performance** | Q4 sales are 35% higher than other quarters due to seasonality.           | Optimize inventory and marketing spend in Q3 to prepare for the Q4 surge.                       |
| **Product Analysis**  | Premium organic products have the highest profit margins (avg. 23%).      | Expand the premium product line and feature these items in marketing campaigns.                 |
| **Customer Behavior** | 23% of the customer base is "At-Risk" and has not purchased recently.     | Launch a targeted re-engagement campaign with personalized offers to win back these customers.  |
| **Forecasting**       | Random Forest model predicts an 8% sales increase over the next 6 months. | Adjust financial targets and resource allocation to align with the predicted growth trajectory. |


## Next step
🚀 Future Enhancements
- Real-time Dashboard: Develop an interactive Looker Studio or Power BI dashboard for live monitoring of KPIs.
- Deep Learning Models: Implement LSTM networks for potentially more accurate long-term forecasting.
- Automated MLOps Pipeline: Create a pipeline to automatically retrain and deploy models to maintain accuracy.
- Recommendation Engine: Build a product recommendation system based on customer segments and purchase history.






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

