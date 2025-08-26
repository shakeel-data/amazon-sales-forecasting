-- Customer Overview
-- Purpose: Basic customer information with aggregated metrics
SELECT 
    customer_name,
    email,
    phone,
    registration_date,
    -- Calculate days since registration
    DATE_DIFF(CURRENT_DATE(), registration_date, DAY) as days_as_customer
FROM `your-project-id.your-dataset-id.customers`
ORDER BY registration_date DESC
LIMIT 10;

-- Product Catalog Summary
-- Purpose: Overview of products with pricing information
SELECT 
    category,
    product_name,
    list_price,
    unit_measure,
    -- Price categories for analysis
    CASE 
        WHEN list_price < 50 THEN 'Budget'
        WHEN list_price BETWEEN 50 AND 200 THEN 'Mid-Range'
        WHEN list_price > 200 THEN 'Premium'
        ELSE 'Unknown'
    END as price_category
FROM `your-project-id.your-dataset-id.products`
WHERE list_price IS NOT NULL
ORDER BY list_price DESC;

-- Sales Performance by Month
-- Purpose: Track sales trends over time
SELECT
  EXTRACT(YEAR FROM o.order_date) AS year,
  EXTRACT(MONTH FROM o.order_date) AS month,
  COUNT(DISTINCT o.order_id) AS total_orders,
  ROUND(SUM(s.sales_amount), 2) AS total_revenue,
  ROUND(AVG(s.sales_amount), 2) AS avg_order_value
FROM
  `your-project-id.your-dataset-id.orders AS o
  JOIN
  `your-project-id.your-dataset-id.sales AS s
  ON o.order_id = s.order_id
GROUP BY year, month
ORDER BY year DESC, month DESC;


                                                                             -- JOIN Queries--
-- Customer Purchase History (INNER JOIN)
-- Purpose: Show customers who have made purchases
SELECT 
    c.customer_name,
    c.email,
    COUNT(DISTINCT o.order_id) as total_orders,
    ROUND(SUM(s.sales_amount), 2) as lifetime_value,
    MIN(o.order_date) as first_purchase,
    MAX(o.order_date) as last_purchase
FROM `your-project-id.your-dataset-id.customers` c
INNER JOIN `your-project-id.your-dataset-id.orders` o ON c.Custkey = o.customer_id
INNER JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
GROUP BY c.customer_name, c.email
ORDER BY lifetime_value DESC
LIMIT 20;

-- All Customers with Purchase Status (LEFT JOIN)
-- Purpose: Include customers who haven't made purchases
SELECT 
    c.customer_name,
    c.email,
    c.registration_date,
    COALESCE(COUNT(DISTINCT o.order_id), 0) as total_orders,
    COALESCE(ROUND(SUM(s.sales_amount), 2), 0) as lifetime_value,
    CASE 
        WHEN o.customer_id IS NULL THEN 'No Purchases'
        WHEN COUNT(DISTINCT o.order_id) = 1 THEN 'Single Purchase'
        ELSE 'Repeat Customer'
    END as customer_status
FROM `your-project-id.your-dataset-id.customers` c
LEFT JOIN `your-project-id.your-dataset-id.orders` o ON c.Custkey = o.customer_id
LEFT JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
GROUP BY c.customer_name, c.email, c.registration_date, o.customer_id
ORDER BY lifetime_value DESC;

-- QUERY 6: Product Performance Analysis (RIGHT JOIN)
-- Purpose: Show all products with their sales performance
SELECT 
    p.product_name,
    p.category,
    p.list_price,
    COALESCE(COUNT(s.product_id), 0) as times_sold,
    COALESCE(ROUND(SUM(s.sales_amount), 2), 0) as total_revenue,
    COALESCE(ROUND(AVG(s.unit_price), 2), p.list_price) as avg_selling_price
FROM `your-project-id.your-dataset-id.sales` s
RIGHT JOIN `your-project-id.your-dataset-id.products` p ON s.product_id = p.product_id
GROUP BY p.product_name, p.category, p.list_price
ORDER BY total_revenue DESC;

-- Complete Order Details (FULL OUTER JOIN)
-- Purpose: Comprehensive view of all orders and potential data gaps
SELECT 
    COALESCE(o.order_id, s.order_id) as order_id,
    c.customer_name,
    o.order_date,
    p.product_name,
    s.quantity,
    s.unit_price,
    s.sales_amount,
    CASE 
        WHEN o.order_id IS NULL THEN 'Missing Order Info'
        WHEN s.order_id IS NULL THEN 'Missing Sales Info'
        ELSE 'Complete'
    END as data_quality
FROM `your-project-id.your-dataset-id.orders` o
FULL OUTER JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
LEFT JOIN `your-project-id.your-dataset-id.customers` c ON o.customer_id = c.Custkey
LEFT JOIN `your-project-id.your-dataset-id.products` p ON s.product_id = p.product_id
WHERE COALESCE(o.order_id, s.order_id) IS NOT NULL
ORDER BY COALESCE(o.order_date, '1900-01-01') DESC;


                                                                              -- Window Functions --
-- Customer Ranking by Revenue
-- Purpose: Rank customers and show revenue percentiles
SELECT 
    c.customer_name,
    ROUND(SUM(s.sales_amount), 2) as total_revenue,
    -- Ranking functions
    ROW_NUMBER() OVER (ORDER BY SUM(s.sales_amount) DESC) as revenue_rank,
    DENSE_RANK() OVER (ORDER BY SUM(s.sales_amount) DESC) as dense_rank,
    -- Percentile calculation
    PERCENT_RANK() OVER (ORDER BY SUM(s.sales_amount)) as revenue_percentile,
    -- Revenue quartiles
    NTILE(4) OVER (ORDER BY SUM(s.sales_amount)) as revenue_quartile
FROM `your-project-id.your-dataset-id.customers` c
JOIN `your-project-id.your-dataset-id.orders` o ON c.Custkey = o.customer_id
JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
GROUP BY c.customer_name
ORDER BY total_revenue DESC;

-- Running Totals and Moving Averages
-- Purpose: Calculate cumulative sales and trends
SELECT 
    order_date,
    COUNT(DISTINCT order_id) as daily_orders,
    ROUND(SUM(daily_revenue), 2) as daily_revenue,
    -- Running totals
    SUM(COUNT(DISTINCT order_id)) OVER (
        ORDER BY order_date 
        ROWS UNBOUNDED PRECEDING
    ) as cumulative_orders,
    SUM(SUM(daily_revenue)) OVER (
        ORDER BY order_date 
        ROWS UNBOUNDED PRECEDING
    ) as cumulative_revenue,
    -- 7-day moving average
    ROUND(AVG(SUM(daily_revenue)) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) as revenue_7day_avg
FROM (
    SELECT 
        o.order_date,
        o.order_id,
        SUM(s.sales_amount) as daily_revenue
    FROM `your-project-id.your-dataset-id.orders` o
    JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
    GROUP BY o.order_date, o.order_id
) daily_data
GROUP BY order_date
ORDER BY order_date;

-- Product Sales Comparison
-- Purpose: Compare each product's performance to category average
SELECT 
    p.category,
    p.product_name,
    COUNT(s.product_id) as units_sold,
    ROUND(SUM(s.sales_amount), 2) as product_revenue,
    -- Compare to category averages
    ROUND(AVG(SUM(s.sales_amount)) OVER (PARTITION BY p.category), 2) as category_avg_revenue,
    ROUND(SUM(s.sales_amount) - AVG(SUM(s.sales_amount)) OVER (PARTITION BY p.category), 2) as revenue_vs_category_avg,
    -- Rank within category
    ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(s.sales_amount) DESC) as category_rank
FROM `your-project-id.your-dataset-id.products` p
JOIN `your-project-id.your-dataset-id.sales` s ON p.product_id = s.product_id
GROUP BY p.category, p.product_name
ORDER BY p.category, product_revenue DESC;


                                        -- CTEs and Subqueries --
-- Customer Segmentation with CTE
-- Purpose: Segment customers using RFM analysis
WITH customer_metrics AS (
    SELECT 
        c.Custkey,
        c.customer_name,
        -- Recency: Days since last purchase
        DATE_DIFF(CURRENT_DATE(), MAX(o.order_date), DAY) as days_since_last_purchase,
        -- Frequency: Number of orders
        COUNT(DISTINCT o.order_id) as total_orders,
        -- Monetary: Total spent
        ROUND(SUM(s.sales_amount), 2) as total_spent
    FROM `your-project-id.your-dataset-id.customers` c
    JOIN `your-project-id.your-dataset-id.orders` o ON c.Custkey = o.customer_id
    JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
    GROUP BY c.Custkey, c.customer_name
),
rfm_scores AS (
    SELECT 
        *,
        -- RFM Scoring (1-5 scale)
        NTILE(5) OVER (ORDER BY days_since_last_purchase DESC) as recency_score,
        NTILE(5) OVER (ORDER BY total_orders) as frequency_score,
        NTILE(5) OVER (ORDER BY total_spent) as monetary_score
    FROM customer_metrics
)
SELECT 
    customer_name,
    days_since_last_purchase,
    total_orders,
    total_spent,
    recency_score,
    frequency_score,
    monetary_score,
    -- Customer segments based on RFM
    CASE 
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
        WHEN recency_score >= 3 AND frequency_score <= 2 THEN 'Potential Loyalists'
        WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'Cannot Lose Them'
        ELSE 'Others'
    END as customer_segment
FROM rfm_scores
ORDER BY monetary_score DESC, frequency_score DESC, recency_score DESC;

-- Top Products by Category (Subqueries)
-- Purpose: Find best-selling products in each category
SELECT 
    category,
    product_name,
    total_revenue,
    units_sold,
    category_rank
FROM (
    SELECT 
        p.category,
        p.product_name,
        ROUND(SUM(s.sales_amount), 2) as total_revenue,
        SUM(s.quantity) as units_sold,
        ROW_NUMBER() OVER (PARTITION BY p.category ORDER BY SUM(s.sales_amount) DESC) as category_rank
    FROM `your-project-id.your-dataset-id.products` p
    JOIN `your-project-id.your-dataset-id.sales` s ON p.product_id = s.product_id
    GROUP BY p.category, p.product_name
) ranked_products
WHERE category_rank <= 3  -- Top 3 products per category
ORDER BY category, category_rank;

-- Monthly Growth Analysis with CTE
-- Purpose: Calculate month-over-month growth rates
WITH monthly_sales AS (
    SELECT 
        EXTRACT(YEAR FROM o.order_date) as year,
        EXTRACT(MONTH FROM o.order_date) as month,
        COUNT(DISTINCT o.order_id) as orders,
        ROUND(SUM(s.sales_amount), 2) as revenue
    FROM `your-project-id.your-dataset-id.orders` o
    JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
    GROUP BY year, month
),
growth_analysis AS (
    SELECT 
        year,
        month,
        orders,
        revenue,
        -- Previous month values
        LAG(orders) OVER (ORDER BY year, month) as prev_month_orders,
        LAG(revenue) OVER (ORDER BY year, month) as prev_month_revenue
    FROM monthly_sales
)
SELECT 
    year,
    month,
    orders,
    revenue,
    prev_month_orders,
    prev_month_revenue,
    -- Growth calculations
    CASE 
        WHEN prev_month_orders IS NOT NULL THEN
            ROUND(((orders - prev_month_orders) / prev_month_orders) * 100, 2)
        ELSE NULL
    END as order_growth_percent,
    CASE 
        WHEN prev_month_revenue IS NOT NULL THEN
            ROUND(((revenue - prev_month_revenue) / prev_month_revenue) * 100, 2)
        ELSE NULL
    END as revenue_growth_percent
FROM growth_analysis
ORDER BY year, month;


                                                                           -- Advanced Analytics --
-- Cohort Analysis - Customer Retention
-- Purpose: Track customer behavior over time
WITH customer_orders AS (
    SELECT 
        c.Custkey,
        o.order_date,
        ROW_NUMBER() OVER (PARTITION BY c.Custkey ORDER BY o.order_date) as order_sequence,
        MIN(o.order_date) OVER (PARTITION BY c.Custkey) as first_order_date
    FROM `your-project-id.your-dataset-id.customers` c
    JOIN `your-project-id.your-dataset-id.orders` o ON c.Custkey = o.customer_id
),
cohort_data AS (
    SELECT 
        DATE_TRUNC(first_order_date, MONTH) as cohort_month,
        DATE_DIFF(DATE_TRUNC(order_date, MONTH), DATE_TRUNC(first_order_date, MONTH), MONTH) as period_number,
        Custkey
    FROM customer_orders
)
SELECT 
    cohort_month,
    period_number,
    COUNT(DISTINCT Custkey) as customers,
    -- Calculate retention rate
    ROUND(
        COUNT(DISTINCT Custkey) / 
        FIRST_VALUE(COUNT(DISTINCT Custkey)) OVER (
            PARTITION BY cohort_month 
            ORDER BY period_number 
            ROWS UNBOUNDED PRECEDING
        ) * 100, 2
    ) as retention_rate
FROM cohort_data
GROUP BY cohort_month, period_number
ORDER BY cohort_month, period_number;

-- ABC Analysis - Product Classification
-- Purpose: Classify products by revenue contribution
WITH product_revenue AS (
    SELECT 
        p.product_id,
        p.product_name,
        ROUND(SUM(s.sales_amount), 2) as total_revenue
    FROM `your-project-id.your-dataset-id.products` p
    JOIN `your-project-id.your-dataset-id.sales` s ON p.product_id = s.product_id
    GROUP BY p.product_id, p.product_name
),
revenue_analysis AS (
    SELECT 
        *,
        SUM(total_revenue) OVER () as total_company_revenue,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC ROWS UNBOUNDED PRECEDING) as cumulative_revenue
    FROM product_revenue
)
SELECT 
    product_name,
    total_revenue,
    ROUND((total_revenue / total_company_revenue) * 100, 2) as revenue_percentage,
    ROUND((cumulative_revenue / total_company_revenue) * 100, 2) as cumulative_percentage,
    -- ABC Classification
    CASE 
        WHEN ROUND((cumulative_revenue / total_company_revenue) * 100, 2) <= 80 THEN 'A'
        WHEN ROUND((cumulative_revenue / total_company_revenue) * 100, 2) <= 95 THEN 'B'
        ELSE 'C'
    END as abc_category
FROM revenue_analysis
ORDER BY total_revenue DESC;

-- Sales Rep Performance Analysis
-- Purpose: Evaluate sales representative effectiveness
SELECT 
    o.sales_rep,
    COUNT(DISTINCT o.customer_id) as unique_customers,
    COUNT(DISTINCT o.order_id) as total_orders,
    ROUND(AVG(order_totals.order_value), 2) as avg_order_value,
    ROUND(SUM(order_totals.order_value), 2) as total_sales,
    -- Performance metrics
    ROUND(SUM(order_totals.order_value) / COUNT(DISTINCT o.order_id), 2) as sales_per_order,
    ROUND(SUM(order_totals.order_value) / COUNT(DISTINCT o.customer_id), 2) as sales_per_customer,
    -- Ranking
    RANK() OVER (ORDER BY SUM(order_totals.order_value) DESC) as sales_rank
FROM `your-project-id.your-dataset-id.orders` o
JOIN (
    SELECT 
        order_id,
        SUM(sales_amount) as order_value
    FROM `your-project-id.your-dataset-id.sales`
    GROUP BY order_id
) order_totals ON o.order_id = order_totals.order_id
WHERE o.sales_rep IS NOT NULL
GROUP BY o.sales_rep
ORDER BY total_sales DESC;


                                                                      -- Business Intelligence Queries --
-- Customer Lifetime Value Prediction
-- Purpose: Estimate future customer value
WITH customer_behavior AS (
    SELECT 
        c.Custkey,
        c.customer_name,
        c.registration_date,
        COUNT(DISTINCT o.order_id) as total_orders,
        ROUND(SUM(s.sales_amount), 2) as total_spent,
        ROUND(AVG(s.sales_amount), 2) as avg_order_value,
        DATE_DIFF(MAX(o.order_date), MIN(o.order_date), DAY) + 1 as customer_lifespan_days,
        DATE_DIFF(CURRENT_DATE(), MAX(o.order_date), DAY) as days_since_last_order
    FROM `your-project-id.your-dataset-id.customers` c
    JOIN `your-project-id.your-dataset-id.orders` o ON c.Custkey = o.customer_id
    JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
    GROUP BY c.Custkey, c.customer_name, c.registration_date
)
SELECT 
    customer_name,
    total_orders,
    total_spent,
    avg_order_value,
    customer_lifespan_days,
    -- Calculate purchase frequency
    CASE 
        WHEN customer_lifespan_days > 0 THEN 
            ROUND(total_orders / (customer_lifespan_days / 365.0), 2)
        ELSE 0 
    END as orders_per_year,
    -- Estimated CLV (simplified)
    CASE 
        WHEN customer_lifespan_days > 0 THEN 
            ROUND(avg_order_value * (total_orders / (customer_lifespan_days / 365.0)) * 2, 2)
        ELSE avg_order_value 
    END as estimated_2year_clv,
    -- Customer status
    CASE 
        WHEN days_since_last_order <= 30 THEN 'Active'
        WHEN days_since_last_order <= 90 THEN 'At Risk'
        ELSE 'Churned'
    END as customer_status
FROM customer_behavior
ORDER BY estimated_2year_clv DESC;

-- Seasonal Sales Patterns
-- Purpose: Identify seasonal trends in sales
SELECT 
    EXTRACT(MONTH FROM o.order_date) as month,
    FORMAT_DATE('%B', DATE(2024, EXTRACT(MONTH FROM o.order_date), 1)) as month_name,
    COUNT(DISTINCT o.order_id) as total_orders,
    ROUND(SUM(s.sales_amount), 2) as total_revenue,
    ROUND(AVG(s.sales_amount), 2) as avg_order_value,
    -- Compare to annual average
    ROUND(
        (SUM(s.sales_amount) - AVG(SUM(s.sales_amount)) OVER ()) / 
        AVG(SUM(s.sales_amount)) OVER () * 100, 2
    ) as variance_from_avg_percent
FROM `your-project-id.your-dataset-id.orders` o
JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
GROUP BY month, month_name
ORDER BY month;

-- Product Cross-Selling Analysis
-- Purpose: Find products frequently bought together
WITH order_products AS (
    SELECT 
        s1.order_id,
        s1.product_id as product_a,
        s2.product_id as product_b
    FROM `your-project-id.your-dataset-id.sales` s1
    JOIN `your-project-id.your-dataset-id.sales` s2 
        ON s1.order_id = s2.order_id 
        AND s1.product_id < s2.product_id  -- Avoid duplicates
),
product_pairs AS (
    SELECT 
        product_a,
        product_b,
        COUNT(*) as times_bought_together
    FROM order_products
    GROUP BY product_a, product_b
    HAVING COUNT(*) >= 2  -- At least bought together twice
)
SELECT 
    pa.product_name as product_a_name,
    pb.product_name as product_b_name,
    pp.times_bought_together,
    -- Calculate support (percentage of orders containing both items)
    ROUND(
        pp.times_bought_together / 
        (SELECT COUNT(DISTINCT order_id) FROM `your-project-id.your-dataset-id.sales`) * 100, 2
    ) as support_percent
FROM product_pairs pp
JOIN `your-project-id.your-dataset-id.products` pa ON pp.product_a = pa.product_id
JOIN `your-project-id.your-dataset-id.products` pb ON pp.product_b = pb.product_id
ORDER BY times_bought_together DESC
LIMIT 20;

-- Revenue Forecast Base Data
-- Purpose: Prepare data for revenue forecasting
WITH daily_sales AS (
    SELECT 
        o.order_date,
        COUNT(DISTINCT o.order_id) as orders,
        ROUND(SUM(s.sales_amount), 2) as revenue,
        COUNT(DISTINCT o.customer_id) as unique_customers
    FROM `your-project-id.your-dataset-id.orders` o
    JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
    GROUP BY o.order_date
)
SELECT 
    order_date,
    orders,
    revenue,
    unique_customers,
    -- 7-day moving averages for trend analysis
    ROUND(AVG(orders) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) as orders_7day_avg,
    ROUND(AVG(revenue) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 2) as revenue_7day_avg,
    -- Week-over-week growth
    ROUND(
        (revenue - LAG(revenue, 7) OVER (ORDER BY order_date)) / 
        LAG(revenue, 7) OVER (ORDER BY order_date) * 100, 2
    ) as wow_growth_percent
FROM daily_sales
ORDER BY order_date DESC;

-- Executive Dashboard Summary
-- Purpose: Key metrics for executive reporting
WITH summary_stats AS (
    SELECT 
        COUNT(DISTINCT c.Custkey) as total_customers,
        COUNT(DISTINCT p.product_id) as total_products,
        COUNT(DISTINCT o.order_id) as total_orders,
        ROUND(SUM(s.sales_amount), 2) as total_revenue,
        ROUND(AVG(s.sales_amount), 2) as avg_order_value,
        COUNT(DISTINCT o.sales_rep) as active_sales_reps
    FROM `your-project-id.your-dataset-id.customers` c
    CROSS JOIN `your-project-id.your-dataset-id.products` p
    LEFT JOIN `your-project-id.your-dataset-id.orders` o ON c.Custkey = o.customer_id
    LEFT JOIN `your-project-id.your-dataset-id.sales` s ON o.order_id = s.order_id
)
SELECT 
    'Total Customers' as metric, CAST(total_customers as STRING) as value
FROM summary_stats
UNION ALL SELECT 'Total Products', CAST(total_products as STRING) FROM summary_stats
UNION ALL SELECT 'Total Orders', CAST(total_orders as STRING) FROM summary_stats  
UNION ALL SELECT 'Total Revenue', CONCAT('$', CAST(total_revenue as STRING)) FROM summary_stats
UNION ALL SELECT 'Average Order Value', CONCAT('$', CAST(avg_order_value as STRING)) FROM summary_stats
UNION ALL SELECT 'Active Sales Reps', CAST(active_sales_reps as STRING) FROM summary_stats;

                                                -- End of Anlysis --