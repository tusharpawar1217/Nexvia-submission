-- ============================================================================
-- ANALYTICAL QUERIES: Customer Churn Analysis
-- ============================================================================

-- Query 1: Overall churn rate and customer counts
-- Purpose: Baseline metrics for the dataset
SELECT 
    COUNT(*) as total_customers,
    SUM(churn) as churned_customers,
    ROUND(AVG(churn) * 100, 2) as churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges,
    ROUND(AVG(total_charges), 2) as avg_total_charges,
    ROUND(AVG(tenure), 1) as avg_tenure_months
FROM customers;

-- Query 2: Churn rate by contract type
-- Purpose: Identify which contract types retain customers best
SELECT 
    contract,
    COUNT(*) as customer_count,
    SUM(churn) as churned_count,
    ROUND(AVG(churn) * 100, 2) as churn_rate_pct,
    ROUND(AVG(tenure), 1) as avg_tenure,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- Query 3: Churn rate by tenure buckets
-- Purpose: Understand when customers are most at risk
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END as tenure_bucket,
    COUNT(*) as customer_count,
    SUM(churn) as churned_count,
    ROUND(AVG(churn) * 100, 2) as churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges
FROM customers
GROUP BY tenure_bucket
ORDER BY 
    CASE 
        WHEN tenure <= 12 THEN 1
        WHEN tenure <= 24 THEN 2
        WHEN tenure <= 48 THEN 3
        ELSE 4
    END;

-- Query 4: Churn rate by payment method
-- Purpose: Identify if payment method correlates with churn
SELECT 
    payment_method,
    COUNT(*) as customer_count,
    SUM(churn) as churned_count,
    ROUND(AVG(churn) * 100, 2) as churn_rate_pct,
    ROUND(AVG(tenure), 1) as avg_tenure,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- Query 5: Churn rate by internet service type
-- Purpose: Understand which internet service customers churn more
SELECT 
    internet_service,
    COUNT(*) as customer_count,
    SUM(churn) as churned_count,
    ROUND(AVG(churn) * 100, 2) as churn_rate_pct,
    ROUND(AVG(tenure), 1) as avg_tenure,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges,
    ROUND(AVG(CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END) * 100, 2) as online_security_pct,
    ROUND(AVG(CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END) * 100, 2) as tech_support_pct
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- Query 6: Senior citizens analysis
-- Purpose: Understand if seniors have different churn patterns
SELECT 
    CASE WHEN senior_citizen = 1 THEN 'Senior' ELSE 'Non-Senior' END as customer_type,
    COUNT(*) as customer_count,
    SUM(churn) as churned_count,
    ROUND(AVG(churn) * 100, 2) as churn_rate_pct,
    ROUND(AVG(tenure), 1) as avg_tenure,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges,
    ROUND(AVG(CASE WHEN partner = 'Yes' THEN 1 ELSE 0 END) * 100, 2) as has_partner_pct,
    ROUND(AVG(CASE WHEN dependents = 'Yes' THEN 1 ELSE 0 END) * 100, 2) as has_dependents_pct
FROM customers
GROUP BY senior_citizen
ORDER BY churn_rate_pct DESC;

-- Query 7: High-value at-risk customers
-- Purpose: Identify customers worth targeting for retention
-- Note: Requires predictions table to be populated
SELECT 
    c.customer_id,
    c.tenure,
    c.contract,
    c.monthly_charges,
    c.total_charges,
    p.churn_probability,
    p.clv_predicted,
    p.segment_name
FROM customers c
JOIN predictions p ON c.customer_id = p.customer_id
WHERE p.churn_probability >= 0.6
    AND c.total_charges >= 2000
ORDER BY p.clv_predicted DESC, p.churn_probability DESC
LIMIT 100;

-- Query 8: Service adoption and churn
-- Purpose: Understand which service combinations reduce churn
WITH service_summary AS (
    SELECT 
        customer_id,
        churn,
        (CASE WHEN phone_service = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN internet_service != 'No' THEN 1 ELSE 0 END +
         CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END) as service_count
    FROM customers
)
SELECT 
    service_count,
    COUNT(*) as customer_count,
    SUM(churn) as churned_count,
    ROUND(AVG(churn) * 100, 2) as churn_rate_pct
FROM service_summary
GROUP BY service_count
ORDER BY service_count;

-- Query 9: Paperless billing and autopay impact
-- Purpose: Understand billing preference correlation with churn
SELECT 
    paperless_billing,
    CASE 
        WHEN payment_method IN ('Bank transfer (automatic)', 'Credit card (automatic)') 
        THEN 'Autopay' 
        ELSE 'Manual' 
    END as payment_type,
    COUNT(*) as customer_count,
    SUM(churn) as churned_count,
    ROUND(AVG(churn) * 100, 2) as churn_rate_pct,
    ROUND(AVG(tenure), 1) as avg_tenure
FROM customers
GROUP BY paperless_billing, payment_type
ORDER BY churn_rate_pct DESC;

-- Query 10: Revenue at risk
-- Purpose: Calculate total monthly revenue at risk from predicted churners
-- Note: Requires predictions table to be populated
SELECT 
    COUNT(*) as high_risk_customers,
    SUM(c.monthly_charges) as monthly_revenue_at_risk,
    SUM(p.clv_predicted) as total_clv_at_risk,
    ROUND(AVG(p.churn_probability), 4) as avg_churn_probability,
    ROUND(AVG(c.tenure), 1) as avg_tenure
FROM customers c
JOIN predictions p ON c.customer_id = p.customer_id
WHERE p.churn_probability >= 0.5;
