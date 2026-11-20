-- ============================================================================
-- SCHEMA: Customer Churn Database
-- Description: Relational schema for IBM Telco customer churn analysis
-- ============================================================================

-- Drop tables if they exist (for clean re-runs)
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS features CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ============================================================================
-- CUSTOMERS TABLE
-- Core customer demographics, contract, and billing information
-- ============================================================================
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    
    -- Demographics
    gender VARCHAR(10),
    senior_citizen INTEGER CHECK (senior_citizen IN (0, 1)),
    partner VARCHAR(3) CHECK (partner IN ('Yes', 'No')),
    dependents VARCHAR(3) CHECK (dependents IN ('Yes', 'No')),
    
    -- Tenure & Contract
    tenure INTEGER CHECK (tenure >= 0),
    contract VARCHAR(20) CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    
    -- Services
    phone_service VARCHAR(3) CHECK (phone_service IN ('Yes', 'No')),
    multiple_lines VARCHAR(20),
    internet_service VARCHAR(20),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20),
    
    -- Billing
    paperless_billing VARCHAR(3) CHECK (paperless_billing IN ('Yes', 'No')),
    payment_method VARCHAR(50),
    monthly_charges NUMERIC(10, 2) CHECK (monthly_charges >= 0),
    total_charges NUMERIC(10, 2) CHECK (total_charges >= 0),
    
    -- Target
    churn INTEGER CHECK (churn IN (0, 1)),
    
    -- Metadata
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX idx_customers_churn ON customers(churn);
CREATE INDEX idx_customers_contract ON customers(contract);
CREATE INDEX idx_customers_tenure ON customers(tenure);
CREATE INDEX idx_customers_internet_service ON customers(internet_service);

-- ============================================================================
-- FEATURES TABLE
-- Engineered features for modeling
-- ============================================================================
CREATE TABLE features (
    customer_id VARCHAR(50) PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    
    -- Tenure features
    tenure_months INTEGER,
    tenure_bucket VARCHAR(20),  -- '0-12', '13-24', '25-48', '49+'
    
    -- Financial features
    charges_per_tenure_month NUMERIC(10, 4),
    charge_change_ratio NUMERIC(10, 4),  -- MonthlyCharges / (TotalCharges/tenure)
    
    -- Service features
    service_count INTEGER CHECK (service_count >= 0),
    has_phone_service INTEGER CHECK (has_phone_service IN (0, 1)),
    has_internet_service INTEGER CHECK (has_internet_service IN (0, 1)),
    has_online_security INTEGER CHECK (has_online_security IN (0, 1)),
    has_tech_support INTEGER CHECK (has_tech_support IN (0, 1)),
    has_streaming_services INTEGER CHECK (has_streaming_services IN (0, 1)),
    
    -- Contract features
    is_month_to_month INTEGER CHECK (is_month_to_month IN (0, 1)),
    is_one_year INTEGER CHECK (is_one_year IN (0, 1)),
    is_two_year INTEGER CHECK (is_two_year IN (0, 1)),
    
    -- Payment features
    has_autopay INTEGER CHECK (has_autopay IN (0, 1)),
    
    -- Proxy features
    potential_complaint_flag INTEGER CHECK (potential_complaint_flag IN (0, 1)),
    -- Flag customers who: month-to-month + no tech support + no online security
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_features_tenure_bucket ON features(tenure_bucket);
CREATE INDEX idx_features_service_count ON features(service_count);

-- ============================================================================
-- PREDICTIONS TABLE
-- Model predictions and analysis results
-- ============================================================================
CREATE TABLE predictions (
    customer_id VARCHAR(50) PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    
    -- Churn prediction
    churn_probability NUMERIC(5, 4) CHECK (churn_probability BETWEEN 0 AND 1),
    churn_prediction INTEGER CHECK (churn_prediction IN (0, 1)),
    churn_model_version VARCHAR(50),
    
    -- Survival analysis
    median_survival_months NUMERIC(10, 2),
    expected_lifetime_months NUMERIC(10, 2),
    survival_12m NUMERIC(5, 4),
    survival_24m NUMERIC(5, 4),
    survival_36m NUMERIC(5, 4),
    
    -- CLV
    clv_predicted NUMERIC(12, 2),
    clv_historical NUMERIC(12, 2),
    
    -- Segmentation
    segment_id INTEGER,
    segment_name VARCHAR(50),
    
    -- Uplift (simulated for Telco)
    uplift_score NUMERIC(10, 6),
    uplift_segment VARCHAR(50),  -- Persuadables, Sure Things, Lost Causes, Sleeping Dogs
    
    -- Metadata
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_churn_prob ON predictions(churn_probability);
CREATE INDEX idx_predictions_segment ON predictions(segment_name);
CREATE INDEX idx_predictions_uplift_segment ON predictions(uplift_segment);

-- ============================================================================
-- VIEWS
-- Useful analytical views
-- ============================================================================

-- Complete customer profile view
CREATE OR REPLACE VIEW customer_profile AS
SELECT 
    c.*,
    f.tenure_bucket,
    f.service_count,
    f.charges_per_tenure_month,
    f.is_month_to_month,
    f.has_autopay,
    p.churn_probability,
    p.churn_prediction,
    p.expected_lifetime_months,
    p.clv_predicted,
    p.segment_name,
    p.uplift_score,
    p.uplift_segment
FROM customers c
LEFT JOIN features f ON c.customer_id = f.customer_id
LEFT JOIN predictions p ON c.customer_id = p.customer_id;

-- High-risk customers view
CREATE OR REPLACE VIEW high_risk_customers AS
SELECT 
    customer_id,
    churn_probability,
    clv_predicted,
    tenure,
    contract,
    monthly_charges,
    segment_name
FROM customer_profile
WHERE churn_probability >= 0.5
ORDER BY churn_probability DESC, clv_predicted DESC;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE customers IS 'Core customer demographics, services, and billing information';
COMMENT ON TABLE features IS 'Engineered features for machine learning models';
COMMENT ON TABLE predictions IS 'Model predictions including churn, survival, CLV, and segmentation';
COMMENT ON COLUMN features.potential_complaint_flag IS 'Proxy for potential complaints: month-to-month contract + no tech support + no online security';
COMMENT ON COLUMN predictions.uplift_segment IS 'Customer segment based on uplift modeling (treatment effect)';
