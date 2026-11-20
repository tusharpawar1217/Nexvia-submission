-- ============================================================================
-- PostgreSQL Database Setup Script
-- Run this as postgres superuser to create the churn database and user
-- ============================================================================
-- 
-- Usage:
-- 1. Open PowerShell
-- 2. Run: psql -U postgres -f sql/setup_db.sql
-- 3. You will be prompted for the postgres password
--
-- Or run interactively:
-- 1. psql -U postgres
-- 2. Copy and paste the commands below
-- ============================================================================

-- Create database
CREATE DATABASE churn_db;

-- Create user
CREATE USER churn_user WITH PASSWORD 'churn123';

-- Grant database privileges
GRANT ALL PRIVILEGES ON DATABASE churn_db TO churn_user;

-- Connect to the new database
\c churn_db

-- Grant schema privileges (PostgreSQL 15+ requires this)
GRANT ALL ON SCHEMA public TO churn_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO churn_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO churn_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO churn_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO churn_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO churn_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO churn_user;

-- Verify setup
SELECT 'Database setup complete!' as status;
SELECT current_database() as database;
SELECT current_user as user;

-- ============================================================================
-- VERIFICATION QUERIES
-- Run these after setup to verify everything works
-- ============================================================================

-- Check user privileges
-- SELECT grantee, privilege_type 
-- FROM information_schema.role_table_grants 
-- WHERE grantee = 'churn_user';

-- Test connection
-- \q
-- psql -U churn_user -d churn_db -c "SELECT version();"
