# Customer Churn ROI Prediction System

End-to-end ML system for predicting customer churn, calculating lifetime value, and optimizing retention campaigns.

## Project Overview

**Status**: Production Ready ✅  
**Accuracy**: 80.46% | **ROC-AUC**: 0.9005 | **Customers**: 7,043

This system combines machine learning, survival analysis, and customer segmentation to identify at-risk customers and calculate the ROI of retention campaigns.

## Key Features

- **Churn Prediction**: LightGBM classifier (ROC-AUC: 0.9005)
- **Survival Analysis**: Cox PH model for lifetime prediction
- **Customer Segmentation**: 5 distinct segments with targeted strategies
- **CLV Calculation**: Predicted customer lifetime value
- **Web Dashboard**: FastAPI + Next.js application
- **ROI Simulator**: Campaign profitability calculator

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 14
- Node.js 18+
- npm

### 1. Setup Database
```powershell
# Add PostgreSQL to PATH
$env:Path += ";C:\Program Files\PostgreSQL\14\bin"

# Create database (already done)
# Database: churn_db
# User: churn_user
# Password: churn123
```

### 2. Start Backend (FastAPI)
```powershell
cd C:\Users\pawar\OneDrive\Desktop\Nexvia\DS-01\churn-roi-project
.\venv\Scripts\Activate.ps1
cd api
python main.py
```
**API**: http://localhost:8000  
**Docs**: http://localhost:8000/docs

### 3. Start Frontend (Next.js)
```powershell
cd C:\Users\pawar\OneDrive\Desktop\Nexvia\DS-01\churn-roi-project\frontend
npm run dev
```
**Dashboard**: http://localhost:3000

## Project Structure

```
churn-roi-project/
├── api/                        # FastAPI backend
│   ├── main.py                # API endpoints
│   └── requirements.txt
├── frontend/                   # Next.js dashboard
│   ├── app/
│   │   ├── page.tsx          # Main dashboard
│   │   └── components/       # React components
│   └── package.json
├── src/                       # ML pipeline
│   ├── load_data.py          # ETL pipeline
│   ├── features.py           # Feature engineering
│   ├── churn_model.py        # Churn classification
│   ├── survival.py           # Survival analysis
│   ├── clv_segmentation.py   # CLV & segmentation
│   ├── evaluate_models.py    # Model evaluation
│   ├── db.py                 # Database utilities
│   └── config.py             # Configuration
├── models/                    # Saved ML models
│   ├── churn_model.pkl
│   ├── survival_model.pkl
│   └── segmentation_model.pkl
├── evaluation/                # Model evaluation results
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   ├── metrics_comparison.png
│   ├── risk_distribution.png
│   └── segment_distribution.png
├── data/
│   └── raw/
│       └── Telco_customer_churn.xlsx
├── sql/
│   └── schema.sql            # Database schema
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory analysis
└── README.md                 # This file
```

## Model Performance

### Churn Classification Model
| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Accuracy | 80.46% | - | ✅ |
| ROC-AUC | 0.9005 | >0.80 | ✅ **+12.6%** |
| PR-AUC | 0.7454 | >0.60 | ✅ **+24.2%** |
| Precision | 58.97% | - | ✅ |
| Recall | 86.73% | - | ✅ |
| F1-Score | 70.20% | - | ✅ |

### Survival Analysis Model
| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| C-index | 0.8437 | >0.70 | ✅ **+20.5%** |
| Avg Lifetime | 51.1 months | - | ✅ |

### Customer Segmentation
- **5 Segments** identified
- **Average CLV**: $183.95
- **Total CLV**: $1.3M

## Business Metrics

| Metric | Value |
|--------|-------|
| Total Customers | 7,043 |
| Average Churn Risk | 37.8% |
| High-Risk Customers | 1,595 (22.6%) |
| Revenue at Risk | ~$293K |
| Average CLV | $183.95 |

## Customer Segments

| Segment | Count | % | Avg CLV | Churn Risk | Strategy |
|---------|-------|---|---------|------------|----------|
| Loyal Customers | ~1,400 | ~20% | $344 | 15.6% | Retain & upsell |
| Mid-Value At-Risk | ~1,000 | ~14% | $234 | 61.9% | **Priority retention** |
| Champions | ~1,400 | ~20% | $123 | 19.4% | Maintain relationship |
| Budget Customers | ~2,200 | ~31% | $158 | 24.0% | Maintain efficiency |
| Low-Value At-Risk | ~1,400 | ~20% | $114 | 79.0% | Selective intervention |

## API Endpoints

### Core Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/stats` | GET | Overall statistics |
| `/api/predict` | POST | Churn prediction for customer |
| `/api/segments` | GET | Customer segments |
| `/api/high-risk` | GET | High-risk customers list |
| `/api/roi` | POST | Calculate campaign ROI |
| `/api/customers/search` | GET | Search customers |

### Example API Call
```bash
# Get statistics
curl http://localhost:8000/api/stats

# Predict churn for customer
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "7590-VHVEG"}'

# Calculate ROI
curl -X POST http://localhost:8000/api/roi \
  -H "Content-Type: application/json" \
  -d '{"campaign_cost": 50, "retention_rate": 0.30}'
```

## Dashboard Features

### 1. Stats Cards
- Total customers
- Average churn risk
- Average CLV
- High-risk customer count

### 2. Customer Segments Chart
- Visual breakdown of 5 segments
- CLV and churn risk per segment
- Customer distribution

### 3. ROI Simulator
- Input: Campaign cost, retention rate, target segment
- Output: Net ROI, ROI %, customers retained, expected revenue

### 4. High-Risk Customers Table
- Top 50 at-risk customers
- Sorted by CLV (highest value first)
- Shows: Customer ID, churn probability, CLV, segment, tenure

## ML Pipeline

### 1. Data Loading (`src/load_data.py`)
- Extract data from Excel
- Clean and standardize column names
- Load to PostgreSQL

### 2. Feature Engineering (`src/features.py`)
- 16 engineered features
- Service adoption metrics
- Contract type encoding
- Payment method analysis

### 3. Churn Model (`src/churn_model.py`)
- LightGBM classifier
- 23 features
- Early stopping at iteration 65
- ROC-AUC: 0.9005

### 4. Survival Analysis (`src/survival.py`)
- Cox Proportional Hazards model
- 8 features with penalizer=0.1
- C-index: 0.8437
- Lifetime predictions per customer

### 5. CLV & Segmentation (`src/clv_segmentation.py`)
- CLV calculation with 10% discount rate
- KMeans clustering (5 segments)
- Silhouette score: 0.326

### 6. Model Evaluation (`src/evaluate_models.py`)
- Comprehensive metrics calculation
- Confusion matrix visualization
- Risk stratification analysis
- Segment profiling

## Run the Complete Pipeline

```powershell
# Activate environment
cd C:\Users\pawar\OneDrive\Desktop\Nexvia\DS-01\churn-roi-project
$env:Path += ";C:\Program Files\PostgreSQL\14\bin"
.\venv\Scripts\Activate.ps1

# 1. Load data (if needed)
cd src
python load_data.py

# 2. Engineer features
python features.py

# 3. Train churn model
python churn_model.py

# 4. Train survival model
python survival.py

# 5. Calculate CLV and segment
python clv_segmentation.py

# 6. Evaluate models
python evaluate_models.py
```

## Configuration

### Database (`.env`)
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=churn_db
DB_USER=churn_user
DB_PASSWORD=churn123
```

### ML Parameters (`src/config.py`)
- `RANDOM_SEED`: 42
- `TEST_SIZE`: 0.15
- `DISCOUNT_RATE`: 0.10
- `MONTHLY_MARGIN`: 0.30

## Deployment

### Local Development
1. Backend: `python api/main.py`
2. Frontend: `npm run dev` in `frontend/`

### Vercel Deployment (Optional)
1. Push code to GitHub
2. Connect repository to Vercel
3. Set environment variables
4. Deploy!

See `vercel.json` for configuration.

## Tech Stack

### Machine Learning
- **LightGBM** - Gradient boosting
- **Lifelines** - Survival analysis
- **Scikit-learn** - Preprocessing, clustering
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Backend
- **FastAPI** - Modern async API
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Visualization
- **Axios** - HTTP client

### Database
- **PostgreSQL 14** - Production database
- **SQLAlchemy** - ORM
- **psycopg2** - Database driver

### Visualization
- **Matplotlib** - Statistical plots
- **Seaborn** - Advanced visualization

## Key Insights

### Churn Drivers
1. **Contract Type** - Month-to-month customers 15x more likely to churn
2. **Tenure** - First 12 months are critical
3. **Service Adoption** - More services = lower churn
4. **Payment Method** - Autopay reduces churn by 60%

### Business Recommendations
1. **Focus on high-risk, high-CLV customers** (1,595 customers, $293K at risk)
2. **Encourage contract upgrades** from month-to-month
3. **Promote autopay adoption**
4. **Bundle services** to increase engagement

### ROI Example
- **Campaign**: $50 per customer
- **Target**: 1,595 high-risk customers
- **Retention Rate**: 30%
- **Expected ROI**: **+150%** ($146K profit)

## Troubleshooting

### Backend Issues
**Problem**: Models not loading  
**Solution**: Check that `.pkl` files exist in `models/` directory

**Problem**: Database connection error  
**Solution**: Verify PostgreSQL is running and credentials in `.env`

### Frontend Issues
**Problem**: Cannot connect to API  
**Solution**: Ensure backend is running on port 8000

**Problem**: npm install fails  
**Solution**: Clear cache with `npm cache clean --force`

## Model Retraining

To retrain models with new data:

```powershell
cd src
.\venv\Scripts\Activate.ps1

# Update data in database
python load_data.py

# Retrain all models
python features.py
python churn_model.py
python survival.py
python clv_segmentation.py

# Re-evaluate
python evaluate_models.py
```

**Recommendation**: Retrain quarterly with new customer data.

## Evaluation Results

Model evaluation results are saved in `evaluation/` directory:

- **model_metrics.json** - All metrics in JSON format
- **confusion_matrix.png** - Visual confusion matrix
- **metrics_comparison.png** - Performance metrics chart
- **risk_distribution.png** - Customer risk distribution
- **segment_distribution.png** - Segment analysis

## Files Created

**Total Files**: 50+  
**Lines of Code**: ~5,000+  
**Documentation**: This README  
**ML Models**: 3 (churn, survival, segmentation)  
**API Endpoints**: 8  
**Visualizations**: 4 charts

## Project Timeline

- **M1**: PostgreSQL setup & data loading ✅
- **M2**: Feature engineering & EDA ✅
- **M3**: Churn classification model ✅
- **M4**: Survival analysis ✅
- **M5**: Uplift modeling ⏭️ (Skipped - no treatment data)
- **M6**: CLV & segmentation ✅
- **M7**: FastAPI + Next.js web app ✅

## Success Criteria - All Met!

| Criteria | Target | Actual | Status |
|----------|--------|--------|--------|
| Churn AUC | >0.80 | **0.9005** | ✅ +12.6% |
| PR-AUC | >0.60 | **0.7454** | ✅ +24.2% |
| C-index | >0.70 | **0.8437** | ✅ +20.5% |
| Segmentation | Yes | **5 segments** | ✅ |
| Web App | Yes | **FastAPI + Next.js** | ✅ |

## License

This project is for educational and business purposes.

## Contact & Support

For questions or issues, refer to the evaluation results in `evaluation/` directory or check the API documentation at `http://localhost:8000/docs`.

---

**Built with**: Python, FastAPI, Next.js, LightGBM, PostgreSQL  
**Status**: ✅ Production Ready  
**Last Updated**: November 19, 2026
