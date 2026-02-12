# 🏥 Medicare Hospital Charges Prediction System

Production-ready Machine Learning system that predicts **Average Medicare Reimbursement Payments** using real CMS inpatient hospital data.

This project combines healthcare analytics, statistical modeling, API development, CI/CD automation, and cloud deployment.

---

## 🎯 Problem Statement

Medicare reimbursement is policy-driven and economically significant.  
Hospitals and analysts need structured insight into:

- Payment patterns across DRGs  
- Geographic reimbursement differences  
- High-cost case variability  

This project builds a predictive and observable ML system for that purpose.

---

## 📊 Dataset

**CMS Inpatient Charges Dataset**

Features:
- Total Discharges  
- Provider State  
- DRG Definition  

Target:
- Average Medicare Payments  

Real-world healthcare economic data. No synthetic dataset.

---

## 🧠 Modeling Approach

### Data Processing
- Cleaned currency fields
- OneHotEncoding for categorical variables
- Standard scaling for numeric features
- Reusable scikit-learn pipeline

### Model Comparison
Compared:
- Linear Regression
- Random Forest Regressor

### Statistical Improvement
- Identified heteroscedasticity
- Applied log transformation
- Stabilized variance across payment strata

---

## 📈 Final Model Performance

Log-Transformed Linear Regression:

- R² ≈ 0.88  
- MAE ≈ 1275  
- MAPE ≈ 14%

Linear model outperformed Random Forest, indicating structured reimbursement behavior.

---

## 🏗️ System Architecture

```
Training Pipeline
↓
Model Artifact Generation
↓
FastAPI Backend
↓
/predict | /health | /metrics
↓
Streamlit Frontend
↓
Render Cloud Deployment

```


---

## 🔧 Tech Stack

Backend:
- FastAPI
- Scikit-learn
- Pandas
- NumPy

Frontend:
- Streamlit

Infrastructure:
- GitHub Actions (CI/CD)
- Render (Cloud Deployment)
- Pytest (API Testing)
- Flake8 (Linting)

---

## 📡 API Endpoints

### GET `/`
Service status

### GET `/health`
System health check

### GET `/metrics`
Returns:
- Selected model
- MAE, RMSE, R², MAPE
- API version

### POST `/predict`
Returns predicted Medicare reimbursement:

```json
{
  "predicted_medicare_payment": 7421.34,
  "currency": "USD",
  "model_version": "2.0.0"
}

## ⚙️ MLOps Practices Implemented

- Model artifacts not stored in Git  
- Model retrained automatically in CI  
- Backend tests run before deployment  
- Linting enforced in CI  
- Metrics exposed via API  
- Cross-platform path handling  
- Environment variable configuration  

---

## 🚀 Outcome

This project demonstrates:

- Healthcare data preprocessing  
- Model comparison and statistical reasoning  
- Variance stabilization using log transformation  
- Structured error analysis  
- Production-grade API development  
- CI/CD integration for ML systems  
- Cloud deployment with observability  

It is a complete, reproducible, deployable healthcare ML system — not just a notebook model.
