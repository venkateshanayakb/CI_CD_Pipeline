# MLOps CI/CD Pipeline Project

## What is this project?

This project demonstrates an **end-to-end CI/CD pipeline for a Machine Learning application**.

It shows how a trained ML model can be:
- Exposed as an API
- Connected to a user-facing frontend
- Automatically tested and deployed to the cloud

The project follows **real-world MLOps practices**, not just local experimentation.

---

## What does it contain?

- **FastAPI Backend**
  - Serves a trained ML model for prediction
  - Exposes a `/predict` API endpoint
  - Deployed as a cloud service on Render

- **Streamlit Frontend**
  - Provides a simple UI for users
  - Sends input data to the backend API
  - Displays predicted results

- **CI/CD with GitHub Actions**
  - Runs automated tests on every push
  - Deploys backend only if tests pass
  - Ensures reliable and repeatable deployments

- **Cloud Deployment (Render)**
  - Backend and frontend deployed as separate services
  - Environment variables used for configuration
  - No hardcoded secrets or URLs

---

## Why is this needed?

In real-world ML systems:
- Models are not run manually
- Code changes should not break production
- Deployments must be automated and reproducible
- Frontend and backend must be independently scalable

This project solves those problems by:
- Turning an ML model into a production-ready service
- Preventing faulty code from being deployed
- Separating concerns between UI, API, and model
- Demonstrating how MLOps works beyond notebooks

---

## What this project demonstrates

- End-to-end ML deployment
- CI/CD pipelines for ML applications
- Monorepo project structure
- Cloud-native configuration using environment variables
- Practical debugging of deployment issues

---

## Who is this useful for?

- ML and Data Science students learning MLOps
- Engineers transitioning from notebooks to production
- Anyone who wants a **realistic ML deployment example**
- Portfolio demonstration for interviews

---

## Summary

This project shows how a Machine Learning model moves from:
**local code → tested service → automated cloud deployment**

It focuses on **reliability, automation, and real-world readiness**, not just model accuracy.
