# ML Explainer API

A backend-focused Flask application for processing datasets, training machine learning models, and generating interpretable predictions using SHAP and LIME.

## Overview
This project provides RESTful API endpoints that support a machine learning workflow, including dataset ingestion, model training, and explainability generation.

## Features
- CSV dataset upload and validation
- REST API endpoints for backend processing
- Machine learning model training using scikit-learn
- Model explainability using SHAP and LIME
- JSON-based responses for frontend integration
- Backend logging for request tracking

## API Endpoints

| Endpoint | Method | Description |
|--------|--------|-------------|
| /upload | POST | Upload dataset |
| /train | POST | Trigger model training |
| /explain | GET | Generate model explanations |
| /health | GET | Check system status |
| /info | GET | Retrieve project metadata |

## My Contributions
- Implemented RESTful API endpoints to handle dataset ingestion, model training, and explanation workflows
- Developed dataset validation logic to ensure data integrity and handle edge cases
- Integrated machine learning models into backend processing pipelines
- Added logging and debugging to improve system reliability and trace request flow

## Tech Stack
- Python, Flask
- Pandas, Scikit-learn
- SHAP, LIME
- SQLite

## How to Run

```bash
pip install -r requirements.txt
python src/app1.py