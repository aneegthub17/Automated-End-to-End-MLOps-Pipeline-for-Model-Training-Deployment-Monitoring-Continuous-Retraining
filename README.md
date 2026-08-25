# Automated End-to-End MLOps Pipeline for Model Training, Deployment, Monitoring & Continuous Retraining

An end-to-end Machine Learning Operations (MLOps) pipeline for customer churn prediction.

The project automates the complete ML lifecycle:

- Data ingestion
- Data validation
- Data profiling
- Data preprocessing
- Model training
- Model evaluation
- MLflow experiment tracking
- Production model registry
- FastAPI deployment
- Prediction monitoring
- Data drift detection
- Automated retraining
- Candidate model evaluation
- Safe model promotion
- Automated CI testing with GitHub Actions

The system is designed so that a newly trained model is **not automatically promoted to production**. A candidate model must first demonstrate a meaningful improvement over the existing production model.

---

# Project Architecture

```mermaid
flowchart TD

    A[Customer Churn Dataset] --> B[Data Ingestion]

    B --> C[Data Validation]

    C --> D[Data Profiling]

    D --> E[Data Splitting]

    E --> F[Data Preprocessing]

    F --> G[Model Training]

    G --> H[Model Evaluation]

    H --> I[MLflow Experiment Tracking]

    H --> J[Best Model]

    J --> K[Production Model Registry]

    K --> L[FastAPI Prediction API]

    L --> M[Prediction Monitoring]

    M --> N[Data Drift Detection]

    N --> O{Significant Drift?}

    O -->|No| M

    O -->|Yes| P[Automated Retraining]

    P --> Q[Candidate Model]

    Q --> R[Candidate Evaluation]

    R --> S{Candidate Better?}

    S -->|Yes| K

    S -->|No| T[Keep Existing Production Model]

    U[GitHub Actions] --> V[Automated Tests]

    V --> W[CI Validation]

    W --> L