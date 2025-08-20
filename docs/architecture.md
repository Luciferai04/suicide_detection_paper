# System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Suicide Risk Detection Research Pipeline                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐
│   Data Sources  │    │  Data Ingestion │    │      Preprocessing & ETL        │
├─────────────────┤    ├─────────────────┤    ├─────────────────────────────────┤
│ • Kaggle        │───▶│ • Kaggle API    │───▶│ • Anonymization (PII removal)   │
│ • Reddit Posts  │    │ • Manual DL     │    │ • Text cleaning & normalization │
│ • Mendeley      │    │ • Git LFS       │    │ • Train/Val/Test splitting      │
│ • MentalLLaMA   │    │ • DVC tracking  │    │ • Feature engineering (TF-IDF)  │
└─────────────────┘    └─────────────────┘    └─────────────────────────────────┘
                                                             │
        ┌──────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Model Training Hub                                  │
├─────────────────┬─────────────────┬─────────────────────────────────────────┤
│   SVM Baseline  │   BiLSTM+Attn   │           BERT/RoBERTa                  │
├─────────────────┼─────────────────┼─────────────────────────────────────────┤
│ • TF-IDF feats  │ • Word embeddings│ • Transformer fine-tuning              │
│ • GridSearch    │ • Attention vis │ • HuggingFace integration               │
│ • SMOTE         │ • Seq modeling  │ • GPU/MPS optimization                  │
│ • Class weights │ • Dropout/L2    │ • Gradient accumulation                 │
└─────────────────┴─────────────────┴─────────────────────────────────────────┘
        │                   │                           │
        └───────────────────┼───────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Evaluation & Metrics Engine                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ • ROC/PR AUC curves        • Statistical tests (McNemar, Bootstrap CI)      │
│ • Confusion matrices       • Cross-dataset generalization                   │
│ • Clinical metrics         • Error analysis & interpretability              │
│ • Fairness auditing        • Performance comparison tables                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                ┌──────────────────────────────────────┐
                │                      │               │
                ▼                      ▼               ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐
    │ MLflow Tracking │    │   Bias Audits   │    │  Reporting   │
    ├─────────────────┤    ├─────────────────┤    ├──────────────┤
    │ • Experiment    │    │ • Demographic   │    │ • HTML report│
    │   logging       │    │   parity        │    │ • Plots/figs │
    │ • Model registry│    │ • Equalized     │    │ • CSV tables │
    │ • Artifact mgmt │    │   opportunity   │    │ • Paper LaTeX│
    └─────────────────┘    │ • Fairness      │    └──────────────┘
                           │   metrics       │
                           └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            Quality Assurance                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ • GitHub Actions CI/CD     • Pre-commit hooks (black, ruff, mypy)           │
│ • Automated testing        • Ethics review checkpoints                      │
│ • Code quality checks      • Documentation generation                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       Human-in-the-Loop Oversight                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ • IRB approval required    • No clinical deployment without validation      │
│ • Privacy-by-design        • Responsible AI principles enforced             │
│ • Manual result review     • Bias mitigation strategies                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

- **Ethics-First Design**: Built with privacy protection, bias auditing, and human oversight
- **Multi-Model Comparison**: SVM baseline, BiLSTM with attention, BERT fine-tuning
- **Comprehensive Evaluation**: Clinical metrics, statistical testing, fairness analysis
- **Research Reproducibility**: Config-driven, version-controlled, automated CI/CD
- **Data Protection**: Anonymization, secure storage, no PII in git history

## Technology Stack

- **ML Frameworks**: scikit-learn, PyTorch, HuggingFace Transformers
- **Experiment Tracking**: MLflow, DVC
- **Data Processing**: pandas, spaCy, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **CI/CD**: GitHub Actions, pre-commit hooks
- **Environment**: Python 3.10+, conda/venv
