# Suicide Risk Detection Research Implementation

Sensitive content notice: This repository contains code for research into suicide risk detection using social media text. It must be used responsibly and ethically. Do not deploy this code in clinical or production settings without IRB approval, licensed clinician oversight, rigorous validation, and compliance with all applicable laws and policies.

Project goals
- Implement three model families: TF‑IDF+SVM baseline, BiLSTM with attention, and BERT/RoBERTa fine‑tuning
- Provide clinical‑focused evaluation and error analysis
- Enforce strict privacy protection and ethical safeguards
- Enable reproducible research with configs, tracking, and documentation

Repository structure
- data/: raw and processed datasets (not versioned in Git). Place your datasets in data/raw.
- src/suicide_detection/: modular Python package for data processing, models, training, evaluation, bias auditing, and utilities.
- notebooks/: exploration, preprocessing, and model training notebooks.
- results/: model outputs, visualizations, and comparison tables.
- ethics/: IRB documentation templates, privacy compliance references, and bias audit reports.
- configs/: YAML configuration files for models, data paths, and evaluation options.
- scripts/: CLI utilities for preprocessing, training, and launching MLflow UI.
- tests/: unit tests for critical components.

Ethics and safety (critical)
- Never generate or simulate synthetic suicide‑related data.
- Perform thorough anonymization and PII removal before any modeling.
- Incorporate human‑in‑the‑loop review checkpoints.
- Include bias detection/mitigation; only run audits if demographic fields exist.
- Document all assumptions and limitations; prioritize sensitivity to high‑risk cases.

Quick start
1) Create and activate a Python 3.10 virtual environment.
2) Install dependencies with pip using requirements.txt.
3) Datasets (ethics-first; no synthetic data generated):
   - Kaggle SuicideWatch: set Kaggle API creds, then run scripts/download_kaggle.py and scripts/prepare_kaggle.py
   - Mendeley datasets: run scripts/download_mendeley.py (or download manually), then scripts/prepare_mendeley.py
   - MentalLLaMA: run scripts/clone_mentallama.py (optional; not used for classification directly)
4) Train models via scripts/ (uses data/splits if present, or data/raw/dataset.csv by default).

Disclaimer: This project is for research purposes only and is not a diagnostic tool.

