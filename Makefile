PYTHON=python3.10
VENV=.venv

.PHONY: setup format lint test clean

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

format:
	. $(VENV)/bin/activate && black src tests

lint:
	. $(VENV)/bin/activate && ruff check src tests

test:
	. $(VENV)/bin/activate && pytest -q

report:
	. $(VENV)/bin/activate && python scripts/generate_report.py

train-svm:
	. $(VENV)/bin/activate && python -m suicide_detection.training.train --model svm --data_path data/processed/dataset.csv --output_dir results/model_outputs/svm

train-bilstm:
	. $(VENV)/bin/activate && python -m suicide_detection.training.train --model bilstm --data_path data/processed/dataset.csv --output_dir results/model_outputs/bilstm

train-bert:
	. $(VENV)/bin/activate && python -m suicide_detection.training.train --model bert --data_path data/processed/dataset.csv --output_dir results/model_outputs/bert

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache

lock:
	. $(VENV)/bin/activate || true; bash scripts/compile_requirements.sh || true

dvc-repro:
	dvc repro || true

