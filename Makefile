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

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache

