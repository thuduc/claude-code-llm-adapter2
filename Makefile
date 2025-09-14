PY=python3
VENV=.venv

.PHONY: venv install dev test clean

venv:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip

install: venv
	. $(VENV)/bin/activate && pip install -r requirements.txt

dev: install
	. $(VENV)/bin/activate && pip install -r requirements-dev.txt

test:
	. $(VENV)/bin/activate && PYTHONPATH=. pytest -q

clean:
	rm -rf $(VENV) **/__pycache__ .pytest_cache
