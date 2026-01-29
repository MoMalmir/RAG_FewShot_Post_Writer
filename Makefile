PYTHON = python3

install:
	$(PYTHON) -m pip install --upgrade pip && \
	$(PYTHON) -m pip install -r requirements.txt

# Testing LLM model connectivity
check-api:
	@echo "Checking API connectivity..."
	$(PYTHON) src/test_api.py
	
format:
	black src/*.py

lint:
	pylint --disable=R,C src/*.py

# Make 'run' depend on 'check-api' to ensure API connectivity is tested before running
run: check-api
	$(PYTHON) src/main.py

all: install format lint