SHELL = /bin/bash

.PHONY: help
help:
	@echo "Comandos:"
	@echo "venv    : crea un virtual environment."
	@echo "style   : ejecuta el formato de estilo."
	@echo "clean   : limpia todos los archivos innecesarios."
	@echo "test    : ejecuta tests en código, datos y modelos."

# Styling
.PHONY: style
style:
	black coe_template
	flake8 coe_template
	isort coe_template

# Environment
.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install -e ".[dev]" && \
	pre-commit install && \
	pre-commit autoupdate

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

# Test
.PHONY: test
test:
	pytest -m "not training"
	cd tests && great_expectations checkpoint run tags

.PHONY: dvc
dvc:
	dvc add data/projects.csv
	dvc add data/tags.csv
	dvc add data/labeled_projects.csv
	dvc push
