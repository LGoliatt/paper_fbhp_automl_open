# Makefile for paper_fbhp_automl_open
# Provides standardized commands for environment setup, verification, and reproducibility.

PYTHON ?= python3
PIP ?= pip

.PHONY: help install test analysis comparison all clean

help:
	@echo "Available targets for paper_fbhp_automl_open:"
	@echo "  make install     - Install Python dependencies from requirements.txt"
	@echo "  make test        - Run the quickstart verification script (examples/run_quickstart_demo.py)"
	@echo "  make analysis    - Run AutoML training and evaluation (fbhp_automl_analysis_v_0p3.py)"
	@echo "  make comparison  - Generate tables (tab/) and figures (img/) from json_automl_f1/"
	@echo "  make all         - Run both analysis and comparison pipelines sequentially"
	@echo "  make clean       - Remove temporary Python cache files and demo outputs"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) examples/run_quickstart_demo.py

analysis:
	$(PYTHON) read_data_fbhp.py
	$(PYTHON) fbhp_automl_analysis_v_0p3.py

comparison:
	$(PYTHON) fbhp_automl_comparison_v_0p3.py

all: analysis comparison

clean:
	rm -rf __pycache__ examples/__pycache__ examples/demo_output
	rm -rf structured_data_regressor __pycache__ *.png *.tex table_best_models___f1__test* *.log AutogluonModels 
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +