# Flowing Bottom-Hole Pressure (FBHP) Prediction Using AutoML

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository provides the official open-source code, workflows, evaluation scripts, and reproducibility materials accompanying our manuscript on predicting Flowing Bottom-Hole Pressure (FBHP / $P_{wf}$) in multiphase oil and gas wells using Automated Machine Learning (AutoML).

> **Note for Reviewers & Users:** All datasets, intermediate JSON evaluation logs, output tables, and figures in this repository are stored in open, uncompressed formats (`.csv`, `.xlsx`, `.json`, `.tex`, `.png`, `.pdf`) to guarantee transparency, security compliance, and long-term reusability.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Computational Requirements](#computational-requirements)
3. [Installation Instructions](#installation-instructions)
4. [Quickstart & Basic Usage](#quickstart--basic-usage)
5. [Reproducing Paper Results](#reproducing-paper-results)
6. [Documentation (User Guide & Tutorials)](#documentation-user-guide--tutorials)
7. [Data & Code Availability](#data--code-availability)
8. [License](#license)
9. [Citation](#citation)

---

## Repository Structure

```text
paper_fbhp_automl_open/
├── data/                                # Uncompressed input datasets for FBHP modeling
├── json_automl_f1/                      # Stored AutoML fold predictions, metrics, and configurations (JSON)
├── tab/                                 # Generated summary tables (CSV / LaTeX / Excel)
├── img/                                 # Generated figures and comparative plots
├── examples/
│   └── run_quickstart_demo.py           # Self-contained smoke test & synthetic FBHP demo
├── read_data_fbhp.py                    # Data loading, preprocessing, and train/test splitting module
├── fbhp_automl_analysis_v_0p3.py        # Main AutoML model training, evaluation, and logging script
├── fbhp_automl_comparison_v_0p3.py      # Statistical comparison, table export, and plotting script
├── Makefile                             # Automated workflow targets for testing and reproduction
├── requirements.txt                     # Python package dependencies
├── environment.yml                      # Conda environment specification
├── USER_GUIDE.md                        # Detailed guide on inputs, outputs, options, and expected behavior
├── TUTORIAL.md                          # Step-by-step how-to tutorials for typical use cases
├── LICENSE                              # GNU General Public License v2.0 (GPL-2.0)
└── README.md                            # Project overview and setup instructions
```

---

## Computational Requirements

### Software Environment
* **Operating System:** Linux (Ubuntu 20.04/22.04 LTS recommended; tested on standard x86\_64 Linux environments), macOS, or Windows (via WSL2 for Linux-specific AutoML backends).
* **Python Version:** Python 3.9, 3.10, or 3.11.

### Hardware & Execution Time
* **Minimum Hardware (for reproducing tables/figures from `json_automl_f1/` and running quickstart demos):**
  * CPU: 2+ cores
  * RAM: 4 GB
  * Storage: < 500 MB free disk space
  * Execution time: **< 1 minute**
* **Recommended Hardware (for full AutoML retraining from scratch):**
  * CPU: 8+ cores (multi-core execution accelerates cross-validation and AutoML search)
  * RAM: 16 GB or higher
  * GPU: Not required (all models run on CPU)
  * Execution time: Depends on the configured AutoML search budget per fold (typically 15–90 minutes for a complete multi-framework benchmark).

---

## Installation Instructions

### Option 1: Using `pip` and `venv` (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LGoliatt/paper_fbhp_automl_open.git
   cd paper_fbhp_automl_open
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # On Linux/macOS
   # venv\Scripts\activate         # On Windows
   ```

3. **Install required dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Option 2: Using Conda

```bash
git clone https://github.com/LGoliatt/paper_fbhp_automl_open.git
cd paper_fbhp_automl_open
conda env create -f environment.yml
conda activate fbhp-automl
```

---

## Quickstart & Basic Usage

### 1. Run the Quickstart Verification Demo
To verify that your Python environment is properly configured and test the end-to-end FBHP workflow using a synthetic test case (without overwriting paper results), run:

```bash
python examples/run_quickstart_demo.py
```

Or using `make`:
```bash
make test
```

### 2. Run the Full Pipeline via `Makefile`
The repository includes a `Makefile` to streamline execution:

```bash
make help          # Display all available targets
make test          # Run the self-contained quickstart verification script
make analysis      # Execute AutoML modeling and log results to json_automl_f1/
make comparison    # Generate comparative tables (tab/) and figures (img/)
make all           # Run the complete analysis and comparison pipeline
```

### 3. Run Individual Python Modules Directly
```bash
# Step 1: Verify data ingestion and preprocessing
python read_data_fbhp.py

# Step 2: Run AutoML training and cross-validation analysis
python fbhp_automl_analysis_v_0p3.py

# Step 3: Aggregate results and generate manuscript tables and plots
python fbhp_automl_comparison_v_0p3.py
```

---

## Reproducing Paper Results

1. **Reproducing Tables and Figures from Precomputed AutoML Runs:**
   All fold-level AutoML evaluation outputs generated during the study are archived in `json_automl_f1/`. Running:
   ```bash
   python fbhp_automl_comparison_v_0p3.py
   ```
   will parse the JSON logs in `json_automl_f1/` and regenerate the summary performance tables in `tab/` and the comparison figures in `img/`.

2. **Reproducing AutoML Training from Scratch:**
   Running:
   ```bash
   python fbhp_automl_analysis_v_0p3.py
   ```
   re-executes the data loading (`read_data_fbhp.py`), cross-validation splits (with fixed random seeds for reproducibility), and AutoML model search, saving structured JSON logs into `json_automl_f1/`.

---

## Documentation (User Guide & Tutorials)

* **[`USER_GUIDE.md`](USER_GUIDE.md):** Detailed documentation of input datasets, variable definitions and physical units, JSON result schemas, module parameters, output artifacts, and expected behavior.
* **[`TUTORIAL.md`](TUTORIAL.md):** Step-by-step tutorials covering environment verification, reproducing paper figures/tables, running AutoML benchmarks, and adapting the pipeline to new well datasets.

---

## Data & Code Availability

* All data and result files are provided directly in this repository in uncompressed, human-readable formats (`data/`, `json_automl_f1/`, `tab/`, `img/`). No `.zip`, `.rar`, or `.7z` archives are used.
* If you adapt this workflow to proprietary field data that cannot be publicly distributed, `examples/run_quickstart_demo.py` provides a physics-inspired synthetic FBHP dataset generator for unit and integration testing.

---

## License

This project is licensed under the **GNU General Public License v2.0 (GPL-2.0)** — see the [`LICENSE`](LICENSE) file for full details.

---

## Citation

If you use this code, data, or workflow in your research, please cite our repository and accompanying manuscript:

```bibtex
@misc{goliatt_paper_fbhp_automl_open,
  author       = {Goliatt, Leonardo and collaborators},
  title        = {Flowing Bottom-Hole Pressure Prediction Using Automated Machine Learning (paper\_fbhp\_automl\_open)},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/LGoliatt/paper_fbhp_automl_open}}
}
```