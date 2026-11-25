# Flowing Bottom-Hole Pressure Prediction Using AutoML

This repository contains code, input workbooks, archived JSON results, and analysis artifacts for a study of flowing bottom-hole pressure (FBHP, also denoted as `Pwf`) prediction in multiphase oil and gas wells.

**Repository:** https://github.com/LGoliatt/paper_fbhp_automl_open

> Before submission, include the public repository URL in the manuscript's Code Availability or Data and Code Availability statement. This repository does not contain the manuscript itself, so the manuscript link cannot be verified here.

## What is included

- `read_data_fbhp.py`: loads the F1 (Al-Shammari) and F2 (Nwanwe) workbooks.
- `fbhp_automl_comparison_v_0p3.py`: trains the selected AutoML frameworks and writes experiment records as JSON. Despite its filename, this is the training script.
- `fbhp_automl_analysis_v_0p3.py`: reads experiment JSON files and produces statistical summaries, tables, and figures. Despite its filename, this is the analysis/plotting script.
- `json_automl_f1/`: archived run records used as inputs to the analysis script.
- `data/`: Excel input workbooks. Check the data-source terms before redistributing or reusing them.
- `img/` and `tab/`: figures and tables already present in the repository.
- `examples/run_quickstart_demo.py`: a small synthetic-data smoke test using scikit-learn estimators. It is not the paper's AutoML benchmark and does not validate all seven AutoML backends.
- `USER_GUIDE.md` and `TUTORIAL.md`: user and workflow documentation.
- `LICENSE`: GNU General Public License, version 2.

## Repository Structure

```text
paper_fbhp_automl_open/
├── data/                                # Excel input workbooks for the FBHP datasets
├── json_automl_f1/                      # Archived AutoML run records in JSON format
├── tab/                                 # Existing analysis tables, primarily LaTeX
├── img/                                 # Existing figures and comparative plots
├── examples/
│   └── run_quickstart_demo.py           # Synthetic-data scikit-learn smoke test
├── read_data_fbhp.py                    # Dataset loading and train/test split functions
├── fbhp_automl_comparison_v_0p3.py      # AutoML training and JSON result generation
├── fbhp_automl_analysis_v_0p3.py        # Statistical analysis, tables, and plotting
├── makefile                             # Convenience targets for install, test, train, analyze
├── requirements.txt                     # Base Python dependencies
├── environment.yml                      # Conda environment specification
├── USER_GUIDE.md                        # Inputs, outputs, options, and known limitations
├── TUTORIAL.md                          # Step-by-step workflows
├── LICENSE                              # GNU General Public License, version 2
└── README.md                            # Project overview, installation, and usage
```

## Requirements

The analysis and demonstration scripts require Python and the packages listed in `requirements.txt`. `environment.yml` describes a Conda environment based on Python 3.10. The main training script also has optional framework-specific dependencies. Their installation is platform- and version-sensitive; install only the frameworks you intend to run, and record the exact package versions and environment used for the reported experiments. The repository currently does not provide a complete, version-pinned environment for all seven backends.

The full AutoML training workload can require substantially more time, memory, and disk space than the smoke test. Resource needs depend on the selected frameworks, time budget, and number of runs. GPU is not required by the scripts' default configuration, although an individual framework may support or configure accelerators.

## Installation

From the repository root:

```bash
git clone https://github.com/LGoliatt/paper_fbhp_automl_open.git
cd paper_fbhp_automl_open
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, create the Conda environment:

```bash
conda env create -f environment.yml
conda activate fbhp-automl
```

The base requirements do not install the optional AutoML backends. Install the dependencies for each backend you select, following that framework's installation guidance and compatibility constraints. For `--frameworks`, use the names accepted by the training script, such as `FLAML`, `TPOT`, `Fedot`, `AutoGluon`, `AutoKeras`, `H2O`, and `AutoSklearn`.

## Quick smoke test

After installing the base requirements, run:

```bash
python examples/run_quickstart_demo.py
```

The script creates `examples/demo_output/` with a generated synthetic CSV, fold-level metrics in JSON, and a summary CSV. It trains Extra Trees, Random Forest, and Gradient Boosting models. These results are only a software smoke test; they are not the manuscript results and are not a substitute for the AutoML experiment.

## Reproducing and analyzing results

### Analyze the archived experiment records

The JSON files in `json_automl_f1/` are consumed by the analysis script. Run:

```bash
python fbhp_automl_analysis_v_0p3.py \
  --input-dir ./json_automl_f1 \
  --prefix f1_ \
  --fig-dir ./img \
  --tab-dir ./tab
```

The script generates analysis outputs from the JSON records. It may overwrite files in the selected output directories. The exact outputs depend on the content of the input records and on the code version.

### Run AutoML training

The training script's defaults are 30 runs and a 240-second time budget per AutoML run. A framework must be installed before it can be selected. For example, after installing FLAML and the base requirements:

```bash
python fbhp_automl_comparison_v_0p3.py \
  --dataset F1 \
  --frameworks FLAML \
  --runs 1 \
  --time-budget 60
```

This is an example smoke run, not the full study protocol. Use the manuscript's precise dataset split, framework versions, seeds, budgets, and other settings to reproduce reported values. Store new JSON results in a separate directory with `--out-dir` to avoid replacing the archived records.

The script accepts `F1`, `F2`, `SYN`, and `CUSTOM` as dataset choices. **The current public repository does not include the synthetic workbook referenced by `SYN` (`data/synthetic/synthetic_fbhp.xlsx`), so that option is not runnable from this checkout without adding the expected file or correcting the loader.** `CUSTOM` requires an Excel workbook with a numeric `FBHP` target and numeric feature columns.

## Documentation

- [User guide](USER_GUIDE.md): inputs, outputs, actual command-line options, and known limitations.
- [Tutorials](TUTORIAL.md): installation, smoke test, analysis of archived records, and a training example.

## Data and licensing

The code is licensed under GPL-2.0 as specified in `LICENSE`. The data workbooks and generated research artifacts may have separate rights or source conditions; the software license should not be assumed to cover third-party data. Verify and document the data license and provenance before redistribution or reuse. The repository contains no `.zip`, `.rar`, or `.7z` archive files in the inspected checkout.

## Citation

Please cite the associated manuscript and this repository using the final bibliographic information. Replace the placeholder below once the manuscript has a stable citation:

```text
Manuscript: [insert full citation and DOI or preprint URL]
Code: https://github.com/LGoliatt/paper_fbhp_automl_open
```

The manuscript should contain the repository link in its Code Availability statement. Do not submit the placeholder citation as final metadata.
