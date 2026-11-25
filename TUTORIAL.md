# Tutorials and Typical Workflows

Run commands from the repository root. The examples below describe the scripts as they currently exist in the repository.

## 1. Install the base environment

With Python 3:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or use Conda:

```bash
conda env create -f environment.yml
conda activate fbhp-automl
```

The base environment does not install all optional AutoML backends. Install the dependencies for the framework you intend to run, then verify its compatibility with your Python and operating-system versions. The repository currently does not specify one fully pinned environment for all seven backends.

## 2. Run the synthetic smoke test

```bash
python examples/run_quickstart_demo.py
```

Expected artifacts are written to `examples/demo_output/`:

- `synthetic_fbhp_sample.csv`: generated synthetic input and target data;
- `demo_summary.csv`: mean and standard deviation of metrics across the five folds for each scikit-learn model;
- `demo_metrics.json`: fold-level metrics and runtime records.

This test uses Extra Trees, Random Forest, and Gradient Boosting from scikit-learn. It does not run the AutoML frameworks or reproduce the manuscript's field-data results. Do not rely on a fixed metric threshold; verify the output generated in your own environment.

## 3. Analyze the archived JSON results

The analysis script reads JSON files from `json_automl_f1/` and writes tables and figures. To inspect all options:

```bash
python fbhp_automl_analysis_v_0p3.py --help
```

Example:

```bash
python fbhp_automl_analysis_v_0p3.py \
  --input-dir ./json_automl_f1 \
  --prefix f1_ \
  --fig-dir ./img \
  --tab-dir ./tab
```

The script may overwrite files in `img/` and `tab/`. To keep the checked-in outputs unchanged, copy the input JSON directory and choose new output directories before running it.

## 4. Run a small AutoML training test

Install the base requirements and the selected AutoML framework first. For example, after installing FLAML:

```bash
python fbhp_automl_comparison_v_0p3.py \
  --dataset F1 \
  --frameworks FLAML \
  --runs 1 \
  --time-budget 60 \
  --out-dir ./results/flaml_smoke
```

This trains one selected framework on F1 and writes JSON results to the specified output folder. It is a small functional test, not the full paper protocol. Use `--help` for the complete options. To reproduce manuscript results, use the exact parameters and software versions reported in the manuscript.

The training script lists `SYN` as an option, but its loader expects `data/synthetic/synthetic_fbhp.xlsx`, which is absent from the current repository. Do not use `--dataset SYN` until that workbook is supplied or the loader is fixed.

## 5. Re-run the paper training workflow

Before launching a full benchmark, preserve the archived records and confirm the experiment protocol. A general command pattern is:

```bash
python fbhp_automl_comparison_v_0p3.py \
  --dataset F1 \
  --frameworks Fedot AutoSklearn TPOT AutoGluon AutoKeras H2O FLAML \
  --runs 30 \
  --start-run 0 \
  --time-budget 240 \
  --out-dir ./results/f1_reproduction
```

This command requires every named framework to be installed and compatible with the environment. The defaults and numbers above mirror the script's CLI defaults; they should not be treated as the verified manuscript protocol unless they match the manuscript. Record Python and package versions, operating system, machine specifications, seeds, and the full command line.

Analyze the resulting JSON folder by passing it to the analysis script:

```bash
python fbhp_automl_analysis_v_0p3.py \
  --input-dir ./results/f1_reproduction \
  --prefix f1_reproduction_ \
  --fig-dir ./results/figures \
  --tab-dir ./results/tables
```

## 6. Use a custom workbook

Prepare an Excel workbook with a numeric `FBHP` target column and numeric feature columns. An optional `SN` identifier column is dropped by the loader. Choose how many leading rows to use for training with `--n-train`; the remaining rows are used for testing. If omitted, the loader uses approximately 75% of the rows for training.

Example:

```bash
python fbhp_automl_comparison_v_0p3.py \
  --dataset CUSTOM \
  --data-file ./data/custom_wells.xlsx \
  --n-train 150 \
  --frameworks FLAML \
  --runs 1 \
  --time-budget 60 \
  --out-dir ./results/custom_smoke
```

Confirm units, missing-value handling, row ordering, and split design before interpreting results. Do not place confidential or restricted field data in a public repository.
