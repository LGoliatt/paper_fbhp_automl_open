# User Guide — `paper_fbhp_automl_open`

This guide reflects the files in the public repository checkout inspected on 2026-11-25. Repository: https://github.com/LGoliatt/paper_fbhp_automl_open

## Purpose and workflow

The repository contains two distinct stages:

1. `fbhp_automl_comparison_v_0p3.py` trains selected AutoML frameworks and writes JSON run records. The filename is misleading: this is the training/benchmarking script.
2. `fbhp_automl_analysis_v_0p3.py` reads JSON records and generates statistical analyses, tables, and figures. This is the analysis script.

`examples/run_quickstart_demo.py` is a separate synthetic-data smoke test. It uses scikit-learn estimators rather than the AutoML frameworks and must not be presented as a reproduction of the manuscript benchmark.

## Inputs

### Public workbooks

`read_data_fbhp.py` implements these dataset keys:

- `F1`: reads `data/data_shammari/al_shammari.xlsx`; the loader uses the `FBHP` target and, by default, the first 596 usable rows for training and the remaining rows for testing.
- `F2`: reads `data/data_nwanwe/nwanwe.xlsx`, using worksheets named `Train` and `Test`. The loader drops identifier/extra columns as coded in `read_data_fbhp.py`.

The exact source, data-use terms, and license for each workbook should be documented separately. The GPL-2.0 code license must not be assumed to license third-party data.

### Custom workbook

The training script supports `--dataset CUSTOM --data-file PATH`. The workbook must contain a numeric column named `FBHP` and numeric input columns. An `SN` identifier column is dropped by the loader if present. `--n-train N` sets the number of leading rows for training; if omitted, the loader uses approximately 75% of the rows. The remaining rows are used for testing. The custom loader reads an Excel workbook.

### Synthetic data limitation

The training CLI lists `SYN` as a dataset choice, but `read_data_fbhp.py` points it to `./data/synthetic/synthetic_fbhp.xlsx`. That file is absent from the inspected repository. Therefore `SYN` cannot be used from the current checkout without supplying the workbook or changing the loader.

The quickstart demo instead generates 220 synthetic rows in memory using a fixed seed, then writes outputs under `examples/demo_output/`. It does not feed those rows through the AutoML training script.

## Installation and dependencies

The base installation is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`environment.yml` provides a Conda environment specification. The base requirements cover common data, plotting, analysis, and smoke-test dependencies. The analysis script imports `permetrics`, `SkillMetrics`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `numpy`, and scikit-learn; the loader also uses `openpyxl` to read Excel files.

The training script imports framework packages defensively, but selected frameworks must be installed to run them. The defaults are `Fedot`, `AutoSklearn`, `TPOT`, `AutoGluon`, `AutoKeras`, `H2O`, and `FLAML`. Framework compatibility depends on Python version, operating system, and versions of its own dependencies. The repository does not currently pin a complete tested combination for all backends. Record the exact environment used for published results and install only the backends selected for a run.

## Training CLI

Show the implemented options with:

```bash
python fbhp_automl_comparison_v_0p3.py --help
```

Options implemented by the training script:

| Option | Default | Meaning |
|---|---:|---|
| `--dataset` | `F1` | `F1`, `F2`, `SYN`, or `CUSTOM`; note the missing `SYN` workbook above. |
| `--data-file` | none | Excel workbook path; required for `CUSTOM`. |
| `--n-train` | none | Number of leading rows for the custom dataset's training split; otherwise approximately 75% is used. |
| `--frameworks` | the seven paper frameworks | One or more framework names. Optional names include `TabPFN`, `EvalML`, `RandomForest`, and `HistGB`. |
| `--runs` | `30` | Upper bound used for run indices; combined with `--start-run`. |
| `--start-run` | `0` | First run index. |
| `--time-budget` | `240` | Per-run time budget in seconds (framework-specific handling applies). |
| `--epochs` | `100` | AutoKeras training epochs. |
| `--pfi-repeats` | `30` | Repetitions for permutation feature importance. |
| `--uncertainty-samples` | `100000` | Samples for uncertainty calculations. |
| `--out-dir` | `./json_automl_<dataset>` | Directory for the generated JSON files. |

Example, after installing FLAML and the base requirements:

```bash
python fbhp_automl_comparison_v_0p3.py \
  --dataset F1 --frameworks FLAML --runs 1 --time-budget 60 \
  --out-dir ./results/flaml_smoke
```

The example is a smoke run only. It does not reproduce the paper's full protocol. The script uses the dataset loader and saves records in JSON format. Existing JSON records should be preserved by choosing a new `--out-dir` for new runs.

## Analysis CLI

Show the implemented options with:

```bash
python fbhp_automl_analysis_v_0p3.py --help
```

Options implemented by the analysis script:

| Option | Default | Meaning |
|---|---|---|
| `--input-dir` | `./json_automl_f1` | Directory containing JSON run records. |
| `--prefix` | `f1_` | Prefix used for analysis output names. |
| `--fig-dir` | `./img` | Figure output directory. |
| `--tab-dir` | `./tab` | Table output directory. |
| `--exclude` | `AutoKeras TabPFN TabICL` | Estimators omitted from the analyses; pass `--exclude` with no values to retain all. |

Run the archived-result analysis with:

```bash
python fbhp_automl_analysis_v_0p3.py \
  --input-dir ./json_automl_f1 --prefix f1_ \
  --fig-dir ./img --tab-dir ./tab
```

This command writes or may overwrite analysis artifacts in the requested output directories. Use copies or new output folders if you need to preserve the checked-in figures and tables.

## Quickstart smoke test

```bash
python examples/run_quickstart_demo.py
```

This generates `examples/demo_output/synthetic_fbhp_sample.csv`, `demo_summary.csv`, and `demo_metrics.json`. It uses 5-fold cross-validation with three scikit-learn regressors. The script itself prints its completion message, but successful execution and metric values must be verified in the target environment; the results are synthetic and are not evidence for the manuscript's field-data results.

## Reproducing manuscript results

Use the exact data split, framework versions, run count, random seeds, time/epoch budgets, and post-processing procedure stated in the manuscript. The checked-in JSON logs can be analyzed with the command above. The repository alone does not establish that the JSON files can be regenerated byte-for-byte or that every reported manuscript value is reproduced; verify the result against the paper and preserve the exact execution environment.

## Known repository issues to resolve

- The `SYN` training option references a workbook that is not present.
- The base dependency files should include the analysis import `permetrics` and keep `requirements.txt` and `environment.yml` consistent.
- Documentation should identify the training script and analysis script by their actual roles, as described here.
- A manuscript link cannot be confirmed from the repository. Add the public repository URL to the manuscript's Code Availability statement.
- Confirm third-party data rights and document workbook provenance before submission.
