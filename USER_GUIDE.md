# User Guide — paper_fbhp_automl_open

This document describes the inputs, outputs, configuration options, and expected behaviour of the FBHP AutoML reproducibility package that accompanies the manuscript.

Repository: https://github.com/LGoliatt/paper_fbhp_automl_open

---

## 1. Purpose

The package implements a reproducible pipeline for predicting **Flowing Bottom-Hole Pressure (FBHP / \(P_{wf}\))** in multiphase oil and gas wells using Automated Machine Learning (AutoML) frameworks. It provides:

- data loading and preprocessing for two public field datasets and a synthetic test case;
- multi-framework AutoML training with fixed random seeds;
- structured JSON logging of predictions, metrics and run metadata;
- statistical comparison, tables and publication-ready figures.

---

## 2. Input data

### 2.1 Field datasets (provided)

| Dataset key | Source folder | Description |
|-------------|---------------|-------------|
| `F1` | `data/data_shammari/` | Al-Shammari workbook (`al_shammari.xlsx`) |
| `F2` | `data/data_nwanwe/` | Nwanwe workbook (`nwanwe.xlsx`) with Train / Test sheets |

Loader module: `read_data_fbhp.py`.

Expected target column name: **`FBHP`** (or `Pwf` in the synthetic demo).

Typical input features (oilfield units):

| Feature | Description | Unit |
|---------|-------------|------|
| `Qo` | Oil rate | bbl/day |
| `Qg` | Gas rate | mscf/day |
| `Qw` | Water rate | bbl/day |
| `Id` | Tubing inner diameter | in |
| `Depth` | Well depth / TVD | ft |
| `Api` | Oil gravity | °API |
| `St` | Surface temperature | °F |
| `Bt` | Bottom-hole temperature | °F |
| `Pwh` | Wellhead pressure | psia |

The exact column set is taken from the workbook after dropping identifier columns such as `SN`.

### 2.2 Synthetic dataset (for testing)

`examples/run_quickstart_demo.py` generates a physics-inspired synthetic multiphase dataset (default 220 samples) with the same feature names and target `Pwf`. Outputs are written only under `examples/demo_output/` and **do not** overwrite manuscript artefacts in `tab/` or `img/`.

### 2.3 Custom data

You may point the training script to your own `.xlsx` file (see CLI options below). Requirements:

- numeric target column named `FBHP`;
- numeric input feature columns;
- optional identifier column `SN` (will be dropped).

---

## 3. Main modules and expected behaviour

### 3.1 `read_data_fbhp.py`

**Role:** load and split the chosen dataset.

**Key function:** `read_fbhp(dataset)` where `dataset` is `'F1'` or `'F2'`.

**Returns** a dictionary with:

- `feature_names`, `target_names`
- `X_train`, `y_train`, `X_test`, `y_test`
- `n_samples`, `n_features`, `task` (`'regression'`)
- `reference` (DOI / URL of the data source)

When run as a script it prints basic dataset information.

### 3.2 `fbhp_automl_comparison_v_0p3.py` (training / benchmarking)

**Role:** train selected AutoML frameworks and write one JSON record per framework and run.

**Default frameworks:** Fedot, AutoSklearn, TPOT, AutoGluon, AutoKeras, H2O, FLAML.

**CLI options (selected):**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `F1` | `F1`, `F2`, `SYN` or `CUSTOM` |
| `--data-file` | `None` | Path to custom `.xlsx` (required with `CUSTOM`) |
| `--frameworks` | paper set | Space-separated list of frameworks |
| `--runs` | `30` | Number of independent runs |
| `--start-run` | `0` | First run index (resume support) |
| `--time-budget` | `240` | Seconds per AutoML run |