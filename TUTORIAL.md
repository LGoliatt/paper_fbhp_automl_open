# Step-by-Step Tutorials & Typical Use Cases

This document provides practical, step-by-step tutorials for common workflows in `paper_fbhp_automl_open`. All commands assume your terminal working directory is the root of the cloned repository.

---

## Tutorial 1: Environment Setup and Quick Verification

**Goal:** Verify your local Python installation and execute an end-to-end test run in less than 30 seconds.

1. **Activate your Python environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run the quickstart verification script:**
   ```bash
   python examples/run_quickstart_demo.py
   ```

3. **Inspect the generated test outputs:**
   The script creates an isolated folder `examples/demo_output/` containing:
   * `synthetic_fbhp_sample.csv` — synthetic 200-well multiphase dataset
   * `demo_summary.csv` — 5-fold cross-validation metrics ($R^2$, $\text{RMSE}$, $\text{MAE}$, $\text{MAPE}$)
   * `demo_metrics.json` — structured JSON log matching the repository's logging schema

4. **Expected console output:**
   You should see a summary table printed to the console with $R^2 > 0.90$ on the synthetic test benchmark and a confirmation message: `Quickstart verification completed successfully.`

---

## Tutorial 2: Reproducing Manuscript Tables and Figures

**Goal:** Regenerate the exact performance tables (`tab/`) and figures (`img/`) reported in the manuscript from the archived cross-validation JSON logs in `json_automl_f1/`.

1. **Verify that the precomputed JSON logs are present:**
   ```bash
   ls -la json_automl_f1/
   ```

2. **Run the comparison and visualization script:**
   ```bash
   python fbhp_automl_comparison_v_0p3.py
   ```
   *(Alternatively, run `make comparison`.)*

3. **Review the generated artifacts:**
   * Check `tab/` for updated summary tables (`.csv`, `.xlsx`, `.tex`).
   * Check `img/` for updated high-resolution figures (`.png`, `.pdf`).

---

## Tutorial 3: Re-Running the AutoML Training Pipeline

**Goal:** Re-execute the data ingestion and AutoML training loop from scratch on the dataset in `data/`.

1. **Inspect the dataset loader first:**
   ```bash
   python read_data_fbhp.py
   ```
   Confirm that the input dataset dimensions and column names match the schema documented in [`USER_GUIDE.md`](USER_GUIDE.md).

2. **(Optional) Back up existing JSON results:**
   If you want to preserve the archived logs before overwriting or appending new runs:
   ```bash
   cp -r json_automl_f1 json_automl_f1_backup
   ```

3. **Execute the AutoML analysis script:**
   ```bash
   python fbhp_automl_analysis_v_0p3.py
   ```
   *(Alternatively, run `make analysis`.)*

4. **Re-generate updated comparison tables and plots:**
   ```bash
   python fbhp_automl_comparison_v_0p3.py
   ```

---

## Tutorial 4: Adapting the Workflow to a Custom Well Dataset

**Goal:** Evaluate the AutoML pipeline on your own field or simulated multiphase well dataset.

1. **Prepare your dataset in uncompressed CSV format:**
   Ensure your file includes the 9 standard input columns (`Qo`, `Qg`, `Qw`, `Id`, `Depth`, `Api`, `St`, `Bt`, `Pwh`) and the target column (`Pwf` or `FBHP`), all in consistent oilfield units ($\text{bbl/day}$, $\text{mscf/day}$, $\text{in}$, $\text{ft}$, $^{\circ}\text{API}$, $^{\circ}\text{F}$, $\text{psia}$).

2. **Place the file inside `data/`:**
   ```bash
   cp /path/to/your_well_data.csv data/custom_fbhp_data.csv
   ```

3. **Update the input filename reference in `read_data_fbhp.py`:**
   Point the loader function in `read_data_fbhp.py` to `data/custom_fbhp_data.csv`.

4. **Run the full pipeline:**
   ```bash
   make all
   ```
   The new AutoML evaluation records will be saved in `json_automl_f1/`, and updated tables and figures will be exported to `tab/` and `img/`.