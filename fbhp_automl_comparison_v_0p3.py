#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FBHP AutoML Benchmark

This script compares several AutoML frameworks on FBHP datasets.

Main steps:
- Load FBHP datasets using `read_fbhp`.
- Train multiple AutoML frameworks (AutoSklearn, TPOT, AutoGluon, AutoKeras,
  H2O AutoML, FLAML; optionally Fedot and TabPFN).
- Evaluate models using R² and RMSE, and compute permutation feature
  importance (see `permutation_feature_importance` and scikit-learn's
  permutation importance documentation for the underlying concept). [web:6][web:9][web:35]
- Perform a simple uncertainty analysis by sampling the input space.
- Save experiment metadata and results as JSON, converting NumPy types via
  `converter_numpy`. [web:33][web:36]
"""

# =============================================================================
# Imports
# =============================================================================
import os
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    # make_scorer,
    # root_mean_squared_error,
)

# AutoML frameworks
autosklearn_v1 = False
try:
    # Preferred: official auto-sklearn
    from autosklearn.regression import AutoSklearnRegressor
    autosklearn_v1 = True
except ImportError:
    # Fallback: lightweight auto_sklearn2 package
    from auto_sklearn2 import AutoSklearnRegressor

try:
    from evalml.automl import AutoMLSearch
except ImportError:
    AutoMLSearch = None  # EvalML is optional

from flaml import AutoML
from tpot import TPOTRegressor
from autokeras import StructuredDataRegressor
from autogluon.tabular import TabularPredictor
from fedot.api.main import Fedot

tbpfn_ok = False
try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    tbpfn_ok = True
except ImportError:
    TabPFNRegressor = None  # TabPFN is optional

import h2o
from h2o.automl import H2OAutoML

# Local data loader (provides `read_fbhp` and dataset structure)
from read_data_fbhp import read_fbhp


# =============================================================================
# Global configuration and compatibility
# =============================================================================
warnings.filterwarnings("ignore")

# Legacy NumPy aliases (kept for compatibility with older code)
np.float = np.float64   # type: ignore[attr-defined]
np.bool = np.bool_      # type: ignore[attr-defined]
np.object = object      # type: ignore[attr-defined]

pd.options.display.float_format = "{:.3f}".format

# Initialize H2O cluster
h2o.init()

# Experiment configuration
BASENAME = "fbhp_automl_"
TIME_BUDGET = 240          # seconds per AutoML run
N_RUNS = 30
EPOCHS = 100               # for AutoKeras
DEFAULT_SCORING = "neg_root_mean_squared_error"
N_REPEATS = 30            # for feature importance

# =============================================================================
# Helper functions
# =============================================================================
def converter_numpy(obj):
    """
    Convert NumPy scalars/arrays to native Python types for JSON.

    This helper is used as `default=converter_numpy` in `json.dump` to
    avoid "NumPy array is not JSON serializable" errors when saving
    experiment results. [web:33][web:36]

    Parameters
    ----------
    obj : Any
        Object possibly containing NumPy dtypes or ndarrays.

    Returns
    -------
    Any
        JSON-serializable Python object.

    Raises
    ------
    TypeError
        If the object type is not supported.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Objeto do tipo {type(obj)} não é serializável em JSON")


def permutation_feature_importance(
    model_name,
    model,
    X_test,
    y_test,
    feature_names,
    target,
    scoring_func=mean_squared_error,
    n_repeats=100,
):
    """
    Compute permutation feature importance for a fitted model.

    Follows the same idea as scikit-learn's permutation importance:
    shuffle each feature column independently and measure the degradation
    in performance (increase in the chosen metric). [web:6][web:9][web:35]

    Parameters
    ----------
    model_name : str
        Framework name ('AutoGluon', 'H2O', 'FLAML', etc.), used to choose
        the correct prediction API.
    model : object
        Trained model with a `predict` method.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True target values.
    feature_names : list of str
        Column names of the input features.
    target : str
        Target column name (needed for AutoGluon and H2O).
    scoring_func : callable, default=mean_squared_error
        Metric function `(y_true, y_pred) -> float`.
    n_repeats : int, default=100
        Number of permutations per feature.

    Returns
    -------
    dict
        Mapping `feature_index -> mean_importance`.
    """
    # Baseline score with original (unshuffled) data
    if model_name == "AutoGluon":
        test_data = pd.DataFrame(X_test, columns=feature_names)
        test_data[target] = y_test
        baseline_score = scoring_func(
            y_test,
            model.predict(test_data).values,
        )
    elif model_name == "H2O":
        test_data = pd.DataFrame(X_test, columns=feature_names)
        test_data[target] = y_test
        test_df = h2o.H2OFrame(test_data)
        baseline_score = scoring_func(
            y_test,
            model.predict(test_df).as_data_frame().values.ravel(),
        )
    else:
        baseline_score = scoring_func(y_test, model.predict(X_test))

    feature_importances = {j: [] for j in range(X_test.shape[1])}

    for j in range(X_test.shape[1]):
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            np.random.shuffle(X_shuffled[:, j])

            if model_name == "AutoGluon":
                test_data = pd.DataFrame(X_shuffled, columns=feature_names)
                test_data[target] = y_test
                shuffled_score = scoring_func(
                    y_test,
                    model.predict(test_data).values,
                )
            elif model_name == "H2O":
                test_data = pd.DataFrame(X_shuffled, columns=feature_names)
                test_data[target] = y_test
                test_df = h2o.H2OFrame(test_data)
                shuffled_score = scoring_func(
                    y_test,
                    model.predict(test_df).as_data_frame().values.ravel(),
                )
            else:
                shuffled_score = scoring_func(
                    y_test,
                    model.predict(X_shuffled),
                )

            feature_importances[j].append(shuffled_score - baseline_score)

    return {j: float(np.mean(vals)) for j, vals in feature_importances.items()}


# =============================================================================
# Main experiment loop
# =============================================================================

for run in range(3, N_RUNS):
    # Seed schedule for reproducibility across frameworks
    seed = run * 37 + 1001
    print("=" * 80)
    print(f"Run {run:02d} (seed={seed})")
    print("=" * 80)

    # Datasets to evaluate (extend this list if needed)
    datasets = [
        read_fbhp("F1"),
        # read_fbhp("F2"),
    ]

    for dataset in datasets:
        dataset_base_name = dataset["name"]
        feature_names = dataset["feature_names"]
        task = dataset["task"]
        normalize = dataset["normalize"]

        # Output directory for this dataset
        dr = dataset_base_name.replace(" ", "_").replace("'", "").lower()
        path = f"./json_automl_{dr}/"
        os.system("mkdir -p " + path.replace(" ", "_").replace("-", "_").lower())

        for tk, tn in enumerate(dataset["target_names"]):
            target = dataset["target_names"][tk]
            dataset_name = f"{dataset_base_name}-{tn}"

            X_train = dataset["X_train"]
            X_test = dataset["X_test"]
            y_train = dataset["y_train"][tk]
            y_test = dataset["y_test"][tk]

            n_samples_train = dataset["n_samples"]
            n_features = dataset["n_features"]
            n_samples_test = len(y_test)

            # Dataset summary for this target
            print("-" * 80)
            print(f"Dataset                    : {dataset_name} -- {target}")
            print(f"Output                     : {tn}")
            print(f"Number of training samples : {n_samples_train}")
            print(f"Number of testing  samples : {n_samples_test}")
            print(f"Number of features         : {n_features}")
            print(f"Normalization              : {normalize}")
            print(f"Task                       : {task}")
            print("-" * 80)

            scoring = (
                "f1_micro"
                if task == "classification"
                else DEFAULT_SCORING
            )

            # ----------------------------------------------------------------
            # Data preparation (pandas and H2O)
            # ----------------------------------------------------------------
            train_data = pd.DataFrame(X_train, columns=feature_names)
            train_data[target] = y_train
            test_data = pd.DataFrame(X_test, columns=feature_names)
            test_data[target] = y_test

            train_df = h2o.H2OFrame(train_data)
            test_df = h2o.H2OFrame(test_data)

            # ----------------------------------------------------------------
            # Framework-specific configuration
            # ----------------------------------------------------------------
            # FLAML
            flaml = AutoML()
            flaml_settings = {
                "time_budget": TIME_BUDGET,
                "metric": "rmse",
                "task": "regression",
                "log_file_name": "ucs.log",
                "estimator_list": [
                    "lgbm",
                    "rf",
                    "xgboost",
                    "extra_tree",
                    "xgb_limitdepth",
                    "sgd",
                    "kneighbor",
                    "histgb",
                ],
                "seed": seed,
                "verbose": False,
            }

            # TPOT
            pipeline_optimizer = TPOTRegressor(
                generations=20,
                population_size=20,
                cv=5,
                max_eval_time_mins=TIME_BUDGET / 60.0,
                random_state=seed,
                verbosity=True,
            )

            # AutoSklearn
            include_autosklearn = {
                "regressor": [
                    # "adaboost", "ard_regression", "decision_tree",
                    "extra_trees",
                    "gaussian_process",
                    "gradient_boosting",
                    "k_nearest_neighbors",
                    "liblinear_svr",
                    "libsvm_svr",
                    "mlp",
                    "random_forest",
                    "sgd",
                ],
                # "data_preprocessor": ["no_preprocessing"],
                "feature_preprocessor": ["no_preprocessing"],
            }

            if autosklearn_v1:
                autosklearn_params = {
                    "time_left_for_this_task": TIME_BUDGET,
                    "max_models_on_disc": 5,
                    "memory_limit": 102400,
                    "ensemble_size": 3,
                    "seed": seed,
                    "include": include_autosklearn,
                }
            else:
                autosklearn_params = {
                    "time_limit": TIME_BUDGET,
                    "n_jobs": -1,
                    "random_state": seed,
                    "scoring": "r2",
                    "include": include_autosklearn,
                }

            # Data for uncertainty analysis
            n_outcomes = 100_000
            data = np.random.uniform(
                low=X_test.min(axis=0),
                high=X_test.max(axis=0),
                size=(n_outcomes, X_test.shape[1]),
            )

            # ----------------------------------------------------------------
            # AutoML frameworks to evaluate
            # ----------------------------------------------------------------
            frameworks = [
                "Fedot",
                #"AutoSklearn",
                #"TPOT",
                #"AutoGluon",
                #"AutoKeras",
                #"H2O",
                #"FLAML",
                ## "TabPFN",
            ]

            for auto in frameworks:
                start_time = time.time()
                print(auto)

                # ==========================================================
                # Model training and prediction
                # ==========================================================
                if auto == "EvalML":
                    if AutoMLSearch is None:
                        raise ImportError("EvalML is not installed.")
                    automl = AutoMLSearch(
                        X_train=pd.DataFrame(
                            X_train, columns=feature_names
                        ),
                        y_train=pd.DataFrame(
                            y_train, columns=[target]
                        ),
                        problem_type="regression",
                        max_time=TIME_BUDGET,
                        ensembling=True,
                        random_seed=seed,
                    )
                    automl.search()
                    y_pred = automl.predict(X_test).values.ravel()

                elif auto == "Fedot":
                    automl = Fedot(
                        problem="regression",
                        timeout=TIME_BUDGET / 60,
                        preset="best_quality",
                        n_jobs=-1,
                        seed=seed,
                    )
                    automl.fit(features=X_train, target=y_train)
                    y_pred = automl.predict(X_test)

                elif auto == "TabPFN":
                    if not tbpfn_ok or TabPFNRegressor is None:
                        raise ImportError("TabPFN is not installed.")
                    automl = TabPFNRegressor.create_default_for_version(
                        ModelVersion.V2
                    )
                    automl.fit(X=X_train, y=y_train)
                    y_pred = automl.predict(X_test)

                elif auto == "FLAML":
                    automl = flaml
                    automl.fit(
                        X_train=X_train,
                        y_train=y_train,
                        **flaml_settings,
                    )
                    y_pred = automl.predict(X_test)

                elif auto == "TPOT":
                    automl = pipeline_optimizer
                    automl.fit(X_train, y_train)
                    y_pred = automl.predict(X_test)

                elif auto == "AutoGluon":
                    automl = TabularPredictor(
                        label=target,
                        learner_kwargs={"random_state": seed},
                    ).fit(
                        train_data=train_data,
                        verbosity=False,
                    )
                    y_pred = automl.predict(test_data).values

                elif auto == "AutoSklearn":
                    automl = AutoSklearnRegressor(**autosklearn_params)
                    automl.fit(X_train, y_train)
                    y_pred = automl.predict(X_test)

                elif auto == "AutoKeras":
                    automl = StructuredDataRegressor(
                        max_trials=50,
                        column_names=list(feature_names),
                        overwrite=True,
                        loss="mean_absolute_error",
                        seed=seed,
                    )
                    automl.fit(
                        x=X_train,
                        y=y_train,
                        epochs=EPOCHS,
                        verbose=False,
                    )
                    y_pred = automl.predict(X_test).ravel()

                elif auto == "H2O":
                    automl = H2OAutoML(
                        max_runtime_secs=TIME_BUDGET,
                        seed=seed,
                        sort_metric="rmse",
                    )
                    automl.train(
                        x=list(feature_names),
                        y=target,
                        training_frame=train_df,
                    )
                    y_pred = (
                        automl.predict(test_df)
                        .as_data_frame()
                        .values
                        .ravel()
                    )

                else:
                    continue

                elapsed_time = time.time() - start_time
                print(
                    ">> ",
                    run,
                    auto,
                    "\t\t",
                    r2_score(y_test, y_pred),
                    elapsed_time,
                )

                # ==========================================================
                # Permutation feature importance
                # ==========================================================
                feature_importances = permutation_feature_importance(
                    model_name=auto,
                    model=automl,
                    X_test=X_test,
                    y_test=y_test,
                    feature_names=list(feature_names),
                    target=target,
                    scoring_func=mean_squared_error,
                    n_repeats=N_REPEATS,
                )

                # ==========================================================
                # Uncertainty analysis
                # ==========================================================
                if auto in [
                    "Fedot",
                    "TabPFN",
                    "FLAML",
                    "TPOT",
                    "AutoSklearn",
                    "AutoKeras",
                ]:
                    predict = automl.predict(data)
                    if hasattr(predict, "ravel"):
                        predict = predict.ravel()
                elif auto == "AutoGluon":
                    data_df = pd.DataFrame(
                        data, columns=feature_names
                    )
                    data_df[target] = 0
                    predict = automl.predict(data_df).values
                elif auto == "H2O":
                    data_df = pd.DataFrame(
                        data, columns=feature_names
                    )
                    data_df[target] = 0
                    data_df_h2o = h2o.H2OFrame(data_df)
                    predict = automl.predict(
                        data_df_h2o
                    ).as_data_frame().values.ravel()
                else:
                    raise ValueError(
                        f"Unknown AutoML framework in uncertainty analysis: {auto}"
                    )

                median = float(np.median(predict))
                mad = float(np.abs(predict - median).mean())
                uncertainty = 100 * mad / median if median != 0 else np.nan

                print(
                    auto,
                    median,
                    mad,
                    n_features,
                    uncertainty / n_features,
                    uncertainty,
                )

                mask = y_pred > -1e12
                unc_p = y_pred.reshape(-1)[mask.flatten()]
                unc_t = y_test.reshape(-1)[mask.flatten()]
                residuals = unc_p - unc_t
                unc_m = float(residuals.mean())
                if len(residuals) > 1:
                    unc_s = float(
                        np.sqrt(
                            np.sum((residuals - unc_m) ** 2)
                            / (len(residuals) - 1)
                        )
                    )
                else:
                    unc_s = float("nan")
                pei95 = [unc_m - 1.96 * unc_s, unc_m + 1.96 * unc_s]

                print(auto, unc_m, unc_s, pei95)

                # ==========================================================
                # Result packaging and JSON export
                # ==========================================================
                record = {
                    "run": run,
                    "elapsed_time": elapsed_time,
                    "seed": seed,
                    "estimator": auto,
                    "y_pred": y_pred.tolist(),
                    "y_test": y_test.tolist(),
                    "dataset": dataset_name,
                    "target": target,
                    "feature_names": list(feature_names),
                    "feature_importances": list(
                        feature_importances.values()
                    ),
                    "mad": mad,
                    "uncertainty": uncertainty,
                    "pei95": pei95,
                    "unc_m": unc_m,
                }

                pk = (
                    path
                    + BASENAME
                    + "_run_"
                    + f"{run:02d}_"
                    + f"{dataset_name:>15}".replace(" ", "_")
                    + f"{auto:>15}".replace(" ", "_")
                    + f"{target:>15}".replace(" ", "_")
                    + ".json"
                )

                for ch in [" ", "'", "(", ")", "[", "]", "-", "{", "}", "$"]:
                    pk = pk.replace(ch, "_")
                pk = pk.lower()

                with open(pk, "w") as fp:
                    json.dump([record], fp, default=converter_numpy)
