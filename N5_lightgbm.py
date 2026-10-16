# =============================================================================
# Script: LightGBM models for MCI-to-AD conversion prediction
#
# Description:
#   Predict MCI-to-AD conversion using two separate feature sets:
#   (1) GMV features + age + sex + APOE.
#   (2) TA coupling features + age + sex + APOE.
#
#   Each feature set is analyzed independently using LightGBM, Optuna
#   hyperparameter optimization, repeated nested cross-validation, and
#   SHAP-based model interpretation.
#
# Analysis Workflow:
#   1. Load features, conversion labels, and covariates for Year 1-Year 4.
#   2. Fit separate prediction models for GMV and TA coupling features.
#   3. Run repeated stratified five-fold outer cross-validation
#      (five repeats; 25 outer folds per year).
#   4. Tune LightGBM hyperparameters within each outer training set using
#      five-fold inner cross-validation and 100 Optuna trials per outer fold.
#   5. Evaluate each selected model on its held-out outer test fold.
#   6. Calculate feature importance and SHAP values for outer test samples.
#   7. Summarize predictive performance and feature importance across folds.
#
# Key Settings:
#   - Outer cross-validation: 5 folds x 5 repeats (25 outer folds per year).
#   - Inner cross-validation: 5 folds for hyperparameter selection.
#   - Hyperparameter optimization: 100 Optuna trials per outer fold.
#   - Optimization target: mean inner-fold ROC-AUC.
#   - Part 1 SHAP: interventional TreeSHAP on the probability scale, using
#     the entire outer training set as background data.
#   - Part 2 SHAP: TreeSHAP using the default model-output scale.
#     The actual output scale depends on the installed SHAP/LightGBM versions
#     and should be verified before interpreting the values.
#   - SHAP values are calculated for all samples in each outer test fold.
#   - Each subject contributes one held-out test record per repeat.
#   - Part 2 feature selection is performed on the full outer training set
#     before inner CV. Therefore, inner validation labels influence feature
#     selection, potentially making inner-CV performance optimistic.
#   - Random seeds are fixed where specified to improve reproducibility.
#

# =============================================================================


# =============================================================================
# Part 1: GMV features + age + sex + APOE

# =============================================================================

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import warnings
import matplotlib
matplotlib.use("Agg")  # Non-interactive plotting backend
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    matthews_corrcoef, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import lightgbm as lgb
import optuna
from joblib import Parallel, delayed

# Fixed LightGBM parameters shared by the final models.
MODEL_FIXED_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "is_unbalance": True,
    "random_state": 42,
    "n_jobs": 1,
    "importance_type": "gain",
}

# ==========================================
# Global Configuration
# ==========================================
years_to_process = ["Year1", "Year2", "Year3", "Year4"]

label_path = (
    "/Users/linqing/Desktop/Projects/2025_RMN/data/"
    "schaefer400_7network_name.csv"
)


# ==========================================
# 1. Helper Functions
# ==========================================
def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate discrimination and threshold-dependent classification metrics.

    Sensitivity, specificity, balanced accuracy, and MCC are calculated from
    the supplied class predictions. In this script, model.predict() uses
    LightGBM's default binary classification threshold.
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def positive_class_shap_values(explainer, X):
    """Return class-1 SHAP values as a samples-by-features array.

    Handles common SHAP output formats for binary classification.
    The expected output shape is (n_samples, n_features).
    """
    values = explainer.shap_values(X)

    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError(
                f"Expected two SHAP class arrays; received {len(values)}."
            )
        values = values[1]

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape == (X.shape[0], X.shape[1], 2):
            values = values[:, :, 1]
        elif values.shape == (2, X.shape[0], X.shape[1]):
            values = values[1]
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

    if values.shape != X.shape or not np.isfinite(values).all():
        raise ValueError(
            f"Invalid positive-class SHAP output shape or values: "
            f"{values.shape}"
        )

    return values


def create_objective(X_train_clean, y_train_outer):
    """Create the Optuna objective using five-fold inner cross-validation.

    The supplied data are the complete outer training set. Inner-fold
    performance is used to select hyperparameters by mean ROC-AUC.
    No outer test samples are used in this objective.
    """
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "is_unbalance": True,
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "num_leaves": trial.suggest_int("num_leaves", 4, 30),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", 500, 1000
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-3, 10.0, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-3, 10.0, log=True
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.1, 0.9
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.1, 0.9
            ),
            "bagging_freq": trial.suggest_int(
                "bagging_freq", 1, 5
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 40
            ),
            "max_bin": trial.suggest_int("max_bin", 31, 127),
            "random_state": 42,
            "importance_type": "gain",
        }

        params.update(MODEL_FIXED_PARAMS)

        skf_inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        )

        aucs = []

        for tr_idx, val_idx in skf_inner.split(
            X_train_clean, y_train_outer
        ):
            X_tr, X_val = (
                X_train_clean[tr_idx],
                X_train_clean[val_idx],
            )
            y_tr, y_val = (
                y_train_outer[tr_idx],
                y_train_outer[val_idx],
            )

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)

            prob = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, prob))

        return np.mean(aucs)

    return objective


# ==========================================
# 2. Outer Fold Function
# ==========================================
def run_fold_all_features(
    fold_num, train_idx, test_idx, X_data, y_data
):
    """Fit and evaluate one outer fold using all available features.

    Hyperparameters are selected by inner five-fold CV. The final model is
    fitted on the complete outer training set and evaluated on the held-out
    outer test fold.

    SHAP is calculated for every outer test sample on the probability scale,
    using the entire outer training set as interventional background data.
    """
    X_tr, y_tr = X_data[train_idx], y_data[train_idx]
    X_te, y_te = X_data[test_idx], y_data[test_idx]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42 + fold_num
        ),
    )

    study.optimize(
        create_objective(X_tr, y_tr),
        n_trials=100,
        n_jobs=1,
    )

    best_params = study.best_params.copy()
    best_params.update(MODEL_FIXED_PARAMS)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr)

    # Gain-based feature importance from the fitted outer-fold model.
    importances = model.feature_importances_

    # Predictions and performance on the held-out outer test fold.
    prob = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    metrics = calculate_metrics(y_te, pred, prob)

    # Verify that the positive class is encoded as 1.
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError(
            "Expected numeric classes 0 and 1, with class 1 positive."
        )

    # Use the entire outer training set as the interventional SHAP background.
    # No random sampling of background subjects is performed.
    # Probability-scale SHAP requires a compatible interventional background.
    explainer = shap.TreeExplainer(
        model,
        data=X_tr,
        feature_perturbation="interventional",
        model_output="probability",
    )

    # Calculate positive-class SHAP values for all outer test samples.
    shap_vals = positive_class_shap_values(explainer, X_te)

    # Mean absolute SHAP for this outer fold, on the probability scale.
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    return metrics, importances, mean_abs_shap, shap_vals, X_te


# ==========================================
# 3. Main Loop
# ==========================================
for current_year in years_to_process:

    print(
        f"\n{'=' * 40}\n"
        f" PROCESSING: {current_year}\n"
        f"{'=' * 40}"
    )

    year_output_dir = (
        "/Users/linqing/Desktop/Projects/2024 structure_FC/"
        "RMN network/Representation mapping network/6.Disease/"
        "predict_model/LightGBM_TA/results/"
        f"{current_year}/GMV+agesex+apoe"
    )
    os.makedirs(year_output_dir, exist_ok=True)

    year_mat_path = (
        "/Users/linqing/Desktop/Projects/2024 structure_FC/"
        "RMN network/Representation mapping network/6.Disease/"
        "predict_model/LightGBM_TA/data/"
        f"{current_year}/gray_dat_harmonized.mat"
    )

    year_cov_path = (
        "/Users/linqing/Desktop/Projects/2024 structure_FC/"
        "RMN network/Representation mapping network/6.Disease/"
        "predict_model/LightGBM_TA/data/"
        f"{current_year}/{current_year}_cov.csv"
    )

    # ------------------------------------------
    # Data loading
    # ------------------------------------------
    try:
        mat_data = sio.loadmat(year_mat_path)

        X_year_gmv = mat_data["gray_dat_harmonized"].T
        cov_df_year = pd.read_csv(year_cov_path)

        y_year = cov_df_year["state"].values

        # Covariates: age, sex and APOE.
        covariates_year = cov_df_year[
            ["age", "sex", "apoe"]
        ].values

        # Combine all GMV features with the three covariates.
        X_year = np.hstack(
            (X_year_gmv, covariates_year)
        )
        n_features = X_year.shape[1]

    except Exception as e:
        print(f"Error loading {current_year}: {e}")
        continue

    # ------------------------------------------
    # Data validation
    # ------------------------------------------
    if (
        X_year_gmv.ndim != 2
        or X_year.ndim != 2
        or X_year_gmv.shape[1] == 0
    ):
        raise ValueError(
            "GMV must be a nonempty subjects-by-features matrix."
        )

    if X_year.shape[0] != len(y_year):
        raise ValueError(
            "MAT subject count and covariate CSV row count differ."
        )

    if (
        not np.issubdtype(X_year.dtype, np.number)
        or not np.isfinite(X_year).all()
    ):
        raise ValueError(
            "GMV, age, sex and APOE must be numeric without NaN/Inf."
        )

    if not np.array_equal(np.unique(y_year), [0, 1]):
        raise ValueError(
            "state must contain exactly numeric labels 0 and 1."
        )

    y_year = y_year.astype(int)

    class_counts = np.bincount(y_year, minlength=2)

    # Check that the outer training sets can support five-fold inner CV.
    outer_train_counts = (
        class_counts - np.ceil(class_counts / 5).astype(int)
    )

    if outer_train_counts.min() < 5:
        raise ValueError(
            "Insufficient subjects per class for five-fold inner CV."
        )

    # ------------------------------------------
    # Feature names
    # ------------------------------------------
    try:
        label_df = pd.read_csv(label_path)
        feature_names = label_df["label"].tolist()

    except (OSError, pd.errors.ParserError):
        warnings.warn(
            "Region label file unavailable; using numbered ROI names."
        )
        feature_names = [
            f"ROI_{i}" for i in range(X_year_gmv.shape[1])
        ]

    if (
        len(feature_names) != X_year_gmv.shape[1]
        or pd.isna(feature_names).any()
    ):
        raise ValueError(
            "ROI labels must contain one nonmissing name per GMV feature."
        )

    feature_names = [str(name) for name in feature_names]
    feature_names.extend(["age", "sex", "apoe"])

    if len(set(feature_names)) != n_features:
        raise ValueError(
            "GMV and age/sex/APOE feature names must be unique."
        )

    # ------------------------------------------
    # Nested cross-validation
    # ------------------------------------------
    # Five-fold outer CV repeated five times, yielding 25 outer test folds.
    # Each subject appears in an outer test fold once per repeat.
    rskf_outer = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42,
    )

    folds = list(rskf_outer.split(X_year, y_year))

    print(f"[{current_year}] Running {len(folds)} Folds...")

    results_fs = Parallel(n_jobs=10)(
        delayed(run_fold_all_features)(
            i, tr, te, X_year, y_year
        )
        for i, (tr, te) in enumerate(folds)
    )

    # ------------------------------------------
    # Unpack results
    # ------------------------------------------
    fs_metrics = [res[0] for res in results_fs]
    all_fold_importances = [res[1] for res in results_fs]
    all_fold_mean_shap = [res[2] for res in results_fs]
    all_shap_values = [res[3] for res in results_fs]
    all_X_te = [res[4] for res in results_fs]

    # ------------------------------------------
    # Save performance metrics
    # ------------------------------------------
    fs_perf_df = pd.DataFrame(fs_metrics)

    mean_fs = fs_perf_df.mean().to_frame().T
    mean_fs.insert(0, "Fold", "Mean")

    fs_perf_df["Fold"] = [
        f"Fold_{i + 1}" for i in range(len(fs_perf_df))
    ]

    pd.concat(
        [fs_perf_df, mean_fs],
        ignore_index=True,
    ).to_csv(
        os.path.join(
            year_output_dir,
            f"GMV_{current_year}_performance_agesex_apoe.csv",
        ),
        index=False,
    )

    # ------------------------------------------
    # Feature selection summary
    # ------------------------------------------
    # No feature filter is applied: all GMV features and covariates
    # are retained in every outer-fold model.
    feature_selection_df = pd.DataFrame({
        "Feature_Name": feature_names,
        "Selection_Method": "No feature filter; all model inputs retained",
        "Selection_Frequency": 1.0,
    })
    for fold_num in range(len(folds)):
        feature_selection_df[f"Fold_{fold_num + 1}_Selected"] = 1
    feature_selection_df.to_csv(
        os.path.join(
            year_output_dir,
            f"GMV_{current_year}_feature_selection_agesex_apoe.csv",
        ),
        index=False,
    )

    # ------------------------------------------
    # Save Gain CSV
    # ------------------------------------------
    # Save raw gain and fold-wise min-max normalized gain.
    # Average_Gain_Norm is the mean of normalized values across folds,
    # not the mean of raw gain values.
    gain_df = pd.DataFrame(
        {"Feature_Name": feature_names}
    )

    gain_norm_cols = []

    for f_idx, g in enumerate(all_fold_importances):

        gain_df[f"Fold_{f_idx + 1}_Gain_Raw"] = g

        mn, mx = g.min(), g.max()

        normed = (
            (g - mn) / (mx - mn)
            if mx > mn
            else np.zeros_like(g)
        )

        col_name = f"Fold_{f_idx + 1}_Gain_Norm"
        gain_df[col_name] = normed
        gain_norm_cols.append(col_name)

    gain_df["Average_Gain_Norm"] = gain_df[
        gain_norm_cols
    ].mean(axis=1)

    gain_df.to_csv(
        os.path.join(
            year_output_dir,
            f"GMV_{current_year}_gain_summary_agesex_apoe.csv",
        ),
        index=False,
    )

    # ------------------------------------------
    # Save SHAP CSV
    # ------------------------------------------
    # Per-fold mean absolute SHAP values are calculated on the probability
    # scale. The across-fold mean summarizes feature importance over models
    # fitted to different outer training sets.
    shap_summary_df = pd.DataFrame(
        {"Feature_Name": feature_names}
    )

    for f_idx, sv in enumerate(all_fold_mean_shap):
        shap_summary_df[
            f"Fold_{f_idx + 1}_SHAP_AbsMean"
        ] = sv

    shap_summary_df["Average_SHAP_AbsMean"] = (
        shap_summary_df.iloc[:, 1:].mean(axis=1)
    )

    shap_summary_df.to_csv(
        os.path.join(
            year_output_dir,
            f"GMV_{current_year}_shap_summary_agesex_apoe.csv",
        ),
        index=False,
    )

    # ==========================================
    # 4. Visualization (Beeswarm & Barplot)
    # ==========================================

    # Concatenate held-out test records from all outer folds.
    # Each subject contributes five records across the five CV repeats.
    # These repeated records should not be treated as independent subjects.
    concat_shap = np.vstack(all_shap_values)
    concat_X = np.vstack(all_X_te)
    pd.DataFrame(concat_shap, columns=feature_names).to_csv(
        os.path.join(
            year_output_dir,
            f"GMV_{current_year}_Raw_SHAP_Values_agesex_apoe.csv",
        ),
        index=False,
    )

    # Select the 20 features with the highest average mean absolute SHAP.
    top20_idx = (
        shap_summary_df["Average_SHAP_AbsMean"]
        .nlargest(20)
        .index.values
    )

    # Order selected features from highest to lowest mean absolute SHAP.
    top20_sorted_idx = top20_idx[
        np.argsort(
            shap_summary_df.loc[
                top20_idx, "Average_SHAP_AbsMean"
            ].values
        )[::-1]
    ]

    shap_top20 = concat_shap[:, top20_sorted_idx]
    X_te_top20 = concat_X[:, top20_sorted_idx]

    feat_top20 = [
        feature_names[i] for i in top20_sorted_idx
    ]

    # ------------------------------------------
    # 4a. Beeswarm Plot (10 x 8)
    # ------------------------------------------
    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_top20,
        X_te_top20,
        feature_names=feat_top20,
        max_display=20,
        plot_type="dot",
        show=False,
        color_bar=True,
        plot_size=None,
        sort=False,
    )

    ax1 = plt.gca()

    ax1.set_xlabel(
        "SHAP value (probability scale; positive → higher predicted probability)",
        fontsize=10,
    )

    ax1.set_title(
        f"Top 20 features - SHAP beeswarm ({current_year})",
        fontsize=12,
        pad=10,
    )

    ax1.tick_params(labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    ax1.axvline(
        0,
        color="0.7",
        lw=0.8,
        ls="--",
        zorder=0,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            year_output_dir,
            f"GMV_{current_year}_top20_shap_beeswarm_agesex_apoe.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------
    # 4b. Barplot (9 x 7)
    # ------------------------------------------
    # Rank features by average mean absolute SHAP across outer folds.
    top20_bar_df = (
        shap_summary_df
        .nlargest(20, "Average_SHAP_AbsMean")
        .sort_values("Average_SHAP_AbsMean", ascending=True)
    )

    fig, ax2 = plt.subplots(figsize=(9, 7))

    bars = ax2.barh(
        top20_bar_df["Feature_Name"],
        top20_bar_df["Average_SHAP_AbsMean"],
        color="orange",
        height=0.65,
    )

    ax2.set_xlabel(
        "Mean |SHAP value| (averaged over folds)",
        fontsize=11,
    )

    ax2.set_title(
        f"Top 20 features by mean absolute SHAP ({current_year})",
        fontsize=12,
        pad=12,
    )

    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=8)

    # Add numerical labels to the bars.
    max_val = top20_bar_df[
        "Average_SHAP_AbsMean"
    ].max()

    for bar in bars:
        val = bar.get_width()

        ax2.text(
            val + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=7.5,
            color="0.35",
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            year_output_dir,
            f"GMV_{current_year}_top20_shap_barplot_agesex_apoe.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[{current_year}] Done.")

print("\nPart 1 finished.")


# =============================================================================
# Part 2: TA coupling features + age + sex + APOE
#
# Feature selection:
#   Welch's two-sample t-test is applied to brain features using the complete
#   outer training set. Features with p < 0.01 are retained. If no features
#   pass the threshold, the 100 features with the smallest p-values are used.
#   Age, sex and APOE are always retained.

# =============================================================================

# LightGBM + Optuna + TreeSHAP (all outer test samples; no subsampling)

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import optuna

from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from joblib import Parallel, delayed

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef
)

warnings.filterwarnings("ignore")

# ==========================================
# Feature names
# ==========================================
region_name_path = "/Users/linqing/Desktop/Projects/2025_RMN/data/schaefer400_7network_name.csv"
term_name_path = "/Users/linqing/Desktop/Projects/2025_RMN/data/Term.csv"

region_names = pd.read_csv(region_name_path)["label"].values
term_names = pd.read_csv(term_name_path)["Term"].values

n_regions = 400
n_terms = 539
n_brain_features = n_regions * n_terms

if len(region_names) != n_regions or len(term_names) != n_terms:
    raise ValueError("Region/term label counts do not match expected dimensions.")

if pd.isna(region_names).any() or pd.isna(term_names).any():
    raise ValueError("Region/term labels contain missing values.")

# Feature order follows the flattened (region, term) dimensions:
# each region contributes 539 term-related coupling features.
region_idx_all = [r for r in range(n_regions) for _ in range(n_terms)]
term_idx_all = [t for _ in range(n_regions) for t in range(n_terms)]

feature_names_all = [
    f"{region_names[r]}__{term_names[t]}"
    for r in range(n_regions)
    for t in range(n_terms)
] + ["age", "sex", "apoe"]

# Covariates do not correspond to a region-term pair.
region_idx_all.extend([-1, -1, -1])
term_idx_all.extend([-1, -1, -1])

n_features = len(feature_names_all)


# ==========================================
# T-test feature selection
# Applied to the outer training set before inner CV.
# ==========================================
def fast_ttest_filter(X_tr, y_tr, p_threshold=0.01):
    """Select brain features using Welch's two-sample t-test.

    This function is called on the complete outer training set in the current
    workflow. Because it is called before the inner CV split, the resulting
    selected features are not independent of inner validation labels.
    """
    X_0 = X_tr[y_tr == 0]
    X_1 = X_tr[y_tr == 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_vals = ttest_ind(
            X_0, X_1, axis=0, equal_var=False
        )

    p_vals = np.nan_to_num(
        p_vals, nan=1.0, posinf=1.0, neginf=1.0
    )

    selected_idx = np.where(p_vals < p_threshold)[0]

    # Fallback: retain up to 100 features with the smallest p-values.
    if len(selected_idx) == 0:
        selected_idx = np.argsort(p_vals)[:min(100, X_tr.shape[1])]

    return selected_idx


# ==========================================
# Metrics
# ==========================================
def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate discrimination and threshold-dependent classification metrics.

    The class predictions are supplied by model.predict(), which uses
    LightGBM's default binary classification threshold.
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    bacc = balanced_accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Balanced Accuracy": bacc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "MCC": mcc
    }


# ==========================================
# Optuna objective
# Inner CV uses the selected features from the outer training set.
# Feature selection is not repeated inside each inner training fold.
# ==========================================
def create_objective(X_train_outer, y_train_outer):

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "is_unbalance": True,
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "num_leaves": trial.suggest_int("num_leaves", 4, 30),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", 500, 1000
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-3, 10.0, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-3, 10.0, log=True
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.1, 0.9
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.1, 0.9
            ),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 40
            ),
            "max_bin": trial.suggest_int("max_bin", 31, 127),
            "random_state": 42,
            "n_jobs": 1,
            "importance_type": "gain"
        }

        skf_inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        )
        aucs = []

        for tr_idx, val_idx in skf_inner.split(
            X_train_outer, y_train_outer
        ):
            X_tr = X_train_outer[tr_idx]
            X_val = X_train_outer[val_idx]
            y_tr = y_train_outer[tr_idx]
            y_val = y_train_outer[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)

            prob = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, prob))

        return float(np.mean(aucs))

    return objective


# ==========================================
# Year loop
# ==========================================
for year in [1, 2, 3, 4]:

    print(f"\n{'=' * 50}")
    print(f"Processing Year {year} (coupling + age + sex + APOE)")
    print(f"{'=' * 50}\n")

    output_dir = (
        "/Users/linqing/Desktop/Projects/2024 structure_FC/"
        "RMN network/Representation mapping network/6.Disease/"
        "predict_model/LightGBM_TA/results/"
        f"Year{year}/coupling_agesex+apoe"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------
    # Load data
    # ------------------------------------------
    mat_path = (
        "/Users/linqing/Desktop/Projects/2024 structure_FC/"
        "RMN network/Representation mapping network/6.Disease/"
        "predict_model/LightGBM_TA/data/"
        f"Year{year}/dat_harmonized.mat"
    )

    cov_path = (
        "/Users/linqing/Desktop/Projects/2024 structure_FC/"
        "RMN network/Representation mapping network/6.Disease/"
        "predict_model/LightGBM_TA/data/"
        f"Year{year}/Year{year}_cov.csv"
    )

    mat_data = sio.loadmat(mat_path)
    raw_data = mat_data["dat_harmonized"]

    if raw_data.ndim != 3 or raw_data.shape[:2] != (
        n_regions, n_terms
    ):
        raise ValueError(
            f"Year{year}: coupling dimensions do not match expected dimensions."
        )

    if (
        not np.issubdtype(raw_data.dtype, np.number)
        or not np.isfinite(raw_data).all()
    ):
        raise ValueError(
            f"Year{year}: coupling data must be finite numeric values."
        )

    # Original data shape: (n_regions, n_terms, n_subjects).
    # Reorder to (n_subjects, n_regions, n_terms), then flatten the brain
    # dimensions so each subject has n_regions * n_terms features.
    transposed_data = np.transpose(raw_data, (2, 0, 1))
    X_brain = transposed_data.reshape(
        transposed_data.shape[0], -1
    )

    cov_df = pd.read_csv(cov_path)

    required_columns = ["state", "age", "sex", "apoe"]
    missing_columns = [
        col for col in required_columns if col not in cov_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Year{year}: missing covariate columns: {missing_columns}"
        )

    y = cov_df["state"].to_numpy()
    covariates = cov_df[["age", "sex", "apoe"]].to_numpy()

    if raw_data.shape[2] != len(y):
        raise ValueError(
            f"Year{year}: coupling subject count and covariate rows differ."
        )

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"Year{year}: state must be numeric 0/1 labels.")

    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError(
            f"Year{year}: state must contain exactly labels 0 and 1."
        )

    if not np.isfinite(covariates).all():
        raise ValueError(
            f"Year{year}: age, sex and APOE must be finite numeric values."
        )

    y = y.astype(int)

    class_counts = np.bincount(y, minlength=2)

    if class_counts.min() < 5:
        raise ValueError(
            f"Year{year}: insufficient subjects for five-fold CV."
        )

    if np.min(class_counts - np.ceil(class_counts / 5).astype(int)) < 5:
        raise ValueError(
            f"Year{year}: insufficient subjects per class for inner five-fold CV."
        )

    # Append covariates. Age, sex and APOE are always retained as predictors.
    X = np.hstack((X_brain, covariates))

    if X.shape[1] != n_features:
        raise ValueError(
            f"Year{year}: unexpected feature count {X.shape[1]}."
        )

    # ------------------------------------------
    # Outer CV: 5 folds x 5 repeats
    # ------------------------------------------
    # Each subject is held out once per repeat, resulting in 25 outer folds.
    rskf_outer = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=42
    )
    folds = list(rskf_outer.split(X, y))

    def run_fold_feature_selection(fold_num, train_idx, test_idx):

        print(f"Year {year}, Fold {fold_num + 1}/{len(folds)}")

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Select brain features using the complete outer training set only.
        # Important: this occurs before inner CV, so inner validation labels
        # influence feature selection. Inner-CV scores may be optimistic.
        selected_brain_idx = fast_ttest_filter(
            X_tr[:, :n_brain_features], y_tr
        )

        # Always include age, sex and APOE, regardless of T-test results.
        covariate_idx = np.arange(n_brain_features, n_features)
        selected_idx = np.concatenate(
            (selected_brain_idx, covariate_idx)
        )
        selection_mask = np.zeros(n_features, dtype=np.uint8)
        selection_mask[selected_idx] = 1

        X_tr_sel = X_tr[:, selected_idx]
        X_te_sel = X_te[:, selected_idx]

        print("Selected features:", len(selected_idx))
        print("Outer test samples:", len(test_idx))

        # Tune hyperparameters using five-fold inner CV on the selected
        # outer-training features. Feature selection is not repeated within
        # each inner training fold in the current implementation.
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42 + fold_num
            )
        )

        study.optimize(
            create_objective(X_tr_sel, y_tr),
            n_trials=100,
            n_jobs=1
        )

        best_params = study.best_params.copy()
        best_params.update({
            "objective": "binary",
            "is_unbalance": True,
            "verbosity": -1,
            "n_jobs": 1,
            "importance_type": "gain",
            "random_state": 42
        })

        # Fit the final model on the selected features from the complete
        # outer training set.
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_tr_sel, y_tr)

        # Evaluate predictions on the held-out outer test fold.
        pred = model.predict(X_te_sel)
        prob = model.predict_proba(X_te_sel)[:, 1]
        metrics = calculate_metrics(y_te, pred, prob)

        # ------------------------------------------
        # Gain importance mapped to global features
        # ------------------------------------------
        # Features excluded by T-test selection receive zero in the
        # full-feature-space representation.
        importances = model.feature_importances_

        global_gain = np.zeros(n_features, dtype=float)
        global_gain[selected_idx] = importances

        # Split counts mapped to the same global feature space.
        split_counts = model.booster_.feature_importance(
            importance_type="split"
        )

        global_split = np.zeros(n_features, dtype=float)
        global_split[selected_idx] = split_counts

        # ------------------------------------------
        # TreeSHAP: all outer test samples
        # No random sampling or subsampling
        # ------------------------------------------
        if not np.array_equal(model.classes_, np.array([0, 1])):
            raise ValueError("Expected classes 0 and 1.")

        # Use the default TreeExplainer configuration.
        # The output scale should be verified for the installed SHAP and
        # LightGBM versions; do not automatically interpret these values
        # as probability-scale SHAP values.
        explainer = shap.TreeExplainer(model)

        shap_output = explainer.shap_values(
            X_te_sel,
            check_additivity=False
        )

        # Support common SHAP output formats:
        # 1. A list containing class-0 and class-1 SHAP arrays.
        # 2. An array with shape (n_samples, n_features).
        # 3. An array with shape (n_samples, n_features, n_classes).
        if isinstance(shap_output, list):
            if len(shap_output) != 2:
                raise ValueError(
                    f"Unexpected SHAP list length: {len(shap_output)}"
                )
            shap_vals = np.asarray(shap_output[1])
        else:
            shap_vals = np.asarray(shap_output)

            if (
                shap_vals.ndim == 3
                and shap_vals.shape[0] == X_te_sel.shape[0]
                and shap_vals.shape[1] == X_te_sel.shape[1]
            ):
                shap_vals = shap_vals[:, :, 1]

        if shap_vals.shape != X_te_sel.shape:
            raise ValueError(
                f"Unexpected TreeSHAP shape: {shap_vals.shape}; "
                f"expected {X_te_sel.shape}"
            )

        # Per-fold mean absolute SHAP for selected features.
        # Values are mapped to the full feature space; excluded features
        # receive zero because they were not included in this fold's model.
        global_shap_abs = np.zeros(n_features, dtype=float)
        global_shap_abs[selected_idx] = np.abs(
            shap_vals
        ).mean(axis=0)

        # Signed SHAP values for every outer test sample.
        # Features excluded from the fold's model receive zero for alignment
        # with the full feature space; this is not an estimated contribution
        # from a model in which those features were included.
        shap_global = np.zeros(
            (X_te_sel.shape[0], n_features),
            dtype=np.float32
        )
        shap_global[:, selected_idx] = shap_vals.astype(np.float32)

        # Retain the original values of all features for beeswarm colouring.
        X_te_global = X_te.astype(np.float32)

        return (
            metrics,
            global_split,
            global_gain,
            global_shap_abs,
            shap_global,
            X_te_global,
            selection_mask
        )

    # ==========================================
    # Run outer folds
    # ==========================================
    results = Parallel(n_jobs=10)(
        delayed(run_fold_feature_selection)(i, tr, te)
        for i, (tr, te) in enumerate(folds)
    )

    # ==========================================
    # Collect results
    # ==========================================
    fs_performance_records = [res[0] for res in results]
    feature_split_list = [res[1] for res in results]
    gain_raw_list = [res[2] for res in results]
    shap_abs_list = [res[3] for res in results]
    shap_raw_list = [res[4] for res in results]
    X_te_global_list = [res[5] for res in results]
    selection_mask_list = [res[6] for res in results]

    # ==========================================
    # Performance summary
    # ==========================================
    # Report fold-wise metrics and their arithmetic mean.
    # The mean row is a descriptive summary across the 25 outer folds.
    fs_perf_df = pd.DataFrame(fs_performance_records)

    fs_perf_df.insert(
        0,
        "Fold",
        [f"Fold_{i + 1}" for i in range(len(fs_perf_df))]
    )

    mean_row = fs_perf_df.drop(columns=["Fold"]).mean()
    mean_row["Fold"] = "Mean"

    fs_perf_combined_df = pd.concat(
        [fs_perf_df, pd.DataFrame([mean_row])],
        ignore_index=True
    )

    fs_perf_combined_df.to_csv(
        os.path.join(
            output_dir,
            f"Year{year}_feature_selection_performance_agesex_apoe.csv"
        ),
        index=False
    )

    # ==========================================
    # T-test Feature Selection Frequency
    # ==========================================
    # Selection frequency is calculated across the 25 outer training sets.
    # Covariates are always selected and therefore have frequency 1.
    feature_selection_df = pd.DataFrame({
        "Global_Feature_Index": np.arange(n_features),
        "Region_Index": region_idx_all,
        "Term_Index": term_idx_all,
        "Feature_Name": feature_names_all
    })
    for fold_num, selected_mask in enumerate(selection_mask_list):
        feature_selection_df[
            f"Fold_{fold_num + 1}_TTest_Selected"
        ] = selected_mask
    selection_cols = [
        f"Fold_{i + 1}_TTest_Selected"
        for i in range(len(selection_mask_list))
    ]
    feature_selection_df["TTest_Selection_Frequency"] = (
        feature_selection_df[selection_cols].mean(axis=1)
    )
    feature_selection_df.to_csv(
        os.path.join(
            output_dir,
            f"Year{year}_feature_selection_agesex_apoe.csv"
        ),
        index=False
    )

    # ==========================================
    # Gain values and normalized Gain CSV
    # ==========================================
    # Save raw gain and fold-wise min-max normalized gain.
    # Average_Gain_Norm is the mean of normalized gain values across folds,
    # not the mean of raw gain values.
    gain_df = pd.DataFrame({
        "Global_Feature_Index": np.arange(n_features),
        "Region_Index": region_idx_all,
        "Term_Index": term_idx_all,
        "Feature_Name": feature_names_all
    })

    gain_norm_cols = []

    for fold_num, g in enumerate(gain_raw_list):
        gain_df[f"Fold_{fold_num + 1}_Gain_Raw"] = g

        col_name = f"Fold_{fold_num + 1}_Gain_Norm"
        mn, mx = g.min(), g.max()

        normed = (
            (g - mn) / (mx - mn)
            if mx > mn
            else np.zeros_like(g)
        )

        gain_df[col_name] = normed
        gain_norm_cols.append(col_name)

    gain_df["Average_Gain_Norm"] = gain_df[
        gain_norm_cols
    ].mean(axis=1)

    gain_df.to_csv(
        os.path.join(
            output_dir,
            f"Year{year}_feature_gain_agesex_apoe.csv"
        ),
        index=False
    )

    # ==========================================
    # Mean absolute SHAP CSV
    # ==========================================
    # Average_SHAP_AbsMean averages fold-wise mean absolute SHAP values.
    # Because excluded features receive zero, this summary reflects both
    # feature-selection frequency and the magnitude of SHAP values in folds
    # where a feature was selected.
    shap_df = pd.DataFrame({
        "Global_Feature_Index": np.arange(n_features),
        "Region_Index": region_idx_all,
        "Term_Index": term_idx_all,
        "Feature_Name": feature_names_all
    })

    for fold_num, sv in enumerate(shap_abs_list):
        shap_df[f"Fold_{fold_num + 1}_SHAP_AbsMean"] = sv

    shap_abs_cols = [
        f"Fold_{i + 1}_SHAP_AbsMean"
        for i in range(len(shap_abs_list))
    ]

    shap_df["Average_SHAP_AbsMean"] = shap_df[
        shap_abs_cols
    ].mean(axis=1)

    shap_df.to_csv(
        os.path.join(
            output_dir,
            f"Year{year}_feature_shap_agesex_apoe.csv"
        ),
        index=False
    )

    # ==========================================
    # SHAP beeswarm
    # All outer test records; no subsampling
    # ==========================================
    # Each subject contributes one held-out test record per repeat.
    # Consequently, the concatenated matrix contains repeated records for
    # each subject across the five CV repeats; these are not independent
    # subject-level observations.
    shap_all = np.vstack(shap_raw_list)
    X_te_all = np.vstack(X_te_global_list)
    pd.DataFrame(shap_all, columns=feature_names_all).to_csv(
        os.path.join(
            output_dir,
            f"Year{year}_Raw_SHAP_Values_agesex_apoe.csv"
        ),
        index=False
    )

    if shap_all.shape != X_te_all.shape:
        raise ValueError(
            f"Year{year}: SHAP matrix and feature matrix differ: "
            f"{shap_all.shape} vs {X_te_all.shape}"
        )

    print(
        f"Year {year}: beeswarm includes {shap_all.shape[0]} "
        "outer-test prediction records; all folds retained."
    )

    # Select the 20 features with the largest average mean absolute SHAP.
    top20_global_idx = (
        shap_df["Average_SHAP_AbsMean"]
        .nlargest(20)
        .index.to_numpy()
    )

    # Order the selected features from highest to lowest average importance.
    importance_order = np.argsort(
        shap_df.loc[
            top20_global_idx, "Average_SHAP_AbsMean"
        ].to_numpy()
    )[::-1]

    top20_sorted_idx = top20_global_idx[importance_order]

    shap_top20 = shap_all[:, top20_sorted_idx]
    X_te_top20 = X_te_all[:, top20_sorted_idx]

    feat_top20 = [
        feature_names_all[i] for i in top20_sorted_idx
    ]

    # ------------------------------------------
    # Beeswarm Plot
    # ------------------------------------------
    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_top20,
        X_te_top20,
        feature_names=feat_top20,
        max_display=20,
        plot_type="dot",
        show=False,
        plot_size=None,
        color_bar=True
    )

    ax = plt.gca()

    # The label describes the direction of the model output. Confirm the
    # actual TreeSHAP output scale before interpreting it as log-odds.
    ax.set_xlabel(
        "SHAP value (positive → higher model output for state = 1)",
        fontsize=10
    )

    ax.set_title(
        f"Top 20 features — TreeSHAP beeswarm "
        f"(Year {year}, age + sex + APOE)",
        fontsize=12,
        pad=10
    )

    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="0.7", lw=0.8, ls="--", zorder=0)

    plt.tight_layout()

    fig = plt.gcf()
    fig.savefig(
        os.path.join(
            output_dir,
            f"Year{year}_top20_shap_beeswarm_agesex_apoe.pdf"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close("all")

    # ==========================================
    # Top-20 SHAP feature bar plot
    # ==========================================
    # Bar lengths represent average mean absolute SHAP values across folds.
    # In this feature-selection workflow, the average also reflects
    # how frequently each feature was selected.
    top20 = shap_df.nlargest(
        20, "Average_SHAP_AbsMean"
    ).sort_values(
        "Average_SHAP_AbsMean", ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.barh(
        top20["Feature_Name"],
        top20["Average_SHAP_AbsMean"],
        color="orange",
        edgecolor="none",
        height=0.65
    )

    ax.set_xlabel(
        "Mean |SHAP value| (averaged over outer folds)",
        fontsize=11
    )

    ax.set_title(
        f"Top 20 features by mean |TreeSHAP| "
        f"(Year {year}, age + sex + APOE)",
        fontsize=12,
        pad=12
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=9)

    max_shap = top20["Average_SHAP_AbsMean"].max()

    for bar, val in zip(
        bars, top20["Average_SHAP_AbsMean"]
    ):
        ax.text(
            val + max_shap * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="0.35"
        )

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            f"Year{year}_top20_shap_barplot_agesex_apoe.pdf"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"\nYear {year} done. Outputs saved to: {output_dir}\n"
    )

print("\nPart 2 finished.")
print("\nAll tasks finished.")