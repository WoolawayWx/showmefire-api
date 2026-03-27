import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from config import (
    ALL_CATEGORY_IDS,
    BASELINE_MAX_MACRO_F1_DEGRADATION,
    CATEGORY_LABELS,
    HIGH_IMPACT_CATEGORY_IDS,
    LATEST_MODEL_META_PATH,
    LATEST_MODEL_PATH,
    MIN_HIGH_IMPACT_SUPPORT,
    PRIMARY_GATE_MIN_MACRO_F1,
    REPORTS_DIR,
    TEST_DATA_PATH,
)
from utils import (
    calculate_fire_danger_category,
    category_support,
    ensure_dirs,
    map_score_to_category,
    utc_timestamp,
    write_json,
)


def evaluate(model_path=None, model_meta_path=None, test_data_path=None):
    model_path = LATEST_MODEL_PATH if model_path is None else model_path
    model_meta_path = LATEST_MODEL_META_PATH if model_meta_path is None else model_meta_path
    test_data_path = TEST_DATA_PATH if test_data_path is None else test_data_path

    with open(model_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]
    target_score_col = meta["target_score_col"]
    target_category_col = meta["target_category_col"]
    category_thresholds = meta.get("category_thresholds")

    df_test = pd.read_csv(test_data_path)
    X_test = df_test[feature_cols]
    y_test = df_test[target_score_col]
    y_test_cat = df_test[target_category_col].astype(int)

    booster = xgb.Booster()
    booster.load_model(model_path)

    dtest = xgb.DMatrix(X_test)
    pred_score = booster.predict(dtest)
    pred_cat = map_score_to_category(pred_score, thresholds=category_thresholds)

    regression_metrics = {
        "mae": float(mean_absolute_error(y_test, pred_score)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_score))),
        "r2": float(r2_score(y_test, pred_score)),
    }

    macro_f1 = float(
        f1_score(
            y_test_cat,
            pred_cat,
            labels=ALL_CATEGORY_IDS,
            average="macro",
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            y_test_cat,
            pred_cat,
            labels=ALL_CATEGORY_IDS,
            average="weighted",
            zero_division=0,
        )
    )
    per_class = precision_recall_fscore_support(
        y_test_cat, pred_cat, labels=ALL_CATEGORY_IDS, zero_division=0
    )
    cm = confusion_matrix(y_test_cat, pred_cat, labels=ALL_CATEGORY_IDS)

    baseline_available = all(c in df_test.columns for c in ["target_fm", "rel_humidity", "wind_speed_ms"])
    baseline_pred_cat = None
    if baseline_available:
        wind_kts = df_test["wind_speed_ms"] * 1.94384
        baseline_pred_cat = np.array(
            [
                calculate_fire_danger_category(fm, rh, wk)
                for fm, rh, wk in zip(df_test["target_fm"], df_test["rel_humidity"], wind_kts)
            ],
            dtype=int,
        )
    else:
        # Fallback for datasets without rule inputs.
        baseline_pred_cat = y_test_cat.to_numpy(copy=True)

    baseline_macro_f1 = float(
        f1_score(
            y_test_cat,
            baseline_pred_cat,
            labels=ALL_CATEGORY_IDS,
            average="macro",
            zero_division=0,
        )
    )
    baseline_weighted_f1 = float(
        f1_score(
            y_test_cat,
            baseline_pred_cat,
            labels=ALL_CATEGORY_IDS,
            average="weighted",
            zero_division=0,
        )
    )
    baseline_cm = confusion_matrix(y_test_cat, baseline_pred_cat, labels=ALL_CATEGORY_IDS)

    support_by_class = category_support(y_test_cat)
    support_gate_failures = {
        str(class_id): support_by_class[class_id]
        for class_id in HIGH_IMPACT_CATEGORY_IDS
        if support_by_class[class_id] < MIN_HIGH_IMPACT_SUPPORT
    }
    support_gate_pass = len(support_gate_failures) == 0
    primary_gate_pass = macro_f1 >= PRIMARY_GATE_MIN_MACRO_F1 and support_gate_pass
    baseline_gate_pass = (macro_f1 + BASELINE_MAX_MACRO_F1_DEGRADATION) >= baseline_macro_f1
    overall_gate_pass = primary_gate_pass and baseline_gate_pass

    category_metrics = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": classification_report(
            y_test_cat,
            pred_cat,
            labels=ALL_CATEGORY_IDS,
            target_names=CATEGORY_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "per_class": {
            label: {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i, label in enumerate(CATEGORY_LABELS)
        },
        "confusion_matrix": cm.tolist(),
        "support_by_class": {str(k): int(v) for k, v in support_by_class.items()},
    }

    baseline_metrics = {
        "available": baseline_available,
        "macro_f1": baseline_macro_f1,
        "weighted_f1": baseline_weighted_f1,
        "confusion_matrix": baseline_cm.tolist(),
        "comparison": {
            "model_minus_baseline_macro_f1": float(macro_f1 - baseline_macro_f1),
            "model_minus_baseline_weighted_f1": float(weighted_f1 - baseline_weighted_f1),
        },
    }

    gates = {
        "primary_macro_f1": {
            "min_required": PRIMARY_GATE_MIN_MACRO_F1,
            "value": macro_f1,
            "support_gate_required": True,
            "passed": primary_gate_pass,
        },
        "class_support": {
            "min_support": MIN_HIGH_IMPACT_SUPPORT,
            "categories_checked": HIGH_IMPACT_CATEGORY_IDS,
            "support_by_class": {str(k): int(v) for k, v in support_by_class.items()},
            "failing_classes": support_gate_failures,
            "passed": support_gate_pass,
        },
        "baseline_comparison": {
            "max_allowed_macro_f1_degradation": BASELINE_MAX_MACRO_F1_DEGRADATION,
            "model_macro_f1": macro_f1,
            "baseline_macro_f1": baseline_macro_f1,
            "passed": baseline_gate_pass,
        },
        "overall": {"passed": overall_gate_pass},
    }

    ts = utc_timestamp()
    ensure_dirs(REPORTS_DIR)
    report_json_path = REPORTS_DIR / f"evaluation_{ts}.json"
    report_md_path = REPORTS_DIR / f"evaluation_{ts}.md"
    cm_path = REPORTS_DIR / f"confusion_matrix_{ts}.png"

    report = {
        "created_utc": ts,
        "model_path": str(model_path),
        "test_data_path": str(test_data_path),
        "regression": regression_metrics,
        "category": category_metrics,
        "baseline": baseline_metrics,
        "gates": gates,
    }
    write_json(report_json_path, report)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Fire Danger Model Evaluation\n\n")
        f.write(f"- Model: {model_path}\n")
        f.write(f"- Test Data: {test_data_path}\n")
        f.write(f"- Macro F1 (primary gate): {macro_f1:.4f}\n\n")
        f.write(f"- Weighted F1: {weighted_f1:.4f}\n")
        f.write(f"- Baseline Macro F1: {baseline_macro_f1:.4f}\n")
        f.write(f"- Overall Gate Pass: {overall_gate_pass}\n\n")
        f.write("## Regression Metrics\n")
        for k, v in regression_metrics.items():
            f.write(f"- {k}: {v:.6f}\n")
        f.write("\n## Class Support (test)\n")
        for class_id, class_name in enumerate(CATEGORY_LABELS):
            f.write(f"- {class_name}: {support_by_class[class_id]}\n")
        f.write("\n## Gates\n")
        f.write(f"- Primary Macro F1 Gate: {primary_gate_pass} (>= {PRIMARY_GATE_MIN_MACRO_F1:.3f})\n")
        f.write(
            f"- High-Impact Support Gate: {support_gate_pass} (min support per class: {MIN_HIGH_IMPACT_SUPPORT})\n"
        )
        f.write(
            f"- Baseline Gate: {baseline_gate_pass} (max degradation: {BASELINE_MAX_MACRO_F1_DEGRADATION:.3f})\n"
        )
        f.write("\n## Confusion Matrix (rows=true, cols=pred)\n")
        f.write(f"\n{cm.tolist()}\n")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Fire Danger Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(CATEGORY_LABELS))
    plt.xticks(ticks, CATEGORY_LABELS, rotation=45, ha="right")
    plt.yticks(ticks, CATEGORY_LABELS)
    plt.ylabel("True")
    plt.xlabel("Predicted")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()

    print("Standalone fire danger model evaluation complete")
    print(f"  report: {report_json_path}")
    print(f"  macro_f1: {macro_f1:.4f}")
    print(f"  baseline macro_f1: {baseline_macro_f1:.4f}")
    print(f"  overall gate pass: {overall_gate_pass}")
    print(f"  confusion matrix plot: {cm_path}")



def main():
    parser = argparse.ArgumentParser(description="Evaluate standalone fire danger model.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-meta-path", default=None)
    parser.add_argument("--test-data-path", default=None)
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        model_meta_path=args.model_meta_path,
        test_data_path=args.test_data_path,
    )


if __name__ == "__main__":
    main()
