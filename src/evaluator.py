"""
HireScope — Evaluator
SHAP explainability for XGBoost demand scoring model.
Falls back to feature importance if SHAP has compatibility issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_COLS = [
    "rolling_avg_4weeks",
    "week_over_week_change",
    "momentum_score",
    "is_peak_hiring_season",
    "total_market_volume",
]


def _get_latest_per_skill(features_df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest row per skill from features data."""
    return (
        features_df
        .sort_values("week_date")
        .groupby("skill")
        .last()
        .reset_index()
    )


def _fallback_explanations(model, features_df: pd.DataFrame,
                            top_n_features: int = 3) -> dict[str, list[dict]]:
    """
    Fallback: use XGBoost feature importance instead of SHAP
    when SHAP has compatibility issues.
    """
    latest = _get_latest_per_skill(features_df)
    X = latest[FEATURE_COLS].fillna(0)
    skills = latest["skill"].tolist()

    # get global feature importances
    importances = model.feature_importances_
    feat_imp = list(zip(FEATURE_COLS, importances))
    feat_imp.sort(key=lambda x: abs(x[1]), reverse=True)

    explanations = {}
    for i, skill in enumerate(skills):
        row = X.iloc[i]
        top = []
        for feat, imp in feat_imp[:top_n_features]:
            val = row[feat]
            # direction: positive feature value with high importance = pushes up
            direction = "↑ pushes demand up" if val > 0 else "↓ pushes demand down"
            top.append({
                "feature": feat,
                "contribution": round(float(imp * val), 2),
                "direction": direction,
            })
        explanations[skill] = top

    print(f"✅ Feature importance explanations generated for {len(explanations)} skills (fallback)")
    return explanations


def explain_predictions(model, features_df: pd.DataFrame,
                        top_n_features: int = 3) -> dict[str, list[dict]]:
    """
    Explain demand predictions per skill.
    Tries SHAP first, falls back to feature importance.
    """
    try:
        import shap

        latest = _get_latest_per_skill(features_df)
        X = latest[FEATURE_COLS].fillna(0)
        skills = latest["skill"].tolist()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        explanations = {}
        for i, skill in enumerate(skills):
            sv = shap_values[i]
            contributions = list(zip(FEATURE_COLS, sv))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            top = []
            for feat, val in contributions[:top_n_features]:
                top.append({
                    "feature": feat,
                    "contribution": round(float(val), 2),
                    "direction": "↑ pushes demand up" if val > 0 else "↓ pushes demand down",
                })
            explanations[skill] = top

        print(f"✅ SHAP explanations generated for {len(explanations)} skills")
        return explanations

    except Exception as e:
        print(f"⚠ SHAP failed ({e}), using feature importance fallback")
        return _fallback_explanations(model, features_df, top_n_features)


def get_shap_df(model, features_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Return explanation values DataFrame. Uses SHAP if available,
    otherwise feature importance * feature value as proxy.
    """
    try:
        import shap

        latest = _get_latest_per_skill(features_df)
        X = latest[FEATURE_COLS].fillna(0)
        skills = latest["skill"].tolist()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
        shap_df["skill"] = skills
        return shap_df, FEATURE_COLS

    except Exception as e:
        print(f"⚠ SHAP DataFrame failed ({e}), using feature importance proxy")
        latest = _get_latest_per_skill(features_df)
        X = latest[FEATURE_COLS].fillna(0)
        skills = latest["skill"].tolist()

        importances = model.feature_importances_
        proxy_df = X.copy()
        for j, col in enumerate(FEATURE_COLS):
            proxy_df[col] = proxy_df[col] * importances[j]

        proxy_df["skill"] = skills
        return proxy_df, FEATURE_COLS


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_all
    from src.aggregator import aggregate_weekly
    from src.feature_engineer import compute_features, compute_demand_scores

    df = load_all()
    weekly, top_skills = aggregate_weekly(df)

    features = compute_features(weekly, top_skills)
    scores, model = compute_demand_scores(features)

    print("\n─── SHAP Explanations ───")
    explanations = explain_predictions(model, features)
    for skill in list(explanations.keys())[:5]:
        print(f"\n{skill}:")
        for item in explanations[skill]:
            print(f"  {item['feature']}: {item['contribution']:+.2f} ({item['direction']})")
