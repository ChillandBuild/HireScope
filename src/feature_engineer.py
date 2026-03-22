"""
HireScope — Feature Engineer
Compute per-skill features for XGBoost demand scoring layer.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_features(weekly_df: pd.DataFrame, top_skills: list[str]) -> pd.DataFrame:
    """
    Compute features per skill from weekly posting counts:
      - rolling_avg_4weeks
      - week_over_week_change (%)
      - momentum_score (acceleration = 2nd derivative)
      - is_peak_hiring_season (Jan-Mar=1, Jul-Sep=1)
      - total_market_volume (all postings that week across all skills)

    Args:
        weekly_df: aggregated weekly data (combined source)
        top_skills: list of skill names

    Returns:
        DataFrame with features per skill per week
    """
    combined = weekly_df[weekly_df["source"] == "combined"].copy()
    combined = combined.sort_values(["skill", "week_date"]).reset_index(drop=True)

    # total market volume per week (across all skills)
    market_vol = (
        combined.groupby("week_date")["posting_count"]
        .sum()
        .reset_index()
        .rename(columns={"posting_count": "total_market_volume"})
    )

    all_features = []

    for skill in top_skills:
        skill_data = combined[combined["skill"] == skill].copy()
        if len(skill_data) < 2:
            continue

        skill_data = skill_data.sort_values("week_date").reset_index(drop=True)

        # rolling 4-week average
        skill_data["rolling_avg_4weeks"] = (
            skill_data["posting_count"]
            .rolling(window=4, min_periods=1)
            .mean()
        )

        # week-over-week change (%)
        skill_data["week_over_week_change"] = (
            skill_data["posting_count"]
            .pct_change()
            .fillna(0) * 100
        )

        # momentum score (acceleration = change of change)
        skill_data["momentum_score"] = (
            skill_data["week_over_week_change"]
            .diff()
            .fillna(0)
        )

        # peak hiring season flag
        skill_data["month"] = skill_data["week_date"].dt.month
        skill_data["is_peak_hiring_season"] = skill_data["month"].apply(
            lambda m: 1 if m in [1, 2, 3, 7, 8, 9] else 0
        )

        # merge market volume
        skill_data = skill_data.merge(market_vol, on="week_date", how="left")

        all_features.append(skill_data)

    if not all_features:
        return pd.DataFrame()

    features_df = pd.concat(all_features, ignore_index=True)
    features_df = features_df.drop(columns=["month"], errors="ignore")

    print(f"✅ Features computed: {len(features_df)} rows, "
          f"{features_df['skill'].nunique()} skills")
    return features_df


def compute_demand_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use XGBoost to compute demand scores (0-100), along with
    trend (RISING/STABLE/DECLINING) and confidence (HIGH/MEDIUM/LOW).

    Uses the latest row per skill for scoring.
    """
    from xgboost import XGBRegressor
    from sklearn.preprocessing import MinMaxScaler

    feature_cols = [
        "rolling_avg_4weeks",
        "week_over_week_change",
        "momentum_score",
        "is_peak_hiring_season",
        "total_market_volume",
    ]

    # take only the latest week per skill for scoring
    latest = (
        features_df
        .sort_values("week_date")
        .groupby("skill")
        .last()
        .reset_index()
    )

    if latest.empty:
        return pd.DataFrame()

    # train XGBoost on full history to learn demand patterns
    # target = posting_count (we want to rank skills by demand)
    train_df = features_df.dropna(subset=feature_cols + ["posting_count"]).copy()

    if len(train_df) < 5:
        # fallback: score based on raw posting count
        latest["demand_score"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
            latest[["posting_count"]]
        ).flatten()
        latest["trend"] = "STABLE"
        latest["confidence"] = "LOW"
        return latest

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["posting_count"]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # predict on latest values
    X_latest = latest[feature_cols].fillna(0)
    predicted = model.predict(X_latest)

    # normalize to 0–100 score
    scaler = MinMaxScaler(feature_range=(0, 100))
    latest["demand_score"] = scaler.fit_transform(
        predicted.reshape(-1, 1)
    ).flatten().round(1)

    # classify trend
    latest["trend"] = latest["week_over_week_change"].apply(
        lambda x: "RISING" if x > 15 else ("DECLINING" if x < -15 else "STABLE")
    )

    # confidence based on data quality
    latest["confidence"] = latest["rolling_avg_4weeks"].apply(
        lambda x: "HIGH" if x > 100 else ("MEDIUM" if x > 30 else "LOW")
    )

    result = latest[[
        "skill", "posting_count", "demand_score", "trend", "confidence",
        "rolling_avg_4weeks", "week_over_week_change", "momentum_score",
        "is_peak_hiring_season", "total_market_volume",
    ]].sort_values("demand_score", ascending=False).reset_index(drop=True)

    print(f"✅ Demand scores computed for {len(result)} skills")
    return result, model


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_all
    from src.aggregator import aggregate_weekly

    df = load_all()
    weekly, top_skills = aggregate_weekly(df)

    print("\n─── Computing features ───")
    features = compute_features(weekly, top_skills)

    print("\n─── Computing demand scores ───")
    scores, model = compute_demand_scores(features)
    print(f"\nDemand Score Rankings:")
    print(scores[["skill", "demand_score", "trend", "confidence"]].to_string(index=False))
