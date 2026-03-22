"""
HireScope — Forecaster
Prophet-based time-series forecasting per skill, 90 days forward.
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cmdstanpy.*")


def forecast_skill(skill_weekly: pd.DataFrame, periods: int = 13) -> pd.DataFrame:
    """
    Train Prophet on weekly posting counts for a single skill
    and forecast ~90 days forward (13 weeks).

    Args:
        skill_weekly: DataFrame with columns [week_date, posting_count]
        periods: number of weekly periods to forecast (13 ≈ 90 days)

    Returns:
        DataFrame with columns:
            ds, yhat, yhat_lower, yhat_upper, actual (NaN for future)
    """
    from prophet import Prophet

    # prophet needs columns named 'ds' and 'y'
    prophet_df = skill_weekly[["week_date", "posting_count"]].copy()
    prophet_df = prophet_df.rename(columns={"week_date": "ds", "posting_count": "y"})
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    # need at least 2 data points for Prophet
    if len(prophet_df) < 2:
        return pd.DataFrame()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq="W-MON")
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result = result.merge(prophet_df, on="ds", how="left")
    result = result.rename(columns={"y": "actual"})

    # clamp forecasts to >= 0 (can't have negative posting counts)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        result[col] = result[col].clip(lower=0)

    return result


def forecast_all_skills(weekly_df: pd.DataFrame,
                        top_skills: list[str],
                        periods: int = 13) -> dict[str, pd.DataFrame]:
    """
    Run Prophet forecast for each skill in top_skills.

    Args:
        weekly_df: output of aggregator.aggregate_weekly() (combined source only)
        top_skills: list of skill names to forecast
        periods: forecast horizon in weeks

    Returns:
        dict mapping skill name → forecast DataFrame
    """
    combined = weekly_df[weekly_df["source"] == "combined"].copy()
    forecasts = {}

    for i, skill in enumerate(top_skills):
        skill_data = combined[combined["skill"] == skill].copy()
        if skill_data.empty:
            continue

        print(f"  [{i+1}/{len(top_skills)}] Forecasting: {skill}...", end=" ")
        try:
            fc = forecast_skill(skill_data, periods=periods)
            if not fc.empty:
                fc["skill"] = skill
                forecasts[skill] = fc
                last_actual = skill_data["posting_count"].iloc[-1]
                last_forecast = fc["yhat"].iloc[-1]
                pct_change = ((last_forecast - last_actual) / max(last_actual, 1)) * 100
                print(f"done ({pct_change:+.0f}% in 90d)")
            else:
                print("skipped (insufficient data)")
        except Exception as e:
            print(f"error: {e}")

    print(f"\n✅ Forecasted {len(forecasts)} skills")
    return forecasts


def get_forecast_summary(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Produce a summary table of forecasts: current vs predicted counts,
    growth %, and simple trend classification.
    """
    rows = []
    for skill, fc in forecasts.items():
        actuals = fc.dropna(subset=["actual"])
        futures = fc[fc["actual"].isna()]

        if actuals.empty or futures.empty:
            continue

        last_actual = actuals["actual"].iloc[-1]
        avg_forecast = futures["yhat"].mean()
        pct_change = ((avg_forecast - last_actual) / max(last_actual, 1)) * 100

        if pct_change > 15:
            trend = "RISING"
        elif pct_change < -15:
            trend = "DECLINING"
        else:
            trend = "STABLE"

        rows.append({
            "skill": skill,
            "current_weekly": last_actual,
            "forecast_avg_weekly": round(avg_forecast, 1),
            "change_pct": round(pct_change, 1),
            "trend": trend,
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("change_pct", ascending=False).reset_index(drop=True)
    return summary


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_all
    from src.aggregator import aggregate_weekly

    df = load_all()
    weekly, top_skills = aggregate_weekly(df)

    print("\n─── Running Prophet forecasts ───")
    forecasts = forecast_all_skills(weekly, top_skills[:5])  # test with top 5 only

    print("\n─── Forecast Summary ───")
    summary = get_forecast_summary(forecasts)
    print(summary.to_string(index=False))
