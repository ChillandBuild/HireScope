"""
HireScope — Aggregator
Group merged job data into weekly skill demand counts.
Filter to top 30 most-mentioned skills.
"""

import pandas as pd
from pathlib import Path


def aggregate_weekly(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Aggregate job postings into weekly skill demand counts.

    Args:
        df: merged dataframe from data_loader.load_all()
        top_n: keep only skills with this many total mentions (default 30)

    Returns:
        DataFrame with columns: week_date, skill, posting_count, source
    """
    # drop rows without dates
    df = df.dropna(subset=["date"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # floor dates to Monday of each week
    df["week_date"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    # ── identify top N skills by total mentions ─────────────────────────────
    skill_counts = df["skill"].value_counts()
    top_skills = skill_counts.head(top_n).index.tolist()
    df_top = df[df["skill"].isin(top_skills)]

    # ── aggregate by skill + week ───────────────────────────────────────────
    # combined (all sources)
    combined = (
        df_top
        .groupby(["week_date", "skill"])
        .agg(posting_count=("job_id", "nunique"))
        .reset_index()
    )
    combined["source"] = "combined"

    # per-source breakdown
    per_source = (
        df_top
        .groupby(["week_date", "skill", "source"])
        .agg(posting_count=("job_id", "nunique"))
        .reset_index()
    )

    result = pd.concat([combined, per_source], ignore_index=True)
    result = result.sort_values(["skill", "week_date"]).reset_index(drop=True)

    print(f"✅ Aggregated: {len(result):,} rows, {result['skill'].nunique()} skills, "
          f"{result['week_date'].nunique()} weeks")
    return result, top_skills


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_all

    df = load_all()
    weekly, top_skills = aggregate_weekly(df)

    print(f"\nTop 5 skills weekly summary:")
    for skill in top_skills[:5]:
        subset = weekly[(weekly["skill"] == skill) & (weekly["source"] == "combined")]
        total = subset["posting_count"].sum()
        weeks = len(subset)
        avg = subset["posting_count"].mean()
        print(f"  {skill}: {total:,} total postings over {weeks} weeks (avg {avg:.0f}/week)")

    print(f"\nWeek range: {weekly['week_date'].min()} → {weekly['week_date'].max()}")
    print(f"Sample data:")
    print(weekly[weekly["source"] == "combined"].head(10).to_string(index=False))
