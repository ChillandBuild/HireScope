"""
HireScope — Data Loader
Load and clean LinkedIn + Naukri job postings into a unified dataframe.
One row per job per skill.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LINKEDIN_POSTINGS = DATA_DIR / "job_postings.csv"
LINKEDIN_SKILLS   = DATA_DIR / "job_skills.csv"
LINKEDIN_SUMMARY  = DATA_DIR / "job_summary.csv"

NAUKRI_FILES = [
    DATA_DIR / "NaukriData_Data Science.csv",
    DATA_DIR / "NaukriData_data analytics.csv",
    DATA_DIR / "Naukri_Data_Scientist_and_Data_Analytics_Jobs_Data.csv",
]

# Reference date for Naukri relative timestamps
NAUKRI_REFERENCE_DATE = pd.Timestamp("2024-01-15")


# ── helpers ────────────────────────────────────────────────────────────────────

def _split_naukri_skills(raw: str) -> list[str]:
    """
    Naukri skills are concatenated without clear delimiters, e.g.:
      "Text miningCareer developmentdata scienceFinanceMachine learning"
    Strategy: split on transitions where a lowercase letter is followed
    by an uppercase letter (camelCase boundary).
    """
    if not isinstance(raw, str) or raw.strip() == "":
        return []

    # Insert a pipe '|' at camelCase boundaries:  "...ingCareer..." → "...ing|Career..."
    spaced = re.sub(r"([a-z])([A-Z])", r"\1|\2", raw)
    parts = [s.strip() for s in spaced.split("|") if s.strip()]
    return parts


def _parse_naukri_post_time(post_time: str, ref_date: pd.Timestamp) -> pd.Timestamp:
    """
    Convert Naukri relative timestamps like '1 Day Ago', '30+ Days Ago'
    into approximate absolute dates.
    """
    if not isinstance(post_time, str):
        return pd.NaT

    post_time = post_time.strip().lower()
    match = re.search(r"(\d+)\+?\s*(day|hour|month|week)", post_time)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if unit == "hour":
            return ref_date - pd.Timedelta(hours=num)
        elif unit == "day":
            return ref_date - pd.Timedelta(days=num)
        elif unit == "week":
            return ref_date - pd.Timedelta(weeks=num)
        elif unit == "month":
            return ref_date - pd.DateOffset(months=num)

    if "just now" in post_time or "few hours" in post_time:
        return ref_date

    return pd.NaT


# ── loaders ────────────────────────────────────────────────────────────────────

def load_linkedin() -> pd.DataFrame:
    """
    Load LinkedIn job_postings + job_skills, merge on job_link,
    explode skills into one row per job per skill.
    """
    # ── postings ──
    postings = pd.read_csv(LINKEDIN_POSTINGS)
    postings = postings.rename(columns={
        "job_link":     "job_id",
        "job_title":    "title",
        "company":      "company",
        "job_location": "location",
        "first_seen":   "date",
        "job_level":    "job_level",
    })
    postings["date"] = pd.to_datetime(postings["date"], errors="coerce")
    postings = postings[["job_id", "title", "company", "location", "date", "job_level"]]

    # ── skills ──
    skills = pd.read_csv(LINKEDIN_SKILLS)
    skills = skills.rename(columns={
        "job_link":   "job_id",
        "job_skills": "skills_raw",
    })
    # explode comma-separated skills → one row per skill
    skills["skill"] = skills["skills_raw"].str.split(",")
    skills = skills.explode("skill")
    skills["skill"] = skills["skill"].str.strip()
    skills = skills[skills["skill"].notna() & (skills["skill"] != "")]
    skills = skills[["job_id", "skill"]]

    # ── merge ──
    merged = postings.merge(skills, on="job_id", how="inner")
    merged["source"] = "linkedin"

    print(f"  LinkedIn: {len(merged):,} rows, {merged['skill'].nunique()} unique skills, "
          f"{merged['job_id'].nunique():,} unique jobs")
    return merged


def load_naukri() -> pd.DataFrame:
    """
    Load all Naukri CSVs, normalise columns, split concatenated skills,
    convert relative dates to approximate absolutes.
    """
    frames = []

    for fpath in NAUKRI_FILES:
        if not fpath.exists():
            print(f"  ⚠ Naukri file not found: {fpath.name}")
            continue

        df = pd.read_csv(fpath)

        # normalise column names — handle both 'Job_Titles' and 'Job Titles' forms
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower().replace(" ", "_")
            if "title" in cl:
                col_map[c] = "title"
            elif "company" in cl:
                col_map[c] = "company"
            elif "location" in cl:
                col_map[c] = "location"
            elif "skill" in cl:
                col_map[c] = "skills_raw"
            elif "post_time" in cl:
                col_map[c] = "post_time"
            elif "post_url" in cl or "url" in cl:
                col_map[c] = "job_id"
            elif "package" in cl:
                col_map[c] = "package"
            elif "experience" in cl:
                col_map[c] = "experience"

        df = df.rename(columns=col_map)

        # ensure required columns exist
        for col in ["title", "company", "location", "skills_raw"]:
            if col not in df.columns:
                df[col] = np.nan

        # generate job_id if not present
        if "job_id" not in df.columns:
            df["job_id"] = [f"naukri_{fpath.stem}_{i}" for i in range(len(df))]

        # parse date from post_time
        if "post_time" in df.columns:
            df["date"] = df["post_time"].apply(
                lambda x: _parse_naukri_post_time(x, NAUKRI_REFERENCE_DATE)
            )
        else:
            df["date"] = pd.NaT

        # explode skills
        df["skill"] = df["skills_raw"].apply(_split_naukri_skills)
        df = df.explode("skill")
        df["skill"] = df["skill"].str.strip()
        df = df[df["skill"].notna() & (df["skill"] != "")]

        df["source"] = "naukri"
        df["job_level"] = np.nan

        frames.append(df[["job_id", "title", "company", "location", "date", "skill", "source", "job_level"]])
        print(f"  Naukri ({fpath.name}): {len(df):,} rows")

    if not frames:
        return pd.DataFrame(columns=["job_id", "title", "company", "location",
                                      "date", "skill", "source", "job_level"])

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Naukri total: {len(combined):,} rows, {combined['skill'].nunique()} unique skills, "
          f"{combined['job_id'].nunique():,} unique jobs")
    return combined


def load_all() -> pd.DataFrame:
    """
    Load LinkedIn + Naukri, combine into single unified dataframe.

    Returns DataFrame with columns:
        job_id, title, company, location, date, skill, source, job_level
    """
    print("Loading LinkedIn data...")
    linkedin = load_linkedin()

    print("Loading Naukri data...")
    naukri = load_naukri()

    combined = pd.concat([linkedin, naukri], ignore_index=True)

    # standardise skill names: title case, strip whitespace
    combined["skill"] = combined["skill"].str.strip().str.title()

    # drop rows without a skill or date
    combined = combined.dropna(subset=["skill"])

    print(f"\n✅ Combined dataset: {len(combined):,} rows, "
          f"{combined['skill'].nunique()} unique skills, "
          f"{combined['job_id'].nunique():,} unique jobs")
    return combined


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_all()
    print(f"\nData loaded: {len(df):,} rows, {df['skill'].nunique()} skills")
    print(f"\nTop 20 skills:")
    print(df["skill"].value_counts().head(20).to_string())
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
