"""
HireScope — CrewAI Orchestration
Sequential pipeline: Gap Analysis → Opportunity Scouting → Strategy Advice
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def run_full_pipeline(resume_text: str,
                       job_description: str,
                       rising_skills: list[dict],
                       forecast_summary: list[dict]) -> dict:
    """
    Run the full CrewAI pipeline:
    1. Gap Analyser analyses resume against forecast
    2. Opportunity Scout finds best-timed opportunities
    3. Strategist advises on specific JD

    Args:
        resume_text: extracted text from resume PDF
        job_description: JD text for strategy analysis
        rising_skills: list of dicts with skill, demand_score, trend, confidence
        forecast_summary: list of dicts with skill, trend, change_pct

    Returns:
        dict with keys: gap_report, opportunities, strategy
    """
    from crewai import Agent, Task, Crew, Process

    results = {}

    # ── Agent 1: Gap Analysis ──────────────────────────────────────────────
    print("\n🔍 Running Career Gap Analysis...")
    gap_agent = Agent(
        role="Career Intelligence Analyst",
        goal="Identify skill gaps between student profile and 90-day market forecast",
        backstory="Expert career analyst specialising in India's tech job market. "
                  "You base analysis on real data from LinkedIn and Naukri postings.",
        verbose=True,
        llm="groq/llama-3.3-70b-versatile",
        allow_delegation=False,
    )

    skills_list = "\n".join(
        [f"- {s['skill']}: score {s['demand_score']}/100, {s['trend']}"
         for s in rising_skills[:20]]
    )

    gap_task = Task(
        description=f"""Analyse this resume against rising market skills.
        
RESUME: {resume_text}

TOP 20 RISING SKILLS:
{skills_list}

Find skills in the forecast NOT in the resume. Rank gaps by demand score.
Output: prioritised skill gap report with learning recommendations.""",
        expected_output="Structured skill gap analysis with recommendations",
        agent=gap_agent,
    )

    # ── Agent 2: Opportunity Scout ─────────────────────────────────────────
    print("\n🔭 Running Opportunity Scouting...")
    scout_agent = Agent(
        role="Job Market Scout",
        goal="Find best matching opportunities based on timing and skill demand",
        backstory="Seasoned market scout tracking hiring patterns across "
                  "LinkedIn and Naukri in India.",
        verbose=True,
        llm="groq/llama-3.3-70b-versatile",
        allow_delegation=False,
    )

    forecast_str = "\n".join(
        [f"- {s['skill']}: {s['trend']}, {s.get('change_pct', 'N/A')}%"
         for s in forecast_summary[:15]]
    )

    scout_task = Task(
        description=f"""Using the gap analysis results, find top 5 opportunities.

MARKET FORECAST:
{forecast_str}

Match skills to roles at peak hiring timing. Provide urgency scores (1-10).
Output: top 5 opportunities with timing advice.""",
        expected_output="Top 5 job opportunities with urgency scores",
        agent=scout_agent,
    )

    # ── Agent 3: Strategy Advisor ──────────────────────────────────────────
    print("\n🎯 Running Strategy Analysis...")
    strategy_agent = Agent(
        role="Application Strategy Advisor",
        goal="Advise student when to apply for a specific job",
        backstory="Strategic advisor combining data-driven market analysis "
                  "with practical application advice for Indian tech market.",
        verbose=True,
        llm="groq/llama-3.3-70b-versatile",
        allow_delegation=False,
    )

    strategy_task = Task(
        description=f"""Analyse this JD and advise whether to apply.

JOB DESCRIPTION: {job_description}

MARKET FORECAST:
{forecast_str}

Calculate match score. Check demand trends.
Recommend: APPLY NOW / WAIT / SKIP with specific reasoning.""",
        expected_output="Apply/wait/skip recommendation with reasoning",
        agent=strategy_agent,
    )

    # ── Execute sequential pipeline ────────────────────────────────────────
    crew = Crew(
        agents=[gap_agent, scout_agent, strategy_agent],
        tasks=[gap_task, scout_task, strategy_task],
        process=Process.sequential,
        verbose=True,
    )

    output = crew.kickoff()

    # extract individual task results
    task_outputs = output.tasks_output if hasattr(output, 'tasks_output') else []

    results["gap_report"] = str(task_outputs[0]) if len(task_outputs) > 0 else str(output)
    results["opportunities"] = str(task_outputs[1]) if len(task_outputs) > 1 else ""
    results["strategy"] = str(task_outputs[2]) if len(task_outputs) > 2 else ""

    print("\n✅ Full pipeline complete!")
    return results


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing full CrewAI pipeline (requires GROQ_API_KEY in .env)...")
    try:
        result = run_full_pipeline(
            resume_text="Python developer with SQL, Flask, JavaScript, Git. B.Tech CS from VIT.",
            job_description="Data Scientist at Infosys. Requirements: Python, ML, SQL, TensorFlow.",
            rising_skills=[
                {"skill": "Machine Learning", "demand_score": 95, "trend": "RISING", "confidence": "HIGH"},
                {"skill": "Python", "demand_score": 82, "trend": "STABLE", "confidence": "HIGH"},
            ],
            forecast_summary=[
                {"skill": "Machine Learning", "trend": "RISING", "change_pct": 34.5},
                {"skill": "Python", "trend": "STABLE", "change_pct": 5.2},
            ],
        )
        for key, val in result.items():
            print(f"\n{'='*60}\n{key.upper()}\n{'='*60}\n{val[:500]}")
    except Exception as e:
        print(f"Pipeline test failed (expected if no API key): {e}")
