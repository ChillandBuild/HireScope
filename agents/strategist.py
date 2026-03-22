"""
HireScope — Application Strategy Advisor Agent
CrewAI agent that analyses a specific job description and advises
the student on whether to apply NOW, wait, or skip.
"""

import os
from pathlib import Path


def create_strategist_agent():
    """Create and return the Strategy Advisor CrewAI agent."""
    from crewai import Agent

    return Agent(
        role="Application Strategy Advisor",
        goal="Tell a student exactly when and whether to apply for a specific job, based on their profile match and market timing",
        backstory="""You are a strategic career advisor who combines data-driven 
        market analysis with practical job application advice. You understand 
        that timing is everything in the job market — applying too early when 
        skills don't match, or too late when demand has peaked, both reduce 
        success rates. You give specific, actionable advice based on real 
        market data from the Indian tech job market.""",
        verbose=True,
        llm="groq/llama-3.3-70b-versatile",
        allow_delegation=False,
    )


def create_strategy_task(agent, job_description: str,
                          student_skills: list[str],
                          forecast_summary: list[dict]):
    """Create the strategy analysis task."""
    from crewai import Task

    skills_str = ", ".join(student_skills)
    forecast_str = "\n".join(
        [f"- {s['skill']}: trend {s['trend']}, "
         f"change {s.get('change_pct', 'N/A')}%"
         for s in forecast_summary[:15]]
    )

    return Task(
        description=f"""Analyse this job description and advise the student 
        on whether to apply.

JOB DESCRIPTION:
{job_description}

STUDENT'S SKILLS: {skills_str}

MARKET FORECAST DATA:
{forecast_str}

YOUR TASK:
1. Extract all required skills from the job description
2. Calculate match score (% of required skills the student has)
3. Check forecast trend for this role type — is demand rising/falling?
4. Make a clear recommendation: APPLY NOW / WAIT / SKIP

DECISION CRITERIA:
- APPLY NOW: match ≥70% AND demand is rising or stable
- WAIT: match 40-69% — student should learn missing skills first
- SKIP: match <40% OR demand is sharply declining

OUTPUT FORMAT:
## Job Analysis

### Required Skills Identified
- List each skill from the JD

### Match Score: X%
- **Skills you have**: [list]
- **Skills you're missing**: [list]

### Market Timing
- Demand trend for this role: [rising/stable/declining]
- Forecast confidence: [high/medium/low]

## 🟢 RECOMMENDATION: [APPLY NOW / ⏳ WAIT / 🔴 SKIP]

### Reasoning
[2-3 sentences explaining the decision]

### If WAIT — Action Plan
- Skills to learn: [list with estimated time]
- Target application window: [when to apply]

### If APPLY NOW — Tips
- Highlight these skills: [list]
- Cover letter focus: [specific advice]
""",
        expected_output="A clear apply/wait/skip recommendation with detailed reasoning",
        agent=agent,
    )


def run_strategy_analysis(job_description: str,
                           student_skills: list[str],
                           forecast_summary: list[dict]) -> str:
    """Execute the strategy analysis pipeline."""
    from crewai import Crew, Process

    agent = create_strategist_agent()
    task = create_strategy_task(agent, job_description, student_skills, forecast_summary)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing strategist (requires GROQ_API_KEY in .env)...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        result = run_strategy_analysis(
            job_description="""
            Data Scientist at TCS, Bangalore
            Requirements: Python, Machine Learning, SQL, TensorFlow, 
            Data Visualization, 2+ years experience
            """,
            student_skills=["Python", "SQL", "Flask", "JavaScript"],
            forecast_summary=[
                {"skill": "Machine Learning", "trend": "RISING", "change_pct": 34.5},
                {"skill": "Python", "trend": "STABLE", "change_pct": 5.2},
                {"skill": "TensorFlow", "trend": "RISING", "change_pct": 22.1},
            ]
        )
        print(result)
    except Exception as e:
        print(f"Agent test failed (expected if no API key): {e}")
