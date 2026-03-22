"""
HireScope — Opportunity Scout Agent
CrewAI agent that finds best-timed job opportunities based on
skill match and forecast data.
"""

import os
from pathlib import Path


def create_opportunity_scout_agent():
    """Create and return the Opportunity Scout CrewAI agent."""
    from crewai import Agent

    return Agent(
        role="Job Market Scout",
        goal="Find the best matching job opportunities for a student based on current market timing and skill demand forecasts",
        backstory="""You are a seasoned job market scout who specialises in the
        Indian tech job market. You track hiring patterns across LinkedIn and 
        Naukri, and you know exactly when companies are most actively hiring 
        for specific skills. You help students find opportunities that match 
        their skills AND are at peak hiring timing for maximum success.""",
        verbose=True,
        llm="groq/llama-3.3-70b-versatile",
        allow_delegation=False,
    )


def create_opportunity_task(agent, student_skills: list[str],
                            gap_report: str,
                            forecast_summary: list[dict]):
    """Create the opportunity scouting task."""
    from crewai import Task

    skills_str = ", ".join(student_skills)
    forecast_str = "\n".join(
        [f"- {s['skill']}: trend {s['trend']}, "
         f"change {s.get('change_pct', 'N/A')}%, confidence {s.get('confidence', 'N/A')}"
         for s in forecast_summary[:20]]
    )

    return Task(
        description=f"""Based on the student's skill profile and the gap analysis,
        find the best job opportunities right now.

STUDENT SKILLS: {skills_str}

GAP ANALYSIS REPORT:
{gap_report}

MARKET FORECAST (90-day):
{forecast_str}

YOUR TASK:
1. Match the student's skills to roles that are currently in high demand
2. Check the forecast — is demand for these roles rising or falling?
3. Identify the TOP 5 opportunities where timing is best right now
4. For each opportunity, provide an urgency score (1-10)

OUTPUT FORMAT:
## Top 5 Opportunities

### 1. [Role Title]
- **Matching skills**: which of the student's skills apply
- **Market demand**: current trend for this role type
- **Urgency score**: X/10 (10 = apply immediately)
- **Why now**: explain the timing advantage
- **Skills to highlight**: which skills to emphasise in application
- **Potential companies**: types of companies hiring for this

### 2. [Role Title]
... (repeat for all 5)

## Timing Summary
- Best time to apply: [specific advice]
- Skills to learn first: [if any gaps are critical]
- Market outlook: [brief 1-2 sentence summary]
""",
        expected_output="Top 5 job opportunities with urgency scores and timing advice",
        agent=agent,
    )


def run_opportunity_scouting(student_skills: list[str],
                              gap_report: str,
                              forecast_summary: list[dict]) -> str:
    """Execute the opportunity scouting pipeline."""
    from crewai import Crew, Process

    agent = create_opportunity_scout_agent()
    task = create_opportunity_task(agent, student_skills, gap_report, forecast_summary)

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
    print("Testing opportunity scout (requires GROQ_API_KEY in .env)...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        result = run_opportunity_scouting(
            student_skills=["Python", "SQL", "Flask", "JavaScript"],
            gap_report="Student lacks Machine Learning, Data Analysis, Cloud Computing skills.",
            forecast_summary=[
                {"skill": "Machine Learning", "trend": "RISING", "change_pct": 34.5, "confidence": "HIGH"},
                {"skill": "Python", "trend": "STABLE", "change_pct": 5.2, "confidence": "HIGH"},
            ]
        )
        print(result)
    except Exception as e:
        print(f"Agent test failed (expected if no API key): {e}")
