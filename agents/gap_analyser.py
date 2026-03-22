"""
HireScope — Career Gap Analyser Agent
CrewAI agent that identifies skill gaps between a student's resume
and the 90-day market forecast.
"""

import os
from pathlib import Path


def extract_resume_text(pdf_path: str) -> str:
    """Extract text from a resume PDF using PyMuPDF."""
    import fitz  # pymupdf
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def create_gap_analyser_agent():
    """Create and return the Career Gap Analyser CrewAI agent."""
    from crewai import Agent

    return Agent(
        role="Career Intelligence Analyst",
        goal="Identify skill gaps between a student's current profile and the 90-day market forecast for India's tech job market",
        backstory="""You are an expert career intelligence analyst specialising in 
        the Indian tech job market. You have deep knowledge of which skills are 
        rising in demand and which are declining. You help students identify which 
        skills they should learn next to maximise their employability. You base 
        your analysis on real data from LinkedIn and Naukri job postings.""",
        verbose=True,
        llm="groq/llama-3.3-70b-versatile",
        allow_delegation=False,
    )


def create_gap_analysis_task(agent, resume_text: str, rising_skills: list[dict]):
    """Create the gap analysis task for the agent."""
    from crewai import Task

    skills_summary = "\n".join(
        [f"- {s['skill']}: demand score {s['demand_score']}/100, "
         f"trend {s['trend']}, confidence {s['confidence']}"
         for s in rising_skills[:20]]
    )

    return Task(
        description=f"""Analyse the following resume and compare it against the 
        top rising skills in the Indian job market.

RESUME TEXT:
{resume_text}

TOP 20 RISING SKILLS (from 90-day forecast):
{skills_summary}

YOUR TASK:
1. Extract all technical skills mentioned in the resume
2. Compare against the top 20 rising skills from the forecast
3. Identify skills that are IN DEMAND but NOT in the resume
4. Rank these gaps by forecast growth rate (highest demand first)
5. For each gap, explain WHY this skill matters and HOW to learn it

OUTPUT FORMAT:
## Skills Found in Resume
- List each skill

## Critical Skill Gaps (Priority Order)
For each gap:
### [Skill Name] — Demand Score: X/100
- **Why it matters**: brief explanation
- **How to learn**: specific free resources (courses, projects)
- **Time to learn**: estimated weeks

## Summary
- Total skills in resume: X
- Skills matching market demand: X  
- Critical gaps to fill: X
- Recommended learning path (1-2 sentences)
""",
        expected_output="A structured skill gap analysis with prioritised recommendations",
        agent=agent,
    )


def run_gap_analysis(resume_text: str, rising_skills: list[dict]) -> str:
    """Execute the gap analysis pipeline."""
    from crewai import Crew, Process

    agent = create_gap_analyser_agent()
    task = create_gap_analysis_task(agent, resume_text, rising_skills)

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
    sample_resume = """
    John Doe — Software Engineer
    Skills: Python, SQL, Flask, HTML, CSS, JavaScript, Git
    Experience: 2 years building web applications
    Education: B.Tech Computer Science, VIT Vellore
    Projects: E-commerce website, REST API development
    """
    
    sample_skills = [
        {"skill": "Machine Learning", "demand_score": 95.0, "trend": "RISING", "confidence": "HIGH"},
        {"skill": "Data Analysis", "demand_score": 88.0, "trend": "RISING", "confidence": "HIGH"},
        {"skill": "Python", "demand_score": 82.0, "trend": "STABLE", "confidence": "HIGH"},
        {"skill": "SQL", "demand_score": 78.0, "trend": "RISING", "confidence": "HIGH"},
        {"skill": "Cloud Computing", "demand_score": 72.0, "trend": "RISING", "confidence": "MEDIUM"},
    ]

    print("Testing gap analyser (requires GROQ_API_KEY in .env)...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        result = run_gap_analysis(sample_resume, sample_skills)
        print(result)
    except Exception as e:
        print(f"Agent test failed (expected if no API key): {e}")
