from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from recommender import PathFinderRecommender
from groq import Groq

app = FastAPI(title="PathFinder SaaS API")

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load engine ONCE in memory when Uvicorn boots
recommender = PathFinderRecommender(index_path="search_index.pkl")

# In-memory user store for workspace authentication
registered_users = {
    "demo_learner@pathfinder.ai": "demo1234"
}

# Initialize Groq Client
def get_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import toml
            secrets = toml.load(".streamlit/secrets.toml")
            api_key = secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    return Groq(api_key=api_key) if api_key else None

client = get_groq()

# Schemas
class AuthRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    auth_type: str = "login"

class RecommendRequest(BaseModel):
    user_goal: str
    top_k: int = 4

class ExplainRequest(BaseModel):
    user_goal: str
    course_title: str
    skill_level: str = "Intermediate"

class RoadmapRequest(BaseModel):
    user_goal: str
    target_domain: str = "Data Science & AI"
    skill_level: str = "Intermediate"

# Authentication Endpoint
@app.post("/api/auth")
async def authenticate_user(req: AuthRequest):
    email = req.email.strip().lower()
    if not email or not req.password:
        raise HTTPException(status_code=400, detail="Email and password required.")
        
    if req.auth_type == "login":
        if email not in registered_users or registered_users[email] != req.password:
            raise HTTPException(status_code=401, detail="Invalid credentials.")
        return {"status": "success", "username": email.split("@")[0].title()}
    else:
        registered_users[email] = req.password
        display_name = req.name.strip() if req.name else email.split("@")[0].title()
        return {"status": "success", "username": display_name}

# Core Endpoints
@app.post("/api/recommend")
async def recommend_courses(req: RecommendRequest):
    if not req.user_goal.strip():
        raise HTTPException(status_code=400, detail="User goal cannot be empty.")
    
    df_results = recommender.recommend(req.user_goal, top_k=req.top_k)
    return {"results": df_results.to_dict(orient="records")}

@app.post("/api/explain")
async def explain_course(req: ExplainRequest):
    fallback_text = f"""- **Prerequisites Needed:** Foundational knowledge in core concepts and syntax.
- **Key Topics Covered:** Industry workflows, practical tools, and system design.
- **Recommended Hands-on Project:** Build an end-to-end working prototype for {req.course_title}.
- **Goal Relevance:** Directly provides the practical framework needed to achieve: "{req.user_goal}"."""

    if not client:
        return {"explanation": fallback_text}
    
    prompt = f"""
    You are a technical curriculum specialist.
    Learner Goal: "{req.user_goal}"
    Selected Course: "{req.course_title}"
    Target Level: {req.skill_level}

    Provide a direct, concise breakdown in bullet points using this exact format:
    - **Prerequisites Needed:** [List 2 core prerequisites]
    - **Key Topics Covered:** [List 3-4 key technical topics covered in this specific course]
    - **Recommended Hands-on Project:** [1 concrete real-world project/capstone to build]
    - **Goal Relevance:** [1 short sentence explaining why this course helps achieve the goal]
    """
    
    try:
        # Enforce 4-second max timeout so Groq queue delays never hang the frontend
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a concise technical advisor."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_tokens=250,
            timeout=4.0
        )
        return {"explanation": response.choices[0].message.content.strip()}
    except Exception:
        # Returns instant fallback formatting if Groq rate-limits or times out
        return {"explanation": fallback_text}

@app.post("/api/roadmap")
async def get_roadmap(req: RoadmapRequest):
    user_goal = req.user_goal.strip() if req.user_goal else "General Learning"
    
    # Dynamic fallback based on user input topic
    dynamic_fallback = [
        {"Step": 1, "Milestone": f"Foundations & Syntax for {user_goal}", "Topic": "Core Concepts, Environment & Tooling Setup", "Duration": "2 Weeks", "Status": "Completed"},
        {"Step": 2, "Milestone": f"Applied Skills & Building Blocks", "Topic": "Intermediate Implementations & Hands-on Projects", "Duration": "3 Weeks", "Status": "In Progress"},
        {"Step": 3, "Milestone": f"Advanced Architecture & Best Practices", "Topic": "Optimization, Design Patterns & Performance Tuning", "Duration": "4 Weeks", "Status": "Upcoming"},
        {"Step": 4, "Milestone": f"Production Deployment & Portfolio", "Topic": "Testing, Deployment & Capstone Integration", "Duration": "3 Weeks", "Status": "Upcoming"}
    ]

    if not client:
        return {"roadmap": dynamic_fallback}

    prompt = f"""
    Create a custom 4-step learning roadmap tailored specifically to achieving this goal: "{user_goal}".
    Return ONLY a raw JSON array of 4 objects with keys: "Step" (int 1-4), "Milestone" (str), "Topic" (str), "Duration" (str), "Status" (str: Completed, In Progress, or Upcoming).
    Do NOT include markdown block quotes or extra text.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise curriculum architect. You output strict raw JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.4,
            max_tokens=350,
            timeout=5.0
        )
        clean = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return {"roadmap": json.loads(clean)}
    except Exception:
        return {"roadmap": dynamic_fallback}