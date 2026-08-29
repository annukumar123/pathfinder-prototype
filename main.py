from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
import os
import json
import sqlite3
import re
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

# --- SQLite Database Initialization ---
DB_FILE = "pathfinder.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT
        )
    """)
    # Seed default demo account
    cursor.execute("INSERT OR IGNORE INTO users (email, password, name) VALUES (?, ?, ?)",
                   ("demo_learner@pathfinder.ai", "demo1234", "Demo Learner"))
    conn.commit()
    conn.close()

init_db()

# Load engine ONCE in memory when Uvicorn boots
recommender = PathFinderRecommender(index_path="search_index.pkl")

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
    email: EmailStr
    password: str
    name: str = ""
    auth_type: str = "login"  # "login" or "register"

    @validator("password")
    def validate_password(cls, v, values):
        # Apply strict password rule only during registration
        if values.get("auth_type") == "register":
            if len(v) < 8:
                raise ValueError("Password must be at least 8 characters long.")
            if not re.search(r"[A-Za-z]", v) or not re.search(r"\d", v):
                raise ValueError("Password must contain both letters and digits.")
        return v

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
        
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if req.auth_type == "login":
        cursor.execute("SELECT password, name FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="No account found with this email. Please register first."
            )
        if row[0] != req.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Incorrect password entered. Please try again."
            )
        
        display_name = row[1] if row[1] else email.split("@")[0].title()
        return {"status": "success", "username": display_name, "email": email}

    else:  # Registration Flow
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="An account already exists with this email address. Please sign in."
            )

        display_name = req.name.strip() if req.name else email.split("@")[0].title()
        cursor.execute("INSERT INTO users (email, password, name) VALUES (?, ?, ?)", (email, req.password, display_name))
        conn.commit()
        conn.close()

        return {"status": "success", "username": display_name, "email": email, "message": "Account created successfully."}

# Core Search & AI Endpoints
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
        return {"explanation": fallback_text}

@app.post("/api/roadmap")
async def get_roadmap(req: RoadmapRequest):
    user_goal = req.user_goal.strip() if req.user_goal else "General Learning"
    
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