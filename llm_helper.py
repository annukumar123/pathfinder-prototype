import os
import json
import streamlit as st
from groq import Groq

def get_groq_client():
    if "GROQ_API_KEY" in st.secrets:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    elif os.getenv("GROQ_API_KEY"):
        return Groq(api_key=os.getenv("GROQ_API_KEY"))
    return None

def explain_recommendation(user_goal, course_title, skill_level="Intermediate"):
    client = get_groq_client()
    
    if not client:
        return "⚠️ API Key Missing: Set GROQ_API_KEY in .streamlit/secrets.toml."
    
    prompt = f"""
    You are a technical curriculum specialist.
    Learner Goal: "{user_goal}"
    Selected Course: "{course_title}"
    Target Level: {skill_level}

    Provide a direct, concise breakdown in bullet points using this exact format:
    - **Prerequisites Needed:** [List 2 core prerequisites]
    - **Key Topics Covered:** [List 3-4 key technical topics covered in this specific course]
    - **Goal Relevance:** [1 short sentence explaining why this course helps achieve the goal]
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a concise technical advisor. Give direct factual breakdowns without conversational fluff."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"🤖 AI Analysis: '{course_title}' directly aligns with {user_goal}. (API Error: {str(e)})"

def generate_dynamic_roadmap(user_goal, target_domain="Data Science & AI", skill_level="Intermediate"):
    client = get_groq_client()
    
    default_roadmap = [
        {"Step": 1, "Milestone": "Foundational Concepts & Setup", "Topic": "Core Syntax, Tools, & Environment Setup", "Duration": "2 Weeks", "Status": "Completed"},
        {"Step": 2, "Milestone": "Intermediate Applied Skills", "Topic": "Data Processing, Libraries & API Integration", "Duration": "3 Weeks", "Status": "In Progress"},
        {"Step": 3, "Milestone": "Advanced Architecture & Models", "Topic": "Model Architecture, Optimization & Fine-tuning", "Duration": "4 Weeks", "Status": "Upcoming"},
        {"Step": 4, "Milestone": "Production Deployment & Portfolio", "Topic": "Containerization, Cloud CI/CD & Capstone Project", "Duration": "3 Weeks", "Status": "Upcoming"}
    ]
    
    if not client:
        return default_roadmap

    prompt = f"""
    Create a 4-step learning roadmap for a learner targeting the domain '{target_domain}' with the goal: '{user_goal}'.
    Their current level is '{skill_level}'.
    
    Return ONLY a JSON array with exactly 4 objects using keys: "Step" (int), "Milestone" (str), "Topic" (str), "Duration" (str), "Status" (str - choose from Completed, In Progress, Upcoming).
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You output strict raw JSON only. Do not include markdown formatting or extra text."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.5,
            max_tokens=350
        )
        clean_text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception:
        return default_roadmap