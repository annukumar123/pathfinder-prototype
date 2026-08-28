# 🎯 PathFinder SaaS — AI-Powered Career & Learning Path Recommender

> **Round 2 Hackathon Prototype** | Live Web App & API Architecture

PathFinder SaaS is an AI-driven learning platform designed to help learners discover personalized course recommendations, analyze skill gaps, and generate dynamic step-by-step learning roadmaps based on natural language career goals.

---

## 🌟 Key Features

* **💬 Conversational Goal Recommender:** Enter free-form natural language objectives to query our sub-50ms TF-IDF vector search engine.
* **🤖 Groq LLM AI Justifications:** Generates real-time course breakdowns including prerequisites, covered topics, real-world capstone project recommendations, and goal alignment.
* **🗺️ Dynamic Roadmap Generator:** AI-powered step-by-step milestone sequencing tailored specifically to the user's input prompt.
* **📊 Skill & Competency Analytics:** Visual dashboards tracking overall goal completion percentages, learning velocity, and module mastery.
* **👤 Learner Profiling Engine:** Captures experience levels, target domains, existing skill baselines, and weekly hour commitments.
* **⚡ High-Performance Architecture:** Decoupled FastAPI + Uvicorn backend with async timeout protection and instant offline fallbacks.

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** HTML5, Tailwind CSS (CDN), Vanilla JavaScript
* **AI & Recommendation Engine:**
  * **Groq API** (`openai/gpt-oss-20b`) for dynamic explanations & roadmaps
  * **Scikit-learn** (`TfidfVectorizer`) for course candidate retrieval
  * **Pandas & NumPy** for dataset indexing and vectorized matrix matching

---

## 📁 Project Directory Structure

```text
pathfinder-prototype/
│
├── .streamlit/
│   └── secrets.toml         # GROQ_API_KEY storage
│
├── build_index.py           # Pre-computes search_index.pkl from train.csv
├── main.py                  # FastAPI + Uvicorn server (REST API Endpoints)
├── recommender.py           # TF-IDF vector search engine & candidate deduplication
├── train.csv                # Dataset containing course titles and learner reviews
├── search_index.pkl         # Serialized vector search index
│
└── index.html               # Single-Page Frontend Application (Tailwind CSS + JS)


🚀 Quickstart Guide

1️⃣ Clone & Install Dependencies

git clone [https://github.com/your-username/pathfinder-prototype.git](https://github.com/your-username/pathfinder-prototype.git)
cd pathfinder-prototype
pip install fastapi uvicorn scikit-learn pandas numpy groq toml

2️⃣ Build Search Index

python build_index.py

3️⃣ Launch Backend API Server

uvicorn main:app --reload --port 8000

4️⃣ Open Frontend
Open index.html directly in any standard browser or via VS Code Live Server.
