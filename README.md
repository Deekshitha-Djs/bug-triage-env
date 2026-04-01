🚀 Bug Report Triage Environment (OpenEnv Compatible)

📌 Overview

This project implements an OpenEnv-compatible environment for automated bug triage.
It simulates a real-world software issue pipeline where an agent classifies bug reports and suggests fixes.

The environment evaluates:

- Bug severity
- Responsible engineering team
- Duplicate detection
- Fix suggestion relevance

It is designed for AI agent benchmarking and real-world debugging scenarios.

---

🧠 Environment Design

🔹 Observation Space

A natural language bug report string:

"Sometimes the profile image does not load when navigating back from settings."

---

🔹 Action Space

The agent must return a JSON object:

{
  "severity": "low | medium | high",
  "team": "frontend | backend | infra",
  "duplicate": "yes | no",
  "fix_suggestion": "short technical fix explanation"
}

---

🎯 Reward System

The environment evaluates agent performance on a scale of -3.0 to +4.0.

✅ Classification Rewards

Field| Correct| Incorrect
severity| +1| -1
team| +1| -1
duplicate| +1| -1

---

🛠 Fix Suggestion Scoring

Based on keyword relevance:

- +1.0 → Matches >50% of expected keywords
- +0.5 → Matches at least one keyword
- 0.0 → No relevant keywords

---

📊 Accuracy Metric

The environment also returns:

- Accuracy percentage
- Reward breakdown per field

---

📈 Difficulty Levels

Level| Description
Easy| UI bugs, typos, obvious crashes
Medium| Caching issues, API delays
Hard| Race conditions, time sync, infra bugs

---

⚙️ API Endpoints

🟢 Health Check

GET /

Response:

{
  "message": "Bug triage API is running"
}

---

🔁 Run Environment

GET /reset

Returns:

{
  "observation": "...",
  "action": {...},
  "reward": ...,
  "info": {...}
}

---

🤖 Model Strategy

This project uses a hybrid approach:

- Hugging Face model ("distilbert-base-uncased")
- Rule-based classification logic

This ensures:

- No API cost 💸
- Deterministic outputs
- Hackathon compliance

---

🐳 Docker Setup

Build Image

docker build -t bug-triage-env .

Run Container

docker run -p 7860:7860 bug-triage-env

---

🌐 Access API

- Home → http://localhost:7860/
- Reset → http://localhost:7860/reset

---

🧪 Running Locally (Without Docker)

Demo Scenarios

python run.py

Shows:

- Good agent
- Partial agent
- Poor agent

---

Interactive Mode

python run_interactive.py

Manually test your own predictions.

---

🤗 Hugging Face Deployment

This project is designed to run as a Docker-based Hugging Face Space.

Requirements:

- Dockerfile ✔
- openenv.yaml ✔
- FastAPI server ✔

---

📂 Project Structure

.
├── Dockerfile
├── openenv.yaml
├── README.md
├── requirements.txt
├── dataset.json
├── env.py
├── inference.py
├── run.py
├── run_interactive.py

---

🏆 Key Features

- OpenEnv-compliant environment
- Dynamic reward scoring system
- Hybrid AI + rule-based agent
- Dockerized deployment
- Hugging Face ready

---

⚠️ Notes

- No external paid APIs used
- Fully reproducible environment
- Designed for evaluation pipelines

---

👩‍💻 Author

Built as part of OpenEnv Hackathon 🚀
