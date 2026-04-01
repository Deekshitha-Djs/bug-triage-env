🚀 Bug Report Triage Environment (OpenEnv Compatible)

🔗 Live Demo:
https://deekshithadjs-bug-triage-env.hf.space/

---

📌 Overview

This project implements an OpenEnv-compatible environment for automated bug triage.

It simulates a real-world software issue pipeline where an AI agent:

- Classifies incoming bug reports
- Assigns severity and responsible team
- Detects duplicates
- Suggests meaningful technical fixes

👉 Designed for AI agent evaluation, benchmarking, and real-world debugging workflows

---

🧠 Environment Design

🔹 Observation Space

A natural language bug report:

"Sometimes the profile image does not load when navigating back from settings."

---

🔹 Action Space

The agent must return:

{
"severity": "low | medium | high",
"team": "frontend | backend | infra",
"duplicate": "yes | no",
"fix_suggestion": "short technical fix explanation"
}

---

🎯 Reward System

The environment evaluates performance on a scale of -3.0 to +4.0

✅ Classification Scoring

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

📊 Additional Metrics

- Accuracy percentage
- Reward breakdown per field

---

📈 Difficulty Levels

Level| Description
Easy| UI bugs, typos, crashes
Medium| Caching issues, API delays
Hard| Race conditions, infra issues

---

⚙️ API Endpoints

🟢 Health Check

GET /

Response:
{
"message": "Bug triage API is running"
}

---

🔁 Reset Environment

GET /reset

Returns a new randomized bug scenario:

{
"observation": "...",
"action": {...},
"reward": ...,
"info": {...}
}

---

🧪 How to Test (For Judges)

1. Open the live demo
2. Visit "/reset"
3. Refresh multiple times to see different bug scenarios

👉 Confirms:

- Dynamic dataset
- Working reward system
- Proper API behavior

---

🤖 Model Strategy

Hybrid approach combining:

- Hugging Face model ("distilbert-base-uncased")
- Rule-based classification logic

✅ Benefits

- No API cost 💸
- Fully offline execution
- Deterministic & reproducible results

---

🐳 Docker Setup

Build Image

docker build -t bug-triage-env .

Run Container

docker run -p 7860:7860 bug-triage-env

---

🌐 Local Access

- Home → http://localhost:7860/
- Reset → http://localhost:7860/reset

---

🧪 Local Testing (Without Docker)

▶ Demo Mode

python run.py

Simulates:

- Good agent
- Partial agent
- Poor agent

---

🎮 Interactive Mode

python run_interactive.py

- Manually act as the agent
- Test different predictions

---

🤗 Hugging Face Deployment

Deployed as a Docker-based Hugging Face Space

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

- ✅ OpenEnv-compliant environment
- ✅ Dynamic reward scoring system
- ✅ Hybrid AI + rule-based agent
- ✅ Fully Dockerized deployment
- ✅ Hugging Face ready

---

⚠️ Notes

- No external paid APIs used
- Fully reproducible environment
- Designed for evaluation pipelines

---

👩‍💻 Author

Built as part of OpenEnv Hackathon 🚀
