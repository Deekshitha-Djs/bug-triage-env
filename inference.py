import json
from fastapi import FastAPI
from transformers import pipeline
from env import BugTriageEnv
import os

API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
HF_TOKEN = os.getenv("HF_TOKEN")

try:
    from openai import OpenAI
except:
    pass

# Load dataset
with open("dataset.json") as f:
    dataset = json.load(f)

app = FastAPI()

print("Using Hugging Face model")

# Load model
hf_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Environment
env = BugTriageEnv(dataset_path="dataset.json")


def classify_bug(text: str, sentiment_result: dict) -> dict:
    text_lower = text.lower()

    # Severity
    if any(word in text_lower for word in ["crash", "fatal", "critical", "down"]):
        severity = "high"
    elif sentiment_result["label"] == "NEGATIVE":
        severity = "medium"
    else:
        severity = "low"

    # Team
    if any(word in text_lower for word in ["ui", "button", "screen", "frontend", "display", "layout"]):
        team = "frontend"
    elif any(word in text_lower for word in ["database", "server", "api", "backend", "connection", "sql"]):
        team = "backend"
    else:
        team = "infra"

    # Duplicate
    if any(word in text_lower for word in ["same as", "already reported", "duplicate", "seen this"]):
        duplicate = "yes"
    else:
        duplicate = "no"

    return {
        "severity": severity,
        "team": team,
        "duplicate": duplicate
    }


@app.get("/")
def home():
    return {"message": "Bug triage API is running"}


@app.get("/reset")
def reset_env():
    print("Processing bug report")

    observation = env.reset(difficulty="medium")

    truncated_obs = observation[:512]
    hf_result = hf_classifier(truncated_obs)[0]

    action = classify_bug(observation, hf_result)

    print("Final JSON output:")
    print(json.dumps(action, indent=2))

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs,
        "action": action,
        "reward": reward,
        "info": info
    }
