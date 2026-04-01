import json
from fastapi import FastAPI
from transformers import pipeline
import uvicorn
from env import BugTriageEnv
import random

with open("dataset.json") as f:
    dataset = json.load(f)

app = FastAPI()

print("Using Hugging Face model")
# Load the pre-downloaded pipeline (runs locally without API keys)
hf_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Keep existing environment intact
env = BugTriageEnv(dataset_path="dataset.json")

def classify_bug(text: str, sentiment_result: dict) -> dict:
    """Hybrid approach: Hugging Face pipeline + Rule-based logic."""
    text_lower = text.lower()
    
    # 1. Severity logic
    if any(word in text_lower for word in ["crash", "fatal", "critical", "down"]):
        severity = "high"
    elif sentiment_result["label"] == "NEGATIVE":
        severity = "medium"
    else:
        severity = "low"
        
    # 2. Team logic
    if any(word in text_lower for word in ["ui", "button", "screen", "frontend", "display", "layout"]):
        team = "frontend"
    elif any(word in text_lower for word in ["database", "server", "api", "backend", "connection", "sql"]):
        team = "backend"
    else:
        team = "infra"
        
    # 3. Duplicate logic
    if any(word in text_lower for word in ["same as", "already reported", "duplicate", "seen this"]):
        duplicate = "yes"
    else:
        duplicate = "no"
        
    return {
        "severity": severity,
        "team": team,
        "duplicate": duplicate
    }

@app.get("/reset")
def reset_env():
    print("Processing bug report")
    
    # 1. Reset BugTriageEnv and get bug observation
    observation = env.reset(difficulty="medium")
    
    # 2. Run observation through Hugging Face pipeline (truncated to 512 chars)
    truncated_obs = observation[:512]
    hf_result = hf_classifier(truncated_obs)[0]
    
    # 3. Use hybrid rule-based classification to generate reliable JSON output
    action = classify_bug(observation, hf_result)
    
    print("Final JSON output:")
    print(json.dumps(action, indent=2))
    
    # 4. Step environment to compute rewards
    obs, reward, done, info = env.step(action)
    
    # 5. Return structured JSON response
    return {
        "observation": obs,
        "action": action,
        "reward": reward,
        "info": info
    }
@app.get("/")
def home():
    return {"message":"Bug triage API is running"}

@app.post("/reset")
def reset():
    sample = random.choice(dataset)
    return sample

if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=7860)
