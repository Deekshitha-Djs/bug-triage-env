import json
import os
from fastapi import FastAPI
from openai import OpenAI
from env import BugTriageEnv

# ---- Required env variables ----
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")

# ---- Initialize OpenAI client (IMPORTANT for validator) ----
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# ---- Load dataset ----
with open("dataset.json") as f:
    dataset = json.load(f)

# ---- Initialize FastAPI ----
app = FastAPI()

print("Using LLM via provided proxy")

# ---- Environment ----
env = BugTriageEnv(dataset_path="dataset.json")


# ---- LLM Classification Logic ----
def classify_bug_with_llm(text: str) -> dict:
    truncated_text = text[:512]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Classify bug into severity (low/medium/high), team (frontend/backend/infra), duplicate (yes/no). Return JSON."
            },
            {
                "role": "user",
                "content": truncated_text
            }
        ]
    )

    output = response.choices[0].message.content

    try:
        result = json.loads(output)
    except:
        # fallback (IMPORTANT safety)
        result = {
            "severity": "medium",
            "team": "backend",
            "duplicate": "no"
        }

    return result


# ---- HEALTH CHECK ----
@app.get("/")
def home():
    return {"message": "Bug triage API is running"}


# ---- MAIN REQUIRED ENDPOINT ----
@app.post("/reset")
def reset():
    print("[START] task=bug_triage", flush=True)

    observation = env.reset(difficulty="medium")

    action = classify_bug_with_llm(observation)

    obs, reward, done, info = env.step(action)

    print(f"[STEP] step=1 reward={reward}", flush=True)
    print(f"[END] task=bug_triage score={reward} steps=1", flush=True)

    return {
        "observation": obs,
        "action": action,
        "reward": reward,
        "info": info
    }


# ---- For validator direct run ----
if __name__ == "__main__":
    print("[START] task=bug_triage", flush=True)

    observation = env.reset(difficulty="medium")

    action = classify_bug_with_llm(observation)

    obs, reward, done, info = env.step(action)

    print(f"[STEP] step=1 reward={reward}", flush=True)
    print(f"[END] task=bug_triage score={reward} steps=1", flush=True)
