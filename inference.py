import json
import os
from fastapi import FastAPI
from env import BugTriageEnv

# ✅ REQUIRED ENV VARIABLES (from hackathon)
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# ✅ OpenAI client (through LiteLLM proxy)
from openai import OpenAI

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# ---- Load dataset ----
with open("dataset.json") as f:
    dataset = json.load(f)

# ---- Initialize app ----
app = FastAPI()

print("Using LLM via provided proxy", flush=True)

# ---- Environment ----
env = BugTriageEnv(dataset_path="dataset.json")


# ✅ LLM CLASSIFICATION (SAFE VERSION)
def classify_bug_with_llm(text: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Classify the bug into severity (low/medium/high), team (frontend/backend/infra), duplicate (yes/no). Return ONLY JSON."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )

        output = response.choices[0].message.content.strip()

        # Convert string → dict
        return json.loads(output)

    except Exception as e:
        print(f"LLM ERROR: {e}", flush=True)

        # 🔥 FALLBACK (VERY IMPORTANT)
        return {
            "severity": "medium",
            "team": "backend",
            "duplicate": "no"
        }


# ---- HEALTH CHECK ----
@app.get("/")
def home():
    return {"message": "Bug triage API is running"}


# ---- MAIN ENDPOINT ----
@app.post("/reset")
def reset():
    try:
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

    except Exception as e:
        print(f"RESET ERROR: {e}", flush=True)

        # fallback safe response
        print("[START] task=bug_triage", flush=True)
        print("[STEP] step=1 reward=0", flush=True)
        print("[END] task=bug_triage score=0 steps=1", flush=True)

        return {
            "observation": "",
            "action": {
                "severity": "medium",
                "team": "backend",
                "duplicate": "no"
            },
            "reward": 0,
            "info": {}
        }


# ---- LOCAL RUN (IMPORTANT FOR VALIDATOR) ----
if __name__ == "__main__":
    try:
        print("[START] task=bug_triage", flush=True)

        observation = env.reset(difficulty="medium")

        action = classify_bug_with_llm(observation)

        obs, reward, done, info = env.step(action)

        print(f"[STEP] step=1 reward={reward}", flush=True)
        print(f"[END] task=bug_triage score={reward} steps=1", flush=True)

    except Exception as e:
        print(f"MAIN ERROR: {e}", flush=True)

        print("[START] task=bug_triage", flush=True)
        print("[STEP] step=1 reward=0", flush=True)
        print("[END] task=bug_triage score=0 steps=1", flush=True)
