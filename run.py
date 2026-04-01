from env import BugTriageEnv

def format_reward_string(val):
    """Helper method to ensure positive numbers have a leading + symbol."""
    return f"+{val}" if val > 0 else str(val)

def test_agent(env, action, scenario_name):
    print(f"=== {scenario_name} ===")
    
    observation = env.state()
    print(f"Observation (Bug Report): '{observation}'\n")
    
    print("[Agent Action]")
    for k, v in action.items():
        print(f"  {k}: {v}")
        
    # Step into the environment properly
    next_obs, reward, done, info = env.step(action)
    
    # Check for graceful error handling
    if info.get("error"):
        print(f"\nEnvironment Error: {info['error']}")
        print("="*60 + "\n")
        return
        
    breakdown = info["reward_breakdown"]
    acc_pct = info["accuracy_percentage"]
    
    print("\n[Evaluation Results]")
    print(f"Total Reward: {reward} out of 4")
    print(f"Accuracy:     {acc_pct:.1f}% correct")
    
    print("\n[Reward Breakdown]")
    print(f"  - Severity:  {format_reward_string(breakdown['severity'])}")
    print(f"  - Team:      {format_reward_string(breakdown['team'])}")
    print(f"  - Duplicate: {format_reward_string(breakdown['duplicate'])}")
    print(f"  - Fix:       {format_reward_string(breakdown['fix'])}")
    
    print("="*60 + "\n")


def main():
    print("Starting Bug Report Triage Environment Demo...\n")
    # Initialize the robust environment
    env = BugTriageEnv(dataset_path="dataset.json")
    
    # --- Scenario 1: GOOD AGENT ---
    env.reset()
    # Mocking the state to guarantee deterministic output
    env.current_bug = {
        "text": "App crashes immediately after clicking the login button with valid credentials.",
        "severity": "high",
        "team": "backend",
        "duplicate": "no",
        "expected_fix_keywords": ["null", "exception", "api", "validation", "crash"]
    }
    good_action = {
        "severity": "high",
        "team": "backend",
        "duplicate": "no",
        # Including majority (>50%) keywords: exception, api, validation -> +1 Fix Reward
        "fix_suggestion": "The api is likely throwing an unhandled exception during login validation."
    }
    test_agent(env, good_action, "SCENARIO 1: GOOD AGENT")
    
    # --- Scenario 2: PARTIALLY CORRECT AGENT ---
    env.reset()
    env.current_bug = {
        "text": "Occasional 502 Bad Gateway errors during peak hours on the payment processing endpoint.",
        "severity": "high",
        "team": "infra",
        "duplicate": "no",
        "expected_fix_keywords": ["scale", "timeout", "load", "balancer", "gateway"]
    }
    partial_action = {
        "severity": "high",
        "team": "backend",  # Incorrect -> -1 Team Reward
        "duplicate": "no",
        # Uses 1 keyword out of 5 ('timeout'). Matches > 0% but <= 50% -> +0.5 Partial Fix Reward
        "fix_suggestion": "We should increase the server timeout config."
    }
    test_agent(env, partial_action, "SCENARIO 2: PARTIALLY CORRECT AGENT")

    # --- Scenario 3: POOR AGENT ---
    env.reset()
    env.current_bug = {
        "text": "Dashboard shows '$NaN' instead of the updated account balance after deposit.",
        "severity": "high",
        "team": "frontend",
        "duplicate": "no",
        "expected_fix_keywords": ["number", "parse", "format", "float"]
    }
    poor_action = {
        "severity": "low",
        # Missing 'team'
        # Missing 'duplicate'
        # 'fix_suggestion' completely devoid of target keywords -> 0 Fix Reward
        "fix_suggestion": "Restart the database server to clear the cache."
    }
    test_agent(env, poor_action, "SCENARIO 3: POOR AGENT")


if __name__ == "__main__":
    main()
