import sys
from env import BugTriageEnv

def format_reward_string(val):
    """Helper method to ensure positive numbers have a leading + symbol."""
    return f"+{val}" if val > 0 else str(val)

def interactive_test(env, difficulty, step_num, total_steps):
    print(f"\n[{step_num}/{total_steps}] Getting a(n) {difficulty.upper()} bug report...")
    
    # Reset environment with specific difficulty
    observation = env.reset(difficulty=difficulty)
    
    print("\n" + "="*60)
    print("BUG REPORT:")
    print(observation)
    print("="*60)
    
    print("\n--- You are the AI Agent ---")
    print("Please analyze the bug report and provide triage details.")
    
    try:
        # Get user input
        severity = input("1. Severity (low, medium, high): ").strip()
        team = input("2. Team (frontend, backend, infra): ").strip()
        duplicate = input("3. Duplicate (yes, no): ").strip()
        fix = input("4. Fix Suggestion: ").strip()
        
        # Build action dictionary
        action = {
            "severity": severity,
            "team": team,
            "duplicate": duplicate,
            "fix_suggestion": fix
        }
        
        print("\nSubmitting action to environment...")
        next_obs, reward, done, info = env.step(action)
        
        if info.get("error"):
            print(f"Error executing step: {info['error']}")
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
        
        # Provide learning feedback by revealing the expected keywords
        keywords = info.get("ground_truth", {}).get("expected_fix_keywords", [])
        print(f"\n(Ground Truth expected fix keywords were: {keywords})")
        
    except KeyboardInterrupt:
        print("\nInteractive testing interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

def main():
    print("Welcome to Bug Report Triage - Interactive Mode!")
    print("You will act as the triage agent and attempt to classify bug reports.")
    
    env = BugTriageEnv(dataset_path="dataset.json")
    
    levels = ["easy", "medium", "hard"]
    
    for i, difficulty in enumerate(levels, 1):
        interactive_test(env, difficulty, i, len(levels))
        
        if i < len(levels):
            try:
                input("\nPress Enter to continue to the next difficulty level...")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
                
    print("\nInteractive testing complete! Great job!")

if __name__ == "__main__":
    main()
