import json
import random
from typing import Dict, Any, Tuple, Optional

class BugTriageEnv:
    """
    Bug Report Triage Environment.
    Evaluates an agent on categorizing bugs into: Severity, Team, and Duplicate.
    """
    
    def __init__(self, dataset_path: str = "dataset.json"):
        """
        Initializes the environment by loading the dataset.
        
        Args:
            dataset_path (str): The path to the JSON dataset file.
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
            
        self.current_bug = None
        
    def reset(self, difficulty: Optional[str] = None) -> str:
        """
        Resets the environment and returns a new random bug report observation.
        
        Args:
            difficulty (str, optional): Filter bugs by difficulty ("easy", "medium", "hard").
        """
        pool = self.dataset
        
        # If dataset uses "difficulty" field, filter accordingly
        if difficulty:
            pool = [bug for bug in self.dataset if bug.get("difficulty") == difficulty.lower()]
            if not pool:
                print(f"Warning: No bugs found for difficulty '{difficulty}'. Falling back to full dataset.")
                pool = self.dataset
                
        self.current_bug = random.choice(pool)
        return self.state()
        
    def state(self) -> str:
        """
        Returns the current observation (bug report text).
        """
        if self.current_bug is None:
            return "No active bug. Call reset() first."
        return self.current_bug.get("text", "")

    def step(self, action: Dict[str, str]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Takes an action and evaluates it against the ground truth.
        
        Args:
            action (dict): Predicted severity, team, and duplicate status.
                
        Returns:
            observation (str): The current bug report text.
            reward (float): +1 for each correct field, -1 for incorrect.
            done (bool): Always True, as a triage is a single-step episode.
            info (dict): Detailed reward breakdown and ground truth context.
        """
        if self.current_bug is None:
            return self.state(), 0.0, True, {"error": "Environment not initialized."}
            
        if not isinstance(action, dict):
            return self.state(), -3.0, True, {"error": "Action must be a valid JSON dictionary."}

        # Extract predicted values safely
        pred_severity = str(action.get("severity", "")).strip().lower()
        pred_team = str(action.get("team", "")).strip().lower()
        pred_duplicate = str(action.get("duplicate", "")).strip().lower()

        # Get expected ground truth values
        exp_severity = self.current_bug.get("severity", "").lower()
        exp_team = self.current_bug.get("team", "").lower()
        exp_duplicate = self.current_bug.get("duplicate", "").lower()

        # Compute rewards: +1 if correct, else -1
        reward_severity = 1.0 if pred_severity == exp_severity else -1.0
        reward_team = 1.0 if pred_team == exp_team else -1.0
        reward_duplicate = 1.0 if pred_duplicate == exp_duplicate else -1.0

        # Total reward for this step
        total_reward = reward_severity + reward_team + reward_duplicate

        info = {
            "reward_breakdown": {
                "severity": reward_severity,
                "team": reward_team,
                "duplicate": reward_duplicate
            },
            "ground_truth": {
                "severity": exp_severity,
                "team": exp_team,
                "duplicate": exp_duplicate
            }
        }

        # done=True after one step
        done = True
        
        return self.state(), total_reward, done, info
