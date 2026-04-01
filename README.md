# Bug Report Triage Environment

## Project Overview
This is an OpenEnv-compatible environment designed for bug triage with advanced fix suggestion evaluation. The environment simulates a real-world software issue queue where an AI agent (or testing user) acts as a triage manager. The objective is to correctly categorize incoming bug reports and propose practically helpful technical fixes.

## How It Works
The environment is structured as a single-turn scenario per bug.

- **Observation**: A string detailing an incoming bug report.
- **Action**: The agent must output a Python dictionary containing:
  - `severity`: "low", "medium", or "high"
  - `team`: "frontend", "backend", or "infra"
  - `duplicate`: "yes" or "no"
  - `fix_suggestion`: A short string proposing how to resolve the issue.

The agent's action is then processed by a robust **reward system**, scoring exactly how well the predictions align with the actual ground truth, strictly tracking fractional relevance scoring for fix suggestions.

## Running the Project

### Scenario Demo (Recommended for Judges)
Run the following script to watch fully-simulated agents perform within the environment:
```bash
python run.py
```
This script acts as the evaluation system showcase by demonstrating three different scenarios:
1. **Good Agent**: Identifies variables perfectly and scores maximum points.
2. **Partially Correct Agent**: Assesses some categories correctly and uses some matching context clues for the fix.
3. **Poor Agent**: Completely hallucinates missing variables and writes an irrelevant fix suggestion.

### Interactive Mode (Manual Testing)
If you want to step into the role of the AI agent, you can manually interact with the issues! Run:
```bash
python run_interactive.py
```
- In this mode, you act as the triage agent.
- You will be given an observation (bug report) cycling natively across the 3 varying difficulty levels.
- You will be prompted to manually type the triage responses. Test different inputs and see how the reward system scores you based on what you submit!

## Reward System
The environment calculates rewards dynamically on a scale of **-3.0 to +4.0**, assigning points for each categorized field:

- **Classification (+1 / -1)**:
  - `severity`, `team`, and `duplicate` each grant **+1** if exactly correct.
  - Getting them wrong (or omitting them from the action) penalizes the agent by **-1**.

- **Fix Suggestion Relevance (+1 / +0.5 / 0)**: 
  The environment checks the agent's fix string against a dynamic set of `expected_fix_keywords`.
  - **+1.0**: Highly relevant (the fix uses a majority `> 50%` of the expected keywords).
  - **+0.5**: Partially relevant (the fix identifies at least `> 0%` of the keywords).
  - **0.0**: Irrelevant (no keywords matched, or fix omitted completely).

The results also return an isolated **Accuracy Percentage**, displaying exactly how many categories correctly resolved against a total pool of 4 potential points.

## Difficulty Levels
The dataset scales across 3 completely fleshed-out difficulty levels:
- **Easy**: Simple typos, obvious UI bugs, and explicit system crashes.
- **Medium**: Disguised caching errors, delayed cron jobs, synchronization issues.
- **Hard**: Timezone drifting issues, intermittent WebSocket leaks, concurrent race transactions.
