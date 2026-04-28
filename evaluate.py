"""
evaluate.py - Comprehensive evaluation and comparison of agents.

Compares three agents across multiple metrics:
  1. PPO RL Agent (trained)
  2. Rule-Based Baseline (heuristic)
  3. AT-DRAC Baseline (adaptive)

Metrics tracked:
  - CSR: Critical Service Resilience
  - Blackouts: Total unserved energy (kWh)
  - Diesel usage: Backup generator consumption (kWh)
  - Renewable utilization: Solar energy used (kWh)
  - Episode reward: Total training reward

Evaluation modes:
  - Normal: Standard operation
  - Adversarial: With HILP disturbances (stress test)

Run: python evaluate.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.monitor import Monitor
from config import MODEL_SAVE_PATH, RESULTS_DIR, STEPS_PER_EPISODE, ADVERSARY_ENABLED
from env.microgrid_env import MicrogridEnv
from agents.baselines import RuleBasedBaseline, ATDRACBaseline


# ─────────────────────────────────────────────────────────────────
# Agent Wrappers
# ─────────────────────────────────────────────────────────────────

class PPOAgentWrapper:
    """Wrapper for Stable Baselines 3 PPO model."""

    def __init__(self, model_path: str):
        # Try loading with or without the .zip suffix for robustness
        try:
            self.model = PPO.load(model_path)
        except Exception:
            self.model = PPO.load(model_path + ".zip")
        self.name = "PPO RL Agent"

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """Predict action using PPO policy."""
        action, _ = self.model.predict(observation, deterministic=True)
        return np.array(action, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# Evaluation Function
# ─────────────────────────────────────────────────────────────────

def evaluate_agent(
    agent,
    env: MicrogridEnv,
    num_episodes: int = 10,
    agent_name: str = "Agent"
) -> dict:
    """
    Evaluate an agent over multiple episodes.

    Args:
        agent: Agent with select_action(obs) method
        env: MicrogridEnv environment
        num_episodes: Number of episodes to evaluate
        agent_name: Name for logging

    Returns:
        Dict with aggregated metrics across all episodes
    """

    episode_results = []

    for ep in range(num_episodes):
        reset_res = env.reset()
        # Support both VecEnv and raw Env
        if isinstance(reset_res, tuple):
            obs, info = reset_res
        else:
            obs = reset_res
            info = {}

        if hasattr(obs, 'ndim') and obs.ndim > 1:
            obs = obs[0]

        done = False
        truncated = False
        episode_reward = 0.0
        step_count = 0

        while not (done or truncated):
            # Get action from agent
            action = agent.select_action(obs)

            # If env is VecEnv expect batched actions/obs
            if hasattr(env, 'step') and isinstance(env, VecEnv):
                action_batch = np.array([action], dtype=np.float32)
                step_res = env.step(action_batch)
                obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = step_res
                obs = obs_batch[0]
                reward = float(reward_batch[0])
                done = bool(terminated_batch[0])
                truncated = bool(truncated_batch[0])
                info = info_batch[0] if isinstance(info_batch, (list, tuple)) else info_batch
            else:
                step_res = env.step(action)
                obs, reward, done, truncated, info = step_res

            episode_reward += float(reward)
            step_count += 1

        # End of episode: collect metrics
        csr = env.get_episode_csr()
        episode_log = env._episode_log.copy()

        episode_results.append({
            "episode": ep + 1,
            "steps": step_count,
            "total_reward": episode_reward,
            "csr": csr,
            "blackout_kwh": episode_log["blackout_kwh"],
            "diesel_kwh": episode_log["diesel_kwh"],
            "renewable_kwh": episode_log["renewable_kwh"],
            "total_demand_kwh": episode_log["total_demand_kwh"],
            "hilp_scenario": info.get("hilp_scenario", "None"),
        })

        # Print progress
        if (ep + 1) % max(1, num_episodes // 5) == 0:
            print(f"  [{agent_name}] Episode {ep+1}/{num_episodes} complete")

    # Aggregate results
    results_df = pd.DataFrame(episode_results)

    aggregated = {
        "agent": agent_name,
        "num_episodes": num_episodes,
        "avg_reward": results_df["total_reward"].mean(),
        "std_reward": results_df["total_reward"].std(),
        "avg_csr": results_df["csr"].mean(),
        "std_csr": results_df["csr"].std(),
        "avg_blackouts": results_df["blackout_kwh"].mean(),
        "std_blackouts": results_df["blackout_kwh"].std(),
        "avg_diesel": results_df["diesel_kwh"].mean(),
        "std_diesel": results_df["diesel_kwh"].std(),
        "avg_renewable": results_df["renewable_kwh"].mean(),
        "std_renewable": results_df["renewable_kwh"].std(),
        "episode_details": results_df,
    }

    return aggregated


# ─────────────────────────────────────────────────────────────────
# Comparison and Reporting
# ─────────────────────────────────────────────────────────────────

def generate_comparison_report(
    results: dict,
    save_path: str = None
) -> pd.DataFrame:
    """
    Generate a comparison report across all agents.

    Args:
        results: Dict with agent_name -> aggregated metrics
        save_path: Path to save CSV report

    Returns:
        DataFrame with comparison metrics
    """

    # Build summary table
    summary_rows = []
    for agent_name, metrics in results.items():
        summary_rows.append({
            "Agent": agent_name,
            "Avg Reward": f"{metrics['avg_reward']:.2f}",
            "Std Reward": f"{metrics['std_reward']:.2f}",
            "Avg CSR": f"{metrics['avg_csr']:.3f}",
            "Blackouts (kWh)": f"{metrics['avg_blackouts']:.1f}",
            "Diesel (kWh)": f"{metrics['avg_diesel']:.1f}",
            "Renewable (kWh)": f"{metrics['avg_renewable']:.1f}",
        })

    comparison_df = pd.DataFrame(summary_rows)

    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"[Evaluate] Comparison report saved to {save_path}")

    return comparison_df


# ─────────────────────────────────────────────────────────────────
# Main Evaluation Script
# ─────────────────────────────────────────────────────────────────

def main():
    """Run comprehensive evaluation of all agents."""

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[Evaluate] ================================================")
    print("[Evaluate] MICROGRID BALANCER: AGENT EVALUATION")
    print("[Evaluate] ================================================\n")

    # ─────────────────────────────────────────────────────────────
    # Test 1: Normal Operation (No Adversarial)
    # ─────────────────────────────────────────────────────────────

    print("[Evaluate] TEST 1: NORMAL OPERATION (No Adversarial Disturbances)")
    print("[Evaluate] ─────────────────────────────────────────────────────\n")

    env_normal = MicrogridEnv(adversarial=False, mode="eval")

    # Load trained PPO model
    ppo_model_path = os.path.join(MODEL_SAVE_PATH, "ppo_final")
    if os.path.exists(ppo_model_path + ".zip"):
        print(f"[Evaluate] Loading PPO model from {ppo_model_path}")
        ppo_agent = PPOAgentWrapper(ppo_model_path)
    else:
        print("[Evaluate] WARNING: PPO model not found!")
        print(f"[Evaluate] Expected: {ppo_model_path}.zip")
        print("[Evaluate] Run 'python train.py' first to train the model\n")
        ppo_agent = None

    # Initialize baselines
    rule_based = RuleBasedBaseline()
    at_drac = ATDRACBaseline()

    results_normal = {}

    if ppo_agent:
        print("[Evaluate] Evaluating PPO RL Agent (Normal)...")
        results_normal["PPO RL Agent"] = evaluate_agent(
            ppo_agent, env_normal, num_episodes=10, agent_name="PPO"
        )

    print("[Evaluate] Evaluating Rule-Based Baseline (Normal)...")
    results_normal["Rule-Based"] = evaluate_agent(
        rule_based, env_normal, num_episodes=10, agent_name="Rule-Based"
    )

    print("[Evaluate] Evaluating AT-DRAC Baseline (Normal)...")
    results_normal["AT-DRAC"] = evaluate_agent(
        at_drac, env_normal, num_episodes=10, agent_name="AT-DRAC"
    )

    env_normal.close()

    # Generate report
    print("\n[Evaluate] Normal Operation Results:")
    print("[Evaluate] ─────────────────────────────────────────────────────")
    comparison_normal = generate_comparison_report(
        results_normal,
        save_path=os.path.join(RESULTS_DIR, "evaluation_normal.csv")
    )
    print(comparison_normal.to_string(index=False))

    # ─────────────────────────────────────────────────────────────
    # Test 2: Adversarial Operation (HILP Disturbances)
    # ─────────────────────────────────────────────────────────────

    print("\n[Evaluate] TEST 2: ADVERSARIAL OPERATION (With HILP Disturbances)")
    print("[Evaluate] ─────────────────────────────────────────────────────\n")

    env_adversarial = MicrogridEnv(adversarial=True, mode="eval")

    results_adversarial = {}

    if ppo_agent:
        print("[Evaluate] Evaluating PPO RL Agent (Adversarial)...")
        results_adversarial["PPO RL Agent"] = evaluate_agent(
            ppo_agent, env_adversarial, num_episodes=10, agent_name="PPO (Adv)"
        )

    print("[Evaluate] Evaluating Rule-Based Baseline (Adversarial)...")
    results_adversarial["Rule-Based"] = evaluate_agent(
        rule_based, env_adversarial, num_episodes=10, agent_name="Rule-Based (Adv)"
    )

    print("[Evaluate] Evaluating AT-DRAC Baseline (Adversarial)...")
    results_adversarial["AT-DRAC"] = evaluate_agent(
        at_drac, env_adversarial, num_episodes=10, agent_name="AT-DRAC (Adv)"
    )

    env_adversarial.close()

    # Generate report
    print("\n[Evaluate] Adversarial Operation Results:")
    print("[Evaluate] ─────────────────────────────────────────────────────")
    comparison_adversarial = generate_comparison_report(
        results_adversarial,
        save_path=os.path.join(RESULTS_DIR, "evaluation_adversarial.csv")
    )
    print(comparison_adversarial.to_string(index=False))

    # ─────────────────────────────────────────────────────────────
    # Save Detailed Results
    # ─────────────────────────────────────────────────────────────

    for agent_name, metrics in results_normal.items():
        if "episode_details" in metrics:
            details_df = metrics.pop("episode_details")
            save_path = os.path.join(
                RESULTS_DIR, f"episodes_normal_{agent_name.lower().replace(' ', '_')}.csv"
            )
            details_df.to_csv(save_path, index=False)

    for agent_name, metrics in results_adversarial.items():
        if "episode_details" in metrics:
            details_df = metrics.pop("episode_details")
            save_path = os.path.join(
                RESULTS_DIR, f"episodes_adversarial_{agent_name.lower().replace(' ', '_')}.csv"
            )
            details_df.to_csv(save_path, index=False)

    # ─────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────

    print("\n[Evaluate] ================================================")
    print("[Evaluate] EVALUATION COMPLETE")
    print("[Evaluate] ================================================")
    print(f"[Evaluate] Results saved to: {RESULTS_DIR}/")
    print("[Evaluate] Files generated:")
    print("[Evaluate]   - evaluation_normal.csv")
    print("[Evaluate]   - evaluation_adversarial.csv")
    print("[Evaluate]   - episodes_normal_*.csv")
    print("[Evaluate]   - episodes_adversarial_*.csv")


if __name__ == "__main__":
    main()
