"""
train.py - Train the PPO agent on the microgrid environment.

This script trains a Proximal Policy Optimization (PPO) agent from Stable Baselines 3
on the MicrogridEnv, with support for adversarial training (AT-DRAC-EBD).

Features:
  - Continuous control (battery, diesel, load shedding, curtailment)
  - Real solar data (Kaggle dataset) or synthetic fallback
  - Adversarial HILP events (load surge, DER tripping, cyber attacks, etc.)
  - Episode logging (CSR, blackouts, diesel usage, renewable utilization)
  - Checkpoint saving every N episodes
  - Tensorboard integration for monitoring

Run: python train.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from config import (
    PPO_CONFIG, TOTAL_TIMESTEPS, MODEL_SAVE_PATH, RESULTS_DIR,
    STEPS_PER_EPISODE, ADVERSARY_ENABLED
)
from env.microgrid_env import MicrogridEnv

from config import PROCESSED_DATA_PATH, ADVERSARY_BUDGET, TIMESTEP_MINUTES


# ─────────────────────────────────────────────────────────────────
# Custom Callback for Episode Logging
# ─────────────────────────────────────────────────────────────────

class MicrogridCallback(BaseCallback):
    """
    Custom callback to log microgrid-specific metrics at episode end.
    Tracks: CSR, blackouts, diesel usage, renewable utilization.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_metrics = []
        self.logged_episode_ids = set()

    def _on_step(self) -> bool:
        """Called after each step. Check if episode ended."""
        # Get the wrapped environment (support VecEnv or raw env)
        env = self.model.get_env().envs[0] if hasattr(self.model.get_env(), 'envs') else self.model.get_env()

        # Check episode end and log once per episode
        if hasattr(env, 'episode_data'):
            ep_id = getattr(env, 'current_episode_id', None)
            if ep_id is not None and env.current_step >= STEPS_PER_EPISODE and ep_id not in self.logged_episode_ids:
                metrics = {
                    "episode": self.num_episodes,
                    "timestep": getattr(self, 'num_timesteps', None),
                    "csr": env.get_episode_csr(),
                    "blackout_kwh": env._episode_log["blackout_kwh"],
                    "diesel_kwh": env._episode_log["diesel_kwh"],
                    "renewable_kwh": env._episode_log["renewable_kwh"],
                    "total_demand_kwh": env._episode_log["total_demand_kwh"],
                    "hilp_scenario": env.active_hilp,
                }
                self.episode_metrics.append(metrics)
                self.logged_episode_ids.add(ep_id)
                self.num_episodes += 1

                if self.verbose >= 1 and self.num_episodes % 10 == 0:
                    print(
                        f"[Episode {self.num_episodes:4d}] "
                        f"CSR: {metrics['csr']:.3f} | "
                        f"Blackouts: {metrics['blackout_kwh']:.1f} kWh | "
                        f"Diesel: {metrics['diesel_kwh']:.1f} kWh | "
                        f"HILP: {metrics['hilp_scenario'] or 'None'}"
                    )

        return True

    def _on_training_start(self) -> None:
        self.num_episodes = 0


# ─────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────

def train_ppo_agent(
    adversarial: bool = ADVERSARY_ENABLED,
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_interval: int = 10000,
    at_drac_iters: int = 0,
    timesteps_per_iter: int = 20000,
):
    """
    Train a PPO agent on the microgrid environment.

    Args:
        adversarial: Whether to use adversarial training (AT-DRAC-EBD)
        total_timesteps: Total training steps (default: 500k from config)
        save_interval: Save checkpoint every N timesteps
    """

    # Create output directories
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[Train] Creating vectorized environments...")

    def make_env(seed_offset: int = 0):
        def _init():
            e = MicrogridEnv(adversarial=adversarial, mode="train")
            e = Monitor(e)
            try:
                e.seed(seed_offset + 0)
            except Exception:
                pass
            return e
        return _init

    n_envs = min(8, max(1, (os.cpu_count() or 1)))
    env_fns = [make_env(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Normalize observations (not rewards for stability) and keep stats
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    env = vec_env

    # set deterministic seed for reproducibility
    set_random_seed(0)

    # Configure PPO from config.py
    print(f"[Train] PPO Config: {PPO_CONFIG}")

    # Create PPO agent
    model = PPO(
        policy=PPO_CONFIG["policy"],
        env=env,
        learning_rate=PPO_CONFIG["learning_rate"],
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        ent_coef=PPO_CONFIG["ent_coef"],
        verbose=PPO_CONFIG["verbose"],
        tensorboard_log=os.path.join(RESULTS_DIR, "tensorboard_logs")
    )

    # Set up tensorboard logging
    log_dir = os.path.join(RESULTS_DIR, "tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)
    model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))

    # Custom callback for microgrid metrics
    # Callbacks: evaluation (save best) and periodic checkpoints
    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_PATH,
        log_path=RESULTS_DIR,
        n_eval_episodes=5,
        eval_freq=max(1000, total_timesteps // 50),
        deterministic=True,
    )

    checkpoint_cb = CheckpointCallback(save_freq=save_interval, save_path=MODEL_SAVE_PATH, name_prefix='checkpoint')
    custom_cb = MicrogridCallback(verbose=1)
    callback = CallbackList([eval_callback, checkpoint_cb, custom_cb])

    print("[Train] Starting training...")
    print(f"[Train] Total timesteps: {total_timesteps:,}")
    print(f"[Train] Adversarial: {adversarial}")

    def generate_adversary_plans(data_path: str, budget_kwh: float):
        """Generate in-memory adversary plans per episode.
        Returns: dict[episode_id] -> {'demand_delta': [...], 'solar_delta': [...]}
        Greedy allocation that prioritizes high-criticality steps (evening peaks).
        All perturbations are bounded by the provided energy budget (kWh) per episode.
        """
        df = pd.read_csv(data_path, parse_dates=["DATE_TIME"]) if isinstance(data_path, str) else data_path
        plans = {}
        dt_hours = TIMESTEP_MINUTES / 60.0
        episodes = df["EPISODE"].unique()
        for ep in episodes:
            ep_rows = df[df["EPISODE"] == ep].reset_index(drop=True)
            n = len(ep_rows)
            demand = ep_rows["VILLAGE_DEMAND_KW"].values.copy()
            solar = ep_rows["SOLAR_POWER_KW"].values.copy()

            # Base deficits (kW)
            deficits = np.clip(demand - solar, 0.0, None)

            # Compute per-step priority weights: evening hours get boosted weight
            hours = ep_rows["HOUR"].values if "HOUR" in ep_rows.columns else np.zeros_like(deficits)
            weights = np.ones_like(deficits)
            evening_mask = (hours >= 17) & (hours <= 21)
            weights[evening_mask] = 2.0

            # Weighted deficits guide allocation priority
            weighted_deficits = deficits * weights

            remaining = budget_kwh
            demand_delta = np.zeros(n, dtype=float)
            solar_delta = np.zeros(n, dtype=float)

            # Allocate to largest weighted deficits first
            order = np.argsort(-weighted_deficits)
            for idx in order:
                if remaining <= 1e-6:
                    break
                d_kw = deficits[idx]
                if d_kw <= 0:
                    continue
                # allow up to the deficit scaled by weight (more attack on critical steps)
                max_alloc_kwh = min(remaining, d_kw * dt_hours * weights[idx])
                if max_alloc_kwh <= 0:
                    continue
                delta_kw = max_alloc_kwh / dt_hours
                demand_delta[idx] += delta_kw
                remaining -= max_alloc_kwh

            # If budget remains, reduce solar on high-production steps (secondary attack)
            if remaining > 1e-6:
                for idx in order:
                    if remaining <= 1e-6:
                        break
                    s_kw = solar[idx]
                    if s_kw <= 0:
                        continue
                    max_alloc_kwh = min(remaining, s_kw * dt_hours)
                    if max_alloc_kwh <= 0:
                        continue
                    delta_kw = max_alloc_kwh / dt_hours
                    solar_delta[idx] -= delta_kw
                    remaining -= max_alloc_kwh

            plans[ep] = {"demand_delta": demand_delta.tolist(), "solar_delta": solar_delta.tolist()}
        return plans

    # Train the agent: if AT-DRAC iterations requested, run iterative adversarial training
    try:
        if adversarial and at_drac_iters > 0:
            print(f"[Train] Running AT-DRAC iterative training: {at_drac_iters} iterations, {timesteps_per_iter} timesteps/iter")
            for it in range(at_drac_iters):
                scaled_budget = ADVERSARY_BUDGET * (1.0 + 0.18 * float(it))
                print(f"[AT-DRAC] Iteration {it+1}/{at_drac_iters}: generating plans (budget={scaled_budget:.1f} kWh)")
                plans = generate_adversary_plans(PROCESSED_DATA_PATH, scaled_budget)

                # Apply plans to each underlying env inside the VecNormalize wrapper
                try:
                    base_venv = env.venv
                except Exception:
                    base_venv = env
                if hasattr(base_venv, 'envs'):
                    for v in base_venv.envs:
                        # v is Monitor wrapper; get inner env
                        inner = getattr(v, 'env', v)
                        micro = getattr(inner, 'env', inner)
                        setattr(micro, 'adversary_plan', plans)

                # Train for this iteration
                model.learn(total_timesteps=timesteps_per_iter, callback=callback)
                # Save intermediate checkpoint
                model.save(os.path.join(MODEL_SAVE_PATH, f"ppo_atdrac_iter_{it+1}"))
        else:
            model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("[Train] Training interrupted by user")

    # Save final model
    model_path = os.path.join(MODEL_SAVE_PATH, "ppo_final")
    model.save(model_path)
    print(f"[Train] Model saved to {model_path}")

    # Save VecNormalize statistics so evaluation can use same normalization
    try:
        if hasattr(env, 'save'):
            vecstat_path = os.path.join(MODEL_SAVE_PATH, 'vecnormalize.pkl')
            env.save(vecstat_path)
            print(f"[Train] VecNormalize stats saved to {vecstat_path}")
    except Exception:
        pass

    # Save episode metrics to CSV
    if custom_cb.episode_metrics:
        metrics_df = pd.DataFrame(custom_cb.episode_metrics)
        metrics_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[Train] Metrics saved to {metrics_path}")

        # Print summary statistics
        print("\n[Train] ===== TRAINING SUMMARY =====")
        print(f"Episodes completed: {len(custom_cb.episode_metrics)}")
        print(f"Avg CSR: {metrics_df['csr'].mean():.3f}")
        print(f"Avg Blackouts: {metrics_df['blackout_kwh'].mean():.1f} kWh")
        print(f"Avg Diesel Usage: {metrics_df['diesel_kwh'].mean():.1f} kWh")
        print(f"Avg Renewable Util: {metrics_df['renewable_kwh'].mean():.1f} kWh")

    env.close()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train PPO agent on microgrid environment"
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        default=ADVERSARY_ENABLED,
        help="Enable adversarial training (AT-DRAC-EBD)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--no-adversarial",
        action="store_true",
        help="Disable adversarial training"
    )
    parser.add_argument(
        "--at-drac-iters",
        type=int,
        default=0,
        help="Number of AT-DRAC adversarial iterations (default 0 = off)"
    )
    parser.add_argument(
        "--timesteps-per-iter",
        type=int,
        default=20000,
        help="Timesteps per AT-DRAC iteration when enabled"
    )

    args = parser.parse_args()

    # Handle conflicting flags
    if args.no_adversarial:
        adversarial = False
    else:
        adversarial = args.adversarial

    train_ppo_agent(
        adversarial=adversarial,
        total_timesteps=args.timesteps,
        at_drac_iters=args.at_drac_iters,
        timesteps_per_iter=args.timesteps_per_iter,
    )
