"""
Configuration file for microgrid balancer RL agent.
All hyperparameters and settings in one place.
"""


# --- Microgrid Physical Parameters ---
BATTERY_CAPACITY_KWH = 100.0        # Total battery storage capacity (kWh)
BATTERY_MIN_SOC = 0.10              # Minimum state-of-charge (10%) to prevent damage
BATTERY_MAX_SOC = 0.95              # Maximum SOC (95%) to prevent overcharge
BATTERY_CHARGE_EFFICIENCY = 0.95    # Charging efficiency (5% loss)
BATTERY_DISCHARGE_EFFICIENCY = 0.95 # Discharging efficiency
BATTERY_MAX_CHARGE_RATE = 30.0      # Max charge power per step (kW)
BATTERY_MAX_DISCHARGE_RATE = 30.0   # Max discharge power per step (kW)

DIESEL_CAPACITY_KW = 50.0           # Max backup generator output (kW)
DIESEL_COST_PER_KWH = 15.0          # Cost in INR per kWh from diesel (expensive!)
DIESEL_EMISSION_FACTOR = 0.82       # kg CO2 per kWh from diesel

CRITICAL_LOAD_FRACTION = 0.6        # 60% of demand is critical (hospitals, lights)
NON_CRITICAL_LOAD_FRACTION = 0.4    # 40% can be shed (irrigation pumps, etc.)

# --- Simulation Parameters ---
TIMESTEP_MINUTES = 15               # 15-minute intervals (matches real dataset)
EPISODE_HOURS = 24                  # Simulate 24 hours per episode
STEPS_PER_EPISODE = EPISODE_HOURS * (60 // TIMESTEP_MINUTES)  # = 96 steps

# --- Reward Function Weights (from paper's objective) ---
BLACKOUT_PENALTY = -100.0           # Severe penalty for each kWh of unmet demand
DIESEL_PENALTY = -5.0               # Penalty per kWh of diesel usage
RENEWABLE_REWARD = +2.0             # Reward per kWh of renewable utilization
BATTERY_WEAR_PENALTY = -0.1         # Small penalty for battery cycling

# --- Adversarial Training Parameters (AT-DRAC-EBD) ---
ADVERSARY_BUDGET = 50.0             # Energy budget B for adversary (kWh)
ADVERSARY_DISTURBANCE_SCALE = 0.3   # Max perturbation as fraction of nominal value
ADVERSARY_ENABLED = True            # Set False to train standard PPO

# HILP (High-Impact Low-Probability) Disturbance Scenarios
HILP_SCENARIOS = {
    "load_surge":       {"demand_multiplier": 2.5, "probability": 0.05},
    "der_tripping":     {"solar_dropout": 0.8,     "probability": 0.05},
    "frequency_attack": {"oscillation_hz": 0.2,    "probability": 0.03},
    "cyber_noise":      {"sensor_bias": 0.25,       "probability": 0.04},
}

# --- PPO Hyperparameters (Stable Baselines 3) ---
PPO_CONFIG = {
    "policy": "MlpPolicy",          # Multi-layer perceptron policy
    "learning_rate": 3e-4,          # Adam optimizer learning rate
    "n_steps": 2048,                # Steps per update
    "batch_size": 64,               # Mini-batch size
    "n_epochs": 10,                 # Epochs per update
    "gamma": 0.99,                  # Discount factor γ (from paper)
    "gae_lambda": 0.95,             # GAE lambda for advantage estimation
    "clip_range": 0.2,              # PPO clip range
    "ent_coef": 0.01,               # Entropy coefficient for exploration
    "verbose": 1,
}
TOTAL_TIMESTEPS = 500_000           # Total training steps (increase for better results)

# --- File Paths ---
DATA_DIR = "data/raw"
PROCESSED_DATA_PATH = "data/processed_solar_data.csv"
MODEL_SAVE_PATH = "models/ppo_microgrid"
RESULTS_DIR = "results"

# --- Dataset Info ---
# Real dataset: Kaggle "Solar Power Generation Data" by Anil Karanam
# URL: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
# Contains: Plant 1 & Plant 2 generation + weather sensor data from India (2020)
KAGGLE_DATASET = "anikannal/solar-power-generation-data"
GENERATION_FILE = "Plant_1_Generation_Data.csv"   # Solar generation
WEATHER_FILE = "Plant_1_Weather_Sensor_Data.csv"  # Temperature, irradiation

