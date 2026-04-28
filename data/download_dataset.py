"""
data/download_dataset.py
========================
Downloads the real Kaggle Solar Power Generation dataset (India solar plants).
Falls back to synthetic data generation if Kaggle credentials are not available.

GitHub Copilot: This file handles all data acquisition.
Run with: python data/download_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, KAGGLE_DATASET, STEPS_PER_EPISODE

# ─────────────────────────────────────────────────────────
# Option A: Download Real Kaggle Dataset
# Requires: pip install kaggle + ~/.kaggle/kaggle.json API key
# ─────────────────────────────────────────────────────────

def download_kaggle_dataset():
    """
    Downloads Solar Power Generation Data from Kaggle.
    Real data from 2 solar plants in India (May-June 2020).
    
    Setup: Place kaggle.json in ~/.kaggle/ or set env vars:
        KAGGLE_USERNAME=your_username
        KAGGLE_KEY=your_api_key
    """
    try:
        import kaggle
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"[Kaggle] Downloading: {KAGGLE_DATASET}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=DATA_DIR,
            unzip=True
        )
        print(f"[Kaggle] [OK] Downloaded to {DATA_DIR}/")
        return True
    except Exception as e:
        print(f"[Kaggle] FAILED: {e}")
        print("[Fallback] Generating synthetic data instead...")
        return False


# ─────────────────────────────────────────────────────────
# Option B: Synthetic Data Generator (Fallback)
# Mimics the real Kaggle dataset structure + realistic patterns
# ─────────────────────────────────────────────────────────

def generate_synthetic_solar_data(n_days: int = 34) -> pd.DataFrame:
    """
    Generates synthetic solar generation data matching the Kaggle dataset structure.
    Uses realistic irradiation curves, temperature patterns, and demand profiles.
    
    Args:
        n_days: Number of days to simulate (default 34, matches real dataset)
    
    Returns:
        DataFrame with columns matching Plant_1_Generation_Data.csv
    """
    print(f"[Synthetic] Generating {n_days} days of realistic solar data...")
    
    np.random.seed(42)  # Reproducible results
    
    # Generate timestamps at 15-minute intervals (matches real dataset)
    timestamps = pd.date_range(
        start="2020-05-15 00:00:00",
        periods=n_days * STEPS_PER_EPISODE,
        freq="15T"
    )
    
    records = []
    
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60.0
        day_of_year = ts.day_of_year
        
        # ── Solar Irradiation Model ─────────────────────────────────
        # Bell curve: peaks at solar noon (hour=12), zero at night
        # Varies with season (day_of_year) and random cloud cover
        
        solar_elevation = max(0, np.sin(np.pi * (hour - 6) / 12))  # 0 at 6am and 6pm
        
        # Seasonal variation: India summer has high irradiation
        seasonal_factor = 0.85 + 0.15 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Random cloud events (10% chance of partial cloud, 5% full cloud)
        cloud_factor = np.random.choice(
            [1.0, 0.6, 0.1],
            p=[0.85, 0.10, 0.05]
        )
        
        # Add slight Gaussian noise for realism
        irradiation = solar_elevation * seasonal_factor * cloud_factor * 0.95
        irradiation = max(0, irradiation + np.random.normal(0, 0.02))
        
        # ── Power Generation ─────────────────────────────────────────
        # Based on actual Plant 1: ~30kW peak DC, ~28kW peak AC
        # Some inverters are less efficient (0.92-0.97 range)
        
        peak_dc_power = 30.0 * (1 + np.random.normal(0, 0.05))   # kW with noise
        inverter_efficiency = np.random.uniform(0.92, 0.97)
        
        dc_power = peak_dc_power * irradiation
        ac_power = dc_power * inverter_efficiency
        
        # Add small inverter noise
        dc_power = max(0, dc_power + np.random.normal(0, 0.3))
        ac_power = max(0, ac_power + np.random.normal(0, 0.2))
        
        # ── Temperature Model ────────────────────────────────────────
        # India summer: 25°C night → 42°C peak
        ambient_temp = 32 + 10 * solar_elevation + np.random.normal(0, 1.5)
        module_temp = ambient_temp + 15 * solar_elevation + np.random.normal(0, 2)
        
        # ── Village Demand Model ─────────────────────────────────────
        # Morning peak (7-9am): cooking, pumps
        # Evening peak (18-21h): lights, fans, cooking
        
        morning_peak = 0.7 * np.exp(-((hour - 8.0) ** 2) / 2)
        evening_peak = 1.0 * np.exp(-((hour - 19.0) ** 2) / 3)
        base_demand = 0.3  # Constant base load
        
        demand_kw = 40 * (base_demand + morning_peak + evening_peak)
        demand_kw = max(5, demand_kw + np.random.normal(0, 3))  # min 5kW load
        
        records.append({
            "DATE_TIME": ts,
            "PLANT_ID": 4135001,
            "SOURCE_KEY": "1BY6WEcLGh8j5v7",
            "DC_POWER": round(dc_power, 4),
            "AC_POWER": round(ac_power, 4),
            "DAILY_YIELD": 0.0,  # Will be computed in preprocessing
            "TOTAL_YIELD": 0.0,
            "AMBIENT_TEMPERATURE": round(ambient_temp, 2),
            "MODULE_TEMPERATURE": round(module_temp, 2),
            "IRRADIATION": round(irradiation, 4),
            "VILLAGE_DEMAND_KW": round(demand_kw, 3),  # Extra column we add
        })
    
    df = pd.DataFrame(records)
    
    # Compute cumulative daily yield (kWh) — 15 min intervals = /4 for hourly
    df["DAILY_YIELD"] = df.groupby(df["DATE_TIME"].dt.date)["AC_POWER"].cumsum() / 4.0
    df["TOTAL_YIELD"] = df["AC_POWER"].cumsum() / 4.0
    
    print(f"[Synthetic] [OK] Generated {len(df)} rows across {n_days} days")
    return df


def generate_synthetic_weather_data(generation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates weather sensor data to match generation data timestamps.
    Mirrors Plant_1_Weather_Sensor_Data.csv structure.
    """
    weather_df = generation_df[["DATE_TIME", "AMBIENT_TEMPERATURE", 
                                 "MODULE_TEMPERATURE", "IRRADIATION"]].copy()
    weather_df["PLANT_ID"] = 4135001
    weather_df["SOURCE_KEY"] = "HmiyD2TTLFNqkNe"
    return weather_df


# ─────────────────────────────────────────────────────────
# Main: Download or Generate
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Try Kaggle first; fall back to synthetic data
    kaggle_success = download_kaggle_dataset()
    
    if not kaggle_success:
        # Generate synthetic data with same structure as real Kaggle dataset
        gen_df = generate_synthetic_solar_data(n_days=34)
        weather_df = generate_synthetic_weather_data(gen_df)
        
        # Save with same filenames as real Kaggle dataset
        gen_path = os.path.join(DATA_DIR, "Plant_1_Generation_Data.csv")
        weather_path = os.path.join(DATA_DIR, "Plant_1_Weather_Sensor_Data.csv")
        
        gen_df.to_csv(gen_path, index=False)
        weather_df.to_csv(weather_path, index=False)
        
        print(f"[Synthetic] Saved generation data: {gen_path}")
        print(f"[Synthetic] Saved weather data: {weather_path}")

    print("\n[OK] Data acquisition complete. Run: python data/preprocess.py")