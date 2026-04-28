"""
data/preprocess.py
==================
Cleans and merges the real Kaggle solar dataset for use in the RL environment.

Input:  data/raw/Plant_1_Generation_Data.csv
        data/raw/Plant_1_Weather_Sensor_Data.csv
Output: data/processed_solar_data.csv

GitHub Copilot: This creates the training dataset for the microgrid environment.
Run with: python data/preprocess.py
"""

import os
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    PROCESSED_DATA_PATH,
    GENERATION_FILE,
    WEATHER_FILE,
    TIMESTEP_MINUTES,
    STEPS_PER_EPISODE,
)


def _parse_datetime(df: pd.DataFrame, col: str = "DATE_TIME") -> pd.DataFrame:
    # Generic parser (caller can pass strings that are day-first or iso)
    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
    return df


def load_and_prepare() -> pd.DataFrame:
    gen_path = os.path.join(DATA_DIR, GENERATION_FILE)
    weather_path = os.path.join(DATA_DIR, WEATHER_FILE)

    if not os.path.exists(gen_path) or not os.path.exists(weather_path):
        raise FileNotFoundError(
            f"Missing raw files. Expected: {gen_path} and {weather_path}"
        )

    print(f"[Preprocess] Loading generation: {gen_path}")
    gen = pd.read_csv(gen_path)
    # generation file uses day-first format like '15-05-2020 00:00'
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True, errors="coerce")

    print(f"[Preprocess] Loading weather: {weather_path}")
    weather = pd.read_csv(weather_path)
    # weather file uses ISO format 'YYYY-MM-DD HH:MM:SS'
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], dayfirst=False, errors="coerce")

    # Check for parsing failures
    if gen["DATE_TIME"].isna().any():
        nbad = gen["DATE_TIME"].isna().sum()
        print(f"[Preprocess] Warning: {nbad} generation rows failed to parse as datetime and will be dropped")
        gen = gen.dropna(subset=["DATE_TIME"]).reset_index(drop=True)
    if weather["DATE_TIME"].isna().any():
        nbad = weather["DATE_TIME"].isna().sum()
        print(f"[Preprocess] Warning: {nbad} weather rows failed to parse as datetime and will be dropped")
        weather = weather.dropna(subset=["DATE_TIME"]).reset_index(drop=True)

    # Aggregate AC_POWER across inverters for each timestamp
    gen_agg = gen.groupby("DATE_TIME", as_index=False).agg({
        "AC_POWER": "sum",
        "DC_POWER": "sum",
        "DAILY_YIELD": "sum",
    })

    # Take first weather row per timestamp
    weather_agg = weather.groupby("DATE_TIME", as_index=False).first()[
        ["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    ]

    # Merge; keep union of timestamps then fill/resample per-day to ensure full episodes
    merged = pd.merge(gen_agg, weather_agg, on="DATE_TIME", how="outer")
    merged = merged.sort_values("DATE_TIME").reset_index(drop=True)

    # Resample per-day to fixed 15-min grid (STEPS_PER_EPISODE per day)
    all_days = merged["DATE_TIME"].dt.date.unique()
    print(f"[Preprocess] Found {len(all_days)} unique calendar days in raw timestamps")
    rows = []
    freq = f"{TIMESTEP_MINUTES}min"

    for d in all_days:
        day_start = pd.Timestamp(d)
        full_idx = pd.date_range(start=day_start, periods=STEPS_PER_EPISODE, freq=freq)

        day_df = merged[(merged["DATE_TIME"] >= full_idx[0]) & (merged["DATE_TIME"] < full_idx[-1] + pd.Timedelta(minutes=TIMESTEP_MINUTES))]
        if day_df.empty:
            # nothing for this day — create empty scaffold
            scaffold = pd.DataFrame({"DATE_TIME": full_idx})
            scaffold["AC_POWER"] = 0.0
            scaffold["IRRADIATION"] = 0.0
            scaffold["AMBIENT_TEMPERATURE"] = np.nan
            scaffold["MODULE_TEMPERATURE"] = np.nan
            rows.append(scaffold)
            continue

        day_df = day_df.set_index("DATE_TIME").reindex(full_idx)

        # Fill generation: assume missing generation at night = 0, otherwise interpolate
        if "AC_POWER" in day_df:
            day_df["AC_POWER"] = day_df["AC_POWER"].interpolate(limit_direction="both").fillna(0.0)
        else:
            day_df["AC_POWER"] = 0.0

        # Fill irradiation: interpolate then clip to >=0
        if "IRRADIATION" in day_df:
            day_df["IRRADIATION"] = day_df["IRRADIATION"].interpolate(limit_direction="both").fillna(0.0)
            day_df["IRRADIATION"] = day_df["IRRADIATION"].clip(lower=0.0)
        else:
            day_df["IRRADIATION"] = 0.0

        # Temperatures: forward-fill then back-fill
        for col in ("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"):
            if col in day_df:
                day_df[col] = day_df[col].ffill().bfill()
            else:
                day_df[col] = np.nan

        day_df = day_df.reset_index().rename(columns={"index": "DATE_TIME"})
        rows.append(day_df)

    df = pd.concat(rows, ignore_index=True)

    # Ensure columns exist
    for c in ["AC_POWER", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]:
        if c not in df.columns:
            df[c] = 0.0

    return df


def add_demand_features(df: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    # Build demand from time-of-day patterns with small stochasticity
    hours = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60.0
    # Tuned baseline and peaks to match observed dataset mean (~16 kW)
    base = 10.0
    morning_peak = 10.0 * np.exp(-((hours - 8.0) ** 2) / 2.0)
    evening_peak = 14.0 * np.exp(-((hours - 19.0) ** 2) / 2.5)

    dow = df["DATE_TIME"].dt.dayofweek
    weekend = (dow >= 5).astype(float)
    weekend_boost = weekend * 2.0

    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 1.2, len(df))

    df["VILLAGE_DEMAND_KW"] = (
        base + morning_peak + evening_peak + weekend_boost + noise
    ).clip(lower=8.0)

    return df


def compute_energy_balance(df: pd.DataFrame) -> pd.DataFrame:
    # Source `AC_POWER` in raw files is in Watts; convert to kW for consistency
    df["SOLAR_POWER_KW"] = (df["AC_POWER"].clip(lower=0.0) / 1000.0)
    df["NET_BALANCE_KW"] = df["SOLAR_POWER_KW"] - df["VILLAGE_DEMAND_KW"]
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek
    df["IS_DAYTIME"] = (df["IRRADIATION"] > 0.05).astype(int)
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use robust denominators to avoid huge outliers affecting norms
    eps = 1e-6
    solar_denom = max(1.0, df["SOLAR_POWER_KW"].quantile(0.99)) + eps
    demand_denom = max(1.0, df["VILLAGE_DEMAND_KW"].quantile(0.99)) + eps
    irr_denom = max(1.0, df["IRRADIATION"].quantile(0.99)) + eps

    df["SOLAR_NORM"] = (df["SOLAR_POWER_KW"] / solar_denom).clip(0, 1)
    df["DEMAND_NORM"] = (df["VILLAGE_DEMAND_KW"] / demand_denom).clip(0, 1)
    df["IRRAD_NORM"] = (df["IRRADIATION"] / irr_denom).clip(0, 1)
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["HOUR"] / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["HOUR"] / 24)
    return df


def create_episode_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    df["DATE"] = df["DATE_TIME"].dt.date
    # Count rows per calendar date and keep only full days
    counts_by_date = df.groupby("DATE").size()
    valid_dates = counts_by_date[counts_by_date == STEPS_PER_EPISODE].index
    dropped = len(counts_by_date) - len(valid_dates)
    if dropped > 0:
        print(f"[Preprocess] Dropping {dropped} incomplete day(s) that do not have {STEPS_PER_EPISODE} steps")

    df = df[df["DATE"].isin(valid_dates)].reset_index(drop=True)

    # Re-map episodes to contiguous integers and compute STEP
    sorted_dates = sorted(valid_dates)
    date_to_ep = {d: i for i, d in enumerate(sorted_dates)}
    df["EPISODE"] = df["DATE"].map(date_to_ep)
    df["STEP"] = df.groupby("EPISODE").cumcount()

    # Sanity check
    if not df.empty:
        assert df.groupby("EPISODE").size().nunique() == 1, "Episodes have inconsistent lengths"
    return df


def print_dataset_summary(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("PROCESSED DATASET SUMMARY")
    print("=" * 55)
    print(f"Total rows:         {len(df):,}")
    print(f"Total episodes:     {df['EPISODE'].nunique()} days")
    print(f"Steps per episode:  {df.groupby('EPISODE').size().mode().iloc[0] if not df.empty else 0}")
    print(f"\nSolar Power:  min={df['SOLAR_POWER_KW'].min():.1f}  "
          f"max={df['SOLAR_POWER_KW'].max():.1f}  "
          f"mean={df['SOLAR_POWER_KW'].mean():.1f} kW")
    print(f"Village Demand: min={df['VILLAGE_DEMAND_KW'].min():.1f}  "
          f"max={df['VILLAGE_DEMAND_KW'].max():.1f}  "
          f"mean={df['VILLAGE_DEMAND_KW'].mean():.1f} kW")
    print(f"Irradiation:  min={df['IRRADIATION'].min():.3f}  "
          f"max={df['IRRADIATION'].max():.3f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    df = load_and_prepare()
    df = add_demand_features(df)
    df = compute_energy_balance(df)
    df = normalize_features(df)
    df = create_episode_index(df)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print_dataset_summary(df)
    print(f"[Preprocess] [OK] Saved to {PROCESSED_DATA_PATH}")
    print("[Preprocess] Next step: python train.py")