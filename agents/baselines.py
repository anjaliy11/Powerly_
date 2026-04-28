"""
agents/baselines.py
===================
Baseline agents for comparison:
  1. RuleBasedBaseline: Greedy load-following with simple heuristics
  2. ATDRACBaseline: Adaptive Time-based Demand-Response and Battery Control

These agents return 4-dimensional continuous actions (no RL training needed):
  [battery_action, diesel_action, load_shed, curtail]
"""

import numpy as np
from config import (
    NON_CRITICAL_LOAD_FRACTION,
    BATTERY_MAX_DISCHARGE_RATE,
    DIESEL_CAPACITY_KW,
    BATTERY_MIN_SOC,
    BATTERY_DISCHARGE_EFFICIENCY,
    TIMESTEP_MINUTES,
)


class RuleBasedBaseline:
    """
    Simple rule-based baseline: greedy load-following.

    Rules:
    - Match battery discharge/charge to solar-demand mismatch
    - Use diesel only when battery is depleted
    - Shed non-critical loads during scarcity
    """

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Observation vector: [solar_kw, demand_kw, battery_soc, diesel_on,
                             hour_sin, hour_cos, irradiation, net_balance]

        Returns: [battery_action, diesel_action, load_shed, curtail]
        """
        solar_norm = observation[0]
        demand_norm = observation[1]
        battery_soc = observation[2]
        net_balance = observation[7]

        # 1. Battery control: balance solar and demand
        if solar_norm > demand_norm:
            # Solar excess: charge battery
            battery_action = 0.8 if battery_soc < 0.9 else 0.0
        elif battery_soc > 0.2:
            # Solar deficit + battery available: discharge
            battery_action = -0.7
        else:
            # No action
            battery_action = 0.0

        # 2. Diesel control: backup when battery low
        if battery_soc < 0.15:
            diesel_action = 0.9  # Turn on diesel
        elif battery_soc > 0.5:
            diesel_action = 0.0  # Turn off diesel
        else:
            diesel_action = 0.3  # Partial operation

        # 3. Load shedding: shed when critical shortage
        if net_balance < -0.6:
            load_shed = 0.7  # Shed 70% of non-critical load
        elif net_balance < -0.3:
            load_shed = 0.3
        else:
            load_shed = 0.0

        # 4. Curtailment: avoid wasting solar
        curtail = 0.0  # Never curtail solar for baseline

        return np.array([battery_action, diesel_action, load_shed, curtail],
                       dtype=np.float32)


class ATDRACBaseline:
    """
    AT-DRAC-EBD: Adaptive Time-based Demand-Response and Battery Control.

    Enhanced rule-based policy that considers:
    - Time of day (solar availability prediction)
    - Battery state (planning for evening peaks)
    - Demand profile (anticipatory charging)
    """

    def select_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Smarter baseline that anticipates evening demand and solar production.

        Returns: [battery_action, diesel_action, load_shed, curtail]
        """
        solar_norm = observation[0]
        demand_norm = observation[1]
        battery_soc = observation[2]
        hour_sin = observation[4]
        hour_cos = observation[5]
        irradiation = observation[6]
        net_balance = observation[7]

        # Decode time of day
        hour = np.arctan2(hour_sin, hour_cos) / (2 * np.pi) * 24
        if hour < 0:
            hour += 24
        hour = float(np.clip(hour, 0, 24))

        # Simple prioritized strategy to maximize CSR while minimizing diesel:
        # 1) Use battery where possible
        # 2) Shed non-critical loads (doesn't affect CSR)
        # 3) Use diesel only as last resort

        deficit = -net_balance  # positive if demand > supply

        # Battery policy: discharge to cover small/medium deficits if SOC available
        if deficit > 0.05 and battery_soc > 0.15:
            # discharge proportionally to deficit and SOC headroom from min
            discharge_strength = min(1.0, deficit / 1.0)
            battery_action = float(-0.9 * discharge_strength * min(1.0, (battery_soc - 0.15) / 0.8))
        elif deficit < -0.05 and battery_soc < 0.9:
            # charge if surplus and room in battery
            charge_strength = min(1.0, -deficit / 1.0)
            battery_action = float(0.6 * charge_strength * (0.95 - battery_soc))
        else:
            battery_action = 0.0

        # Load shedding: prefer shedding non-critical loads before diesel
        if deficit > 0.8:
            load_shed = 0.9
        elif deficit > 0.4:
            load_shed = 0.6
        elif deficit > 0.2:
            load_shed = 0.3
        else:
            load_shed = 0.0

        # After battery discharge and shedding, estimate residual deficit in normalized space.
        # Compute a realistic estimate of battery support (kW) that accounts for SOC
        # headroom and discharge efficiency, then convert to a fraction of diesel capacity.
        est_batt_support_frac = 0.0
        if battery_action < 0:
            # magnitude of requested discharge (kW)
            requested_discharge_kw = (-battery_action) * BATTERY_MAX_DISCHARGE_RATE

            # fraction of usable energy available in battery above min SOC
            soc_headroom_frac = 0.0
            if battery_soc > BATTERY_MIN_SOC:
                soc_headroom_frac = (battery_soc - BATTERY_MIN_SOC) / max(1e-6, 1.0 - BATTERY_MIN_SOC)

            # expected delivered power (kW) accounting for SOC headroom and efficiency
            delivered_kw = requested_discharge_kw * min(1.0, soc_headroom_frac) * BATTERY_DISCHARGE_EFFICIENCY

            est_batt_support_frac = min(1.0, delivered_kw / max(1e-6, DIESEL_CAPACITY_KW))

        residual_frac = max(0.0, deficit - est_batt_support_frac - load_shed * NON_CRITICAL_LOAD_FRACTION)

        # Diesel policy: allow modest diesel use to avoid blackouts while keeping usage limited.
        MAX_DIESEL_FRACTION = 0.6
        diesel_action = 0.0

        if residual_frac > 0.15:
            diesel_action = min(MAX_DIESEL_FRACTION, residual_frac * 1.2)
            # if battery has good headroom, scale diesel back a bit
            if battery_soc > 0.35:
                diesel_action *= 0.6

        # Conservative fallback: if SOC is low but small residual exists, provide moderate diesel
        if battery_soc < 0.20 and residual_frac > 0.05:
            diesel_action = max(diesel_action, 0.35)

        # Emergency safeguard: stronger diesel when residual is large or battery critically low
        if residual_frac > 0.30 or (battery_soc < 0.12 and deficit > 0.05):
            diesel_action = max(diesel_action, 0.6)

        # Curtailment: apply minimal curtailment if battery full to avoid wasting renewables
        if battery_soc > 0.92 and solar_norm > 0.6:
            curtail = 0.3
        else:
            curtail = 0.0

        return np.array([battery_action, diesel_action, load_shed, curtail],
                       dtype=np.float32)
