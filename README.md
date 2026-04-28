# Microgrid Balancer: AT-DRAC-EBD RL Agent

Autonomous Microgrid Balancer using **PPO RL** with **Adversarial Training** for resilience under High-Impact Low-Probability (HILP) disturbances.

![Status](https://img.shields.io/badge/status-active-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

This project implements an RL agent for autonomous control of rural village microgrids. The agent optimizes:
- **Renewable utilization** (solar power)
- **Battery management** (charge/discharge strategy)
- **Diesel usage** (cost & emissions)
- **Load shedding** (maintain critical supplies)
- **Resilience** (under adversarial HILP events)

### Key Features

✓ **Continuous Control**: 4D action space (battery, diesel, load shedding, curtailment)  
✓ **Real Data**: Kaggle solar dataset (India, 2020)  
✓ **Adversarial Training**: Energy-bounded HILP disturbances  
✓ **Baselines**: Rule-based + AT-DRAC comparison  
✓ **Evaluation**: Normal & adversarial operation modes  
✓ **Dashboard**: Streamlit visualization  

---

## Project Structure

```
microgrid_balancer/
├── config.py                  # All hyperparameters (1 file!)
├── train.py                   # PPO training script
├── evaluate.py                # Agent comparison & metrics
├── requirements.txt           # Dependencies
├── env/
│   └── microgrid_env.py      # Gymnasium environment
├── agents/
│   └── baselines.py          # Rule-based & AT-DRAC baselines
├── data/
│   ├── download_dataset.py   # Kaggle data downloader
│   └── preprocess.py         # Dataset preprocessing
├── models/                    # Trained model checkpoints (auto-created)
├── results/                   # Evaluation results (auto-created)
└── dashboard/
    └── app.py                # Streamlit dashboard
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd microgrid_balancer
pip install -r requirements.txt
```

### 2. Download & Preprocess Data (Optional)

```bash
# Use Kaggle dataset (requires kaggle API setup)
python data/download_dataset.py

# Preprocess
python data/preprocess.py
```

If Kaggle API fails, the environment auto-generates synthetic data.

### 3. Train the RL Agent

```bash
# Standard training with adversarial enabled
python train.py

# Without adversarial training
python train.py --no-adversarial

# Custom timesteps (default: 500k)
python train.py --timesteps 1000000
```

**Training output:**
- `models/ppo_final.zip` - Trained model
- `results/training_metrics.csv` - Episode metrics
- `results/tensorboard_logs/` - TensorBoard logs

### 4. Evaluate & Compare Agents

```bash
python evaluate.py
```

**Evaluation output (10 episodes each):**
- `results/evaluation_normal.csv` - Normal operation
- `results/evaluation_adversarial.csv` - With HILP disturbances
- `results/episodes_*.csv` - Detailed per-episode data

### 5. View Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

---

## Configuration

All hyperparameters in **config.py**:

```python
# Physical Parameters
BATTERY_CAPACITY_KWH = 100.0
BATTERY_MAX_SOC = 0.95
BATTERY_MIN_SOC = 0.10
DIESEL_CAPACITY_KW = 50.0

# Reward Weights
BLACKOUT_PENALTY = -100.0      # Highest priority
DIESEL_PENALTY = -5.0
RENEWABLE_REWARD = +2.0
BATTERY_WEAR_PENALTY = -0.1

# PPO Hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    ...
}

# Adversarial Training (AT-DRAC-EBD)
ADVERSARY_ENABLED = True
ADVERSARY_BUDGET = 50.0         # Energy budget for disturbances
HILP_SCENARIOS = {
    "load_surge": {...},        # 2.5x demand spike
    "der_tripping": {...},      # 80% solar dropout
    "frequency_attack": {...},  # Oscillatory perturbation
    "cyber_noise": {...},       # Sensor bias attack
}
```

---

## Environment Details

### Observation Space (8D)
```
[solar_generation, demand, battery_soc, diesel_on,
 hour_sin, hour_cos, irradiation, net_balance]
```

### Action Space (4D Continuous)
```
[battery_action, diesel_action, load_shed, curtail]
  [-1: discharge] [-1: off]  [0: no-shed] [0: no-curtail]
  [+1: charge]    [+1: on]   [1: shed-all] [1: curtail-all]
```

### Reward Function
```
R = -100*(blackout_kw) - 5*(diesel_kw) + 2*(renewable_util) - 0.1*|battery_action|
```

---

## Results & Metrics

### Key Performance Indicators

| Metric | Description |
|--------|-------------|
| **CSR** | Critical Service Resilience (% of critical demand met) |
| **Blackouts** | Total unserved energy (kWh) |
| **Diesel Usage** | Backup generator consumption (kWh) |
| **Renewable** | Solar energy utilized (kWh) |
| **Reward** | Episode cumulative reward |

### Expected Performance

After training on 500k timesteps:

| Agent | Normal CSR | Adv. CSR | Diesel (kWh) |
|-------|-----------|---------|--------------|
| PPO RL | 0.98+ | 0.92+ | <10 |
| Rule-Based | 0.95 | 0.85 | ~15 |
| AT-DRAC | 0.96 | 0.88 | ~12 |

---

## Advanced Usage

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=results/tensorboard_logs
```

### Load & Use Trained Model

```python
from stable_baselines3 import PPO
from env.microgrid_env import MicrogridEnv

# Load model
model = PPO.load("models/ppo_final")

# Use in environment
env = MicrogridEnv()
obs, _ = env.reset()

for _ in range(96):  # 24 hours
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.close()
```

### Custom Training Configuration

Edit `config.py`:

```python
TOTAL_TIMESTEPS = 1_000_000  # Increase training length
PPO_CONFIG["learning_rate"] = 1e-3  # Adjust learning rate
ADVERSARY_BUDGET = 100.0  # Harder adversary
```

Then:
```bash
python train.py
```

---

## Troubleshooting

### Issue: "Dataset not found"
**Solution**: Auto-generates synthetic data. For real data:
```bash
# Set up Kaggle API (https://github.com/Kaggle/kaggle-api)
python data/download_dataset.py
python data/preprocess.py
```

### Issue: "PPO model not found" during evaluation
**Solution**: Train first:
```bash
python train.py --timesteps 100000  # Quick test
python evaluate.py
```

### Issue: Slow training
**Solution**: Reduce hyperparameters in `config.py`:
```python
TOTAL_TIMESTEPS = 100_000  # From 500k
PPO_CONFIG["n_steps"] = 1024  # From 2048
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `config.py`:
```python
PPO_CONFIG["batch_size"] = 32  # From 64
```

---

## Research Paper Reference

This implementation is based on:
- **AT-DRAC-EBD**: Adversarial Training for Demand-Response and Battery Control
- **Microgrid Control**: Energy-bounded adversarial learning
- **HILP Events**: High-Impact Low-Probability disturbance modeling

---

## Dataset Info

**Kaggle Dataset**: [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- **Location**: India (Plant 1 & 2)
- **Period**: Jan 2020 - Dec 2020
- **Frequency**: 15-minute intervals
- **Features**: Solar generation, weather sensors (irradiation, temperature)

---

## License

MIT License - See LICENSE file for details

---

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

## Citation

If you use this code in research, please cite:

```bibtex
@article{microgrid_balancer_2024,
  title={Autonomous Microgrid Balancing with Adversarial Training},
  author={Your Name},
  year={2024}
}
```

---

## Contact & Support

- **Issues**: GitHub issues
- **Questions**: Open a discussion
- **Email**: your-email@example.com

Happy training! 🚀
