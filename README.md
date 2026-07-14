# GBPUSD Intraday ML/RL Trading System

Price action–driven M15/M30 algo using Python ↔ MT5 bridge, LightGBM supervised classifier, and PPO RL agent.

## Architecture

```
MT5 Terminal (Windows)
    ↕  MetaTrader5 Python library
data/pipeline.py        — fetch & store OHLCV bars
features/engineer.py    — price action feature engineering
models/supervised/      — LightGBM directional bias classifier
models/rl/              — PPO agent (Stable Baselines3)
env/trading_env.py      — custom Gym environment
live/executor.py        — live trading loop
backtest/runner.py      — walk-forward backtester
```

## Setup

```bash
pip install -r requirements.txt
```

MT5 terminal must be running on Windows (or Wine/VPS).

## Run Order

1. `python data/pipeline.py --mode history`     # pull historical data
2. `python features/engineer.py`                # build feature matrix
3. `python models/supervised/train.py`          # train LightGBM classifier
4. `python models/rl/train.py`                  # train PPO agent
5. `python backtest/runner.py`                  # walk-forward evaluation
6. `python live/executor.py`                    # live paper/real trading
