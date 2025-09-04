# TheVarenneBot — ML‑driven BTCEUR trading bot (Binance)

> End‑to‑end research → training → simulation → live trading on the BTCEUR pair using the Binance API.

[![Exchange](https://img.shields.io/badge/Exchange-Binance-yellow)](#) [![Pair](https://img.shields.io/badge/Pair-BTCEUR-blue)](#) [![Interval](https://img.shields.io/badge/Candles-5m-lightgrey)](#)

## Contents

* [Overview](#overview)
* [August 2025 live results](#august-2025-live-results)
* [How it works](#how-it-works)
* [Install & setup](#install--setup)
* [Reproduce the pipeline](#reproduce-the-pipeline)
* [Run live](#run-live)
* [Risk, assumptions & limitations](#risk-assumptions--limitations)
* [Project structure](#project-structure)
* [Roadmap](#roadmap)

## Overview

**TheVarenneBot** is a full research–to–production pipeline for short‑horizon trading on **BTCEUR**:

* **Data extraction** from Binance (5‑minute klines). Robust retry/backoff and rate‑limit friendly pagination.
* **Dataset creation**: rolling windows of features/labels suitable for deep learning.
* **Modeling**: two supervised models learn to forecast the next‑window **high** and **low** (normalized), which the execution engine translates into actionable orders.
* **Simulation** on a held‑out evaluation set.
* **Live trading** via the Binance API.

---

## August 2025 live results

Period: **2025‑08‑01 → 2025‑08‑31** on BTCEUR, compared to a buy‑and‑hold benchmark.

**Headline:** BTC fell in August; the bot finished **positive** and with materially lower drawdowns.

| Metric                          |       TheVarenneBot | Buy & Hold (BTCEUR) |
| ------------------------------- | ------------------: | ------------------: |
| Total return (Aug 1→31)         |          **+2.04%** |          **−5.37%** |
| Outperformance (Aug 1→31)       |        **+7.41 pp** |                   — |
| Return vs 31 Jul close (Aug 31) |          **+2.04%** |          **−8.65%** |
| Max drawdown                    |          **−4.72%** |         **−11.62%** |
| Sharpe (annualized, daily)      |            **1.59** |           **−3.26** |
| Sortino (annualized, daily)     |            **2.60** |           **−4.65** |
| Hit rate (days > 0)             |          **51.61%** |          **41.94%** |
| Best / worst day                | **+1.88% / −1.89%** | **+2.13% / −4.13%** |
| Return correlation vs BTC       |            **0.62** |                   — |
| Beta vs BTC (daily)             |            **0.32** |                   — |

**Interpretation.** The bot delivered **positive absolute return in a declining month** and behaved defensively (β≈0.32). Its shallower drawdown and better downside‑risk profile (higher Sortino) suggest the model’s signals avoided a chunk of the mid/late‑month selloff and limited exposure during spikes in realized volatility.

> 📈 *Equity curves (normalized to 1.0 on 2025‑07‑31):*
>
> `![August 2025: TheVarenneBot vs Buy & Hold]([docs/grafico.png](https://github.com/filippogalavotti/TheVarenneBot/blob/main/docs/grafico.png))`

---

## How it works

**Pipeline at a glance**

1. **Data extraction** (`csv_generator.py`) — pulls 5‑minute BTCEUR klines across a date range, with retries and incremental windows to respect API limits. Saves a compact `dataset.csv` with `Open Time`, `Open`, `High`, `Low`.
2. **Data QA** (`errors.py`) — verifies there are **no missing 5‑minute intervals**; prints any gaps.
3. **Dataset building** (`dataset_generator.py`) — converts `dataset.csv` into rolling **features**/**labels**:

   * Feature window: **512 bars** (\~42.7 hours).
   * Label window: **256 bars** (\~21.3 hours).
   * Two targets: next‑window **max High** and **min Low**, each **z‑scored** using the feature window stats.
   * Output: compressed `.npz` files for training/validation and a raw JSON for evaluation.
4. **Modeling** (`HIGHModel.py`, `LOWModel.py`) — train two regressors/classifiers that predict the normalized next‑window **high**/**low**.
5. **Simulation** (`simulator.py`) — replays the evaluation set to estimate performance, slippage/fees assumptions, and risk.
6. **Live trading** (`varenne.py`) — polls for fresh candles, creates signals from model outputs, and routes orders through the Binance API on **BTCEUR**.

> **Note:** the modeling/execution code is modular—swap architectures or modify execution logic without touching the upstream data pipeline.

---

## Install & setup

**Requirements**

* Python ≥ 3.10
* Recommended: create a virtualenv (`python -m venv .venv && source .venv/bin/activate`)

Install dependencies (adjust to your stack):

```bash
pip install python-binance pandas numpy requests scikit-learn torch
```

**Credentials & environment**
Create a `.env` (or export as environment variables) and **never hard‑code keys**:

```bash
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
SYMBOL=BTCEUR
INTERVAL=5m
```

---

## Reproduce the pipeline

1. **Pull historical data**

   ```bash
   python csv_generator.py
   # produces dataset.csv with 5m candles and columns: Open Time, Open, High, Low
   ```
2. **Sanity‑check for gaps**

   ```bash
   python errors.py
   # prints any missing 5m timestamps
   ```
3. **Build ML datasets**

   ```bash
   python dataset_generator.py
   # saves high/low .npz files (train/test) and an evaluation JSON
   ```
4. **Train models**

   ```bash
   python HIGHModel.py
   python LOWModel.py
   ```
5. **Simulate**

   ```bash
   python simulator.py
   ```

---

## Run live

Make sure your models’ weights are saved and discoverable by `varenne.py`.

```bash
python varenne.py
```

Operational tips:

* Run behind a process manager (tmux/screen/systemd) and log to file.
* Monitor balances and open orders; enforce a kill‑switch on API failures.
* Keep clock synced (NTP) to avoid timestamp/recvWindow errors.

---

## Risk, assumptions & limitations

* **Fees/slippage.** Simulations should include realistic taker/maker fees and a slippage model; real fills vary by liquidity/time of day.
* **Model drift.** Features are stationary only locally; periodically re‑train and re‑calibrate.
* **Regime shifts.** Crypto microstructure changes across volatility/liquidity regimes; validate out‑of‑sample and stress test.
* **Small sample caution.** The August 2025 live window is one month; statistics are noisy. Annualized ratios are reported for comparability, not precision.
* **Key management.** Use read‑only keys for backtests and minimal permissions for live trading.

---

## Project structure

```
.
├── csv_generator.py        # Binance 5m klines → dataset.csv
├── errors.py               # Detect missing 5m intervals
├── dataset_generator.py    # Rolling features/labels; save .npz & eval JSON
├── HIGHModel.py            # Model for next-window High (normalized)
├── LOWModel.py             # Model for next-window Low (normalized)
├── simulator.py            # Evaluation harness & PnL simulation
├── varenne.py              # Live bot using Binance API on BTCEUR
├── README.md               # ← You are here
└── docs/
    └── aug-2025-equity.png # Equity curves image
```

---

## Roadmap

* Add **unit tests** for data integrity and feature construction.
* Export a **config file** for hyperparameters (window sizes, symbol, fees).
* Add **Prometheus/Grafana** metrics and alerting.
* Support **multiple symbols** and automatic **daily re‑training**.
* Package as a **Docker image** with a one‑liner deploy.
