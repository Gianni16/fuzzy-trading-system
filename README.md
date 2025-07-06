# Fuzzy Trading System

This project implements an **algorithmic trading system** based on **fuzzy logic** and technical indicators (RSI, MACD, Stochastic). The fuzzy model is optimized using a **genetic algorithm (GA)** to maximize the performance of a long-short trading strategy.

## 🎯 Objectives

- Compute technical indicators on weekly time series (e.g., Nasdaq, S&P500)
- Define fuzzy rules to generate buy/sell signals
- Optimize fuzzy membership parameters using a genetic algorithm
- Visualize the equity curve and compare it with the actual market price

## ⚙️ Technologies & Libraries

- `pandas`, `numpy`, `matplotlib`
- `pandas_ta`: technical indicators
- `scikit-fuzzy`, `fuzzy-expert`: fuzzy logic engine
- `pygad`: genetic algorithm for optimization

## Configuration

The main hyperparameters you can customize in `main.py` are:

- `num_generations`: Number of generations for the genetic algorithm optimization. Increasing this value may improve the model accuracy but will increase the training time. Default is 2.
- `sol_per_pop`: Number of solutions per population in the genetic algorithm.
- `list_series`: List of stock/index symbols to analyze (e.g. Nasdaq)
- `list_indicators`: List of technical indicators to use (`'RSI'`, `'MACD'`, `'STO'`).

Feel free to modify these parameters to suit your needs and experiment with different setups.

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Gianni16/fuzzy-trading-system.git
cd fuzzy-trading-system