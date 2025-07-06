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

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Gianni16/fuzzy-trading-system.git
cd fuzzy-trading-system