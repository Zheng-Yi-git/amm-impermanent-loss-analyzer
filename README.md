# Impermanent Loss Analysis & Data Pipeline

This project implements a comprehensive pipeline for analyzing Impermanent Loss (IL) in AMMs using real-world data. It combines Deribit option chain data with Uniswap V3 pool data to calculate liquidity-weighted implied volatility surfaces and simulate IL under risk-neutral pricing.

## 🎯 Overview

The core analysis is based on the theoretical integral form of impermanent loss:

```math
IL(P_1|P_0, \ell) = \int_{0}^{P_0} Put(K) \cdot \frac{\ell(K)}{2K^{3/2}} dK + \int_{P_0}^{\infty} Call(K) \cdot \frac{\ell(K)}{2K^{3/2}} dK
```

This repository provides tools to:

1.  **Fetch Data**: Efficiently retrieve historical option chains (Deribit) and pool ticks (Uniswap V3).
2.  **Reconstruct Pools**: Rebuild Uniswap V3 liquidity distributions from on-chain data.
3.  **Analyze Risk**: Calculate IL and build Implied Volatility (IV) surfaces.
4.  **Visualize**: Generate 3D interactive plots of IV surfaces and term structures.

## 📁 Directory Structure

### Core Analysis

- `il_core.py`: **Core Engine**. Implements the IL integral calculation and `OptionChainProcessor`.
- `compute_iv_surface.py`: **Main Analysis**. Computes liquidity-weighted IV surfaces and generates visualizations.

### Data Pipeline

- `unified_data_fetcher.py`: **Orchestrator**. Unified controller that launches both data-fetch flows.
- `parallel_deribit_data_fetcher.py`: **Options Fetcher**. Multi-threaded Deribit API client for high-performance data retrieval.
- `uniswap-v3-pool-reconstructor/`: **Pool Fetcher**. Node.js submodule for reconstructing Uniswap V3 pools.

### Output & Support

- `artifacts/`: **Results**. Contains generated CSVs (`artifacts/csv/`) and interactive plots (`artifacts/plot/`).
- `deribit_data_test/`: **Raw Data**. Downloaded Deribit option chains.
- `pool_data_test/`: **Raw Data**. Reconstructed Uniswap V3 pool ticks.
- `requirements.txt`: Python dependencies.

## 🚀 Quick Start

### 1. Environment Setup

**Prerequisites**: Python 3.10+, Node.js >= 16.

```bash
# 1. Clone repository with submodule
git clone --recursive https://github.com/YOUR_USERNAME/amm-impermanent-loss-analyzer.git
cd amm-impermanent-loss-analyzer

# If you cloned without --recursive, initialize the submodule manually:
# git submodule update --init --recursive

# 2. Python Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Node.js Setup (for Pool Reconstructor)
cd uniswap-v3-pool-reconstructor
npm install
# IMPORTANT: Edit src/config.ts to add your Infura/Alchemy API Key
npm run build
cd ..
```

### 2. Run the Unified Pipeline

The `unified_data_fetcher.py` script orchestrates the entire data collection process:

```bash
python3 unified_data_fetcher.py
```

**What it does:**

1.  Fetches option chain data for specified symbols (default: BTC, ETH).
2.  Invokes the Node.js submodule to fetch Uniswap V3 pool ticks.
3.  Saves aligned data to `deribit_data_test/` and `pool_data_test/`.

### 3. Run Analysis & Visualization

Calculate Liquidity-Weighted IV Surface and generate plots:

```bash
python3 compute_iv_surface.py
```

**Output:**

- **CSVs**: `artifacts/csv/{COIN}_liquidity_weighted_iv.csv`
- **Plots**:
  - `artifacts/plot/{COIN}_IV_surface_3D.html` (Interactive 3D Surface)
  - `artifacts/plot/{COIN}_IV_term_structure.html` (Term Structure Evolution)

## 📊 Key Results

Based on the analysis of Deribit option chain data:

| Currency | Total IL | Put % | Call % | Risk Level |
| -------- | -------- | ----- | ------ | ---------- |
| BTC_USDC | 4.365    | 58.5% | 41.5%  | High       |
| ETH_USDC | 1.341    | 41.4% | 58.6%  | Medium     |

## 🔧 Configuration & Customization

- **Data Fetching**: Edit `unified_data_fetcher.py` to change symbols (`symbols = [...]`) or expiry dates.
- **Pool Config**: Edit `uniswap-v3-pool-reconstructor/src/config.ts` to change target pools or RPC endpoints.
- **Analysis**: Edit `compute_iv_surface.py` to adjust integration parameters or plotting styles.

## 🛠️ Troubleshooting

- **Node executable not found**: Ensure `node` is in your system PATH, or update the command in `unified_data_fetcher.py`.
- **RPC Errors**: Ensure your Infura/Alchemy key in `uniswap-v3-pool-reconstructor/src/config.ts` is valid and has active limits.
- **Missing Data**: Ensure `parallel_deribit_data_fetcher.py` completed successfully.

## 📚 Theoretical Background

The implementation is based on the theoretical framework for impermanent loss in automated market makers (AMMs). The integral form captures the expected loss under risk-neutral pricing, where:

1.  **Put options** contribute to IL when the price falls below the current level.
2.  **Call options** contribute to IL when the price rises above the current level.
3.  **Liquidity profiles** determine the weight of different price levels.
