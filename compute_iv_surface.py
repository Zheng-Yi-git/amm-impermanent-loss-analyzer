"""
Liquidity-Weighted Implied Volatility Surface Builder

Computes implied volatility term structures from Deribit options and Uniswap V3 pool data.
For each observation timestamp and expiry date, calculates market-observed IL using
liquidity-weighted integration, then inverts Black-Scholes to recover implied volatility.
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from math import exp, log, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import griddata
import plotly.graph_objects as go

from il_core import get_uniswap_data, OptionChainProcessor


BASE_DIR = os.getcwd()
OPTIONS_DIR = os.path.join(BASE_DIR, "deribit_data_test", "options", "pqt")
POOL_DIR = os.path.join(BASE_DIR, "pool_data_test")
TS_PATTERN = re.compile(r"(\d{8}_\d{6})")
MS_IN_YEAR = 365 * 24 * 60 * 60 * 1000


COIN_CONFIG: Dict[str, Dict[str, str]] = {
    "BTC": {
        "currency": "BTC_USDC",
        "pool_dir": os.path.join(POOL_DIR, "BTC"),
        "pool_prefix": "WBTC-USDC_ticks_",
    },
    "ETH": {
        "currency": "ETH_USDC",
        "pool_dir": os.path.join(POOL_DIR, "ETH"),
        "pool_prefix": "WETH-USDC_ticks_",
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class IntegrationSchedule:
    """Discrete integration grid for IL calculation"""
    strikes: np.ndarray
    dq: np.ndarray
    weights: np.ndarray
    liquidity: np.ndarray
    option_types: np.ndarray

    def subset(self, mask: np.ndarray) -> "IntegrationSchedule":
        return IntegrationSchedule(
            strikes=self.strikes[mask],
            dq=self.dq[mask],
            weights=self.weights[mask],
            liquidity=self.liquidity[mask],
            option_types=self.option_types[mask],
        )

    @property
    def weighted_liquidity_sum(self) -> float:
        """Compute weighted liquidity integral: ∫ ℓ(K)/(K^(3/2)) * dK"""
        return float(np.sum(self.weights * self.dq, dtype=np.float64))


def extract_timestamp(name: str) -> Optional[str]:
    match = TS_PATTERN.search(name)
    return match.group(1) if match else None


def enumerate_option_files() -> List[Tuple[str, str]]:
    if not os.path.isdir(OPTIONS_DIR):
        logger.warning("Options directory missing: %s", OPTIONS_DIR)
        return []
    files: List[Tuple[str, str]] = []
    for fname in sorted(os.listdir(OPTIONS_DIR)):
        if not fname.endswith(".parquet"):
            continue
        ts = extract_timestamp(fname)
        if ts:
            files.append((ts, os.path.join(OPTIONS_DIR, fname)))
    logger.info("Discovered %d option files under %s", len(files), OPTIONS_DIR)
    return files


def load_option_frame(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    numeric_cols = ["strike", "best_bid", "best_ask", "mark_price", "mid_price"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mid_price" not in df.columns:
        df["mid_price"] = np.where(
            (df["best_bid"].notna()) & (df["best_ask"].notna()),
            (df["best_bid"] + df["best_ask"]) / 2,
            df["mark_price"],
        )
    return df


def load_pool_snapshot(path: str) -> Tuple[float, pd.DataFrame]:
    metadata, df_ticks = get_uniswap_data(path)
    current_tick = int(metadata.get("Current Tick", "0"))
    current_tick_row = df_ticks[df_ticks["tick_index"] == current_tick]

    if len(current_tick_row) > 0:
        current_price = float(current_tick_row.iloc[0]["user_price"])
    else:
        df_sorted = df_ticks.sort_values("tick_index").reset_index(drop=True)
        insert_pos = df_sorted["tick_index"].searchsorted(current_tick)
        if insert_pos == 0:
            current_price = float(df_sorted.iloc[0]["user_price"])
        elif insert_pos == len(df_sorted):
            current_price = float(df_sorted.iloc[-1]["user_price"])
        else:
            left_tick = df_sorted.iloc[insert_pos - 1]
            right_tick = df_sorted.iloc[insert_pos]
            current_price = float(
                (left_tick["user_price"] + right_tick["user_price"]) / 2
            )

    return current_price, df_ticks


def compute_time_to_maturity(df: pd.DataFrame) -> Optional[float]:
    if "time_to_expiration" in df.columns:
        tte = pd.to_numeric(df["time_to_expiration"], errors="coerce").dropna()
        tte = tte[tte > 0]
        if not tte.empty:
            return float(np.median(tte))

    if {"expiration_timestamp", "orderbook_timestamp"}.issubset(df.columns):
        exp_ms = pd.to_numeric(df["expiration_timestamp"], errors="coerce").dropna()
        obs_ms = pd.to_numeric(df["orderbook_timestamp"], errors="coerce").dropna()
        if not exp_ms.empty and not obs_ms.empty:
            delta_ms = float(np.median(exp_ms)) - float(np.median(obs_ms))
            if delta_ms > 0:
                return delta_ms / MS_IN_YEAR

    if "expiry_date" in df.columns and "orderbook_timestamp" in df.columns:
        expiry_val = df["expiry_date"].dropna().iloc[0]
        obs_ms = pd.to_numeric(df["orderbook_timestamp"], errors="coerce").dropna()
        if isinstance(expiry_val, str) and not obs_ms.empty:
            try:
                expiry_dt = datetime.strptime(expiry_val, "%Y-%m-%d")
                obs_dt = datetime.utcfromtimestamp(float(np.median(obs_ms)) / 1000)
                delta = (expiry_dt - obs_dt).total_seconds()
                if delta > 0:
                    return delta / (365.0 * 24 * 3600)
            except ValueError:
                pass

    return None


def prepare_option_slice(df: pd.DataFrame) -> pd.DataFrame:
    required = {"strike", "mid_price", "option_type"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    cleaned = df.copy()
    cleaned["strike"] = pd.to_numeric(cleaned["strike"], errors="coerce")
    cleaned["mid_price"] = pd.to_numeric(cleaned["mid_price"], errors="coerce")
    cleaned = cleaned.dropna(subset=["strike", "mid_price", "option_type"])
    cleaned = cleaned[(cleaned["strike"] > 0) & (cleaned["mid_price"] > 0)]
    cleaned = cleaned.drop_duplicates(subset=["strike", "option_type"], keep="first")
    return cleaned.sort_values("strike").reset_index(drop=True)


def build_integration_schedule(
    df_ticks: pd.DataFrame,
    current_price: float,
    k_lower: float,
    k_upper: float,
) -> Optional[IntegrationSchedule]:
    df_filtered = df_ticks[
        (df_ticks["user_price"] >= k_lower) & (df_ticks["user_price"] <= k_upper)
    ].sort_values("user_price")
    if len(df_filtered) < 2:
        return None

    prices = pd.to_numeric(df_filtered["user_price"], errors="coerce").to_numpy()
    liquidity = pd.to_numeric(df_filtered["cumulative_liquidity"], errors="coerce").to_numpy()
    liquidity = np.nan_to_num(liquidity, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    strikes = (prices[:-1] + prices[1:]) / 2.0
    dq = prices[1:] - prices[:-1]
    liquidity = liquidity[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = liquidity / (2.0 * np.power(strikes, 1.5))
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    option_types = np.where(strikes <= current_price, "P", "C")
    mask = (
        np.isfinite(strikes)
        & np.isfinite(dq)
        & np.isfinite(weights)
        & (strikes > 0)
        & (np.abs(dq) >= 1e-10)
        & (liquidity >= 0)
        & (weights >= 0)
    )

    if not mask.any():
        return None

    return IntegrationSchedule(
        strikes=strikes[mask],
        dq=dq[mask],
        weights=weights[mask],
        liquidity=liquidity[mask],
        option_types=option_types[mask],
    )


def compute_market_unit_il(
    processor: OptionChainProcessor,
    schedule: IntegrationSchedule,
) -> Tuple[Optional[float], IntegrationSchedule]:
    processor.split_by_type()
    if processor.calls is None or processor.puts is None:
        return None, schedule
    if len(processor.calls) < 2 or len(processor.puts) < 2:
        return None, schedule

    strikes = schedule.strikes
    prices = processor.interpolate_prices(strikes, "both")
    call_prices = np.asarray(prices.get("calls", np.full_like(strikes, np.nan)))
    put_prices = np.asarray(prices.get("puts", np.full_like(strikes, np.nan)))
    selected = np.where(schedule.option_types == "P", put_prices, call_prices)

    mask = np.isfinite(selected)
    if not mask.any():
        return None, schedule

    schedule = schedule.subset(mask)
    selected = selected[mask]

    weighted_liq_sum = schedule.weighted_liquidity_sum
    if weighted_liq_sum <= 0:
        return None, schedule

    total_il = float(np.sum(selected * schedule.weights * schedule.dq))
    unit_il = total_il / weighted_liq_sum
    return unit_il, schedule


def bs_price(S: float, K: float, T: float, sigma: float, option_type: str, r: float = 0.0) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type.upper() == "C":
        return float(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))
    return float(K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def compute_bs_unit_il(
    schedule: IntegrationSchedule,
    current_price: float,
    T: float,
    sigma: float,
) -> Optional[float]:
    weighted_liq_sum = schedule.weighted_liquidity_sum
    if weighted_liq_sum <= 0 or not np.isfinite(T) or T <= 0:
        return None
    if not np.isfinite(sigma) or sigma <= 0:
        return None

    model_prices = [
        bs_price(current_price, strike, T, sigma, opt_type)
        for strike, opt_type in zip(schedule.strikes, schedule.option_types)
    ]
    total_il = float(np.sum(np.asarray(model_prices) * schedule.weights * schedule.dq))
    return total_il / weighted_liq_sum


def solve_implied_vol(
    target_unit_il: float,
    schedule: IntegrationSchedule,
    current_price: float,
    time_to_maturity: float,
) -> Optional[float]:
    if not np.isfinite(target_unit_il) or target_unit_il <= 0:
        return None
    weighted_liq_sum = schedule.weighted_liquidity_sum
    if weighted_liq_sum <= 0 or not np.isfinite(time_to_maturity) or time_to_maturity <= 0:
        return None

    def objective(vol: float) -> float:
        model_il = compute_bs_unit_il(schedule, current_price, time_to_maturity, vol)
        if model_il is None or not np.isfinite(model_il):
            return np.inf
        return model_il - target_unit_il

    try:
        return float(brentq(objective, 1e-4, 5.0, maxiter=80))
    except ValueError:
        try:
            return float(brentq(objective, 1e-4, 10.0, maxiter=100))
        except ValueError:
            return None


def locate_pool_file(coin: str, timestamp: str) -> Optional[str]:
    cfg = COIN_CONFIG[coin]
    fname = f"{cfg['pool_prefix']}{timestamp}.csv"
    path = os.path.join(cfg["pool_dir"], fname)
    return path if os.path.exists(path) else None


def process_coin_timestamp(
    coin: str,
    timestamp: str,
    df_options: pd.DataFrame,
    current_price: float,
    df_ticks: pd.DataFrame,
) -> List[Dict[str, object]]:
    cfg = COIN_CONFIG[coin]
    df_coin = df_options[df_options["currency"] == cfg["currency"]].copy()
    if df_coin.empty:
        return []

    if "mid_price" not in df_coin.columns:
        df_coin["mid_price"] = np.where(
            (df_coin["best_bid"].notna()) & (df_coin["best_ask"].notna()),
            (df_coin["best_bid"] + df_coin["best_ask"]) / 2,
            df_coin["mark_price"],
        )

    results: List[Dict[str, object]] = []
    group_col = "expiry_date" if "expiry_date" in df_coin.columns else "expiration_timestamp"

    for expiry_value, df_slice in df_coin.groupby(group_col, dropna=True):
        cleaned = prepare_option_slice(df_slice)
        if cleaned.empty:
            continue

        time_to_maturity = compute_time_to_maturity(df_slice)
        if time_to_maturity is None or time_to_maturity <= 0:
            continue

        strike_min = float(cleaned["strike"].min())
        strike_max = float(cleaned["strike"].max())
        pool_price_min = float(pd.to_numeric(df_ticks["user_price"], errors="coerce").min())
        pool_price_max = float(pd.to_numeric(df_ticks["user_price"], errors="coerce").max())

        k_lower = max(strike_min, pool_price_min)
        k_upper = min(strike_max, pool_price_max)
        if not (k_upper > k_lower > 0):
            continue

        schedule = build_integration_schedule(df_ticks, current_price, k_lower, k_upper)
        if schedule is None:
            continue

        processor = OptionChainProcessor("<dataframe>")
        processor.df = cleaned
        unit_il, filtered_schedule = compute_market_unit_il(processor, schedule)
        if unit_il is None or filtered_schedule.weighted_liquidity_sum <= 0:
            continue

        implied_vol = solve_implied_vol(unit_il, filtered_schedule, current_price, time_to_maturity)
        if implied_vol is None or not np.isfinite(implied_vol):
            continue

        logger.info(
            "%s %s expiry=%s T=%.4f IV=%.4f unit_il=%.4e",
            coin,
            timestamp,
            expiry_value,
            time_to_maturity,
            implied_vol,
            unit_il,
        )
        results.append(
            {
                "coin": coin,
                "timestamp": timestamp,
                "expiry": expiry_value,
                "time_to_maturity": time_to_maturity,
                "unit_il": unit_il,
                "implied_vol": implied_vol,
                "option_count": len(cleaned),
                "integration_points": len(filtered_schedule.strikes),
            }
        )

    return results


def build_liquidity_weighted_iv_surface(coins: List[str]) -> Dict[str, pd.DataFrame]:
    option_files = enumerate_option_files()
    aggregated: Dict[str, List[Dict[str, object]]] = {coin: [] for coin in coins}

    for timestamp, option_path in option_files:
        df_options = load_option_frame(option_path)
        for coin in coins:
            pool_path = locate_pool_file(coin, timestamp)
            if not pool_path:
                continue
            current_price, df_ticks = load_pool_snapshot(pool_path)
            rows = process_coin_timestamp(coin, timestamp, df_options, current_price, df_ticks)
            aggregated[coin].extend(rows)

    return {
        coin: pd.DataFrame(rows).sort_values(["timestamp", "time_to_maturity"]).reset_index(drop=True)
        for coin, rows in aggregated.items()
        if rows
    }


class IVSurfaceVisualizer:
    """Visualization tools for IV surfaces"""

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for plotting"""
        df = df.copy()
        
        # Ensure datetime column exists
        if "datetime" not in df.columns and "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S")
            
        # Calculate hours elapsed
        first_time = df["datetime"].min()
        df["hours_elapsed"] = (df["datetime"] - first_time).dt.total_seconds() / 3600
        
        return df

    @staticmethod
    def create_3d_surface_plot(df: pd.DataFrame, coin: str) -> go.Figure:
        """Create 3D surface plot"""
        df = IVSurfaceVisualizer.preprocess_data(df)
        
        # Grid interpolation
        x = df["time_to_maturity"].values
        y = df["hours_elapsed"].values
        z = df["implied_vol"].values
        
        # Create regular grid
        x_grid = np.linspace(x.min(), x.max(), 100)
        y_grid = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate
        Z = griddata((x, y), z, (X, Y), method="cubic")
        
        # Plot
        fig = go.Figure(data=[
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Viridis",
                colorbar=dict(title="Implied Vol"),
                hovertemplate="T: %{x:.4f}<br>Hours: %{y:.2f}<br>IV: %{z:.4f}<extra></extra>"
            )
        ])
        
        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=2, color="red", opacity=0.6),
            name="Data Points",
            hovertemplate="T: %{x:.4f}<br>Hours: %{y:.2f}<br>IV: %{z:.4f}<extra></extra>"
        ))
        
        # Format axes
        unique_hours = sorted(df["hours_elapsed"].unique())
        # Limit ticks to ~10
        step = max(1, len(unique_hours) // 10)
        y_tickvals = unique_hours[::step]
        y_ticktext = [
            df[df["hours_elapsed"] == h]["datetime"].iloc[0].strftime("%m-%d %H:%M")
            for h in y_tickvals
        ]
        
        fig.update_layout(
            title=f"{coin} Liquidity-Weighted IV Surface",
            scene=dict(
                xaxis=dict(title="Time to Maturity (years)"),
                yaxis=dict(title="Timestamp", tickvals=y_tickvals, ticktext=y_ticktext),
                zaxis=dict(title="Implied Volatility"),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1200, height=800,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig

    @staticmethod
    def create_term_structure_lines(df: pd.DataFrame, coin: str) -> go.Figure:
        """Create term structure evolution plot"""
        df = IVSurfaceVisualizer.preprocess_data(df)
        unique_times = sorted(df["datetime"].unique())
        
        # Sample ~10 lines
        step = max(1, len(unique_times) // 10)
        sample_times = unique_times[::step]
        
        fig = go.Figure()
        n_times = len(sample_times)
        
        for idx, dt in enumerate(sample_times):
            subset = df[df["datetime"] == dt].sort_values("time_to_maturity")
            
            # Color gradient (Blue: Light -> Dark)
            intensity = idx / max(n_times - 1, 1)
            r = int(173 * (1 - intensity))
            g = int(216 * (1 - intensity))
            b = int(230 * (1 - intensity * 0.4))
            color = f"rgb({r},{g},{b})"
            
            fig.add_trace(go.Scatter(
                x=subset["time_to_maturity"],
                y=subset["implied_vol"],
                mode="lines+markers",
                name=dt.strftime("%m-%d %H:%M"),
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                hovertemplate="T: %{x:.4f}<br>IV: %{y:.4f}<extra></extra>"
            ))
            
        fig.update_layout(
            title=f"{coin} IV Term Structure Evolution (Light→Dark = Early→Late)",
            xaxis_title="Time to Maturity (years)",
            yaxis_title="Implied Volatility",
            width=1200, height=600,
            legend=dict(title=dict(text="Timestamp"))
        )
        return fig


def main():
    coins = list(COIN_CONFIG.keys())
    surfaces = build_liquidity_weighted_iv_surface(coins)

    if not surfaces:
        print("No implied volatility surfaces were generated.")
        return

    # Setup output directories
    artifacts_dir = os.path.join(BASE_DIR, "artifacts")
    csv_dir = os.path.join(artifacts_dir, "csv")
    plot_dir = os.path.join(artifacts_dir, "plot")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    for coin, df in surfaces.items():
        print(f"\n=== {coin} Liquidity-Weighted IV Surface ===")
        preview_cols = ["timestamp", "expiry", "time_to_maturity", "implied_vol"]
        available_cols = [col for col in preview_cols if col in df.columns]
        print(df[available_cols].head(20))
        
        # Save CSV
        output_path = os.path.join(csv_dir, f"{coin}_liquidity_weighted_iv.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")

        # Visualization
        try:
            print(f"Generating plots for {coin}...")
            
            # 3D Surface
            fig_3d = IVSurfaceVisualizer.create_3d_surface_plot(df, coin)
            plot_path_3d = os.path.join(plot_dir, f"{coin}_IV_surface_3D.html")
            fig_3d.write_html(plot_path_3d)
            
            # Term Structure
            fig_ts = IVSurfaceVisualizer.create_term_structure_lines(df, coin)
            plot_path_ts = os.path.join(plot_dir, f"{coin}_IV_term_structure.html")
            fig_ts.write_html(plot_path_ts)
            
            print(f"✅ Plots saved to {plot_dir}")
        except Exception as e:
            logger.error(f"Failed to generate plots for {coin}: {e}")


if __name__ == "__main__":
    main()
