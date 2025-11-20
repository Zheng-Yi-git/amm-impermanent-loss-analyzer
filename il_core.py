"""
Empirical Simulation of Impermanent Loss via Option Chain Data

This module implements the theoretical integral form of impermanent loss (IL)
under risk-neutral pricing using Deribit option chain data.

Theoretical Framework:
IL(P1|P0,ℓ) = ∫[0,P0] Put(K=q) * ℓ(q)/(2q^(3/2)) dq + ∫[P0,∞] Call(K=q) * ℓ(q)/(2q^(3/2)) dq

where:
- P0: current spot/forward price
- ℓ(q): liquidity profile (flat, uniform, or custom)
- Put(K), Call(K): risk-neutral option prices with strike K and maturity T
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import trapezoid
from typing import Dict, List, Tuple, Optional, Callable
import warnings

warnings.filterwarnings("ignore")

# Set pandas plotting backend to plotly
pd.options.plotting.backend = "plotly"

# Try to import plotly for advanced plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.io as pio

    print("Plotly available - using interactive charts")
except ImportError:
    print("Plotly not available - using static matplotlib charts")


class OptionChainProcessor:
    """Process and clean option chain data from Deribit CSV files."""

    def __init__(self, csv_path: str):
        """
        Initialize with CSV file path.

        Args:
            csv_path: Path to the Deribit options CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.calls = None
        self.puts = None

    def load_data(self) -> pd.DataFrame:
        """Load and basic clean the option chain data."""
        self.df = pd.read_csv(self.csv_path)

        # Clean data
        self.df["strike"] = pd.to_numeric(self.df["strike"], errors="coerce")
        self.df["best_bid"] = pd.to_numeric(self.df["best_bid"], errors="coerce")
        self.df["best_ask"] = pd.to_numeric(self.df["best_ask"], errors="coerce")
        self.df["mark_price"] = pd.to_numeric(self.df["mark_price"], errors="coerce")

        # Compute mid prices where bid/ask available
        self.df["mid_price"] = np.where(
            (self.df["best_bid"].notna()) & (self.df["best_ask"].notna()),
            (self.df["best_bid"] + self.df["best_ask"]) / 2,
            self.df["mark_price"],
        )

        # Remove rows with invalid strikes or prices
        self.df = self.df.dropna(subset=["strike", "mid_price"])
        self.df = self.df[self.df["strike"] > 0]
        self.df = self.df[self.df["mid_price"] > 0]

        return self.df

    def split_by_type(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into calls and puts."""
        if self.df is None:
            self.load_data()

        self.calls = self.df[self.df["option_type"] == "C"].copy()
        self.puts = self.df[self.df["option_type"] == "P"].copy()

        # Sort by strike for interpolation
        self.calls = self.calls.sort_values("strike").reset_index(drop=True)
        self.puts = self.puts.sort_values("strike").reset_index(drop=True)

        return self.calls, self.puts

    def interpolate_prices(
        self, strikes: np.ndarray, option_type: str = "both"
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate option prices for given strikes with non-negative constraint.

        Args:
            strikes: Array of strike prices to interpolate
            option_type: 'call', 'put', or 'both'

        Returns:
            Dictionary with interpolated prices (all non-negative)
        """
        if self.calls is None or self.puts is None:
            self.split_by_type()

        result = {}

        if option_type in ["call", "both"]:
            if len(self.calls) > 1:
                # Use cubic spline interpolation for smooth curves
                call_interp = interpolate.interp1d(
                    self.calls["strike"],
                    self.calls["mid_price"],
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                call_prices = call_interp(strikes)

                # Apply non-negative constraint
                call_prices = np.maximum(call_prices, 0.0)

                # For extrapolation beyond data range, use nearest boundary value
                min_strike = self.calls["strike"].min()
                max_strike = self.calls["strike"].max()

                # For strikes below minimum, use minimum strike price
                below_min = strikes < min_strike
                if np.any(below_min):
                    min_price = self.calls[self.calls["strike"] == min_strike][
                        "mid_price"
                    ].iloc[0]
                    call_prices[below_min] = min_price

                # For strikes above maximum, use maximum strike price
                above_max = strikes > max_strike
                if np.any(above_max):
                    max_price = self.calls[self.calls["strike"] == max_strike][
                        "mid_price"
                    ].iloc[0]
                    call_prices[above_max] = max_price

                result["calls"] = call_prices
            else:
                result["calls"] = np.full_like(strikes, np.nan)

        if option_type in ["put", "both"]:
            if len(self.puts) > 1:
                put_interp = interpolate.interp1d(
                    self.puts["strike"],
                    self.puts["mid_price"],
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                put_prices = put_interp(strikes)

                # Apply non-negative constraint
                put_prices = np.maximum(put_prices, 0.0)

                # For extrapolation beyond data range, use nearest boundary value
                min_strike = self.puts["strike"].min()
                max_strike = self.puts["strike"].max()

                # For strikes below minimum, use minimum strike price
                below_min = strikes < min_strike
                if np.any(below_min):
                    min_price = self.puts[self.puts["strike"] == min_strike][
                        "mid_price"
                    ].iloc[0]
                    put_prices[below_min] = min_price

                # For strikes above maximum, use maximum strike price
                above_max = strikes > max_strike
                if np.any(above_max):
                    max_price = self.puts[self.puts["strike"] == max_strike][
                        "mid_price"
                    ].iloc[0]
                    put_prices[above_max] = max_price

                result["puts"] = put_prices
            else:
                result["puts"] = np.full_like(strikes, np.nan)

        return result


class LiquidityProfile:
    """Define different liquidity profiles for IL calculation."""

    @staticmethod
    def flat_profile(q: np.ndarray, K_lower: float, K_upper: float) -> np.ndarray:
        """
        Flat liquidity profile: ℓ(q) = 1 for q ∈ [K_lower, K_upper], 0 otherwise.

        Args:
            q: Strike prices
            K_lower: Lower bound
            K_upper: Upper bound

        Returns:
            Liquidity values
        """
        return np.where((q >= K_lower) & (q <= K_upper), 1.0, 0.0)

    @staticmethod
    def uniform_profile(q: np.ndarray) -> np.ndarray:
        """
        Uniform liquidity profile: ℓ(q) = 1 for all q.

        Args:
            q: Strike prices

        Returns:
            Liquidity values (all ones)
        """
        return np.ones_like(q)

    @staticmethod
    def bounded_profile(
        q: np.ndarray, K_lower: float, K_upper: float, decay_factor: float = 0.1
    ) -> np.ndarray:
        """
        Bounded liquidity with decay: ℓ(q) = exp(-decay_factor * |q - center|).

        Args:
            q: Strike prices
            K_lower: Lower bound
            K_upper: Upper bound
            decay_factor: Decay rate

        Returns:
            Liquidity values
        """
        center = (K_lower + K_upper) / 2
        return np.where(
            (q >= K_lower) & (q <= K_upper),
            np.exp(-decay_factor * np.abs(q - center)),
            0.0,
        )


class ImpermanentLossCalculator:
    """Calculate impermanent loss using option chain data."""

    def __init__(self, processor: OptionChainProcessor):
        """
        Initialize calculator with option chain processor.

        Args:
            processor: OptionChainProcessor instance
        """
        self.processor = processor
        self.P0 = None  # Current/forward price
        self.T = None  # Time to maturity

    def set_parameters(self, P0: float, T: float = None):
        """
        Set current price and time to maturity.

        Args:
            P0: Current spot/forward price
            T: Time to maturity in years (optional)
        """
        self.P0 = P0
        self.T = T if T is not None else 0.25  # Default 3 months

    def derive_forward_price(self) -> float:
        """
        Derive forward price using put-call parity.

        Returns:
            Estimated forward price
        """
        if self.processor.calls is None or self.processor.puts is None:
            self.processor.split_by_type()

        # Find common strikes between calls and puts
        common_strikes = set(self.processor.calls["strike"]).intersection(
            set(self.processor.puts["strike"])
        )

        if len(common_strikes) < 3:
            print("Warning: Insufficient common strikes for put-call parity")
            return self.P0

        # Calculate forward using put-call parity: F = K + (C - P
        forwards = []
        for strike in common_strikes:
            call_price = self.processor.calls[self.processor.calls["strike"] == strike][
                "mid_price"
            ].iloc[0]
            put_price = self.processor.puts[self.processor.puts["strike"] == strike][
                "mid_price"
            ].iloc[0]

            forward = strike + call_price - put_price
            forwards.append(forward)

        # Use median to reduce impact of outliers
        return np.median(forwards)

    def calculate_il(
        self,
        liquidity_func: Callable = None,
        K_lower: float = None,
        K_upper: float = None,
        num_points: int = 1000,
    ) -> Dict[str, float]:
        """
        Calculate impermanent loss using numerical integration.

        Args:
            liquidity_func: Custom liquidity function
            K_lower: Lower bound for flat liquidity
            K_upper: Upper bound for flat liquidity
            num_points: Number of integration points

        Returns:
            Dictionary with IL components and total
        """
        if self.P0 is None:
            raise ValueError("Must set P0 using set_parameters()")

        # Get data bounds
        all_strikes = np.concatenate(
            [
                self.processor.calls["strike"].values,
                self.processor.puts["strike"].values,
            ]
        )

        K_min = np.min(all_strikes)
        K_max = np.max(all_strikes)

        # Set default bounds if not provided
        if K_lower is None:
            K_lower = K_min
        if K_upper is None:
            K_upper = K_max

        # Create integration grid
        strikes = np.linspace(K_lower, K_upper, num_points)

        # Interpolate option prices
        option_prices = self.processor.interpolate_prices(strikes, "both")

        # Define liquidity profile
        if liquidity_func is None:
            liquidity_func = lambda q: LiquidityProfile.flat_profile(
                q, K_lower, K_upper
            )

        liquidity = liquidity_func(strikes)

        # Calculate weights: w(K) = ℓ(K) / (2 * K^(3/2))
        weights = liquidity / (2 * strikes ** (3 / 2))

        # Split into put and call regions
        put_mask = strikes <= self.P0
        call_mask = strikes >= self.P0

        # Calculate IL components
        put_contrib = trapezoid(
            option_prices["puts"][put_mask] * weights[put_mask], strikes[put_mask]
        )

        call_contrib = trapezoid(
            option_prices["calls"][call_mask] * weights[call_mask], strikes[call_mask]
        )

        total_il = put_contrib + call_contrib

        return {
            "total_il": total_il,
            "put_contribution": put_contrib,
            "call_contribution": call_contrib,
            "strikes": strikes,
            "weights": weights,
            "liquidity": liquidity,
            "option_prices": option_prices,
        }

    def calculate_il_by_currency(self, currencies: List[str]) -> Dict[str, Dict]:
        """
        Calculate IL for multiple currencies.

        Args:
            currencies: List of currency symbols

        Returns:
            Dictionary with IL results for each currency
        """
        results = {}

        for currency in currencies:
            # Filter data for this currency
            currency_data = self.processor.df[
                self.processor.df["currency"] == currency
            ].copy()

            if len(currency_data) == 0:
                print(f"No data found for {currency}")
                continue

            # Create temporary processor for this currency
            temp_processor = OptionChainProcessor(self.processor.csv_path)
            temp_processor.df = currency_data
            temp_processor.split_by_type()

            # Calculate IL
            temp_calc = ImpermanentLossCalculator(temp_processor)

            # Estimate P0 from data (use median strike as proxy)
            P0_estimate = currency_data["strike"].median()
            temp_calc.set_parameters(P0_estimate)

            # Try to derive better forward price
            try:
                forward_price = temp_calc.derive_forward_price()
                temp_calc.set_parameters(forward_price)
            except:
                print(f"Could not derive forward for {currency}, using median strike")

            # Calculate IL with bounded liquidity profile
            # Uniform distribution within data range, no liquidity outside range
            il_results = temp_calc.calculate_il(
                liquidity_func=lambda q: LiquidityProfile.flat_profile(
                    q, currency_data["strike"].min(), currency_data["strike"].max()
                )
            )

            results[currency] = il_results

        return results


class ILVisualizer:
    """Create visualizations for impermanent loss analysis."""

    @staticmethod
    def plot_option_chain(processor: OptionChainProcessor, currency: str = None):
        """Plot option chain data using pandas plotly backend."""
        if currency:
            data = processor.df[processor.df["currency"] == currency]
        else:
            data = processor.df

        # Plot calls
        calls = data[data["option_type"] == "C"]
        if len(calls) > 0:
            fig_calls = calls.plot.scatter(
                x="strike",
                y="mid_price",
                title=f'Call Options - {currency or "All"}',
                labels={"strike": "Strike Price", "mid_price": "Option Price"},
            )
            fig_calls.update_traces(marker_color="green")
            fig_calls.show()

        # Plot puts
        puts = data[data["option_type"] == "P"]
        if len(puts) > 0:
            fig_puts = puts.plot.scatter(
                x="strike",
                y="mid_price",
                title=f'Put Options - {currency or "All"}',
                labels={"strike": "Strike Price", "mid_price": "Option Price"},
            )
            fig_puts.update_traces(marker_color="red")
            fig_puts.show()

    @staticmethod
    def plot_il_analysis(il_results: Dict, currency: str):
        """Plot IL analysis results using pandas plotly backend."""
        strikes = il_results["strikes"]
        weights = il_results["weights"]
        liquidity = il_results["liquidity"]
        option_prices = il_results["option_prices"]
        P0 = il_results.get("P0", strikes[len(strikes) // 2])

        # Create DataFrame for plotting
        df_analysis = pd.DataFrame(
            {
                "strike": strikes,
                "weights": weights,
                "liquidity": liquidity,
                "put_prices": option_prices["puts"],
                "call_prices": option_prices["calls"],
            }
        )

        # Plot weights and liquidity with dual y-axis
        fig1 = df_analysis.plot.line(
            x="strike",
            y="weights",
            title=f"Bounded Liquidity Profile - {currency}",
            labels={"strike": "Strike Price", "weights": "Weights ℓ(K)/(2K^(3/2))"},
        )
        fig1.update_traces(line_color="blue")

        # Add liquidity line with secondary y-axis
        fig1.add_scatter(
            x=df_analysis["strike"],
            y=df_analysis["liquidity"],
            name="Liquidity ℓ(K)",
            line=dict(dash="dash", color="orange"),
            yaxis="y2",
        )

        fig1.update_layout(
            yaxis2=dict(title="Liquidity ℓ(K)", side="right", overlaying="y"),
            height=400,
        )
        fig1.show()

        # Plot option prices - plot each line separately
        fig2 = df_analysis.plot.line(
            x="strike",
            y="put_prices",
            title=f"Option Prices - {currency}",
            labels={"strike": "Strike Price", "value": "Option Price"},
        )
        fig2.update_traces(name="Put Prices", line_color="red")

        # Add call prices
        fig2.add_scatter(
            x=df_analysis["strike"],
            y=df_analysis["call_prices"],
            name="Call Prices",
            line=dict(color="green"),
        )

        # Add P0 vertical line
        fig2.add_vline(x=P0, line_dash="dash", line_color="black", annotation_text="P0")
        fig2.update_layout(height=400)
        fig2.show()

    @staticmethod
    def plot_il_comparison(results_by_currency: Dict):
        """Plot IL comparison across currencies using pandas plotly backend."""
        currencies = list(results_by_currency.keys())
        ils = [results_by_currency[curr]["total_il"] for curr in currencies]

        # Create DataFrame for plotting
        df_comparison = pd.DataFrame({"currency": currencies, "il": ils})

        # Interactive bar chart
        fig = df_comparison.plot.bar(
            x="currency",
            y="il",
            title="Impermanent Loss Comparison Across Currencies",
            labels={"currency": "Currency", "il": "Impermanent Loss"},
            text=[f"{il:.3f}" for il in ils],
            text_auto=True,
        )
        fig.update_traces(
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(currencies)]
        )

        fig.update_layout(height=600, showlegend=False)

        fig.show()


def main():
    """Main function to run the impermanent loss simulation."""

    # Load data
    # Load data
    # csv_path = "path/to/your/deribit_options_data.csv"  # Update this path
    # processor = OptionChainProcessor(csv_path)
    # df = processor.load_data()
    
    print("Please import this module and use the classes directly, or update the csv_path in main()")
    return {}




def get_uniswap_data(csv_path):
    """Get pool data from Uniswap V3 ticks CSV"""
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Parse metadata
    # Parse metadata and find header
    metadata = {}
    header_row_index = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            if ":" in line:
                key = line.split(":")[0].replace("#", "").strip()
                value = line.split(":")[1].strip()
                metadata[key] = value
        elif "tick_index" in line:
            header_row_index = i
            break

    # Read tick data
    df_ticks = pd.read_csv(csv_path, skiprows=header_row_index)

    return metadata, df_ticks


if __name__ == "__main__":
    results = main()
