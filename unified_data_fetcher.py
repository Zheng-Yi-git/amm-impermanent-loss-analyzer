"""
Unified Data Fetcher
Fetches both pool data and option data with consistent timestamps
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parallel_deribit_data_fetcher import (
    ParallelDeribitAPIClient,
    ParallelDeribitOptionsFetcher,
)


def get_unified_timestamp():
    """Get unified timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fetch_options_data(symbols, base_timestamp):
    """Fetch options data"""
    api_client = ParallelDeribitAPIClient(max_workers=10)
    fetcher = ParallelDeribitOptionsFetcher(api_client)

    try:
        options_data = fetcher.fetch_options_by_expiry_parallel(symbols, max_workers=10)

        if not options_data.empty:
            # Save options data
            parquet_file = fetcher.save_to_parquet(
                options_data, data_type="options", base_timestamp=base_timestamp
            )
            csv_file = fetcher.save_to_csv(options_data, base_timestamp=base_timestamp)

            print(f"✅ Options data saved:")
            print(f"   Parquet: {parquet_file}")
            print(f"   CSV: {csv_file}")
            print(f"   Rows: {len(options_data)}")
            expiry_array = (
                options_data["expiry_date"].dropna().astype(str).unique()
                if "expiry_date" in options_data.columns
                else []
            )
            expiry_list = sorted(expiry_array.tolist()) if len(expiry_array) else []
            if expiry_list:
                preview_expiries = ", ".join(expiry_list[:10])
                if len(expiry_list) > 10:
                    preview_expiries += ", ..."
                print(f"   Expiry dates count: {len(expiry_list)}")
                print(f"   Expiry examples: {preview_expiries}")
            if "time_to_expiration" in options_data.columns:
                tte_series = options_data["time_to_expiration"].dropna()
                if not tte_series.empty:
                    print(
                        "   Time to Expiration (years): "
                        f"min {tte_series.min():.6f}, "
                        f"avg {tte_series.mean():.6f}, "
                        f"max {tte_series.max():.6f}"
                    )
            if "orderbook_timestamp" in options_data.columns:
                ts_series = options_data["orderbook_timestamp"].dropna()
                if not ts_series.empty:
                    order_time_range = datetime.fromtimestamp(
                        ts_series.min() / 1000
                    ).isoformat()
                    latest_order_time = datetime.fromtimestamp(
                        ts_series.max() / 1000
                    ).isoformat()
                    print(
                        f"   Orderbook time range: {order_time_range} ~ {latest_order_time}"
                    )

            return True, options_data
        else:
            print("❌ No options data retrieved")
            return False, None

    except Exception as e:
        print(f"❌ Options data fetch failed: {e}")
        return False, None


def fetch_pool_data(base_timestamp):
    """Fetch pool data"""
    # Define token list
    tokens = ["WBTC", "WETH"]

    pool_data_dir = Path("pool_data_test")
    pool_data_dir.mkdir(exist_ok=True)

    success_count = 0
    total_tokens = len(tokens)

    for token in tokens:
        try:
            print(f"📊 Fetching {token} pool data...")

            # Build Node.js command (with full path)
            cmd = [
                "node",
                "dist/index.js",
                f"--token0={token}",
                f"--timestamp={base_timestamp}",
            ]

            # Switch to uniswap-v3-pool-reconstructor directory
            original_dir = os.getcwd()
            pool_reconstructor_dir = Path("uniswap-v3-pool-reconstructor")

            if not pool_reconstructor_dir.exists():
                print(f"❌ Directory not found: {pool_reconstructor_dir}")
                continue

            os.chdir(pool_reconstructor_dir)

            # Execute command
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print(f"✅ {token} pool data fetched successfully")
                success_count += 1

                # Find generated files
                output_files = list(Path(".").glob(f"*{token}*ticks*.csv"))
                if output_files:
                    latest_file = max(output_files, key=os.path.getctime)
                    print(f"   File: {latest_file}")
            else:
                print(f"❌ {token} pool data fetch failed:")
                print(f"   Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"⏰ {token} pool data fetch timeout")
        except Exception as e:
            print(f"❌ {token} pool data fetch exception: {e}")
        finally:
            # Restore original directory
            os.chdir(original_dir)

    print(f"\n📊 Pool data fetch complete: {success_count}/{total_tokens}")
    return success_count > 0, None


def main():
    """Main function - Unified data fetch"""
    print("=" * 60)
    print("🔄 Unified Data Fetcher")
    print("=" * 60)

    # Get unified timestamp
    base_timestamp = get_unified_timestamp()
    print(f"📅 Unified timestamp: {base_timestamp}")

    # Parameter settings
    symbols = ["BTC_USDC", "ETH_USDC"]

    success_count = 0
    total_tasks = 2

    # 1. Fetch options data
    print("\n" + "=" * 40)
    print("📊 Fetching Options Data")
    print("=" * 40)
    options_success, options_data = fetch_options_data(symbols, base_timestamp)
    if options_success:
        success_count += 1

    # 2. Fetch pool data
    print("\n" + "=" * 40)
    print("🏊 Fetching Pool Data")
    print("=" * 40)
    pool_success, pool_data = fetch_pool_data(base_timestamp)
    if pool_success:
        success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("📋 Data Fetch Summary")
    print("=" * 60)
    print(f"✅ Success: {success_count}/{total_tasks}")
    print(f"📅 Timestamp: {base_timestamp}")

    if success_count == total_tasks:
        print("🎉 All data fetched successfully!")
    else:
        print("⚠️  Partial data fetch failed, please check logs")


if __name__ == "__main__":
    main()
