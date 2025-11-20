"""
Parallel Deribit Data Fetcher
Fetches option data in parallel using multi-threading for improved performance.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Optional, Union, Set
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class ParallelDeribitAPIClient:
    """Parallel Deribit API Client"""

    BASE_URL = "https://www.deribit.com/api/v2/"

    def __init__(self, timeout: int = 30, max_workers: int = 10):
        """
        Initialize parallel API client

        Args:
            timeout: Request timeout in seconds
            max_workers: Maximum number of concurrent threads
        """
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ParallelDeribitFetcher/1.0"})

        # Thread lock for rate limiting
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms minimum interval

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Generic GET request method"""
        try:
            # Rate limiting
            with self.rate_limit_lock:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)
                self.last_request_time = time.time()

            url = f"{self.BASE_URL}{endpoint}"
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if "error" in data and data["error"]:
                raise Exception(f"API error: {data['error']}")

            return data.get("result", {})

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_instruments(
        self, currency: str, kind: str = "option", expired: bool = False
    ) -> List[Dict]:
        """Get list of instruments"""
        params = {"currency": currency, "kind": kind, "expired": str(expired).lower()}
        return self._make_request("public/get_instruments", params)

    def get_order_book(self, instrument_name: str) -> Dict:
        """Get order book data"""
        params = {"instrument_name": instrument_name}
        return self._make_request("public/get_order_book", params)

    def get_ticker(self, instrument_name: str) -> Dict:
        """Get ticker data"""
        params = {"instrument_name": instrument_name}
        return self._make_request("public/ticker", params)

    def get_tradingview_chart_data(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        resolution: str = "1D",
    ) -> Dict:
        """Get historical candlestick data"""
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "resolution": resolution,
        }
        return self._make_request("public/get_tradingview_chart_data", params)


class ParallelOptionDataProcessor:
    """Parallel Option Data Processor"""

    @staticmethod
    def parse_instrument_name(instrument_name: str) -> Dict[str, Union[str, float]]:
        """Parse option instrument name"""
        try:
            parts = instrument_name.split("-")
            if len(parts) >= 4:
                currency = parts[0]
                expiry = parts[1]

                # Handle scientific notation for strike price (e.g., 2d4 = 2.4, 5d6 = 5.6)
                strike_str = parts[2]
                if "d" in strike_str:
                    # Convert scientific notation to float
                    # Example: 2d4 -> 2.4, 5d6 -> 5.6, 1d2 -> 1.2
                    base, exponent = strike_str.split("d")
                    # Correction logic: 2d4 = 2.4, 5d6 = 5.6
                    strike = float(base) + float(exponent) / 10
                else:
                    strike = float(strike_str)

                option_type = parts[3]

                return {
                    "currency": currency,
                    "expiry": expiry,
                    "strike": strike,
                    "option_type": option_type,
                }
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse instrument name {instrument_name}: {e}")

        return {}

    @staticmethod
    def clean_price_data(price: Optional[float]) -> float:
        """Clean price data"""
        if price is None or price == 0:
            return np.nan
        return float(price)


class ParallelDeribitOptionsFetcher:
    """Parallel Deribit Options Data Fetcher"""

    MS_IN_YEAR = 365 * 24 * 60 * 60 * 1000

    def __init__(
        self,
        api_client: Optional[ParallelDeribitAPIClient] = None,
        data_dir: str = "deribit_data_test",
    ):
        """Initialize parallel fetcher"""
        self.api = api_client or ParallelDeribitAPIClient()
        self.processor = ParallelOptionDataProcessor()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.options_dir = self.data_dir / "options"
        self.pqt_dir = self.options_dir / "pqt"
        self.csv_dir = self.options_dir / "csv"
        self.options_dir.mkdir(exist_ok=True)
        self.pqt_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)

    @staticmethod
    def _parse_symbol(symbol: str) -> Dict[str, str]:
        """Parse symbol"""
        parts = symbol.upper().split("_")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid symbol format: {symbol}. Expected like 'BTC_USDC'."
            )
        return {"currency": parts[0], "settlement": parts[1]}

    def _fetch_single_instrument_data(
        self, instrument: Dict, symbol_label: str, expiry_date: str
    ) -> Optional[Dict]:
        """Fetch data for a single instrument (for parallel processing)"""
        try:
            instrument_name = instrument["instrument_name"]

            # Parallel fetch order book (contains mark_price and stats)
            # Optimization: Removed get_ticker call to save API limits
            order_book = self.api.get_order_book(instrument_name)
            
            order_timestamp_raw = order_book.get("timestamp")
            expiration_timestamp_raw = instrument.get("expiration_timestamp")

            order_timestamp = (
                int(order_timestamp_raw) if order_timestamp_raw is not None else pd.NA
            )
            expiration_timestamp = (
                int(expiration_timestamp_raw)
                if expiration_timestamp_raw is not None
                else pd.NA
            )

            time_to_expiration = np.nan
            if order_timestamp_raw is not None and expiration_timestamp_raw is not None:
                try:
                    time_diff_ms = int(expiration_timestamp_raw) - int(
                        order_timestamp_raw
                    )
                    if time_diff_ms < 0:
                        logger.debug(
                            "Negative time to expiration for %s, setting to 0",
                            instrument_name,
                        )
                        time_diff_ms = 0
                    time_to_expiration = time_diff_ms / self.MS_IN_YEAR
                except (TypeError, ValueError) as e:
                    logger.warning(
                        "Failed to compute time to expiration for %s: %s",
                        instrument_name,
                        e,
                    )

            # Parse instrument info
            parsed_info = self.processor.parse_instrument_name(instrument_name)

            # Extract price fields
            best_bid = self.processor.clean_price_data(order_book.get("best_bid_price"))
            best_ask = self.processor.clean_price_data(order_book.get("best_ask_price"))
            mark_price = self.processor.clean_price_data(order_book.get("mark_price"))

            # Extract stats (volume is usually in stats object)
            stats = order_book.get("stats", {})
            volume = stats.get("volume") if stats else order_book.get("volume", 0)
            open_interest = order_book.get("open_interest", 0)

            # Build record
            record = {
                "currency": symbol_label,
                "instrument_name": instrument_name,
                "expiry_date": expiry_date,
                "strike": parsed_info.get("strike", np.nan),
                "option_type": parsed_info.get("option_type", "Unknown"),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mark_price": mark_price,
                "bid_size": order_book.get("best_bid_amount", 0),
                "ask_size": order_book.get("best_ask_amount", 0),
                "volume": volume,
                "open_interest": open_interest,
                "orderbook_timestamp": order_timestamp,
                "expiration_timestamp": expiration_timestamp,
                "time_to_expiration": time_to_expiration,
            }

            return record

        except Exception as e:
            logger.error(
                f"Error processing {instrument.get('instrument_name', 'Unknown')}: {e}"
            )
            return None

    def fetch_options_by_expiry_parallel(
        self,
        symbols: List[str],
        expiry_date: Optional[str] = None,
        max_workers: int = 10,
    ) -> pd.DataFrame:
        """
        Fetch options data in parallel

        Args:
            symbols: List of symbols
            expiry_date: Expiry date, leave empty to fetch all unexpired options
            max_workers: Maximum number of concurrent workers

        Returns:
            DataFrame with options data
        """
        target_dates: Optional[Set[date]] = None
        if expiry_date:
            target_dates = {datetime.strptime(expiry_date, "%Y-%m-%d").date()}

        all_data = []

        # Identify required currencies from symbols (e.g., BTC_USDC -> USDC, BTC_BTC -> BTC)
        desired_underlyings = {}
        required_currencies = set()
        
        for s in symbols:
            try:
                parsed = self._parse_symbol(s)
                currency = parsed["currency"]
                settlement = parsed["settlement"]
                desired_underlyings[currency] = s
                required_currencies.add(settlement)
            except ValueError:
                logger.warning(f"Skipping invalid symbol format: {s}")

        # Fetch instruments for all required currencies
        all_instruments = []
        for settlement_currency in required_currencies:
            try:
                logger.info(f"Fetching instruments for settlement currency: {settlement_currency}")
                instruments = self.api.get_instruments(settlement_currency)
                all_instruments.extend(instruments)
            except Exception as e:
                logger.error(f"Failed to fetch instruments for {settlement_currency}: {e}")

        # Filter instruments
        filtered_instruments = []
        for instrument in all_instruments:
            try:
                instrument_expiry = datetime.utcfromtimestamp(
                    instrument["expiration_timestamp"] / 1000
                ).date()
                if target_dates and instrument_expiry not in target_dates:
                    continue
                
                # Check if this instrument matches our desired underlyings
                base_ccy = instrument.get("base_currency") or instrument.get("currency")
                settlement_ccy = instrument.get("settlement_currency")
                
                # We only want instruments where the base currency is in our list
                # AND the settlement currency matches what we expect for that base currency
                # (Simplify: just check if base currency is in our desired list)
                if base_ccy in desired_underlyings:
                     # Optional: Strict check if settlement matches the symbol definition
                     # For now, we trust the API returned instruments for the requested settlement currencies
                    filtered_instruments.append(instrument)
            except (ValueError, KeyError) as e:
                logger.warning(f"Error processing instrument: {e}")
                continue

        logger.info(
            f"Found {len(filtered_instruments)} instruments for parallel processing"
        )

        # Prepare parallel tasks
        tasks = []
        for instrument in filtered_instruments:
            base_ccy = instrument.get("base_currency") or instrument.get("currency")
            symbol_label = desired_underlyings.get(base_ccy, f"{base_ccy}_USDC")
            instrument_expiry = datetime.utcfromtimestamp(
                instrument["expiration_timestamp"] / 1000
            ).strftime("%Y-%m-%d")
            tasks.append((instrument, symbol_label, instrument_expiry))

        # Execute in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self._fetch_single_instrument_data, task[0], task[1], task[2]
                ): task
                for task in tasks
            }

            # Collect results
            for future in as_completed(future_to_task):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        end_time = time.time()
        logger.info(
            f"Parallel processing completed in {end_time - start_time:.2f} seconds"
        )

        # Build DataFrame
        df = pd.DataFrame(all_data)

        if not df.empty:
            sort_cols = ["currency", "expiry_date", "option_type", "strike"]
            existing_cols = [col for col in sort_cols if col in df.columns]
            df = df.sort_values(existing_cols).reset_index(drop=True)
            logger.info(f"Successfully fetched {len(df)} option rows in parallel")
            unique_expiries = df["expiry_date"].dropna().unique()
            logger.info(
                f"Captured {len(unique_expiries)} unique expiries: {unique_expiries}"
            )
        else:
            logger.warning("No option data fetched")

        return df

    def _set_data_types(
        self, df: pd.DataFrame, data_type: str = "options"
    ) -> pd.DataFrame:
        """Set data types"""
        if data_type == "options":
            # Option data types
            dtype_mapping = {
                "currency": "category",
                "instrument_name": "string",
                "expiry_date": "string",
                "strike": "float64",
                "option_type": "category",
                "best_bid": "float64",
                "best_ask": "float64",
                "mark_price": "float64",
                "bid_size": "float64",
                "ask_size": "float64",
                "volume": "float64",
                "open_interest": "float64",
                "orderbook_timestamp": "Int64",
                "expiration_timestamp": "Int64",
                "time_to_expiration": "float64",
            }
        else:
            return df

        # Apply data types
        for col, dtype in dtype_mapping.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Failed to convert column {col} to {dtype}: {e}")

        return df

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        filename: str = None,
        data_type: str = "options",
        base_timestamp: str = None,
    ) -> str:
        """Save DataFrame to Parquet format"""
        if filename is None:
            if base_timestamp:
                timestamp = base_timestamp
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deribit_{data_type}_{timestamp}.parquet"

        # Set data types
        df = self._set_data_types(df, data_type)

        # Select save directory
        if data_type == "options":
            save_dir = self.pqt_dir
        else:
            save_dir = self.data_dir

        # Save file
        filepath = save_dir / filename
        df.to_parquet(filepath, index=False, compression="snappy")
        logger.info(f"Saved Parquet to: {filepath}")
        return str(filepath)

    def save_to_csv(
        self, df: pd.DataFrame, filename: str = None, base_timestamp: str = None
    ) -> str:
        """Save DataFrame to CSV"""
        if filename is None:
            if base_timestamp:
                timestamp = base_timestamp
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deribit_options_{timestamp}.csv"

        # Save to CSV subdirectory
        filepath = self.csv_dir / filename
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.info(f"Saved CSV to: {filepath}")
        return str(filepath)


def main():
    """Example usage"""
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("🚀 Parallel Deribit Data Fetcher")
    print("=" * 60)

    # Create parallel client and fetcher
    api_client = ParallelDeribitAPIClient(max_workers=10)
    fetcher = ParallelDeribitOptionsFetcher(api_client)

    # Parameters
    symbols = ["BTC_USDC", "ETH_USDC"]
    expiry_date = None

    try:
        # Fetch options data in parallel
        logger.info("Starting parallel options data fetch...")
        start_time = time.time()

        options_data = fetcher.fetch_options_by_expiry_parallel(
            symbols, expiry_date, max_workers=10
        )

        end_time = time.time()

        if not options_data.empty:
            # Data Quality Check
            n_vol = (options_data["volume"] > 0).sum()
            n_oi = (options_data["open_interest"] > 0).sum()
            n_bid = (options_data["best_bid"] > 0).sum()
            
            print("\n=== Data Quality ===")
            print(f"Rows with Volume > 0: {n_vol} ({n_vol/len(options_data)*100:.1f}%)")
            print(f"Rows with Open Interest > 0: {n_oi} ({n_oi/len(options_data)*100:.1f}%)")
            print(f"Rows with Best Bid > 0: {n_bid} ({n_bid/len(options_data)*100:.1f}%)")

            # Statistics
            print("\n=== Breakdown ===")
            print(
                options_data.groupby(["currency", "option_type"], observed=True).size()
            )

            # Sample
            print("\n=== Data Sample (Top 5) ===")
            print(options_data[["instrument_name", "mark_price", "volume", "open_interest"]].head(5))

            # Save with unified timestamp
            base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fetcher.save_to_parquet(
                options_data, data_type="options", base_timestamp=base_timestamp
            )

            # Save CSV as backup
            fetcher.save_to_csv(options_data, base_timestamp=base_timestamp)

        else:
            print("No data fetched. Check network/API availability or symbol support.")

    except Exception as e:
        logger.error(f"Program error: {e}")
        raise


if __name__ == "__main__":
    main()
