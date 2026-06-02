from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pandas.tseries.offsets import MonthEnd
from tqdm import tqdm

import config
from src.utils import read_json


def unix_timestamp(date_text: str) -> int:
    dt = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def cache_safe_ticker(ticker: str) -> str:
    return ticker.replace("/", "_").replace(".", "-")


def price_cache_path(ticker: str) -> Path:
    return config.PRICE_CACHE_DIR / f"{cache_safe_ticker(ticker)}.csv"


def load_cached_price(
    ticker: str,
    start_date: str,
    end_date: str,
    last_sp500_year: int,
) -> pd.DataFrame | None:
    path = price_cache_path(ticker)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    required = {"close", "adj_close"}
    if frame.empty or not required.issubset(frame.columns):
        return None

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if frame.index.min() > start + pd.Timedelta(days=10):
        return None
    if last_sp500_year >= config.END_YEAR and frame.index.max() < end - pd.Timedelta(days=10):
        return None
    return frame.sort_index()


def write_cached_price(ticker: str, frame: pd.DataFrame) -> None:
    path = price_cache_path(ticker)
    tmp_path = path.with_suffix(".tmp")
    frame[["close", "adj_close"]].to_csv(tmp_path)
    tmp_path.replace(path)


def fetch_chart(symbol: str, start_date: str, end_date: str) -> tuple[str, pd.DataFrame | None, str | None]:
    url = config.YAHOO_CHART_URL.format(symbol=symbol)
    params = {
        "period1": unix_timestamp(start_date),
        "period2": unix_timestamp(end_date),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers=config.HTTP_HEADERS,
                timeout=config.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            chart = payload.get("chart", {})
            error = chart.get("error")
            if error:
                return symbol, None, str(error)
            result = (chart.get("result") or [None])[0]
            if not result:
                return symbol, None, "empty result"

            timestamps = result.get("timestamp") or []
            quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
            adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]

            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
                    "close": quote.get("close", []),
                    "adj_close": adj.get("adjclose", quote.get("close", [])),
                }
            )
            frame = frame.dropna(subset=["date"]).drop_duplicates("date")
            frame = frame.set_index("date").sort_index()
            if frame.empty:
                return symbol, None, "empty dataframe"
            return symbol, frame, None
        except Exception as exc:
            if attempt == 2:
                return symbol, None, f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))

    return symbol, None, "unknown failure"


def load_price_symbols() -> list[dict[str, Any]]:
    if not config.SP500_UNIVERSE_JSON.exists():
        raise FileNotFoundError(f"Run step0 first: {config.SP500_UNIVERSE_JSON}")
    universe = read_json(config.SP500_UNIVERSE_JSON)
    symbols = []
    for item in universe:
        symbols.append(
            {
                "ticker": item["ticker"],
                "price_ticker": item.get("price_ticker") or item["ticker"],
                "last_year": int(item.get("last_year") or config.END_YEAR),
            }
        )
    return symbols


def collect_prices(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbols = load_price_symbols()
    close_frames = []
    adj_frames = []
    failures = []
    pending = []

    for row in symbols:
        cached = load_cached_price(row["ticker"], start_date, end_date, row["last_year"])
        if cached is None:
            pending.append(row)
            continue
        close_frames.append(cached[["close"]].rename(columns={"close": row["ticker"]}))
        adj_frames.append(cached[["adj_close"]].rename(columns={"adj_close": row["ticker"]}))

    print(f"[step1] cached tickers: {len(symbols) - len(pending)}")
    print(f"[step1] tickers to fetch: {len(pending)}")

    with ThreadPoolExecutor(max_workers=config.YAHOO_MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_chart, row["price_ticker"], start_date, end_date): row
            for row in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="[step1] Yahoo chart"):
            row = futures[future]
            _, frame, error = future.result()
            ticker = row["ticker"]
            if error or frame is None:
                failures.append(
                    {
                        "ticker": ticker,
                        "price_ticker": row["price_ticker"],
                        "reason": error,
                    }
                )
                continue
            write_cached_price(ticker, frame)
            close_frames.append(frame[["close"]].rename(columns={"close": ticker}))
            adj_frames.append(frame[["adj_close"]].rename(columns={"adj_close": ticker}))
            time.sleep(config.YAHOO_SLEEP_SECONDS)

    close = pd.concat(close_frames, axis=1).sort_index() if close_frames else pd.DataFrame()
    adj_close = pd.concat(adj_frames, axis=1).sort_index() if adj_frames else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    return close, adj_close, failures_df


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    monthly = df.resample("ME").last()
    monthly.index = monthly.index + MonthEnd(0)
    return monthly.dropna(how="all")


def run(
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    config.ensure_dirs()
    start_date = start_date or f"{config.START_YEAR}-01-01"
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    print(f"[step1] Collecting Yahoo prices: {start_date} to {end_date}")

    close, adj_close, failures = collect_prices(start_date, end_date)
    failures.to_csv(config.PRICE_FAILURES_CSV, index=False)
    if close.empty:
        print(f"[step1] failed tickers: {len(failures)}")
        print(f"[step1] saved failures: {config.PRICE_FAILURES_CSV}")
        raise RuntimeError("No Yahoo price data was collected")

    monthly_close = to_monthly(close)
    monthly_adj = to_monthly(adj_close)

    close.to_csv(config.PRICE_DAILY_CLOSE_CSV)
    adj_close.to_csv(config.PRICE_DAILY_ADJ_CLOSE_CSV)
    monthly_close.to_csv(config.PRICE_MONTHLY_CLOSE_CSV)
    monthly_adj.to_csv(config.PRICE_MONTHLY_ADJ_CLOSE_CSV)

    print(f"[step1] daily close shape: {close.shape}")
    print(f"[step1] monthly close shape: {monthly_close.shape}")
    print(f"[step1] failed tickers: {len(failures)}")
    print(f"[step1] saved: {config.PRICE_MONTHLY_CLOSE_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=f"{config.START_YEAR}-01-01")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    run(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
