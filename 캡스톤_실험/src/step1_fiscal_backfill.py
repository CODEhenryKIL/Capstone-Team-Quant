from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

import config
from src import step1_prices


FISCAL_BASE_URL = "https://api.fiscal.ai"
COMPANIES_LIST_CACHE = config.CACHE_DIR / "fiscal" / "companies_list.csv"
ATTEMPTS_CSV = config.INTERIM_DIR / "fiscal_price_backfill_attempts.csv"


def fiscal_headers(api_key: str) -> dict[str, str]:
    return {
        "X-Api-Key": api_key,
        "User-Agent": "STAI-CARL-Capstone/1.0",
    }


def fetch_companies_list(api_key: str, refresh: bool = False) -> pd.DataFrame:
    if COMPANIES_LIST_CACHE.exists() and not refresh:
        return pd.read_csv(COMPANIES_LIST_CACHE, dtype=str)

    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        response = requests.get(
            f"{FISCAL_BASE_URL}/v2/companies-list",
            params={"pageNumber": page, "pageSize": 1000},
            headers=fiscal_headers(api_key),
            timeout=config.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("data") or [])
        pagination = payload.get("pagination") or {}
        if not pagination.get("hasNextPage"):
            break
        page += 1
        time.sleep(0.1)

    frame = pd.DataFrame(rows)
    COMPANIES_LIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(COMPANIES_LIST_CACHE, index=False)
    return frame


def load_missing_tickers(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing price failures file: {path}")
    frame = pd.read_csv(path, dtype=str)
    if "ticker" not in frame.columns:
        raise ValueError(f"{path} must contain a ticker column")
    return frame.dropna(subset=["ticker"]).drop_duplicates("ticker")


def stock_prices_available(value: object) -> bool:
    return "stock_prices" in str(value)


def strict_us_ticker_matches(missing: pd.DataFrame, companies: pd.DataFrame) -> pd.DataFrame:
    tickers = set(missing["ticker"].astype(str))
    candidates = companies.copy()
    for column in ["countryCode", "availableDatasets", "ticker"]:
        if column not in candidates.columns:
            candidates[column] = ""

    candidates = candidates[
        (candidates["countryCode"].astype(str).eq("US"))
        & candidates["availableDatasets"].map(stock_prices_available)
        & candidates["ticker"].astype(str).isin(tickers)
    ].copy()

    columns = [
        "ticker",
        "name",
        "exchangeSymbol",
        "micCode",
        "tradingStatus",
        "cik",
        "figi",
    ]
    for column in columns:
        if column not in candidates.columns:
            candidates[column] = ""
    return candidates[columns].drop_duplicates("ticker", keep="first")


def fetch_stock_prices(
    api_key: str,
    ticker: str,
    exchange: str | None,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame | None, str | None]:
    params: dict[str, str] = {
        "ticker": ticker,
        "startDate": start_date,
        "endDate": end_date,
    }
    if exchange:
        params["exchange"] = exchange

    try:
        response = requests.get(
            f"{FISCAL_BASE_URL}/v2/company/stock-prices",
            params=params,
            headers=fiscal_headers(api_key),
            timeout=config.REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return None, f"{response.status_code}: {response.text[:500]}"
        payload = response.json()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    rows = payload if isinstance(payload, list) else payload.get("data", [])
    if not rows:
        return None, "empty response"

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns or "close_price" not in frame.columns:
        return None, f"unexpected columns: {list(frame.columns)}"

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close_price"], errors="coerce")
    if "adjusted_close_price" in frame.columns:
        frame["adj_close"] = pd.to_numeric(frame["adjusted_close_price"], errors="coerce")
    else:
        # Fiscal.ai documents stock prices as split-adjusted. It does not expose a
        # separate dividend-adjusted close in this endpoint, so keep both fields
        # aligned with the available close for compatibility with the pipeline.
        frame["adj_close"] = frame["close"]

    frame = (
        frame.dropna(subset=["date", "close"])
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )
    if frame.empty:
        return None, "empty parsed dataframe"
    return frame[["close", "adj_close"]], None


def rebuild_price_outputs(start_date: str, end_date: str) -> pd.DataFrame:
    symbols = step1_prices.load_price_symbols()
    close_frames = []
    adj_frames = []
    failures = []
    for row in symbols:
        cached = step1_prices.load_cached_price(
            row["ticker"],
            start_date,
            end_date,
            int(row.get("last_year") or config.END_YEAR),
        )
        if cached is None:
            failures.append(
                {
                    "ticker": row["ticker"],
                    "price_ticker": row.get("price_ticker") or row["ticker"],
                    "reason": "missing cache after Fiscal.ai backfill",
                }
            )
            continue
        close_frames.append(cached[["close"]].rename(columns={"close": row["ticker"]}))
        adj_frames.append(cached[["adj_close"]].rename(columns={"adj_close": row["ticker"]}))

    close = pd.concat(close_frames, axis=1).sort_index() if close_frames else pd.DataFrame()
    adj_close = pd.concat(adj_frames, axis=1).sort_index() if adj_frames else pd.DataFrame()

    if not close.empty:
        config.PRICE_DAILY_CLOSE_CSV.parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(config.PRICE_DAILY_CLOSE_CSV)
        adj_close.to_csv(config.PRICE_DAILY_ADJ_CLOSE_CSV)
        step1_prices.to_monthly(close).to_csv(config.PRICE_MONTHLY_CLOSE_CSV)
        step1_prices.to_monthly(adj_close).to_csv(config.PRICE_MONTHLY_ADJ_CLOSE_CSV)

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(config.PRICE_FAILURES_CSV, index=False)
    return failures_df


def run(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh_companies: bool = False,
    rebuild_outputs: bool = True,
) -> None:
    config.ensure_dirs()
    start_date = start_date or f"{config.START_YEAR}-01-01"
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    missing = load_missing_tickers(config.PRICE_FAILURES_CSV)
    companies = fetch_companies_list(api_key, refresh=refresh_companies)
    matches = strict_us_ticker_matches(missing, companies)
    match_by_ticker = matches.set_index("ticker").to_dict("index") if not matches.empty else {}

    attempts = []
    success_count = 0
    for _, row in missing.iterrows():
        ticker = str(row["ticker"])
        match = match_by_ticker.get(ticker)
        if not match:
            attempts.append(
                {
                    "ticker": ticker,
                    "status": "skipped",
                    "reason": "no strict US Fiscal.ai stock_prices match",
                }
            )
            continue

        frame, error = fetch_stock_prices(
            api_key=api_key,
            ticker=ticker,
            exchange=str(match.get("exchangeSymbol") or ""),
            start_date=start_date,
            end_date=end_date,
        )
        if error or frame is None:
            attempts.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "reason": error,
                    "fiscal_name": match.get("name"),
                    "exchange": match.get("exchangeSymbol"),
                    "tradingStatus": match.get("tradingStatus"),
                }
            )
            continue

        step1_prices.write_cached_price(ticker, frame)
        attempts.append(
            {
                "ticker": ticker,
                "status": "success",
                "reason": "",
                "rows": len(frame),
                "first_date": frame.index.min().date(),
                "last_date": frame.index.max().date(),
                "fiscal_name": match.get("name"),
                "exchange": match.get("exchangeSymbol"),
                "tradingStatus": match.get("tradingStatus"),
            }
        )
        success_count += 1
        time.sleep(0.15)

    attempts_df = pd.DataFrame(attempts)
    attempts_df.to_csv(ATTEMPTS_CSV, index=False)

    failures_df = None
    if success_count and rebuild_outputs:
        failures_df = rebuild_price_outputs(start_date, end_date)

    print(f"[fiscal] missing tickers before: {len(missing)}")
    print(f"[fiscal] strict US matches: {len(matches)}")
    print(f"[fiscal] successful backfills: {success_count}")
    if failures_df is not None:
        print(f"[fiscal] remaining price failures: {len(failures_df)}")
    print(f"[fiscal] attempts saved: {ATTEMPTS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Yahoo price failures with Fiscal.ai")
    parser.add_argument("--start-date", default=f"{config.START_YEAR}-01-01")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--refresh-companies", action="store_true")
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("FISCAL_AI_API_KEY")
    if not api_key:
        raise SystemExit("Set FISCAL_AI_API_KEY before running Fiscal.ai backfill")

    run(
        api_key=api_key,
        start_date=args.start_date,
        end_date=args.end_date,
        refresh_companies=args.refresh_companies,
        rebuild_outputs=not args.no_rebuild,
    )


if __name__ == "__main__":
    main()
