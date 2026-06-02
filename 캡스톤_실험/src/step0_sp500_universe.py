from __future__ import annotations

import argparse
import io
from copy import deepcopy
from datetime import datetime
from typing import Any

import pandas as pd
import requests

import config
from src.utils import (
    clean_text,
    find_column,
    flatten_columns,
    format_cik,
    load_overrides,
    normalize_name,
    normalize_ticker,
    write_json,
)


def fetch_wikipedia_tables() -> list[pd.DataFrame]:
    response = requests.get(
        config.WIKI_SP500_URL,
        headers=config.HTTP_HEADERS,
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return pd.read_html(io.StringIO(response.text))


def load_cached_cik_mapping() -> dict[str, str]:
    mapping = {}
    if config.SP500_UNIVERSE_JSON.exists():
        for item in pd.read_json(config.SP500_UNIVERSE_JSON).to_dict("records"):
            ticker = normalize_ticker(item.get("ticker"))
            cik = format_cik(item.get("cik"))
            if ticker and cik:
                mapping[ticker] = cik

    for path in [
        config.SP500_CURRENT_CSV,
        config.SP500_MEMBERSHIP_CSV,
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ticker_col = "ticker" if "ticker" in df.columns else "Ticker" if "Ticker" in df.columns else None
        if not ticker_col or "cik" not in df.columns:
            continue
        for _, row in df.dropna(subset=[ticker_col, "cik"]).iterrows():
            ticker = normalize_ticker(row[ticker_col])
            cik = format_cik(row["cik"])
            if ticker and cik:
                mapping[ticker] = cik
    return mapping


def fetch_sec_cik_mapping(allow_network: bool = True) -> dict[str, str]:
    cached = load_cached_cik_mapping()
    if not allow_network:
        return cached

    response = requests.get(
        config.SEC_COMPANY_TICKERS_URL,
        headers=config.SEC_HEADERS,
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    fetched = {
        item["ticker"].upper(): str(item["cik_str"]).zfill(10)
        for item in data.values()
    }
    cached.update(fetched)
    return cached


def load_cached_current_members() -> dict[str, dict[str, Any]]:
    if not config.SP500_CURRENT_CSV.exists():
        raise FileNotFoundError(config.SP500_CURRENT_CSV)
    current = pd.read_csv(config.SP500_CURRENT_CSV)
    members = {}
    for _, row in current.iterrows():
        ticker = normalize_ticker(row.get("ticker"))
        if not ticker:
            continue
        date_added = pd.to_datetime(row.get("date_added"), errors="coerce")
        members[ticker] = {
            "ticker": ticker,
            "cik": format_cik(row.get("cik")),
            "sector": clean_text(row.get("sector")) or "Unknown",
            "industry": clean_text(row.get("industry")) or "Unknown",
            "name": clean_text(row.get("name")) or ticker,
            "date_added": date_added.strftime("%Y-%m-%d") if pd.notna(date_added) else None,
            "date_added_dt": date_added if pd.notna(date_added) else pd.NaT,
            "is_current_constituent": True,
            "source": "cached_current",
        }
    return members


def load_cached_changes() -> pd.DataFrame:
    if not config.SP500_CHANGES_CSV.exists():
        raise FileNotFoundError(config.SP500_CHANGES_CSV)
    changes = pd.read_csv(config.SP500_CHANGES_CSV)
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    for col in ["added_ticker", "removed_ticker"]:
        if col in changes.columns:
            changes[col] = changes[col].map(normalize_ticker)
    return changes.dropna(subset=["date"]).sort_values("date")


def parse_current_members(
    tables: list[pd.DataFrame],
    sec_cik_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    current = flatten_columns(tables[0])
    symbol_col = find_column(current.columns, "symbol")
    security_col = find_column(current.columns, "security")
    sector_col = find_column(current.columns, "gics", "sector")
    industry_col = find_column(current.columns, "gics", "sub", "industry")
    cik_col = find_column(current.columns, "cik")
    date_added_col = find_column(current.columns, "date", "added")

    rows = []
    members: dict[str, dict[str, Any]] = {}
    for _, row in current.iterrows():
        ticker = normalize_ticker(row.get(symbol_col))
        if not ticker:
            continue

        date_added = pd.to_datetime(row.get(date_added_col), errors="coerce")
        metadata = {
            "ticker": ticker,
            "cik": format_cik(row.get(cik_col)) or sec_cik_map.get(ticker),
            "sector": clean_text(row.get(sector_col)) or "Unknown",
            "industry": clean_text(row.get(industry_col)) or "Unknown",
            "name": clean_text(row.get(security_col)) or ticker,
            "date_added": date_added.strftime("%Y-%m-%d")
            if pd.notna(date_added)
            else None,
            "date_added_dt": date_added if pd.notna(date_added) else pd.NaT,
            "is_current_constituent": True,
            "source": "wikipedia_current",
        }
        members[ticker] = metadata
        rows.append({k: v for k, v in metadata.items() if k != "date_added_dt"})

    pd.DataFrame(rows).sort_values("ticker").to_csv(
        config.SP500_CURRENT_CSV,
        index=False,
    )
    return members


def parse_changes(tables: list[pd.DataFrame]) -> pd.DataFrame:
    candidates = []
    for table in tables[1:]:
        df = flatten_columns(table)
        names = [
            col.lower().replace(" ", "_").replace("-", "_")
            for col in df.columns
        ]
        if any("added" in name and "ticker" in name for name in names):
            candidates.append(df)

    if not candidates:
        raise ValueError("Wikipedia S&P 500 changes table was not found")

    changes = candidates[0]
    date_col = find_column(changes.columns, "date")
    added_ticker_col = find_column(changes.columns, "added", "ticker")
    added_security_col = find_column(changes.columns, "added", "security")
    removed_ticker_col = find_column(changes.columns, "removed", "ticker")
    removed_security_col = find_column(changes.columns, "removed", "security")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(changes[date_col], errors="coerce"),
            "added_ticker": changes[added_ticker_col].map(normalize_ticker),
            "added_security": changes[added_security_col].map(clean_text),
            "removed_ticker": changes[removed_ticker_col].map(normalize_ticker),
            "removed_security": changes[removed_security_col].map(clean_text),
        }
    )
    parsed = parsed.dropna(subset=["date"]).sort_values("date")
    parsed.to_csv(config.SP500_CHANGES_CSV, index=False)
    return parsed


def remove_added_member(
    members: dict[str, dict[str, Any]],
    change: pd.Series,
) -> None:
    added_ticker = change.get("added_ticker")
    if added_ticker and added_ticker in members:
        members.pop(added_ticker, None)
        return

    added_name = normalize_name(change.get("added_security"))
    date_matches = []
    name_matches = []
    for ticker, metadata in members.items():
        date_added_dt = metadata.get("date_added_dt")
        if pd.notna(date_added_dt) and pd.notna(change.get("date")):
            if pd.Timestamp(date_added_dt).date() == pd.Timestamp(change["date"]).date():
                date_matches.append(ticker)

        current_name = normalize_name(metadata.get("name"))
        if added_name and current_name:
            if added_name in current_name or current_name in added_name:
                name_matches.append(ticker)

    if len(name_matches) == 1:
        members.pop(name_matches[0], None)
    elif len(date_matches) == 1:
        members.pop(date_matches[0], None)


def metadata_for_removed(
    ticker: str,
    security: str | None,
    sec_cik_map: dict[str, str],
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "cik": sec_cik_map.get(ticker),
        "sector": "Unknown",
        "industry": "Unknown",
        "name": security or ticker,
        "date_added": None,
        "date_added_dt": pd.NaT,
        "is_current_constituent": False,
        "source": "wikipedia_removed",
    }


def snapshot_for_year(
    year: int,
    current_members: dict[str, dict[str, Any]],
    changes: pd.DataFrame,
    sec_cik_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    today = pd.Timestamp(datetime.now().date())
    as_of = pd.Timestamp(year=year, month=12, day=31)
    if year == today.year and as_of > today:
        as_of = today

    members = deepcopy(current_members)
    future_changes = changes[changes["date"] > as_of].sort_values(
        "date",
        ascending=False,
    )

    for _, change in future_changes.iterrows():
        removed = change.get("removed_ticker")
        if removed and removed not in members:
            members[removed] = metadata_for_removed(
                removed,
                change.get("removed_security"),
                sec_cik_map,
            )
        remove_added_member(members, change)

    for metadata in members.values():
        metadata["as_of_date"] = as_of.strftime("%Y-%m-%d")
        metadata.pop("date_added_dt", None)
    return members


def apply_overrides(
    membership: pd.DataFrame,
    sec_cik_map: dict[str, str],
) -> pd.DataFrame:
    membership = membership.copy()
    membership["price_ticker"] = membership["ticker"]
    membership["sec_ticker"] = membership["ticker"]

    overrides = load_overrides(config.OVERRIDES_CSV)
    if not overrides.empty:
        for _, row in overrides.iterrows():
            mask = membership["ticker"] == row["ticker"]
            for col in ["cik", "price_ticker", "sec_ticker", "sector", "industry", "name"]:
                if col in overrides.columns and pd.notna(row.get(col)):
                    membership.loc[mask, col] = row[col]

    missing_cik = membership["cik"].isna() | (membership["cik"].astype(str) == "")
    membership.loc[missing_cik, "cik"] = membership.loc[missing_cik, "sec_ticker"].map(sec_cik_map)

    period_overrides = load_period_overrides(config.PERIOD_OVERRIDES_CSV)
    if not period_overrides.empty:
        for _, row in period_overrides.iterrows():
            mask = (
                (membership["ticker"] == row["ticker"])
                & (membership["year"].astype(int) >= int(row["start_year"]))
                & (membership["year"].astype(int) <= int(row["end_year"]))
            )
            for col in ["cik", "price_ticker", "sec_ticker", "sector", "industry", "name"]:
                if col in period_overrides.columns and pd.notna(row.get(col)):
                    membership.loc[mask, col] = row[col]
    return membership


def load_period_overrides(path: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"ticker", "start_year", "end_year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} must contain columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
    df["end_year"] = pd.to_numeric(df["end_year"], errors="coerce")
    return df.dropna(subset=["ticker", "start_year", "end_year"])


def build_universe(membership: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = []
    dropped = []
    for ticker, group in membership.groupby("ticker", sort=True):
        group = group.sort_values("year")
        years = sorted(int(year) for year in group["year"].unique())
        latest = group.iloc[-1].to_dict()
        known_sector = group[group["sector"].fillna("Unknown") != "Unknown"]
        if not known_sector.empty:
            latest = known_sector.iloc[-1].to_dict()

        cik_values = [
            str(value).zfill(10)
            for value in group["cik"].dropna().unique()
            if str(value).strip() and str(value).lower() != "nan"
        ]
        cik = cik_values[0] if cik_values else None
        sector = latest.get("sector") or "Unknown"
        cik_by_year = {
            str(int(row["year"])): str(row["cik"]).split(".", 1)[0].zfill(10)
            for _, row in group.dropna(subset=["cik"]).iterrows()
            if str(row["cik"]).strip() and str(row["cik"]).lower() != "nan"
        }

        if not cik:
            dropped.append({"ticker": ticker, "reason": "No CIK mapping", "years": years})
            continue
        if sector in config.EXCLUDED_SECTORS:
            dropped.append(
                {
                    "ticker": ticker,
                    "reason": f"Excluded sector ({sector})",
                    "years": years,
                }
            )
            continue

        valid.append(
            {
                "ticker": ticker,
                "price_ticker": latest.get("price_ticker") or ticker,
                "sec_ticker": latest.get("sec_ticker") or ticker,
                "cik": cik,
                "sector": sector,
                "industry": latest.get("industry") or "Unknown",
                "name": latest.get("name") or ticker,
                "first_year": years[0],
                "last_year": years[-1],
                "years": years,
                "cik_by_year": cik_by_year,
            }
        )
    return valid, dropped


def run(
    start_year: int = config.START_YEAR,
    end_year: int = config.END_YEAR,
    refresh: bool = False,
) -> None:
    config.ensure_dirs()
    print(f"[step0] Building historical S&P 500 universe: {start_year}-{end_year}")

    use_cached_source = (
        not refresh
        and config.SP500_CURRENT_CSV.exists()
        and config.SP500_CHANGES_CSV.exists()
    )
    sec_cik_map = fetch_sec_cik_mapping(allow_network=not use_cached_source)
    if use_cached_source:
        print("[step0] using cached Wikipedia source CSVs")
        current_members = load_cached_current_members()
        changes = load_cached_changes()
    else:
        tables = fetch_wikipedia_tables()
        current_members = parse_current_members(tables, sec_cik_map)
        changes = parse_changes(tables)

    rows = []
    for year in range(start_year, end_year + 1):
        snapshot = snapshot_for_year(year, current_members, changes, sec_cik_map)
        rows.extend({"year": year, **metadata} for metadata in snapshot.values())

    membership = pd.DataFrame(rows).sort_values(["year", "ticker"]).reset_index(drop=True)
    membership = apply_overrides(membership, sec_cik_map)
    valid, dropped = build_universe(membership)

    valid_tickers = {item["ticker"] for item in valid}
    membership["in_dataset_universe"] = membership["ticker"].isin(valid_tickers)
    membership.to_csv(config.SP500_MEMBERSHIP_CSV, index=False)
    write_json(config.SP500_MEMBERSHIP_JSON, membership.to_dict(orient="records"))
    write_json(config.SP500_UNIVERSE_JSON, valid)
    write_json(config.SP500_DROPPED_JSON, dropped)

    summary = (
        membership.groupby("year")
        .agg(raw_members=("ticker", "nunique"), usable_members=("in_dataset_universe", "sum"))
        .reset_index()
    )
    summary.to_csv(config.SP500_SUMMARY_CSV, index=False)

    print(f"[step0] raw historical tickers: {membership['ticker'].nunique()}")
    print(f"[step0] usable universe tickers: {len(valid)}")
    print(f"[step0] dropped tickers: {len(dropped)}")
    print(f"[step0] saved: {config.SP500_MEMBERSHIP_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=config.START_YEAR)
    parser.add_argument("--end-year", type=int, default=config.END_YEAR)
    parser.add_argument("--refresh", action="store_true", help="Refetch Wikipedia and SEC source data")
    args = parser.parse_args()
    run(start_year=args.start_year, end_year=args.end_year, refresh=args.refresh)


if __name__ == "__main__":
    main()
