from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

import config
from src.utils import read_json


def load_universe() -> list[dict[str, Any]]:
    if not config.SP500_UNIVERSE_JSON.exists():
        raise FileNotFoundError(f"Run step0 first: {config.SP500_UNIVERSE_JSON}")
    return read_json(config.SP500_UNIVERSE_JSON)


def cache_path(cik: str) -> Path:
    return config.SEC_CACHE_DIR / f"CIK{cik}.json"


def fetch_companyfacts(cik: str) -> dict[str, Any]:
    path = cache_path(cik)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    url = config.SEC_COMPANY_FACTS_URL.format(cik=cik)
    response = requests.get(
        url,
        headers=config.SEC_HEADERS,
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(config.SEC_SLEEP_SECONDS)
    return payload


def iter_fact_units(tag_payload: dict[str, Any]) -> list[dict[str, Any]]:
    units = tag_payload.get("units") or {}
    for preferred in ["USD", "shares", "pure"]:
        if preferred in units:
            return units[preferred]
    if not units:
        return []
    return next(iter(units.values()))


def extract_tag_records(
    facts: dict[str, Any],
    field: str,
    tag: str,
    priority: int,
    start_year: int,
) -> list[dict[str, Any]]:
    records = []
    for taxonomy, taxonomy_facts in facts.items():
        if tag not in taxonomy_facts:
            continue
        for item in iter_fact_units(taxonomy_facts[tag]):
            form = item.get("form")
            if form not in {"10-K", "10-Q"}:
                continue
            fy = item.get("fy")
            fp = item.get("fp")
            end = item.get("end")
            filed = item.get("filed")
            value = item.get("val")
            if fy is None or fp not in {"FY", "Q1", "Q2", "Q3"}:
                continue
            if int(fy) < start_year or value is None or not end or not filed:
                continue
            records.append(
                {
                    "fiscal_year": int(fy),
                    "period": fp,
                    "period_end": end,
                    "filed_date": filed,
                    "form": form,
                    "field": field,
                    "tag": tag,
                    "taxonomy": taxonomy,
                    "tag_priority": priority,
                    "value": float(value),
                }
            )
    return records


def parse_companyfacts(
    ticker: str,
    companyfacts: dict[str, Any],
    start_year: int,
) -> pd.DataFrame:
    facts = companyfacts.get("facts") or {}
    field_frames = []

    for field, tags in config.XBRL_TAGS.items():
        records = []
        for priority, tag in enumerate(tags):
            records.extend(extract_tag_records(facts, field, tag, priority, start_year))
        if not records:
            continue

        field_df = pd.DataFrame(records)
        field_df["period_end_dt"] = pd.to_datetime(field_df["period_end"], errors="coerce")
        field_df["filed_date_dt"] = pd.to_datetime(field_df["filed_date"], errors="coerce")
        field_df = field_df.sort_values(
            ["fiscal_year", "period", "tag_priority", "period_end_dt", "filed_date_dt"],
            ascending=[True, True, True, False, True],
        )
        field_df = field_df.drop_duplicates(["fiscal_year", "period"], keep="first")
        field_df = field_df[
            [
                "fiscal_year",
                "period",
                "period_end",
                "filed_date",
                "form",
                "tag",
                "value",
            ]
        ].rename(columns={"tag": f"{field}_tag", "value": field})
        field_frames.append((field, field_df))

    if not field_frames:
        return pd.DataFrame()

    base = pd.concat(
        [
            frame[["fiscal_year", "period", "period_end", "filed_date", "form"]]
            for field, frame in field_frames
            if field != "Shares"
        ]
        or [
            frame[["fiscal_year", "period", "period_end", "filed_date", "form"]]
            for _, frame in field_frames
        ],
        ignore_index=True,
    )
    base["period_end_dt"] = pd.to_datetime(base["period_end"], errors="coerce")
    base["filed_date_dt"] = pd.to_datetime(base["filed_date"], errors="coerce")
    base = base.sort_values(
        ["fiscal_year", "period", "period_end_dt", "filed_date_dt"],
        ascending=[True, True, False, True],
    )
    base = base.drop_duplicates(["fiscal_year", "period"], keep="first")
    base = base.drop(columns=["period_end_dt", "filed_date_dt"])

    result = base
    for _, frame in field_frames:
        value_cols = [
            col
            for col in frame.columns
            if col not in {"period_end", "filed_date", "form"}
        ]
        result = result.merge(frame[value_cols], on=["fiscal_year", "period"], how="left")

    result.insert(0, "Ticker", ticker)
    return result


def cik_year_groups(item: dict[str, Any]) -> list[tuple[str, set[int] | None]]:
    cik_by_year = item.get("cik_by_year") or {}
    grouped: dict[str, set[int]] = {}
    for year_text, cik in cik_by_year.items():
        if not cik:
            continue
        grouped.setdefault(str(cik).zfill(10), set()).add(int(year_text))

    if len(grouped) <= 1:
        return [(str(item["cik"]).zfill(10), None)]
    return sorted(grouped.items(), key=lambda pair: min(pair[1]))


def run(start_year: int = config.START_YEAR) -> None:
    config.ensure_dirs()
    universe = load_universe()
    rows = []
    failures = []
    print(f"[step2] Collecting SEC companyfacts for {len(universe)} tickers")

    for item in tqdm(universe, desc="[step2] SEC"):
        ticker = item["ticker"]
        for cik, valid_years in cik_year_groups(item):
            try:
                facts = fetch_companyfacts(cik)
                df = parse_companyfacts(ticker, facts, start_year)
                if valid_years is not None:
                    df = df[df["fiscal_year"].isin(valid_years)]
                if df.empty:
                    failures.append({"ticker": ticker, "cik": cik, "reason": "no parsed facts"})
                    continue
                df["CIK"] = cik
                df["Sector"] = item.get("sector", "Unknown")
                df["Industry"] = item.get("industry", "Unknown")
                rows.append(df)
            except Exception as exc:
                failures.append({"ticker": ticker, "cik": cik, "reason": f"{type(exc).__name__}: {exc}"})

    if not rows:
        raise RuntimeError("No SEC fundamental data was collected")

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["Ticker", "fiscal_year", "period"]).reset_index(drop=True)
    result.to_csv(config.FUNDAMENTALS_CSV, index=False)
    pd.DataFrame(failures).to_csv(config.FUNDAMENTALS_FAILURES_CSV, index=False)

    print(f"[step2] rows: {len(result)}")
    print(f"[step2] tickers: {result['Ticker'].nunique()}")
    print(f"[step2] failures: {len(failures)}")
    print(f"[step2] saved: {config.FUNDAMENTALS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=config.START_YEAR)
    args = parser.parse_args()
    run(start_year=args.start_year)


if __name__ == "__main__":
    main()
