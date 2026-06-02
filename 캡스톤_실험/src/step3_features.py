from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

import config


PERIOD_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4}


def calculate_safe_lag_date(
    period_end: pd.Series,
    filed_date: pd.Series,
    lag_months: int = config.SAFE_LAG_MONTHS,
) -> pd.Series:
    base = pd.to_datetime(period_end, utc=True) + pd.DateOffset(months=lag_months)
    valuation = base + MonthEnd(0)
    filed = pd.to_datetime(filed_date, utc=True) + MonthEnd(0)
    return pd.to_datetime(np.maximum(valuation.values, filed.values), utc=True).tz_convert(None)


def match_monthly_price(df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    prices = price_df.copy()
    prices.index = pd.to_datetime(prices.index)
    prices.index = prices.index + MonthEnd(0)
    stacked = prices.stack().reset_index()
    stacked.columns = ["ValuationDate", "Ticker", "Close"]
    stacked["ValuationDate"] = pd.to_datetime(stacked["ValuationDate"])

    merged = df.merge(stacked, on=["Ticker", "ValuationDate"], how="left")
    return merged


def restate_annual(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in df.groupby("Ticker", sort=False):
        group = group.sort_values("fiscal_year").copy()
        investment = group["RD"].fillna(0) + 0.3 * group["SGA"].fillna(0)
        k_values = []
        amortization = []
        k_prev = None
        for z_t in investment:
            if k_prev is None:
                k_prev = z_t / 0.33 if 0.33 else z_t
                k_values.append(k_prev)
                amortization.append(z_t)
                continue
            a_t = 0.33 * k_prev
            k_t = (1 - 0.33) * k_prev + z_t
            k_values.append(k_t)
            amortization.append(a_t)
            k_prev = k_t

        group["Investment"] = investment
        group["IntangibleCapital"] = k_values
        group["Amortization"] = amortization
        group["AdjEquity"] = group["Equity"] + group["IntangibleCapital"]
        group["AdjOI"] = group["OI"] + group["Investment"] - group["Amortization"]
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def calculate_m3(group: pd.DataFrame) -> pd.Series:
    inv_t1 = group["Investment"].shift(1)
    inv_t2 = group["Investment"].shift(2)
    avg_inv = (inv_t1 + inv_t2) / 2
    return group["Investment"] / avg_inv.replace(0, np.nan)


def filter_membership(features: pd.DataFrame) -> pd.DataFrame:
    if not config.SP500_MEMBERSHIP_CSV.exists():
        return features
    membership = pd.read_csv(config.SP500_MEMBERSHIP_CSV)
    if "in_dataset_universe" in membership.columns:
        membership = membership[membership["in_dataset_universe"].astype(bool)]
    cols = ["year", "ticker"]
    if "as_of_date" in membership.columns:
        cols.append("as_of_date")
    keys = membership[cols].drop_duplicates()
    keys["year"] = keys["year"].astype(int)
    keys["ticker"] = keys["ticker"].astype(str)

    before = len(features)
    out = features.merge(
        keys,
        left_on=["FiscalYear", "Ticker"],
        right_on=["year", "ticker"],
        how="inner",
    )
    out["SP500MemberYear"] = out["year"]
    if "as_of_date" in out.columns:
        out["SP500AsOfDate"] = out["as_of_date"]
        out = out.drop(columns=["as_of_date"])
    else:
        out["SP500AsOfDate"] = pd.NA
    out = out.drop(columns=["year", "ticker"])
    print(f"[step3] S&P membership filter: {before} -> {len(out)} rows")
    return out


def run() -> None:
    config.ensure_dirs()
    if not config.FUNDAMENTALS_CSV.exists():
        raise FileNotFoundError(f"Run step2 first: {config.FUNDAMENTALS_CSV}")
    if not config.PRICE_MONTHLY_CLOSE_CSV.exists():
        raise FileNotFoundError(f"Run step1 first: {config.PRICE_MONTHLY_CLOSE_CSV}")

    fundamentals = pd.read_csv(config.FUNDAMENTALS_CSV)
    prices = pd.read_csv(config.PRICE_MONTHLY_CLOSE_CSV, index_col=0, parse_dates=True)

    df = fundamentals[fundamentals["period"] == "FY"].copy()
    df = df.sort_values(["Ticker", "fiscal_year"])
    df = restate_annual(df)
    df["ValuationDate"] = calculate_safe_lag_date(df["period_end"], df["filed_date"])

    merged = match_monthly_price(df, prices)
    merged["MarketCap"] = merged["Close"] * merged["Shares"]
    merged["PBR"] = merged["MarketCap"] / merged["AdjEquity"]
    merged["PBR_Clipped"] = merged["PBR"].clip(0.01, 100.0)
    merged["y"] = np.log(merged["PBR_Clipped"])
    merged["m1"] = (merged["NetIncome"] / merged["AdjEquity"]).clip(-5.0, 5.0)
    merged["m2"] = (merged["AdjOI"] / np.maximum(merged["OI"].abs(), 1e-6)).clip(-5.0, 5.0)
    merged["m3"] = merged.groupby("Ticker", group_keys=False).apply(calculate_m3).clip(0.0, 10.0)

    final = merged[
        [
            "Ticker",
            "CIK",
            "fiscal_year",
            "period_end",
            "filed_date",
            "ValuationDate",
            "Sector",
            "Industry",
            "m1",
            "m2",
            "m3",
            "y",
            "Close",
            "MarketCap",
            "PBR",
            "AdjEquity",
            "AdjOI",
            "Revenue",
            "OI",
            "NetIncome",
            "RD",
            "SGA",
            "Shares",
        ]
    ].rename(
        columns={
            "fiscal_year": "FiscalYear",
            "period_end": "EndDate",
        }
    )

    final = filter_membership(final)
    before_drop = len(final)
    final = final.dropna(subset=["m1", "m2", "m3", "y", "Close"])
    print(f"[step3] model-ready dropna: {before_drop} -> {len(final)} rows")

    final.to_csv(config.FEATURES_ANNUAL_CSV, index=False)
    final[["Ticker", "FiscalYear", "Sector", "m1", "m2", "m3", "y"]].to_csv(
        config.MODEL_DATASET_ANNUAL_CSV,
        index=False,
    )
    print(f"[step3] saved: {config.FEATURES_ANNUAL_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
