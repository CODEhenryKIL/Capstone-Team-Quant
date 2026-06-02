from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config


REQUIRED_FOR_STEP3 = [
    ("membership", config.SP500_MEMBERSHIP_CSV),
    ("monthly prices", config.PRICE_MONTHLY_CLOSE_CSV),
    ("fundamentals", config.FUNDAMENTALS_CSV),
]

REQUIRED_FOR_EXPERIMENT = [
    ("features annual", config.FEATURES_ANNUAL_CSV),
    ("monthly prices", config.PRICE_MONTHLY_CLOSE_CSV),
]

OPTIONAL_FOR_FF5 = [
    ("FF5 monthly factors", config.FF5_MONTHLY_FACTORS_CSV),
]


def describe_csv(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path == config.FF5_MONTHLY_FACTORS_CSV:
        try:
            from run_experiment import load_ff5_factors

            df = load_ff5_factors(path)
        except Exception as exc:
            return f"unreadable ({type(exc).__name__}: {exc})"
        if df.empty:
            return "empty"
        return (
            f"ok rows={len(df)} cols={len(df.columns)}, "
            f"months={df['MonthEnd'].min().date()}-{df['MonthEnd'].max().date()}"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return f"unreadable ({type(exc).__name__}: {exc})"
    if "FiscalYear" in df.columns:
        year_info = f", years={int(df['FiscalYear'].min())}-{int(df['FiscalYear'].max())}"
    elif "year" in df.columns:
        year_info = f", years={int(df['year'].min())}-{int(df['year'].max())}"
    else:
        year_info = ""
    ticker_info = f", tickers={df['Ticker'].nunique()}" if "Ticker" in df.columns else ""
    return f"ok rows={len(df)} cols={len(df.columns)}{ticker_info}{year_info}"


def check_group(title: str, items: list[tuple[str, Path]]) -> bool:
    print(f"\n[{title}]")
    ok = True
    for label, path in items:
        status = describe_csv(path)
        exists = status.startswith("ok")
        ok = ok and exists
        mark = "OK" if exists else "NO"
        print(f"{mark:>2} {label:<18} {path.relative_to(config.ROOT_DIR)} :: {status}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when experiment inputs are missing")
    args = parser.parse_args()

    config.ensure_dirs()
    step3_ready = check_group("needed to build features", REQUIRED_FOR_STEP3)
    experiment_ready = check_group("needed to run experiments", REQUIRED_FOR_EXPERIMENT)
    check_group("optional for Fama-French alpha", OPTIONAL_FOR_FF5)

    print("\n[summary]")
    print(f"step3 ready: {step3_ready}")
    print(f"experiment ready: {experiment_ready}")
    if args.strict and not experiment_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
