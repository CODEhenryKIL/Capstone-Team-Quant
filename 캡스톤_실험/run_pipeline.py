from __future__ import annotations

import argparse
from datetime import datetime

import config
from src import step0_sp500_universe, step1_prices, step2_sec_fundamentals, step3_features, step4_ff_factors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capstone S&P 500 dataset pipeline")
    parser.add_argument(
        "--steps",
        default="01234",
        help="Steps to run. Examples: 0, 01, 0123, 4, 01234",
    )
    parser.add_argument("--start-year", type=int, default=config.START_YEAR)
    parser.add_argument("--end-year", type=int, default=config.END_YEAR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--refresh-step0", action="store_true", help="Refetch Wikipedia/SEC source data for step0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    if "0" in args.steps:
        step0_sp500_universe.run(
            start_year=args.start_year,
            end_year=args.end_year,
            refresh=args.refresh_step0,
        )
    if "1" in args.steps:
        step1_prices.run(
            start_date=args.start_date or f"{args.start_year}-01-01",
            end_date=args.end_date,
        )
    if "2" in args.steps:
        step2_sec_fundamentals.run(start_year=args.start_year)
    if "3" in args.steps:
        step3_features.run()
    if "4" in args.steps:
        step4_ff_factors.run()


if __name__ == "__main__":
    main()
