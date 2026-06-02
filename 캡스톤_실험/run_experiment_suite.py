from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import config
from run_experiment import execute_experiment


BASE_EXPERIMENTS = [
    {"model_type": "mse", "sector_scope": "all"},
    {"model_type": "mse", "sector_scope": "bio_it"},
    {"model_type": "ranknet", "sector_scope": "all"},
    {"model_type": "ranknet", "sector_scope": "bio_it"},
]
WINDOWS = [5, 3]


def experiment_name(model_type: str, sector_scope: str, window: int) -> str:
    return f"sp500_{model_type}_{sector_scope}_w{window}"


def read_csv_if_exists(path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def collect_suite_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    backtest_frames = []
    ff5_frames = []
    for base in BASE_EXPERIMENTS:
        for window in WINDOWS:
            name = experiment_name(base["model_type"], base["sector_scope"], window)
            exp_dir = config.EXPERIMENTS_DIR / name
            metadata = {
                "Experiment": name,
                "Universe": "S&P500",
                "Model": base["model_type"],
                "SectorScope": base["sector_scope"],
                "WindowYears": window,
            }

            backtest = read_csv_if_exists(exp_dir / "backtest_summary.csv")
            if not backtest.empty:
                for key, value in metadata.items():
                    backtest[key] = value
                backtest_frames.append(backtest)

            ff5 = read_csv_if_exists(exp_dir / "ff5_alpha_summary.csv")
            if not ff5.empty:
                for key, value in metadata.items():
                    ff5[key] = value
                ff5_frames.append(ff5)

    backtest_summary = pd.concat(backtest_frames, ignore_index=True) if backtest_frames else pd.DataFrame()
    ff5_summary = pd.concat(ff5_frames, ignore_index=True) if ff5_frames else pd.DataFrame()
    return backtest_summary, ff5_summary


def run_suite(start_year: int, alpha: float) -> None:
    config.ensure_dirs()
    run_rows = []
    for base in BASE_EXPERIMENTS:
        for window in WINDOWS:
            name = experiment_name(base["model_type"], base["sector_scope"], window)
            print(f"\n[suite] running {name}")
            result = execute_experiment(
                name=name,
                window=window,
                alpha=alpha,
                start_year=start_year,
                model_type=base["model_type"],
                sector_scope=base["sector_scope"],
            )
            run_rows.append(result)

    backtest_summary, ff5_summary = collect_suite_outputs()
    run_log = pd.DataFrame(run_rows)
    run_log["created_at"] = datetime.now().isoformat(timespec="seconds")

    run_log.to_csv(config.EXPERIMENTS_DIR / "capstone_8_run_log.csv", index=False)
    backtest_summary.to_csv(config.EXPERIMENTS_DIR / "capstone_8_backtest_summary.csv", index=False)
    ff5_summary.to_csv(config.EXPERIMENTS_DIR / "capstone_8_ff5_alpha_summary.csv", index=False)

    print("\n[suite] saved aggregate summaries:")
    print(f"- {config.EXPERIMENTS_DIR / 'capstone_8_run_log.csv'}")
    print(f"- {config.EXPERIMENTS_DIR / 'capstone_8_backtest_summary.csv'}")
    print(f"- {config.EXPERIMENTS_DIR / 'capstone_8_ff5_alpha_summary.csv'}")
    if not backtest_summary.empty:
        cols = ["Experiment", "Strategy", "cagr", "mdd", "sharpe", "years"]
        print(backtest_summary[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 8 capstone experiment variants")
    parser.add_argument("--start-year", type=int, default=config.BACKTEST_START_YEAR)
    parser.add_argument("--alpha", type=float, default=config.RIDGE_ALPHA)
    args = parser.parse_args()
    run_suite(start_year=args.start_year, alpha=args.alpha)


if __name__ == "__main__":
    main()
