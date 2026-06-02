from __future__ import annotations

import argparse
import json
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import config


FEATURE_COLS = ["m1", "m2", "m3"]
FF5_EXPOSURE_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
FF5_REQUIRED_COLS = FF5_EXPOSURE_COLS + ["RF"]
MODEL_TYPES = ("mse", "ranknet")
SECTOR_SCOPES = ("all", "bio_it")


def rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 3:
        return 0.0
    value = pd.Series(pred).corr(pd.Series(target), method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if config.Y_TARGET_MODE == "sector":
        sector_median = df.groupby(["FiscalYear", "Sector"])["y"].transform("median")
        global_median = df.groupby("FiscalYear")["y"].transform("median")
        df["y_original"] = df["y"]
        df["Sector_Median"] = sector_median.fillna(global_median)
        df["y"] = df["y"] - df["Sector_Median"]
    return df


def filter_sector_scope(features: pd.DataFrame, sector_scope: str) -> pd.DataFrame:
    if sector_scope == "all":
        return features.copy()
    if sector_scope != "bio_it":
        raise ValueError(f"Unknown sector scope: {sector_scope}")

    sector_match = (
        features["Sector"].isin(config.BIO_IT_SECTORS)
        if "Sector" in features.columns
        else pd.Series(False, index=features.index)
    )
    industry_match = (
        features["Industry"].isin(config.BIO_IT_INDUSTRIES)
        if "Industry" in features.columns
        else pd.Series(False, index=features.index)
    )
    return features[sector_match | industry_match].copy()


def fit_ridge(train: pd.DataFrame, alpha: float) -> dict[str, object]:
    x = train[FEATURE_COLS].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    x_std = (x - mean) / std
    x_aug = np.column_stack([x_std, np.ones(len(x_std))])
    penalty = np.diag([alpha, alpha, alpha, 0.0])
    beta = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)
    return {"model_type": "mse", "beta": beta, "mean": mean, "std": std}


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50, 50)))


def sample_ranknet_pair_diffs(
    x_std: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    diffs = []
    for year in sorted(set(years.tolist())):
        idx = np.where(years == year)[0]
        if len(idx) < 2:
            continue
        n_pairs = min(config.RANKNET_MAX_PAIRS_PER_YEAR, len(idx) * (len(idx) - 1) // 2)
        if n_pairs <= 0:
            continue

        left = rng.choice(idx, size=n_pairs * 3, replace=True)
        right = rng.choice(idx, size=n_pairs * 3, replace=True)
        valid = y[left] != y[right]
        left = left[valid][:n_pairs]
        right = right[valid][:n_pairs]
        if len(left) == 0:
            continue

        higher = np.where(y[left] > y[right], left, right)
        lower = np.where(y[left] > y[right], right, left)
        diffs.append(x_std[higher] - x_std[lower])

    return np.vstack(diffs) if diffs else np.empty((0, x_std.shape[1]))


def fit_ranknet(train: pd.DataFrame) -> dict[str, object]:
    x = train[FEATURE_COLS].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    years = train["FiscalYear"].to_numpy(dtype=int)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    x_std = (x - mean) / std
    rng = np.random.default_rng(config.RANKNET_SEED)
    pair_diffs = sample_ranknet_pair_diffs(x_std, y, years, rng)

    weights = np.zeros(x_std.shape[1], dtype=float)
    final_loss = np.nan
    if len(pair_diffs):
        for _ in range(config.RANKNET_EPOCHS):
            margins = pair_diffs @ weights
            final_loss = float(np.logaddexp(0, -margins).mean() + 0.5 * config.RANKNET_L2 * weights @ weights)
            gradient = -(pair_diffs * sigmoid(-margins)[:, None]).mean(axis=0)
            gradient += config.RANKNET_L2 * weights
            weights -= config.RANKNET_LEARNING_RATE * gradient

    raw_train_scores = x_std @ weights
    calibration_x = np.column_stack([raw_train_scores, np.ones(len(raw_train_scores))])
    if np.std(raw_train_scores) < 1e-8:
        calibration = np.array([0.0, y.mean()])
    else:
        calibration, *_ = np.linalg.lstsq(calibration_x, y, rcond=None)

    return {
        "model_type": "ranknet",
        "mean": mean,
        "std": std,
        "weights": weights,
        "calibration_slope": float(calibration[0]),
        "calibration_intercept": float(calibration[1]),
        "ranknet_loss": final_loss,
        "n_ranknet_pairs": int(len(pair_diffs)),
    }


def fit_model(train: pd.DataFrame, model_type: str, alpha: float) -> dict[str, object]:
    if model_type == "mse":
        return fit_ridge(train, alpha)
    if model_type == "ranknet":
        return fit_ranknet(train)
    raise ValueError(f"Unknown model type: {model_type}")


def model_weights(model: dict[str, object]) -> tuple[float, float, float, float]:
    if model["model_type"] == "mse":
        beta = model["beta"]
        return float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])
    weights = model["weights"]
    return (
        float(weights[0]),
        float(weights[1]),
        float(weights[2]),
        float(model["calibration_intercept"]),
    )


def predict(model: dict[str, object], df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURE_COLS].to_numpy(dtype=float)
    x_std = (x - model["mean"]) / model["std"]
    if model["model_type"] == "mse":
        x_aug = np.column_stack([x_std, np.ones(len(x_std))])
        return x_aug @ model["beta"]

    raw_scores = x_std @ model["weights"]
    return raw_scores * model["calibration_slope"] + model["calibration_intercept"]


def run_predictions(
    features: pd.DataFrame,
    window: int,
    alpha: float,
    start_year: int,
    model_type: str = "mse",
    sector_scope: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = filter_sector_scope(features, sector_scope)
    features = prepare_target(features)
    years = sorted(features["FiscalYear"].astype(int).unique())
    if not years:
        return pd.DataFrame(), pd.DataFrame()
    first_prediction_year = max(min(years) + window, start_year)

    predictions = []
    metrics = []
    for test_year in years:
        if test_year < first_prediction_year:
            continue
        train = features[
            (features["FiscalYear"] >= test_year - window)
            & (features["FiscalYear"] <= test_year - 1)
        ].dropna(subset=FEATURE_COLS + ["y"])
        test = features[features["FiscalYear"] == test_year].dropna(subset=FEATURE_COLS + ["y"])
        if len(train) < 20 or len(test) < 5:
            continue

        model = fit_model(train, model_type, alpha)
        scores = predict(model, test)
        y_actual = test["y"].to_numpy(dtype=float)
        gaps = scores - y_actual
        w1, w2, w3, bias = model_weights(model)

        metrics.append(
            {
                "FiscalYear": test_year,
                "model_type": model_type,
                "sector_scope": sector_scope,
                "train_start": test_year - window,
                "train_end": test_year - 1,
                "n_train": len(train),
                "n_test": len(test),
                "rank_ic": rank_ic(scores, y_actual),
                "w1": w1,
                "w2": w2,
                "w3": w3,
                "bias": bias,
                "ranknet_loss": model.get("ranknet_loss", np.nan),
                "n_ranknet_pairs": model.get("n_ranknet_pairs", 0),
            }
        )

        for row, score, gap, y_val in zip(test.to_dict("records"), scores, gaps, y_actual):
            predictions.append(
                {
                    "Ticker": row["Ticker"],
                    "FiscalYear": test_year,
                    "model_type": model_type,
                    "sector_scope": sector_scope,
                    "Sector": row.get("Sector"),
                    "MarketCap": row.get("MarketCap"),
                    "m1": row.get("m1"),
                    "m2": row.get("m2"),
                    "m3": row.get("m3"),
                    "SP500MemberYear": row.get("SP500MemberYear"),
                    "SP500AsOfDate": row.get("SP500AsOfDate"),
                    "y_actual": y_val,
                    "score": float(score),
                    "gap": float(gap),
                }
            )

    return pd.DataFrame(predictions), pd.DataFrame(metrics)


def price_at_or_before(
    price_df: pd.DataFrame,
    date: pd.Timestamp,
    ticker: str,
    min_date: pd.Timestamp | None = None,
) -> tuple[float | None, pd.Timestamp | None]:
    if ticker not in price_df.columns:
        return None, None
    series = price_df[ticker].dropna()
    if min_date is not None:
        series = series[series.index >= min_date]
    series = series[series.index <= date]
    if series.empty:
        return None, None
    actual_date = series.index[-1]
    value = series.iloc[-1]
    if pd.isna(value) or value <= 0:
        return None, None
    return float(value), actual_date


def first_price_on_or_after(
    price_df: pd.DataFrame,
    date: pd.Timestamp,
    ticker: str,
) -> tuple[float | None, pd.Timestamp | None]:
    if ticker not in price_df.columns:
        return None, None
    series = price_df[ticker].dropna()
    series = series[series.index >= date]
    if series.empty:
        return None, None
    actual_date = series.index[0]
    value = series.iloc[0]
    if pd.isna(value) or value <= 0:
        return None, None
    return float(value), actual_date


def passes_price_rule(ticker: str, rebalance_date: pd.Timestamp, price_df: pd.DataFrame) -> bool:
    if ticker not in price_df.columns:
        return False
    recent = price_df.loc[rebalance_date - pd.DateOffset(months=12) : rebalance_date, ticker].dropna()
    if recent.empty:
        return False
    return bool((recent <= config.PRICE_THRESHOLD).mean() < config.PRICE_BELOW_RATIO)


def position_return_detail(
    ticker: str,
    rebalance_date: pd.Timestamp,
    price_df: pd.DataFrame,
) -> dict[str, object]:
    sell_date = rebalance_date + pd.DateOffset(months=config.HOLDING_MONTHS)
    price_history_end = pd.to_datetime(price_df.index).max()
    if pd.notna(price_history_end) and sell_date > price_history_end:
        return {
            "Ticker": ticker,
            "buy_date": None,
            "sell_date": None,
            "target_sell_date": sell_date.date().isoformat(),
            "buy_price": np.nan,
            "sell_price": np.nan,
            "position_return": np.nan,
            "price_status": "target_sell_date_beyond_price_history",
        }

    buy, buy_date = price_at_or_before(price_df, rebalance_date, ticker)
    if buy is None or buy_date is None:
        return {
            "Ticker": ticker,
            "buy_date": None,
            "sell_date": None,
            "buy_price": np.nan,
            "sell_price": np.nan,
            "position_return": np.nan,
            "price_status": "no_buy_price",
        }

    buy_staleness = (rebalance_date - buy_date).days
    if buy_staleness > config.BUY_PRICE_MAX_STALENESS_DAYS:
        return {
            "Ticker": ticker,
            "buy_date": buy_date.date().isoformat(),
            "sell_date": None,
            "buy_price": buy,
            "sell_price": np.nan,
            "position_return": np.nan,
            "price_status": "stale_buy_price",
        }

    sell, actual_sell_date = first_price_on_or_after(price_df, sell_date, ticker)
    if sell is None or actual_sell_date is None:
        return {
            "Ticker": ticker,
            "buy_date": buy_date.date().isoformat(),
            "sell_date": None,
            "target_sell_date": sell_date.date().isoformat(),
            "buy_price": buy,
            "sell_price": 0.0,
            "position_return": -1.0,
            "price_status": "delisted_no_sell_price_zero",
        }

    status = "normal"
    if actual_sell_date > sell_date:
        status = "next_available_after_sell_date"

    return {
        "Ticker": ticker,
        "buy_date": buy_date.date().isoformat(),
        "sell_date": actual_sell_date.date().isoformat(),
        "target_sell_date": sell_date.date().isoformat(),
        "buy_price": buy,
        "sell_price": sell,
        "position_return": sell / buy - 1.0,
        "price_status": status,
    }


def portfolio_return_with_details(
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    price_df: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    details = [position_return_detail(ticker, rebalance_date, price_df) for ticker in tickers]
    detail_df = pd.DataFrame(details)
    returns = detail_df["position_return"].dropna() if "position_return" in detail_df.columns else pd.Series(dtype=float)
    return (float(returns.mean()) if len(returns) else np.nan), detail_df


def portfolio_return(tickers: list[str], rebalance_date: pd.Timestamp, price_df: pd.DataFrame) -> float:
    ret, _ = portfolio_return_with_details(tickers, rebalance_date, price_df)
    return ret


def build_monthly_portfolio_returns(
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    target_sell_date = rebalance_date + pd.DateOffset(months=config.HOLDING_MONTHS)
    price_history_end = pd.to_datetime(price_df.index).max()
    if pd.isna(price_history_end) or target_sell_date > price_history_end:
        return pd.DataFrame()

    month_ends = pd.date_range(
        start=rebalance_date + pd.offsets.MonthEnd(1),
        end=target_sell_date,
        freq="ME",
    )
    if len(month_ends) == 0:
        return pd.DataFrame()

    positions: dict[str, dict[str, object]] = {}
    for ticker in tickers:
        buy, buy_date = price_at_or_before(price_df, rebalance_date, ticker)
        if buy is None or buy_date is None:
            continue
        if (rebalance_date - buy_date).days > config.BUY_PRICE_MAX_STALENESS_DAYS:
            continue
        series = price_df[ticker].dropna().sort_index() if ticker in price_df.columns else pd.Series(dtype=float)
        positions[ticker] = {
            "active": True,
            "last_date": buy_date,
            "last_price": buy,
            "series": series,
            "value": 1.0,
        }

    if not positions:
        return pd.DataFrame()

    start_weight = 1.0 / len(positions)
    for position in positions.values():
        position["value"] = start_weight

    rows = []
    for month_end in month_ends:
        start_value = float(sum(position["value"] for position in positions.values() if position["active"]))
        if start_value <= 0:
            break

        carried_missing = 0
        terminal_zero = 0
        for position in positions.values():
            if not position["active"]:
                continue
            series = position["series"]
            last_date = position["last_date"]
            last_price = position["last_price"]
            current_prices = series[(series.index > last_date) & (series.index <= month_end)]
            if not current_prices.empty:
                current_date = current_prices.index[-1]
                current_price = float(current_prices.iloc[-1])
                position["value"] = float(position["value"]) * (current_price / float(last_price))
                position["last_date"] = current_date
                position["last_price"] = current_price
                continue

            future_prices = series[series.index > month_end]
            if future_prices.empty:
                position["value"] = 0.0
                position["active"] = False
                terminal_zero += 1
            else:
                carried_missing += 1

        end_value = float(sum(position["value"] for position in positions.values() if position["active"]))
        monthly_return = end_value / start_value - 1.0 if start_value > 0 else np.nan
        rows.append(
            {
                "MonthEnd": month_end.date().isoformat(),
                "monthly_return": monthly_return,
                "portfolio_value": end_value,
                "n_start_positions": len(positions),
                "n_active_positions": int(sum(1 for position in positions.values() if position["active"])),
                "n_carried_missing": carried_missing,
                "n_terminal_zero": terminal_zero,
            }
        )

    return pd.DataFrame(rows)


def normalize_factor_column_name(name: object) -> str:
    normalized = str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    mapping = {
        "mktrf": "MKT_RF",
        "mkt": "MKT_RF",
        "market": "MKT_RF",
        "smb": "SMB",
        "hml": "HML",
        "rmw": "RMW",
        "cma": "CMA",
        "rf": "RF",
        "riskfree": "RF",
    }
    return mapping.get(normalized, str(name).strip())


def parse_factor_month(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    is_yyyymm = compact.str.match(r"^\d{6}$")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if is_yyyymm.any():
        parsed.loc[is_yyyymm] = pd.to_datetime(compact[is_yyyymm], format="%Y%m", errors="coerce")
    if (~is_yyyymm).any():
        parsed.loc[~is_yyyymm] = pd.to_datetime(text[~is_yyyymm], errors="coerce")
    return parsed + pd.offsets.MonthEnd(0)


def read_ff5_factor_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    header_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if "Mkt-RF" in line and "SMB" in line and "HML" in line and "RMW" in line and "CMA" in line
        ),
        None,
    )
    if header_idx is None:
        return pd.read_csv(path)

    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        first_value = line.split(",", 1)[0].strip()
        if first_value.isdigit() and len(first_value) == 6:
            data_lines.append(line)
            continue
        break
    return pd.read_csv(StringIO("\n".join(data_lines)))


def load_ff5_factors(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    raw = read_ff5_factor_csv(path)
    raw = raw.rename(columns={column: normalize_factor_column_name(column) for column in raw.columns})
    date_col = next((column for column in raw.columns if column not in FF5_REQUIRED_COLS), raw.columns[0])
    raw["MonthEnd"] = parse_factor_month(raw[date_col])

    missing = [column for column in FF5_REQUIRED_COLS if column not in raw.columns]
    if missing:
        raise ValueError(f"FF5 factor file is missing columns: {missing}")

    factors = raw[["MonthEnd"] + FF5_REQUIRED_COLS].copy()
    for column in FF5_REQUIRED_COLS:
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    factors = factors.dropna(subset=["MonthEnd"] + FF5_REQUIRED_COLS).sort_values("MonthEnd")

    factor_max = factors[FF5_REQUIRED_COLS].abs().max().max()
    if pd.notna(factor_max) and factor_max > 1.0:
        factors[FF5_REQUIRED_COLS] = factors[FF5_REQUIRED_COLS] / 100.0
    return factors


def run_ff5_alpha(
    monthly_returns: pd.DataFrame,
    factor_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if monthly_returns.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "status": "no_monthly_portfolio_returns",
                        "factor_file": str(factor_path.relative_to(config.ROOT_DIR)),
                    }
                ]
            ),
            pd.DataFrame(),
        )

    factors = load_ff5_factors(factor_path)
    if factors.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "status": "missing_factor_file",
                        "factor_file": str(factor_path.relative_to(config.ROOT_DIR)),
                    }
                ]
            ),
            pd.DataFrame(),
        )

    monthly = monthly_returns.copy()
    monthly["MonthEnd"] = pd.to_datetime(monthly["MonthEnd"]) + pd.offsets.MonthEnd(0)
    regression_data = monthly.merge(factors, on="MonthEnd", how="inner")
    if regression_data.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "status": "no_overlapping_factor_months",
                        "factor_file": str(factor_path.relative_to(config.ROOT_DIR)),
                    }
                ]
            ),
            regression_data,
        )

    regression_data["excess_return"] = regression_data["monthly_return"] - regression_data["RF"]
    rows = []
    for strategy, group in regression_data.groupby("Strategy"):
        group = group.dropna(subset=["excess_return"] + FF5_EXPOSURE_COLS)
        n_obs = len(group)
        n_params = len(FF5_EXPOSURE_COLS) + 1
        if n_obs <= n_params:
            rows.append(
                {
                    "Strategy": strategy,
                    "status": "insufficient_observations",
                    "n_obs": n_obs,
                    "required_obs": n_params + 1,
                }
            )
            continue

        y = group["excess_return"].to_numpy(dtype=float)
        x = group[FF5_EXPOSURE_COLS].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residuals = y - fitted
        dof = n_obs - x.shape[1]
        sigma2 = float(residuals.T @ residuals / dof)
        covariance = sigma2 * np.linalg.pinv(x.T @ x)
        standard_errors = np.sqrt(np.diag(covariance))
        alpha_monthly = float(beta[0])
        alpha_se = float(standard_errors[0]) if standard_errors[0] > 0 else np.nan
        ss_total = float(((y - y.mean()) ** 2).sum())
        ss_resid = float((residuals**2).sum())
        r_squared = 1.0 - ss_resid / ss_total if ss_total > 0 else np.nan

        rows.append(
            {
                "Strategy": strategy,
                "status": "ok",
                "n_obs": n_obs,
                "alpha_monthly": alpha_monthly,
                "alpha_annualized": float((1 + alpha_monthly) ** 12 - 1),
                "alpha_tstat": float(alpha_monthly / alpha_se) if alpha_se and not pd.isna(alpha_se) else np.nan,
                "beta_mkt_rf": float(beta[1]),
                "beta_smb": float(beta[2]),
                "beta_hml": float(beta[3]),
                "beta_rmw": float(beta[4]),
                "beta_cma": float(beta[5]),
                "mean_monthly_return": float(group["monthly_return"].mean()),
                "mean_monthly_excess_return": float(group["excess_return"].mean()),
                "r_squared": float(r_squared) if not pd.isna(r_squared) else np.nan,
            }
        )

    return pd.DataFrame(rows), regression_data


def run_backtest(
    predictions: pd.DataFrame,
    price_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty or "FiscalYear" not in predictions.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rows = []
    holdings = []
    monthly_rows = []
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    price_history_end = price_df.index.max()

    for year, year_df in predictions.groupby("FiscalYear"):
        rebalance_date = pd.Timestamp(year=int(year) + 1, month=3, day=31)
        target_sell_date = rebalance_date + pd.DateOffset(months=config.HOLDING_MONTHS)
        if target_sell_date > price_history_end:
            continue

        ranked = year_df.sort_values("gap", ascending=False)
        valid = [ticker for ticker in ranked["Ticker"].tolist() if passes_price_rule(ticker, rebalance_date, price_df)]
        if not valid:
            continue

        for ratio in config.LONG_RATIOS:
            n_select = max(1, int(len(valid) * ratio))
            selected = valid[:n_select]
            ret, price_details = portfolio_return_with_details(selected, rebalance_date, price_df)
            strategy = f"gap_top_{int(ratio * 100)}pct"
            rows.append(
                {
                    "FiscalYear": int(year),
                    "RebalanceDate": rebalance_date.date().isoformat(),
                    "Strategy": strategy,
                    "n_selected": len(selected),
                    "n_priced": int(price_details["position_return"].notna().sum()) if not price_details.empty else 0,
                    "n_delisted_zero": int(
                        (price_details["price_status"] == "delisted_no_sell_price_zero").sum()
                    )
                    if not price_details.empty
                    else 0,
                    "n_price_missing": int(price_details["position_return"].isna().sum()) if not price_details.empty else 0,
                    "return": ret,
                }
            )
            selected_rows = ranked[ranked["Ticker"].isin(selected)].copy()
            selected_rows["Strategy"] = strategy
            selected_rows["RebalanceDate"] = rebalance_date.date().isoformat()
            selected_rows = selected_rows.merge(price_details, on="Ticker", how="left")
            holdings.append(selected_rows)

            monthly = build_monthly_portfolio_returns(selected, rebalance_date, price_df)
            if not monthly.empty:
                monthly["FiscalYear"] = int(year)
                monthly["RebalanceDate"] = rebalance_date.date().isoformat()
                monthly["Strategy"] = strategy
                monthly_rows.append(monthly)

    returns = pd.DataFrame(rows)
    holdings_df = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    monthly_returns = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    summary = summarize_returns(returns)
    return returns, summary, holdings_df, monthly_returns


def summarize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    rows = []
    for strategy, group in returns.dropna(subset=["return"]).groupby("Strategy"):
        values = group["return"].to_numpy(dtype=float)
        cumulative = float(np.prod(1 + values) - 1)
        cagr = float(np.prod(1 + values) ** (1 / len(values)) - 1)
        equity = np.cumprod(1 + values)
        peak = np.maximum.accumulate(equity)
        mdd = float(((equity - peak) / peak).min())
        sharpe = float(values.mean() / (values.std(ddof=1) + 1e-8))
        rows.append(
            {
                "Strategy": strategy,
                "years": len(values),
                "mean_return": float(values.mean()),
                "cumulative_return": cumulative,
                "cagr": cagr,
                "mdd": mdd,
                "sharpe": sharpe,
                "avg_selected": float(group["n_selected"].mean()),
                "avg_priced": float(group["n_priced"].mean()) if "n_priced" in group.columns else np.nan,
                "total_delisted_zero": int(group["n_delisted_zero"].sum())
                if "n_delisted_zero" in group.columns
                else 0,
                "total_price_missing": int(group["n_price_missing"].sum()) if "n_price_missing" in group.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    exp_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    ff5_summary: pd.DataFrame,
    model_type: str,
    sector_scope: str,
    window: int,
    alpha: float,
) -> None:
    report = [
        "# Capstone Experiment Report",
        "",
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- model_type: {model_type}",
        f"- sector_scope: {sector_scope}",
        f"- train_window_years: {window}",
        f"- ridge_alpha: {alpha}",
        f"- ranknet_epochs: {config.RANKNET_EPOCHS}",
        f"- ranknet_max_pairs_per_year: {config.RANKNET_MAX_PAIRS_PER_YEAR}",
        f"- y_target_mode: {config.Y_TARGET_MODE}",
        f"- bio_it_definition: Sector in {config.BIO_IT_SECTORS} or Industry in {config.BIO_IT_INDUSTRIES}",
        f"- membership_rule: selected from FiscalYear S&P 500 members; held positions are not removed just because the ticker exits S&P 500 later",
        f"- evaluation_policy: only completed holding windows are evaluated; incomplete final windows are excluded",
        f"- delisting_policy: S&P 500 exits keep using available market prices; missing sell price inside the available price-history horizon is treated as -100%",
        f"- evaluation_tracks: raw annual backtest plus optional Fama-French 5-factor alpha",
        f"- ff5_factor_file: {config.FF5_MONTHLY_FACTORS_CSV.relative_to(config.ROOT_DIR)}",
        "",
        "## Training",
        "",
    ]
    if metrics.empty:
        report.append("No prediction metrics were generated.")
    else:
        report.append(f"- prediction_years: {int(metrics['FiscalYear'].min())}-{int(metrics['FiscalYear'].max())}")
        report.append(f"- avg_rank_ic: {metrics['rank_ic'].mean():.4f}")
    report.extend(["", "## Backtest", ""])
    if summary.empty:
        report.append("No backtest summary was generated.")
    else:
        report.append("```text")
        report.append(summary.to_string(index=False))
        report.append("```")
    report.extend(["", "## Monthly Portfolio Returns", ""])
    if monthly_returns.empty:
        report.append("No monthly portfolio return series was generated.")
    else:
        report.append(f"- rows: {len(monthly_returns)}")
        report.append(
            f"- months: {monthly_returns['MonthEnd'].min()} to {monthly_returns['MonthEnd'].max()}"
        )
        report.append("- file: `monthly_portfolio_returns.csv`")
    report.extend(["", "## Fama-French 5-Factor Alpha", ""])
    if ff5_summary.empty:
        report.append("No FF5 alpha summary was generated.")
    else:
        report.append("```text")
        report.append(ff5_summary.to_string(index=False))
        report.append("```")
    (exp_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def execute_experiment(
    name: str,
    window: int,
    alpha: float,
    start_year: int,
    model_type: str,
    sector_scope: str,
) -> dict[str, object]:
    config.ensure_dirs()
    if not config.FEATURES_ANNUAL_CSV.exists():
        raise FileNotFoundError(f"Missing features: {config.FEATURES_ANNUAL_CSV}")
    if not config.PRICE_MONTHLY_CLOSE_CSV.exists():
        raise FileNotFoundError(f"Missing prices: {config.PRICE_MONTHLY_CLOSE_CSV}")

    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}: {model_type}")
    if sector_scope not in SECTOR_SCOPES:
        raise ValueError(f"sector_scope must be one of {SECTOR_SCOPES}: {sector_scope}")

    exp_dir = config.EXPERIMENTS_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(config.FEATURES_ANNUAL_CSV)
    price_df = pd.read_csv(config.PRICE_MONTHLY_CLOSE_CSV, index_col=0, parse_dates=True)
    predictions, metrics = run_predictions(
        features,
        window=window,
        alpha=alpha,
        start_year=start_year,
        model_type=model_type,
        sector_scope=sector_scope,
    )
    returns, summary, holdings, monthly_returns = run_backtest(predictions, price_df)
    ff5_summary, ff5_regression_data = run_ff5_alpha(monthly_returns, config.FF5_MONTHLY_FACTORS_CSV)

    predictions.to_csv(exp_dir / "predictions.csv", index=False)
    metrics.to_csv(exp_dir / "training_metrics.csv", index=False)
    returns.to_csv(exp_dir / "backtest_returns.csv", index=False)
    summary.to_csv(exp_dir / "backtest_summary.csv", index=False)
    holdings.to_csv(exp_dir / "holdings.csv", index=False)
    monthly_returns.to_csv(exp_dir / "monthly_portfolio_returns.csv", index=False)
    ff5_summary.to_csv(exp_dir / "ff5_alpha_summary.csv", index=False)
    ff5_regression_data.to_csv(exp_dir / "ff5_regression_data.csv", index=False)
    write_report(
        exp_dir,
        metrics,
        summary,
        monthly_returns,
        ff5_summary,
        model_type=model_type,
        sector_scope=sector_scope,
        window=window,
        alpha=alpha,
    )
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "window": window,
                "alpha": alpha,
                "start_year": start_year,
                "model_type": model_type,
                "sector_scope": sector_scope,
                "bio_it_sectors": config.BIO_IT_SECTORS,
                "bio_it_industries": config.BIO_IT_INDUSTRIES,
                "target_mode": config.Y_TARGET_MODE,
                "long_ratios": config.LONG_RATIOS,
                "delisting_policy": "sp500_exit_uses_market_price_missing_sell_price_minus_100pct",
                "buy_price_max_staleness_days": config.BUY_PRICE_MAX_STALENESS_DAYS,
                "ranknet_learning_rate": config.RANKNET_LEARNING_RATE,
                "ranknet_epochs": config.RANKNET_EPOCHS,
                "ranknet_l2": config.RANKNET_L2,
                "ranknet_max_pairs_per_year": config.RANKNET_MAX_PAIRS_PER_YEAR,
                "ranknet_seed": config.RANKNET_SEED,
                "evaluation_tracks": ["raw_annual_backtest", "fama_french_5_factor_alpha"],
                "ff5_factor_file": str(config.FF5_MONTHLY_FACTORS_CSV.relative_to(config.ROOT_DIR)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"experiment saved: {exp_dir}")
    if not summary.empty:
        print(summary.to_string(index=False))
    return {
        "name": name,
        "exp_dir": str(exp_dir),
        "model_type": model_type,
        "sector_scope": sector_scope,
        "window": window,
        "n_predictions": len(predictions),
        "n_return_rows": len(returns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline_mse_all_w5")
    parser.add_argument("--window", type=int, default=config.TRAIN_WINDOW_YEARS)
    parser.add_argument("--alpha", type=float, default=config.RIDGE_ALPHA)
    parser.add_argument("--start-year", type=int, default=config.BACKTEST_START_YEAR)
    parser.add_argument("--model", choices=MODEL_TYPES, default="mse")
    parser.add_argument("--sector-scope", choices=SECTOR_SCOPES, default="all")
    args = parser.parse_args()

    execute_experiment(
        name=args.name,
        window=args.window,
        alpha=args.alpha,
        start_year=args.start_year,
        model_type=args.model,
        sector_scope=args.sector_scope,
    )


if __name__ == "__main__":
    main()
