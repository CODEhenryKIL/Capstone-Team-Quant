from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config
from run_experiment import FF5_EXPOSURE_COLS, load_ff5_factors


REPORT_DIR = config.ROOT_DIR / "reports"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
EXPERIMENTS = [
    "sp500_mse_all_w5",
    "sp500_mse_all_w3",
    "sp500_mse_bio_it_w5",
    "sp500_mse_bio_it_w3",
    "sp500_ranknet_all_w5",
    "sp500_ranknet_all_w3",
    "sp500_ranknet_bio_it_w5",
    "sp500_ranknet_bio_it_w3",
]
STRATEGY_ORDER = ["gap_top_5pct", "gap_top_10pct", "gap_top_20pct", "gap_top_30pct"]
MODEL_LABELS = {"mse": "MSE", "ranknet": "RankNet", "MSE": "MSE", "RankNet": "RankNet"}
SECTOR_SCOPE_LABELS = {"all": "전체", "bio_it": "바이오_IT", "전체": "전체", "바이오_IT": "바이오_IT"}
STRATEGY_LABELS = {
    "gap_top_5pct": "상위 5%",
    "gap_top_10pct": "상위 10%",
    "gap_top_20pct": "상위 20%",
    "gap_top_30pct": "상위 30%",
}
FIGURE_LABELS = {
    "annual_mean_return": "연평균 수익률",
    "annual_sharpe": "연간 샤프 비율",
    "terminal_zero_count": "최종 -100% 처리 건수",
    "ff5_alpha_hac_tstat": "FF5 알파 HAC t-통계량",
    "rank_ic_mean": "평균 Rank IC",
    "annual_returns_top10": "상위 10% 전략 연도별 수익률",
    "monthly_equity_top10": "상위 10% 전략 월별 누적가치",
}
TABLE_LABELS = {
    "data_quality_summary": "데이터 품질 요약",
    "run_log": "실험 실행 로그",
    "backtest_summary": "백테스트 요약",
    "annual_return_tests": "연간 수익률 유의성 검정",
    "monthly_return_tests": "월간 수익률 유의성 검정",
    "ff5_alpha_tests": "Fama-French 5요인 알파 검정",
    "rank_ic_summary": "Rank IC 및 학습 지표",
    "annual_returns_raw": "연간 수익률 원자료",
    "monthly_returns_raw": "월간 수익률 원자료",
    "training_metrics_raw": "학습 지표 원자료",
}
METRIC_LABELS = {
    "historical_universe_tickers": "전체 후보 티커 수",
    "sp500_membership_rows": "S&P500 연도별 멤버십 행 수",
    "monthly_price_months": "월별 가격 기간 수",
    "monthly_price_tickers": "월별 가격 티커 수",
    "price_failure_tickers": "가격 누락 티커 수",
    "manual_filled_numeric_price_rows": "수동 보완 가격 숫자 행 수",
    "fundamental_rows": "재무 데이터 행 수",
    "fundamental_tickers": "재무 데이터 티커 수",
    "fundamental_failure_tickers": "재무 데이터 실패 티커 수",
    "feature_rows_model_ready": "모델 입력 가능 feature 행 수",
    "feature_tickers_model_ready": "모델 입력 가능 티커 수",
    "feature_year_min": "feature 시작연도",
    "feature_year_max": "feature 종료연도",
}
COLUMN_LABELS = {
    "metric": "항목",
    "value": "값",
    "name": "실험명",
    "exp_dir": "산출물 경로",
    "model_type": "모델",
    "sector_scope": "섹터 범위",
    "window": "롤링 윈도우(년)",
    "n_predictions": "예측 행 수",
    "n_return_rows": "백테스트 행 수",
    "created_at": "생성 시각",
    "Experiment": "실험",
    "Universe": "유니버스",
    "Model": "모델",
    "SectorScope": "섹터 범위",
    "WindowYears": "롤링 윈도우(년)",
    "Strategy": "전략",
    "years": "평가 연수",
    "mean_return": "평균 연수익률",
    "cumulative_return": "누적수익률",
    "cagr": "CAGR",
    "mdd": "최대낙폭(MDD)",
    "sharpe": "샤프 비율",
    "avg_selected": "평균 선정 종목 수",
    "avg_priced": "평균 가격 확인 종목 수",
    "total_delisted_zero": "-100% 처리 건수",
    "total_price_missing": "가격 누락 건수",
    "annual_return_n": "연간 표본 수",
    "annual_return_mean": "평균 연수익률",
    "annual_return_std": "연수익률 표준편차",
    "annual_return_tstat_mean_eq_0": "t-통계량(평균=0)",
    "annual_return_pvalue_two_sided": "p값(양측)",
    "annual_return_ci95_low": "95% 신뢰구간 하한",
    "annual_return_ci95_high": "95% 신뢰구간 상한",
    "bootstrap_iter": "부트스트랩 반복 수",
    "bootstrap_mean_p05": "부트스트랩 평균 5%",
    "bootstrap_mean_p50": "부트스트랩 평균 중앙값",
    "bootstrap_mean_p95": "부트스트랩 평균 95%",
    "bootstrap_prob_mean_le_zero": "평균수익률<=0 확률",
    "monthly_return_n": "월간 표본 수",
    "monthly_return_mean": "평균 월수익률",
    "monthly_return_std": "월수익률 표준편차",
    "monthly_return_tstat_mean_eq_0": "t-통계량(평균=0)",
    "monthly_return_pvalue_two_sided": "p값(양측)",
    "monthly_return_ci95_low": "95% 신뢰구간 하한",
    "monthly_return_ci95_high": "95% 신뢰구간 상한",
    "status": "상태",
    "n_obs": "관측치 수",
    "alpha_monthly": "월간 알파",
    "alpha_annualized": "연환산 알파",
    "alpha_tstat": "OLS 알파 t-통계량",
    "beta_mkt_rf": "시장 베타",
    "beta_smb": "SMB 베타",
    "beta_hml": "HML 베타",
    "beta_rmw": "RMW 베타",
    "beta_cma": "CMA 베타",
    "mean_monthly_return": "평균 월수익률",
    "mean_monthly_excess_return": "평균 월초과수익률",
    "r_squared": "R제곱",
    "ff5_hac_n_obs": "HAC 관측치 수",
    "ff5_hac_lag": "HAC lag",
    "alpha_monthly_hac": "HAC 월간 알파",
    "alpha_annualized_hac": "HAC 연환산 알파",
    "alpha_tstat_hac": "HAC 알파 t-통계량",
    "alpha_pvalue_hac": "HAC 알파 p값",
    "alpha_pvalue_ols_two_sided": "OLS 알파 p값(양측)",
    "ic_years": "IC 평가 연수",
    "rank_ic_mean": "평균 Rank IC",
    "rank_ic_std": "Rank IC 표준편차",
    "rank_ic_tstat_mean_eq_0": "Rank IC t-통계량",
    "rank_ic_pvalue_two_sided": "Rank IC p값(양측)",
    "rank_ic_ci95_low": "Rank IC 95% 신뢰구간 하한",
    "rank_ic_ci95_high": "Rank IC 95% 신뢰구간 상한",
    "n_train_mean": "평균 학습 표본 수",
    "n_test_mean": "평균 테스트 표본 수",
    "w1_mean": "평균 w1",
    "w2_mean": "평균 w2",
    "w3_mean": "평균 w3",
    "bias_mean": "평균 bias",
    "ranknet_loss_mean": "평균 RankNet loss",
    "n_ranknet_pairs_mean": "평균 RankNet pair 수",
}


def set_korean_plot_theme() -> None:
    font_path = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=plt.rcParams.get("font.family", "sans-serif"))


def display_model(value: object) -> str:
    return MODEL_LABELS.get(str(value), str(value))


def display_sector_scope(value: object) -> str:
    return SECTOR_SCOPE_LABELS.get(str(value), str(value))


def display_strategy(value: object) -> str:
    return STRATEGY_LABELS.get(str(value), str(value))


def display_experiment(value: object) -> str:
    text = str(value)
    parts = text.split("_")
    if len(parts) >= 4 and parts[0] == "sp500":
        model = display_model(parts[1])
        sector = display_sector_scope("_".join(parts[2:-1]))
        window = parts[-1].replace("w", "")
        return f"{model} {sector} {window}년"
    return text


def display_status(value: object) -> str:
    mapping = {
        "ok": "정상",
        "insufficient_observations": "관측치 부족",
        "missing_factor_file": "요인 파일 없음",
        "no_monthly_portfolio_returns": "월간 수익률 없음",
        "no_overlapping_factor_months": "요인 기간 겹침 없음",
    }
    return mapping.get(str(value), str(value))


def display_report_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "metric" in out.columns:
        out["metric"] = out["metric"].map(lambda x: METRIC_LABELS.get(str(x), str(x)))
    if "Experiment" in out.columns:
        out["Experiment"] = out["Experiment"].map(display_experiment)
    if "name" in out.columns:
        out["name"] = out["name"].map(display_experiment)
    if "Model" in out.columns:
        out["Model"] = out["Model"].map(display_model)
    if "model_type" in out.columns:
        out["model_type"] = out["model_type"].map(display_model)
    if "SectorScope" in out.columns:
        out["SectorScope"] = out["SectorScope"].map(display_sector_scope)
    if "sector_scope" in out.columns:
        out["sector_scope"] = out["sector_scope"].map(display_sector_scope)
    if "Strategy" in out.columns:
        out["Strategy"] = out["Strategy"].map(display_strategy)
    if "status" in out.columns:
        out["status"] = out["status"].map(display_status)
    return out.rename(columns={column: COLUMN_LABELS.get(column, column) for column in out.columns})


def add_plot_labels(df: pd.DataFrame, include_strategy: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["모델"] = out["Model"].map(display_model) if "Model" in out.columns else ""
    if "Experiment" in out.columns:
        out["실험명"] = out["Experiment"].map(display_experiment)
    if include_strategy and "Strategy" in out.columns:
        out["전략명"] = out["Strategy"].map(display_strategy)
        out["라벨"] = out["실험명"] + " | " + out["전략명"]
    elif "실험명" in out.columns:
        out["라벨"] = out["실험명"]
    return out


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def experiment_metadata(name: str) -> dict[str, object]:
    parts = name.split("_")
    return {
        "Experiment": name,
        "Universe": "S&P500",
        "Model": parts[1],
        "SectorScope": "_".join(parts[2:-1]),
        "WindowYears": int(parts[-1].replace("w", "")),
    }


def read_experiment_file(filename: str) -> pd.DataFrame:
    frames = []
    for name in EXPERIMENTS:
        path = config.EXPERIMENTS_DIR / name / filename
        frame = read_csv(path)
        if frame.empty:
            continue
        for key, value in experiment_metadata(name).items():
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def one_sample_test(values: pd.Series | np.ndarray, mu: float = 0.0) -> dict[str, float]:
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    n = len(arr)
    if n < 2:
        return {
            "n": n,
            "mean": float(arr.mean()) if n else np.nan,
            "std": np.nan,
            "t_stat": np.nan,
            "p_value_two_sided": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    se = std / math.sqrt(n) if std > 0 else np.nan
    t_stat = (mean - mu) / se if se and not pd.isna(se) else np.nan
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if not pd.isna(t_stat) else np.nan
    critical = float(stats.t.ppf(0.975, df=n - 1))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "t_stat": float(t_stat) if not pd.isna(t_stat) else np.nan,
        "p_value_two_sided": p_value,
        "ci95_low": mean - critical * se if se and not pd.isna(se) else np.nan,
        "ci95_high": mean + critical * se if se and not pd.isna(se) else np.nan,
    }


def bootstrap_mean(values: pd.Series | np.ndarray, n_iter: int = 10000, seed: int = 42) -> dict[str, float]:
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return {
            "bootstrap_iter": n_iter,
            "bootstrap_mean_p05": np.nan,
            "bootstrap_mean_p50": np.nan,
            "bootstrap_mean_p95": np.nan,
            "bootstrap_prob_mean_le_zero": np.nan,
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_iter, len(arr)), replace=True).mean(axis=1)
    return {
        "bootstrap_iter": n_iter,
        "bootstrap_mean_p05": float(np.quantile(draws, 0.05)),
        "bootstrap_mean_p50": float(np.quantile(draws, 0.50)),
        "bootstrap_mean_p95": float(np.quantile(draws, 0.95)),
        "bootstrap_prob_mean_le_zero": float((draws <= 0).mean()),
    }


def build_annual_tests(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["Experiment", "Universe", "Model", "SectorScope", "WindowYears", "Strategy"]
    for keys, group in returns.groupby(group_cols):
        stats_row = one_sample_test(group["return"])
        boot_row = bootstrap_mean(group["return"])
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "annual_return_n": stats_row.pop("n"),
                "annual_return_mean": stats_row.pop("mean"),
                "annual_return_std": stats_row.pop("std"),
                "annual_return_tstat_mean_eq_0": stats_row.pop("t_stat"),
                "annual_return_pvalue_two_sided": stats_row.pop("p_value_two_sided"),
                "annual_return_ci95_low": stats_row.pop("ci95_low"),
                "annual_return_ci95_high": stats_row.pop("ci95_high"),
            }
        )
        row.update(boot_row)
        rows.append(row)
    return pd.DataFrame(rows)


def build_monthly_tests(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["Experiment", "Universe", "Model", "SectorScope", "WindowYears", "Strategy"]
    for keys, group in monthly_returns.groupby(group_cols):
        stats_row = one_sample_test(group["monthly_return"])
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "monthly_return_n": stats_row["n"],
                "monthly_return_mean": stats_row["mean"],
                "monthly_return_std": stats_row["std"],
                "monthly_return_tstat_mean_eq_0": stats_row["t_stat"],
                "monthly_return_pvalue_two_sided": stats_row["p_value_two_sided"],
                "monthly_return_ci95_low": stats_row["ci95_low"],
                "monthly_return_ci95_high": stats_row["ci95_high"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def hac_covariance(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    n, _ = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for t in range(n):
        xt = x[t : t + 1].T
        meat += float(residuals[t] ** 2) * (xt @ xt.T)
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1)
        gamma = np.zeros_like(meat)
        for t in range(l, n):
            xt = x[t : t + 1].T
            xl = x[t - l : t - l + 1].T
            gamma += float(residuals[t] * residuals[t - l]) * (xt @ xl.T)
        meat += weight * (gamma + gamma.T)
    return xtx_inv @ meat @ xtx_inv


def run_ff5_hac_tests(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    factors = load_ff5_factors(config.FF5_MONTHLY_FACTORS_CSV)
    monthly = monthly_returns.copy()
    monthly["MonthEnd"] = pd.to_datetime(monthly["MonthEnd"]) + pd.offsets.MonthEnd(0)
    factors["MonthEnd"] = pd.to_datetime(factors["MonthEnd"]) + pd.offsets.MonthEnd(0)
    reg = monthly.merge(factors, on="MonthEnd", how="inner")
    reg["excess_return"] = reg["monthly_return"] - reg["RF"]

    rows = []
    group_cols = ["Experiment", "Universe", "Model", "SectorScope", "WindowYears", "Strategy"]
    for keys, group in reg.groupby(group_cols):
        group = group.dropna(subset=["excess_return"] + FF5_EXPOSURE_COLS)
        n = len(group)
        k = len(FF5_EXPOSURE_COLS) + 1
        row = dict(zip(group_cols, keys))
        row["ff5_hac_n_obs"] = n
        row["ff5_hac_lag"] = np.nan
        if n <= k:
            for col in ["alpha_monthly_hac", "alpha_annualized_hac", "alpha_tstat_hac", "alpha_pvalue_hac"]:
                row[col] = np.nan
            rows.append(row)
            continue
        y = group["excess_return"].to_numpy(dtype=float)
        x = group[FF5_EXPOSURE_COLS].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        lag = max(1, int(math.floor(4 * (n / 100.0) ** (2 / 9))))
        cov = hac_covariance(x, resid, lag)
        se = math.sqrt(cov[0, 0]) if cov[0, 0] > 0 else np.nan
        t_stat = beta[0] / se if se and not pd.isna(se) else np.nan
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - k)) if not pd.isna(t_stat) else np.nan
        row.update(
            {
                "ff5_hac_lag": lag,
                "alpha_monthly_hac": float(beta[0]),
                "alpha_annualized_hac": float((1 + beta[0]) ** 12 - 1),
                "alpha_tstat_hac": float(t_stat) if not pd.isna(t_stat) else np.nan,
                "alpha_pvalue_hac": p_value,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ic_summary(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["Experiment", "Universe", "Model", "SectorScope", "WindowYears"]
    for keys, group in training.groupby(group_cols):
        test = one_sample_test(group["rank_ic"])
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "ic_years": test["n"],
                "rank_ic_mean": test["mean"],
                "rank_ic_std": test["std"],
                "rank_ic_tstat_mean_eq_0": test["t_stat"],
                "rank_ic_pvalue_two_sided": test["p_value_two_sided"],
                "rank_ic_ci95_low": test["ci95_low"],
                "rank_ic_ci95_high": test["ci95_high"],
                "n_train_mean": float(group["n_train"].mean()),
                "n_test_mean": float(group["n_test"].mean()),
                "w1_mean": float(group["w1"].mean()),
                "w2_mean": float(group["w2"].mean()),
                "w3_mean": float(group["w3"].mean()),
                "bias_mean": float(group["bias"].mean()),
                "ranknet_loss_mean": float(group["ranknet_loss"].dropna().mean())
                if group["ranknet_loss"].notna().any()
                else np.nan,
                "n_ranknet_pairs_mean": float(group["n_ranknet_pairs"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_data_quality_summary() -> pd.DataFrame:
    rows = []
    universe = json.loads(config.SP500_UNIVERSE_JSON.read_text(encoding="utf-8"))
    prices = pd.read_csv(config.PRICE_MONTHLY_CLOSE_CSV, index_col=0)
    price_failures = read_csv(config.PRICE_FAILURES_CSV)
    manual_filled_path = config.ROOT_DIR.parent / "sp500_price_collection_request_filled.xlsx"
    manual_numeric_prices = 0
    if manual_filled_path.exists():
        try:
            manual = pd.read_excel(manual_filled_path, sheet_name="필요월별가격_채우기")
            manual_numeric_prices = int(pd.to_numeric(manual.get("close"), errors="coerce").notna().sum())
        except Exception:
            manual_numeric_prices = 0
    fundamentals = pd.read_csv(config.FUNDAMENTALS_CSV)
    fundamental_failures = read_csv(config.FUNDAMENTALS_FAILURES_CSV)
    features = pd.read_csv(config.FEATURES_ANNUAL_CSV)
    membership = pd.read_csv(config.SP500_MEMBERSHIP_CSV)

    rows.extend(
        [
            ("historical_universe_tickers", len(universe)),
            ("sp500_membership_rows", len(membership)),
            ("monthly_price_months", len(prices)),
            ("monthly_price_tickers", prices.shape[1]),
            ("price_failure_tickers", len(price_failures)),
            ("manual_filled_numeric_price_rows", manual_numeric_prices),
            ("fundamental_rows", len(fundamentals)),
            ("fundamental_tickers", fundamentals["Ticker"].nunique()),
            ("fundamental_failure_tickers", len(fundamental_failures)),
            ("feature_rows_model_ready", len(features)),
            ("feature_tickers_model_ready", features["Ticker"].nunique()),
            ("feature_year_min", int(features["FiscalYear"].min())),
            ("feature_year_max", int(features["FiscalYear"].max())),
        ]
    )
    return pd.DataFrame(rows, columns=["metric", "value"])


def add_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Strategy" in out.columns:
        out["_strategy_order"] = out["Strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)}).fillna(99)
    else:
        out["_strategy_order"] = 0
    if "Experiment" in out.columns:
        out["_experiment_order"] = out["Experiment"].map({e: i for i, e in enumerate(EXPERIMENTS)}).fillna(99)
        return out.sort_values(["_experiment_order", "_strategy_order"]).drop(
            columns=["_experiment_order", "_strategy_order"], errors="ignore"
        )
    return out.drop(columns=["_strategy_order"], errors="ignore")


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        add_sort_columns(table).to_csv(TABLE_DIR / f"{name}.csv", index=False)

    workbook_path = REPORT_DIR / "capstone_experiment_report_tables.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, table in tables.items():
            sheet_name = name[:31]
            add_sort_columns(table).to_excel(writer, sheet_name=sheet_name, index=False)


def plot_metric_bars(backtest: pd.DataFrame, ff5: pd.DataFrame, ic: pd.DataFrame) -> list[Path]:
    set_korean_plot_theme()
    paths = []

    ordered = add_plot_labels(add_sort_columns(backtest))
    for metric, title, ylabel, filename in [
        ("mean_return", "연평균 수익률: 실험/전략별", "연평균 수익률", "annual_mean_return.png"),
        ("sharpe", "연간 샤프비율: 실험/전략별", "샤프비율", "annual_sharpe.png"),
        ("total_delisted_zero", "최종 -100% 처리 건수: 실험/전략별", "건수", "terminal_zero_count.png"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=ordered, x="라벨", y=metric, hue="모델", dodge=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("실험 / 전략")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=90)
        if ax.legend_ is not None:
            ax.legend(title="모델")
        fig.tight_layout()
        path = FIG_DIR / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    ff5_ordered = add_plot_labels(add_sort_columns(ff5))
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=ff5_ordered, x="라벨", y="alpha_tstat_hac", hue="모델", dodge=False, ax=ax)
    ax.axhline(1.96, color="red", linestyle="--", linewidth=1)
    ax.axhline(-1.96, color="red", linestyle="--", linewidth=1)
    ax.set_title("Fama-French 5요인 알파 HAC t통계량")
    ax.set_xlabel("실험 / 전략")
    ax.set_ylabel("알파 HAC t통계량")
    ax.tick_params(axis="x", rotation=90)
    if ax.legend_ is not None:
        ax.legend(title="모델")
    fig.tight_layout()
    path = FIG_DIR / "ff5_alpha_hac_tstat.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    ic_ordered = add_plot_labels(ic, include_strategy=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=ic_ordered, x="라벨", y="rank_ic_mean", hue="모델", dodge=False, ax=ax)
    ax.set_title("실험별 평균 Rank IC")
    ax.set_xlabel("실험")
    ax.set_ylabel("평균 Rank IC")
    ax.tick_params(axis="x", rotation=45)
    if ax.legend_ is not None:
        ax.legend(title="모델")
    fig.tight_layout()
    path = FIG_DIR / "rank_ic_mean.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def plot_return_lines(returns: pd.DataFrame, monthly: pd.DataFrame) -> list[Path]:
    set_korean_plot_theme()
    paths = []
    top10 = returns[returns["Strategy"] == "gap_top_10pct"].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    for exp, group in top10.groupby("Experiment"):
        group = group.sort_values("FiscalYear")
        ax.plot(group["FiscalYear"], group["return"], marker="o", label=display_experiment(exp))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("연간 수익률: 상위 10% 전략")
    ax.set_xlabel("회계연도")
    ax.set_ylabel("연간 수익률")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = FIG_DIR / "annual_returns_top10.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    monthly_top10 = monthly[monthly["Strategy"] == "gap_top_10pct"].copy()
    monthly_top10["MonthEnd"] = pd.to_datetime(monthly_top10["MonthEnd"])
    fig, ax = plt.subplots(figsize=(12, 6))
    for exp, group in monthly_top10.groupby("Experiment"):
        group = group.sort_values("MonthEnd")
        equity = (1 + group["monthly_return"].fillna(0)).cumprod()
        ax.plot(group["MonthEnd"], equity, label=display_experiment(exp))
    ax.set_title("월간 자본곡선: 상위 10% 전략")
    ax.set_xlabel("월")
    ax.set_ylabel("누적 가치")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = FIG_DIR / "monthly_equity_top10.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    table = df if columns is None else df[columns]
    table = display_report_table(table)
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, sep] + rows)


def write_report(
    tables: dict[str, pd.DataFrame],
    figure_paths: list[Path],
) -> Path:
    report_path = REPORT_DIR / "capstone_experiment_report.md"
    data_quality = tables["data_quality_summary"]
    backtest = add_sort_columns(tables["backtest_summary"])
    annual_tests = add_sort_columns(tables["annual_return_tests"])
    monthly_tests = add_sort_columns(tables["monthly_return_tests"])
    ff5 = add_sort_columns(tables["ff5_alpha_tests"])
    ic = add_sort_columns(tables["rank_ic_summary"])
    run_log = tables["run_log"]

    lines = [
        "# 캡스톤 S&P500 실험 정량 보고서",
        "",
        f"- 생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        "- 실험 조합: S&P500 / {MSE, RankNet} / {전체, 바이오_IT} / {5년, 3년 롤링 윈도우}",
        "- 전략: 예측 괴리율 기준 상위 5%, 10%, 20%, 30% 포트폴리오",
        "- 예측 대상: 섹터 중립 ln(PBR) 괴리율",
        "- 유의성 검정: 연간 수익률 단일표본 t-test, 월간 수익률 단일표본 t-test, 부트스트랩 평균수익률 분포, Fama-French 5요인 알파 Newey-West/HAC t-통계량, Rank IC t-test",
        "",
        "## 데이터 품질 및 처리 기준",
        "",
        markdown_table(data_quality),
        "",
        "### 결측치 및 이상치 처리",
        "",
        "- SEC companyfacts 재무 데이터에서 R&D/SG&A가 없는 경우, 기존 CARL 파이프라인 및 PDF의 정책과 맞춰 무형투자 계산에서 0으로 처리했습니다.",
        "- 모델 필수 변수(`m1`, `m2`, `m3`, `y`, `Close`)가 없는 행은 학습/예측 전에 제거했습니다.",
        "- 로그 변환 전 PBR은 [0.01, 100]으로 제한했고, `m1`, `m2`는 [-5, 5], `m3`는 [0, 10] 범위로 제한했습니다.",
        "- look-ahead bias를 줄이기 위해 valuation 기준일은 `period_end + 3개월`과 실제 공시일 중 더 늦은 날짜의 월말로 잡았습니다.",
        "- 백테스트 매수가는 리밸런싱일 기준 45일 이내에 존재해야 합니다.",
        "- 보유기간이 완전히 끝난 연도만 평가했습니다. 목표 매도일이 현재 가격 데이터 범위를 넘어서는 마지막 미완료 연도는 제외했습니다.",
        "- 목표 매도일이 가격 데이터 범위 안에 있는데 개별 종목 매도가격이 없으면 상폐/거래중단으로 보고 해당 종목 수익률을 -100%로 처리했습니다.",
        "- 리밸런싱 전 12개월 중 가격이 5달러 이하인 비율이 80% 이상이면 가격 필터를 통과하지 못하게 했습니다.",
        "- 수동 보완 엑셀 확인 결과 숫자로 채워진 가격 행은 0건이어서 추가 병합하지 않았습니다.",
        "",
        "## 시각화",
        "",
    ]
    for path in figure_paths:
        rel = path.relative_to(REPORT_DIR)
        label = FIGURE_LABELS.get(path.stem, path.stem)
        lines.append(f"![{label}]({rel.as_posix()})")
        lines.append("")

    lines.extend(
        [
            "## 실험 실행 로그",
            "",
            markdown_table(run_log),
            "",
            "## 백테스트 요약: 전체 정량 지표",
            "",
            markdown_table(backtest),
            "",
            "## 연간 수익률 유의성 검정",
            "",
            markdown_table(annual_tests),
            "",
            "## 월간 수익률 유의성 검정",
            "",
            markdown_table(monthly_tests),
            "",
            "## Fama-French 5요인 알파 검정",
            "",
            markdown_table(ff5),
            "",
            "## Rank IC 및 학습 지표",
            "",
            markdown_table(ic),
            "",
            "## 전체 CSV 표",
            "",
        ]
    )
    for name in tables:
        label = TABLE_LABELS.get(name, name)
        lines.append(f"- [{label}](tables/{name}.csv)")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    ensure_dirs()

    run_log = read_csv(config.EXPERIMENTS_DIR / "capstone_8_run_log.csv")
    backtest_summary = read_csv(config.EXPERIMENTS_DIR / "capstone_8_backtest_summary.csv")
    returns = read_experiment_file("backtest_returns.csv")
    monthly = read_experiment_file("monthly_portfolio_returns.csv")
    training = read_experiment_file("training_metrics.csv")
    ff5_base = read_csv(config.EXPERIMENTS_DIR / "capstone_8_ff5_alpha_summary.csv")
    annual_tests = build_annual_tests(returns)
    monthly_tests = build_monthly_tests(monthly)
    ic_summary = build_ic_summary(training)
    ff5_hac = run_ff5_hac_tests(monthly)
    ff5 = ff5_base.merge(
        ff5_hac,
        on=["Experiment", "Universe", "Model", "SectorScope", "WindowYears", "Strategy"],
        how="left",
    )
    ff5["alpha_pvalue_ols_two_sided"] = ff5.apply(
        lambda row: float(2 * stats.t.sf(abs(row["alpha_tstat"]), df=row["n_obs"] - 6))
        if row.get("status") == "ok" and pd.notna(row.get("alpha_tstat")) and row.get("n_obs", 0) > 6
        else np.nan,
        axis=1,
    )

    tables = {
        "data_quality_summary": build_data_quality_summary(),
        "run_log": run_log,
        "backtest_summary": backtest_summary,
        "annual_return_tests": annual_tests,
        "monthly_return_tests": monthly_tests,
        "ff5_alpha_tests": ff5,
        "rank_ic_summary": ic_summary,
        "annual_returns_raw": returns,
        "monthly_returns_raw": monthly,
        "training_metrics_raw": training,
    }
    save_tables(tables)
    figures = []
    figures.extend(plot_metric_bars(backtest_summary, ff5, ic_summary))
    figures.extend(plot_return_lines(returns, monthly))
    report_path = write_report(tables, figures)
    print(f"report: {report_path}")
    print(f"tables: {TABLE_DIR}")
    print(f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
