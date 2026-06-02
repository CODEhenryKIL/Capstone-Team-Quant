# Capstone S&P 500 Dataset Pipeline

This directory is independent from the original STAI-CARL repository.

## Goal

Build a capstone-only experiment dataset from:

- Wikipedia S&P 500 current constituents and historical changes
- Yahoo Finance chart API prices
- SEC EDGAR CompanyFacts fundamentals

Default range is `2011-2025`, which is 15 fully completed fiscal/calendar years.

## Setup

```bash
cd 캡스톤_실험
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

When the network is slow, first run only Step 0 from the cached Wikipedia source files:

```bash
.venv/bin/python run_pipeline.py --steps 0
```

To refetch Wikipedia/SEC source data later:

```bash
.venv/bin/python run_pipeline.py --steps 0 --refresh-step0
```

Then run data download/build steps when the network is usable:

```bash
.venv/bin/python run_pipeline.py --steps 1
.venv/bin/python run_pipeline.py --steps 2
.venv/bin/python run_pipeline.py --steps 3
.venv/bin/python run_pipeline.py --steps 4
```

Or run all:

```bash
.venv/bin/python run_pipeline.py --steps 01234
```

Check readiness:

```bash
.venv/bin/python validate_dataset.py
```

Run one experiment after `features_annual.csv` and monthly prices exist:

```bash
.venv/bin/python run_experiment.py --name sp500_mse_all_w5 --model mse --sector-scope all --window 5
```

Run the full capstone experiment suite:

```bash
.venv/bin/python run_experiment_suite.py
```

The experiment uses one selection/backtest pipeline and writes two evaluation tracks:

- Raw annual backtest: realized holding-period return, CAGR, MDD, Sharpe, delisting counts
- Fama-French 5-factor track: monthly portfolio returns, and FF5 alpha when the factor file exists

The full suite runs 8 variants:

```text
1. S&P500 / MSE     / all sectors / 5-year rolling
2. S&P500 / MSE     / all sectors / 3-year rolling
3. S&P500 / MSE     / bio & IT    / 5-year rolling
4. S&P500 / MSE     / bio & IT    / 3-year rolling
5. S&P500 / RankNet / all sectors / 5-year rolling
6. S&P500 / RankNet / all sectors / 3-year rolling
7. S&P500 / RankNet / bio & IT    / 5-year rolling
8. S&P500 / RankNet / bio & IT    / 3-year rolling
```

`bio & IT` means `Sector == Information Technology` or `Industry == Biotechnology`.
Override with `CAPSTONE_BIO_IT_SECTORS` and `CAPSTONE_BIO_IT_INDUSTRIES` if needed.
The MSE path uses the existing linear MSE objective with optional L2 alpha
(`CAPSTONE_RIDGE_ALPHA`, default `1.0`; pass `--alpha 0` for unregularized MSE).

For FF5 alpha, place the Kenneth French monthly 5-factor CSV at:

```text
data/raw/fama_french_5_factors_monthly.csv
```

You can also download it with:

```bash
.venv/bin/python run_pipeline.py --steps 4
```

The parser accepts the original Kenneth French CSV format with `Mkt-RF, SMB, HML, RMW, CMA, RF`.
If the file is missing, `run_experiment.py` still creates `monthly_portfolio_returns.csv` and records
`missing_factor_file` in `ff5_alpha_summary.csv`.

## Outputs

- `data/interim/sp500_membership_by_year.csv`
- `data/interim/sp500_universe.json`
- `data/interim/prices_monthly_close.csv`
- `data/interim/fundamentals_sec_companyfacts.csv`
- `data/raw/fama_french_5_factors_monthly.csv`
- `data/processed/features_annual.csv`
- `data/processed/dataset_annual_model.csv`
- `experiments/sp500_mse_all_w5/predictions.csv`
- `experiments/sp500_mse_all_w5/backtest_summary.csv`
- `experiments/sp500_mse_all_w5/monthly_portfolio_returns.csv`
- `experiments/sp500_mse_all_w5/ff5_alpha_summary.csv`
- `experiments/sp500_mse_all_w5/ff5_regression_data.csv`
- `experiments/sp500_mse_all_w5/report.md`
- `experiments/capstone_8_backtest_summary.csv`
- `experiments/capstone_8_ff5_alpha_summary.csv`
- `experiments/capstone_8_run_log.csv`

## Evaluation Interpretation

The experiment itself is shared:

```text
S&P 500 universe -> features -> model prediction -> stock selection -> portfolio holding
```

The output interpretation is split:

- `backtest_summary.csv`: practical strategy performance. Use this to answer whether the selected portfolio made money.
- `ff5_alpha_summary.csv`: risk-adjusted abnormal performance. Use this to answer whether returns remain after controlling for market, size, value, profitability, and investment factors.
- `monthly_portfolio_returns.csv`: the bridge between both views. It is built from the same selected holdings, but sampled monthly so it can be joined with Fama-French monthly factors.

## Membership and Delisting Rules

- Candidate universe: any company included in the S&P 500 at least once during `2011-2025`.
- Selection rule: for each prediction/backtest fiscal year, candidates are filtered to tickers that were S&P 500 members in that same fiscal year.
- Holding rule: once selected, a ticker is still evaluated through the holding period even if it exits the S&P 500 in the following year.
- New additions: a ticker can only become selectable from the fiscal year in which it appears in the annual membership snapshot.
- S&P 500 deletion rule: deletion from the index does not imply `-100%`; if the stock keeps trading, the backtest uses its market price at the sell date.
- Delisting/price stop rule: only when no usable sell-date-or-later price exists after purchase, the position return is set to `-100%` and marked `delisted_no_sell_price_zero` in `holdings.csv`.
- If a ticker has no usable buy price near the rebalance date, it is not selected.

## Overrides

Historical tickers can require manual mapping for Yahoo or SEC. Create:

`data/raw/ticker_overrides.csv`

with optional columns:

```csv
ticker,price_ticker,sec_ticker,cik,sector,industry,name
FB,META,META,0001326801,Communication Services,Interactive Media & Services,Meta Platforms
```
