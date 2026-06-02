from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
SEC_CACHE_DIR = CACHE_DIR / "sec"
PRICE_CACHE_DIR = CACHE_DIR / "prices"

START_YEAR = int(os.getenv("CAPSTONE_START_YEAR", "2011"))
END_YEAR = int(os.getenv("CAPSTONE_END_YEAR", "2025"))

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FF5_MONTHLY_FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

REQUEST_TIMEOUT = 30
SEC_SLEEP_SECONDS = float(os.getenv("CAPSTONE_SEC_SLEEP_SECONDS", "0.15"))
YAHOO_SLEEP_SECONDS = float(os.getenv("CAPSTONE_YAHOO_SLEEP_SECONDS", "0.05"))
YAHOO_MAX_WORKERS = int(os.getenv("CAPSTONE_YAHOO_MAX_WORKERS", "8"))
SAFE_LAG_MONTHS = int(os.getenv("CAPSTONE_SAFE_LAG_MONTHS", "3"))

# Keep empty for the capstone default: the universe itself is S&P 500.
# Example if you want the legacy filter:
# ("Financials", "Utilities", "Energy", "Materials", "Real Estate", "Unknown")
EXCLUDED_SECTORS: tuple[str, ...] = ()

SEC_HEADERS = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT",
        "STAI-CARL Capstone Dataset contact@example.com",
    ),
    "Accept-Encoding": "gzip, deflate",
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; STAI-CARL-Capstone/1.0)",
}

OVERRIDES_CSV = RAW_DIR / "ticker_overrides.csv"
PERIOD_OVERRIDES_CSV = RAW_DIR / "ticker_period_overrides.csv"

SP500_CURRENT_CSV = RAW_DIR / "sp500_current_constituents.csv"
SP500_CHANGES_CSV = RAW_DIR / "sp500_changes.csv"
SP500_MEMBERSHIP_CSV = INTERIM_DIR / "sp500_membership_by_year.csv"
SP500_MEMBERSHIP_JSON = INTERIM_DIR / "sp500_membership_by_year.json"
SP500_UNIVERSE_JSON = INTERIM_DIR / "sp500_universe.json"
SP500_DROPPED_JSON = INTERIM_DIR / "sp500_dropped_tickers.json"
SP500_SUMMARY_CSV = INTERIM_DIR / "sp500_membership_summary.csv"

PRICE_DAILY_CLOSE_CSV = INTERIM_DIR / "prices_daily_close.csv"
PRICE_DAILY_ADJ_CLOSE_CSV = INTERIM_DIR / "prices_daily_adj_close.csv"
PRICE_MONTHLY_CLOSE_CSV = INTERIM_DIR / "prices_monthly_close.csv"
PRICE_MONTHLY_ADJ_CLOSE_CSV = INTERIM_DIR / "prices_monthly_adj_close.csv"
PRICE_FAILURES_CSV = INTERIM_DIR / "price_failures.csv"

FUNDAMENTALS_CSV = INTERIM_DIR / "fundamentals_sec_companyfacts.csv"
FUNDAMENTALS_FAILURES_CSV = INTERIM_DIR / "fundamentals_failures.csv"
FEATURES_ANNUAL_CSV = PROCESSED_DIR / "features_annual.csv"
MODEL_DATASET_ANNUAL_CSV = PROCESSED_DIR / "dataset_annual_model.csv"
FF5_MONTHLY_FACTORS_CSV = RAW_DIR / "fama_french_5_factors_monthly.csv"

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
TRAIN_WINDOW_YEARS = int(os.getenv("CAPSTONE_TRAIN_WINDOW_YEARS", "5"))
BACKTEST_START_YEAR = int(os.getenv("CAPSTONE_BACKTEST_START_YEAR", str(START_YEAR + TRAIN_WINDOW_YEARS)))
RIDGE_ALPHA = float(os.getenv("CAPSTONE_RIDGE_ALPHA", "1.0"))
Y_TARGET_MODE = os.getenv("CAPSTONE_Y_TARGET_MODE", "sector")
LONG_RATIOS = tuple(float(x) for x in os.getenv("CAPSTONE_LONG_RATIOS", "0.05,0.10,0.20,0.30").split(","))
HOLDING_MONTHS = int(os.getenv("CAPSTONE_HOLDING_MONTHS", "12"))
PRICE_THRESHOLD = float(os.getenv("CAPSTONE_PRICE_THRESHOLD", "5.0"))
PRICE_BELOW_RATIO = float(os.getenv("CAPSTONE_PRICE_BELOW_RATIO", "0.80"))
BUY_PRICE_MAX_STALENESS_DAYS = int(os.getenv("CAPSTONE_BUY_PRICE_MAX_STALENESS_DAYS", "45"))
BIO_IT_SECTORS = tuple(
    item.strip()
    for item in os.getenv("CAPSTONE_BIO_IT_SECTORS", "Information Technology").split(",")
    if item.strip()
)
BIO_IT_INDUSTRIES = tuple(
    item.strip()
    for item in os.getenv("CAPSTONE_BIO_IT_INDUSTRIES", "Biotechnology").split(",")
    if item.strip()
)
RANKNET_LEARNING_RATE = float(os.getenv("CAPSTONE_RANKNET_LEARNING_RATE", "0.05"))
RANKNET_EPOCHS = int(os.getenv("CAPSTONE_RANKNET_EPOCHS", "150"))
RANKNET_L2 = float(os.getenv("CAPSTONE_RANKNET_L2", "0.01"))
RANKNET_MAX_PAIRS_PER_YEAR = int(os.getenv("CAPSTONE_RANKNET_MAX_PAIRS_PER_YEAR", "5000"))
RANKNET_SEED = int(os.getenv("CAPSTONE_RANKNET_SEED", "42"))

XBRL_TAGS = {
    "Shares": [
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "RD": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentExpenseSoftwareExcludingAmortization",
    ],
    "SGA": [
        "SellingGeneralAndAdministrativeExpense",
        "SellingAndMarketingExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "OI": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenuesNetOfInterestExpense",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "NetIncome": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
}


def ensure_dirs() -> None:
    for path in [
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        CACHE_DIR,
        SEC_CACHE_DIR,
        PRICE_CACHE_DIR,
        EXPERIMENTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
