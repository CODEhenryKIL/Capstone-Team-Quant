from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_ticker(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    ticker = str(value).strip().upper()
    if not ticker or ticker.lower() == "nan":
        return None
    ticker = re.sub(r"\[[^\]]+\]", "", ticker)
    ticker = ticker.replace(".", "-").replace(" ", "")
    return ticker or None


def clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\[[^\]]+\]", "", str(value)).strip()
    return text or None


def normalize_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    stopwords = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "class",
        "plc",
        "ltd",
        "limited",
        "the",
    }
    return " ".join(token for token in text.split() if token not in stopwords)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        columns = []
        for column in df.columns:
            parts = [
                str(part).strip()
                for part in column
                if str(part).strip()
                and not str(part).startswith("Unnamed")
                and str(part).lower() != "nan"
            ]
            columns.append("_".join(parts))
        df.columns = columns
    else:
        df.columns = [str(column).strip() for column in df.columns]
    return df


def find_column(columns: Iterable[str], *tokens: str) -> str:
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_"): column
        for column in columns
    }
    for normalized_name, original in normalized.items():
        if all(token in normalized_name for token in tokens):
            return original
    raise KeyError(f"Could not find column matching {tokens}: {list(columns)}")


def format_cik(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if "." in text:
        text = text.split(".", 1)[0]
    digits = re.sub(r"\D", "", text)
    return digits.zfill(10) if digits else None


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} must contain a ticker column")
    df["ticker"] = df["ticker"].map(normalize_ticker)
    return df.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")
