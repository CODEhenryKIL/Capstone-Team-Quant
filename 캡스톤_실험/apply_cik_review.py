from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import config


NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def column_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - 64
    return index - 1


def node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext())


def normalize_cik(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits.zfill(10) if digits else ""


def normalize_year_range(value: object) -> tuple[int, int]:
    years = [int(match) for match in re.findall(r"\d{4}", str(value or ""))]
    if not years:
        raise ValueError(f"Could not parse year range: {value}")
    return min(years), max(years)


def find_default_workbook() -> Path:
    candidates = [
        *config.ROOT_DIR.parent.glob("*sp500_cik_review_filled.xlsx"),
        *config.ROOT_DIR.glob("*sp500_cik_review_filled.xlsx"),
        config.DATA_DIR / "manual" / "sp500_cik_review_filled.xlsx",
        config.DATA_DIR / "manual" / "sp500_cik_review.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find filled CIK review workbook")


def read_xlsx(path: Path) -> dict[str, list[dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = ["".join(item.itertext()) for item in root.findall("m:si", NS)]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relmap = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pr:Relationship", REL_NS)
        }

        sheets: dict[str, str] = {}
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = relmap[rel_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheets[name] = target

        def read_sheet(target: str) -> list[dict[str, str]]:
            root = ET.fromstring(archive.read(target))
            raw_rows: list[dict[int, str]] = []
            width = 0
            for row in root.findall("m:sheetData/m:row", NS):
                values: dict[int, str] = {}
                for cell in row.findall("m:c", NS):
                    idx = column_index(cell.attrib.get("r", "A1"))
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find("m:v", NS)
                    if cell_type == "s":
                        raw = node_text(value_node)
                        value = shared_strings[int(raw)] if raw.isdigit() else raw
                    elif cell_type == "inlineStr":
                        value = node_text(cell.find("m:is", NS))
                    else:
                        value = node_text(value_node)
                    values[idx] = value.strip()
                if values:
                    width = max(width, max(values) + 1)
                    raw_rows.append(values)

            matrix = [[row.get(idx, "") for idx in range(width)] for row in raw_rows]
            if not matrix:
                return []
            headers = matrix[0]
            return [dict(zip(headers, row + [""] * (len(headers) - len(row)))) for row in matrix[1:]]

        return {name: read_sheet(target) for name, target in sheets.items()}


def write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_general_overrides(sheets: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    if "override_template" in sheets:
        source_rows = sheets["override_template"]
        rows = []
        for row in source_rows:
            cik = normalize_cik(row.get("cik"))
            if not row.get("ticker") or not cik:
                continue
            rows.append(
                {
                    "ticker": row.get("ticker", ""),
                    "price_ticker": row.get("price_ticker", "") or row.get("ticker", ""),
                    "sec_ticker": row.get("sec_ticker", "") or row.get("ticker", ""),
                    "cik": cik,
                    "sector": row.get("sector", ""),
                    "industry": row.get("industry", ""),
                    "name": row.get("name", ""),
                    "notes": row.get("notes", ""),
                    "source_url": row.get("source_url", ""),
                }
            )
        return rows

    rows = []
    for sheet_name, cik_col, name_col in [
        ("missing_cik_170", "correct_cik_to_fill", "company_name"),
        ("suspected_wrong_cik", "correct_cik_to_fill", "expected_company_name"),
    ]:
        for row in sheets.get(sheet_name, []):
            cik = normalize_cik(row.get(cik_col))
            if not row.get("ticker") or not cik:
                continue
            rows.append(
                {
                    "ticker": row.get("ticker", ""),
                    "price_ticker": row.get("price_ticker_to_fill", "") or row.get("ticker", ""),
                    "sec_ticker": row.get("sec_ticker_to_fill", "") or row.get("ticker", ""),
                    "cik": cik,
                    "sector": row.get("sector_if_known", ""),
                    "industry": row.get("industry_if_known", ""),
                    "name": row.get(name_col, ""),
                    "notes": row.get("notes", ""),
                    "source_url": row.get("source_url", ""),
                }
            )
    return rows


def build_period_overrides(sheets: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    rows = []
    for row in sheets.get("period_specific_exceptions", []):
        ticker = row.get("ticker", "")
        cik = normalize_cik(row.get("cik"))
        if not ticker or not cik:
            continue
        start_year, end_year = normalize_year_range(row.get("period"))
        rows.append(
            {
                "ticker": ticker,
                "start_year": start_year,
                "end_year": end_year,
                "price_ticker": ticker,
                "sec_ticker": ticker,
                "cik": cik,
                "sector": "",
                "industry": "",
                "name": row.get("company_name", ""),
                "notes": row.get("note", ""),
                "source_url": row.get("source_url", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    workbook = args.input or find_default_workbook()
    sheets = read_xlsx(workbook)
    general_rows = build_general_overrides(sheets)
    period_rows = build_period_overrides(sheets)

    write_csv(
        config.OVERRIDES_CSV,
        ["ticker", "price_ticker", "sec_ticker", "cik", "sector", "industry", "name", "notes", "source_url"],
        general_rows,
    )
    write_csv(
        config.PERIOD_OVERRIDES_CSV,
        [
            "ticker",
            "start_year",
            "end_year",
            "price_ticker",
            "sec_ticker",
            "cik",
            "sector",
            "industry",
            "name",
            "notes",
            "source_url",
        ],
        period_rows,
    )

    print(f"input: {workbook}")
    print(f"general overrides: {len(general_rows)} -> {config.OVERRIDES_CSV}")
    print(f"period overrides: {len(period_rows)} -> {config.PERIOD_OVERRIDES_CSV}")


if __name__ == "__main__":
    main()
