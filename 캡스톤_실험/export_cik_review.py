from __future__ import annotations

import csv
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

import config


MANUAL_DIR = config.DATA_DIR / "manual"
MISSING_CSV = MANUAL_DIR / "sp500_missing_cik_170.csv"
SUSPECT_CSV = MANUAL_DIR / "sp500_suspected_wrong_cik.csv"
OVERRIDE_TEMPLATE_CSV = MANUAL_DIR / "ticker_overrides_template.csv"
REVIEW_XLSX = MANUAL_DIR / "sp500_cik_review.xlsx"


def column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def safe_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[][\\/*?:]", "_", name)
    return cleaned[:31] or "Sheet"


def cell_xml(row_idx: int, col_idx: int, value: object) -> str:
    ref = f"{column_letter(col_idx)}{row_idx}"
    if value is None or pd.isna(value):
        return f'<c r="{ref}"/>'
    text = escape(str(value), {'"': "&quot;"})
    preserve = ' xml:space="preserve"' if text.strip() != text else ""
    return f'<c r="{ref}" t="inlineStr"><is><t{preserve}>{text}</t></is></c>'


def sheet_xml(headers: list[str], rows: list[dict[str, object]]) -> str:
    total_rows = len(rows) + 1
    total_cols = len(headers)
    dimension = f"A1:{column_letter(total_cols)}{max(total_rows, 1)}"
    widths = "".join(
        f'<col min="{idx}" max="{idx}" width="{min(max(len(header) + 2, 12), 36)}" customWidth="1"/>'
        for idx, header in enumerate(headers, start=1)
    )

    xml_rows = [
        '<row r="1">'
        + "".join(cell_xml(1, idx, header) for idx, header in enumerate(headers, start=1))
        + "</row>"
    ]
    for row_idx, row in enumerate(rows, start=2):
        xml_rows.append(
            f'<row r="{row_idx}">'
            + "".join(cell_xml(row_idx, idx, row.get(header, "")) for idx, header in enumerate(headers, start=1))
            + "</row>"
        )

    autofilter = f'<autoFilter ref="A1:{column_letter(total_cols)}{total_rows}"/>' if headers else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        "</sheetView></sheetViews>"
        f"<cols>{widths}</cols>"
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        f"{autofilter}"
        "</worksheet>"
    )


def write_xlsx(path: Path, sheets: list[tuple[str, list[str], list[dict[str, object]]]]) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, len(sheets) + 1)
    )
    workbook_sheets = "".join(
        f'<sheet name="{escape(safe_sheet_name(name), {chr(34): "&quot;"})}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, (name, _, _) in enumerate(sheets, start=1)
    )
    workbook_rels = "".join(
        f'<Relationship Id="rId{idx}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, len(sheets) + 1)
    )
    style_rid = len(sheets) + 1

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" '
            'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            f"{sheet_overrides}</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
            'Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
            'Target="docProps/app.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "docProps/core.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:creator>capstone_export</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
            "</cp:coreProperties>",
        )
        archive.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>capstone_export</Application>"
            "</Properties>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets>"
            "</workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{workbook_rels}"
            f'<Relationship Id="rId{style_rid}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
            "<fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>"
            "<borders count=\"1\"><border/></borders>"
            "<cellStyleXfs count=\"1\"><xf/></cellStyleXfs>"
            "<cellXfs count=\"1\"><xf xfId=\"0\"/></cellXfs>"
            "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>"
            "</styleSheet>",
        )
        for idx, (_, headers, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(headers, rows))


def write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def non_unknown(values: pd.Series) -> str:
    cleaned = [
        str(value).strip()
        for value in values.dropna().tolist()
        if str(value).strip() and str(value).strip().lower() not in {"nan", "unknown"}
    ]
    return cleaned[-1] if cleaned else ""


def load_expected_company(ticker: str, membership: pd.DataFrame) -> dict[str, object]:
    group = membership[membership["ticker"] == ticker]
    years = sorted(int(year) for year in group["year"].dropna().unique())
    return {
        "name": non_unknown(group["name"]) or ticker,
        "sector": non_unknown(group["sector"]),
        "industry": non_unknown(group["industry"]),
        "first_year": years[0] if years else "",
        "last_year": years[-1] if years else "",
        "year_count": len(years),
        "sp500_years": ",".join(str(year) for year in years),
    }


def load_sec_entity_name(cik: str) -> str:
    cache_path = config.SEC_CACHE_DIR / f"CIK{str(cik).zfill(10)}.json"
    if not cache_path.exists():
        return ""
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return str(payload.get("entityName") or "")


def build_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    dropped = json.loads(config.SP500_DROPPED_JSON.read_text(encoding="utf-8"))
    membership = pd.read_csv(config.SP500_MEMBERSHIP_CSV)
    membership["ticker"] = membership["ticker"].astype(str)

    missing_rows = []
    override_rows = []
    for item in sorted(dropped, key=lambda row: row["ticker"]):
        ticker = item["ticker"]
        expected = load_expected_company(ticker, membership)
        missing_rows.append(
            {
                "ticker": ticker,
                "company_name": expected["name"],
                "first_year": expected["first_year"],
                "last_year": expected["last_year"],
                "year_count": expected["year_count"],
                "sp500_years": expected["sp500_years"],
                "reason": item["reason"],
                "correct_cik_to_fill": "",
                "sec_entity_name_check": "",
                "price_ticker_to_fill": ticker,
                "sec_ticker_to_fill": ticker,
                "sector_if_known": expected["sector"],
                "industry_if_known": expected["industry"],
                "notes": "",
            }
        )
        override_rows.append(
            {
                "ticker": ticker,
                "cik": "",
                "price_ticker": ticker,
                "sec_ticker": ticker,
                "sector": expected["sector"],
                "industry": expected["industry"],
                "name": expected["name"],
                "notes": "",
            }
        )

    suspect_rows = []
    if config.FUNDAMENTALS_FAILURES_CSV.exists():
        failures = pd.read_csv(config.FUNDAMENTALS_FAILURES_CSV)
        for _, failure in failures.iterrows():
            ticker = str(failure["ticker"])
            expected = load_expected_company(ticker, membership)
            cik = str(failure.get("cik", "")).split(".", 1)[0].zfill(10)
            suspect_rows.append(
                {
                    "ticker": ticker,
                    "expected_company_name": expected["name"],
                    "mapped_cik_now": cik,
                    "sec_cache_entity_name_now": load_sec_entity_name(cik),
                    "first_year": expected["first_year"],
                    "last_year": expected["last_year"],
                    "sp500_years": expected["sp500_years"],
                    "failure_reason": failure.get("reason", ""),
                    "correct_cik_to_fill": "",
                    "price_ticker_to_fill": ticker,
                    "sec_ticker_to_fill": ticker,
                    "notes": "Likely historical ticker reuse or wrong SEC mapping.",
                }
            )

    return missing_rows, suspect_rows, override_rows


def main() -> None:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    missing_rows, suspect_rows, override_rows = build_rows()

    missing_headers = [
        "ticker",
        "company_name",
        "first_year",
        "last_year",
        "year_count",
        "sp500_years",
        "reason",
        "correct_cik_to_fill",
        "sec_entity_name_check",
        "price_ticker_to_fill",
        "sec_ticker_to_fill",
        "sector_if_known",
        "industry_if_known",
        "notes",
    ]
    suspect_headers = [
        "ticker",
        "expected_company_name",
        "mapped_cik_now",
        "sec_cache_entity_name_now",
        "first_year",
        "last_year",
        "sp500_years",
        "failure_reason",
        "correct_cik_to_fill",
        "price_ticker_to_fill",
        "sec_ticker_to_fill",
        "notes",
    ]
    override_headers = [
        "ticker",
        "cik",
        "price_ticker",
        "sec_ticker",
        "sector",
        "industry",
        "name",
        "notes",
    ]
    instruction_headers = ["field", "meaning"]
    instruction_rows = [
        {
            "field": "correct_cik_to_fill / cik",
            "meaning": "Fill the historical company's correct SEC CIK as a 10-digit text value.",
        },
        {
            "field": "price_ticker_to_fill / price_ticker",
            "meaning": "Usually keep the historical ticker. Change only if Yahoo uses a different symbol.",
        },
        {
            "field": "sec_ticker_to_fill / sec_ticker",
            "meaning": "Usually keep the historical ticker. It is mainly used for override bookkeeping.",
        },
        {
            "field": "sec_entity_name_check",
            "meaning": "Optional manual check that the CIK's SEC entity name matches company_name.",
        },
        {
            "field": "After filling",
            "meaning": "Save the override_template sheet as data/raw/ticker_overrides.csv, then rerun step0-step3.",
        },
    ]

    write_csv(MISSING_CSV, missing_headers, missing_rows)
    write_csv(SUSPECT_CSV, suspect_headers, suspect_rows)
    write_csv(OVERRIDE_TEMPLATE_CSV, override_headers, override_rows)
    write_xlsx(
        REVIEW_XLSX,
        [
            ("missing_cik_170", missing_headers, missing_rows),
            ("suspected_wrong_cik", suspect_headers, suspect_rows),
            ("override_template", override_headers, override_rows),
            ("instructions", instruction_headers, instruction_rows),
        ],
    )

    print(f"missing CIK rows: {len(missing_rows)}")
    print(f"suspected wrong CIK rows: {len(suspect_rows)}")
    print(f"wrote: {REVIEW_XLSX}")
    print(f"wrote: {MISSING_CSV}")
    print(f"wrote: {SUSPECT_CSV}")
    print(f"wrote: {OVERRIDE_TEMPLATE_CSV}")


if __name__ == "__main__":
    main()
