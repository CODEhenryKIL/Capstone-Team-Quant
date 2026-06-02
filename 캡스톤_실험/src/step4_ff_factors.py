from __future__ import annotations

import argparse
from io import BytesIO
from zipfile import ZipFile

import requests

import config


def run() -> None:
    config.ensure_dirs()
    print("[step4] Downloading Fama-French 5-factor monthly data")
    response = requests.get(
        config.FF5_MONTHLY_FACTORS_URL,
        headers=config.HTTP_HEADERS,
        timeout=config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        csv_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
        ]
        if not csv_names:
            raise RuntimeError("No CSV file found in Fama-French factor ZIP")
        payload = archive.read(csv_names[0])

    config.FF5_MONTHLY_FACTORS_CSV.write_bytes(payload)
    print(f"[step4] saved: {config.FF5_MONTHLY_FACTORS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
