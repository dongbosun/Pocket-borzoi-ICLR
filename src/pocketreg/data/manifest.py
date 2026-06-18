"""Small table I/O helpers with optional parquet support."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterable


def _stringify(value) -> str:
    if value is None:
        return ""
    return str(value)


def write_rows_csv_gz(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key)) for key in fieldnames})


def write_rows_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_rows_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_table(rows: list[dict], path: str | Path) -> None:
    """Write parquet when pandas+pyarrow are present, otherwise JSONL fallback."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(rows).to_parquet(path, index=False)
            return
        except Exception:
            # Keep the requested filename for pipeline compatibility; content is JSONL.
            write_rows_jsonl(rows, path)
            return
    if path.suffix == ".jsonl":
        write_rows_jsonl(rows, path)
        return
    raise ValueError(f"Unsupported table path: {path}")


def read_table(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore

            return pd.read_parquet(path).to_dict(orient="records")
        except Exception:
            return read_rows_jsonl(path)
    if path.suffix == ".jsonl":
        return read_rows_jsonl(path)
    raise ValueError(f"Unsupported table path: {path}")


def atomic_write_table(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write_table(rows, tmp)
    tmp.replace(path)
