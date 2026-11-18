"""Read raw CSVs once, validate, write typed parquet to data/interim/. Prints a short summary."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from fraud.data import IEEE_LABEL, INTERIM, LC_LABEL, load_ieee_raw, load_lc_raw
from fraud.schemas import validate_ieee, validate_lc


def _hash(df) -> str:
    raw = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, loader, validate, label in [
        ("ieee", load_ieee_raw, validate_ieee, IEEE_LABEL),
        ("lc", load_lc_raw, validate_lc, LC_LABEL),
    ]:
        df = validate(loader())
        df.to_parquet(INTERIM / f"{name}.parquet", index=False)
        meta[name] = {
            "rows": len(df),
            "cols": df.shape[1],
            "pos_rate": round(float(df[label].mean()), 5),
            "span_days": round(float(df["t_days"].max()), 1),
            "hash": _hash(df),
        }
        print(name, meta[name])
    (INTERIM / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
