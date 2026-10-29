"""Save and load a built index as JSON."""

from __future__ import annotations

import json
import os


def save_index(index: dict, path: str) -> None:
    """Persist chunks, idf and vectors so you don't rebuild every time."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f)


def load_index(path: str) -> dict:
    """Load a previously saved index."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(records: list[dict], idf: dict[str, float],
                vectors: list[dict[str, float]]) -> dict:
    """Bundle everything ask-time needs into one serializable dict."""
    return {"chunks": records, "idf": idf, "vectors": vectors}
