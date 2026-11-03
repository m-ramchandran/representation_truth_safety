"""Inventory each collection's canonical inputs to the paper build."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


REQUIRED_PATTERNS = (
    "per_dataset/tables/metrics_*.csv",
    "leave_out/tables/metrics_*.csv",
    "merged/tables/metrics_*.csv",
    "per_dataset/tables/error_overlap_raw_last_layer*.csv",
    "leave_out/tables/margin_hist_raw_last_layer*_task.csv",
    "coefficients/tables/similarity_*_raw_last_layer*.csv",
    "coefficients/tables/mls_raw_last_layer*.csv",
)


def inventory(root: Path) -> tuple[list[dict], list[str]]:
    files, missing = [], []
    seen = set()
    for pattern in REQUIRED_PATTERNS:
        matches = sorted(root.glob(pattern))
        if not matches:
            missing.append(pattern)
        for path in matches:
            relative = str(path.relative_to(root))
            if relative in seen:
                continue
            seen.add(relative)
            stat = path.stat()
            files.append({
                "path": relative,
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            })
    return files, missing


def _write(root: Path, collection: str, *, status: str,
           files: list[dict] | None = None,
           missing: list[str] | None = None,
           failures: list[str] | None = None) -> Path:
    destination = root / "paper_inputs" / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": collection,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "status": status,
        "files": files or [],
        "missing_patterns": missing or [],
        "failures": failures or [],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def mark_incomplete(root: Path, collection: str,
                    failures: list[str] | None = None) -> Path:
    """Invalidate any older complete manifest while a new run is active."""
    return _write(root, collection, status="incomplete", failures=failures)


def write_manifest(root: Path, collection: str) -> Path:
    files, missing = inventory(root)
    return _write(root, collection, status="complete",
                  files=files, missing=missing)


def load_manifest(root: Path, collection: str) -> dict:
    path = root / "paper_inputs" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run analysis_scripts.run_all for {collection}")
    payload = json.loads(path.read_text())
    if payload.get("collection") != collection:
        raise ValueError(
            f"{path} describes {payload.get('collection')!r}, "
            f"not {collection!r}")
    if payload.get("status") != "complete":
        raise RuntimeError(
            f"{collection} analysis did not complete successfully; see {path}")
    if payload.get("missing_patterns"):
        missing = ", ".join(payload["missing_patterns"])
        raise FileNotFoundError(
            f"{collection} paper inputs were incomplete: {missing}")
    return payload
