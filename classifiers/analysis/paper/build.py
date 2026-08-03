"""Compute cross-collection paper tables and curated figures."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .studies import load
from .inputs import load_manifest


COLLECTIONS = ("data_unwrap", "tf_wrapped_a")
CORE_STAGES = ("synthesis", "covariance", "pc-selection")
STAGES = (*CORE_STAGES, "boundaries")


def validate_manifests(results_root: Path,
                       collections: tuple[str, ...]) -> None:
    for collection in collections:
        load_manifest(results_root / collection, collection)


def build_boundaries(results_root: Path, output_root: Path) -> None:
    boundary = load("boundary_common")
    boundary.configure_paths(
        results_root=results_root, output_root=output_root)
    required = [boundary.ARTIFACTS / f"{scope}.npz"
                for scope in boundary.SCOPES]
    required += [boundary.ARTIFACTS / f"{scope}_two_pc_{method}.npz"
                 for scope in boundary.SCOPES
                 for method in ("rbf", "mlp")]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "boundary artifacts are incomplete:\n  - " + formatted)

    refit = load("boundary_refit")
    refit.OUT = boundary.ARTIFACTS
    refit.FIGURES = boundary.FIGURES
    refit.TABLES = boundary.TABLES
    refit.combine()

    for module_name in (
            "boundary_slices",
            "boundary_grid",
            "mlp_boundary_comparison"):
        module = load(module_name)
        module.ARTIFACTS = boundary.ARTIFACTS
        module.FIGURES = boundary.FIGURES
        if hasattr(module, "TABLES"):
            module.TABLES = boundary.TABLES
        module.main()


def build(*, results_root: Path, output_root: Path,
          collections: tuple[str, ...] = COLLECTIONS,
          stages: tuple[str, ...] = CORE_STAGES,
          check_inputs: bool = True) -> Path:
    if set(collections) != set(COLLECTIONS):
        raise ValueError(
            "the current paper analyses require data_unwrap and tf_wrapped_a")
    if check_inputs:
        validate_manifests(results_root, collections)
    output_root.mkdir(parents=True, exist_ok=True)

    completed = []
    if "synthesis" in stages:
        synthesis = load("synthesis")
        synthesis.main(results_root=results_root, output_root=output_root)
        completed.append("synthesis")
    if "covariance" in stages:
        covariance = load("covariance_transfer")
        covariance.main(results_root=results_root, output_root=output_root)
        completed.append("covariance")
    if "pc-selection" in stages:
        pc_selection = load("label_free_pcs")
        pc_selection.main(
            results_root=results_root, output_root=output_root)
        completed.append("pc-selection")
    if "boundaries" in stages:
        build_boundaries(results_root, output_root)
        completed.append("boundaries")

    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collections": list(collections),
        "results_root": str(results_root.resolve()),
        "stages": completed,
    }, indent=2) + "\n")
    return manifest
