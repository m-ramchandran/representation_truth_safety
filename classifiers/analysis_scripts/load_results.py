"""Shared loader for the runners' outputs under classifiers/results/.

Metrics come back as pandas DataFrames (the csvs already carry features /
layer as columns, so concatenating the per-(tag, layer) files gives the
full table). Predictions come back as plain dicts of numpy arrays; the
`params` / `full_params` entries, stored as `repr` strings, are parsed
back into Python objects.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import COLLECTIONS, DEFAULT_COLLECTION

RESULTS = Path(__file__).parents[1] / "results"

EXPERIMENTS = ("per_dataset", "leave_out", "merged")


def add_data_args(ap: argparse.ArgumentParser) -> None:
    """The --data / --root pair every analysis CLI takes. Results live
    under results/<collection>/<experiment>/, so --root defaults to
    results/<data>; an explicit --root (e.g. a fixture tree) wins."""
    ap.add_argument("--data", default=DEFAULT_COLLECTION, choices=sorted(COLLECTIONS))
    ap.add_argument("--root", type=Path, default=None,
                    help="results root (default: results/<data>)")


def resolve_root(args: argparse.Namespace) -> Path:
    return args.root if args.root is not None else RESULTS / args.data


def analysis_dirs(root: Path, experiment: str = None) -> tuple[Path, Path]:
    """Where an analysis module writes: (tables, figures), both created.
    Experiment-scoped output joins the runner's own tables/ and figures/
    under <root>/<experiment>/ (user reorg 2026-07-23 — no nested
    analysis/ level; runner and analysis files share the two
    directories, told apart by name: metrics_* / auc_* are the
    runner's). The cross-experiment coefficient modules (experiment
    None) write under <root>/coefficients/."""
    base = root / experiment if experiment else root / "coefficients"
    tables, figures = base / "tables", base / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def load_metrics(experiment: str, root: Path = RESULTS) -> pd.DataFrame:
    """Every metrics csv of one experiment concatenated into one frame,
    with `auc` numeric. `experiment` is "per_dataset", "leave_out", or
    "merged". Only the runner's metrics_*.csv are read — analysis csvs
    share the same tables/ directory since the 2026-07-23 reorg."""
    if experiment not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {experiment!r}; "
                       f"known: {EXPERIMENTS}")
    paths = sorted((root / experiment / "tables").glob("metrics_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no metrics csvs under "
                                f"{root / experiment / 'tables'}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.rename(columns={"variant": "features"})   # pre-2026-07-20 csvs
    df["auc"] = df["auc"].astype(float)
    return df


def auc_or_nan(y, proba) -> float:
    """AUC, NaN when the slice has a single class — the one spelling of
    that convention (errors / margins / per_slice all score slices that
    can be single-class)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return roc_auc_score(y, proba) if 0 < y.mean() < 1 else np.nan


def is_wrong(proba, y) -> np.ndarray:
    """The shared error predicate: predicted positive at 0.5 vs y == 1
    (the label convention set in scripts/)."""
    return (np.asarray(proba) >= 0.5) != (np.asarray(y) == 1)


def load_prediction(path, keys=None) -> dict:
    """One prediction npz as {key: numpy array}, with `params` (and, for
    merged, `full_params`) parsed from their repr back into Python
    objects (a dict, or for merged's `params` a per-outer-fold list of
    dicts). `keys` limits which arrays are decompressed (params /
    full_params always included) — the per-sample score arrays dwarf
    everything else, so callers that only need coefficients or
    hyperparameters should pass the keys they read."""
    with np.load(path, allow_pickle=True) as d:
        names = d.files if keys is None else \
            [k for k in d.files
             if k in keys or k in ("params", "full_params")]
        out = {k: d[k] for k in names}
    for key in ("params", "full_params"):
        if key in out:
            out[key] = ast.literal_eval(str(out[key]))
    return out


def load_booster(d: dict, prefix: str = ""):
    """An xgboost fit reconstituted from the `booster_ubj` bytes its npz
    carries (`prefix` is "" or "full_" for the merged run's whole-corpus
    fit) — the one method whose predictions cannot be rebuilt from a
    summary vector, so the booster is saved instead. Returns an
    xgboost.Booster; score new data with
    `booster.predict(xgboost.DMatrix(X), output_margin=True)` for the
    log-odds margin (proba = sigmoid of it), on data put through the
    fit's own featurization (directions.apply_featurization) — the
    booster takes raw feature values, with no standardization in front.
    xgboost is imported lazily, so this module still loads without it."""
    import xgboost

    key = prefix + "booster_ubj"
    if key not in d:
        raise KeyError(f"no {key!r} in the npz (not an xgboost fit, or a "
                       "run from before 2026-07-24)")
    booster = xgboost.Booster()
    booster.load_model(bytearray(np.asarray(d[key], dtype=np.uint8).tobytes()))
    return booster


def npz_path(experiment: str, tag: str, layer: int, method: str,
             dataset: str = None, root: Path = RESULTS) -> Path:
    """The prediction npz for one fit. Method and dataset names contain
    underscores, so filenames are built from the metrics-csv columns
    rather than parsed: `dataset` is the training dataset (per_dataset)
    or the held-out unit (leave_out); merged has one file per method."""
    name = f"{tag}_layer{layer}_{method}"
    if experiment != "merged":
        name += f"_{dataset}"
    return root / experiment / "predictions" / f"{name}.npz"


def cv_predictions(experiment: str, tag: str, layer: int,
                   root: Path = RESULTS) -> pd.DataFrame:
    """The CV predictions of every method run at this (tag, layer), as
    one long frame: sample_id / dataset / topic / y / method / proba /
    margin (margin NaN where the method has none; topic "" where the
    npz stores no cv_topic). Works for per_dataset (one file per
    training dataset) and merged (dataset read from cv_dataset);
    leave_out's pools overlap across holdouts, so it has no single
    per-sample CV frame."""
    if experiment == "leave_out":
        raise ValueError("leave_out has no unique CV prediction per "
                         "sample; read its npz files directly")
    m = load_metrics(experiment, root)
    m = m[(m["features"] == tag) & (m["layer"] == layer)]
    runs = (m[["method", "train_dataset"]].drop_duplicates().itertuples()
            if experiment == "per_dataset"
            else ((r, None) for r in m["method"].unique()))
    frames = []
    keys = {"cv_sample_id", "cv_dataset", "cv_topic", "cv_y", "cv_proba",
            "cv_margin"}
    for run in runs:
        method, dataset = (run.method, run.train_dataset) \
            if experiment == "per_dataset" else run
        d = load_prediction(npz_path(experiment, tag, layer, method,
                                     dataset, root), keys=keys)
        frames.append(pd.DataFrame({
            "sample_id": d["cv_sample_id"],
            "dataset": d["cv_dataset"] if experiment == "merged" else dataset,
            "topic": d.get("cv_topic",
                           np.full(len(d["cv_y"]), "")).astype(str),
            "y": d["cv_y"].astype(int),
            "method": method,
            "proba": d["cv_proba"],
            "margin": d.get("cv_margin", np.full(len(d["cv_y"]), np.nan)),
        }))
    if not frames:
        raise FileNotFoundError(f"no {experiment} runs for tag={tag!r} "
                                f"layer={layer}")
    return pd.concat(frames, ignore_index=True)


def _platt(margin: np.ndarray, params, fold: np.ndarray = None):
    """`margin` on the log-odds scale: as stored for methods whose margin
    is already a native log-odds, through the Platt (a, b) otherwise.
    merged's `params` is a per-outer-fold list, so each CV sample takes
    its own fold's (a, b) — one shared (a, b) would put the folds on
    scales that differ by a constant, and |log-odds| ranks (what the
    confidence quantiles are built from) are not invariant to that."""
    if isinstance(params, dict):
        return params["platt_a"] * margin + params["platt_b"] \
            if "platt_a" in params else margin
    if not isinstance(params, list) or fold is None \
            or "platt_a" not in params[0]:
        return margin
    a = np.array([params[f]["platt_a"] for f in fold])
    b = np.array([params[f]["platt_b"] for f in fold])
    return a * margin + b


def _score_frame(d: dict, prefix: str, base: dict) -> pd.DataFrame:
    """One npz's scores under `prefix` ("cv" / "transfer" / "holdout" /
    "eval_only") as a frame, with the margin also on the log-odds scale
    ("logodds"; NaN when there is no margin at all — knn). eval_only
    scores come from the whole-corpus fit, so they calibrate through
    `full_params`, not the per-fold nested-CV ones."""
    n = len(d[f"{prefix}_y"])
    margin = d.get(f"{prefix}_margin", np.full(n, np.nan))
    params = d["full_params"] if prefix == "eval_only" and "full_params" in d \
        else d["params"]
    logodds = _platt(margin, params,
                     d.get("cv_fold") if prefix == "cv" else None)
    return pd.DataFrame({
        **base,
        "dataset": d.get(f"{prefix}_dataset", base.get("fit")),
        "sample_id": d[f"{prefix}_sample_id"],
        "topic": d.get(f"{prefix}_topic",
                       np.full(n, "")).astype(str),
        "y": d[f"{prefix}_y"].astype(int),
        "proba": d[f"{prefix}_proba"],
        "margin": margin, "logodds": logodds,
    })


def eval_predictions(experiment: str, tag: str, layer: int,
                     root: Path = RESULTS,
                     with_held: bool = True) -> tuple[pd.DataFrame,
                                                      pd.DataFrame]:
    """Each fit's CV predictions next to its predictions on data it
    never trained on, as two long frames (cv, held). per_dataset: fit =
    the training dataset, held = its transfer_* scores; leave_out: fit =
    "<held-out unit> (dataset|task)" (the CV frame is the training
    pool's), held = its holdout_* scores; merged: fit = "pooled", and
    since nested CV already scores every sample with a fit that did not
    train on it, held = the CV frame itself (one evaluation per dataset
    slice) plus any eval_only_* scores. Columns as in cv_predictions
    plus fit / logodds
    (and holdout_kind for leave_out). `with_held=False` returns
    (cv, None) and never decompresses the transfer/holdout arrays —
    the largest in the npz — for callers that only read the CV side."""
    m = load_metrics(experiment, root)
    m = m[(m["features"] == tag) & (m["layer"] == layer)]
    if experiment == "per_dataset":
        runs = [(r.method, r.train_dataset, {})
                for r in m[["method", "train_dataset"]]
                .drop_duplicates().itertuples()]
        prefix = "transfer"
    elif experiment == "leave_out":
        # the fit label carries the kind: a name can be held out both as
        # a dataset and as a single-dataset task (social_iqa, claude_vds
        # in the original collection), and those are two different
        # learners — labelling both by the bare holdout name silently
        # pooled their samples wherever a caller grouped by fit
        runs = [(r.method, f"{r.holdout_kind}_{r.holdout}",
                 {"fit": f"{r.holdout} ({r.holdout_kind})",
                  "holdout_kind": r.holdout_kind})
                for r in m[["method", "holdout_kind", "holdout"]]
                .drop_duplicates().itertuples()]
        prefix = "holdout"
    else:                                   # merged
        runs = [(method, None, {"fit": "pooled"})
                for method in m["method"].unique()]
        prefix = "eval_only"
    keys = {f"{p}_{k}" for p in (("cv", prefix) if with_held else ("cv",))
            for k in ("dataset", "sample_id", "topic", "y", "proba",
                      "margin")}
    keys.add("cv_fold")                     # merged's per-fold Platt
    cv, held = [], []
    for method, name, extra in runs:
        d = load_prediction(npz_path(experiment, tag, layer, method, name,
                                     root), keys=keys)
        base = {"fit": name, "method": method, **extra}
        cv.append(_score_frame(d, "cv", base))
        if not with_held:
            continue
        if experiment != "merged":
            # A pool_excluded unit's per_dataset fit is evaluated on
            # nothing but its own CV folds — no transfer arrays exist, by
            # design — so it contributes a CV frame and no held frame.
            if f"{prefix}_y" in d:
                held.append(_score_frame(d, prefix, base))
            continue
        # the pooled learner has no transfer set: the data it never
        # trained on IS its nested-CV samples, so each dataset's slice of
        # the CV frame is one evaluation. Eval-only units (scored by the
        # whole-corpus fit, never in any training pool) join them.
        held.append(cv[-1])
        if f"{prefix}_y" in d:
            held.append(_score_frame(d, prefix, base))
    if not cv:
        raise FileNotFoundError(f"no {experiment} runs for tag={tag!r} "
                                f"layer={layer}")
    return (pd.concat(cv, ignore_index=True),
            pd.concat(held, ignore_index=True) if with_held else None)
