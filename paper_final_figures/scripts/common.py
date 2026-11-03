"""Shared loaders for the results deep dive.

Builds tidy cross-collection frames from classifiers/results/<collection>/
for all four collections (2 models x 2 framings), mirroring the regime
definitions in the classifier analysis pipeline but
parameterized over collections. Everything is cached under
paper_final_figures/data/.

Layer alignment: Gemma-3-4B (34 layers) probes 17/22/29/33; Qwen2.5-7B
(28 layers) probes 14/18/24/27 at matched relative depth. We map both to
depth positions L1..L4 (~53%, 68%, 88%, 100% of the stack).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "classifiers" / "results"
DATA = REPO / "paper_final_figures" / "data"
FIGS = REPO / "paper_final_figures" / "figures"

COLLECTIONS = {
    ("gemma", "bare"): "gemma_original",
    ("gemma", "true_false"): "gemma_tf_a",
    ("qwen", "bare"): "qwen_original",
    ("qwen", "true_false"): "qwen_tf_a",
}
MODEL_LABEL = {"gemma": "Gemma-3-4B", "qwen": "Qwen2.5-7B"}
PERT_LABEL = {"bare": "Original", "true_false": "True/False framing"}

LAYERS = {"gemma": (17, 22, 29, 33), "qwen": (14, 18, 24, 27)}
# depth position shared across models (fraction of stack, rounded)
DEPTH_OF = {("gemma", 17): 1, ("gemma", 22): 2, ("gemma", 29): 3,
            ("gemma", 33): 4, ("qwen", 14): 1, ("qwen", 18): 2,
            ("qwen", 24): 3, ("qwen", 27): 4}
# depth = (scope layer + 1) / n_layers, i.e. the HF hidden_states index of
# the file actually loaded; layer 4 is the model's final block for both
# models (gemma .529/.676/.882/1.0, qwen .536/.679/.893/1.0)
DEPTH_LABEL = {1: "53%", 2: "68%", 3: "88%", 4: "100%"}

DOMAINS = {
    "true_false": ("got", "common_claim", "counterfact"),
    "medical": ("pubmedqa_artificial", "pubmedqa_labeled", "claude_anatomy",
                "claude_blood_test_ranges"),
    "jailbreak": ("harmful_harmless", "wildjailbreak_vanilla",
                  "wildjailbreak_adversarial", "xstest"),
    "social": ("social_iqa",),
    "statistics": ("claude_vds",),
}
DATASETS = tuple(d for members in DOMAINS.values() for d in members)
DOMAIN_OF = {d: dom for dom, members in DOMAINS.items() for d in members}
# the paper roster excludes blood_test_ranges; we keep a flag either way
PAPER_DATASETS = tuple(d for d in DATASETS if d != "claude_blood_test_ranges")
POOL_EXCLUDED = ("claude_blood_test_ranges", "claude_vds",
                 "wildjailbreak_adversarial")

DATASET_LABEL = {
    "got": "Geometry of Truth", "common_claim": "CommonClaim",
    "counterfact": "CounterFact", "pubmedqa_artificial": "PubMedQA-art",
    "pubmedqa_labeled": "PubMedQA-lab", "claude_anatomy": "Anatomy",
    "claude_blood_test_ranges": "Blood ranges",
    "harmful_harmless": "Harmful/harmless",
    "wildjailbreak_vanilla": "WJB vanilla",
    "wildjailbreak_adversarial": "WJB adversarial", "xstest": "XSTest",
    "social_iqa": "Social IQa", "claude_vds": "VDS",
}
DOMAIN_RENAME = {"tf": "true_false", "medical": "medical",
                 "jailbreak": "jailbreak", "social_iqa": "social",
                 "claude_vds": "statistics",
                 "claude_blood_test_ranges": "medical"}

LINEAR_METHODS = ("diff", "l1reg", "l2reg", "svm_linear", "lda")
NONLINEAR_METHODS = ("qda", "knn", "svm_rbf", "mlp", "xgboost")
METHODS = LINEAR_METHODS + NONLINEAR_METHODS
POOLED_EXTRA = ("random_slopes", "random_slopes_task", "moe", "moe_hard")
FAMILY_OF = {**{m: "linear" for m in LINEAR_METHODS},
             **{m: "nonlinear" for m in NONLINEAR_METHODS},
             **{m: "pooled_extra" for m in POOLED_EXTRA}}
METHOD_LABEL = {
    "diff": "Diff-in-means", "l1reg": "L1 logistic", "l2reg": "L2 logistic",
    "svm_linear": "Linear SVM", "lda": "LDA", "qda": "QDA", "knn": "kNN",
    "svm_rbf": "RBF SVM", "mlp": "MLP", "xgboost": "XGBoost",
    "random_slopes": "Random slopes (dataset)",
    "random_slopes_task": "Random slopes (domain)",
    "moe": "Mixture of experts", "moe_hard": "MoE (hard)",
}

REGIMES = ("within_cv", "single_source", "merged_included",
           "leave_dataset_out", "leave_domain_out")
REGIME_LABEL = {
    "within_cv": "Within target (CV)",
    "single_source": "One source dataset",
    "merged_included": "Pooled, target included",
    "leave_dataset_out": "Target dataset absent",
    "leave_domain_out": "Entire domain absent",
}


def _read(pattern: str) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def long_results(feature_prefix: str = "raw_last",
                 keep_all_methods: bool = False,
                 roster: tuple = PAPER_DATASETS) -> pd.DataFrame:
    """One row per (model, perturbation, regime, dataset, method, layer).

    `roster` is the set of datasets kept, as targets and as single-source
    training sets. It defaults to the paper roster (PAPER_DATASETS), which
    drops claude_blood_test_ranges everywhere -- the 2026-07-22 decision in
    classifiers/Planning.md; pass DATASETS to keep all thirteen units.

    single_source keeps one row per (source, target) — collapse later.
    Mirrors from_results.long_results, including the eval_only handling.
    """
    rows = []

    def add(frame, *, regime, dataset_col, model, pert, **extra):
        if frame.empty:
            return
        keep = frame[frame[dataset_col].isin(roster)]
        if not keep_all_methods and "method" in keep:
            keep = keep[keep.method.isin(METHODS)]
        if keep.empty:
            return
        keep = keep.assign(regime=regime, dataset=keep[dataset_col],
                           model=model, perturbation=pert, **extra)
        rows.append(keep)

    for (model, pert), coll in COLLECTIONS.items():
        root = RESULTS / coll
        pd_ = _read(str(root / f"per_dataset/tables/metrics_{feature_prefix}_layer*.csv"))
        if not pd_.empty:
            add(pd_[pd_.kind == "cv"], regime="within_cv",
                dataset_col="train_dataset", model=model, pert=pert)
            tr = pd_[pd_.kind == "transfer"]
            tr = tr[tr.train_dataset.isin(roster)]
            add(tr, regime="single_source", dataset_col="test_dataset",
                model=model, pert=pert, source_dataset=tr.train_dataset)
        mg = _read(str(root / f"merged/tables/metrics_{feature_prefix}_layer*.csv"))
        if not mg.empty:
            add(mg[mg.kind == "per_dataset"], regime="merged_included",
                dataset_col="test_dataset", model=model, pert=pert)
            add(mg[mg.kind == "eval_only"], regime="leave_dataset_out",
                dataset_col="test_dataset", model=model, pert=pert)
        ldo = _read(str(root / f"leave_out/tables/metrics_{feature_prefix}_layer*_dataset.csv"))
        if not ldo.empty:
            add(ldo[ldo.holdout == ldo.test_dataset], regime="leave_dataset_out",
                dataset_col="test_dataset", model=model, pert=pert)
        ldt = _read(str(root / f"leave_out/tables/metrics_{feature_prefix}_layer*_task.csv"))
        if not ldt.empty:
            target_domain = ldt.test_dataset.map(DOMAIN_OF)
            heldout_domain = ldt.holdout.map(DOMAIN_RENAME)
            add(ldt[target_domain == heldout_domain], regime="leave_domain_out",
                dataset_col="test_dataset", model=model, pert=pert)
            if not mg.empty:
                # permanently pool-excluded units whose whole domain has no
                # other pool member: vds (statistics). Its eval_only row is
                # also its entire-domain-absent row.
                add(mg[(mg.kind == "eval_only")
                       & (mg.test_dataset.map(DOMAIN_OF) == "statistics")],
                    regime="leave_domain_out", dataset_col="test_dataset",
                    model=model, pert=pert)

    out = pd.concat(rows, ignore_index=True)
    out["domain"] = out.dataset.map(DOMAIN_OF)
    out["family"] = out.method.map(FAMILY_OF)
    out["auc"] = pd.to_numeric(out.auc, errors="coerce")
    out["depth"] = out.apply(lambda r: DEPTH_OF.get((r.model, r.layer)), axis=1)
    return out.dropna(subset=["auc", "domain", "depth"])


def collapse_single_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Median over source datasets for single_source rows (one value per
    target), leaving other regimes untouched."""
    single = frame[frame.regime == "single_source"]
    other = frame[frame.regime != "single_source"]
    keys = ["model", "perturbation", "dataset", "domain", "regime",
            "method", "family", "layer", "depth"]
    single = single.groupby(keys, as_index=False).auc.median()
    return pd.concat([other, single], ignore_index=True)


def cached(name: str, builder, refresh: bool = False) -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    frame = builder()
    frame.to_csv(path, index=False)
    return frame


def dataset_sizes() -> pd.Series:
    """n per dataset per collection, from per_dataset CV rows."""
    f = long_results()
    cv = f[f.regime == "within_cv"]
    return cv.groupby(["model", "perturbation", "dataset"]).n_test.max()
