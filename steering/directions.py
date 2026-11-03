"""Build every direction for the replacement steering run — saved fits only.

Runs on compute cluster (classifiers/results and the activation cache live there).
Outputs: directions_layer{L}.npz, manifest_layer{L}.csv,
directions_meta_layer{L}.json under steering/out/.

Reported configuration:

  * Five targets: common_claim, claude_vds, wildjailbreak_vanilla, xstest,
    pubmedqa_labeled. wildjailbreak_adversarial is dropped.
  * Every fitted coefficient vector — single-dataset, pooled, held-out, and
    the random-slopes rule (shared + that dataset's adjustment combined
    into ONE vector first) — is converted into a movement by multiplying
    with the TARGET dataset's own covariance: the plain covariance of that
    dataset's stored layer-L vectors, both classes together, no class or
    cell centering, regularized with Ledoit-Wolf shrinkage. Mean
    differences are movements already and are not converted.
  * Every real direction is oriented so that adding it pushes toward the
    class whose correct answer is "True" (label 1). Orientation is checked
    on the target's own rows; a direction that natively points the wrong
    way is flipped and flagged in the manifest (`flipped`), because a
    held-out fit that anti-correlates with its target is itself a finding.
  * Controls, per target: 8 draws of covariance-times-random at EACH
    power (the null for the fits converted at that power), 8
    label-shuffled mean differences (same estimator and rows as the real
    mean difference with labels destroyed),
    and the top-variance direction with the dataset's own mean difference
    projected out. Eight plain random vectors, shared across targets, are
    the p=0 rung of the covariance-times-random null.
  * The pooled-vs-per-dataset blending grid is excluded.

Additional Qwen layer-24 configuration:

  * --collection / --targets, so the same build serves both models.
  * wildjailbreak_adversarial is a target again. Gemma would not answer its
    True/False question (0.602), but Qwen does (0.877), so the cell is
    live on this model. It is pool-excluded like claude_vds, so its
    in-pool fits come from the expanded pool and its held-out fits from the
    standard pool, which never contained it. Unlike claude_vds it also
    gets a family-held-out fit, and that is the only number to trust for
    it: the vanilla and adversarial halves share source ids, so any fit
    whose pool contains vanilla is optimistic about adversarial.
  * A COVARIANCE POWER SWEEP replaces the single covariance
    multiplication. `p` is an exponent applied through the covariance's
    eigendecomposition: raise its eigenvalues to the power p and rebuild.
    p=0 leaves the coefficients alone, p=1 is one multiplication (what the
    Gemma run did), p=0.5 and p=2 are half and twice that.

    Do NOT justify this as "the multiplication undoes the fit and returns
    the class-mean difference". That identity needs an unregularized
    logistic fit, homoscedastic Gaussian classes, and the WITHIN-class
    covariance; these fits are ridge-penalized on standardized features
    and the penalty strength is not even saved, so the fit's position
    between the two limits is unknown. The angle cannot settle it either:
    covariance-times-RANDOM reaches cosine 0.586 median / 0.982 max with
    the mean difference (14 draws, 2026-08-16), against 0.917 for mapped
    fits — a high cosine is mostly a statement about the covariance having
    few effective dimensions with the mean difference near the leading
    one. What separated fits from noise there was whether the converted
    vector still SEPARATES THE CLASSES: AUC 0.838 vs 0.42.

    What the Gemma run actually measured is narrower and is a measurement,
    not a mechanism: every converted fit steered like the TARGET'S OWN
    mean difference whatever pool it came from (wildjailbreak_vanilla,
    per 1000 units: own 1.41, pooled 4.95, converted pooled 1.40; xstest
    0.53 / 4.70 / 0.59), while covariance-times-random steered nearly as
    well. That fits substitution rather than recovery — the multiplication
    replaces most of the input with the target's own leading geometry.

    The sweep turns that into a curve and tests it. If the substitution
    account is right, p=0 is the only rung where a pooled fit can behave
    differently from an own fit, which makes p=0 the most informative rung
    rather than a formality. Every power gets its own matched null: the
    same transform applied to random vectors, eight draws at each.
  * Coefficients with the class-mean-difference component projected out are
    excluded.

Mid-depth configuration (Gemma L22, Qwen L18):

  * --cov-centering class: the dial uses the pooled WITHIN-class
    covariance, not the total one. With balanced classes the total
    covariance contains the class separation itself as variance (a quarter
    of d d^T for mean difference d), so its leading eigenvector nearly
    coincides with the mean difference and multiplying by it pulls ANY
    vector toward that difference -- which is what the L29/L24 runs
    measured. Class-centering removes that term. It does NOT fully
    decouple the two: the within-class leading direction still sits at
    cosine 0.92-0.97 to the mean difference on six of eight cells, so
    covariance-times-random still drifts, just less.
  * --dial ridge REPLACES the multiplier rungs. Instead of Sigma^p w, the
    map is ((1-t) * Shat + t * I)^(1/2) w, where Shat is the covariance
    rescaled to mean eigenvalue 1 and t runs over RIDGE_TS. The rescaling
    is what makes t meaningful: on the raw covariance every eigenvalue
    dwarfs 1, so adding the identity would do nothing. t = 1 is the
    coefficients exactly as fitted and t = 0 is Sigma^(1/2) w, so the path
    contains two of the old rungs as its endpoints and nothing beyond
    them. Both models are run this way; the two power-dial runs at L29/L24
    stay as they were.
  * Mid-depth runs drop claude_vds and wildjailbreak_adversarial: the
    expanded-pool fits that serve them were never made at those layers, and
    nothing is ever refit for this experiment.

Nothing is refit: fitted vectors come from saved fit files (verified
unchanged at the end); covariances, orientations, and the shuffled-label
control are summary statistics of the stored activation cache.

    python steering/directions.py [--layer 29]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "classifiers" / "results"
OUT = REPO / "steering" / "out"

import sys
sys.path.insert(0, str(REPO))
from classifiers.data import _load_units  # noqa: E402
import steering.manifest_schema as MS  # noqa: E402

DEFAULT_COLLECTION = "gemma_tf_a"
# Set from --collection in main(); the expanded pool is the same tree with a
# suffix and supplies the datasets the standard pool never contained.
COLLECTION = DEFAULT_COLLECTION
EXPANDED_POOL = f"{COLLECTION}_expanded_pool"
# tag added to every output file, so the two models' builds sit side by side
OUT_TAG = ""
# which covariance the dial uses; set from --cov-centering in main()
COV_CENTERING = "total"
# which family of maps the dial applies to the fits; --dial in main()
DIAL = "power"
# t values for the ridge dial: ((1-t) Shat + t I)^(1/2) w, Shat the
# covariance rescaled to mean eigenvalue 1. t=0 is half of one
# covariance multiplication, t=1 is the coefficients exactly as fitted.
# Grid used for the reported runs: mostly even in t, with 0.9 kept as
# the one point near the identity end. 0.99 was dropped as too close
# to t=1 to carry its own information.
RIDGE_TS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)

TARGETS = ("common_claim", "claude_vds", "wildjailbreak_vanilla",
           "xstest", "pubmedqa_labeled", "wildjailbreak_adversarial")
SHORT = {"common_claim": "cc", "claude_vds": "vds",
         "wildjailbreak_vanilla": "wjv", "xstest": "xs",
         "pubmedqa_labeled": "pql", "wildjailbreak_adversarial": "wja"}
# Datasets that never enter the standard training pool, so their in-pool
# fits have to come from the expanded pool instead.
POOL_EXCLUDED = ("claude_vds", "wildjailbreak_adversarial")
# powers p in Sigma^p w; 0 = the raw coefficients, 1 = the Gemma run
POWERS = (0.0, 0.5, 1.0, 2.0)
POWER_TAG = {0.0: "p0", 0.5: "p05", 1.0: "p1", 2.0: "p2"}
# Eight draws at every rung. The bands are compared
# across powers and across models, and "how many null draws beat the real
# direction" is only meaningful if every band is built from the same
# number of draws. p=0 draws 0 here because Sigma^0 x random IS a plain
# random vector, and the eight plain randoms below already serve every
# target -- so the p=0 band is also eight deep.
POWER_DRAWS = {0.0: 0, 0.5: 8, 1.0: 8, 2.0: 8}
N_PLAIN_RANDOM = 8      # the p=0 null, shared across targets
N_SHUFDIFF = 8
# Task family used for the task-level random-slopes adjustment and for the
# family-held-out fits. claude_vds is natively a true/false dataset, so it
# uses the tf adjustment; it belongs to no task
# in the pool, so it gets no family-held-out block (its held-out block is
# the standard pool, which never contained it at all).
# Two datasets whose prompts are rewrites of each other's, same labels.
# A training set that contains one of them is NOT clean for the other, even
# though the other was never in it by name.
SIBLING = {"wildjailbreak_adversarial": "wildjailbreak_vanilla"}
TASK = {"common_claim": "tf", "claude_vds": "tf",
        "wildjailbreak_vanilla": "jailbreak", "xstest": "jailbreak",
        "pubmedqa_labeled": "medical", "wildjailbreak_adversarial": "jailbreak"}


# ---- readers (unchanged conventions) --------------------------------------

READ: list[Path] = []   # every fit file opened, for the nothing-changed check


def cache_units(z, key: str, prefix: str = "") -> np.ndarray:
    """coef -> raw activation units: divide by the fit's stored scaler when
    it has one (diff-in-means stores none — already in raw units)."""
    w = np.asarray(z[prefix + key], dtype=np.float64).ravel()
    for k in (prefix + "standardize_scale", "standardize_scale"):
        if k in z.files:
            return w / np.asarray(z[k], dtype=np.float64)
    return w


def rs_parts(path: Path, prefix: str) -> tuple[np.ndarray, dict]:
    """(shared, {group: adjustment}) in raw activation units. Both live in
    the same standardized space, so one scaler maps them all; the map is
    linear, so map-then-add equals add-then-map."""
    READ.append(path)
    with np.load(path) as z:
        shared = cache_units(z, "coef", prefix)
        scale = np.asarray(z[prefix + "standardize_scale"], dtype=np.float64)
        devs = {str(g): np.asarray(d, np.float64) / scale
                for g, d in zip(z[prefix + "group_labels"],
                                z[prefix + "dev_w"])}
    return shared, devs


def merged(run: str, method: str, layer: int) -> Path:
    return RESULTS / run / "merged" / "predictions" / f"raw_last_layer{layer}_{method}.npz"


def single(dataset: str, method: str, layer: int) -> Path:
    return (RESULTS / COLLECTION / "per_dataset" / "predictions"
            / f"raw_last_layer{layer}_{method}_{dataset}.npz")


def leave_out(method: str, holdout: str, layer: int) -> Path:
    return (RESULTS / COLLECTION / "leave_out" / "predictions"
            / f"raw_last_layer{layer}_{method}_dataset_{holdout}.npz")


def leave_domain(method: str, task: str, layer: int) -> Path:
    return (RESULTS / COLLECTION / "leave_out" / "predictions"
            / f"raw_last_layer{layer}_{method}_task_{task}.npz")


def load(path: Path, prefix: str = "") -> np.ndarray:
    READ.append(path)
    with np.load(path) as z:
        return cache_units(z, "coef", prefix)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0:
        raise ValueError("cannot normalize a zero / non-finite vector")
    return v / n


# ---- per-target rows and covariances --------------------------------------

def target_rows(layer: int) -> dict:
    """target -> (X float32 [n, d], y int [n]) from the activation cache."""
    units = _load_units("last", layer, COLLECTION, groups=False)
    missing = [t for t in TARGETS if t not in units]
    if missing:
        raise SystemExit(f"units missing from the {COLLECTION} cache: {missing}")
    out = {}
    for t in TARGETS:
        X = np.asarray(units[t]["X"], dtype=np.float32)
        y = np.asarray(units[t]["y"], dtype=int)
        if set(np.unique(y)) != {0, 1}:
            raise SystemExit(f"{t}: labels are not binary 0/1")
        out[t] = (X, y)
        print(f"  rows {t}: {len(y)} ({int(y.sum())} label-1)", flush=True)
    return out


def lw_cov(X: np.ndarray, y: np.ndarray | None = None,
           centering: str = "total"):
    """Ledoit-Wolf-shrunk covariance of the dataset's rows.

    centering="total": both classes together,
    grand mean removed as part of the standard covariance definition, no
    class or cell centering. With balanced classes this matrix contains the
    class separation itself as variance (a quarter of d d^T for mean
    difference d), so its top eigenvector nearly coincides with the mean
    difference and multiplying by it pulls ANY vector toward the mean
    difference — the L29/L24 runs measured exactly that.

    centering="class": subtract each class's own
    mean first — the pooled within-class covariance. No separation term, so
    the power dial stops manufacturing the mean difference from arbitrary
    input, and Sigma * coefficients becomes the honest version of the
    discriminant identity."""
    from sklearn.covariance import LedoitWolf
    if centering == "class":
        Xc = X.astype(np.float64).copy()
        for cls in (0, 1):
            Xc[y == cls] -= Xc[y == cls].mean(0)
        lw = LedoitWolf(assume_centered=True).fit(Xc)
    else:
        lw = LedoitWolf(assume_centered=False).fit(X)
    return lw.covariance_.astype(np.float64), float(lw.shrinkage_)


# ---- build ----------------------------------------------------------------

def build(layer: int):
    vecs: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    geom: list[dict] = []
    meta: dict = {"layer": layer, "collection": COLLECTION, "targets": {},
                  "dial": DIAL,
                  "rungs": (list(RIDGE_TS) if DIAL == "ridge"
                            else list(POWERS)),
                  "cov_centering": COV_CENTERING}

    def gap_of(u: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        p = X.astype(np.float64) @ u
        return float(p[y == 1].mean() - p[y == 0].mean())

    def add(name, v, *, target, kind, uses_target_data, source, X=None,
            y=None, orient=False, covariance="", description=""):
        """Register one vector.

        kind              which sort of vector this is, for grouping
        uses_target_data  "yes" if this vector was built using rows from
                          the dataset it will be applied to, "no" if it
                          never saw them, "comparison" for the matched
                          comparison sets
        description       one plain sentence saying what this vector is;
                          required, because a short name is not an
                          explanation
        """
        if name in vecs:
            raise ValueError(f"duplicate vector {name}")
        if not description:
            raise ValueError(f"{name}: every vector needs a description")
        v = np.asarray(v, dtype=np.float64)
        if not np.isfinite(v).all():
            raise ValueError(f"{name}: non-finite entries")
        u = unit(v)
        gap, flipped = np.nan, False
        if X is not None:
            gap = gap_of(u, X, y)
            if orient and gap < 0:
                u, gap, flipped = -u, -gap, True
        vecs[name] = u
        rows.append({"name": name, "target": target, "kind": kind,
                     "uses_target_data": uses_target_data,
                     "description": description, "source": source,
                     "covariance": covariance, "gap": round(float(gap), 3),
                     "flipped": flipped})

    # ---- pooled fits: standard pool, plus the expanded pool for the two
    # datasets the standard pool never contained ------------------------
    std_l2 = load(merged(COLLECTION, "l2reg", layer), "full_")
    std_diff = load(merged(COLLECTION, "diff", layer), "full_")
    std_rs, std_rs_dev = rs_parts(merged(COLLECTION, "random_slopes", layer),
                                  "full_")
    std_rst, std_rst_dev = rs_parts(
        merged(COLLECTION, "random_slopes_task", layer), "full_")
    # The expanded pool exists to serve the pool-excluded targets. When none
    # of them is requested (e.g. a layer where those fits were never made),
    # the files are not required at all — the mid-layer runs drop
    # claude_vds and wildjailbreak_adversarial for exactly this reason.
    if any(t in POOL_EXCLUDED for t in TARGETS):
        sp_l2 = load(merged(EXPANDED_POOL, "l2reg", layer), "full_")
        sp_diff = load(merged(EXPANDED_POOL, "diff", layer), "full_")
        sp_rs, sp_rs_dev = rs_parts(
            merged(EXPANDED_POOL, "random_slopes", layer), "full_")
        sp_rst, sp_rst_dev = rs_parts(
            merged(EXPANDED_POOL, "random_slopes_task", layer), "full_")
        for t in POOL_EXCLUDED:
            if t in TARGETS and t not in sp_rs_dev:
                raise AssertionError(
                    f"the expanded-pool random-slopes fit has no {t} "
                    "adjustment — wrong run, refusing to build")
    else:
        sp_l2 = sp_diff = sp_rs = sp_rst = None
        sp_rs_dev, sp_rst_dev = {}, {}

    rows_by_t = target_rows(layer)
    dim = len(std_l2)

    # plain random controls, shared across targets: the null for the raw
    # coefficients, which are not covariance-shaped
    for seed in range(N_PLAIN_RANDOM if DIAL == "power" else 0):
        rng = np.random.default_rng(seed)
        add(f"rand_p0_{seed}", rng.standard_normal(dim), target="ALL",
            kind="rand_p0", uses_target_data="comparison",
            source=f"seed {seed}",
            description="A vector of random numbers. This is the "
                        "comparison for the coefficients used as fitted "
                        "(power 0). Raising the covariance to the power 0 "
                        "does nothing, so one set of random vectors "
                        "serves every dataset.")

    skipped = []

    for ti, t in enumerate(TARGETS):
        s = SHORT[t]
        X, y = rows_by_t[t]
        Sigma, shrink = lw_cov(X, y, COV_CENTERING)
        cov_tag = (f"{t}_rows_lw_within" if COV_CENTERING == "class"
                   else f"{t}_rows_lw")
        eigval, eigvec = np.linalg.eigh(Sigma)
        pc1 = eigvec[:, -1]
        lam = np.clip(eigval, 1e-12, None)      # Ledoit-Wolf keeps these > 0

        def conv(w, power: float):
            """Sigma^power times w. power 0 leaves the coefficients alone."""
            w = np.asarray(w, dtype=np.float64)
            if power == 0.0:
                return w
            return eigvec @ ((lam ** power) * (eigvec.T @ w))

        # ---- the dial: which family of maps is applied to the fits ------
        # "power"  Sigma^p w, p in POWERS (the L29/L24 runs)
        # "ridge"  ((1-t) Shat + t I)^(1/2) w, t in RIDGE_TS, where Shat is
        #          the covariance rescaled to mean eigenvalue 1 — replaces
        #          the multiplier rungs entirely
        lamhat = lam / lam.mean()

        def rmap(w, tt):
            w = np.asarray(w, dtype=np.float64)
            return eigvec @ (np.sqrt((1.0 - tt) * lamhat + tt)
                             * (eigvec.T @ w))

        if DIAL == "ridge":
            # (tag, map, n_null_draws, prose)
            RUNGS = [(f"r{int(round(v * 100))}", (lambda w, v=v: rmap(w, v)),
                      N_PLAIN_RANDOM,
                      ("used exactly as fitted (t = 1 is the identity end "
                       "of the path)" if v == 1.0 else
                       "mapped through ((1-t) x this dataset's within-class "
                       "covariance, rescaled to mean eigenvalue 1, + t x "
                       f"the identity) to the power one half, at t = {v}"))
                     for v in RIDGE_TS]
            # the rung the single-point vectors use (held-out fits, the
            # hierarchy pieces): chosen from the ridge geometry BEFORE any
            # GPU time as the t where fits stand furthest above their
            # matched randoms, the same read-the-geometry-first rule the
            # README already sets for the dial
            SINGLE_TAG = "r90"
            single_map = lambda w: rmap(w, 0.9)      # noqa: E731
            single_prose = ("mapped through the ridge path at t = 0.9, the "
                            "rung where the geometry separates fits from "
                            "matched random draws most sharply")
        else:
            RUNGS = [(POWER_TAG[p], (lambda w, p=p: conv(w, p)),
                      POWER_DRAWS[p],
                      ("used exactly as fitted (the covariance step is "
                       "switched off at power 0)" if p == 0.0 else
                       f"then multiplied by {t}'s own covariance raised to "
                       f"the power {p}"))
                     for p in POWERS]
            SINGLE_TAG = "p1"
            single_map = lambda w: conv(w, 1.0)      # noqa: E731
            single_prose = f"multiplied by {t}'s covariance once"

        excluded = t in POOL_EXCLUDED
        pool_name = "expanded_pool" if excluded else "standard"
        pl_l2 = sp_l2 if excluded else std_l2
        pl_diff = sp_diff if excluded else std_diff
        rs, rs_dev = (sp_rs, sp_rs_dev) if excluded else (std_rs, std_rs_dev)
        rst, rst_dev = (sp_rst, sp_rst_dev) if excluded else (std_rst,
                                                              std_rst_dev)
        if t not in rs_dev:
            raise AssertionError(f"the {pool_name} random-slopes fit has no "
                                 f"{t} adjustment")
        if TASK[t] not in rst_dev:
            raise AssertionError(f"the {pool_name} task random-slopes fit "
                                 f"has no {TASK[t]} adjustment")

        kw = dict(target=t, X=X, y=y, orient=True)
        USES, UNSEEN = "yes", "no"
        # how each rung of the ladder gets described, in plain words
        from_own = f"{t}'s own prompts"
        from_pool = ("all the combined training datasets"
                     + (" (the expanded-pool version, the only combined set "
                        f"that contains {t})" if excluded else ""))

        # ---- class-mean differences: movements already, never converted --
        add(f"{s}_own_diff", load(single(t, "diff", layer)),
            kind="own_diff", uses_target_data=USES,
            source="per_dataset/diff",
            description=f"The average difference: average the numbers over "
                        f"{t}'s True prompts, average them over its False "
                        f"prompts, subtract. Built from {from_own}.", **kw)
        add(f"{s}_pooled_diff", pl_diff, kind="pooled_diff",
            uses_target_data=USES, source=f"{pool_name}/diff",
            description="The same True-minus-False average difference, but "
                        f"computed over {from_pool} at once instead of over "
                        f"{t} alone.", **kw)

        # ---- the covariance power sweep, on three coefficient vectors ----
        own_l2 = load(single(t, "l2reg", layer))
        rs_combined = rs + rs_dev[t]
        for tag, dmap, _, dial in RUNGS:
            add(f"{s}_own_fit_{tag}", dmap(own_l2),
                kind=f"own_fit_{tag}", uses_target_data=USES,
                source="per_dataset/l2reg", covariance=cov_tag,
                description="Coefficients of a penalized logistic "
                            f"regression fitted on {from_own}, {dial}.",
                **kw)
            add(f"{s}_pooled_fit_{tag}", dmap(pl_l2),
                kind=f"pooled_fit_{tag}", uses_target_data=USES,
                source=f"{pool_name}/l2reg", covariance=cov_tag,
                description="Coefficients of a penalized logistic "
                            f"regression fitted on {from_pool}, {dial}.",
                **kw)
            add(f"{s}_rs_{tag}", dmap(rs_combined),
                kind=f"rs_{tag}", uses_target_data=USES,
                source=f"{pool_name}/random_slopes", covariance=cov_tag,
                description="The hierarchical (random-slopes) fit on "
                            f"{from_pool}: its one shared coefficient "
                            f"vector plus {t}'s own adjustment, added "
                            f"together first and {dial}.", **kw)

        # ---- taking the random-slopes fit apart, at power 1 -------------
        add(f"{s}_rs_shared_{SINGLE_TAG}", single_map(rs),
            kind=f"rs_shared_{SINGLE_TAG}",
            uses_target_data=USES, source=f"{pool_name}/random_slopes",
            covariance=cov_tag,
            description="The shared coefficient vector of the hierarchical "
                        f"fit ALONE, without {t}'s adjustment, multiplied "
                        "by the covariance once.", **kw)
        add(f"{s}_rs_adj_{SINGLE_TAG}", single_map(rs_dev[t]),
            kind=f"rs_adj_{SINGLE_TAG}",
            uses_target_data=USES, source=f"{pool_name}/random_slopes",
            covariance=cov_tag,
            description=f"{t}'s adjustment from the hierarchical fit "
                        "ALONE, without the shared vector, multiplied by "
                        "the covariance once. Asks whether the "
                        "dataset-specific part does anything by itself.",
            **kw)
        add(f"{s}_rs_task_{SINGLE_TAG}", single_map(rst + rst_dev[TASK[t]]),
            kind=f"rs_task_{SINGLE_TAG}", uses_target_data=USES,
            source=f"{pool_name}/random_slopes_task", covariance=cov_tag,
            description="The task-level hierarchical fit: its shared "
                        f"vector plus the adjustment for the {TASK[t]} "
                        "family of datasets rather than for this one "
                        "dataset, multiplied by the covariance once.",
            **kw)

        # ---- rung 3: this dataset taken out of the training set -------
        # Each vector is added in its own try, and the two average-based
        # ones go first: a missing hierarchical file must not also cost us
        # the plain difference, which is the vector this run leans on.
        sib = SIBLING.get(t)
        caveat = ("" if not sib else
                  f" NOTE: this training set still contains {sib}, whose "
                  f"prompts are the originals that {t}'s prompts were "
                  "rewritten from, with the same labels — so this is NOT a "
                  "clean never-seen-it number for this dataset. Use the "
                  "family-removed vectors for that.")
        if excluded:
            gone = (f"{t} is never part of any combined training set in "
                    "this project, so the standard combined set already "
                    "excludes it." + caveat)
            for nm, kind_, vec, src, extra in (
                    ("datasetout_diff", "datasetout_diff", std_diff,
                     "standard/diff",
                     "The True-minus-False average difference over the "
                     "standard combined training set. "),
                    (f"datasetout_fit_{SINGLE_TAG}", f"datasetout_fit_{SINGLE_TAG}",
                     single_map(std_l2), "standard/l2reg",
                     "Logistic-regression coefficients fitted on the "
                     "standard combined training set, multiplied by "
                     f"{t}'s covariance once. "),
                    (f"datasetout_rs_{SINGLE_TAG}", f"datasetout_rs_{SINGLE_TAG}",
                     single_map(std_rs), "standard/random_slopes",
                     "The shared coefficient vector of the hierarchical "
                     "fit on the standard combined training set "
                     "(there is no adjustment for a dataset that was "
                     f"never in it), multiplied by {t}'s covariance "
                     "once. ")):
                add(f"{s}_{nm}", vec, kind=kind_, uses_target_data=UNSEEN,
                    source=src,
                    covariance="" if "diff" in nm else cov_tag,
                    description=extra + gone, **kw)
        else:
            gone = (f"The training set had {t} removed before fitting."
                    + caveat)
            try:
                add(f"{s}_datasetout_diff",
                    load(leave_out("diff", t, layer)),
                    kind="datasetout_diff", uses_target_data=UNSEEN,
                    source="leave_out/diff",
                    description="The True-minus-False average difference "
                                "over the combined training set. " + gone,
                    **kw)
            except (FileNotFoundError, KeyError) as e:
                skipped.append(f"{t} dataset-removed average difference: {e}")
            try:
                add(f"{s}_datasetout_fit_{SINGLE_TAG}",
                    single_map(load(leave_out("l2reg", t, layer))),
                    kind=f"datasetout_fit_{SINGLE_TAG}", uses_target_data=UNSEEN,
                    source="leave_out/l2reg", covariance=cov_tag,
                    description="Logistic-regression coefficients fitted "
                                "on the combined training set, multiplied "
                                f"by {t}'s covariance once. " + gone, **kw)
            except (FileNotFoundError, KeyError) as e:
                skipped.append(f"{t} dataset-removed coefficients: {e}")
            try:
                lo_rs, lo_rs_dev = rs_parts(
                    leave_out("random_slopes", t, layer), "")
                if t in lo_rs_dev:
                    raise AssertionError(
                        f"this fit still carries a {t} adjustment — {t} "
                        "was not actually removed from it")
                add(f"{s}_datasetout_rs_{SINGLE_TAG}", single_map(lo_rs),
                    kind=f"datasetout_rs_{SINGLE_TAG}", uses_target_data=UNSEEN,
                    source="leave_out/random_slopes", covariance=cov_tag,
                    description="The shared coefficient vector of the "
                                "hierarchical fit on the combined "
                                "training set, multiplied by "
                                f"{t}'s covariance once. " + gone, **kw)
            except (FileNotFoundError, KeyError) as e:
                skipped.append(f"{t} dataset-removed hierarchical fit: {e}")

        # ---- rung 4: this dataset's whole family taken out ------------
        # claude_vds gets none: it belongs to no family in the training
        # set, so rung 3 above is already as clean as it gets for it.
        # wildjailbreak_adversarial DOES get one, and this is the only
        # clean rung for that dataset, because removing the whole
        # jailbreak family also removes its sibling.
        if t != "claude_vds":
            fam = (f"The training set had every {TASK[t]} dataset removed "
                   "before fitting"
                   + (f", including {sib}, so unlike the rung above this "
                      "one is clean." if sib else "."))
            try:
                add(f"{s}_taskout_diff",
                    load(leave_domain("diff", TASK[t], layer)),
                    kind="taskout_diff", uses_target_data=UNSEEN,
                    source=f"leave_out/diff task={TASK[t]}",
                    description="The True-minus-False average difference "
                                "over the combined training set. " + fam,
                    **kw)
            except (FileNotFoundError, KeyError) as e:
                skipped.append(f"{t} family-removed average difference: {e}")
            try:
                add(f"{s}_taskout_fit_{SINGLE_TAG}",
                    single_map(load(leave_domain("l2reg", TASK[t], layer))),
                    kind=f"taskout_fit_{SINGLE_TAG}", uses_target_data=UNSEEN,
                    source=f"leave_out/l2reg task={TASK[t]}",
                    covariance=cov_tag,
                    description="Logistic-regression coefficients fitted "
                                "on the combined training set, multiplied "
                                f"by {t}'s covariance once. " + fam, **kw)
            except (FileNotFoundError, KeyError) as e:
                skipped.append(f"{t} family-removed coefficients: {e}")
            try:
                do_rs, _ = rs_parts(
                    leave_domain("random_slopes", TASK[t], layer), "")
                add(f"{s}_taskout_rs_{SINGLE_TAG}", single_map(do_rs),
                    kind=f"taskout_rs_{SINGLE_TAG}", uses_target_data=UNSEEN,
                    source=f"leave_out/random_slopes task={TASK[t]}",
                    covariance=cov_tag,
                    description="The shared coefficient vector of the "
                                "hierarchical fit on the combined training "
                                f"set, multiplied by {t}'s covariance "
                                "once. " + fam, **kw)
            except (FileNotFoundError, KeyError) as e:
                skipped.append(f"{t} family-removed hierarchical fit: {e}")

        # ---- the matched comparison sets ------------------------------
        cmp_kw = dict(target=t, X=X, y=y, orient=False,
                      uses_target_data="comparison")
        u_diff = vecs[f"{s}_own_diff"]
        add(f"{s}_topvar_minus_diff", pc1 - float(pc1 @ u_diff) * u_diff,
            kind="topvar_minus_diff", covariance=cov_tag,
            source="largest-variance direction, own average difference "
                   "projected out",
            description="The single direction along which this dataset's "
                        "numbers vary most, with the real True-minus-False "
                        "average difference subtracted off. Tests whether "
                        "simply being a high-variance direction is enough "
                        "to move the answer.", **cmp_kw)
        # one comparison set per setting of the dial: the SAME
        # multiplication, applied to random numbers instead of to a fit
        # One comparison set per rung of the dial: the SAME map, applied
        # to random numbers instead of to a fit. Under the ridge dial the
        # eight underlying draws are shared across rungs, so each draw
        # traces a curve along the path rather than being fresh noise.
        ridge_draws = [np.random.default_rng([2000 + ti, k])
                       .standard_normal(dim) for k in range(N_PLAIN_RANDOM)]
        for tag, dmap, ndraw, dial in RUNGS:
            for k in range(ndraw):
                g = (ridge_draws[k] if DIAL == "ridge" else
                     np.random.default_rng([2000 + ti, tag, k])
                     .standard_normal(dim))
                add(f"{s}_rand_{tag}_{k}", dmap(g),
                    kind=f"rand_{tag}", covariance=cov_tag,
                    source=f"random draw {k}, {tag}",
                    description="A vector of random numbers put through "
                                "exactly the same step as the fitted "
                                f"vectors at rung {tag} — {dial} (draw "
                                f"{k} of {ndraw}). This is what a fitted "
                                "vector at that rung has to beat.",
                    **cmp_kw)
        for k in range(N_SHUFDIFF):
            yp = np.random.default_rng([3000 + ti, k]).permutation(y)
            d = (X[yp == 1].astype(np.float64).mean(0)
                 - X[yp == 0].astype(np.float64).mean(0))
            add(f"{s}_shufdiff_{k}", d, kind="shufdiff",
                source=f"label-shuffled average difference, draw {k}",
                description="The True-minus-False average difference "
                            "recipe run on the same rows after shuffling "
                            f"the labels (draw {k} of {N_SHUFDIFF}). Same "
                            "recipe, same data, no real grouping. This is "
                            "what the real average differences have to "
                            "beat.", **cmp_kw)

        # ---- how these vectors actually relate to each other ----------
        # Written before any GPU time is spent. For every vector that will
        # run on this dataset: how well it tells the two groups apart on
        # this dataset's own prompts, and how close it points to three
        # reference vectors. Two vectors that agree at cosine 0.99 will
        # behave the same no matter what the sweep costs to run, and this
        # is where that shows up.
        from sklearn.metrics import roc_auc_score
        ref = {"own_diff": u_diff,
               "unseen_diff": vecs.get(f"{s}_taskout_diff",
                                       vecs.get(f"{s}_datasetout_diff")),
               "topvar": pc1}
        here = [r for r in rows if r["target"] in (t, "ALL")]
        Xf = X.astype(np.float64)
        for r in here:
            u = vecs[r["name"]]
            a = float(roc_auc_score(y, Xf @ u))
            geom.append({
                "target": t, "name": r["name"], "kind": r["kind"],
                "uses_target_data": r["uses_target_data"],
                # sign is arbitrary for a comparison vector, so report the
                # sign-free version: 0.5 means it cannot tell the groups
                # apart at all, 1.0 means perfectly
                "tells_groups_apart": round(max(a, 1.0 - a), 4),
                "auc_signed": round(a, 4),
                **{f"cos_{k}": (np.nan if v is None
                                else round(float(abs(u @ v)), 4))
                   for k, v in ref.items()},
            })

        meta["targets"][t] = {
            "n_rows": int(len(y)), "n_label1": int(y.sum()),
            "lw_shrinkage": round(shrink, 6),
            "top_eig_share": round(float(eigval[-1] / eigval.sum()), 4),
            "cos_pc1_own_diff": round(float(abs(pc1 @ u_diff)), 4),
            "pool": pool_name,
        }
        print(f"  {t}: pool={pool_name}, LW shrinkage {shrink:.4f}, "
              f"top-eig share {eigval[-1] / eigval.sum():.3f}, "
              f"|cos(pc1, own diff)| {abs(pc1 @ u_diff):.3f}", flush=True)

    return vecs, pd.DataFrame(rows), pd.DataFrame(geom), meta, skipped

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layer", type=int, default=29)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help="fit tree and activation cache to build from "
                         "(gemma_tf_a, qwen_tf_a)")
    ap.add_argument("--targets", default="",
                    help="comma-separated subset of TARGETS (default: all)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files, e.g. qwen")
    ap.add_argument("--cov-centering", choices=("total", "class"),
                    default="total",
                    help="covariance for the power dial: 'total' = both "
                         "classes together (the L29/L24 runs), 'class' = "
                         "pooled within-class, which removes the class "
                         "separation from the matrix")
    ap.add_argument("--dial", choices=("power", "ridge"), default="power",
                    help="which map family the fits go through: 'power' = "
                         "Sigma^p (the L29/L24 runs), 'ridge' = "
                         "((1-t) Shat + t I)^(1/2), which REPLACES the "
                         "multiplier rungs")
    args = ap.parse_args()

    global COLLECTION, EXPANDED_POOL, TARGETS, OUT_TAG, COV_CENTERING, DIAL
    COLLECTION = args.collection
    EXPANDED_POOL = f"{COLLECTION}_expanded_pool"
    OUT_TAG = f"_{args.tag}" if args.tag else ""
    COV_CENTERING = args.cov_centering
    DIAL = args.dial
    if DIAL == "ridge" and COV_CENTERING != "class":
        raise SystemExit("--dial ridge needs --cov-centering class: "
                         "the ridge path is defined on the "
                         "within-class covariance")
    if args.targets:
        want = tuple(x.strip() for x in args.targets.split(",") if x.strip())
        unknown = [x for x in want if x not in TARGETS]
        if unknown:
            raise SystemExit(f"unknown targets: {unknown}")
        TARGETS = want
    print(f"building from {COLLECTION} (expanded pool {EXPANDED_POOL}) at layer "
          f"{args.layer}: {len(TARGETS)} targets"
          + f", {DIAL} dial on the {COV_CENTERING}-centered "
            "covariance", flush=True)

    vecs, manifest, geometry, meta, skipped = build(args.layer)
    before = {p: p.stat().st_mtime for p in set(READ)}

    OUT.mkdir(parents=True, exist_ok=True)
    names = list(manifest.name)
    W = np.vstack([vecs[n] for n in names]).astype(np.float32)
    norms = np.linalg.norm(W.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        raise AssertionError(f"not unit-norm: {norms.min()} .. {norms.max()}")

    np.savez_compressed(OUT / f"directions{OUT_TAG}_layer{args.layer}.npz",
                        names=np.array(names), W=W,
                        collection=np.array(COLLECTION),
                        layer=np.array(args.layer))
    # The readers agreed to these column names; say so here, where a
    # rename is being made, rather than six hours later on a GPU.
    MS.validate(manifest, "directions.py")
    manifest.to_csv(OUT / f"manifest{OUT_TAG}_layer{args.layer}.csv", index=False)
    geometry.to_csv(OUT / f"geometry{OUT_TAG}_layer{args.layer}.csv",
                    index=False)
    (OUT / f"directions_meta{OUT_TAG}_layer{args.layer}.json").write_text(
        json.dumps(meta, indent=2))

    # no fit file may be touched — this experiment refits nothing
    after = {p: p.stat().st_mtime for p in set(READ)}
    if before != after:
        raise AssertionError("a fit file changed during the build")
    if OUT.resolve().is_relative_to(RESULTS.resolve()):
        raise AssertionError("outputs must not be written into the fit tree")

    n_real = int((manifest.uses_target_data != "comparison").sum())
    n_flip = int(manifest.flipped.sum())
    print(f"\nwrote directions{OUT_TAG}_layer{args.layer}.npz: {len(names)} "
          f"length-1 "
          f"vectors ({n_real} real, {len(names) - n_real} comparison), "
          f"d={W.shape[1]}, {n_flip} flipped to point toward True, "
          f"read {len(set(READ))} fit files (all unchanged)")
    for msg in skipped:
        print(f"  SKIPPED {msg}")
    pd.set_option("display.width", 220)
    real = manifest[manifest.uses_target_data != "comparison"]
    print(real[["name", "target", "kind", "uses_target_data", "gap",
                "flipped", "source"]].to_string(index=False))

    print("\n== how well each vector tells the two groups apart, and how "
          "close it points to the reference vectors ==")
    print("   (0.5 = cannot tell them apart at all; 1.0 = perfectly. "
          "Read each real vector against the\n    comparison vectors "
          "built the same way — a fitted vector at some power has to beat "
          "random\n    numbers put through the same step.)")
    for t, g in geometry.groupby("target", sort=False):
        print(f"\n-- {t} --")
        real_g = g[g.uses_target_data != "comparison"]
        print(real_g.drop(columns=["target", "uses_target_data"])
              .to_string(index=False))
        cmp_g = g[g.uses_target_data == "comparison"]
        if len(cmp_g):
            print("   comparison sets, min / median / max:")
            for k, gg in cmp_g.groupby("kind"):
                print(f"     {k:20s} n={len(gg):2d}  tells apart "
                      f"{gg.tells_groups_apart.min():.3f} / "
                      f"{gg.tells_groups_apart.median():.3f} / "
                      f"{gg.tells_groups_apart.max():.3f}   "
                      f"cos to own difference "
                      f"{gg.cos_own_diff.min():.3f} / "
                      f"{gg.cos_own_diff.median():.3f} / "
                      f"{gg.cos_own_diff.max():.3f}")


if __name__ == "__main__":
    main()
