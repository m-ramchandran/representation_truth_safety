"""Check a collection has everything the steering build needs, before any
GPU time is spent.

Nothing here fits anything — it opens each saved fit file, confirms the
arrays the build reads are present, and reports what is missing. Run it
for a new model/layer first; the build itself tolerates missing held-out
fits (it skips them), but a missing pooled fit or a missing activation
cache is fatal and should be found in two minutes, not two hours.

    python steering/preflight.py --collection qwen_tf_a --layer 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

import steering.directions as D  # noqa: E402
import steering.manifest_schema as MS  # noqa: E402
from classifiers.data import COLLECTIONS, _load_units  # noqa: E402


def probe(path: Path, need: tuple, prefix: str = "") -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with np.load(path) as z:
            have = set(z.files)
            gone = [prefix + k for k in need if prefix + k not in have]
            if gone:
                return False, f"present but has no {gone}"
            n = len(np.asarray(z[prefix + need[0]]).ravel())
        return True, f"ok (d={n})"
    except Exception as exc:                        # noqa: BLE001
        return False, f"unreadable: {type(exc).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collection", default="qwen_tf_a")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--targets", default="",
                    help="comma-separated subset of the default targets; "
                         "checks are limited to what these actually need")
    args = ap.parse_args()

    c = COLLECTIONS[args.collection]
    layer = args.layer if args.layer is not None else c.layers[2]
    D.COLLECTION = args.collection
    D.EXPANDED_POOL = f"{args.collection}_expanded_pool"
    if args.targets:
        want = tuple(x.strip() for x in args.targets.split(",") if x.strip())
        unknown = [x for x in want if x not in D.TARGETS]
        if unknown:
            raise SystemExit(f"unknown targets: {unknown}")
        D.TARGETS = want
    # The expanded pool only serves the pool-excluded targets; when none is
    # requested, its files are not required and are not checked.
    need_sp = any(t in D.POOL_EXCLUDED for t in D.TARGETS)
    print(f"preflight: {args.collection} at layer {layer} "
          f"(model {c.model}), {len(D.TARGETS)} targets"
          + ("" if need_sp else " — expanded pool not needed") + "\n")

    fatal, skippable = [], []

    print("== pooled fits ==")
    for run in ((D.COLLECTION, D.EXPANDED_POOL) if need_sp
                else (D.COLLECTION,)):
        for method in ("l2reg", "diff", "random_slopes", "random_slopes_task"):
            need = ("coef",) if method == "diff" else ("coef",
                                                       "standardize_scale")
            if method.startswith("random_slopes"):
                need = need + ("group_labels", "dev_w")
            ok, msg = probe(D.merged(run, method, layer), need, "full_")
            print(f"  {run:28s} {method:20s} {msg}")
            if not ok:
                fatal.append(f"{run}/{method}")

    print("\n== the expanded-pool hierarchical fit must carry an "
          "adjustment for every dataset the standard pool never had ==")
    if not need_sp:
        print("  no pool-excluded target requested — skipped")
    else:
        try:
            _, dev = D.rs_parts(
                D.merged(D.EXPANDED_POOL, "random_slopes", layer), "full_")
            for t in D.POOL_EXCLUDED:
                here = t in dev
                print(f"  {t:28s} {'ok' if here else 'MISSING'}")
                if not here:
                    fatal.append(f"expanded pool has no {t} adjustment")
        except Exception as exc:                    # noqa: BLE001
            print(f"  could not read: {type(exc).__name__}")
            fatal.append("expanded-pool random slopes unreadable")

    print("\n== per-dataset fits ==")
    for t in D.TARGETS:
        for method in ("diff", "l2reg"):
            need = ("coef",) if method == "diff" else ("coef",
                                                       "standardize_scale")
            ok, msg = probe(D.single(t, method, layer), need)
            print(f"  {t:28s} {method:8s} {msg}")
            if not ok:
                fatal.append(f"per_dataset/{method}/{t}")

    # What each method needs is what the build actually reads: the plain
    # coefficient for diff and l2reg, and the whole hierarchical structure
    # for random_slopes, which the build takes apart.
    NEED = {"diff": ("coef",), "l2reg": ("coef", "standardize_scale"),
            "random_slopes": ("coef", "standardize_scale", "group_labels",
                              "dev_w")}

    print("\n== fits with this dataset taken out (the build skips any "
          "that are absent) ==")
    for t in D.TARGETS:
        if t in D.POOL_EXCLUDED:
            print(f"  {t:28s} never in any combined training set, so the "
                  "standard one already excludes it")
            continue
        missing = [m for m in NEED
                   if not probe(D.leave_out(m, t, layer), NEED[m])[0]]
        skippable += [f"{t}, {m} fit with {t} removed" for m in missing]
        print(f"  {t:28s} "
              + ("all three present" if not missing
                 else f"MISSING {', '.join(missing)}"))

    print("\n== fits with this dataset's whole family taken out ==")
    for task in sorted({D.TASK[t] for t in D.TARGETS if t != "claude_vds"}):
        missing = [m for m in NEED
                   if not probe(D.leave_domain(m, task, layer), NEED[m])[0]]
        skippable += [f"{task} family, {m} fit with the family removed"
                      for m in missing]
        print(f"  family={task:21s} "
              + ("all three present" if not missing
                 else f"MISSING {', '.join(missing)}"))

    print("\n== the task-level hierarchical fit must carry an adjustment "
          "for every family we actually take from that pool ==")
    # Each pool only has to cover the datasets that draw from it: the
    # standard one serves every ordinary dataset, the expanded pool serves
    # only the ones the standard pool never contained. Demanding more
    # would report a fatal that is not one.
    needs = {
        D.COLLECTION: sorted({D.TASK[t] for t in D.TARGETS
                              if t not in D.POOL_EXCLUDED}),
        D.EXPANDED_POOL: sorted({D.TASK[t] for t in D.TARGETS
                              if t in D.POOL_EXCLUDED}),
    }
    for run, tasks in needs.items():
        if not tasks:
            print(f"  {run:28s} nothing drawn from this pool")
            continue
        try:
            _, dev = D.rs_parts(D.merged(run, "random_slopes_task", layer),
                                "full_")
            for task in tasks:
                here = task in dev
                print(f"  {run:28s} {task:12s} "
                      f"{'ok' if here else 'MISSING'}")
                if not here:
                    fatal.append(f"{run} task fit has no {task} adjustment")
        except Exception as exc:                    # noqa: BLE001
            print(f"  {run}: could not read ({type(exc).__name__})")
            fatal.append(f"{run} task-level hierarchical fit unreadable")

    print("\n== activation cache ==")
    try:
        units = _load_units("last", layer, args.collection, groups=False)
        for t in D.TARGETS:
            if t not in units:
                print(f"  {t:28s} MISSING from the cache")
                fatal.append(f"cache/{t}")
            else:
                y = np.asarray(units[t]["y"], dtype=int)
                X = units[t]["X"]
                print(f"  {t:28s} {len(y):6d} rows, {int(y.sum()):5d} "
                      f"label-1, d={X.shape[1]}")
    except Exception as exc:                        # noqa: BLE001
        print(f"  could not load: {type(exc).__name__}: {exc}")
        fatal.append("activation cache")

    print("\n== manifest schema, for any manifest already on disk ==")
    # On a fresh layer the build has not run, so there is nothing to check
    # and that is fine — directions.py validates its own output as it
    # writes. This is here for the rerun case, where a manifest from an
    # older build is sitting in out/ and steer.py would read it.
    found = sorted(D.OUT.glob(f"manifest*_layer{layer}.csv"))
    if not found:
        print(f"  none at layer {layer} yet — the build writes one and "
              "checks it against the same contract")
    for path in found:
        try:
            MS.validate(pd.read_csv(path), f"preflight/{path.name}")
            print(f"  {path.name:34s} ok")
        except SystemExit as exc:                   # noqa: PERF203
            print(f"  {path.name:34s} STALE — {exc}")
            fatal.append(f"{path.name} does not match manifest_schema.py")

    print("\n" + "=" * 60)
    if skippable:
        print(f"WOULD BE SKIPPED ({len(skippable)}):")
        for m in skippable:
            print(f"  {m}")
    if fatal:
        print(f"\nFATAL ({len(fatal)}) — do not launch:")
        for m in fatal:
            print(f"  {m}")
        sys.exit(1)
    print("\nall required fits present — safe to build directions")


if __name__ == "__main__":
    main()
