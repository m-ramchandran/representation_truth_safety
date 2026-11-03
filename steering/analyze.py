"""Turn a finished sweep into a few numbers per vector (laptop, no GPU).

For each vector applied to a dataset:

  effect_per_1000     how much the model's preference for the positive
                      word shifts per 1000 units of pushing, fitted
                      through the origin over the push sizes that count:
                      size 1 and below, where the rest of the model's
                      output survived (see the KL cutoff below), and
                      where the push was big enough to survive 16-bit
                      rounding. Blank when no size qualifies.
  effect_per_1pct     the same effect, restated as a push of 1% of the
                      typical length of the model's numbers. Use this one
                      to compare across models: the raw per-1000 figure
                      is in units of whatever that model's numbers happen
                      to be, and the two models differ.
  moved_right_way     share of individual prompts that moved the way the
                      push was pointed, at sizes +1 and -1. Averages hide
                      half the prompts going the other way, so this is
                      printed next to every average.
  largest_safe_shift  biggest average shift over the push sizes where the
                      rest of the output still survived.
  pins_true/false     the check that a vector's sign is right: at the
                      largest push the answer should be pinned at the
                      positive word, and at the largest push the other
                      way, at the negative word. sign_ok when both hold.
  removal_cost_auc    how much worse the model's answers separate the two
                      groups after subtracting off this vector's
                      component, next to removal_share, how much of the
                      total length that subtraction removed. Different
                      vectors remove very different amounts, so a fair
                      comparison has to report both.
  landed_on_*         mean matching: each prompt moved so it lands on one
                      group's average position along this vector, instead
                      of being pushed a fixed amount. Per side: the mean
                      change in the answer, the share of prompts that
                      moved the intended way, the mean probability of
                      that side afterwards, whether the rest of the
                      output survived, and how far prompts had to travel
                      — that last one matters, because a vector that only
                      works by moving prompts a long way is not doing the
                      same job as one that works from where they are.

The matched comparison sets are summarized as a band (smallest / middle /
largest absolute effect) per set, so a real vector can be read against
random numbers put through exactly the same construction.

A sweep saves every word pair the extraction pipeline scores, so --pair
re-reads the same run under a different answer vocabulary (true, yes,
safe, harmless). Sweeps from before that change carry True/False only.

    python steering/analyze.py --run-dir steering/out/runs_qwen_layer24 \
                               --manifest steering/out/manifest_qwen_layer24.csv
                               [--pair true|yes|safe|harmless]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "steering" / "out"

# How far the model's whole next-word distribution is allowed to move
# from the unedited version before we stop counting an edit as evidence.
# An edit that flips the answer while wrecking everything else the model
# was about to say tells us nothing. Measured in nats.
KL_SLOPE = 0.5      # push sizes allowed into the effect estimate
KL_MAX = 1.0        # push sizes allowed into largest_safe_shift
# A push smaller than this fraction of the length of the model's numbers
# is lost to 16-bit rounding, so it cannot be evidence either.
ROUNDING_FLOOR = 1e-3

# answer vocabulary -> its columns in the run csv (positive side, negative
# side, share of next-token probability the pair receives)
PAIRS = {
    "true": ("true_lse", "false_lse", "tf_mass"),
    "yes": ("yes_lse", "no_lse", "yes_no_mass"),
    "safe": ("safe_lse", "unsafe_lse", "safe_unsafe_mass"),
    "harmless": ("harmless_lse", "harmful_lse", "harmful_harmless_mass"),
}


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def summarize_target(df: pd.DataFrame, man: pd.DataFrame, dose_units: dict,
                     target: str, pair: str = "true"):
    pos, neg, mass = PAIRS[pair]
    missing = [c for c in (pos, neg, mass) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{target}: this run has no {missing} column — it predates "
            "multi-pair saving and can only be read as --pair true")
    base = df[df.direction == "baseline"].set_index("sample_id")
    base_lo = base[pos] - base[neg]
    base_auc = auc(base.y, base_lo)
    base_mass = float(base[mass].median())
    med_hnorm = float(base.extra.median())
    floor = ROUNDING_FLOOR * med_hnorm

    kind_of = dict(zip(man.name, man.kind))
    uses_of = dict(zip(man.name, man.uses_target_data))
    flip_of = dict(zip(man.name, man.flipped)) if "flipped" in man else {}
    unknown = sorted(set(df.direction.unique()) - set(kind_of) - {"baseline"})
    if unknown:
        raise SystemExit(
            f"{target}: {len(unknown)} vectors in this run are not in the "
            f"manifest (e.g. {unknown[:3]}) — wrong --manifest for this "
            "run. Every vector would be reported under the wrong heading.")

    rows, curves = [], []
    for d_ in df.direction.unique():
        if d_ == "baseline":
            continue
        g = df[(df.direction == d_) & (df.intervention == "add")]
        gap = dose_units.get("gaps", {}).get(d_, np.nan)
        per_frac = {}
        for fr, gf in g.groupby("frac"):
            gf = gf.set_index("sample_id")
            shift = (gf[pos] - gf[neg]) - base_lo.reindex(gf.index)
            p_pos = 1.0 / (1.0 + np.exp(-(gf[pos] - gf[neg])))
            per_frac[fr] = {
                "alpha": float(gf.extra.iloc[0]),
                "mean_shift": float(shift.mean()),
                "right": float((np.sign(shift) == np.sign(fr)).mean()),
                "med_kl": float(gf.kl.median()),
                "med_mass": float(gf[mass].median()),
                "mean_p_pos": float(p_pos.mean()),
            }
            curves.append({"target": target, "direction": d_, "frac": fr,
                           **per_frac[fr]})

        usable = [fr for fr, r in per_frac.items()
                  if abs(fr) <= 1 and r["med_kl"] <= KL_SLOPE
                  and abs(r["alpha"]) >= floor]
        if usable:
            a = np.array([per_frac[fr]["alpha"] for fr in usable])
            m = np.array([per_frac[fr]["mean_shift"] for fr in usable])
            slope = float((a @ m) / (a @ a)) * 1000.0
        else:
            slope = float("nan")
        right1 = np.mean([per_frac[fr]["right"]
                          for fr in (1.0, -1.0) if fr in per_frac])
        coh = [abs(r["mean_shift"]) for fr, r in per_frac.items()
               if r["med_kl"] <= KL_MAX]
        top = max(per_frac) if per_frac else None
        sat_pos = per_frac.get(top, {}).get("mean_p_pos", np.nan) \
            if top else np.nan
        sat_neg = per_frac.get(-top, {}).get("mean_p_pos", np.nan) \
            if top else np.nan
        sat_ok = bool(sat_pos >= 0.99 and sat_neg <= 0.01) \
            if np.isfinite(sat_pos) and np.isfinite(sat_neg) else False

        # mean matching: each prompt moved onto one group's average
        # position along this vector, rather than pushed a fixed amount
        mm_stats = {}
        for src, dst, want in (("match_pos", "landed_on_true", +1),
                               ("match_neg", "landed_on_false", -1)):
            mm = df[(df.direction == d_) & (df.intervention == src)]
            if not len(mm):
                continue
            mm = mm.set_index("sample_id")
            lo = mm[pos] - mm[neg]
            sh = lo - base_lo.reindex(mm.index)
            mm_stats[f"{dst}_shift"] = float(sh.mean())
            mm_stats[f"{dst}_right_way"] = float((np.sign(sh) == want).mean())
            mm_stats[f"{dst}_p"] = float((1.0 / (1.0 + np.exp(-lo))).mean())
            mm_stats[f"{dst}_kl"] = float(mm.kl.median())
            mm_stats[f"{dst}_distance"] = float(mm.extra.abs().mean())

        ab = df[(df.direction == d_) & (df.intervention == "ablate")]
        ab_dauc = ab_share = float("nan")
        if len(ab):
            ab = ab.set_index("sample_id")
            ab_dauc = base_auc - auc(ab.y, ab[pos] - ab[neg])
            ab_share = float(ab.extra.mean()) / float(base.extra.mean())

        rows.append({
            "target": target, "direction": d_,
            "kind": kind_of[d_], "uses_target_data": uses_of[d_],
            "flipped": bool(flip_of.get(d_, False)),
            "gap": gap,
            "effect_per_1000": slope,
            # restated per push of 1% of the typical length of the model's
            # numbers, so the two models can be compared
            "effect_per_1pct": slope * med_hnorm / 1e5,
            "n_usable": len(usable),
            "moved_right_way": float(right1) if per_frac else np.nan,
            "largest_safe_shift": max(coh) if coh else np.nan,
            "pins_true": sat_pos, "pins_false": sat_neg, "sign_ok": sat_ok,
            "removal_cost_auc": ab_dauc, "removal_share": ab_share,
            **mm_stats,
            # the unedited model on this sample, under this word pair:
            # how well its answer tracks the labels, and the dose base
            "baseline_auc": base_auc, "baseline_pair_mass": base_mass,
            "dose_base": dose_units.get("base", np.nan),
        })
    return (pd.DataFrame(rows), pd.DataFrame(curves),
            {"auc": base_auc, "mass": base_mass, "hnorm": med_hnorm})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", default=str(OUT / "runs_layer29"))
    ap.add_argument("--manifest", default=str(OUT / "manifest_layer29.csv"))
    ap.add_argument("--pair", default="true", choices=sorted(PAIRS),
                    help="answer vocabulary to read the sweep under")
    args = ap.parse_args()
    pos, neg, _ = PAIRS[args.pair]
    run_dir = Path(args.run_dir)
    man = pd.read_csv(args.manifest)
    # older builds used different names for the same two columns
    man = man.rename(columns={"family": "kind",
                              "regime": "uses_target_data"})
    if "uses_target_data" in man:
        man["uses_target_data"] = man.uses_target_data.replace(
            {"in_fit": "yes", "out_of_fit": "no", "control": "comparison"})

    # One file per dataset when the sweep ran as an array job, one file
    # for everything when it ran as a single job. Merge either way.
    meta_files = sorted(run_dir.glob("run_meta*.json"))
    if not meta_files:
        raise SystemExit(f"no run_meta*.json in {run_dir}")
    # Merged all the way down: when one target was spread over several
    # parts, each part's file carries only its own vectors' gaps under
    # dose_units[target], and a one-level update would keep the last
    # part's only.
    def merge_into(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge_into(dst[k], v)
            else:
                dst.setdefault(k, v)

    meta: dict = {}
    for mf in meta_files:
        merge_into(meta, json.loads(mf.read_text()))

    all_sum, all_curves = [], []
    stem = "" if args.pair == "true" else f"_{args.pair}"
    skip = {"summary.csv", "curves.csv",
            f"summary{stem}.csv", f"curves{stem}.csv"}
    for f in sorted(run_dir.glob("*.csv")):
        if f.name in skip or f.name.startswith(("summary", "curves")):
            continue
        df = pd.read_csv(f)
        target = df.target.iloc[0]
        du = meta.get("dose_units", {}).get(target, {})
        s, c, b = summarize_target(df, man, du, target, args.pair)
        all_sum.append(s)
        all_curves.append(c)

        print(f"\n===== {target} ({pos}-vs-{neg} baseline AUC "
              f"{b['auc']:.3f}, median mass on the pair "
              f"{b['mass']:.2e}) =====")
        cols = ["kind", "direction", "effect_per_1000", "effect_per_1pct",
                "moved_right_way", "largest_safe_shift", "sign_ok",
                "removal_cost_auc", "removal_share", "n_usable"]
        for uses, heading in (
                ("yes", "built using this dataset's own prompts"),
                ("no", "built without ever seeing this dataset")):
            block = s[s.uses_target_data == uses].sort_values(
                "effect_per_1000", key=lambda x: x.abs(), ascending=False)
            if block.empty:
                continue
            print(f"-- vectors {heading} --")
            print(block[cols].round(3).to_string(index=False))
            if "landed_on_true_shift" in block:
                print("   mean matching (each prompt moved onto a group's "
                      "average position along this vector):")
                mcols = ["direction", "landed_on_true_shift",
                         "landed_on_true_right_way", "landed_on_true_p",
                         "landed_on_false_shift",
                         "landed_on_false_right_way", "landed_on_false_p",
                         "landed_on_true_kl", "landed_on_true_distance"]
                print(block[[c for c in mcols if c in block]]
                      .round(3).to_string(index=False))
            for _, r in block.iterrows():
                if not r.sign_ok:
                    print(f"   SIGN CHECK FAILED {r.direction}: pushing "
                          f"hard the positive way left P({pos.split('_')[0]})"
                          f" at {r.pins_true:.3f} (should pin near 1) and "
                          f"the negative way at {r.pins_false:.3f} (should "
                          "pin near 0)")
                if r.flipped:
                    print(f"   note {r.direction}: this vector pointed away "
                          "from True on this dataset and was reversed at "
                          "build time")
        cmp_ = s[s.uses_target_data == "comparison"]
        if len(cmp_):
            print("-- matched comparison sets: |effect_per_1000| "
                  "smallest / middle / largest --")
            for k, gg in cmp_.groupby("kind"):
                a = gg.effect_per_1000.abs()
                print(f"   {k:20s} n={len(gg):2d}  "
                      f"{a.min():6.2f} / {a.median():6.2f} / {a.max():6.2f}")

    summary = pd.concat(all_sum, ignore_index=True)
    curves = pd.concat(all_curves, ignore_index=True)
    summary.insert(0, "pair", args.pair)
    curves.insert(0, "pair", args.pair)
    summary.to_csv(run_dir / f"summary{stem}.csv", index=False)
    curves.to_csv(run_dir / f"curves{stem}.csv", index=False)
    print(f"\nwrote {run_dir}/summary{stem}.csv and curves{stem}.csv")


if __name__ == "__main__":
    main()
