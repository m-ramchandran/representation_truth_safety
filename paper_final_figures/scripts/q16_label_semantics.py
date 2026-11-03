"""Q16: does transfer follow label semantics, and can one pooled direction
carry several label types at once? (Original prompts only.)

On the bare collections y=1 means three different things (label polarity
verified in classifiers/Planning.md, 2026-07-25):
  truth  - the shown statement is true (got, common_claim, counterfact,
           claude_anatomy, claude_blood_test_ranges, claude_vds, social_iqa)
  yesno  - the correct answer to the shown question is "yes"
           (pubmedqa_artificial, pubmedqa_labeled)
  safety - the shown request is safe/benign (harmful_harmless,
           wildjailbreak_vanilla, wildjailbreak_adversarial, xstest)

Three questions, all at the F13 depth band (65-85%; the 50% and 97%
depths are unstable for exactly these cells - recorded in the digest):
  1. Is single-source transfer organized by label type (block structure),
     and is cross-label-type transfer sign-aligned (true = yes = safe on
     the same side of the axis, AUC > 0.5) or anti-aligned (< 0.5)?
  2. Is the label-type block structure the same thing as the domain
     structure (F4)? (No: the medical domain splits - anatomy travels
     with the truth block, the two pubmedqas with each other.)
  3. Leave-domain-out removes an entire label type from the pooled
     corpus twice over: holdout=jailbreak leaves no safe/unsafe data
     (the pool's only safety sets are the three jailbreak members) and
     holdout=medical leaves no yes/no data (both pubmedqas are medical).
     How much of the target's separability does a pooled probe keep when
     it has never seen the target's label type? (Truth has no such test:
     truth-labelled units sit in four of the five domains.)

Input: data/shared_classifier_results.csv. Writes
supplementary_figure5_label_transfer.csv,
q16_group_matrix.csv, q16_labeltype_absent.csv, q16_digest.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C

LABEL_GROUP = {
    "got": "truth", "common_claim": "truth", "counterfact": "truth",
    "claude_anatomy": "truth", "claude_blood_test_ranges": "truth",
    "claude_vds": "truth", "social_iqa": "truth",
    "pubmedqa_artificial": "yesno", "pubmedqa_labeled": "yesno",
    "harmful_harmless": "safety", "wildjailbreak_vanilla": "safety",
    "wildjailbreak_adversarial": "safety", "xstest": "safety",
}
GROUP_ORDER = ("truth", "yesno", "safety")
GROUP_LABEL = {"truth": "True/False", "yesno": "Yes/No",
               "safety": "Safe/Unsafe"}
GROUP_MEANING = {"truth": "y=1: the statement is true",
                 "yesno": "y=1: the correct answer is yes",
                 "safety": "y=1: the request is safe"}
# pool sources and all 13 targets, grouped by label type (pool-excluded
# targets close their block)
SOURCE_ORDER = ("got", "common_claim", "counterfact", "claude_anatomy",
                "social_iqa", "pubmedqa_artificial", "pubmedqa_labeled",
                "harmful_harmless", "wildjailbreak_vanilla", "xstest")
TARGET_ORDER = ("got", "common_claim", "counterfact", "claude_anatomy",
                "social_iqa", "claude_blood_test_ranges", "claude_vds",
                "pubmedqa_artificial", "pubmedqa_labeled",
                "harmful_harmless", "wildjailbreak_vanilla", "xstest",
                "wildjailbreak_adversarial")
MID_DEPTHS = (2, 3)          # the 65% / 85% band (F13)
METHODS = ("diff", "l2reg")  # transfer estimator + the discriminative foil
# targets whose leave_domain_out fit saw no example of their label type
ABSENT_TYPE_TARGETS = ("harmful_harmless", "wildjailbreak_vanilla",
                       "xstest", "wildjailbreak_adversarial",
                       "pubmedqa_artificial", "pubmedqa_labeled")


def main() -> None:
    lr = pd.read_csv(C.DATA / "shared_classifier_results.csv")
    bare = lr[(lr.perturbation == "bare") & lr.method.isin(METHODS)].copy()

    # ---- 1. pair-level single-source transfer ---------------------------
    ss = bare[bare.regime == "single_source"].copy()
    ss["source_group"] = ss.source_dataset.map(LABEL_GROUP)
    ss["target_group"] = ss.dataset.map(LABEL_GROUP)
    pair = ss[["model", "method", "source_dataset", "dataset", "depth",
               "source_group", "target_group", "auc"]].rename(
        columns={"dataset": "target_dataset"})
    pair.to_csv(C.DATA / "supplementary_figure5_label_transfer.csv",
                index=False)

    # ---- 2. group-level matrix, mid band and all depths -----------------
    rows = []
    for agg, sel in (("mid", ss[ss.depth.isin(MID_DEPTHS)]), ("all", ss)):
        g = sel.groupby(["model", "method", "source_group", "target_group"])
        m = g.auc.agg(auc_median="median", n_cells="size",
                      frac_below_048=lambda a: (a < 0.48).mean(),
                      frac_above_05=lambda a: (a > 0.5).mean()).reset_index()
        m.insert(2, "depth_agg", agg)
        rows.append(m)
    grid = pd.concat(rows, ignore_index=True)
    grid.to_csv(C.DATA / "q16_group_matrix.csv", index=False)

    # ---- 3. label-type-absent ladder ------------------------------------
    regs = ("within_cv", "merged_included", "leave_dataset_out",
            "leave_domain_out")
    lt = bare[bare.regime.isin(regs)]

    def band(sel, name):
        r = (sel.groupby(["model", "method", "dataset", "regime"])
             .auc.median().rename(name).reset_index())
        return r
    out = band(lt[lt.depth.isin(MID_DEPTHS)], "auc_mid").merge(
        band(lt, "auc_all"), how="outer",
        on=["model", "method", "dataset", "regime"])
    out["label_group"] = out.dataset.map(LABEL_GROUP)
    out["labeltype_absent_in_ldo"] = out.dataset.isin(ABSENT_TYPE_TARGETS)
    out.to_csv(C.DATA / "q16_labeltype_absent.csv", index=False)

    # ---- digest ----------------------------------------------------------
    lines = ["Q16 digest - label semantics of transfer (Original prompts, "
             "raw_last)", ""]
    dg = grid[(grid.method == "diff") & (grid.depth_agg == "mid")]
    lines.append("Group transfer matrix, diff-in-means, median AUC over "
                 "(source,target) pairs x 65-85% depths:")
    for model in ("gemma", "qwen"):
        p = dg[dg.model == model].pivot(index="source_group",
                                        columns="target_group",
                                        values="auc_median")
        p = p.reindex(index=GROUP_ORDER, columns=GROUP_ORDER)
        lines.append(f"  {C.MODEL_LABEL[model]} (rows=trained-on):")
        for sg in GROUP_ORDER:
            vals = "  ".join(f"{GROUP_LABEL[tg]:>11} {p.loc[sg, tg]:.3f}"
                             for tg in GROUP_ORDER)
            lines.append(f"    {GROUP_LABEL[sg]:>11} ->  {vals}")
    cross = ss[(ss.depth.isin(MID_DEPTHS)) & (ss.method == "diff")
               & (ss.source_group != ss.target_group)]
    for model in ("gemma", "qwen"):
        cm = cross[cross.model == model]
        lines.append(f"  cross-label-type cells above 0.5 "
                     f"({C.MODEL_LABEL[model]}): "
                     f"{(cm.auc > 0.5).mean():.0%} of {len(cm)} "
                     f"(pair x depth); below 0.48: "
                     f"{(cm.auc < 0.48).mean():.0%}")
    # the pairs that are at/below chance at the mid band (pair medians)
    pm = (cross.groupby(["model", "source_dataset", "dataset"])
          .auc.median().reset_index())
    bad = pm[pm.auc < 0.5].sort_values(["model", "auc"])
    lines.append("  cross-type pairs with mid-band median AUC < 0.5:")
    for r in bad.itertuples():
        lines.append(f"    {r.model}: {r.source_dataset} -> "
                     f"{r.dataset}  {r.auc:.3f}")
    lines.append("")

    lines.append("Label-type-absent ladder (median over 65-85% depths; "
                 "ceiling = within-dataset L2):")
    lad = out[out.dataset.isin(ABSENT_TYPE_TARGETS)]
    for model in ("gemma", "qwen"):
        lines.append(f"  {C.MODEL_LABEL[model]}: target  ceiling(l2)  "
                     "pool w/o target (diff)  label type absent (diff)  "
                     "label type absent (l2)")
        for tgt in ABSENT_TYPE_TARGETS:
            def v(method, regime):
                s = lad[(lad.model == model) & (lad.method == method)
                        & (lad.dataset == tgt) & (lad.regime == regime)]
                return float(s.auc_mid.iloc[0]) if len(s) else float("nan")
            lines.append(
                f"    {tgt:<26} {v('l2reg', 'within_cv'):.3f}       "
                f"{v('diff', 'leave_dataset_out'):.3f}                 "
                f"{v('diff', 'leave_domain_out'):.3f}                  "
                f"{v('l2reg', 'leave_domain_out'):.3f}")
    lines += ["", "Notes:",
              "- claude_blood_test_ranges is at-chance even within-dataset "
              "at this band (own L2 CV ceiling 0.45 Gemma / 0.59 Qwen), so "
              "its sub-0.5 transfer cells are a dead target, not "
              "anti-alignment; the substantive at/below-chance cross-type "
              "cells are Gemma's tf -> wildjailbreak ones.",
              "- depth fragility (F13): outside the 65-85% band the Gemma "
              "label-type-absent cells collapse (diff leave_domain_out at "
              "97% depth: xstest 0.14, wjb_vanilla 0.20, hh 0.54); the "
              "band restriction is load-bearing and stated on the figure.",
              "- wildjailbreak_adversarial leave_dataset_out pools "
              "wjb_vanilla = same-label rewrite twins of the test prompts "
              "(the report's pairing caveat); its leave_domain_out row "
              "drops the whole jailbreak domain, so THAT number is "
              "twin-free.",
              "- social_iqa carries the T/F wrapper line on the bare "
              "collections (roster note in classifiers/data.py); it is one "
              "of the 5 truth sources, not a target of interest here.",
              "- truth is never fully absent: truth-labelled units sit in "
              "4 of 5 domains, so no leave-domain-out fit lacks truth "
              "labels - the absent-label-type test exists only for safety "
              "and yes/no.",
              "- merged_included (target in pool) sits at the within "
              "ceiling for these targets (q16_labeltype_absent.csv), i.e. "
              "one pooled direction serves all three label types at once "
              "(F2); the ladder here is about how much survives when a "
              "label type was never in the pool at all."]
    text = "\n".join(lines)
    (C.DATA / "q16_digest.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
