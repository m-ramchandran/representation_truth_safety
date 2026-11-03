"""Q4: the model's own answer (paired-token logits) vs the probes.

Downloads every <stem>_tf_logits.npz for the four collections from the
activations HF repo (they are small), computes the model-readout AUC per
experiment unit under the collection's label convention, and saves:

  q4_logit_auc.csv   model, perturbation, dataset, readout, auc, n,
                     mass_median (how much next-token probability the
                     readout's token pair actually receives)

Readouts: p_true (True/False), p_yes (Yes/No), p_safe (safe/unsafe),
p_harmless (harmless/harmful). The canonical readout per unit:
p_true for statement datasets, p_yes for PubMedQA (bare), p_safe for the
jailbreak units (bare); under True/False framing everything is p_true.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import common as C
from classifiers.data import COLLECTIONS, HF_REPO, binary_labels

READOUT_MASS = {"p_true": "tf_mass", "p_yes": "yes_no_mass",
                "p_safe": "safe_unsafe_mass",
                "p_harmless": "harmful_harmless_mass"}

# unit -> canonical readout on the bare prompts
CANONICAL_BARE = {
    "got": "p_true", "common_claim": "p_true", "counterfact": "p_true",
    "claude_anatomy": "p_true", "claude_blood_test_ranges": "p_true",
    "claude_vds": "p_true", "social_iqa": "p_true",
    "pubmedqa_artificial": "p_yes", "pubmedqa_labeled": "p_yes",
    "harmful_harmless": "p_safe", "wildjailbreak_vanilla": "p_safe",
    "wildjailbreak_adversarial": "p_safe", "xstest": "p_safe",
}


def mask_for(unit: str, topic: np.ndarray | None):
    if topic is None:
        return slice(None)
    t = topic.astype(str)
    if unit in ("got", "common_claim", "counterfact"):
        parts = {"common_claim": "common_claim_true_false",
                 "counterfact": "counterfact_true_false"}
        if unit in parts:
            return t == parts[unit]
        return ~np.isin(t, list(parts.values()))
    if unit.startswith("wildjailbreak_"):
        return np.char.startswith(t, unit.split("_", 1)[1])
    return slice(None)


def main() -> None:
    from huggingface_hub import hf_hub_download

    rows = []
    for (model, pert), coll in C.COLLECTIONS.items():
        c = COLLECTIONS[coll]
        for unit in C.DATASETS:
            source = unit
            if unit in ("got", "common_claim", "counterfact"):
                source = "geometry_truth"
            elif unit.startswith("wildjailbreak_"):
                source = "wildjailbreak"
            hf_dir, stem = c.hf_files[source][0]
            try:
                path = hf_hub_download(HF_REPO, f"{hf_dir}/{stem}_tf_logits.npz",
                                       repo_type="dataset")
                d = np.load(path, allow_pickle=True)
            except Exception as exc:                    # noqa: BLE001
                print(f"  {coll} {unit}: {type(exc).__name__} skipped")
                continue
            y = binary_labels(d["labels"], source, c.label_map.get(source))
            if source in c.flip_labels:
                y = 1 - y
            topic = next((d[k] for k in ("topic", "types", "data_types")
                          if k in d.files), None)
            keep = mask_for(unit, topic)
            yk = y[keep]
            if len(np.unique(yk)) < 2:
                continue
            for readout in ("p_true", "p_yes", "p_safe", "p_harmless"):
                if readout not in d.files:
                    continue
                score = np.asarray(d[readout], dtype=float)[keep]
                mass = np.asarray(d[READOUT_MASS[readout]], dtype=float)[keep]
                rows.append({
                    "model": model, "perturbation": pert, "dataset": unit,
                    "readout": readout, "auc": roc_auc_score(yk, score),
                    "n": int(len(yk)),
                    "mass_median": float(np.median(mass)),
                    "canonical": (readout == "p_true" if pert == "true_false"
                                  else CANONICAL_BARE[unit] == readout),
                })
        print(f"  {coll} done")
    out = pd.DataFrame(rows)
    out.to_csv(C.DATA / "q4_logit_auc.csv", index=False)
    pd.set_option("display.width", 250)
    canon = out[out.canonical]
    print("\n== canonical model-readout AUC ==")
    print(canon.pivot_table(index="dataset", columns=["model", "perturbation"],
                            values="auc").round(3))
    print("\n== best readout per unit (post-hoc oracle over readouts) ==")
    best = out.loc[out.groupby(["model", "perturbation", "dataset"]).auc.idxmax()]
    print(best.pivot_table(index="dataset", columns=["model", "perturbation"],
                           values="auc").round(3))


if __name__ == "__main__":
    main()
