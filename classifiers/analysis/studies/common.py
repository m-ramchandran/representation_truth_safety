"""Shared roster + loaders for the deep dive."""
import numpy as np

from .metrics_frame import load_metrics

EXCLUDED_DATASETS = ("claude_blood_test_ranges",
                     "claude_diagnosis_severity")
DATASETS = ("got", "common_claim", "counterfact", "harmful_harmless",
            "pubmedqa_artificial", "pubmedqa_labeled", "wildjailbreak",
            "xstest", "social_iqa", "claude_anatomy", "claude_vds")
TASKS = {"tf": ("got", "common_claim", "counterfact"),
         "medical": ("pubmedqa_artificial", "pubmedqa_labeled",
                     "claude_anatomy"),
         "jailbreak": ("harmful_harmless", "wildjailbreak", "xstest")}
TASK_OF = {d: t for t, ds in TASKS.items() for d in ds}
TASK_OF["social_iqa"] = "social_iqa"
TASK_OF["claude_vds"] = "claude_vds"

LINEAR = ["diff", "l1reg", "l2reg", "svm_linear", "lda"]
NONLINEAR = ["qda", "knn", "svm_rbf", "mlp", "xgboost"]
POOLED_ONLY = ["random_slopes", "random_slopes_task", "moe"]
FEATURES = ["raw_last", "pca_last", "pca_dict_last", "sae_last",
            "transcoder_last", "binned_activations_last"]
COLLS = ["tf_wrapped_a", "data_unwrap"]
LAYERS = [17, 22, 29, 33]

# xgboost on tf_wrapped_a per_dataset/leave_out is INCOMPLETE (see a2_gaps)
XGB_TFW_MISSING = ["claude_anatomy", "claude_blood_test_ranges",
                   "claude_vds", "social_iqa"]


def metrics(root=None):
    m = load_metrics() if root is None else load_metrics(root=root)
    # The blood-range task is outside the model's domain (chance even in
    # within-dataset CV), while diagnosis-severity is an n=40 eval-only
    # result. Neither belongs in aggregate performance or transfer claims.
    roster_columns = ["train_dataset", "test_dataset", "holdout"]
    m = m[~m[roster_columns].isin(EXCLUDED_DATASETS).any(axis=1)].copy()
    m["task"] = m["test_dataset"].map(TASK_OF)
    m["train_task"] = m["train_dataset"].map(TASK_OF)
    return m


def fmt(df, dec=3):
    return df.round(dec).to_string()


def hdr(s, ch="="):
    rule = ch * 96
    print(f"\n{rule}\n{s}\n{rule}")


def load_standardized_raw(*, layer=17, collection="data_unwrap",
                          max_per_dataset=2000, seed=0):
    """Sample raw activations uniformly and scale them in one shared space."""
    from classifiers.data import load_datasets

    loaded = load_datasets(layer, "last", ("raw",), collection)
    rng = np.random.default_rng(seed)
    raw, labels = {}, {}
    for name in DATASETS:
        X = np.asarray(loaded[name]["blocks"]["raw"], dtype=np.float32)
        y = np.asarray(loaded[name]["y"], dtype=np.int8)
        if len(X) > max_per_dataset:
            take = rng.choice(len(X), max_per_dataset, replace=False)
            X, y = X[take], y[take]
        raw[name], labels[name] = X, y

    stacked = np.concatenate(list(raw.values()))
    mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.
    standardized = {name: (X - mean) / scale for name, X in raw.items()}
    return standardized, labels
