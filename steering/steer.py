"""The sweep itself (runs ON compute cluster, on a GPU).

Read steering/README.md first — it says in plain words what every vector
and every edit in this file is, and why.

What this does, in one line: run each prompt through the model once, then
edit the numbers the model is holding at the last word of the prompt, many
different ways, and record what word it says next each time.

Why one pass per prompt is enough. A language model only ever looks
backwards, so changing the numbers at the LAST word of the prompt cannot
change anything the model already computed at earlier words. That means
the work done on earlier words can be saved once and reused for every
edit:

    full pass over the prompt, once
      -> take the numbers at layer L, last word
      -> keep everything computed for the earlier words
    per edit (three blocks and the output layer only, one word)
      -> feed the edited numbers back in, read the next-word scores

The check that this is legitimate, and it is run at the start of every
job: the shortcut with NO edit applied must reproduce what a full pass
produces. If it does not, the reuse is wrong and every number downstream
would be meaningless, so the job stops.

Three edits per vector:

  push        add the vector, scaled, at 18 sizes either side of zero.
              Does adding it change the answer?
  remove      subtract off however much of the model's numbers lie along
              the vector. Does the model still answer correctly without
              that component?
  land on     move each prompt so it sits exactly on one group's average
              position along the vector, instead of pushing every prompt
              the same amount. The step is per prompt: large for a prompt
              that starts far away, near zero for one already there.
              (Singh et al., ICML 2024, show this is the least-squares
              optimal way to move one group onto another.) The group
              averages come from stored numbers — nothing is refitted.

What is recorded per prompt and edit: for each of four word pairs —
True/False, Yes/No, safe/unsafe, harmless/harmful — how strongly the model
prefers the positive word (pooling over capitalisation variants, using the
extraction pipeline's own token lists), and what share of the next-word
probability that pair holds at all. Plus how far the model's whole
next-word distribution moved from the unedited version, which is the check
on whether an edit broke the model.

The push sizes, every vector's sign, and the shortcut check all use
True/False. The other three pairs are recorded so a finished run can be
re-read under a different answer vocabulary without paying for the sweep
again: a dataset the model will not answer in True/False words may still
carry its verdict in safe/unsafe ones. On Gemma, xstest read 0.51 (chance)
under True/False and 0.67 under safe/unsafe on exactly the same prompts.

    python steering/steer.py [--smoke] [--equivalence-only]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from classifiers.data import COLLECTIONS, binary_labels, load_hf  # noqa: E402
import steering.manifest_schema as MS  # noqa: E402

OUT = REPO / "steering" / "out"
DEFAULT_COLLECTION = "gemma_tf_a"
# Set in main() from --collection. The model id and the layer roster come
# from the collection itself, so the two models stay in step: index 2 of
# each roster is the same relative depth (Gemma 29 = 88.2%, Qwen 24 =
# 89.3%), which is how every other comparison in this repo pairs them.
COLLECTION = DEFAULT_COLLECTION
MODEL_ID = COLLECTIONS[COLLECTION].model
LAYER = COLLECTIONS[COLLECTION].layers[2]   # hidden_states index LAYER+1

_SPEC = importlib.util.spec_from_file_location(
    "tf_next_token_logits",
    REPO / "extract_activations" / "shared_core" / "tf_next_token_logits.py")
tf_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["tf_next_token_logits"] = _SPEC.loader and tf_mod  # dataclass needs it
_SPEC.loader.exec_module(tf_mod)

# Every word pair the extraction pipeline scores, in the order stored:
# (positive side, negative side, mass key). Read off that module so the two
# can never drift apart.
PAIRS = tuple((pr.name, tf_mod._negative_side_name(pr), pr.mass_key)
              for pr in tf_mod.CONCEPT_PAIRS)
STORE_COLS = tuple([c for pos, neg, mass in PAIRS
                    for c in (f"{pos}_lse", f"{neg}_lse", mass)]
                   + ["kl", "extra"])

# target unit -> (HF source, filter over the stored topic array).
#
# wildjailbreak_adversarial was dropped for the Gemma run because Gemma
# barely answers its True/False question (0.602). Qwen does answer it
# (0.877), so it is included for that model. Its two halves
# share source ids, so on this target only the family-held-out fit is
# trustworthy — anything whose pool contains the vanilla half is optimistic.
TARGETS = {
    "common_claim": ("geometry_truth",
                     lambda t: t == "common_claim_true_false"),
    "claude_vds": ("claude_vds", None),
    "wildjailbreak_vanilla": ("wildjailbreak",
                              lambda t: t.startswith("vanilla")),
    "xstest": ("xstest", None),
    "pubmedqa_labeled": ("pubmedqa_labeled", None),
    "wildjailbreak_adversarial": ("wildjailbreak",
                                  lambda t: t.startswith("adversarial")),
}
# Every vector is length 1 (directions.py normalizes and asserts it), and
# on a given dataset every vector gets the SAME push sizes: size x one
# shared base amount. The base is that dataset's True-average-minus-
# False-average distance measured along its OWN average-difference vector,
# so size 1 is the distance that carries the average False prompt onto the
# average True one. Because the sizes are identical, "this vector did
# nothing at the same size where that one worked" is a fair statement, and
# the comparison vectors are pushed exactly as hard as the real ones. Each
# vector's own True-False distance is still recorded for the analysis, but
# it does not change how hard anything is pushed.
#
# The ladder deliberately runs past the point where the answer stops
# moving: the +-64 ends exist only to CHECK that pushing hard the positive
# way pins the answer at True and the other way pins it at False. Effects
# are measured only at sizes where the rest of the model's output survived.
FRACS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
# The largest number fp16 can hold. bf16 reaches ~3e38 and never comes
# near this, so it only matters on cards with no bf16 at all.
FP16_MAX = 65504.0


def load_eval(target: str, layer: int, n_per_class: int, seed: int) -> dict:
    source, topic_filter = TARGETS[target]
    c = COLLECTIONS[COLLECTION]
    d = load_hf(source, "last", layer, COLLECTION)
    y = binary_labels(d["labels"], source, c.label_map.get(source))
    keep = np.ones(len(y), dtype=bool)
    if topic_filter is not None:
        keep = np.array([topic_filter(str(t)) for t in d["topic"]])
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        pool = idx[y[idx] == label]
        chosen.append(rng.choice(pool, size=min(n_per_class, len(pool)),
                                 replace=False))
    sel = np.sort(np.concatenate(chosen))
    return {"prompts": [str(p) for p in d["prompts"][sel]],
            "y": y[sel].astype(np.int8),
            "sample_id": d["sample_id"][sel],
            "stored": np.asarray(d["X"][sel], dtype=np.float32)}


def text_model(model):
    inner = model.model if hasattr(model, "model") else model
    return inner.language_model if hasattr(inner, "language_model") else inner


def templated(tok, prompts, agp: bool):
    return [tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=agp)
            for p in prompts]


def make_batches(texts, tok, token_budget: int, max_rows: int = 16):
    """Tokenize once, sort by length, pack under rows*max_len <= budget."""
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    lengths = np.array([len(x) for x in ids])
    order = np.argsort(lengths, kind="stable")
    groups, batch = [], []
    for i in order:
        cand = batch + [int(i)]
        if batch and (len(cand) > max_rows
                      or len(cand) * lengths[cand].max() > token_budget):
            groups.append(batch)
            batch = [int(i)]
        else:
            batch = cand
    if batch:
        groups.append(batch)
    return [(np.array(g),
             tok.pad({"input_ids": [ids[i] for i in g]}, return_tensors="pt"))
            for g in groups]


class Tail:
    """Re-run the final token as a normal 1-token decode step against the
    saved work from the full pass, replacing the numbers at the edited
    layer with the edited ones.

    Deliberately NOT a hand-rolled stack of the upper blocks: Gemma-3
    interleaves local and global attention with separate rotary embeddings,
    and left-padding needs masking, so reimplementing the tail invites
    version-specific bugs. Going through the model's own forward reuses its
    mask construction, rope selection and cache handling; the cost is running
    the blocks BELOW the steering layer on one token too, which is negligible
    against a full-sequence prefill (and their output is discarded by the
    hook anyway)."""

    def __init__(self, model, layer: int):
        self.model = model
        self.h_new = None
        block = text_model(model).layers[layer]
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.h_new is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] = self.h_new.to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    @torch.no_grad()
    def logits(self, h_new, last_ids, cache, attn_mask, keep: int):
        """h_new: (B, d) the edited numbers. cache: saved work, cropped to
        `keep` positions. Cropped again afterwards so the same cache serves
        every condition."""
        self.h_new = h_new
        try:
            out = self.model(input_ids=last_ids, attention_mask=attn_mask,
                             past_key_values=cache, use_cache=True)
        finally:
            self.h_new = None
            crop(cache, keep)
        return out.logits[:, -1, :].float()


@torch.no_grad()
def prefill(model, enc, layer):
    from transformers import DynamicCache
    cache = DynamicCache()
    res = model(**enc, past_key_values=cache, use_cache=True,
                output_hidden_states=True)
    h = res.hidden_states[layer + 1][:, -1, :].float()
    logits = res.logits[:, -1, :].float()
    return h, cache, logits


def crop(cache, n: int):
    """Drop the last position from every layer's K/V so it is recomputed."""
    if hasattr(cache, "crop"):
        cache.crop(n)
        return cache
    for lyr in cache.layers if hasattr(cache, "layers") else []:
        lyr.keys, lyr.values = lyr.keys[:, :, :n], lyr.values[:, :, :n]
    return cache


def tf_scores(logits, true_ids, false_ids):
    t = torch.logsumexp(logits[:, true_ids], -1)
    f = torch.logsumexp(logits[:, false_ids], -1)
    p = torch.softmax(logits, -1)
    mass = p[:, true_ids].sum(-1) + p[:, false_ids].sum(-1)
    return t.cpu().numpy(), f.cpu().numpy(), mass.cpu().numpy()


def concept_scores(logits, tensors) -> list:
    """Per word pair, in PAIRS order: the pooled log-sum-exp of each side
    and the share of next-token probability that pair receives."""
    p = torch.softmax(logits, -1)
    out = []
    for pos, neg, _ in PAIRS:
        pi, ni = tensors[pos], tensors[neg]
        out.append(torch.logsumexp(logits[:, pi], -1).cpu().numpy())
        out.append(torch.logsumexp(logits[:, ni], -1).cpu().numpy())
        out.append((p[:, pi].sum(-1) + p[:, ni].sum(-1)).cpu().numpy())
    return out


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def ensure_directions(layer: int, tag: str) -> None:
    """Build the direction file in-job if it is not already there.

    The CPU queue on this cluster runs thousands of jobs deep while the GPU
    queue runs hundreds, so waiting on a separate CPU build job can block a
    GPU job that is already near the front. Building here costs about five
    minutes of the GPU's time and removes that coupling. Runs the real
    builder as a subprocess so its assertions, manifest and printout land
    in this job's log unchanged."""
    npz = OUT / f"directions{tag}_layer{layer}.npz"
    if npz.exists():
        print(f"using existing {npz.name}", flush=True)
        return
    if os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        sys.exit(f"{npz.name} is missing and this is an array task — six "
                 "tasks building the same file would race. Run the "
                 "directions job first and make the array depend on it.")
    print(f"{npz.name} missing — building directions in-job", flush=True)
    import subprocess
    cmd = [sys.executable, "-u", str(REPO / "steering" / "directions.py"),
           "--layer", str(layer), "--collection", COLLECTION]
    if tag:
        cmd += ["--tag", tag.lstrip("_")]
    subprocess.run(cmd, check=True, cwd=str(REPO))
    if not npz.exists():
        sys.exit("the direction builder finished but wrote no file")


# Batches are the natural cut point: the outer loop walks batches of
# prompts and the inner loop walks directions, so once a batch returns,
# every one of the ~1,345 keys is complete for that batch's rows and
# nothing later can change them. A snapshot is therefore just the whole
# store, which is n x 14 float32 per key — about 75 MB at n=1000. Cheap
# every ten batches, far too heavy every batch.
CKPT_EVERY = 10


def _ckpt_sig(target: str, keys: list, n: int) -> str:
    """What a checkpoint is FOR. The key list changes with the direction
    set and with --smoke, and n changes with the sample; any of those makes
    an older file meaningless, and silently resuming onto it would mix two
    experiments in one CSV."""
    h = hashlib.sha1("\n".join(keys).encode()).hexdigest()[:12]
    return f"{target}|n={n}|k={len(keys)}|{h}"


def _ckpt_save(path: Path, store: dict, sig: str, n_done: int) -> None:
    tmp = path.with_suffix(".writing.npz")
    np.savez(tmp, arr=np.stack([store[k] for k in store]),
             keys=np.array(list(store)), sig=np.array(sig),
             n_done=np.array(n_done))
    # Atomic: a walltime kill during the write leaves the previous good
    # checkpoint untouched rather than a half-written one.
    os.replace(tmp, path)


def _ckpt_load(path: Path, store: dict, sig: str, target: str) -> int:
    """Refill store from a checkpoint; return how many batches it covers.
    Any doubt returns 0 and the target recomputes from the start — a wrong
    resume is far more expensive than a repeated one."""
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["sig"]) != sig:
                print(f"  {target}: checkpoint is for a different direction "
                      f"set or sample ({str(z['sig'])}), ignoring it",
                      flush=True)
                return 0
            arr, keys = z["arr"], [str(k) for k in z["keys"]]
            n_done = int(z["n_done"])
        for i, k in enumerate(keys):
            store[k] = arr[i]
        print(f"  {target}: resuming, {n_done} batches already done",
              flush=True)
        return n_done
    except Exception as exc:                        # noqa: BLE001
        print(f"  {target}: checkpoint unreadable ({type(exc).__name__}), "
              "starting this target over", flush=True)
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help="gemma_tf_a or qwen_tf_a; sets the model and the "
                         "activation cache")
    ap.add_argument("--layer", type=int, default=None,
                    help="default: index 2 of the collection's layer "
                         "roster (Gemma 29, Qwen 24 — the same depth)")
    ap.add_argument("--targets", default="",
                    help="comma-separated subset, one per array task")
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run without a GPU. Only ever for a tiny local "
                         "check — the real sweep cannot finish on a CPU.")
    ap.add_argument("--dtype", choices=("auto", "bf16", "fp16"),
                    default="auto",
                    help="auto = bf16 where the card has it, fp16 where "
                         "it does not. fp16 holds more detail than bf16 "
                         "but stops at 65,504, so the largest push sizes "
                         "get dropped per dataset and reported.")
    ap.add_argument("--equivalence-only", action="store_true",
                    help="run only the tail-vs-full check and exit")
    ap.add_argument("--directions-tag", default="",
                    help="read directions_<tag>_layer*.npz and write "
                         "runs_<tag>_layer*/ (e.g. 'mapped')")
    ap.add_argument("--max-frac", type=float, default=None,
                    help="run only the push sizes up to this multiple of "
                         "the dose base (1 keeps 0.25, 0.5 and 1 either "
                         "side of zero, the sizes the slope is fitted "
                         "over). Default: all sizes.")
    ap.add_argument("--adds-only", action="store_true",
                    help="skip the removal and mean-matching edits; run "
                         "only the pushes")
    ap.add_argument("--part", default="",
                    help="i/n: run only the i-th (1-based) of n interleaved "
                         "slices of each target's vectors and write "
                         "<target>_part<i>of<n>.csv, so one target can be "
                         "spread over several GPU tasks")
    args = ap.parse_args()
    part_i, part_n = 1, 1
    if args.part:
        a_, b_ = args.part.split("/")
        part_i, part_n = int(a_), int(b_)
        if not 1 <= part_i <= part_n:
            sys.exit(f"--part {args.part}: need 1 <= i <= n")
    part_tag = f"_part{part_i}of{part_n}" if part_n > 1 else ""

    global COLLECTION, MODEL_ID, LAYER, TARGETS
    COLLECTION = args.collection
    MODEL_ID = COLLECTIONS[COLLECTION].model
    LAYER = COLLECTIONS[COLLECTION].layers[2]
    if args.layer is None:
        args.layer = LAYER
    if args.targets:
        want = [x.strip() for x in args.targets.split(",") if x.strip()]
        unknown = [x for x in want if x not in TARGETS]
        if unknown:
            sys.exit(f"unknown targets: {unknown}")
        TARGETS = {k: v for k, v in TARGETS.items() if k in want}
    print(f"collection {COLLECTION} | model {MODEL_ID} | layer "
          f"{args.layer} | targets {list(TARGETS)}", flush=True)

    tag = f"_{args.directions_tag}" if args.directions_tag else ""
    if not tag and COLLECTION != DEFAULT_COLLECTION:
        tag = f"_{COLLECTION.split('_')[0]}"     # directions_qwen_layer24
    smoke_tag = "_smoke" if args.smoke else ""
    # `tag` already carries the collection (set just above), so the run
    # directory must not add it a second time.
    run_dir = OUT / f"runs{tag}{smoke_tag}_layer{args.layer}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Never fall back to a CPU silently: the full sweep is not practical
    # there, while a tiny explicit check remains available via --allow-cpu.
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    elif args.allow_cpu:
        device = "cpu"
        print("WARNING: no GPU; running on CPU because --allow-cpu was "
              "passed. This is only viable for a tiny check.", flush=True)
    else:
        sys.exit("no GPU is visible to torch, and this sweep cannot run "
                 "on a CPU. Submit with --gres=gpu:1 (see "
                 "steering/sbatch/*_steer.sbatch), or pass --allow-cpu if "
                 "you really mean to run a tiny check without one.")

    ensure_directions(args.layer, tag)
    with np.load(OUT / f"directions{tag}_layer{args.layer}.npz") as z:
        names = [str(n) for n in z["names"]]
        W = z["W"].astype(np.float32)
    manifest = pd.read_csv(OUT / f"manifest{tag}_layer{args.layer}.csv")
    MS.validate(manifest, "steer.py")
    per_target = {t: [n for n, tg in zip(manifest.name, manifest.target)
                      if tg in (t, "ALL")] for t in TARGETS}
    uses_of = dict(zip(manifest.name, manifest.uses_target_data))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {MODEL_ID} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # Avoid a full host-memory copy while transferring the model to the GPU.
    if device != "cuda":
        dt, half = torch.float32, False
    elif args.dtype == "fp16" or (args.dtype == "auto"
                                  and not torch.cuda.is_bf16_supported()):
        dt, half = torch.float16, True
        print(f"running in fp16 (card: {torch.cuda.get_device_name(0)}). It "
              f"holds more detail than bf16 but stops at {FP16_MAX:,.0f}, so "
              "any push size that would pass that is dropped for that "
              "dataset and named in the log.", flush=True)
    else:
        if not torch.cuda.is_bf16_supported():
            sys.exit(f"--dtype bf16 asked for, but "
                     f"{torch.cuda.get_device_name(0)} has no bf16. Use "
                     "--dtype fp16, or send the job to a newer card.")
        dt, half = torch.bfloat16, False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dt, low_cpu_mem_usage=True,
    ).to(device).eval()

    t_ids, f_ids = tf_mod.get_tf_token_id_lists(tok)
    true_ids = torch.tensor(t_ids, device=device)
    false_ids = torch.tensor(f_ids, device=device)
    concept_ids, concept_id_lists = tf_mod.concept_token_tensors(tok, device)
    tail = Tail(model, args.layer)

    n_pc = 8 if args.smoke else args.n_per_class
    evals = {t: load_eval(t, args.layer, n_pc, args.seed) for t in TARGETS}
    for t, ev in evals.items():
        print(f"{t}: {len(ev['y'])} rows, {int(ev['y'].sum())} True", flush=True)

    # ---- encoding check: which chat-template setting reproduces the stored
    # activations? extraction never recorded add_generation_prompt ----------
    checks, scales = {}, {}
    for agp in (True, False):
        cs, ns = [], []
        for t, ev in evals.items():
            k = min(16, len(ev["prompts"]))
            texts = templated(tok, ev["prompts"][:k], agp)
            for idx, enc_cpu in make_batches(texts, tok, args.token_budget):
                enc = {k2: v.to(device) for k2, v in enc_cpu.items()}
                h, _, _ = prefill(model, enc, args.layer)
                a = h.cpu().numpy()
                b = ev["stored"][idx]
                na = np.linalg.norm(a, axis=1)
                nb = np.linalg.norm(b, axis=1)
                cs.append((a * b).sum(1) / (na * nb))
                ns.append(na / nb)
        checks[agp] = float(np.median(np.concatenate(cs)))
        scales[agp] = float(np.median(np.concatenate(ns)))
        print(f"  add_generation_prompt={agp}: median cos vs stored "
              f"{checks[agp]:.5f}, median length ratio {scales[agp]:.5f}",
              flush=True)
    agp = max(checks, key=checks.get)
    if checks[agp] < 0.99:
        sys.exit(f"the numbers this job computes do not match the stored "
                 f"ones (best median cos {checks[agp]:.4f}) — refusing to "
                 "edit numbers the saved fits were not built from")
    # Cosine only checks which way they point. Push sizes and the
    # mean-matching destinations are absolute positions taken from the
    # stored numbers, so their LENGTH has to match too.
    # Deliberately loose. Anything that would really invalidate the run --
    # wrong layer, wrong prompt template, activations stored normalized --
    # is off by far more than 10%, while a few percent from a different
    # library version would only bias the mean-matching destinations
    # slightly. The ratio is printed either way, so a small drift is
    # visible in the log rather than silently killing twelve array tasks.
    if not 0.90 <= scales[agp] <= 1.10:
        sys.exit(f"the numbers this job computes point the right way but "
                 f"are {scales[agp]:.4f} times the stored ones in length — "
                 "every push size and every mean-matching destination is "
                 "taken from the stored numbers, so they would be wrong")
    print(f"encoding: add_generation_prompt={agp}", flush=True)

    batches = {t: make_batches(templated(tok, ev["prompts"], agp), tok,
                               args.token_budget)
               for t, ev in evals.items()}

    # ---- tail-vs-full equivalence: the load-bearing check ------------------
    t0 = next(iter(TARGETS))
    idx, enc_cpu = batches[t0][0]
    enc = {k: v.to(device) for k, v in enc_cpu.items()}
    h, cache, full_logits = prefill(model, enc, args.layer)
    T = enc["input_ids"].shape[1]
    last_ids = enc["input_ids"][:, -1:]
    crop(cache, T - 1)
    tail_logits = tail.logits(h, last_ids, cache, enc["attention_mask"], T - 1)
    d = (tail_logits - full_logits).abs().max().item()
    rel = d / full_logits.abs().max().item()
    # gate on the quantity actually used — the T/F log-odds — not on raw
    # logits, whose absolute scale (~58 here) makes bf16 rounding look large
    ta, tb, _ = tf_scores(tail_logits, true_ids, false_ids)
    fa, fb, _ = tf_scores(full_logits, true_ids, false_ids)
    dlo_err = float(np.abs((ta - tb) - (fa - fb)).max())
    spread = float(np.std(fa - fb))
    # How coarse the arithmetic is at this size of number: two logits this
    # large cannot differ by less than one of these steps, whatever the
    # cache does. The log-odds error combines four such quantities, each
    # roundable by up to half a step, so about two steps is the floor the
    # format imposes and four leaves a factor of two of margin.
    step = float(full_logits.abs().max()) * torch.finfo(dt).eps
    tol = max(4.0 * step, 0.05 * spread)
    print(f"tail-vs-full at alpha=0: max logit diff {d:.4f} (relative "
          f"{rel:.2e}) | max d-log-odds error {dlo_err:.4f} vs tolerance "
          f"{tol:.4f} (4 x {step:.4f} format step, or 5% of across-prompt "
          f"sd {spread:.2f})", flush=True)
    if dlo_err > tol:
        sys.exit("tail reconstruction does not reproduce the full forward's "
                 "T/F log-odds by more than the arithmetic's own resolution "
                 "— the KV-cache reuse is wrong and every downstream number "
                 "would be invalid")
    if args.equivalence_only:
        return

    fracs = FRACS[:1] if args.smoke else FRACS
    if args.max_frac is not None:
        fracs = tuple(f for f in fracs if f <= args.max_frac + 1e-9)
        if not fracs:
            sys.exit(f"--max-frac {args.max_frac} keeps no push size")
    signed = [s * f for f in fracs for s in (+1, -1)]
    meta = {"model": MODEL_ID, "layer": args.layer, "collection": COLLECTION,
            "device": (torch.cuda.get_device_name(0)
                       if device == "cuda" else "cpu"),
            "add_generation_prompt": agp, "encoding_cos": checks,
            "tail_vs_full_rel": rel, "true_ids": t_ids, "false_ids": f_ids,
            "word_pairs": [list(pr) for pr in PAIRS],
            "word_pair_token_ids": {k: [int(i) for i in v]
                                  for k, v in concept_id_lists.items()},
            "fracs": list(fracs), "seed": args.seed, "n_per_class": n_pc,
            "adds_only": bool(args.adds_only), "part": args.part or "1/1",
            "dose_unit": ("one shared base per target — the eval sample's "
                          "true-false gap along the target's own mean "
                          "difference; all directions unit norm, "
                          "alpha = frac x base, equal for every direction"),
            "dose_units": {}, "fracs_used": {},
            "dtype": str(dt).replace("torch.", ""),
            "n_rows": {t: int(len(e["y"])) for t, e in evals.items()}}

    for target, ev in evals.items():
        out_path = run_dir / f"{target}{part_tag}.csv"
        if out_path.exists() and not args.smoke:
            print(f"{target}: exists, skip", flush=True)
            continue
        # real directions first, so --smoke exercises them, comparisons
        # after. Tested against the manifest's own name for the value, so
        # a rename in the builder cannot leave this key silently never
        # firing the way "control" did.
        ordered = sorted(per_target[target],
                         key=lambda n: uses_of.get(n, "") == MS.COMPARISON)
        if part_n > 1:
            # interleaved, so every part carries a mix of real and
            # comparison vectors and finishes in about the same time
            ordered = ordered[part_i - 1::part_n]
        dir_names = ordered if not args.smoke else ordered[:3]
        V = {n: torch.tensor(W[names.index(n)], device=device,
                             dtype=torch.float32) for n in dir_names}
        # one shared dose base per target: the eval sample's true-false gap
        # along the target's own mean difference (stored activations match
        # the prefill's to cos 0.99999, so they are the same vectors).
        # Every direction is unit norm, so alpha = frac x base injects the
        # same absolute length for every direction. Per-direction gaps are recorded for
        # the analysis but do not change the dose.
        yv = ev["y"].astype(bool)
        Xe = ev["stored"].astype(np.float64)
        own_diff_name = manifest[(manifest.target == target)
                                 & (manifest.kind == MS.OWN_DIFF)].name.iloc[0]
        gaps = {n: float((Xe[yv] @ W[names.index(n)]).mean()
                         - (Xe[~yv] @ W[names.index(n)]).mean())
                for n in dir_names}
        # where each class sits along each direction — the destinations for
        # the mean-matching intervention
        class_pos = {n: (float((Xe[yv] @ W[names.index(n)]).mean()),
                         float((Xe[~yv] @ W[names.index(n)]).mean()))
                     for n in dir_names}
        # the base comes from the own difference whether or not this part
        # of the run happens to include that vector
        w_own = W[names.index(own_diff_name)]
        ref = float((Xe[yv] @ w_own).mean() - (Xe[~yv] @ w_own).mean())
        if ref <= 0:
            sys.exit(f"{target}: the own mean difference has a non-positive "
                     f"gap ({ref:.2f}) on the eval sample — orientation is "
                     "broken, refusing to run")
        meta["dose_units"][target] = {
            "base": round(ref, 2),
            "gaps": {n_: round(gaps[n_], 2) for n_ in dir_names},
            "class_positions": {n_: [round(class_pos[n_][0], 2),
                                     round(class_pos[n_][1], 2)]
                                for n_ in dir_names}}
        print(f"  {target}: shared dose base {ref:,.0f} "
              f"(own-mean-difference gap); all {len(dir_names)} directions "
              "unit norm, equal multipliers"
              + ("" if args.adds_only else
                 "; plus mean matching to each class average"), flush=True)
        # In fp16, drop only the push sizes that would actually pass the
        # limit — measured, not assumed: the largest number this dataset
        # already holds, plus the largest this size would add on top.
        signed_t, kept, dropped = signed, list(fracs), []
        if half:
            hmax = float(np.abs(Xe).max())
            vmax = max(float(V[d_].abs().max()) for d_ in dir_names)
            room = 0.9 * FP16_MAX
            kept = [f for f in fracs if hmax + f * ref * vmax <= room]
            dropped = [f for f in fracs if f not in kept]
            signed_t = [s * f for f in kept for s in (+1, -1)]
            print(f"  {target}: fp16 — holding up to {hmax:,.0f} already; "
                  f"keeping sizes {kept}"
                  + (f", dropping {dropped} (largest would reach "
                     f"{hmax + max(dropped) * ref * vmax:,.0f})"
                     if dropped else ", none dropped"), flush=True)
        meta["fracs_used"][target] = {"kept": [float(f) for f in kept],
                                      "dropped": [float(f) for f in dropped]}

        n = len(ev["y"])
        keys = (["baseline"]
                + [f"{d_}|add|{fr}" for d_ in dir_names for fr in signed_t])
        if not args.adds_only:
            keys += ([f"{d_}|ablate|0" for d_ in dir_names]
                     + [f"{d_}|{kind}|0" for d_ in dir_names
                        for kind in ("match_pos", "match_neg")])
        cols = {k: {} for k in keys}
        store = {k: np.zeros((n, len(STORE_COLS)), dtype=np.float32)
                 for k in cols}

        # Resume where a killed job left off. Skipped in --smoke, which is
        # a few seconds of work and would only leave stale files behind.
        ckpt_path = run_dir / f"{target}{part_tag}.ckpt.npz"
        sig = _ckpt_sig(target, list(store), n)
        first_batch = 0
        if ckpt_path.exists() and not args.smoke:
            first_batch = _ckpt_load(ckpt_path, store, sig, target)

        t_start = time.time()
        for bi, (idx, enc_cpu) in enumerate(batches[target]):
            if bi < first_batch:
                continue
            enc = {k: v.to(device) for k, v in enc_cpu.items()}
            h, cache, _ = prefill(model, enc, args.layer)
            T = enc["input_ids"].shape[1]
            last_ids = enc["input_ids"][:, -1:]
            crop(cache, T - 1)

            # baseline first: its full next-word distribution is the
            # reference for every KL below
            lg0 = tail.logits(h, last_ids, cache, enc["attention_mask"],
                              T - 1)
            logp0 = torch.log_softmax(lg0, -1)
            p0 = logp0.exp()
            s0 = concept_scores(lg0, concept_ids)
            hnorm = h.norm(dim=-1).cpu().numpy()
            store["baseline"][idx] = np.stack(
                s0 + [np.zeros_like(s0[0]), hnorm], 1)

            def run(key, h_new, extra):
                lg = tail.logits(h_new, last_ids, cache,
                                 enc["attention_mask"], T - 1)
                kl = (p0 * (logp0 - torch.log_softmax(lg, -1))
                      ).sum(-1).cpu().numpy()
                store[key][idx] = np.stack(
                    concept_scores(lg, concept_ids) + [kl, extra], 1)

            for d_ in dir_names:
                v = V[d_]
                for fr in signed_t:
                    alpha = fr * ref
                    run(f"{d_}|add|{fr}", h + alpha * v,
                        np.full(len(idx), alpha, dtype=np.float32))
                if args.adds_only:
                    continue
                proj = h @ v
                run(f"{d_}|ablate|0", h - proj.unsqueeze(-1) * v,
                    proj.abs().cpu().numpy())
                # mean matching: land every prompt on a class's average
                # position along this direction. The dose is per-prompt, so
                # `extra` records the signed distance each prompt moved.
                m_pos, m_neg = class_pos[d_]
                for kind, m in (("match_pos", m_pos), ("match_neg", m_neg)):
                    step = m - proj
                    run(f"{d_}|{kind}|0", h + step.unsqueeze(-1) * v,
                        step.cpu().numpy())
            if half:
                # The rule above bounds what gets injected. It cannot bound
                # what the remaining blocks compute from it, so check rather
                # than assume — a silent inf here would reach the CSV as a
                # missing number and read as a null result.
                blown = [k for k in store
                         if not np.isfinite(store[k][idx]).all()]
                if blown:
                    sys.exit(
                        f"{target}: fp16 overflowed inside the model on batch "
                        f"{bi} for {len(blown)} of {len(store)} edits, e.g. "
                        f"{blown[:3]}. The kept push sizes are still too "
                        "large for this card — rerun on a card with bf16, or "
                        "cut the largest kept size.")
            if bi % 10 == 0:
                print(f"  {target} batch {bi + 1}/{len(batches[target])} "
                      f"({time.time() - t_start:.0f}s)", flush=True)
            if (bi + 1) % CKPT_EVERY == 0 and not args.smoke:
                _ckpt_save(ckpt_path, store, sig, bi + 1)

        rows = []
        for key, arr in store.items():
            if key == "baseline":
                d_, kind, fr = "baseline", "none", 0.0
            else:
                d_, kind, fr = key.split("|")
            rows.append(pd.DataFrame({
                "target": target, "direction": d_, "intervention": kind,
                "frac": float(fr), "sample_id": ev["sample_id"], "y": ev["y"],
                **{c: arr[:, i] for i, c in enumerate(STORE_COLS)}}))
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(out_path, index=False, float_format="%.6g")
        ckpt_path.unlink(missing_ok=True)
        base = df[df.direction == "baseline"]
        print(f"{target}: {len(dir_names)} directions x "
              f"{len(signed_t) + 1} interventions in "
              f"{time.time() - t_start:.0f}s | baseline AUC "
              f"{auc(base.y, base.true_lse - base.false_lse):.3f}, median "
              f"T/F mass {base.tf_mass.median():.3f} -> {out_path.name}",
              flush=True)

    # One file per array task. Six tasks writing one run_meta.json would
    # leave only the last one's push sizes and group positions, and could
    # interleave into unreadable JSON; analyze.py merges run_meta*.json.
    stem = ("run_meta" if os.environ.get("SLURM_ARRAY_TASK_ID") is None
            else "run_meta_" + "_".join(sorted(TARGETS))) + part_tag
    (run_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
