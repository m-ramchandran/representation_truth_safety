"""Tests for the --group-folds path: rows expanded from one source item
stay in one CV fold.

Covers the two halves separately — the group rules in `classifiers.data`
(what counts as one item) and the splitters in `classifiers.scripts.base`
(keeping a group whole) — plus the invariant that matters most in
practice: with `groups=None` the split is bit-identical to what the code
did before grouping existed, so `--no-group-folds` really does reproduce
the results already on disk.

Synthetic prompts throughout: no HF download, no GPU, runs in a second.
`base.py` is loaded by path because importing it the normal way pulls in
`classifiers/scripts/__init__.py`, which imports cuML — fine on a GPU node,
absent on the machine this test is meant to run on.

Run: python classifiers/tests/test_fold_groups.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from classifiers.data import (COLLECTIONS, GROUP_ID_KEYS,  # noqa: E402
                              GROUP_RULES, _split_wrapper, _stored_group_key,
                              group_ids)

_spec = importlib.util.spec_from_file_location(
    "classifiers_base",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "base.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


WRAP = "Determine whether the following is True or False."


def test_split_wrapper():
    """The wrapper line separates context from statement where it is there,
    and where it is not the whole prompt is the statement."""
    ctx, st = _split_wrapper(f"Alex left early.\n{WRAP}\nThe answer is 'x'.")
    assert ctx == "Alex left early." and st == "The answer is 'x'."
    ctx, st = _split_wrapper("The city of Krasnodar is in Russia.")
    assert ctx == "" and st == "The city of Krasnodar is in Russia."
    print("  test_split_wrapper")


def test_social_iqa_choices_share_a_group():
    """Every choice expanded from one question is one group; two questions
    about the same scenario are too (coarser is the safe direction)."""
    rule = GROUP_RULES["social_iqa"]
    keys = {rule(*_split_wrapper(
        f"Alex left early.\n{WRAP}\nThe correct answer for the question "
        f'"How would Alex feel?" is "{choice}".')) for choice in "ABC"}
    assert len(keys) == 1, keys
    other = rule(*_split_wrapper(
        f"Kendall went camping.\n{WRAP}\nThe correct answer for the "
        'question "How would Alex feel?" is "A".'))
    assert other not in keys
    print("  test_social_iqa_choices_share_a_group")


def test_pubmedqa_groups_on_the_abstract():
    """Wrapped or bare, the abstract is the group — so the yes/no expansion
    in TF_prompt_wrappers_plan.md would land in one fold if extracted."""
    rule = GROUP_RULES["pubmedqa_labeled"]
    abstract = "Mitochondria were examined in lace plant leaves."
    q = "Do mitochondria play a role?"
    wrapped = rule(*_split_wrapper(f"{abstract}\n\n{WRAP}\n\n{q}"))
    bare = rule(*_split_wrapper(f"{abstract}\n\n{q}"))
    assert wrapped == bare == abstract, (wrapped, bare)
    assert rule(*_split_wrapper("A different abstract.\n\n" + q)) != wrapped
    print("  test_pubmedqa_groups_on_the_abstract")


def test_geometry_truth_pairs_negations_and_directions():
    """A fact and its negation, and a comparison and its reverse, are one
    group; unrelated facts are not."""
    rule = GROUP_RULES["geometry_truth"]
    def key(s):
        return rule(*_split_wrapper(f"{WRAP} {s}"))

    assert key("The city of Krasnodar is in Russia.") == \
        key("The city of Krasnodar is not in Russia.")
    assert key("The Spanish word 'uno' means 'one'.") == \
        key("The Spanish word 'uno' does not mean 'one'.")
    assert key("Fifty-one is larger than fifty-two.") == \
        key("Fifty-one is smaller than fifty-two.")
    # same subject, swapped object -> same group (the true and false
    # version of one underlying fact)
    assert key("The city of Krasnodar is in Russia.") == \
        key("The city of Krasnodar is in South Africa.")
    assert key("The city of Krasnodar is in Russia.") != \
        key("The city of Toulouse is in France.")
    print("  test_geometry_truth_pairs_negations_and_directions")


def test_group_ids_identity_for_sources_without_a_rule():
    """A source with no rule is one group per row, so its split is exactly
    what an ungrouped run would do."""
    sid = np.array([f"xstest:{i}" for i in range(10)])
    assert (group_ids("xstest", sid, 22, "gemma_tf_a") == sid).all()
    print("  test_group_ids_identity_for_sources_without_a_rule")


def test_stored_ids_win_over_the_text_rule():
    """A source in GROUP_ID_KEYS whose file carries the id uses it, with no
    prompt read at all — the layer/collection arguments are never touched,
    which is why a bogus collection name here must still work."""
    loaded = {"source_row_id": np.array([7, 7, 8, 8])}
    sid = np.array([f"pubmedqa_labeled:{i}" for i in range(4)])
    stored = _stored_group_key("pubmedqa_labeled", loaded)
    got = group_ids("pubmedqa_labeled", sid, 999, "nonexistent", stored=stored)
    assert list(got) == ["pubmedqa_labeled:7", "pubmedqa_labeled:7",
                         "pubmedqa_labeled:8", "pubmedqa_labeled:8"], got
    print("  test_stored_ids_win_over_the_text_rule")


def test_stored_ids_are_namespaced_per_source():
    """source_row_id is a row number into the upstream dataset, so two
    sources' id 0 are unrelated and must not merge once pooled."""
    loaded = {"source_row_id": np.array([0, 1])}
    a = _stored_group_key("pubmedqa_artificial", loaded)
    b = _stored_group_key("pubmedqa_labeled", loaded)
    assert not set(a) & set(b), (a, b)
    print("  test_stored_ids_are_namespaced_per_source")


def test_wildjailbreak_pairs_its_two_halves():
    """source_prompt_id is the vanilla<->adversarial pairing key. The two
    halves become separate units, so this group never binds inside a unit;
    in a pooled corpus containing both halves it binds across them (see
    test_pool_blocks_groups_pair_across_units_of_one_source), and on
    transfer evaluations it is what makes the contamination measurable."""
    assert GROUP_ID_KEYS["wildjailbreak"] == "source_prompt_id"
    loaded = {"source_prompt_id": np.array(["abc123", "abc123", "def456"])}
    g = _stored_group_key("wildjailbreak", loaded)
    assert g[0] == g[1] != g[2]
    print("  test_wildjailbreak_pairs_its_two_halves")


def test_sources_with_no_stored_id_fall_back():
    """A file that predates the stored ids returns None, so group_ids goes
    to the text rule rather than silently to one group per row."""
    assert _stored_group_key("pubmedqa_labeled", {"labels": np.zeros(3)}) is None
    assert _stored_group_key("xstest", {"source_row_id": np.arange(3)}) is None
    print("  test_sources_with_no_stored_id_fall_back")


def test_a_rule_that_collapses_everything_is_an_error():
    """_group_social_iqa keys on the T/F wrapper's context block, so on a
    bare tree it would put every row in one fold. That must raise, not
    quietly destroy the CV."""
    import classifiers.data as D
    sid = np.array([f"social_iqa:{i}" for i in range(5)])
    prompts = {s: "A bare prompt with no wrapper line." for s in sid}
    saved = D._prompts_cached
    D._prompts_cached = lambda *a, **k: prompts
    try:
        group_ids("social_iqa", sid, 22, "gemma_original")
    except ValueError as e:
        assert "one group" in str(e), e
    else:
        raise AssertionError("expected a ValueError on a collapsed grouping")
    finally:
        D._prompts_cached = saved
    print("  test_a_rule_that_collapses_everything_is_an_error")


def test_every_collection_declares_a_group_source_it_has():
    """GROUP_ID_KEYS / GROUP_RULES are keyed on HF SOURCE names, not unit
    names — a typo would silently mean no grouping at all."""
    for name, c in COLLECTIONS.items():
        known = set(c.hf_files)
        for keyed in (GROUP_ID_KEYS, GROUP_RULES):
            unknown = set(keyed) - known
            assert not unknown, f"{name}: {sorted(unknown)} are not sources"
    print("  test_every_collection_declares_a_group_source_it_has")


def _leakage(y, groups, seed=0):
    """Test rows whose group is also in the training part, outer and inner."""
    X = np.zeros((len(y), 1))
    leaked = total = 0
    for otr, ote, inner in base.make_nested_folds(X, y, seed=seed,
                                                  groups=groups):
        seen = set(groups[otr].tolist())
        leaked += sum(g in seen for g in groups[ote].tolist())
        total += len(ote)
        for itr, ite in inner:
            seen_in = set(groups[otr][itr].tolist())
            leaked += sum(g in seen_in for g in groups[otr][ite].tolist())
            total += len(ite)
    return leaked, total


def test_nested_folds_never_split_a_group():
    """The invariant the flag exists for, at both levels of the nested CV."""
    rng = np.random.default_rng(0)
    n_groups = 400
    sizes = rng.integers(1, 4, size=n_groups)         # 1-3 rows per item
    groups = np.concatenate([np.full(s, f"g{i}") for i, s in enumerate(sizes)])
    y = rng.integers(0, 2, size=len(groups))
    leaked, total = _leakage(y, groups)
    assert leaked == 0, f"{leaked}/{total} test rows had a sibling in train"
    assert total > 0
    print(f"  test_nested_folds_never_split_a_group (0/{total})")


def test_grouped_folds_still_cover_every_sample_once():
    """Group-aware or not, the outer folds are still a partition."""
    rng = np.random.default_rng(1)
    groups = np.repeat([f"g{i}" for i in range(300)], 2)
    y = rng.integers(0, 2, size=len(groups))
    X = np.zeros((len(y), 1))
    seen = np.zeros(len(y), dtype=int)
    for _, ote, _ in base.make_nested_folds(X, y, seed=0, groups=groups):
        seen[ote] += 1
    assert (seen == 1).all(), Counter(seen.tolist())
    print("  test_grouped_folds_still_cover_every_sample_once")


def test_ungrouped_is_bit_identical_to_the_old_splitter():
    """--no-group-folds must reproduce the pre-grouping results exactly."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=500)
    X = np.zeros((len(y), 1))
    new = base.make_nested_folds(X, y, seed=0, groups=None)
    old = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1,
                                       random_state=0).split(X, y))
    assert len(new) == len(old)
    for (ntr, nte, ninner), (otr, ote) in zip(new, old):
        assert np.array_equal(ntr, otr) and np.array_equal(nte, ote)
        old_inner = list(StratifiedKFold(5, shuffle=True, random_state=0)
                         .split(X[otr], y[otr]))
        assert len(ninner) == len(old_inner)
        for (a, b), (c, d) in zip(ninner, old_inner):
            assert np.array_equal(a, c) and np.array_equal(b, d)
    print("  test_ungrouped_is_bit_identical_to_the_old_splitter")


def test_stratify_on_still_applies_under_grouping():
    """The pooled runs stratify on (dataset, label); grouping must not
    silently drop that — every fold should still see both datasets."""
    rng = np.random.default_rng(3)
    groups = np.repeat([f"g{i}" for i in range(400)], 2)
    dataset = np.where(np.arange(len(groups)) < len(groups) // 2, "a", "b")
    y = rng.integers(0, 2, size=len(groups))
    key = np.char.add(dataset, y.astype(str))
    X = np.zeros((len(y), 1))
    for _, ote, _ in base.make_nested_folds(X, y, seed=0, stratify_on=key,
                                            groups=groups):
        assert set(np.unique(dataset[ote])) == {"a", "b"}
    print("  test_stratify_on_still_applies_under_grouping")


def _load_features():
    """features.py by path, like `base` above — the package __init__ pulls
    cuML, absent here. Its relative imports need a package shim whose
    __path__ lets `.base` resolve normally (base.py is CPU-only)."""
    import types
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts")
    pkg = types.ModuleType("classifiers.scripts")
    pkg.__path__ = [scripts_dir]
    sys.modules.setdefault("classifiers.scripts", pkg)
    spec = importlib.util.spec_from_file_location(
        "classifiers.scripts.features",
        os.path.join(scripts_dir, "features.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pool_blocks_groups_pair_across_units_of_one_source():
    """Regression, 2026-08-10: pool_blocks used to namespace CV groups per
    UNIT, so wildjailbreak's vanilla/adversarial halves — separate units
    sharing one source_prompt_id per rewrite pair — straddled the fold
    boundary whenever both were pool members (the expanded-pool runs: ~80%
    of adversarial CV-test rows had their same-label vanilla sibling in
    that fold's training set). The namespace is now the unit's SOURCE, so
    the pair shares one fold group; units of distinct sources still
    cannot collide, even on bare row-number ids."""
    features = _load_features()
    mk = lambda n, ids, group: {                      # noqa: E731
        "blocks": {"raw": np.arange(2 * n, dtype=float).reshape(n, 2)},
        "y": np.tile([0, 1], n // 2 + 1)[:n],
        "sample_id": np.asarray(ids), "topic": None,
        "group": np.asarray(group),
    }
    pair_ids = [f"wildjailbreak:{i}" for i in range(8)]
    units = {
        "wildjailbreak_vanilla": mk(8, [f"v{i}" for i in range(8)], pair_ids),
        "wildjailbreak_adversarial": mk(8, [f"a{i}" for i in range(8)],
                                        pair_ids),
        # two single-unit sources whose fallback ids are bare row numbers
        "xstest": mk(4, list("0123"), list("0123")),
        "claude_anatomy": mk(4, list("0123"), list("0123")),
    }
    _, dataset, arrays = features.pool_blocks(units, list(units))
    g = arrays["group"]
    van, adv = g[dataset == "wildjailbreak_vanilla"], \
        g[dataset == "wildjailbreak_adversarial"]
    assert (van == adv).all(), "rewrite pairs must share one fold group"
    xs = set(g[dataset == "xstest"])
    an = set(g[dataset == "claude_anatomy"])
    assert not xs & an, "bare row ids must not collide across sources"
    # and the splitter must keep every pair whole
    y = arrays["y"]
    key = np.char.add(dataset, y.astype(str))
    for tr, te in base.make_folds(np.zeros((len(y), 1)), y, seed=0,
                                  n_splits=2, stratify_on=key, groups=g):
        assert not set(g[tr]) & set(g[te])
    print("  test_pool_blocks_groups_pair_across_units_of_one_source")


if __name__ == "__main__":
    print("fold groups:")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all passed")
