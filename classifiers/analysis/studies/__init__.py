"""Focused studies built on top of the per-module analyses.

Where the modules in ``classifiers.analysis`` each answer one question about
one experiment, these cut across experiments and collections to answer the
questions the paper argues from: which fits exist at all, how methods and
feature sets compare once the comparison is paired, what predicts transfer,
and what the decision boundaries actually look like.

Each module is runnable on its own::

    python -m classifiers.analysis.studies.linear_hypothesis

Most read only the metrics csvs under ``classifiers/results/`` and print to
stdout. ``synthesis``, ``covariance_transfer`` and ``label_free_pcs`` also
load cached activations and write csvs to ``tables/`` and figures to
``figures/`` beside this file; nothing is written back into the results tree.
``boundary_refit`` and the ``two_pc_*`` fits need a GPU and leave reusable
npz artifacts that the plotting modules read.
"""
