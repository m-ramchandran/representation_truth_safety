"""The three experimental designs, plus the feature sets they are run over.

Each runner takes one activation collection (``--data``), one raw variant
(``--raw last|mean``), a set of layers, a set of feature sets and a set of
methods, and writes a metrics csv and a per-sample prediction npz per
(feature set, layer) cell under ``results/<collection>/<experiment>/``:

    run_per_dataset  one fit per dataset, nested CV in domain, then that
                     fit applied to every other dataset (transfer)
    run_leave_out    one fit per held-out unit on the pool of the rest,
                     scored on what was held out (``--holdout dataset``
                     or ``--holdout task``)
    run_merged       one nested-CV fit on the pooled corpus, plus one
                     tuned fit on all of it for the eval-only units

Runs resume: a (tag, layer) cell already present in the metrics csv is
skipped, so a job that stops partway can simply be resubmitted.

The fitting itself lives in ``classifiers.methods`` and needs a GPU;
``smoke_test`` exercises every method and all three runners on small
synthetic data before committing cluster time to a full grid.
"""
