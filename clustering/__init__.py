"""Clustering approaches and sensitivity analyses on activation matrices.

Fitting writes a uniform artifact and every metric and plot reads only
those artifacts, so any two runs — different method, different layer,
different number of components — compare through one code path.

    run_clustering    fit a config's whole grid, one run per cell
    leave_one_out     the same grid with one dataset held out at a time
    run_embeddings    t-SNE / UMAP views of a finished run, with NESS
                      stability across a seed ensemble
    make_figures      every figure, from saved files only

Configs live in configs/, outputs under results/<scope>/. Labels are
used for evaluation only and never for fitting.
"""
