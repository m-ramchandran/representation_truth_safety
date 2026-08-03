"""Binary classifiers on language-model activations.

Four pieces, split by what they need to run:

    data.py       activation loading and the dataset registry — plain
                  numpy, imports anywhere
    methods/      thirteen classifiers behind one fit contract. l1reg /
                  l2reg / svm_linear / svm_rbf / knn go through cuML,
                  mlp / random_slopes / moe through torch, xgboost
                  through the xgboost package, and diff / lda / qda on
                  CPU. Needs a GPU: it imports cuML at load.
    experiments/  the three experimental designs — per-dataset,
                  leave-one-out, pooled — run over those methods, with
                  outputs under results/<collection>/
    analysis/     everything downstream of results/, plain numpy again,
                  so it runs on a laptop

data.py, fig_style.py and auc_figures.py sit at the package root because
both sides import them, which is what keeps the label conventions and
the figure styling from drifting apart.
"""
