"""Classifier module. The experiment code in scripts/ is GPU-homed:
l1reg / l2reg / svm_linear / svm_rbf / knn fit via cuML and mlp /
random_slopes / moe via torch, so scripts/ needs a GPU node (it will not
import on a CPU-only machine; run as python -m
classifiers.scripts.run_per_dataset / run_leave_out / run_merged, outputs
in results/). data.py (activation loading) and analysis/ (post-hoc work
on results/) are plain numpy and run anywhere."""
