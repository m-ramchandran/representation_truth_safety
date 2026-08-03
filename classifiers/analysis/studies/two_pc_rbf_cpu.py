"""Compatibility entry point for the two-PC RBF fitter's CPU backend."""

import argparse

from .boundary_common import SCOPES
from .two_pc_rbf import fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    fit(args.scope, backend="cpu", n_jobs=args.jobs)
