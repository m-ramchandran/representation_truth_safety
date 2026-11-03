# Steering experiments

This package constructs probe-derived directions, intervenes on last-token
activations, and summarizes changes in True/False output odds. Run commands from
the repository root after classifier fits are available.

## Main pipeline

```bash
# 1. Build normalized directions from fitted probes and class means.
python -m steering.directions --help

# 2. Run activation-addition interventions.
python -m steering.steer --help

# 3. Aggregate intervention curves and matched controls.
python -m steering.analyze --help
```

The checked-in Slurm launchers under `steering/sbatch/` record the model, layer,
target, and array settings used for the reported runs. They write large raw
outputs to `steering/out/`, which is intentionally excluded from git.

Figure 6 uses probe coefficients projected onto the leading principal
components of their training activations. `projected_directions.py` constructs
those directions; `pc_alignment.py` and `pc_capture.py` provide its PCA and fit
utilities. The projected launchers under `steering/sbatch/` record the reported
run configuration.

Compact summaries are stored under `paper_final_figures/data/`.
`distill_projected_steering.py` rebuilds them from raw runs, and
`fig_steering_projected.py` renders the submitted figure.
