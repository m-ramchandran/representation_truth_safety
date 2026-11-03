# Final figures

This directory contains only the material needed to inspect or regenerate the
paper figures:

- `data/`: compact plot inputs.
- `scripts/`: analysis and plotting code.
- `figures/`: PNGs and captions.
- `figures_pdf/`: publication-ready PDFs.

## Plot the submitted figures

These are the only plotting entry points. Together they generate the six main
and five supplementary figures used in the submission. Run from the repository
root:

```bash
python paper_final_figures/scripts/q0_activation_geometry.py
python paper_final_figures/scripts/q1_figures.py
python paper_final_figures/scripts/q16_figures.py
python paper_final_figures/scripts/q12_figures.py
python paper_final_figures/scripts/q17_heldout_targets.py
python paper_final_figures/scripts/fig_steering_projected.py
python paper_final_figures/scripts/q6_layer_figure.py
python paper_final_figures/scripts/q5_figures.py
python paper_final_figures/scripts/q7_figures.py
```

The scripts write PNGs to `figures/` and PDFs to `figures_pdf/`. The remaining
files under `scripts/` derive the compact plot inputs and do not write figures.
Data files are named for the figure they support;
`shared_classifier_results.csv` is shared by Figure 5 and Supplementary Figure
2.
