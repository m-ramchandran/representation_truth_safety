"""Post-hoc analyses of the classifier experiments' saved outputs.

Everything here reads only what the runners left under
classifiers/results/ — the metrics csvs and the prediction npz files —
never the fitted models. That makes this package runnable on a laptop,
and it stays that way only if nothing here imports classifiers.methods,
which pulls in cuML at load time and needs a GPU.

`load_results` is the shared loader (metrics tables as DataFrames, npz
files as dicts with the params parsed back). Each analysis is its own
module on top of it; `studies/` holds the cross-experiment ones. The npz
layout and the conventions — margins, Platt calibration, which space
each method's coefficient vector lives in — are in docs/results.md.
"""
