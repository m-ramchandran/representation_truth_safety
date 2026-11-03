"""Post-hoc analyses of the classifier experiments' saved outputs.

Everything here reads only what the runners left under
classifiers/results/ — the metrics csvs and the prediction npz files —
never the fitted models. That makes this package runnable anywhere,
including the Mac: do NOT import from classifiers.scripts, which pulls
in cuML at load time and only imports on a GPU node.

`load_results` is the shared loader (metrics tables as DataFrames, npz
files as dicts with the params parsed back). Each analysis is its own
module on top of it; see README.md for the npz layout and the
conventions (margins, Platt calibration, which space each method's
coefficient vector lives in).
"""
