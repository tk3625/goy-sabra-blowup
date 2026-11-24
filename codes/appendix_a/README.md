# Numerical data for Appendix A

This directory contains the code and numerical data for Appendix A.
The construction is described in the appendix.

## Running the code

Run the following commands from this directory:

```sh
python -m pip install -r requirements.txt
python approximation.py
```

The script saves `approximation.npz` in the same directory as `approximation.py`, replacing any existing file with that name.
To change the parameters, at the end of `approximation.py`, edit the call to `generate_approximation`.
The values used there are:

| Parameter | Value | Meaning |
|---|---|---|
| `h0` | `1e-4` | Base timestep; the adaptive timestep used in the integration is `h0 / max(1, max(abs(z)))`. |
| `return_tol` | `1e-9` | Relative tolerance for comparing successive profiles in the ell^{2} norm. |
| `boundary_tol` | `1e-9` | Relative boundary threshold used to decide whether to enlarge the interval. |
| `taylor_tol` | `1e-9` | Relative coefficient threshold used to determine the Taylor truncation order. |
| `max_steps` | `2000000` | Maximum number of RK4 steps per integration attempt. |

The computations use NumPy double-precision arithmetic. This is a numerical search, we do not claim any rigorous error bounds here.

After regenerating `approximation.npz`, rerun `prepare_data.py` and `verify_orbit.py` (and optionally `plot_orbit.py` to regenerate the figure used in the LaTeX code) in `../section_2`, then run `verify_contraction.py` in `../section_3`.

## Output

The file `approximation.npz` contains:

- `I`: the two integer endpoints of the final shell interval.
- `K`: the Taylor truncation order.
- `delta`: the approximate relative period.
- `a`: the Taylor coefficients, with shape `(I[1]-I[0]+1, K+1)`.

The entry `a[i, k]` represents a_{I[0]+i}^{(k)} in the manuscript.
The shell indices have been shifted by an integer so that the weighted center of this initial profile lies in `[0, 1)`.

The data can be loaded as follows:

```python
import numpy as np

with np.load("approximation.npz", allow_pickle=False) as data:
    I = tuple(data["I"])
    K = int(data["K"])
    delta = float(data["delta"])
    a = data["a"]
```