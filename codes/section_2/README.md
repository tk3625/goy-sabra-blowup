# Computer-assisted verification for Section 2

This directory contains the code and data for the computer-assisted verification in Section 2. 
The formulas and verification steps are described in the paper.

## Running the code

Run the following commands from this directory to verify the supplied `data.npz` and approximation:

```sh
python -m pip install -r requirements.txt
python verify_orbit.py
```

To regenerate the data and verify them, run:

```sh
python prepare_data.py
python verify_orbit.py
```

`prepare_data.py` replaces `data.npz`. Regeneration is required after changing the approximation or the preparation parameters.

The function `prepare_data` computes and returns the unit tangent `q` and the matrices `J` and `A_fin` using ordinary floating-point arithmetic.
The matrix basis is `(1,0)`, followed by `(0,e_nk)`, ordered first by shell index and then by Taylor degree.

The `__main__` block sets the parameters, reads the Appendix A approximation, and saves the three returned quantities and `rho`, `L`, `U`, `S`, and `ell0` in `data.npz`.
`data.npz` also stores `approximation_file = "../appendix_a/approximation.npz"`.
The function `read_data` resolves this path relative to `data.npz` and reads `I`, `K`, `delta`, and `a` from that file; they are not duplicated in `data.npz`.
If the approximation changes, run `prepare_data.py` again before verification.

`verify_orbit.py` verifies the inequalities using interval arithmetic and writes the upper bounds, prescribed constants, and times to `results_orbit.txt`. 
The report ends with `All inequalities verified.` on success.

To edit the parameters, change the values in the `__main__` block of `prepare_data.py`.
To edit the prescribed bounds, radius, or working precision, change the values in the `__main__` block of `verify_orbit.py`.

## Plotting the travelling wave

From this directory, run:

```sh
python -m pip install -r requirements.txt
python plot_orbit.py
```

The script reads the coefficients overline{a}_{n}^{(k)} from `../appendix_a/approximation.npz` and saves `bilateral_profile.png` in this directory.