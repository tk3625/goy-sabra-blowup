# Computer-assisted verification for Section 3

This directory contains the code for the computer-assisted verification in Section 3.
The formulas and verification steps are described in the paper.

## Running the code

Run the following commands from this directory:

```sh
python -m pip install -r requirements.txt
python verify_contraction.py
```

The `__main__` block reads `approximation_file` from `../section_2/data.npz`.
It resolves `approximation_file` relative to `data.npz` and reads `I`, `K`, `delta`, and `a` from the Appendix A approximation.
The radius `r` must have been validated in Section 2 for the same approximation.

`verify_contraction.py` verifies the inequalities using interval arithmetic and writes the intermediate enclosures, final contraction bounds, verification results, and times to `results_contraction.txt`.
The report ends with `All inequalities verified.` on success.

To edit the parameters or working precision, change the values in the `__main__` block of `verify_contraction.py`.
