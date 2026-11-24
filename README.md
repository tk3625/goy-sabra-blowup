# Finite-time blowup in the inviscid GOY and Sabra models

Verification code and numerical data accompanying the paper 
**Finite-time blowup in the inviscid GOY and Sabra models** 
by **Taehun Kim**.

## Contents

- `codes/appendix_a/`: numerical construction of the approximate travelling wave.
- `codes/section_2/`: interval-arithmetic verification of the existence of a travelling wave.
- `codes/section_3/`: interval-arithmetic verification of the contraction estimates for the GOY and Sabra models.

Each directory contains a README with further instructions.

## Verify the supplied data

With Python and pip available, run the following commands from the repository root:

```sh
python -m pip install -r codes/section_2/requirements.txt
python codes/section_2/verify_orbit.py
python codes/section_3/verify_contraction.py
```

Both verification scripts should end with:

```text
All inequalities verified.
```

The scripts use the `.npz` files in the folder. It is not necessary to regenerate the numerical approximation before running the verifications.

The verification scripts overwrite the corresponding `results_*.txt` files in the local working copy. Use a separate copy if you want to retain the original recorded logs for comparison.

## Numerical data generation

The data-generation procedures are described in the README files under `codes/appendix_a/` and `codes/section_2/`.

These procedures use ordinary floating-point arithmetic.
