# Interval arithmetic verification for Section 2.

from decimal import Decimal, ROUND_CEILING, localcontext
from pathlib import Path
from time import perf_counter
import platform
import numpy as np
import flint
from flint import arb, arb_mat, ctx

if not __debug__:
    raise RuntimeError("Verification requires assertions. Run Python without -O or -OO and unset PYTHONOPTIMIZE.")

def exact_float(value):
    """
    Return an interval enclosure of the input floating-point value.
    """
    numerator, denominator = float(value).as_integer_ratio()

    return arb(numerator) / arb(denominator)

def read_data(path):
    """
    input: path of data.npz
    Read data.npz and return exact values or interval enclosures.
    """

    # codes/section_2
    directory = Path(path).resolve().parent

    # The output will be 'data', containing all data.
    data = {}

    with np.load(path) as saved:
        # path to codes/appendix_a/approximation.npz
        approximation_path = directory / str(saved["approximation_file"].item())

        data["L"] = int(saved["L"])
        data["U"] = int(saved["U"])
        data["ell0"] = int(saved["ell0"])

        # Represent \mathcal{S} by a range of integers, including both endpoints.
        data["S"] = range(int(saved["S"][0]), int(saved["S"][1]) + 1)

        # \rho will be represented by its interval enclosure
        data["rho"] = arb(int(saved["rho"][0])) / arb(int(saved["rho"][1]))
        
        data["q"] = [exact_float(value) for value in saved["q"]]
        data["J"] = arb_mat([[exact_float(value) for value in row] for row in saved["J"]])
        data["A_fin"] = arb_mat([[exact_float(value) for value in row] for row in saved["A_fin"]])

    with np.load(approximation_path) as approximation:
        data["I"] = (int(approximation["I"][0]), int(approximation["I"][1]))
        data["K"] = int(approximation["K"])
        data["delta"] = exact_float(approximation["delta"])
        data["a"] = [[exact_float(value) for value in row] for row in approximation["a"]]

    # \dim(PX)
    data["dimension"] = 1 + (data["U"] - data["L"] + 1) * (data["K"] + 1)
    
    return data

def coefficient(data, n, k):
    """
    Return a_n^(k), extended by zero outside its support.
    """
    I_left, I_right = data["I"]
    if I_left <= n <= I_right and 0 <= k <= data["K"]:
        return data["a"][n - I_left][k]
    return arb(0)

def coordinate_index(data, n, k):
    """
    The period coordinate has index 0, followed by $e_{n, k}$, 
    ordered first by n and then by k.
    (L, 0), (L, 1), ..., (L, K), (L+1, 0), ..., (L+1, K), ..., (U, K)
    Return the index of $(0, e_{n, k})$ in the basis of $PX$.
    """
    return 1 + (n - data["L"]) * (data["K"] + 1) + k

def matrix_norm(matrix):
    """
    Enclose the induced $ell^{1}$-norm of a finite matrix.
    """
    result = arb(0)
    for j in range(matrix.ncols()):
        column_sum = arb(0)
        for i in range(matrix.nrows()):
            column_sum += abs(matrix[i, j])
        result = result.max(column_sum)
    return result

def check_parameters(data):
    """
    Check the parameter conditions required in the finite estimates.
    """
    assert data["rho"] > 1, "Could not verify rho > 1."
    assert 0 < data["delta"] < 1, "Could not verify 0 < overline{delta} < 1."
    assert data["ell0"] > data["K"], "ell_0 > K does not hold"
    assert data["K"] + 1 > data["rho"] * data["delta"], "Could not verify K+1 > rho delta."
    assert data["S"][0] <= min(data["L"], data["I"][0] - 2), "mathcal{S} is not big enough."
    assert data["S"][-1] >= max(data["U"], data["I"][1] + 2), "mathcal{S} is not big enough."

def check_z1_inv(data, Z1_inv):
    """
    Compute the matrix norm of I - A_{fin} J.
    Return its upper endpoint, and check if this does not exceed the prescribed constant Z1_inv.
    """

    identity = arb_mat(data["dimension"], data["dimension"])
    for i in range(data["dimension"]):
        identity[i, i] = 1
    bound = matrix_norm(identity - data["A_fin"] * data["J"]).upper()
    assert bound <= Z1_inv, "The bound for Z1_inv exceeds the prescribed constant."
    return bound

def check_z0(data, Z0):
    """
    Evaluate ||A|| ||L_0^(-1)|| ||F(delta,a)|| and return an upper endpoint.
    Check if this does not exceed the prescribed constant Z0.
    """

    norm_A = arb(1).max(matrix_norm(data["A_fin"]))
    norm_L0_inverse = ((data["rho"] + 1) * data["delta"]).exp() / (1 - (-data["delta"]).exp()) + (data["rho"] * data["delta"]).exp()

    K = data["K"] 
    # F(delta, a) is supported in [n_min, n_max] \times [0, 2K+1].
    n_min = data["I"][0] - 2
    n_max = data["I"][1] + 2

    # F[n - n_min, k] stores F_{n}^{(k)}.
    F = arb_mat(n_max - n_min + 1, 2 * K + 2)

    # Compute F_{n}^{(0)}.
    for n in range(n_min, n_max + 1):
        F[n - n_min, 0] = -coefficient(data, n, 0)
        for j in range(K + 1):
            F[n - n_min, 0] += coefficient(data, n + 1, j)

    # Compute F_{n}^{(k)} for k \ge 1.
    for n in range(n_min, n_max + 1):
        for k in range(1, 2 * K + 2):
            F[n - n_min, k] = coefficient(data, n, k) + data["delta"] / k * coefficient(data, n, k-1)
            
            nonlinear = arb(0)
            for j in range(k):
                nonlinear += (
                    2 * coefficient(data, n - 1, j)
                      * coefficient(data, n - 2, k-1-j)
                    + coefficient(data, n + 1, j)
                      * coefficient(data, n - 1, k-1-j) / 2
                    - coefficient(data, n + 2, j)
                      * coefficient(data, n + 1, k-1-j) / 4
                )
            F[n - n_min, k] -= (data["delta"] / k * nonlinear)

    # Compute ||F(\delta, a)||_{\ell_{\rho}^{1}}.
    residual_norm = arb(0)
    for n in range(n_min, n_max + 1):
        for k in range(2 * K + 2):
            residual_norm += data["rho"] ** k * abs(F[n - n_min, k])

    bound = (norm_A * norm_L0_inverse * residual_norm).upper()
    assert bound <= Z0, "The bound for Z0 exceeds the prescribed constant."
    return bound

def check_z2_z3(data, Z2, Z3):
    """
    Evaluate the two higher-derivative bounds and return their upper endpoints.
    Check if these do not exceed the prescribed constants Z2, Z3.
    """

    norm_A = arb(1).max(matrix_norm(data["A_fin"]))
    norm_L0_inverse = ((data["rho"] + 1) * data["delta"]).exp() / (1 - (-data["delta"]).exp()) + (data["rho"] * data["delta"]).exp()
    C_Q = arb(11) / 4
    norm_a = arb(0)
    for n in range(data["I"][0], data["I"][1] + 1):
        for k in range(data["K"] + 1):
            norm_a += data["rho"] ** k * abs(coefficient(data, n, k))

    bound2 = (norm_A * norm_L0_inverse * (data["rho"] * (1 + 2 * C_Q * norm_a)).max(2 * data["delta"] * data["rho"] * C_Q)).upper()
    bound3 = (norm_A * norm_L0_inverse * data["rho"] * C_Q).upper()
    assert bound2 <= Z2, "The bound for Z2 exceeds the prescribed constant."
    assert bound3 <= Z3, "The bound for Z3 exceeds the prescribed constant."
    return bound2, bound3

def finite_inverse(data, y, M, n_minus, n_plus):
    """
    Return interval enclosures of P(0, L_{0}^{-1}y) and T(y).
    y is supported in [n_minus, n_plus] times [0, M],
    and has shape (n_plus - n_minus + 1, M + 1).
    y[n-n_minus, k] stores y_{n}^{(k)}, using interval enclosures.
    Parameters must satisfy M >= K, n_minus <= L, n_plus >= U, and M+1 > rho delta.
    """

    rho = data["rho"]
    delta = data["delta"]
    K = data["K"]
    L = data["L"]
    U = data["U"]
    dimension = data["dimension"]

    # This condition is used for the alternating-series enclosures below.
    assert 0 < delta < 1, "Could not verify 0 < delta < 1."
    
    assert M + 1 > rho * delta, "Could not verify M+1 > rho delta."

    # h[n - n_minus, k] stores (L_{0}^{-1}y)_{n}^{(k)}.
    h = arb_mat(n_plus - n_minus + 1, M + 1)
    # We also need h_{n_minus-1}^{(0)}
    h_left = arb(0)

    # Compute h_{n}^{(0)} = e^{-\overline{\delta}} h_{n+1}^{(0)} + (summand with j=0)
    for n in range(n_minus, n_plus + 1):
        h[n - n_minus, 0] = -y[n - n_minus, 0]

    for ell in range(1, M + 1):
        lower_sum = arb(1)
        term = arb(1)
        for j in range(1, 40):
            term *= (-delta) / (ell + j)
            lower_sum += term
        term *= (-delta) / (ell + 40)
        upper_sum = lower_sum + term
        sum_enclosure = lower_sum.lower().union(upper_sum.upper())

        for n in range(n_minus, n_plus):
            h[n - n_minus, 0] += sum_enclosure * y[n + 1 - n_minus, ell]
        h_left += sum_enclosure * y[0, ell]

    for n in range(n_plus - 1, n_minus - 1, -1):
        h[n - n_minus, 0] += (-delta).exp() * h[n + 1 - n_minus, 0]
    h_left += (-delta).exp() * h[0, 0]

    # Compute h_{n}^{(k)} = -\frac{\delta}{k} h_{n}^{(k-1)} + y_{n}^{(k)}.
    for k in range(1, M + 1):
        for n in range(n_minus, n_plus + 1):
            h[n - n_minus, k] = y[n - n_minus, k] - delta / k * h[n - n_minus, k-1]

    # Store P(0,h) in the normalized basis of PX.
    v = arb_mat(dimension, 1)
    for n in range(L, U + 1):
        for k in range(K + 1):
            row = coordinate_index(data, n, k)
            v[row, 0] = data["rho"] ** k * h[n - n_minus, k]

    # First term in T(y)
    T_finite = arb(0)
    for n in range(n_minus, n_plus + 1):
        for k in range(M + 1):
            if n < L or n > U or k > K:
                T_finite += data["rho"] ** k * abs(h[n - n_minus, k])

    # Second term in T(y)
    T_Taylor = arb(0)
    for n in range(n_minus, n_plus + 1):
        T_Taylor += data["rho"] ** M * abs(h[n - n_minus, M])
    T_Taylor *= (rho * delta / (M + 1 - rho * delta))

    # Third term in T(y)
    T_spatial = (rho * delta).exp() / (1 - (-delta).exp()) * abs(h_left)

    return v, T_finite + T_Taylor + T_spatial

def check_z1_app(data, Z1_app):
    """
    Evaluate the period, coefficient, and high-Taylor bounds, 
    and return the upper endpoint of their maximum.
    Check if this does not exceed the prescribed constant Z1_app.
    """

    rho = data["rho"]
    delta = data["delta"]
    K = data["K"]
    L = data["L"]
    U = data["U"]
    dimension = data["dimension"]
    J = data["J"]
    A_fin = data["A_fin"]
    ell0 = data["ell0"]
    
    # Compute y_delta.
    M_delta = 2 * K + 1
    n_minus_delta = min(0, L, data["I"][0] - 2)
    n_plus_delta = max(U, data["I"][1] + 2)
    y_delta = arb_mat(n_plus_delta - n_minus_delta + 1, M_delta + 1)
    for k in range(1, M_delta + 1):
        for n in range(data["I"][0] - 2, data["I"][1] + 3):
            y_delta[n - n_minus_delta, k] = coefficient(data, n, k - 1) / k
            nonlinear = arb(0)
            for j in range(k):
                nonlinear += (
                    2 * coefficient(data, n - 1, j)
                      * coefficient(data, n - 2, k - 1 - j)
                    + coefficient(data, n + 1, j)
                      * coefficient(data, n - 1, k - 1 - j) / 2
                    - coefficient(data, n + 2, j)
                      * coefficient(data, n + 1, k - 1 - j) / 4
                )
            y_delta[n - n_minus_delta, k] -= (nonlinear / k)

    # P(0, L_{0}^{-1}y_{delta}) and T(y_{delta})
    v_delta, T_delta = finite_inverse(data, y_delta, M_delta, n_minus_delta, n_plus_delta)
    
    # Period check: ||A_fin (P(0,L_0^(-1)y_delta) - J(1,0))|| + T(y_delta).
    for i in range(dimension):
        v_delta[i, 0] -= J[i, 0]
    period_bound = matrix_norm(A_fin * v_delta) + T_delta

    # Coefficient checks

    coefficient_bound = arb(0)
    for m in data["S"]:
        for ell in range(ell0):
            n_minus_m_ell = min(0, L, m - 2)
            n_plus_m_ell = max(U, m + 2)
            M_m_ell = ell + K + 1

            # Compute y_{m, ell}.
            y_m_ell = arb_mat(n_plus_m_ell - n_minus_m_ell + 1, M_m_ell + 1)
            for k in range(ell + 1, ell + K + 2):
                y_m_ell[m + 2 - n_minus_m_ell, k] = -2 * delta / k / data["rho"] ** ell * coefficient(data, m + 1, k - 1 - ell)
                y_m_ell[m + 1 - n_minus_m_ell, k] = -2 * delta / k / data["rho"] ** ell * (
                    coefficient(data, m - 1, k - 1 - ell)
                    + coefficient(data, m + 2, k - 1 - ell) / 4
                )
                y_m_ell[m - 1 - n_minus_m_ell, k] = -2 * delta / k / data["rho"] ** ell * (
                    coefficient(data, m - 2, k - 1 - ell) / 4
                    - coefficient(data, m + 1, k - 1 - ell) / 8
                )
                y_m_ell[m - 2 - n_minus_m_ell, k] = 2 * delta / k / data["rho"] ** ell * coefficient(data, m - 1, k - 1 - ell) / 8

            # P(0, L_{0}^{-1}y_{m, \ell}) and T(y_{m, \ell})
            v_m_ell, T_m_ell = finite_inverse(data, y_m_ell, M_m_ell, n_minus_m_ell, n_plus_m_ell)

            # Coefficient check.
            # include q_{m}\mathbf{1}_{\{\ell=0\}} term.
            if L <= m <= U and ell == 0:
                v_m_ell[0, 0] = data["q"][m - L]
            # include I_{PX}P(0, e_{m, \ell}) term.
            if L <= m <= U and ell <= K:
                v_m_ell[coordinate_index(data, m, ell), 0] += 1
                # subtract \overline{J}P(0, e_{m, \ell}).
                for i in range(dimension):
                    v_m_ell[i, 0] -= J[i, coordinate_index(data, m, ell)]

            bound_m_ell = matrix_norm(A_fin * v_m_ell) + T_m_ell
            coefficient_bound = coefficient_bound.max(bound_m_ell)

    # High-Taylor checks

    high_bound = arb(0)
    for m in data["S"]:
        M_m_ell0 = ell0 + K + 1
        n_minus_m_ell0 = min(0, L, m - 2)
        n_plus_m_ell0 = max(U, m + 2)

        # Compute y_{m, \ell_{0}}.
        y_m_ell0 = arb_mat(n_plus_m_ell0 - n_minus_m_ell0 + 1, M_m_ell0 + 1)
        for k in range(ell0 + 1, ell0 + K + 2):
            y_m_ell0[m + 2 - n_minus_m_ell0, k] = -2 * delta / k / data["rho"] ** ell0 * coefficient(data, m + 1, k - 1 - ell0)
            y_m_ell0[m + 1 - n_minus_m_ell0, k] = -2 * delta / k / data["rho"] ** ell0 * (
                coefficient(data, m - 1, k - 1 - ell0)
                + coefficient(data, m + 2, k - 1 - ell0) / 4
            )
            y_m_ell0[m - 1 - n_minus_m_ell0, k] = -2 * delta / k / data["rho"] ** ell0 * (
                coefficient(data, m - 2, k - 1 - ell0) / 4
                - coefficient(data, m + 1, k - 1 - ell0) / 8
            )
            y_m_ell0[m - 2 - n_minus_m_ell0, k] = 2 * delta / k / data["rho"] ** ell0 * coefficient(data, m - 1, k - 1 - ell0) / 8
        
        bound_m_ell0 = arb(0)
        for k in range(ell0 + 1, ell0 + K + 2):
            for n in range(m - 2, m + 3):
                e_n_k_minus = min(0, L, n)
                e_n_k_plus = max(U, n)
                e_n_k = arb_mat(e_n_k_plus - e_n_k_minus + 1, k + 1)
                e_n_k[n - e_n_k_minus, k] = 1 / ( data["rho"] ** k )
                v_n_k, T_n_k = finite_inverse(data, e_n_k, k, e_n_k_minus, e_n_k_plus)
                bound_m_ell0 += data["rho"] ** k * abs(y_m_ell0[n - n_minus_m_ell0, k]) * (matrix_norm(A_fin * v_n_k) + T_n_k)
        high_bound = high_bound.max(bound_m_ell0)

    bound = period_bound.max(coefficient_bound).max(high_bound).upper()

    assert bound <= Z1_app, "Could not verify the prescribed bound Z1_app."
    
    return bound

def check_radius(data, constants, r):
    """
    Check the conditions for the prescribed radius r using the prescribed constants.
    """
    polynomial = (constants["Z0"] + (constants["Z1_inv"] + constants["Z1_app"] - 1) * r + constants["Z2"] * r ** 2 + constants["Z3"] * r ** 3)
    assert r > 0 and polynomial < 0, "The radii inequality failed."
    assert r < data["delta"], "Could not verify delta^{(ast)} > 0."

    norm_a0 = arb(0)
    for n in range(data["I"][0], data["I"][1] + 1):
        norm_a0 += abs(coefficient(data, n, 0))
    assert r < norm_a0.lower(), "Could not verify a^{(0)} is nontrivial."

def upper_decimal(value, digits=10):
    """
    Round an upper endpoint upward to the requested number of significant digits.
    This is only for displaying bounds.
    """
    # value.upper() = mantissa * 2 ** exponent
    mantissa, exponent = map(int, value.upper().man_exp())

    if exponent >= 0:
        numerator, denominator = mantissa << exponent, 1
    else:
        numerator, denominator = mantissa, 1 << (-exponent)

    # Convert the upper endpoint to a decimal string, rounding upward.
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_CEILING
        return str(Decimal(numerator) / Decimal(denominator))

if __name__ == "__main__":

    # Prescribed constants and radius used in the paper.
    prescribed_constants = {
        "Z0": "1e-6",
        "Z1_inv": "1e-13",
        "Z1_app": "0.52",
        "Z2": "3200",
        "Z3": "260",
    }
    prescribed_r = "1e-5"

    # Prescribe the precision we use in interval arithmetic.
    precision = 64
    ctx.prec = precision

    # Store prescribed constants as interval enclosures.
    constants = {name: arb(value) for name, value in prescribed_constants.items()}
    r = arb(prescribed_r)

    start = perf_counter()
    
    # Read the input data and run the checks.
    # directory = codes/section_2
    directory = Path(__file__).resolve().parent

    lines = [f"precision = {precision} bits", 
             f"Python {platform.python_version()}, NumPy {np.__version__}, python-flint {flint.__version__}"]
    results_path = directory / "results_orbit.txt"

    results_path.write_text("Verification started; not yet completed.\n")
    try:
        data = read_data(directory / "data.npz")
        lines += [f"I = {data['I']}, K = {data['K']}, L = {data['L']}, U = {data['U']}",
                  f"delta <= {upper_decimal(data['delta'], 17)}",
                  f"ell_0 = {data['ell0']}, dim(PX) = {data['dimension']}"]
        times = {}

        step = perf_counter()
        print("check_parameters", flush=True)
        check_parameters(data)
        times["check_parameters"] = perf_counter() - step
        lines.append("check_parameters: verified")

        step = perf_counter()
        print("check_z1_inv", flush=True)
        Z1_inv = check_z1_inv(data, constants["Z1_inv"])
        times["check_z1_inv"] = perf_counter() - step
        lines.append(f"check_z1_inv: {upper_decimal(Z1_inv)} <= {prescribed_constants['Z1_inv']}: verified")

        step = perf_counter()
        print("check_z0", flush=True)
        Z0 = check_z0(data, constants["Z0"])
        times["check_z0"] = perf_counter() - step
        lines.append(f"check_z0: {upper_decimal(Z0)} <= {prescribed_constants['Z0']}: verified")

        step = perf_counter()
        print("check_z2_z3", flush=True)
        Z2, Z3 = check_z2_z3(data, constants["Z2"], constants["Z3"])
        times["check_z2_z3"] = perf_counter() - step
        lines.append(f"check_z2_z3: {upper_decimal(Z2)} <= {prescribed_constants['Z2']}, "
                     f"{upper_decimal(Z3)} <= {prescribed_constants['Z3']}: verified")

        step = perf_counter()
        print("check_z1_app", flush=True)
        Z1_app = check_z1_app(data, constants["Z1_app"])
        times["check_z1_app"] = perf_counter() - step
        lines.append(f"check_z1_app: {upper_decimal(Z1_app)} <= {prescribed_constants['Z1_app']}: verified")

        step = perf_counter()
        print("check_radius", flush=True)
        check_radius(data, constants, r)
        times["check_radius"] = perf_counter() - step
        lines += ["check_radius: verified", f"r = {prescribed_r}"]

        lines.append("Run times (seconds):")
        for name, seconds in times.items():
            lines.append(f"  {name}: {seconds:.6f}")
        lines.append(f"  total, including input: {perf_counter() - start:.6f}")
        lines.append("All inequalities verified.")

    except Exception as error:
        lines.append(f"Verification failed: {error}")
        results_path.write_text("\n".join(lines) + "\n")
        raise

    results_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)