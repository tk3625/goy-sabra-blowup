# Interval arithmetic verification for Section 3.

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

def coefficient(data, n, k):
    """
    Return a_n^(k), extended by zero outside its support.
    """
    I_left, I_right = data["I"]
    if I_left <= n <= I_right and 0 <= k <= data["K"]:
        return data["a"][n - I_left][k]
    return arb(0)

def coordinate(data, n, a):
    """
    Real and imaginary parts are the first (a=1) and second (a=2) 
    coordinates of each shell.
    """
    return 2 * (n - data["n_min"]) + a - 1

def matrix_norm(matrix):
    """
    Compute the induced max-norm of a finite matrix.
    """
    result = arb(0)
    for i in range(matrix.nrows()):
        row_sum = arb(0)
        for j in range(matrix.ncols()):
            row_sum += abs(matrix[i, j])
        result = result.max(row_sum)
    return result

def check_parameters(data, model, indices, J, p, r_C, j0):
    """
    Check the required parameter conditions.
    """
    assert model in ("GOY", "Sabra"), "Selected model should be either GOY or Sabra."
    assert len(indices) == (4 if model == "GOY" else 3), "Incorrect number of sampling indices defining pi."
    for n in indices:
        assert data["n_minus"] <= n <= data["n_plus"], "A sampling index defining pi is outside the support of overline{a}."
    assert indices[0] != indices[1], "The real sampling indices defining pi must be distinct."
    if model == "GOY":
        assert indices[2] != indices[3], "The imaginary sampling indices defining pi must be distinct."
    else:
        assert 3 <= j0 <= data["n_plus"], "Could not verify 3 <= j0 <= I_plus."
    assert J >= 0 and p >= 1, "Could not verify J >= 0 and p >= 1."
    assert r_C > 1, "Could not verify r_{C} > 1."
    
def check_invertibility(data, model, indices):
    """
    Return interval enclosures of $overline{cV}$ and $pi overline{cV}$,
    and check that the enclosure of its determinant excludes zero.
    """
    cVbar = arb_mat(data["dimension"], len(indices))
    for n in range(data["n_minus"] - 1, data["n_plus"] + 2):
        nonlinear = (
            2 * coefficient(data, n - 1, 0) * coefficient(data, n - 2, 0)
            + coefficient(data, n + 1, 0) * coefficient(data, n - 1, 0) / 2
            - coefficient(data, n + 2, 0) * coefficient(data, n + 1, 0) / 4
        )
        cVbar[coordinate(data, n, 1), 0] = nonlinear
        cVbar[coordinate(data, n, 1), 1] = nonlinear - coefficient(data, n, 0)
        if model == "GOY":
            if n % 3 == 0:
                cVbar[coordinate(data, n, 2), 2] = -coefficient(data, n, 0)
                cVbar[coordinate(data, n, 2), 3] = -coefficient(data, n, 0)
            elif n % 3 == 1:
                cVbar[coordinate(data, n, 2), 2] = coefficient(data, n, 0)
            else:
                cVbar[coordinate(data, n, 2), 3] = coefficient(data, n, 0)
        else:
            cVbar[coordinate(data, n, 2), 2] = ( (arb(5).sqrt() + 1) / 2 ) ** n * coefficient(data, n, 0)
    picVbar = arb_mat(len(indices), len(indices))
    for i, n in enumerate(indices):
        # i=0, 1 corresponds to real mode => a=1
        # i=2, (3 for GOY) corresponds to imaginary mode => a=2
        a = 1 if i < 2 else 2
        for j in range(len(indices)):
            picVbar[i, j] = cVbar[coordinate(data, n, a), j]
    assert not picVbar.det().contains(0), "Could not verify that pi overline{cV} is invertible."
    return cVbar, picVbar

def compute_phase_error(data, r, j0):
    """
    Verify the Sabra tail condition and return an interval enclosure 
    of epsilon_S.
    """
    C_Q = arb(11) / 4
    max_sum_in_condition = arb(0)
    for n in range(j0 - 2, data["n_plus"] + 1):
        sum_in_condition = arb(0)
        for k in range(data["K"] + 1):
            sum_in_condition += abs(coefficient(data, n, k))
        max_sum_in_condition = max_sum_in_condition.max(sum_in_condition)
    
    assert C_Q * (1 + arb(5).sqrt()) / 2 * ( r + max_sum_in_condition ) <= 1, "Could not verify the Sabra tail condition."
    
    a0_tail = arb(0)
    for n in range(j0, data["n_plus"] + 1):
        a0_tail = a0_tail.max( ((1 + arb(5).sqrt()) / 2) ** n * abs(coefficient(data, n, 0)))
    return (((1 + arb(5).sqrt()) / 2) ** (j0 - 1) * r).max( 
        C_Q * ((1 + arb(5).sqrt()) / 2) ** (j0 + 1) * (r + max_sum_in_condition) ** 2 
        + a0_tail )

def compute_eta(data, model, picVbar, r, epsilon_S):
    """
    Return an interval enclosure of eta, and check eta < 1.
    """
    inverse = picVbar.inv()
    C_Q = arb(11) / 4
    norm_a0 = arb(0)
    for n in range(data["I"][0], data["I"][1] + 1):
        norm_a0 = norm_a0.max(abs(coefficient(data, n, 0)))
    epsilon_V = (1 + 2 * C_Q * (2 * norm_a0 + r)) * r
    if model == "GOY":
        epsilon_V += 2 * r
    else:
        epsilon_V += epsilon_S
    eta = matrix_norm(inverse) * epsilon_V
    assert eta < 1, "Could not verify eta < 1."
    return eta

def compute_contraction_norm(data, model, indices, cVbar, picVbar, eta, J, p, r_C, r):
    """
    Compute the interval enclosure of ||overline{widetilde{M}}^{p}|| 
    + epsilon_{widetilde{M}, p}.
    Return an upper endpoint, and check that it is strictly less than 1.
    """

    dimension = data["dimension"]

    K = data["K"]
    delta = data["delta"]

    # DNa[k] stores DN(a^{(k)}).
    DNa = []

    # DN(a^(k)) has both input and output support in [n_minus-2, n_plus+2].
    for k in range(K + 1):
        DNak = arb_mat(dimension, dimension)
        for n in range(data["n_minus"] - 2, data["n_plus"] + 3):
            for m in range(data["n_minus"] - 2, data["n_plus"] + 3):
                value = arb(0)
                if m == n - 2:
                    value = 2 * coefficient(data, n - 1, k)
                elif m == n - 1:
                    value = 2 * coefficient(data, n - 2, k) + coefficient(data, n + 1, k) / 2
                elif m == n + 1:
                    value = coefficient(data, n - 1, k) / 2 - coefficient(data, n + 2, k) / 4
                elif m == n + 2:
                    value = -coefficient(data, n + 1, k) / 4
                
                imag_value = value
                if model == "GOY":
                    imag_value = -value
                elif m == n - 1:
                    imag_value = 2 * coefficient(data, n - 2, k) - coefficient(data, n + 1, k) / 2
                elif m == n + 1:
                    imag_value = coefficient(data, n - 1, k) / 2 + coefficient(data, n + 2, k) / 4
                DNak[coordinate(data, n, 1), coordinate(data, m, 1)] = value
                DNak[coordinate(data, n, 2), coordinate(data, m, 2)] = imag_value
        DNa.append(DNak)

    identity = arb_mat(dimension, dimension)
    for i in range(dimension):
        identity[i, i] = 1

    # Xi[j] stores overline{Xi}^{(j)}.
    # Mbar at this point only computes \sum_{j=0}^{J}\overline{\Xi}^{(j)}.
    Xi = [identity]
    Mbar = identity
    for j in range(J):
        Xij = arb_mat(dimension, dimension)
        for k in range(min(j, K) + 1):
            Xij += DNa[k] * Xi[j - k]
        Xij *= delta / (j + 1)
        Xi.append(Xij)
        Mbar = Mbar + Xij

    # R^{-1} shifts by one to the left.
    left_shift = arb_mat(dimension, dimension)
    for n in range(data["n_min"], data["n_max"]):
        for a in (1, 2):
            left_shift[coordinate(data, n, a), coordinate(data, n + 1, a)] = 1   
    # Compute \overline{M}.
    Mbar = (-delta).exp() * left_shift * Mbar

    # pi extracts the sampled coordinates.
    pi = arb_mat(len(indices), dimension)
    sampled_indices = []
    for j, n in enumerate(indices):
        a = 1 if j < 2 else 2
        sampled_indices.append((n, a))
        pi[j, coordinate(data, n, a)] = 1

    inverse = picVbar.inv()
    Pibar = cVbar * inverse * pi

    Mtildebar = (identity - Pibar) * Mbar

    # The norm of \overline{M}.
    Mbar_norm = matrix_norm(Mbar).max((-delta).exp())

    # The norm of I-\overline{\Pi}.
    IminusPibar_norm = matrix_norm(identity - Pibar).max(arb(1))

    # The norms of the powers of \overline{\widetilde{M}}.
    Mtildebar_norms = [arb(1)]
    Mtildebar_power = identity
    for j in range(1, p + 1):
        Mtildebar_power = Mtildebar * Mtildebar_power
        power_norm = (-j * delta).exp()
        for n in range(data["n_minus"] - j - 2, data["n_plus"] + 2):
            for a in (1, 2):
                row_sum = arb(0)
                for m in range(data["n_minus"] - 2, data["n_plus"] + j + 2):
                    for b in (1, 2):
                        if (m, b) not in sampled_indices:
                            row_sum += abs(Mtildebar_power[coordinate(data, n, a), coordinate(data, m, b)])
                power_norm = power_norm.max(row_sum)
        Mtildebar_norms.append(power_norm)

    # Compute \epsilon_{\Pi}.
    assert 0 <= eta < 1, "The supplied eta bound must lie in [0,1)."
    epsilon_Pi = eta / (1 - eta) * IminusPibar_norm

    # Compute \Lambda, \epsilon_{var}.
    Lambda = arb(0)
    for k in range(K + 1):
        DNak_norm = matrix_norm(DNa[k])
        Lambda += DNak_norm * delta

    C_Q = arb(11) / 4
    epsilon_var = r / delta * Lambda + 2 * arb(2).sqrt() * C_Q * (delta + r) * r

    # Compute \epsilon_{T}.
    Taylor_exponent = arb(0)
    for k in range(K + 1):
        DNak_norm = matrix_norm(DNa[k])
        Taylor_exponent += DNak_norm * delta * r_C ** (k + 1) / (k + 1) 
    
    epsilon_T = Taylor_exponent.exp() / (r_C ** (J + 1) * (1 - 1 / r_C))
    
    # Compute \epsilon_{M}.
    epsilon_M = (-delta + r).exp() * (epsilon_var * (Lambda + epsilon_var).exp() + r * Lambda.exp()) + (-delta).exp() * epsilon_T

    # Compute \epsilon_{\widetilde{M}}.
    epsilon_Mtilde = epsilon_Pi * (Mbar_norm + epsilon_M) + IminusPibar_norm * epsilon_M

    # Compute \epsilon_{\widetilde{M}, p} recursively.
    epsilon_Mtilde_p = [arb(0)]
    for q in range(1, p + 1):
        error_sum = arb(0)
        for j in range(q):
            error_sum += Mtildebar_norms[q - 1 - j] * (Mtildebar_norms[j] + epsilon_Mtilde_p[j])
        epsilon_Mtilde_p.append(epsilon_Mtilde * error_sum)

    bound = (Mtildebar_norms[p] + epsilon_Mtilde_p[p]).upper()
    assert bound < 1, "Could not verify contraction."
    return bound

def upper_decimal(value, digits=10):
    """
    Round an upper endpoint upward to the requested number of significant digits.
    This is only for displaying bounds.
    """
    # value = mantissa * 2 ** exponent
    mantissa, exponent = map(int, value.man_exp())

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

    # Model, sampling indices, and power p.
    models = (("GOY", (-2, 2, 0, 1), 4), ("Sabra", (0, 2, 2), 5))
    
    J = 80
    prescribed_r_C = "2"

    # The validated radius from Section 2 and $j_{0}$ for Sabra.
    prescribed_r = "1e-5"
    j0 = 9

    # Prescribe the precision we use in interval arithmetic.
    precision = 64
    ctx.prec = precision
    r_C = arb(prescribed_r_C)
    r = arb(prescribed_r)

    start = perf_counter()

    # codes/section_3
    directory = Path(__file__).resolve().parent
    lines = [f"precision = {precision} bits",
             f"Python {platform.python_version()}, NumPy {np.__version__}, python-flint {flint.__version__}"]
    results_path = directory / "results_contraction.txt"

    results_path.write_text("Verification started; not yet completed.\n")
    try:
        # Read the same approximation used in Section 2.
        data_path = directory / "../section_2/data.npz"
        data_directory = data_path.resolve().parent
        data = {}

        with np.load(data_path) as saved:
            approximation_path = data_directory / str(saved["approximation_file"].item())

        with np.load(approximation_path) as approximation:
            data["I"] = (int(approximation["I"][0]), int(approximation["I"][1]))
            data["K"] = int(approximation["K"])
            data["delta"] = exact_float(approximation["delta"])
            data["a"] = [[exact_float(value) for value in row] for row in approximation["a"]]

        data["n_minus"], data["n_plus"] = data["I"]

        # This window contains all rows and columns needed for the 
        # finite formulas for \overline{\widetilde{M}}^{j} (1 \le j \le p).
        max_p = max(p for model, indices, p in models)
        data["n_min"] = data["n_minus"] - max_p - 2
        data["n_max"] = data["n_plus"] + max_p + 1
        data["dimension"] = 2 * (data["n_max"] - data["n_min"] + 1)

        lines += [f"I = {data['I']}, K = {data['K']}",
                  f"delta = {data['delta']}",
                  f"J = {J}, r_C = {prescribed_r_C}",
                  f"matrix window = [{data['n_min']}, {data['n_max']}], dimension = {data['dimension']}",
                  f"validated radius from Section 2: r = {prescribed_r}"]
        times = {}

        for model, indices, p in models:
            lines += [f"{model}: sampling indices = {indices}, p = {p}"]
            step = perf_counter()
            print(model, "check_parameters", flush=True)
            check_parameters(data, model, indices, J, p, r_C, j0)
            times[model + " check_parameters"] = perf_counter() - step
            lines.append("check_parameters: verified")

            # In GOY case, \epsilon_{S} = None.
            epsilon_S = None
            if model == "Sabra":
                step = perf_counter()
                print(model, "compute_phase_error", flush=True)
                epsilon_S = compute_phase_error(data, r, j0)
                times[model + " compute_phase_error"] = perf_counter() - step
                lines.append(f"compute_phase_error: j0 = {j0}, epsilon_S = {epsilon_S}")

            step = perf_counter()
            print(model, "check_invertibility", flush=True)
            cVbar, picVbar = check_invertibility(data, model, indices)
            times[model + " check_invertibility"] = perf_counter() - step
            lines += ["check_invertibility of pi overline{cV}: verified"]

            step = perf_counter()
            print(model, "compute_eta", flush=True)
            eta = compute_eta(data, model, picVbar, r, epsilon_S)
            times[model + " compute_eta"] = perf_counter() - step
            lines.append(f"compute_eta: eta = {eta}, verified < 1")

            step = perf_counter()
            print(model, "compute_contraction_norm", flush=True)
            bound = compute_contraction_norm(data, model, indices, cVbar, picVbar, eta, J, p, r_C, r)
            times[model + " compute_contraction_norm"] = perf_counter() - step
            lines.append(f"compute_contraction_norm: final upper bound <= {upper_decimal(bound)}, verified < 1")
            
        lines.append("Run times (seconds):")
        for name, seconds in times.items():
            lines.append(f"  {name}: {seconds:.6f}")
        for model, *_ in models:
            subtotal = sum( seconds for name, seconds in times.items() if name.startswith(model + " ") )
            lines.append(f"  {model} verification total: {subtotal:.6f}")
        lines.append(f"  total, including input: {perf_counter() - start:.6f}")
        lines.append("All inequalities verified.")

    except Exception as error:
        lines.append(f"Verification failed: {error}")
        results_path.write_text("\n".join(lines) + "\n")
        raise

    results_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
