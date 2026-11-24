# Numerical data for the computer-assisted verification in Section 2.

from pathlib import Path
import numpy as np
from time import perf_counter

def Q(u, v):
    """
    Compute the symmetric bilinear map Q(u, v).
    Values outside the index range of the input arrays are taken to be zero.
    """
    result = np.zeros_like(u)
    result[2:] += u[1:-1] * v[0:-2] + v[1:-1] * u[0:-2]
    result[1:-1] += (u[2:] * v[0:-2] + v[2:] * u[0:-2]) / 4.0
    result[0:-2] -= (u[2:] * v[1:-1] + v[2:] * u[1:-1]) / 8.0
    return result

def finite_inverse(y, delta, K):
    """
    Compute the coordinates of h=L_{0}^{-1}y through Taylor degree K,
    where y is finitely supported.
    y[n, k] denotes $y_{n}^{(k)}$.
    """

    # y_{n}^{(k)}=0 if k>M
    M = y.shape[1] - 1

    h = np.zeros((len(y), K + 1))

    # Compute h_{n}^{(0)} = e^{-\overline{\delta}} h_{n+1}^{(0)} + (summand with j=0)
    h[:, 0] = -y[:, 0]
    for ell in range(1, M + 1):
        # Sum \frac{\ell! (-\delta)^{j}}{(\ell+j)!} for j=0,...,39.
        partial_sum = 1.0
        term = 1.0
        for j in range(1, 40):
            term *= (-delta) / (ell + j)
            partial_sum += term
        h[0:-1, 0] += partial_sum * y[1:, ell]

    for n in range(len(y) - 2, -1, -1):
        h[n, 0] += np.exp(-delta) * h[n + 1, 0]

    # Compute h_{n}^{(k)} = -\frac{\delta}{k} h_{n}^{(k-1)} + y_{n}^{(k)}.
    for k in range(1, K + 1):
        if k <= M:
            h[:, k] = y[:, k] - delta / k * h[:, k - 1]
        else:
            h[:, k] = -delta / k * h[:, k - 1]

    return h

def prepare_data(delta, a, I, K, rho, L, U, S):
    """
    Compute and return q, J, and A_fin for Section 2.
    The matrices use the basis (1, 0), followed by (0, e_{n,k}), ordered
    first by L <= n <= U and then by 0 <= k <= K.
    ((L, 0), (L, 1), ..., (L, K), (L+1, 0), ..., (L+1, K), ..., (U, K))
    """

    # \overline{J} approximates PDG(\overline{x})|_{PX}
    dimension_P = 1 + (U - L + 1) * (K + 1)
    J = np.zeros((dimension_P, dimension_P))

    # The zeroth column of J corresponds to period input
    # This should be P(0, L_{0}^{-1}y_{\delta}).
    # y_{\delta} is supported in $[I[0]-2, I[1]+2] \times [0, 2K+1]$,
    # which is $\mathcal{S} \times [0, 2K+1]$.
     
    n_min, n_max = S
    n = np.arange(n_min, n_max + 1)
    a_extended = np.zeros((len(n), K + 1))
    # Note that $I=[I[0], I[1]]$ is contained in $\mathcal{S}=[n_min, n_max]$.
    # a_extended is the zero extension of a to $\mathcal{S}$.
    a_extended[I[0] - n_min:I[1] - n_min + 1] = a

    # q is the normalization of $N(a^{(0)})-a^{(0)}$.
    a0 = a_extended[:, 0]
    tangent = (Q(a0, a0) - a0)[L - n_min:U - n_min + 1]
    if np.linalg.norm(tangent) == 0:
        raise ValueError("The approximate tangent q is zero on [L, U].")
    q = tangent / np.linalg.norm(tangent)

    # Compute y_{\delta}.
    y_delta = np.zeros((len(n), 2 * K + 2))
    for k in range(1, 2 * K + 2):
        if k - 1 <= K:
            y_delta[:, k] = a_extended[:, k-1]
        for j in range(max(0, k - 1 - K), min(K, k - 1) + 1):
            y_delta[:, k] -= Q(a_extended[:, j], a_extended[:, k - 1 - j])
        y_delta[:, k] /= k

    # Compute L_{0}^{-1}y_{\delta} =: h_{\delta}
    h_delta = finite_inverse(y_delta, delta, K)

    # The matrix entries are with respect to the normalized basis.
    numerical_weights = np.zeros((U - L + 1, K + 1))
    for m in range(L, U + 1):
        for k in range(K + 1):
            numerical_weights[m - L, k] = rho ** k

    J[1:, 0] = (numerical_weights * h_delta[L - n_min: U - n_min + 1]).ravel()

    # The remaining columns of $J$ correspond to $e_{m, \ell}$ with $m \in [L, U], \ell \in [0, K]$.
    # The phase component may be nonzero, and
    # the coefficient component is e_{m, \ell} + L_{0}^{-1}y_{m, \ell}.
    
    for m in range(L, U + 1):
        # one_m stores \mathbf{1}_{\{m\}}.
        one_m = np.zeros(len(n))
        one_m[m - n_min] = 1.0
        for ell in range(K + 1):
            # index of $e_{m, \ell}$ in the basis of $PX$
            column = 1 + (m - L) * (K + 1) + ell
            
            # y_{m, \ell} is supported in $[m-2, m+2] \times [\ell+1, \ell+K+1]$,
            # which is contained in $\mathcal{S} \times [0, 2K+1]$.
            y_m_ell = np.zeros((len(n), 2 * K + 2))
            for j in range(K + 1):
                k = ell + j + 1
                y_m_ell[:, k] = -2.0 * delta / k * rho ** (-ell) * Q(a_extended[:, j], one_m)

            # Compute L_{0}^{-1}y_{m, \ell} =: h_{m, \ell}
            h_m_ell = finite_inverse(y_m_ell, delta, K)
            
            J[1:, column] = (numerical_weights * h_m_ell[L - n_min: U - n_min + 1]).ravel()
            J[column, column] += 1.0

            # There is a phase component when $\ell=0$
            if ell == 0:
                J[0, column] = q[m - L] * rho ** (-ell) 

    # $A_{\mathrm{fin}}$ is the numerical inverse of $\overline{J}$.
    A_fin = np.linalg.inv(J)

    return q, J, A_fin

if __name__ == "__main__":
    start = perf_counter()

    L = -20
    U = 8
    S = (-53, 23)
    ell0 = 39

    # Store the weight \rho as integer (numerator, denominator) pairs.
    rho_ratio = (101, 100)
    rho = rho_ratio[0] / rho_ratio[1]

    # codes/section_2
    directory = Path(__file__).resolve().parent
    approximation_file = "../appendix_a/approximation.npz"
    # codes/appendix_a/approximation.npz
    approximation_path = directory / approximation_file

    with np.load(approximation_path) as approximation:
        I = tuple(int(n) for n in approximation["I"])
        K = int(approximation["K"])
        delta = float(approximation["delta"])
        a = approximation["a"]

    q, J, A_fin = prepare_data(delta, a, I, K, rho, L, U, S)

    # Save q, J, A_fin and the parameters in data.npz.
    # Store a relative path to the Appendix A file, without copying I, K, delta, or a.
    data = dict(q=q, J=J, A_fin=A_fin,
                rho=rho_ratio, L=L, U=U, S=S, ell0=ell0,
                approximation_file=approximation_file)
    np.savez_compressed(directory / "data.npz", **data)
    elapsed = perf_counter() - start

    print(f"I = {I}, K = {K}, delta = {delta:.17g}")
    print(f"L = {L}, U = {U}, ell0 = {ell0}, matrix dimension = {J.shape[0]}")
    print(f"Preparation time, including input and saving: {elapsed:.6f} seconds")