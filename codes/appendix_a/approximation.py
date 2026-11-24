# Numerical data for Appendix A

import numpy as np
from pathlib import Path
from time import perf_counter

def N(z):
    """
    Compute the array $N_{real}(z)$ from the input array $z$. 
    Values of z outside the index range of the input array are taken to be zero.
    N_{real}(z)_{n} = 2z_{n-1}z_{n-2}+z_{n+1}z_{n-1}/2-z_{n+2}z_{n+1}/4.
    """
    result = np.zeros_like(z)
    result[2:] += 2.0 * z[1:-1] * z[0:-2]
    result[1:-1] += z[2:] * z[0:-2] / 2.0
    result[:-2] -= z[2:] * z[1:-1] / 4.0
    return result

def rk4_step(z, dt):
    """
    One step of classical RK4.
    """
    k1 = N(z)
    k2 = N(z + dt * k1 / 2.0)
    k3 = N(z + dt * k2 / 2.0)
    k4 = N(z + dt * k3)
    return z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

def weighted_center(z, n):
    """
    Center of squared mass.
    n is the array of shell indices, e.g. [-10, -9, ..., 9, 10],
    and z is the value at these shell indices.
    """
    weights = (z / np.max(np.abs(z))) ** 2
    return float(np.dot(n, weights) / np.sum(weights))

def check_boundary_small(values, boundary_tol):
    """
    Check if the finite interval $I$ is enough to contain the numerical support of the data.
    Return two flags indicating whether the left or right boundary values are too large.
    """
    threshold = boundary_tol * np.max(np.abs(values))
    left_small = np.max(np.abs(values[0:2])) >= threshold
    right_small = np.max(np.abs(values[-2:])) >= threshold
    return bool(left_small), bool(right_small)

def enlarge_interval(I, is_boundary_small):
    """
    Double only the endpoint(s) of $I$ whose boundary check failed.
    """
    (I_left, I_right) = I
    return (2 * I_left if is_boundary_small[0] else I_left,
            2 * I_right if is_boundary_small[1] else I_right)

def compare_profiles(z_l, z_r):
    """
    Return the amplitude growth A = ||z_r||_2 / ||z_l||_2 
    and the relative return error.
    """
    
    scale_z_l, scale_z_r = np.max(np.abs(z_l)), np.max(np.abs(z_r))
    z_l_renorm, z_r_renorm = z_l / scale_z_l, z_r / scale_z_r
    ratio = np.linalg.norm(z_r_renorm) / np.linalg.norm(z_l_renorm)
    A = (scale_z_r / scale_z_l) * ratio

    difference = np.r_[0.0, z_l_renorm] - np.r_[z_r_renorm / ratio, 0.0]
    relative_error = np.linalg.norm(difference) / np.linalg.norm(z_l_renorm)
    return float(A), float(relative_error)

def integrate(I, h0, return_tol, boundary_tol, max_steps):
    """
    Using finite interval $I$, solve using RK4.
    Return a candidate, or the sides of $I$ requiring expansion.
    """
    n = np.arange(I[0], I[1] + 1)
    z = np.zeros(len(n))
    z[-I[0]] = z[1 - I[0]] = 1.0
    z_l = z.copy()
    target = weighted_center(z_l, n) + 1
    T = 0.0
    crossings = []

    for step in range(1, max_steps + 1):
        is_boundary_small = check_boundary_small(z, boundary_tol)
        if is_boundary_small[0] or is_boundary_small[1]:
            return None, is_boundary_small
        dt = h0 / max(1.0, np.max(np.abs(z)))
        z_r = rk4_step(z, dt)
        if not np.isfinite(z_r).all():
            raise ArithmeticError("The RK4 step overflowed; choose a smaller h0.")
        is_boundary_small = check_boundary_small(z_r, boundary_tol)
        if is_boundary_small[0] or is_boundary_small[1]:
            return None, is_boundary_small

        c_r = weighted_center(z_r, n)
        if c_r < target:
            T += dt
            z = z_r
        else:
            # Roll back to z and take one step over the interpolated duration.
            c_l = weighted_center(z, n)
            dt *= (target - c_l) / (c_r - c_l)
            z_r = rk4_step(z, dt)
            if not np.isfinite(z_r).all():
                raise ArithmeticError("The interpolated RK4 step overflowed.")
            is_boundary_small = check_boundary_small(z_r, boundary_tol)
            if is_boundary_small[0] or is_boundary_small[1]:
                return None, is_boundary_small
            T += dt
            A, relative_error = compare_profiles(z_l, z_r)
            c_r = weighted_center(z_r, n)
            crossings.append({"center": c_r, "A": A, "relative_error": relative_error,
                              "interpolation_error": c_r - target})
            if len(crossings) % 10 == 0 or relative_error < return_tol:
                print(f"  return {len(crossings)}: relative error = {relative_error:.6g}",
                      flush=True)
            if A > 1 and relative_error < return_tol:
                delta = float(np.log(A))
                a0 = (A / (A - 1)) * T * z_l
                return {"delta": delta, "a0": a0, "A": A, "T": T,
                        "steps": step, "crossings": crossings}, (False, False)
            z_l = z_r.copy()
            target = c_r + 1
            T = 0.0
            z = z_r

    raise RuntimeError("No return was accepted within max_steps.")

def taylor_coefficients(delta, a0, taylor_tol=1e-9, max_degree=512):
    """
    Compute the Taylor coefficients a^{(k)} using a^{(0)}.
    Stop at the first k >= 1 with
    ||a^{(k)}||_{inf} / ||a^{(0)}||_{inf} <= taylor_tol.
    """
    
    # Arrays of Taylor coefficients have Taylor index first and shell index second.
    # This will be transposed later.
    # At this point, a_{I[0]+i}^{(k)} corresponds to a[k][i].

    a = [np.array(a0, dtype=float, copy=True)]
    k = 0
    while k < max_degree:
        nonlinear = np.zeros(len(a0))
        for j in range(k + 1):
            u, v = a[j], a[k - j]
            nonlinear[2:] += 2.0 * u[1:-1] * v[0:-2]
            nonlinear[1:-1] += u[2:] * v[0:-2] / 2.0
            nonlinear[0:-2] -= u[2:] * v[1:-1] / 4.0
        a.append(delta / (k + 1) * (nonlinear - a[k]))
        k += 1

        if not np.isfinite(a[k]).all():
            raise ArithmeticError("The Taylor coefficients overflowed.")
        if np.max(np.abs(a[k])) <= taylor_tol * np.max(np.abs(a[0])):
            # Transpose a so that it has shell index first and Taylor index second.
            return np.asarray(a).T
        
    raise RuntimeError("The requested Taylor truncation exceeds max_degree.")

def generate_approximation(h0=1e-4, return_tol=1e-9, boundary_tol=1e-9,
                           taylor_tol=1e-9, max_steps=2000000,
                           max_degree=512, max_restarts=15):
    """
    Return (I, K, delta, a), with the center of a^{(0)} in [0, 1).

    Start on I = [-2, 2]. After boundary failure, double the offending
    endpoint(s) and restart from z_{0}=z_{1}=1 at t=0.
    Output I is the final pair of endpoints AFTER integer shell
    relabeling; a[i, k] corresponds to shell I[0]+i and Taylor degree k.
    """

    I = (-2, 2)
    for attempt in range(max_restarts + 1):
        print(f"Start from t=0 on I={I}", flush=True)
        candidate, is_boundary_small = integrate(I, h0, return_tol, boundary_tol, max_steps)
        if candidate is None:
            print(f"  restart: trajectory boundary, left={is_boundary_small[0]}, " f"right={is_boundary_small[1]}", flush=True)
            I = enlarge_interval(I, is_boundary_small)
            continue
        else:
            a = taylor_coefficients(candidate["delta"], candidate["a0"], taylor_tol=taylor_tol, max_degree=max_degree)
            is_boundary_small = check_boundary_small(a, boundary_tol)

            if is_boundary_small[0] or is_boundary_small[1]:
                print(f"  restart: Taylor coefficient boundary, left={is_boundary_small[0]}, " f"right={is_boundary_small[1]}", flush=True)
                I = enlarge_interval(I, is_boundary_small)
                continue

        n = np.arange(I[0], I[1] + 1)
        shift = int(np.floor(weighted_center(a[:, 0], n)))
        I = (I[0] - shift, I[1] - shift)
        K, delta = a.shape[1] - 1, candidate["delta"]
        return I, K, delta, a

    raise RuntimeError("Too many interval expansions; use larger max_restarts.")

if __name__ == "__main__":
    start = perf_counter()

    I, K, delta, a = generate_approximation(
        h0=1e-4,
        return_tol=1e-9,
        boundary_tol=1e-9,
        taylor_tol=1e-9,
        max_steps=2000000
    )
    # codes/appendix_a
    directory = Path(__file__).resolve().parent
    np.savez_compressed(directory / "approximation.npz", I=np.array(I), K=K, delta=delta, a=a)
    
    elapsed = perf_counter() - start

    print(f"I = {I}, K = {K}, delta = {delta:.17g}")
    print(f"Approximation time, including saving: {elapsed:.6f} seconds")