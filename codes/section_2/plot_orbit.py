# Plot of the travelling wave for Section 2.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_orbit(a, I, figure_save_directory):
    """
    Plot the approximate travelling wave $(overline{Z}(tau))_{n}$ over 
    n-tau/overline{delta}.
    """
    n = np.arange(I[0], I[1] + 1)
    # theta = tau/delta.
    theta = np.linspace(0.0, 1.0, 10001)
    fig, ax = plt.subplots(figsize=(6, 3))

    for i in range(I[1] - I[0] + 1):
        # a[i] contains overline{a}_{n[i]}^{(k)} for all k
        profile = np.polynomial.polynomial.polyval(theta, a[i])
        ax.plot(n[i] - theta, profile, color="teal")

    # Mark the values at \tau=0.
    ax.plot(n, a[:, 0], "o", markeredgecolor="teal", markersize=3, markerfacecolor="white")
    ax.set_xlim(-12.5, 7.5)
    ax.set_ylim(-0.05, 0.8)
    ax.set_xticks([-12, -9, -6, -3, 0, 3, 6])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(r"$n-\frac{\tau}{\delta^{(\ast)}}$")
    ax.set_ylabel("Profile amplitude")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_save_directory / "bilateral_profile.png", dpi=220)
    plt.close(fig)

if __name__ == "__main__":
    # codes/section_2
    directory = Path(__file__).resolve().parent
    with np.load(directory.parent / "appendix_a" / "approximation.npz") as data:
        plot_orbit(data["a"], data["I"], directory)
