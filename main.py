import numpy as np
import matplotlib.pyplot as plt

# ==============================
# PARAMETERS
# ==============================
LENGTH = 100       # Reactor length (cm)
N = 100            # Grid points
DX = LENGTH / N

D = 1.0            # Diffusion coefficient
SIGMA_A = 0.1      # Absorption cross-section
NU_SIGMA_F = 0.12  # Production term

MAX_ITER = 300
TOL = 1e-5


# ==============================
# INITIALIZATION
# ==============================
def initialize_flux(n):
    phi = np.ones(n)
    phi[0] = 0
    phi[-1] = 0
    return phi


# ==============================
# DIFFUSION STEP
# ==============================
def diffusion_step(phi, k):
    phi_new = phi.copy()

    for i in range(1, len(phi) - 1):
        diffusion = D * (phi[i+1] - 2*phi[i] + phi[i-1]) / DX**2
        production = (NU_SIGMA_F / k) * phi[i]

        phi_new[i] = (diffusion + production) / SIGMA_A

    return phi_new


# ==============================
# NORMALIZATION
# ==============================
def normalize(phi):
    return phi / np.max(phi)


# ==============================
# K-EFFECTIVE UPDATE
# ==============================
def update_k(phi_old, phi_new, k_old):
    num = np.sum(NU_SIGMA_F * phi_new)
    den = np.sum(NU_SIGMA_F * phi_old)
    return k_old * (num / den)


# ==============================
# MAIN SOLVER
# ==============================
def solve():
    phi = initialize_flux(N)
    k = 1.0

    for iteration in range(MAX_ITER):
        phi_new = diffusion_step(phi, k)
        phi_new = normalize(phi_new)

        k_new = update_k(phi, phi_new, k)

        if abs(k_new - k) < TOL:
            print(f"Converged in {iteration} iterations")
            break

        phi = phi_new
        k = k_new

    return phi, k


# ==============================
# RESULTS
# ==============================
def plot_results(phi, k):
    x = np.linspace(0, LENGTH, N)

    plt.figure()
    plt.plot(x, phi)
    plt.xlabel("Reactor Length (cm)")
    plt.ylabel("Neutron Flux")
    plt.title(f"Neutron Flux Distribution (k = {k:.4f})")

    plt.savefig("../results/flux.png")
    plt.show()


def print_reactor_state(k):
    print(f"\nk-effective = {k:.4f}")

    if k < 1:
        print("Reactor State: SUBCRITICAL")
    elif abs(k - 1) < 1e-3:
        print("Reactor State: CRITICAL")
    else:
        print("Reactor State: SUPERCRITICAL")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    phi, k = solve()
    print_reactor_state(k)
    plot_results(phi, k)
