import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

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
def initialize_flux(n: int) -> np.ndarray:
    """Initialize neutron flux with boundary conditions.
    
    Args:
        n: Number of grid points
        
    Returns:
        Initial flux array with zeros at boundaries
        
    Raises:
        ValueError: If n < 3
    """
    if n < 3:
        raise ValueError("Grid points must be at least 3")
    
    phi = np.ones(n)
    phi[0] = 0
    phi[-1] = 0
    return phi


# ==============================
# DIFFUSION STEP
# ==============================
def diffusion_step(phi: np.ndarray, k: float) -> np.ndarray:
    """Perform one diffusion step using finite difference method.
    
    Args:
        phi: Current neutron flux
        k: Current k-effective
        
    Returns:
        Updated flux after diffusion step
    """
    phi_new = phi.copy()

    for i in range(1, len(phi) - 1):
        diffusion = D * (phi[i+1] - 2*phi[i] + phi[i-1]) / DX**2
        removal = SIGMA_A * phi[i]
        production = (NU_SIGMA_F / k) * phi[i]

        phi_new[i] = phi[i] + (diffusion - removal + production)

    return phi_new


# ==============================
# NORMALIZATION
# ==============================
def normalize(phi: np.ndarray) -> np.ndarray:
    """Normalize flux to maximum value.
    
    Args:
        phi: Flux array to normalize
        
    Returns:
        Normalized flux array
        
    Raises:
        ValueError: If maximum flux is zero
    """
    max_val = np.max(np.abs(phi))
    if max_val == 0:
        raise ValueError("Cannot normalize zero flux")
    return phi / max_val


# ==============================
# K-EFFECTIVE UPDATE
# ==============================
def update_k(phi_old: np.ndarray, phi_new: np.ndarray, k_old: float) -> float:
    """Update k-effective using power iteration method.
    
    Args:
        phi_old: Previous flux
        phi_new: Current flux
        k_old: Previous k-effective
        
    Returns:
        Updated k-effective value
    """
    num = np.sum(NU_SIGMA_F * phi_new)
    den = np.sum(NU_SIGMA_F * phi_old)
    
    if den == 0:
        raise ValueError("Denominator is zero in k update")
    
    return k_old * (num / den)


# ==============================
# MAIN SOLVER
# ==============================
def solve() -> Tuple[np.ndarray, float, List[float]]:
    """Solve reactor criticality problem using power iteration.
    
    Returns:
        Tuple of (final flux, k-effective, k_history)
        
    Raises:
        RuntimeError: If solution diverges
    """
    phi = initialize_flux(N)
    k = 1.0
    k_history = [k]

    for iteration in range(MAX_ITER):
        phi_new = diffusion_step(phi, k)
        phi_new = normalize(phi_new)

        k_new = update_k(phi, phi_new, k)
        k_history.append(k_new)

        # Check for divergence
        if k_new > 10 or k_new < 0.01:
            raise RuntimeError(f"Solution diverging at iteration {iteration}: k = {k_new:.6f}")

        if abs(k_new - k) < TOL:
            print(f"Converged in {iteration} iterations")
            break

        phi = phi_new
        k = k_new
    else:
        print(f"Warning: Did not converge after {MAX_ITER} iterations")

    return phi, k, k_history


# ==============================
# RESULTS
# ==============================
def plot_results(phi: np.ndarray, k: float) -> None:
    """Plot neutron flux distribution.
    
    Args:
        phi: Neutron flux array
        k: k-effective value
    """
    x = np.linspace(0, LENGTH, N)

    plt.figure(figsize=(10, 6))
    plt.plot(x, phi, 'b-', linewidth=2)
    plt.xlabel("Reactor Length (cm)", fontsize=12)
    plt.ylabel("Neutron Flux", fontsize=12)
    plt.title(f"Neutron Flux Distribution (k = {k:.4f})", fontsize=14)
    plt.grid(True, alpha=0.3)

    os.makedirs("../results", exist_ok=True)
    plt.savefig("../results/flux.png", dpi=300, bbox_inches='tight')
    print("Saved flux plot to ../results/flux.png")
    plt.show()


def plot_convergence(k_history: List[float]) -> None:
    """Plot convergence of k-effective.
    
    Args:
        k_history: List of k-effective values at each iteration
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_history, 'o-', linewidth=2, markersize=4)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("k-effective", fontsize=12)
    plt.title("Convergence of k-effective", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    os.makedirs("../results", exist_ok=True)
    plt.savefig("../results/convergence.png", dpi=300, bbox_inches='tight')
    print("Saved convergence plot to ../results/convergence.png")
    plt.show()


def print_reactor_state(k: float) -> None:
    """Print reactor criticality state.
    
    Args:
        k: k-effective value
    """
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
    try:
        phi, k, k_history = solve()
        print_reactor_state(k)
        plot_results(phi, k)
        plot_convergence(k_history)
    except Exception as e:
        print(f"Error during execution: {e}")
        raise