import numpy as np
import matplotlib.pyplot as plt

def simulate_sde(num_simulations=5, T=1, dt=0.01, S0=None, f=None, g=None):
    """
    Simulates a general Stochastic Differential Equation (SDE):
        dX_t = f(t, X_t) dt + g(t, X_t) dB_t

    Parameters:
    - num_simulations: Number of paths to simulate.
    - T: Total time horizon.
    - dt: Time step size.
    - S0: Array of initial values (length = num_simulations).
    - f: Drift function f(t, X_t).
    - g: Diffusion function g(t, X_t).

    Returns:
    - t: Time array.
    - S: Simulated SDE paths (shape: [num_steps, num_simulations]).
    """
    np.random.seed(42)  # For reproducibility
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)  # Time grid

    # If S0 is not provided, default to ones
    if S0 is None:
        S0 = np.ones(num_simulations)

    # Ensure S0 is a numpy array
    S0 = np.array(S0)

    # Initialize paths
    S = np.zeros((N, num_simulations))
    S[0, :] = S0  # Set initial values

    # Simulate SDE paths using Euler-Maruyama method
    for i in range(1, N):
        dW = np.random.randn(num_simulations) * np.sqrt(dt)  # Brownian increments
        S_prev = S[i-1, :]
        
        drift = f(t[i-1], S_prev) * dt  # Compute drift f(t, X_t) dt
        diffusion = g(t[i-1], S_prev) * dW  # Compute noise g(t, X_t) dB_t
        
        S[i, :] = S_prev + drift + diffusion  # Euler-Maruyama step

    return t, S

def plot_sde(t, S):
    """
    Plots the simulated SDE paths.

    Parameters:
    - t: Time array.
    - S: Simulated SDE paths.
    """
    plt.figure(figsize=(6, 4))
    for j in range(S.shape[1]):  # Iterate over simulations
        plt.plot(t, S[:, j], alpha=1.0)

    plt.xlabel("Time: t")
    plt.ylabel(r"State: $X_t$")
    plt.title(r"SDE Simulation: $dX_t = f(t, X_t) dt + g(t, X_t) dB_t$")
    plt.grid()
    plt.tight_layout()
    plt.show()

# Example usage:
num_simulations = 200
T = 5.5
dt = 0.001

# Generate random initial values for each simulation
S0 = np.random.normal(5.0, 0.2, num_simulations)

# Define the drift and noise functions
def f(t, X):
    return 0.1*X

def g(t, X):
    return 0.0*X

# Simulate SDE paths
t, S = simulate_sde(num_simulations, T, dt, S0, f, g)

# Plot the results
plot_sde(t, S)
