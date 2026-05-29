import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import matplotlib

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
# ==========================================
# 1. Define the Parameters
# ==========================================
g_tilde = 1.0    
v = 1.0          

r_min = 1e-2     # Keeping 0.01 to avoid hard singularities at exactly 0
r_max = 10.0     
N_points = 1000  

# ==========================================
# 2. Define the System of ODEs
# ==========================================
def ode_system(r, y):
    F, W = y
    
    # === YOUR UPDATED EQUATION ===
    dF = (2 * W / r) * (2 - g_tilde * r * W)
    
    # Second equation remains the same
    dW = (F / r) * (1 - g_tilde * r * W) - (W / r)
    
    return np.vstack((dF, dW))

# ==========================================
# 3. Define the Boundary Conditions
# ==========================================
def boundary_conditions(ya, yb):
    return np.array([
        ya[0] - 0.0,                           # F(0) = 0
        yb[1] - 1.0 / (g_tilde * r_max)        # W(inf) = 1 / (\tilde{g} * r_max)
    ])

# ==========================================
# 4. Physically Accurate Initial Guess
# ==========================================
r_mesh = np.linspace(r_min, r_max, N_points)

# Smooth quadratic growth near origin transitioning to constants
F_guess = v * (1 - np.exp(-r_mesh**2)) 
W_guess = (1 - np.exp(-r_mesh**3)) / (g_tilde * r_mesh) 

y_guess = np.vstack((F_guess, W_guess))

# ==========================================
# 5. Solve the BVP
# ==========================================
# max_nodes increased to give the solver plenty of room to work
solution = solve_bvp(ode_system, boundary_conditions, r_mesh, y_guess, max_nodes=100000)

if not solution.success:
    print("\n--- SOLVER FAILED ---")
    print("Reason:", solution.message)
    print("---------------------\n")
else:
    print("\nSolver converged successfully!\n")

    # ==========================================
    # 6. Plot the Results
    # ==========================================
    r_plot = np.linspace(r_min, r_max, 1000)
    y_plot = solution.sol(r_plot)

    F_sol = y_plot[0]
    W_sol = y_plot[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot F(r)
    ax1.plot(r_plot, F_sol, label='$F(r)$', color='blue', linewidth=2)
    ax1.set_xlabel('radial distance $r$',  fontsize=18)
    ax1.set_ylabel('$F(r)$',  fontsize=18)
    ax1.set_title('Profile of the $F$ field', fontsize=20)
    ax1.legend(fontsize=14)
    ax1.grid(True)

    # Plot W(r)
    ax2.plot(r_plot, W_sol, label='$W(r)$', color='red', linewidth=2)
    ax2.plot(r_plot, 1 / (g_tilde * r_plot), color='gray', linestyle='--', 
             label='Asymptotic $1/(\\tilde{g}r)$')
    ax2.set_ylim(-0.1, 1.5) # Constrained to ignore infinite visual spikes near 0
    ax2.set_xlabel('radial distance $r$',  fontsize=18)
    ax2.set_ylabel('$W(r)$',  fontsize=18)
    ax2.set_title('Profile of the $W$ field',  fontsize=20)
    ax2.legend(fontsize=14)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()