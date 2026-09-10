import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import matplotlib

matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)

# ---------------------------------------------------------
# 1. Define the Parameters
# ---------------------------------------------------------
# 
lmbda = 1.0      # \lambda
v = 1.0          # v
g_tilde = 1.0    # \tilde{g}

r_min = 1e-3     # Start slightly above 0 to avoid 1/r singularity
r_max = 10.0     # Outer boundary (approximating infinity)
N_points = 500   # Number of points for the initial mesh

# ---------------------------------------------------------
# 2. Define the System of ODEs
# ---------------------------------------------------------
def ode_system(r, y):
    """
    y[0] = F, y[1] = F'
    y[2] = W, y[3] = W'
    """
    F, dF, W, dW = y
    
    # Equation 2.38 for F''
    term1_F = (2 * F / r**2) * (1 - g_tilde * r * W)**2
    term2_F = lmbda * F * (F**2 - v**2)
    d2F = -(2 / r) * dF + term1_F + term2_F
    
    # Equation 2.39 for W''
    term1_W = (W / r**2) * (1 - g_tilde * r * W) * (2 - g_tilde * r * W)
    term2_W = (g_tilde * F**2 / r) * (1 - g_tilde * r * W)
    d2W = -(2 / r) * dW + term1_W - term2_W
    
    return np.vstack((dF, d2F, dW, d2W))

# ---------------------------------------------------------
# 3. Define the Boundary Conditions
# ---------------------------------------------------------
def boundary_conditions(ya, yb):
    """
    ya: values at r_min
    yb: values at r_max
    
    Standard topological defect boundary conditions assumed:
    At r -> 0: F = 0, W is regular (we'll set W=0)
    At r -> inf: F = v, (1 - g_tilde*r*W) -> 0 => W = 1/(g_tilde*r)
    """
    return np.array([
        ya[0] - 0.0,                           # F(r_min) = 0
        yb[0] - v,                             # F(r_max) = v
        ya[2] - 0.0,                           # W(r_min) = 0
        yb[2] - 1.0 / (g_tilde * r_max)        # W(r_max) = 1 / (\tilde{g} * r_max)
    ])

# ---------------------------------------------------------
# 4. Initial Guess for the Solver
# ---------------------------------------------------------
# BVP solvers require a good initial guess to converge.
r_mesh = np.linspace(r_min, r_max, N_points)

# Smooth transition guesses satisfying the boundary conditions
F_guess = v * (1 - np.exp(-r_mesh))
dF_guess = v * np.exp(-r_mesh)
W_guess = (1 - np.exp(-r_mesh)) / (g_tilde * r_max) 
dW_guess = np.exp(-r_mesh) / (g_tilde * r_max)

y_guess = np.vstack((F_guess, dF_guess, W_guess, dW_guess))

# ---------------------------------------------------------
# 5. Solve the BVP
# ---------------------------------------------------------
solution = solve_bvp(ode_system, boundary_conditions, r_mesh, y_guess)

if not solution.success:
    print("Solver failed to converge:")
    print(solution.message)
else:
    print("Solver converged successfully!")

    # ---------------------------------------------------------
    # 6. Plot the Results
    # ---------------------------------------------------------
    r_plot = np.linspace(r_min, r_max, 1000)
    y_plot = solution.sol(r_plot)

    F_sol = y_plot[0]
    W_sol = y_plot[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot F(r)
    ax1.plot(r_plot, F_sol, label='$F(r)$', color='blue', linewidth=2)
    ax1.axhline(v, color='gray', linestyle='--', label='Vacuum expectation $v$')
    ax1.set_xlabel('radial distance $r$', fontsize=18)
    ax1.set_ylabel('$F(r)$', fontsize=18)
    ax1.set_title('Profile of the $F$ field', fontsize=20)
    ax1.legend(fontsize=14)
    ax1.grid(True)

    # Plot W(r)
    ax2.plot(r_plot, W_sol, label='$W(r)$', color='red', linewidth=2)
    ax2.plot(r_plot, 1/(g_tilde*r_plot), color='gray', linestyle='--', 
             label='Asymptotic $1/(\\tilde{g}r)$')
    ax2.set_ylim(-0.1, 1.5) # Constrain y-axis due to 1/r asymptote visual
    ax2.set_xlabel('radial distance $r$', fontsize=18)
    ax2.set_ylabel('$W(r)$', fontsize=18)
    ax2.set_title('Profile of the $W$ field', fontsize=20)
    ax2.legend(fontsize=14)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()