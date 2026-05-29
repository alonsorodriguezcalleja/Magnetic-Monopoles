import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

# ---------------------------------------------------------
# 1. Physical Parameters (Set to order 1 for visualization)
# ---------------------------------------------------------
g = 1.0          # SU(2) coupling
g_prime = 1.0    # U(1) coupling
lam = 1.0        # Higgs self-coupling lambda
rho0 = 1.0       # Higgs vacuum expectation value v
beta = 2      # Born-Infeld parameter 

# ---------------------------------------------------------
# 2. Radial Grid and Trial Profiles
# ---------------------------------------------------------
# Start slightly above 0 to avoid division by zero warnings
r = np.linspace(0.02, 5.0, 500)

# Trial functions matching the boundary conditions:
# rho(0) = 0, rho(inf) = rho0
# f(0) = 1, f(inf) = 0
rho = rho0 * np.tanh(r)
drho = rho0 * (1.0 / np.cosh(r)**2)  # Derivative of rho

f = 1.0 / np.cosh(r)
df = -np.sinh(r) / np.cosh(r)**2     # Derivative of f

# ---------------------------------------------------------
# 3. Energy Density Components (dE/dr)
# ---------------------------------------------------------
# Common terms for the SU(2) and Higgs sectors (from Eq. 2.63)
# Note: The factor of 4*pi is omitted for the sake of the plot scale
E_Higgs_kinetic = (drho)**2 + (rho**2 / 4.0) * f**2
E_SU2_gauge = ((1 - f**2)**2) / (2 * g**2 * r**2) + (df)**2 / g**2
E_Higgs_potential = (lam / 8.0) * (rho**2 - rho0**2)**2

E_common = E_Higgs_kinetic + E_SU2_gauge + E_Higgs_potential

# Scenario A: Unregularized U(1)_Y
# Diverges as 1/r^2 at the origin
U1_unreg = 2.0 / (g_prime**2 * r**2)
dE_dr_unreg = E_common + U1_unreg

# Scenario B: CKY Regularization
# Multiplied by (rho/rho0)^2 (assuming c0=0, c1=1)
U1_CKY = U1_unreg * (rho / rho0)**2
dE_dr_CKY = E_common + U1_CKY

# Scenario C: Born-Infeld Extension
# Replaces the kinetic term to tame the singularity
# Formula adapted to match the U(1) asymptotic limit
U1_BI = r**2 * beta**2 * (np.sqrt(1 + 4.0 / (g_prime**2 * beta**2 * r**4)) - 1)
dE_dr_BI = E_common + U1_BI

# ---------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(r, dE_dr_unreg, label='Unregularized (SM)', color='red', linestyle='--')
plt.plot(r, dE_dr_CKY, label='CKY Regularization ($c_1=1$)', color='blue', linewidth=2)
plt.plot(r, dE_dr_BI, label=f'Born-Infeld Extension ($\\beta={beta}$)', color='green', linewidth=2)

# Graph formatting
plt.title('Energy Density of the Cho-Maison Monopole', fontsize=24)
plt.xlabel('Radial distance $r$', fontsize=24)
plt.ylabel('Energy Density (a.u.)', fontsize=24)

# Limit y-axis to see the regularized curves clearly despite the divergence
plt.ylim(0, 15)
plt.xlim(0, 5)

plt.grid(True, alpha=0.4)
plt.legend(loc='upper right', fontsize=18)
plt.tight_layout()

plt.show()