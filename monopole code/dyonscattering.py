import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)

def dyon_scattering(q, kappa, mu, v0, b_array):
    chi = 2.0 * np.arctan(np.abs(kappa) / (mu * v0 * b_array))
    
    if q > 0:
        xi_half = np.arctan((np.abs(kappa) * v0 / q) * (1.0 / np.tan(chi/2)))
    elif q < 0:
        xi_half = np.pi - np.arctan((np.abs(kappa) * v0 / np.abs(q)) * (1.0 / np.tan(chi/2)))
    else:
        xi_half = np.ones_like(chi) * (np.pi / 2)

    argument = xi_half / np.cos(chi/2)
    cos_theta_half = np.cos(chi/2) * np.abs(np.sin(argument))
    
    cos_theta_half = np.clip(cos_theta_half, -1.0, 1.0)
    Theta = 2.0 * np.arccos(cos_theta_half)

    dTheta = np.gradient(Theta)
    db = np.gradient(b_array)
    
    valid = np.abs(np.sin(Theta)) > 1e-4
    
    dsig_dOmega = np.zeros_like(b_array)
    dsig_dOmega[valid] = (b_array[valid] / np.sin(Theta[valid])) * np.abs(db[valid] / dTheta[valid])
    
    return Theta, dsig_dOmega

def rutherford_scattering(q_eff, mu, v0, theta_array):
    factor = (q_eff / (2.0 * mu * v0**2))**2
    return factor / (np.sin(theta_array / 2.0)**4)

# ---------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------
mu = 1.0
v0 = 1.0
kappa = 1.0   
q_mag = 1.0   

# Huge number of impact parameters for a dense, clean scatter plot
b_values = np.logspace(-2, 2, 100000)

theta_rep, xsec_rep = dyon_scattering(q=q_mag, kappa=kappa, mu=mu, v0=v0, b_array=b_values)
theta_att, xsec_att = dyon_scattering(q=-q_mag, kappa=kappa, mu=mu, v0=v0, b_array=b_values)

theta_ruth = np.linspace(0.05, np.pi, 1000)
q_effective = np.sqrt(q_mag**2 + kappa**2) 
xsec_ruth = rutherford_scattering(q_effective, mu=v0, v0=v0, theta_array=theta_ruth)

# --- Plotting ---
plt.figure(figsize=(10, 6))

# The repulsive case is mathematically well-behaved (no spiraling), so a line plot works perfectly
sort_rep = np.argsort(theta_rep)
plt.semilogy(theta_rep[sort_rep], xsec_rep[sort_rep], label='Repulsive Dyon (q > 0)', color='red', lw=2)

# Attractive case as a scatter plot with small points (s=0.5) and slight transparency.
# This prevents Matplotlib from drawing lines between different orbital branches.
plt.scatter(theta_att, xsec_att, label='Attractive Dyon (q < 0)', color='blue', s=0.5, alpha=0.3)

plt.semilogy(theta_ruth, xsec_ruth, label='Standard Rutherford', color='black', linestyle='--', lw=2)

# Formatting
plt.title('Differential Cross-Section: Dyon vs. Rutherford', fontsize=20)
plt.xlabel(r'Scattering Angle $\Theta$ (radians)', fontsize=18)
plt.ylabel(r'$d\sigma / d\Omega$', fontsize=18)
plt.xlim(0, np.pi)
plt.ylim(1e-2, 1e4)
plt.grid(True, which="both", ls=":", alpha=0.6)

# Legends
leg = plt.legend(fontsize=16)
leg.legend_handles[1]._sizes = [40]

plt.tight_layout()
plt.show()