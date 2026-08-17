import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. Load SU2 CFD Data
# ============================================================================
print("Loading SU2 data...")
try:
    su2_data = np.genfromtxt("restart_flow.csv", delimiter=",", names=True)
except FileNotFoundError:
    print("ERROR: restart_flow.csv not found. Run SU2 first.")
    exit()

# Extract coordinates and conservative variables
x_su2 = su2_data["x"]
y_su2 = su2_data["y"]
rho = su2_data["Density"]
rhou = su2_data["Momentum_x"]
rhov = su2_data["Momentum_y"]
rhoE = su2_data["Energy"]

# Convert to primitive variables
rho_safe = np.where(rho < 1e-8, 1e-8, rho)
u_su2 = rhou / rho_safe
v_su2 = rhov / rho_safe
gamma = 1.4
p_su2 = (gamma - 1.0) * (rhoE - 0.5 * rho * (u_su2**2 + v_su2**2))

# ============================================================================
# 2. Load PINN Predictions
# ============================================================================
print("Loading PINN predictions...")
try:
    pinn_data = np.genfromtxt("pure_pinn_predictions.csv", delimiter=",", names=True)
except FileNotFoundError:
    print("ERROR: pinn_predictions.csv not found. Did you run the infer_cavity tool?")
    exit()

x_pinn = pinn_data["x"]
y_pinn = pinn_data["y"]
u_pinn = pinn_data["u"]
v_pinn = pinn_data["v"]
p_pinn = pinn_data["p"]

# ============================================================================
# 3. Plotting (3x2 Grid: u, v, p for SU2 vs PINN)
# ============================================================================
print("Generating plots...")
fig, axes = plt.subplots(3, 2, figsize=(12, 16))

variables = [
    {"name": "u-velocity", "su2": u_su2, "pinn": u_pinn, "cmap": "jet"},
    {"name": "v-velocity", "su2": v_su2, "pinn": v_pinn, "cmap": "jet"},
    {"name": "pressure",   "su2": p_su2, "pinn": p_pinn, "cmap": "viridis"}
]

for i, var in enumerate(variables):
    # Calculate a shared color scale based on SU2 ground truth
    vmin = var["su2"].min()
    vmax = var["su2"].max()
    
    # Plot SU2 CFD (Left Column)
    ax_su2 = axes[i, 0]
    tcf_su2 = ax_su2.tricontourf(x_su2, y_su2, var["su2"], levels=50, cmap=var["cmap"], vmin=vmin, vmax=vmax)
    ax_su2.set_title(f"SU2 CFD ({var['name']})", fontsize=14)
    ax_su2.set_xlabel("x", fontsize=12)
    ax_su2.set_ylabel("y", fontsize=12)
    ax_su2.set_aspect('equal')
    fig.colorbar(tcf_su2, ax=ax_su2, label=var['name'])

    # Plot PINN (Right Column)
    ax_pinn = axes[i, 1]
    tcf_pinn = ax_pinn.tricontourf(x_pinn, y_pinn, var["pinn"], levels=50, cmap=var["cmap"], vmin=vmin, vmax=vmax)
    ax_pinn.set_title(f"PINN ({var['name']})", fontsize=14)
    ax_pinn.set_xlabel("x", fontsize=12)
    ax_pinn.set_ylabel("y", fontsize=12)
    ax_pinn.set_aspect('equal')
    fig.colorbar(tcf_pinn, ax=ax_pinn, label=var['name'])

plt.tight_layout()
plt.savefig("pure_pinn_results.png", dpi=300)
print("Plot saved to pure_pinn_results.png")
