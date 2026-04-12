"""
analysis_and_plots.py
=====================
Open-loop nonlinear simulation and visualisation of the
inclined-plane magnetic-ball system.
"""



import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import control


import parameters as p
from nonlinear_dynamics import MagneticBallSystem
from linearisation import get_linearised_system

#  Matplotlib style settings 
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "lines.linewidth":   2,
    "grid.alpha":        0.35,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# linearised model and equilibrium 

A_num, B_num, C_num, D_num, x_eq, i_eq, V_eq, ss_sys = get_linearised_system()

print(f"\nEquilibrium:  x_eq = {x_eq:.4f} m,  i_eq = {i_eq:.6f} A,  V_eq = {V_eq:.4f} V")

#  Simulation setup 
dx0   = -0.02  # [m]   position offset  (−2 cm away from the magnet)
di0   = 0.0    # [A]   no current perturbation

x0_vec = np.array([x_eq + dx0,   # x
                    0.0,          # x_dot
                    i_eq + di0,   # i
                    x_eq + dx0])  # x_m (sensor initialised at ball position)

t_span = (0.0, 1.5)      # simulation window [s]
t_eval = np.linspace(*t_span, 3000)

# Constant voltage input = V_eq (open loop, no controller)
V_input = V_eq

# Nonlinear simulation 

system = MagneticBallSystem()

def ode_rhs(t, state):
    return system.dynamics(t, state, V_input)

print("\nRunning nonlinear simulation (scipy RK45) …")

# Terminal event: stop integration if the ball gets dangerously close to
# the electromagnet (gap < 5 mm) or moves past the spring attachment point.
def event_magnet_proximity(t, state):
    """Triggered when gap to electromagnet < 5 mm."""
    return (p.delta - state[0]) - 0.005

def event_wall(t, state):
    """Triggered when ball position reaches 0 (lower wall)."""
    return state[0]

event_magnet_proximity.terminal  = True
event_magnet_proximity.direction = -1   # gap is decreasing
event_wall.terminal  = True
event_wall.direction = -1               # position decreasing to zero

sol = solve_ivp(
    ode_rhs,
    t_span,
    x0_vec,
    method="RK45",
    t_eval=t_eval,
    events=[event_magnet_proximity, event_wall],
    rtol=1e-9,
    atol=1e-11,
    dense_output=False,
)

if sol.status == 1:
    print(f"  Integration stopped by event at t = {sol.t[-1]:.4f} s.")
elif not sol.success:
    print(f"  WARNING: solver message: {sol.message}")
else:
    print(f"  Done. {sol.t.shape[0]} time steps.")

t_nl   = sol.t
x_nl   = sol.y[0]     # ball position
xd_nl  = sol.y[1]     # ball velocity
i_nl   = sol.y[2]     # coil current
xm_nl  = sol.y[3]     # sensor output

#  Linearised simulation (for comparison) 

# Initial deviation from equilibrium
delta_x0 = np.array([x0_vec[0] - x_eq,
                      x0_vec[1] - 0.0,
                      x0_vec[2] - i_eq,
                      x0_vec[3] - x_eq])

# Deviation input: zero (open loop at V_eq)
def u_lin(t):
    return np.array([[0.0]])

print("Running linearised simulation (control.forced_response) …")
t_lin_arr = t_eval
response   = control.forced_response(
    ss_sys,
    T=t_lin_arr,
    U=np.zeros((1, len(t_lin_arr))),
    X0=delta_x0,
)
# response is a TimeResponseData object
t_lin  = response.t
y_lin  = response.outputs          # shape (1, N)  – output y = x_m deviation
x_lin  = response.states           # shape (4, N)  – state deviations

x_lin_abs  = x_lin[0]  + x_eq     # absolute position
xd_lin_abs = x_lin[1]             # velocity deviation (zero mean)
i_lin_abs  = x_lin[2]  + i_eq     # absolute current
print("  Done.")

#  5. Plotting 
COLORS = {
    "nl":   "#1f77b4",   # blue  – nonlinear
    "lin":  "#d62728",   # red   – linearised
    "eq":   "#2ca02c",   # green – equilibrium
    "ref":  "#7f7f7f",   # grey  – reference / annotation
}

def save_fig(fig, name: str):
    path = f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

#  Figure 1: Ball position 

fig1, ax1 = plt.subplots(figsize=(9, 4.5))

ax1.plot(t_nl, x_nl,       color=COLORS["nl"],  label="Nonlinear")
ax1.plot(t_lin, x_lin_abs, color=COLORS["lin"], linestyle="--", label="Linearised")
ax1.axhline(x_eq, color=COLORS["eq"], linewidth=1.4,
            linestyle=":", label=f"Equilibrium  $x_{{eq}}={x_eq}$ m")
ax1.axhline(p.delta, color=COLORS["ref"], linewidth=1, linestyle="-.",
            label=f"Electromagnet  $\\delta={p.delta}$ m")

ax1.set_xlabel("Time  $t$  [s]")
ax1.set_ylabel("Ball position  $x$  [m]")
ax1.set_title("Open-Loop Response – Ball Position")
ax1.legend(loc="upper right")
ax1.grid(True)
ax1.set_xlim(t_span)

save_fig(fig1, "fig1_position")

#  Figure 2: Ball velocity 

fig2, ax2 = plt.subplots(figsize=(9, 4.5))

ax2.plot(t_nl, xd_nl,       color=COLORS["nl"],  label="Nonlinear")
ax2.plot(t_lin, xd_lin_abs, color=COLORS["lin"], linestyle="--", label="Linearised")
ax2.axhline(0.0, color=COLORS["eq"], linewidth=1.4, linestyle=":", label="Zero velocity")

ax2.set_xlabel("Time  $t$  [s]")
ax2.set_ylabel("Ball velocity  $\\dot{x}$  [m/s]")
ax2.set_title("Open-Loop Response – Ball Velocity")
ax2.legend(loc="upper right")
ax2.grid(True)
ax2.set_xlim(t_span)

save_fig(fig2, "fig2_velocity")

#  Figure 3: Coil current 

fig3, ax3 = plt.subplots(figsize=(9, 4.5))

ax3.plot(t_nl, i_nl,       color=COLORS["nl"],  label="Nonlinear")
ax3.plot(t_lin, i_lin_abs, color=COLORS["lin"], linestyle="--", label="Linearised")
ax3.axhline(i_eq, color=COLORS["eq"], linewidth=1.4, linestyle=":",
            label=f"Equilibrium  $i_{{eq}}={i_eq:.4f}$ A")

ax3.set_xlabel("Time  $t$  [s]")
ax3.set_ylabel("Coil current  $i$  [A]")
ax3.set_title("Open-Loop Response – Coil Current")
ax3.legend(loc="upper right")
ax3.grid(True)
ax3.set_xlim(t_span)

save_fig(fig3, "fig3_current")

#  Figure 4: Phase portrait  ẋ vs x 

fig4, ax4 = plt.subplots(figsize=(6.5, 5.5))

ax4.plot(x_nl,       xd_nl,       color=COLORS["nl"],  label="Nonlinear")
ax4.plot(x_lin_abs,  xd_lin_abs,  color=COLORS["lin"], linestyle="--", label="Linearised")

# Mark start and equilibrium
ax4.scatter([x0_vec[0]], [x0_vec[1]], color="black", zorder=5,
            s=60, label=f"Initial state $x_0={x0_vec[0]:.3f}$ m")
ax4.scatter([x_eq], [0.0], color=COLORS["eq"], marker="*", s=180, zorder=5,
            label=f"Equilibrium $({x_eq}, 0)$")

ax4.set_xlabel("Ball position  $x$  [m]")
ax4.set_ylabel("Ball velocity  $\\dot{x}$  [m/s]")
ax4.set_title("Phase Portrait  $\\dot{x}$ vs $x$")
ax4.legend(loc="upper right")
ax4.grid(True)

save_fig(fig4, "fig4_phase_portrait")

#  Figure 5: Combined summary

fig5 = plt.figure(figsize=(13, 9))
gs   = gridspec.GridSpec(2, 2, figure=fig5, hspace=0.42, wspace=0.35)

axes = [fig5.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]

# (0,0) Position
axes[0].plot(t_nl, x_nl,       color=COLORS["nl"],  label="Nonlinear")
axes[0].plot(t_lin, x_lin_abs, color=COLORS["lin"], linestyle="--", label="Linearised")
axes[0].axhline(x_eq, color=COLORS["eq"], linestyle=":", linewidth=1.2)
axes[0].set_xlabel("$t$ [s]");  axes[0].set_ylabel("$x$ [m]")
axes[0].set_title("Ball Position"); axes[0].legend(fontsize=8); axes[0].grid(True)

# (0,1) Velocity
axes[1].plot(t_nl, xd_nl,       color=COLORS["nl"],  label="Nonlinear")
axes[1].plot(t_lin, xd_lin_abs, color=COLORS["lin"], linestyle="--", label="Linearised")
axes[1].axhline(0.0, color=COLORS["eq"], linestyle=":", linewidth=1.2)
axes[1].set_xlabel("$t$ [s]");  axes[1].set_ylabel("$\\dot{x}$ [m/s]")
axes[1].set_title("Ball Velocity"); axes[1].legend(fontsize=8); axes[1].grid(True)

# (1,0) Current
axes[2].plot(t_nl, i_nl,       color=COLORS["nl"],  label="Nonlinear")
axes[2].plot(t_lin, i_lin_abs, color=COLORS["lin"], linestyle="--", label="Linearised")
axes[2].axhline(i_eq, color=COLORS["eq"], linestyle=":", linewidth=1.2)
axes[2].set_xlabel("$t$ [s]");  axes[2].set_ylabel("$i$ [A]")
axes[2].set_title("Coil Current"); axes[2].legend(fontsize=8); axes[2].grid(True)

# (1,1) Phase portrait
axes[3].plot(x_nl,      xd_nl,      color=COLORS["nl"],  label="Nonlinear")
axes[3].plot(x_lin_abs, xd_lin_abs, color=COLORS["lin"], linestyle="--", label="Linearised")
axes[3].scatter([x_eq], [0.0], color=COLORS["eq"], marker="*", s=120, zorder=5)
axes[3].set_xlabel("$x$ [m]");  axes[3].set_ylabel("$\\dot{x}$ [m/s]")
axes[3].set_title("Phase Portrait"); axes[3].legend(fontsize=8); axes[3].grid(True)

fig5.suptitle(
    "Open-Loop Nonlinear vs Linearised System Response\n"
    f"(Perturbation $\\Delta x_0 = {dx0*100:.0f}$ cm at $x_{{eq}}={x_eq}$ m,  "
    f"$V = V_{{eq}} = {V_eq:.2f}$ V)",
    fontsize=13, y=1.01,
)
save_fig(fig5, "fig5_summary")

print("\nAll figures saved.  Analysis complete.")