"""
test_scenarios.py
=================
Main execution script for the Control Design and Testing phase.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import parameters as p
from controller import build_default_controller
from simulation_engine import (
    ClosedLoopSimulation,
    MismatchedClosedLoopSimulation,
    impulse_velocity_disturbance,
    equilibrium_current_at,
)

# Global Matplotlib style

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "lines.linewidth":   1.8,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLORS = {
    "position":  "#1f77b4",
    "sensor":    "#ff7f0e",
    "setpoint":  "#2ca02c",
    "voltage":   "#9467bd",
    "current":   "#8c564b",
    "error":     "#d62728",
    "nominal":   "#1f77b4",
    "mismatch":  "#d62728",
    "lin":       "#7f7f7f",
}

X_SP       = p.x_eq   # set-point = 0.50 m
X0_NOMINAL = 0.48     # starting position for Scenarios 1 & 3 [m]


def _save(fig, name: str):
    path = f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", format="pdf")
    print(f"  Saved → {path}")
    plt.close(fig)



#  CONTROLLER REPORT + BODE / STABILITY MARGIN PLOTS


def bode_and_stability_plots(pid):
    import control

    print("\n" + "=" * 66)
    print("Bode / Stability Margin Plots")
    print("=" * 66)

    L  = pid.loop_tf()
    T  = pid.closed_loop_tf()

    margins = pid.stability_margins()
    print(f"  GM = {margins['gm_dB']:.2f} dB   PM = {margins['pm']:.2f}°")
    print(f"  ωg = {margins['wg']:.3f} rad/s   ωp = {margins['wp']:.3f} rad/s")
    print(f"  Closed-loop stable: {margins['stable']}")

    omega = np.logspace(-1, 4, 3000)

    import control as ctrl
    mag_L, phase_L, _ = ctrl.bode(L, omega, plot=False)
    mag_L_dB    = 20 * np.log10(np.maximum(np.abs(mag_L.flatten()), 1e-15))
    phase_L_deg = np.degrees(
        np.angle(np.array([ctrl.evalfr(L, 1j * w) for w in omega]).flatten())
    )

    mag_T, _, _ = ctrl.bode(T, omega, plot=False)
    mag_T_dB = 20 * np.log10(np.maximum(np.abs(mag_T.flatten()), 1e-15))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    fig.subplots_adjust(hspace=0.45)

    ax1, ax2, ax3 = axes

    ax1.semilogx(omega, mag_L_dB, color=COLORS["nominal"], linewidth=1.8)
    ax1.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    if margins['wg'] and np.isfinite(margins['wg']):
        ax1.axvline(margins['wg'], color="orange", linewidth=1.2, linestyle="--",
                    label=f"Gain xover $\\omega_g={margins['wg']:.1f}$ rad/s")
    if margins['wp'] and np.isfinite(margins['wp']):
        ax1.axvline(margins['wp'], color="red", linewidth=1.2, linestyle="--",
                    label=f"Phase xover $\\omega_p={margins['wp']:.1f}$ rad/s")
    ax1.set_ylabel(r"$|L(j\omega)|$  [dB]")
    ax1.set_title(r"Bode Diagram — Open-Loop $L(s)=C(s)\,G(s)$")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both")
    ax1.text(0.97, 0.05,
             f"GM = {margins['gm_dB']:.1f} dB\nPM = {margins['pm']:.1f}°",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    ax2.semilogx(omega, phase_L_deg, color=COLORS["position"], linewidth=1.8)
    ax2.axhline(-180, color="grey", linewidth=0.8, linestyle=":")
    ax2.set_ylabel(r"$\angle L(j\omega)$  [°]")
    ax2.set_xlabel("Frequency  [rad/s]")
    ax2.grid(True, which="both")

    ax3.semilogx(omega, mag_T_dB, color=COLORS["sensor"], linewidth=1.8)
    ax3.axhline(-3, color="grey", linewidth=0.8, linestyle=":", label="−3 dB")
    ax3.set_ylabel(r"$|T(j\omega)|$  [dB]")
    ax3.set_xlabel("Frequency  [rad/s]")
    ax3.set_title(r"Closed-Loop Frequency Response $T(s)$")
    ax3.legend(fontsize=8)
    ax3.grid(True, which="both")

    _save(fig, "scenario0_bode_stability")
    return margins



#  SCENARIO 1 – Nominal Set-Point Tracking


def scenario_1(pid):
    print("\n" + "=" * 66)
    print("Scenario 1: Nominal Set-Point Tracking")
    print(f"  x0 = {X0_NOMINAL:.2f} m  →  x_sp = {X_SP:.2f} m")
    print("=" * 66)

    sim   = ClosedLoopSimulation(pid, x_sp=X_SP)
    i0    = equilibrium_current_at(X0_NOMINAL)
    z0    = sim.initial_state(x0=X0_NOMINAL, xdot0=0.0, i0=i0)
    t_end = 2.0
    t_eval = np.linspace(0, t_end, 8000)

    print(f"  Initial current at x0: i0 = {i0:.4f} A")
    print("  Running nonlinear simulation …")
    res = sim.run(z0, (0.0, t_end), t_eval)

    # Linearised step response (deviation model)
    dx_step      = X_SP - X0_NOMINAL
    lin_metrics  = pid.step_response_metrics(t_end=t_end, dt=1e-4)
    x_lin_abs    = X0_NOMINAL + lin_metrics['y'] * dx_step
    t_lin        = lin_metrics['t']

    # Settling time (nonlinear, ±2 mm)
    xm, t_arr = res['x_m'], res['t']
    within = np.where(np.abs(xm - X_SP) <= 0.002)[0]
    if len(within) > 0:
        # Find index where it enters and stays
        for start in within:
            if np.all(np.abs(xm[start:] - X_SP) <= 0.002):
                t_settle_nl = t_arr[start]
                break
        else:
            t_settle_nl = t_arr[within[0]]
    else:
        t_settle_nl = None

    ss_err_mm = abs(xm[-1] - X_SP) * 1e3
    print(f"  Final x     = {res['x'][-1]*1e3:.2f} mm  (target {X_SP*1e3:.0f} mm)")
    print(f"  SS error    = {ss_err_mm:.3f} mm")
    if t_settle_nl is not None:
        print(f"  Settling (±2 mm, nonlinear) = {t_settle_nl:.4f} s")

    #  Figure 
    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    fig.subplots_adjust(hspace=0.42)
    ax1, ax2, ax3 = axes

    ax1.plot(res['t'], res['x'] * 1e2,   color=COLORS["position"],
             label="Ball position $x(t)$")
    ax1.plot(res['t'], res['x_m'] * 1e2, color=COLORS["sensor"],
             linestyle="--", label="Sensor output $x_m(t)$")
    ax1.plot(t_lin,   x_lin_abs * 1e2,   color=COLORS["lin"],
             linestyle=":", linewidth=1.4, label="Linearised model")
    ax1.axhline(X_SP * 1e2, color=COLORS["setpoint"], linewidth=1.2,
                linestyle="-.", label=f"Set-point $x_{{sp}}={X_SP*1e2:.0f}$ cm")
    if t_settle_nl is not None:
        ax1.axvline(t_settle_nl, color="grey", linewidth=0.9, linestyle=":")
        ax1.text(t_settle_nl + 0.02, X0_NOMINAL * 1e2 + 0.3,
                 f"$t_{{s}}={t_settle_nl:.3f}$ s", fontsize=8, color="grey")
    ax1.set_ylabel("Position  [cm]")
    ax1.set_title("Scenario 1 — Nominal Set-Point Tracking")
    ax1.legend(loc="lower right", ncol=2)
    ax1.grid(True)

    ax2.plot(res['t'], res['error'] * 1e3, color=COLORS["error"],
             label="Error $e(t) = x_{sp} - x_m(t)$")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(+2, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.axhline(-2, color="grey", linewidth=0.8, linestyle=":", alpha=0.6,
                label="±2 mm band")
    ax2.set_ylabel("Error  [mm]")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    ax3.plot(res['t'], res['V'] * 1e-3, color=COLORS["voltage"],
             label="Applied voltage $V(t)$")
    ax3.axhline(sim.V_eq * 1e-3, color="grey", linewidth=1, linestyle=":",
                label=f"$V_{{eq}}={sim.V_eq*1e-3:.1f}$ kV")
    ax3.set_xlabel("Time  [s]")
    ax3.set_ylabel("Voltage  [kV]")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    _save(fig, "scenario1_setpoint_tracking")
    return res, lin_metrics, t_settle_nl, ss_err_mm



#  SCENARIO 2 – Disturbance Rejection


def scenario_2(pid):
    T_KICK   = 0.80   # [s]   time of disturbance
    DURATION = 0.05   # [s]   push duration
    DELTA_V  = 0.05   # [m/s] velocity change (positive = toward magnet)

    print("\n" + "=" * 66)
    print("Scenario 2: Disturbance Rejection")
    print(f"  Ball settled at x_sp = {X_SP:.2f} m")
    print(f"  Impulse push: Δv = {DELTA_V} m/s at t = {T_KICK} s")
    print("=" * 66)

    dist_fn = impulse_velocity_disturbance(T_KICK, DURATION, DELTA_V)
    sim     = ClosedLoopSimulation(pid, x_sp=X_SP, disturbance_fn=dist_fn)

    # Start already settled at x_sp with i_eq
    i_eq_sp = equilibrium_current_at(X_SP)
    z0      = sim.initial_state(x0=X_SP, xdot0=0.0, i0=i_eq_sp)
    t_end   = 2.5
    t_eval  = np.linspace(0, t_end, 8000)

    print("  Running disturbance-rejection simulation …")
    res = sim.run(z0, (0.0, t_end), t_eval)

    # Measure recovery time (return within 1 mm after kick ends)
    post_kick  = res['t'] >= T_KICK + DURATION
    t_post     = res['t'][post_kick]
    e_post     = np.abs(res['error'][post_kick])
    within_1mm = np.where(e_post <= 1e-3)[0]
    if len(within_1mm) > 0:
        t_recover = t_post[within_1mm[0]] - (T_KICK + DURATION)
        print(f"  Recovery to within 1 mm: {t_recover*1e3:.1f} ms")
    else:
        t_recover = None
        print("  Ball did not recover within 1 mm in the simulation window.")

    ss_err_mm = abs(res['x_m'][-1] - X_SP) * 1e3
    print(f"  SS error after disturbance = {ss_err_mm:.3f} mm")

    #  Figure 
    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    fig.subplots_adjust(hspace=0.42)
    ax1, ax2, ax3 = axes

    ax1.plot(res['t'], res['x'] * 1e2,   color=COLORS["position"],
             label="Ball position $x(t)$")
    ax1.plot(res['t'], res['x_m'] * 1e2, color=COLORS["sensor"],
             linestyle="--", label="Sensor output $x_m(t)$")
    ax1.axhline(X_SP * 1e2, color=COLORS["setpoint"], linewidth=1.2,
                linestyle="-.", label=f"Set-point $x_{{sp}}={X_SP*1e2:.0f}$ cm")
    ax1.axvspan(T_KICK, T_KICK + DURATION, color="tomato", alpha=0.25,
                label=f"Push ($\\Delta v={DELTA_V}$ m/s, {DURATION*1e3:.0f} ms)")
    ax1.set_ylabel("Position  [cm]")
    ax1.set_title("Scenario 2 — Disturbance Rejection")
    ax1.legend(loc="lower right", ncol=2)
    ax1.grid(True)

    ax2.plot(res['t'], res['error'] * 1e3, color=COLORS["error"],
             label="Error $e(t) = x_{sp} - x_m(t)$")
    ax2.axvspan(T_KICK, T_KICK + DURATION, color="tomato", alpha=0.20)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(+1, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.axhline(-1, color="grey", linewidth=0.8, linestyle=":", alpha=0.6,
                label="±1 mm band")
    ax2.set_ylabel("Error  [mm]")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    ax3.plot(res['t'], res['V'] * 1e-3, color=COLORS["voltage"],
             label="Applied voltage $V(t)$")
    ax3.axhline(sim.V_eq * 1e-3, color="grey", linewidth=1, linestyle=":",
                label=f"$V_{{eq}}={sim.V_eq*1e-3:.1f}$ kV")
    ax3.axvspan(T_KICK, T_KICK + DURATION, color="tomato", alpha=0.20)
    ax3.set_xlabel("Time  [s]")
    ax3.set_ylabel("Voltage  [kV]")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    if t_recover is not None:
        t_ann = T_KICK + DURATION + t_recover
        ax1.annotate(
            f"Recovery: {t_recover*1e3:.0f} ms",
            xy=(t_ann, X_SP * 1e2),
            xytext=(t_ann + 0.15, X_SP * 1e2 - 0.15),
            arrowprops=dict(arrowstyle="->", color="grey"),
            fontsize=8, color="grey",
        )

    _save(fig, "scenario2_disturbance_rejection")
    return res, t_recover, ss_err_mm



#  SCENARIO 3 – Sensitivity / Robustness Analysis


def scenario_3(pid):
    M_FACTOR = 1.15   # +15 % mass
    R_FACTOR = 1.10   # +10 % resistance

    print("\n" + "=" * 66)
    print("Scenario 3: Sensitivity / Robustness Analysis")
    print(f"  Nominal:   m = {p.m:.3f} kg,  R = {p.R:.0f} Ω")
    print(f"  Perturbed: m = {p.m*M_FACTOR:.3f} kg (+15 %), "
          f"R = {p.R*R_FACTOR:.0f} Ω (+10 %)")
    print("=" * 66)

    # Nominal simulation: 2 s is sufficient
    t_end_nom  = 2.0
    t_eval_nom = np.linspace(0, t_end_nom, 8000)
    # Mismatched simulation: integrator convergence is slower (larger gravity
    # force offset because m is 15 % heavier), so run for 15 s to show the
    # integral action eventually bringing the error to zero.
    t_end_mis  = 15.0
    t_eval_mis = np.linspace(0, t_end_mis, 30000)
    i0         = equilibrium_current_at(X0_NOMINAL)

    #  Nominal 
    sim_nom = ClosedLoopSimulation(pid, x_sp=X_SP)
    z0_nom  = sim_nom.initial_state(x0=X0_NOMINAL, xdot0=0.0, i0=i0)
    print("  Running nominal simulation …")
    res_nom = sim_nom.run(z0_nom, (0.0, t_end_nom), t_eval_nom)
    ss_nom  = abs(res_nom['x_m'][-1] - X_SP) * 1e3
    print(f"  Nominal final x_m = {res_nom['x_m'][-1]*1e3:.2f} mm, SS error = {ss_nom:.3f} mm")

    #  Mismatched 
    sim_mis = MismatchedClosedLoopSimulation(
        pid, x_sp=X_SP, m_factor=M_FACTOR, R_factor=R_FACTOR,
    )
    # Use same initial position and current (controller uses nominal V_eq)
    z0_mis = sim_mis.initial_state(x0=X0_NOMINAL, xdot0=0.0, i0=i0)
    print("  Running mismatched simulation (15 s, slower integral convergence) …")
    res_mis = sim_mis.run(z0_mis, (0.0, t_end_mis), t_eval_mis)
    ss_mis  = abs(res_mis['x_m'][-1] - X_SP) * 1e3
    print(f"  Mismatched final x_m = {res_mis['x_m'][-1]*1e3:.2f} mm, SS error = {ss_mis:.3f} mm")

    # Find settling time for mismatched (±5 mm, since convergence is slow)
    for band_mm in [5.0, 10.0, 20.0]:
        within = np.where(np.abs(res_mis['x_m'] - X_SP) <= band_mm*1e-3)[0]
        if len(within) > 0:
            t_settle_mis = res_mis['t'][within[0]]
            print(f"  Mismatched settling (±{band_mm:.0f} mm): {t_settle_mis:.2f} s")
            break

    #  Figure 3a: Full 3-panel comparison 
    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    fig.subplots_adjust(hspace=0.42)
    ax1, ax2, ax3 = axes

    ax1.plot(res_nom['t'], res_nom['x'] * 1e2,
             color=COLORS["nominal"], label="Nominal plant")
    ax1.plot(res_mis['t'], res_mis['x'] * 1e2,
             color=COLORS["mismatch"], linestyle="--",
             label=f"Mismatched ($m\\!\\times\\!{M_FACTOR}$, $R\\!\\times\\!{R_FACTOR}$)")
    ax1.axhline(X_SP * 1e2, color=COLORS["setpoint"], linewidth=1.2,
                linestyle="-.", label=f"$x_{{sp}}={X_SP*1e2:.0f}$ cm")
    ax1.set_ylabel("Position  [cm]")
    ax1.set_title("Scenario 3 — Robustness to Parameter Uncertainty")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    ax2.plot(res_nom['t'], res_nom['error'] * 1e3,
             color=COLORS["nominal"],  label="Nominal")
    ax2.plot(res_mis['t'], res_mis['error'] * 1e3,
             color=COLORS["mismatch"], linestyle="--", label="Mismatched")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Error  [mm]")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    ax3.plot(res_nom['t'], res_nom['V'] * 1e-3,
             color=COLORS["nominal"],  label="Nominal $V(t)$")
    ax3.plot(res_mis['t'], res_mis['V'] * 1e-3,
             color=COLORS["mismatch"], linestyle="--", label="Mismatched $V(t)$")
    ax3.axhline(sim_nom.V_eq * 1e-3, color="grey", linewidth=1, linestyle=":",
                label=f"$V_{{eq}}={sim_nom.V_eq*1e-3:.1f}$ kV")
    ax3.set_xlabel("Time  [s]")
    ax3.set_ylabel("Voltage  [kV]")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    _save(fig, "scenario3_robustness")

    #  Figure 3b: Transient error close-up 
    fig2, ax = plt.subplots(figsize=(7, 4.2))
    t_cut = 0.6
    mn = res_nom['t'] <= t_cut
    mm = res_mis['t'] <= t_cut
    ax.plot(res_nom['t'][mn], res_nom['error'][mn] * 1e3,
            color=COLORS["nominal"], label="Nominal")
    ax.plot(res_mis['t'][mm], res_mis['error'][mm] * 1e3,
            color=COLORS["mismatch"], linestyle="--",
            label=f"Mismatched ($m\\!\\times\\!{M_FACTOR}$, $R\\!\\times\\!{R_FACTOR}$)")
    ax.axhline(0,  color="black", linewidth=0.8)
    ax.axhline(+2, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axhline(-2, color="grey", linewidth=0.8, linestyle=":", alpha=0.6,
               label="±2 mm band")
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Tracking error  [mm]")
    ax.set_title("Scenario 3 — Transient Error Detail  (0–0.6 s)")
    ax.legend()
    ax.grid(True)
    _save(fig2, "scenario3_error_closeup")

    return res_nom, res_mis, ss_nom, ss_mis


#  SUMMARY FIGURE  (2×2 grid across all scenarios)


def summary_figure(res1, res2, res3_nom, res3_mis, pid):
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]

    # (0,0) Scenario 1 – Position
    ax = axes[0]
    ax.plot(res1['t'], res1['x'] * 1e2,   color=COLORS["position"],
            label="Ball $x(t)$")
    ax.plot(res1['t'], res1['x_m'] * 1e2, color=COLORS["sensor"],
            linestyle="--", linewidth=1.2, label="Sensor $x_m(t)$")
    ax.axhline(X_SP * 1e2, color=COLORS["setpoint"], linewidth=1,
               linestyle="-.", label=f"$x_{{sp}}={X_SP*1e2:.0f}$ cm")
    ax.set_title("Sc. 1 — Set-Point Tracking")
    ax.set_xlabel("$t$ [s]");  ax.set_ylabel("Position [cm]")
    ax.legend(fontsize=8);     ax.grid(True)

    # (0,1) Scenario 1 – Error
    ax = axes[1]
    ax.plot(res1['t'], res1['error'] * 1e3, color=COLORS["error"])
    ax.axhline(0,  color="black", linewidth=0.8)
    ax.axhline(+2, color="grey",  linewidth=0.8, linestyle=":")
    ax.axhline(-2, color="grey",  linewidth=0.8, linestyle=":")
    ax.set_title("Sc. 1 — Tracking Error")
    ax.set_xlabel("$t$ [s]");  ax.set_ylabel("Error [mm]")
    ax.grid(True)

    # (1,0) Scenario 2 – Disturbance event (position)
    T_KICK, DUR = 0.80, 0.05
    ax = axes[2]
    ax.plot(res2['t'], res2['x'] * 1e2, color=COLORS["position"],
            label="Ball $x(t)$")
    ax.axhline(X_SP * 1e2, color=COLORS["setpoint"], linewidth=1,
               linestyle="-.", label=f"$x_{{sp}}={X_SP*1e2:.0f}$ cm")
    ax.axvspan(T_KICK, T_KICK + DUR, color="tomato", alpha=0.25,
               label="Push")
    ax.set_title("Sc. 2 — Disturbance Rejection")
    ax.set_xlabel("$t$ [s]");  ax.set_ylabel("Position [cm]")
    ax.legend(fontsize=8);     ax.grid(True)

    # (1,1) Scenario 3 – Robustness (position comparison)
    ax = axes[3]
    ax.plot(res3_nom['t'], res3_nom['x'] * 1e2,
            color=COLORS["nominal"], label="Nominal")
    ax.plot(res3_mis['t'], res3_mis['x'] * 1e2,
            color=COLORS["mismatch"], linestyle="--", label="Mismatched")
    ax.axhline(X_SP * 1e2, color=COLORS["setpoint"], linewidth=1,
               linestyle="-.", label=f"$x_{{sp}}={X_SP*1e2:.0f}$ cm")
    ax.set_title("Sc. 3 — Robustness")
    ax.set_xlabel("$t$ [s]");  ax.set_ylabel("Position [cm]")
    ax.legend(fontsize=8);     ax.grid(True)

    fig.suptitle(
        f"PID Controller — All Scenarios Summary\n"
        f"$K_p={pid.Kp:.0f}$, $K_i={pid.Ki:.0f}$, $K_d={pid.Kd:.0f}$, "
        f"$N={pid.N:.0f}$ rad/s  |  Set-point $x_{{sp}}={X_SP}$ m",
        fontsize=11, y=1.01,
    )
    _save(fig, "scenario_summary")


#  MAIN

if __name__ == "__main__":

    print("\n" + "#" * 66)
    print("#  Control Design and Testing — Full Test Suite")
    print(f"#  Set-point: x_sp = {X_SP} m")
    print("#" * 66)

    pid = build_default_controller()
    pid.print_report()

    margins              = bode_and_stability_plots(pid)
    res1, lin_m, ts1, e1 = scenario_1(pid)
    res2, t_rec, e2      = scenario_2(pid)
    res3_nom, res3_mis, e3_nom, e3_mis = scenario_3(pid)

    summary_figure(res1, res2, res3_nom, res3_mis, pid)

    #  Summary table 
    print("\n" + "=" * 66)
    print("Summary of Key Metrics")
    print("=" * 66)
    print(f"  PID gains:  Kp={pid.Kp:.0f}  Ki={pid.Ki:.0f}  "
          f"Kd={pid.Kd:.0f}  N={pid.N:.0f} rad/s")
    print(f"  GM = {margins['gm_dB']:.2f} dB   PM = {margins['pm']:.2f}°")
    print(f"  Linearised: rise={lin_m['rise_time']:.4f}s  "
          f"settle={lin_m['settling_time']:.4f}s  "
          f"OS={lin_m['overshoot_pct']:.1f}%")
    if ts1 is not None:
        print(f"  Nonlinear:  settling (±2mm) = {ts1:.4f} s")
    print(f"  Scenario 1 SS error  = {e1:.3f} mm")
    if t_rec is not None:
        print(f"  Scenario 2 recovery = {t_rec*1e3:.1f} ms  |  SS error = {e2:.3f} mm")
    print(f"  Scenario 3 SS error: nominal = {e3_nom:.3f} mm, "
          f"mismatched = {e3_mis:.3f} mm")
    print("\nAll figures saved as PDF.")
    print("=" * 66)