"""
controller.py
=============
PID controller design and analysis for the inclined-plane magnetic-ball system.

The PID is designed on the linearised plant G(s) = C*(sI-A)^{-1}*B obtained
from linearisation.py.  All analysis (transfer functions, stability margins,
step-response metrics) is done with the `control` library.

"""

import numpy as np
import control
from linearisation import get_linearised_system


#  Default PID gains 

DEFAULT_KP = 50_000.0    # [V/m]
DEFAULT_KI = 15_000.0    # [V/(m·s)]
DEFAULT_KD =  2_000.0    # [V·s/m]
DEFAULT_N  =     80.0    # [rad/s]  derivative filter pole


class PIDController:
    """
    PID controller analysis class for the magnetic-ball plant.

    Parameters
    ----------
    Kp, Ki, Kd : float
        Proportional, integral, derivative gains.
    N : float
        Derivative filter pole (rad/s).  The derivative term is realised as
        Kd·N·s / (s + N),  equivalent to a first-order lead with corner at N.
    A, B, C, D : ndarray, optional
        Linearised plant matrices.  If not supplied, they are imported
        automatically from linearisation.py.
    """

    def __init__(self,
                 Kp: float = DEFAULT_KP,
                 Ki: float = DEFAULT_KI,
                 Kd: float = DEFAULT_KD,
                 N:  float = DEFAULT_N,
                 A=None, B=None, C=None, D=None):    #The class constructor. The = DEFAULT_KP means: "if you don't provide a value for Kp, use DEFAULT_KP". So PIDController() creates a controller with the tuned gains, but PIDController(Kp=30000) overrides just Kp.

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.N  = N

        # Load plant matrices from linearisation if not provided
        if A is None:
            matrices = get_linearised_system()
            self.A, self.B, self.C, self.D = matrices[0], matrices[1], matrices[2], matrices[3]
            self.x_eq, self.i_eq, self.V_eq = matrices[4], matrices[5], matrices[6]
            self.plant_ss = matrices[7]
        else:
            self.A, self.B, self.C, self.D = A, B, C, D
            self.x_eq = None
            self.V_eq = None
            self.plant_ss = control.ss(A, B, C, D)

        # Build transfer functions
        self._plant_tf  = None   # computed lazily
        self._pid_tf    = None
        self._loop_tf   = None
        self._cl_tf     = None

    #  Transfer-function construction 

    def plant_tf(self) -> control.TransferFunction:
        """Open-loop plant transfer function G(s) = C*(sI–A)^{-1}*B."""
        if self._plant_tf is None:
            self._plant_tf = control.tf(self.plant_ss)
        return self._plant_tf

    def pid_tf(self) -> control.TransferFunction:
        """
        PID transfer function (error → control action, deviation form):
            C(s) = Kp + Ki/s + Kd·N·s/(s+N)
        """
        if self._pid_tf is None:
            s = control.tf('s')
            self._pid_tf = (self.Kp
                            + self.Ki / s
                            + self.Kd * self.N * s / (s + self.N))
        return self._pid_tf

    def loop_tf(self) -> control.TransferFunction:
        """Open-loop transfer function  L(s) = C(s)·G(s)."""
        if self._loop_tf is None:
            self._loop_tf = self.pid_tf() * self.plant_tf()
        return self._loop_tf

    def closed_loop_tf(self) -> control.TransferFunction:
        """
        Closed-loop transfer function  T(s) = L(s) / (1 + L(s)).
        Uses control.feedback for numerical robustness.
        """
        if self._cl_tf is None:
            self._cl_tf = control.feedback(self.loop_tf(), 1)
        return self._cl_tf

    #  Stability analysis 

    def stability_margins(self) -> dict:
        """
        Compute gain margin (GM), phase margin (PM), and crossover frequencies
        using the control library.

        Important note for unstable plants
        ------------------------------------
        The open-loop plant G(s) has one unstable pole (+9.85 rad/s).  The
        standard Bode-based gain/phase margin analysis assumes a stable open-loop
        transfer function.  `control.margin()` may therefore report an incorrect
        (negative) gain margin even when the *closed-loop* is provably stable.

        The definitive stability test for this system is the Nyquist criterion:
            N = Z − P
        where N = number of encirclements of −1, Z = number of RHP closed-loop
        zeros (poles), P = number of RHP open-loop poles (= 1 here).
        For closed-loop stability we need Z = 0, so the Nyquist plot must
        encircle −1 exactly once in the *clockwise* direction.

        We therefore report:
          - closed_loop_stable: True if all closed-loop poles have negative real parts
          - pm: phase margin from Bode (indicative only)
          - gm / gm_dB: gain margin from Bode (may be misleading; treat with caution)

        Returns
        -------
        dict with keys: 'gm', 'pm', 'wg', 'wp', 'gm_dB',
                        'closed_loop_stable', 'stable'
        """
        gm, pm, wg, wp = control.margin(self.loop_tf())
        cl_poles = self.closed_loop_poles()
        cl_stable = bool(np.all(np.array([p.real for p in cl_poles]) < 0))
        return {
            "gm":               gm,
            "gm_dB":            20 * np.log10(abs(gm)) if gm and gm > 0 else float('nan'),
            "pm":               pm,
            "wg":               wg,
            "wp":               wp,
            "closed_loop_stable": cl_stable,
            # 'stable' kept for backwards-compat; uses closed-loop criterion
            "stable":           cl_stable,
        }

    def closed_loop_poles(self) -> np.ndarray:
        """Poles (eigenvalues) of the closed-loop system."""
        return control.poles(self.closed_loop_tf())

    def open_loop_poles(self) -> np.ndarray:
        """Poles of the open-loop plant (eigenvalues of A)."""
        return np.linalg.eigvals(self.A)

    # Step-response metrics

    def step_response_metrics(self, t_end: float = 2.0, dt: float = 1e-4) -> dict:
        """
        Simulate the linearised closed-loop step response and compute standard
        performance metrics.

        Parameters
        ----------
        t_end : float  – simulation duration [s]
        dt    : float  – time step [s]

        Returns
        -------
        dict with: 'rise_time', 'settling_time', 'overshoot_pct', 'peak_time',
                   'steady_state_error', 't', 'y'
        """
        t_arr = np.arange(0.0, t_end, dt)
        response = control.step_response(self.closed_loop_tf(), T=t_arr)
        t = response.t
        y = response.outputs.flatten()

        # Final value (should be 1 for a unity-gain stable closed-loop)
        y_ss = y[-1]

        # Overshoot
        y_max    = np.max(y)
        overshoot = (y_max - y_ss) / y_ss * 100 if y_ss != 0 else np.nan

        # Rise time (10 %→90 % of final value)
        try:
            t_10 = t[np.where(y >= 0.10 * y_ss)[0][0]]
            t_90 = t[np.where(y >= 0.90 * y_ss)[0][0]]
            rise_time = t_90 - t_10
        except IndexError:
            rise_time = np.nan

        # Settling time (within ±2 % of final value)
        settle_band = 0.02 * abs(y_ss)
        settled = np.where(np.abs(y - y_ss) > settle_band)[0]
        settling_time = t[settled[-1]] if len(settled) > 0 else 0.0

        # Peak time
        peak_time = t[np.argmax(y)]

        return {
            "rise_time":          rise_time,
            "settling_time":      settling_time,
            "overshoot_pct":      overshoot,
            "peak_time":          peak_time,
            "steady_state_error": abs(1.0 - y_ss),
            "t":                  t,
            "y":                  y,
        }

    #  Reporting 

    def print_report(self):
        """Print a concise design report to stdout."""
        sep = "=" * 66
        print(sep)
        print("PID Controller Design Report")
        print(sep)
        print(f"  Kp = {self.Kp:.2f}  V/m")
        print(f"  Ki = {self.Ki:.2f}  V/(m·s)")
        print(f"  Kd = {self.Kd:.2f}  V·s/m")
        print(f"  N  = {self.N:.2f}   rad/s  (derivative filter pole)")

        print(f"\nTransfer function C(s)·G(s) [loop gain]:")
        print(self.loop_tf())

        print("\nOpen-loop plant poles:")
        for p in self.open_loop_poles():
            tag = "UNSTABLE" if p.real > 0 else "stable"
            print(f"  {p:+.4f}  [{tag}]")

        print("\nClosed-loop poles:")
        for p in self.closed_loop_poles():
            tag = "UNSTABLE" if p.real > 0 else "stable"
            print(f"  {p:+.4f}  [{tag}]")

        margins = self.stability_margins()
        print(f"\nStability margins:")
        print(f"  Gain  margin  = {margins['gm_dB']:.2f} dB  (|GM| = {margins['gm']:.3f})")
        print(f"  Phase margin  = {margins['pm']:.2f} °")
        print(f"  Gain  crossover  ωg = {margins['wg']:.3f} rad/s")
        print(f"  Phase crossover  ωp = {margins['wp']:.3f} rad/s")
        print(f"  Stable: {margins['stable']}")

        metrics = self.step_response_metrics()
        print(f"\nLinearised closed-loop step-response metrics:")
        print(f"  Rise time     = {metrics['rise_time']:.4f}  s")
        print(f"  Settling time = {metrics['settling_time']:.4f}  s  (±2 %)")
        print(f"  Overshoot     = {metrics['overshoot_pct']:.2f} %")
        print(f"  Peak time     = {metrics['peak_time']:.4f}  s")
        print(f"  SS error      = {metrics['steady_state_error']:.6f}  (normalised)")
        print(sep)


#  Module-level convenience 

def build_default_controller() -> PIDController:
    """Return a PIDController with the default well-tuned gains."""
    ctrl = PIDController()
    return ctrl


if __name__ == "__main__":
    pid = build_default_controller()
    pid.print_report()
