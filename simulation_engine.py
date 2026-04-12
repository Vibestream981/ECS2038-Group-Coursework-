"""
simulation_engine.py
====================
Closed-loop nonlinear simulation engine for the magnetic-ball system.

"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional

import parameters as p
from nonlinear_dynamics import MagneticBallSystem, inductance, d_inductance_dx
from controller import PIDController


#Physical equilibrium helpers 

def equilibrium_current_at(x0: float) -> float:
    """
    Compute the equilibrium coil current required to hold the ball stationary
    at position x0 on the incline (force balance, ẋ = 0).

    This is the physically correct initial current to use when starting a
    simulation from x0 ≠ x_eq.  Using the wrong (x_eq) current at x0 would
    create a large initial transient / singularity because the magnetic force
    scales as 1/(δ−x)², which varies greatly with position.

    Parameters
    ----------
    x0 : float – ball position [m], must satisfy 0 < x0 < delta

    Returns
    -------
    i0 : float – equilibrium current at x0 [A]
    """
    gap = p.delta - x0
    spring_force  = p.k * (x0 - p.d)
    gravity_force = p.m * p.g * np.sin(p.phi)
    required_fmag = spring_force + gravity_force
    if required_fmag <= 0:
        # Below the natural length the spring already assists gravity;
        # minimum current just to hold against gravity.
        required_fmag = gravity_force
    return float(np.sqrt(required_fmag * gap**2 / p.c))


# Saturation helper 

def _sat(u: float, lo: float, hi: float) -> float:
    return float(np.clip(u, lo, hi))


# Augmented closed-loop ODE 

class ClosedLoopSimulation:
    """
    Continuous-time closed-loop simulator.

    State vector  z_aug = [x, ẋ, i, x_m, σ, ξ]  (length 6)

        x    – ball position [m]
        ẋ    – ball velocity [m/s]
        i    – coil current  [A]
        x_m  – filtered sensor output [m]
        σ    – PID integral state (integrator) [m·s]
        ξ    – PID derivative filter state     [m]

    Parameters
    ----------
    controller : PIDController
    x_sp       : float  – set-point position [m]
    V_max      : float  – upper voltage limit [V].  Default = 2·V_eq.
    disturbance_fn : optional callable(t, z_aug) → ndarray shape (4,)
        Adds an additive force/torque term to the mechanical + electrical ODE
        (first 4 states only).
    """

    def __init__(self,
                 controller: PIDController,
                 x_sp: float,
                 V_max: Optional[float] = None,
                 disturbance_fn: Optional[Callable] = None):

        self.pid   = controller
        self.x_sp  = x_sp
        self.V_eq  = controller.V_eq
        self.V_max = V_max if V_max is not None else 2.0 * self.V_eq
        self.V_min = 0.0

        self.plant = MagneticBallSystem()
        self.disturbance_fn = disturbance_fn

        # Anti-windup tracking time constant
        Kd = controller.Kd
        Ki = controller.Ki
        self.Tt = float(np.sqrt(Kd / Ki)) if Ki > 0 else np.inf

    #  Build initial augmented state 

    def initial_state(self,
                      x0: float,
                      xdot0: float = 0.0,
                      i0: Optional[float] = None,
                      sigma0: float = 0.0,
                      xi0: Optional[float] = None) -> np.ndarray:
        """
        Construct the 6-element initial augmented state vector.

        Parameters
        ----------
        x0      : initial ball position [m]
        xdot0   : initial velocity [m/s]
        i0      : initial current [A].  Defaults to the equilibrium current AT
                  x0 (NOT the equilibrium current at x_sp).  Using the set-point
                  equilibrium current at a different position would create a huge
                  initial magnetic force and cause immediate divergence.
        sigma0  : initial integrator state [m·s]
        xi0     : initial derivative filter state [m]  (defaults to x0)
        """
        if i0 is None:
            i0 = equilibrium_current_at(x0)
        if xi0 is None:
            xi0 = x0
        return np.array([x0, xdot0, i0, x0, sigma0, xi0])

    # Compute PID output (deviation from V_eq) 

    def _pid_output(self, x_m: float, sigma: float, xi: float) -> tuple:
        """
        Returns (u_pid, V_total, V_clipped)
            u_pid    – raw PID output [V]
            V_total  – clipped total voltage [V]
            V_uncl   – unclipped total voltage (for anti-windup) [V]
        """
        pid = self.pid
        e = self.x_sp - x_m

        u_pid  = (pid.Kp * e
                  + pid.Ki * sigma
                  - pid.Kd * pid.N * (x_m - xi))

        V_uncl  = self.V_eq + u_pid
        V_clip  = _sat(V_uncl, self.V_min, self.V_max)
        return u_pid, V_clip, V_uncl

    # Augmented ODE RHS 

    def ode_rhs(self, t: float, z_aug: np.ndarray) -> np.ndarray:
        """
        Time derivative of the augmented state vector.

        Parameters
        ----------
        t     : current time [s]
        z_aug : [x, ẋ, i, x_m, σ, ξ]

        Returns
        -------
        dz/dt : ndarray shape (6,)
        """
        x, x_dot, i_c, x_m, sigma, xi = z_aug

        #  PID output 
        _, V_clip, V_uncl = self._pid_output(x_m, sigma, xi)

        #  Plant ODEs 
        plant_state = np.array([x, x_dot, i_c, x_m])
        dz_plant = self.plant.dynamics(t, plant_state, V_clip)

        # Optional external disturbance (additive to plant derivatives)
        if self.disturbance_fn is not None:
            dz_plant = dz_plant + self.disturbance_fn(t, z_aug)

        #  Controller states 
        e = self.x_sp - x_m

        # Anti-windup back-calculation
        aw_correction = (V_clip - V_uncl) / self.Tt if np.isfinite(self.Tt) else 0.0

        d_sigma = e + aw_correction     # integrator with anti-windup
        d_xi    = self.pid.N * (x_m - xi)  # derivative filter

        return np.concatenate([dz_plant, [d_sigma, d_xi]])

    #  Run simulation 

    def run(self,
            z0_aug: np.ndarray,
            t_span: tuple,
            t_eval: Optional[np.ndarray] = None,
            rtol: float = 1e-9,
            atol: float = 1e-11) -> dict:
        """
        Integrate the closed-loop augmented ODE.

        Parameters
        ----------
        z0_aug  : initial augmented state (length 6)
        t_span  : (t_start, t_end)
        t_eval  : time points at which to store the solution
        rtol, atol : solver tolerances

        Returns
        -------
        dict with keys:
            't'        – time array
            'x'        – ball position [m]
            'x_dot'    – ball velocity [m/s]
            'i'        – coil current  [A]
            'x_m'      – sensor output [m]
            'sigma'    – integrator state
            'xi'       – derivative filter state
            'V'        – applied voltage (clipped) [V]
            'u_pid'    – raw PID output [V]
            'error'    – tracking error x_sp – x_m [m]
            'solver'   – full scipy IVP result
        """
        if t_eval is None:
            t_eval = np.linspace(*t_span, 5000)

        # Terminal event: ball touches electromagnet (gap < 1 mm)
        def event_collision(t, z):
            return (p.delta - z[0]) - 0.001
        event_collision.terminal  = True
        event_collision.direction = -1

        # Terminal event: ball hits lower wall
        def event_wall(t, z):
            return z[0] - 0.005
        event_wall.terminal  = True
        event_wall.direction = -1

        sol = solve_ivp(
            self.ode_rhs,
            t_span,
            z0_aug,
            method="RK45",
            t_eval=t_eval,
            events=[event_collision, event_wall],
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if sol.status == 1:
            print(f"  [sim] Integration stopped by terminal event at t = {sol.t[-1]:.4f} s.")
        elif sol.status == -1:
            print(f"  [sim] Solver failed: {sol.message}")

        t    = sol.t
        x    = sol.y[0]
        xd   = sol.y[1]
        i_c  = sol.y[2]
        x_m  = sol.y[3]
        sig  = sol.y[4]
        xi   = sol.y[5]

        # Recompute voltage trace for plotting
        V_arr    = np.zeros_like(t)
        upid_arr = np.zeros_like(t)
        for k in range(len(t)):
            u_pid_k, V_k, _ = self._pid_output(x_m[k], sig[k], xi[k])
            V_arr[k]    = V_k
            upid_arr[k] = u_pid_k

        return {
            "t":      t,
            "x":      x,
            "x_dot":  xd,
            "i":      i_c,
            "x_m":    x_m,
            "sigma":  sig,
            "xi":     xi,
            "V":      V_arr,
            "u_pid":  upid_arr,
            "error":  self.x_sp - x_m,
            "solver": sol,
        }


#  Disturbance factory helpers 

def impulse_velocity_disturbance(t_kick: float,
                                 duration: float,
                                 delta_v: float) -> Callable:
    """
    Returns a disturbance function that applies a brief velocity impulse to
    the ball (i.e., an additive term to dẋ/dt) between t_kick and
    t_kick + duration.

    This models a short physical push, e.g., delta_v = +0.10 m/s over 0.05 s
    ≈ an instantaneous 0.1 m/s velocity change.
    """
    magnitude = delta_v / duration   # force per unit effective mass [m/s²]

    def _dist(t: float, z_aug: np.ndarray) -> np.ndarray:
        d = np.zeros(4)
        if t_kick <= t <= t_kick + duration:
            d[1] = magnitude     # add to ẋ equation
        return d

    return _dist


def parameter_mismatch_plant(m_factor: float = 1.15,
                              R_factor: float = 1.10) -> MagneticBallSystem:
    """
    Return a MagneticBallSystem with perturbed physical parameters to simulate
    model uncertainty (Scenario 3).

    Parameters
    ----------
    m_factor : multiplicative factor on ball mass m
    R_factor : multiplicative factor on coil resistance R
    """
    import copy

    perturbed = MagneticBallSystem()
    perturbed.m     = p.m * m_factor
    perturbed.m_eff = p.m_eff * m_factor   # m_eff = 1.4·m scales together
    perturbed.R     = p.R * R_factor
    return perturbed


class MismatchedClosedLoopSimulation(ClosedLoopSimulation):
    """
    Closed-loop simulation using a deliberately mismatched plant
    (for Scenario 3: sensitivity / robustness analysis).

    The PID gains are kept identical to the nominal design; only the
    simulation plant model is perturbed.
    """

    def __init__(self,
                 controller: PIDController,
                 x_sp: float,
                 V_max: Optional[float] = None,
                 m_factor: float = 1.15,
                 R_factor: float = 1.10):

        super().__init__(controller, x_sp, V_max)
        self.plant = parameter_mismatch_plant(m_factor, R_factor)
        self.m_factor = m_factor
        self.R_factor = R_factor