"""
nonlinear_dynamics.py
=====================
Full nonlinear state-space model of the inclined-plane magnetic-ball system.

"""

import numpy as np
import parameters as p


#  Helper functions 

def inductance(x: float) -> float:
    """Position-dependent inductance L(x) [H]."""
    return p.L0 + p.L1 * np.exp(-p.alpha * (p.delta - x))


def d_inductance_dx(x: float) -> float:
    """Spatial derivative dL/dx [H/m].""" #back EMF term in electrical ODE
    return p.L1 * p.alpha * np.exp(-p.alpha * (p.delta - x))


def magnetic_force(x: float, i: float) -> float:
    """
    Attractive magnetic force on the ball [N].
    Acts in the +x direction (toward the electromagnet at x = delta).
    Undefined / singular when x >= delta; caller must stay in valid range.
    """
    gap = p.delta - x #distance between ball and magnet
    return p.c * i**2 / gap**2


# Nonlinear ODE class 
class MagneticBallSystem:
    """
    Encapsulates the nonlinear dynamics of the inclined magnetic-ball system.

    Usage
    -----
    >>> sys = MagneticBallSystem()
    >>> dxdt = sys.dynamics(t, state, V)
    """

    def __init__(self): #_init__ is a special function that runs automatically when you create the object (MagneticBallSystem()). It copies all parameters from parameters.py into the object itself. self refers to the object being created — self.m_eff means "store this value inside this particular object".
        # Cache parameters for speed
        self.m_eff  = p.m_eff
        self.m      = p.m
        self.g      = p.g
        self.phi    = p.phi
        self.k      = p.k
        self.b      = p.b
        self.d      = p.d
        self.delta  = p.delta
        self.R      = p.R
        self.tau_m  = p.tau_m

    def dynamics(self, t: float, state: np.ndarray, V: float) -> np.ndarray: #takes the current state of the system and returns how fast each state is changing right now
        """
        Compute the time derivative of the state vector.

        Parameters
        ----------
        t     : current time (required by solve_ivp, not used directly)
        state : [x, x_dot, i, x_m]
        V     : applied voltage input [V]

        Returns
        -------
        dstate/dt : numpy array of shape (4,)
        """
        x, x_dot, i, x_m = state

        # Guard against singularity: ball must not touch/pass the electromagnet
        gap = self.delta - x
        if gap <= 0:
            raise ValueError( #stops the program and shows an error message
                f"Ball position x={x:.4f} >= delta={self.delta:.4f}: "
                "physical singularity reached."
            )

        # --- Mechanical ---
        L   = inductance(x)
        dL  = d_inductance_dx(x)
        F_m = magnetic_force(x, i)

        # Net force along incline (positive = away from fixed wall, toward magnet)
        F_net = (
            F_m
            - self.k * (x - self.d)
            - self.b * x_dot
            - self.m * self.g * np.sin(self.phi)
        )
        x_ddot = F_net / self.m_eff

        # --- Electrical ---
        # di/dt = (V - R*i - dL/dx * x_dot * i) / L
        i_dot = (V - self.R * i - dL * x_dot * i) / L

        # --- Sensor ---
        xm_dot = (x - x_m) / self.tau_m #If the ball is ahead of where the sensor thinks it is (x > x_m), the sensor reading increases

        return np.array([x_dot, x_ddot, i_dot, xm_dot])

    def dynamics_callable(self, V_func):
        """
        Return a callable f(t, state) with the input voltage provided by V_func(t).
        Useful for passing directly to scipy.integrate.solve_ivp.

        Parameters
        ----------
        V_func : callable(t) -> float
        """
        def f(t, state):
            return self.dynamics(t, state, V_func(t))
        return f

    def equilibrium_state(self, x_eq: float, i_eq: float) -> np.ndarray:
        """
        Return the equilibrium state vector [x_eq, 0, i_eq, x_eq].
        (Velocity and sensor output are zero/matched at equilibrium.)
        """
        return np.array([x_eq, 0.0, i_eq, x_eq])
