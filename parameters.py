"""
parameters.py
=============
Units (SI unless noted):
    mass        [kg]
    length      [m]
    stiffness   [N/m]
    damping     [Ns/m]
    resistance  [Ohm]
    inductance  [H]
    angle       [rad]
    time const  [s]
"""

import numpy as np


m      = 0.462          # ball mass [kg]
r      = 0.123          # ball radius [m]
g      = 9.81           # gravitational acceleration [m/s²]
phi    = np.deg2rad(41) # incline angle [rad]
m_eff  = 1.4 * m        # effective mass [kg]
k      = 1885.0         # spring stiffness [N/m]
b      = 10.4           # damper coefficient [Ns/m]
d      = 0.42           # spring natural (unstretched) length / attachment point [m]
delta  = 0.65           # electromagnet fixed position [m]
c      = 6.811e-3       # magnetic force constant [N·m²/A²]
R      = 2200.0         # coil resistance [Ω]
L0     = 0.125          # baseline inductance [H]
L1     = 0.0241         # position-dependent inductance coefficient [H]
alpha  = 1.2            # inductance spatial decay constant [m⁻¹]
tau_m  = 0.030          # measurement filter time constant [s]
x_eq   = 0.50           # target equilibrium ball position [m]