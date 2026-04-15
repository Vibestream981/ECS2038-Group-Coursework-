import numpy as np
import sympy as sp
import control


x, x_dot, i_sym, x_m = sp.symbols('x x_dot i x_m', real=True)
V_sym = sp.Symbol('V', real=True)


# Parameter symbols
m_s, m_eff_s, g_s, phi_s    = sp.symbols('m m_eff g phi',      positive=True)
k_s, b_s, d_s                = sp.symbols('k b d',              positive=True)
delta_s, c_s                 = sp.symbols('delta c',            positive=True)
R_s, L0_s, L1_s, alpha_s    = sp.symbols('R L0 L1 alpha',      positive=True)
tau_m_s                      = sp.Symbol('tau_m',               positive=True)

gap_s  = delta_s - x                                           # air-gap
L_s    = L0_s + L1_s * sp.exp(-alpha_s * gap_s)               # inductance
dLdx_s = sp.diff(L_s, x)                                      # dL/dx
F_mag_s = c_s * i_sym**2 / gap_s**2                           # magnetic force




#Symbolic state equations  f = [f1, f2, f3, f4]
f1 = x_dot

f2 = (F_mag_s
      - k_s * (x - d_s)
      - b_s * x_dot
      - m_s * g_s * sp.sin(phi_s)) / m_eff_s

f3 = (V_sym - R_s * i_sym - dLdx_s * x_dot * i_sym) / L_s

f4 = (x - x_m) / tau_m_s

f_vec = sp.Matrix([f1, f2, f3, f4])
state_vec = sp.Matrix([x, x_dot, i_sym, x_m])
input_vec = sp.Matrix([V_sym])



# Symbolic Jacobians
A_sym = f_vec.jacobian(state_vec)
B_sym = f_vec.jacobian(input_vec)

print("=" * 70)
print("Symbolic Jacobian  A = ∂f/∂x  (before substitution)")
print("=" * 70)
sp.pprint(A_sym, use_unicode=True)

print("\n" + "=" * 70)
print("Symbolic Jacobian  B = ∂f/∂u  (before substitution)")
print("=" * 70)
sp.pprint(B_sym, use_unicode=True)



import parameters as p
#Numerical parameter
param_subs = {
    m_s:      p.m,
    m_eff_s:  p.m_eff,
    g_s:      p.g,
    phi_s:    p.phi,
    k_s:      p.k,
    b_s:      p.b,
    d_s:      p.d,
    delta_s:  p.delta,
    c_s:      p.c,
    R_s:      p.R,
    L0_s:     p.L0,
    L1_s:     p.L1,
    alpha_s:  p.alpha,
    tau_m_s:  p.tau_m,
}



#Solve for equilibrium current and voltage 
x_eq_val = p.x_eq
gap_eq   = p.delta - x_eq_val

spring_force  = p.k * (x_eq_val - p.d)
gravity_force = p.m * p.g * np.sin(p.phi)
required_F_mag = spring_force + gravity_force

i_eq_sq  = required_F_mag * gap_eq**2 / p.c
if i_eq_sq < 0:
    raise ValueError(
        "Negative i_eq² – equilibrium is not physically realisable at "
        f"x_eq = {x_eq_val} m.  Check that F_mag can oppose spring + gravity."
    )
i_eq  = float(np.sqrt(i_eq_sq))
V_eq  = p.R * i_eq

print("\n" + "=" * 70)
print("Equilibrium Operating Point")
print("=" * 70)
print(f"  x_eq   = {x_eq_val:.4f}  m")
print(f"  x_dot  = 0.0000  m/s")
print(f"  i_eq   = {i_eq:.6f}  A")
print(f"  V_eq   = {V_eq:.4f}  V")
print(f"  (Spring force    = {spring_force:.4f} N)")
print(f"  (Gravity force   = {gravity_force:.4f} N)")
print(f"  (Required F_mag  = {required_F_mag:.4f} N)")




#Substitute equilibrium + parameters into Jacobians 
eq_subs = {
    x:      x_eq_val,
    x_dot:  0.0,
    i_sym:  i_eq,
    x_m:    x_eq_val,
    V_sym:  V_eq,
}
all_subs = {**param_subs, **eq_subs}

A_num_sym = A_sym.subs(all_subs)
B_num_sym = B_sym.subs(all_subs)





# Simplify and convert to float numpy arrays
A_num = np.array(A_num_sym.evalf(), dtype=float)
B_num = np.array(B_num_sym.evalf(), dtype=float)




# Output matrix: sensor reads x_m  (4th state, index 3)
C_num = np.array([[0.0, 0.0, 0.0, 1.0]])



# Direct feedthrough: none
D_num = np.array([[0.0]])

print("\n" + "=" * 70)
print("Linearised State-Space Matrices (numerical, evaluated at equilibrium)")
print("=" * 70)

np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("\nA =")
print(A_num)
print("\nB =")
print(B_num)
print("\nC =")
print(C_num)
print("\nD =")
print(D_num)







#Build control.StateSpace object
ss_sys = control.ss(A_num, B_num, C_num, D_num) #control.ss(...) creates a state-space system object from the 4 matrices.

print("\n" + "=" * 70)
print("control.StateSpace object")
print("=" * 70)
print(ss_sys)







# Eigenvalues of A → open-loop poles
eigenvalues = np.linalg.eigvals(A_num) #np.linalg.eigvals(A_num) computes the eigenvalues of the A matrix
print("\nOpen-loop poles (eigenvalues of A):")
for ev in eigenvalues:  # loops through each eigenvalue one by one
    stability = "stable" if ev.real < 0 else "UNSTABLE"
    print(f"  {ev:+.4f}   [{stability}]")






#Export for use by other modules
def get_linearised_system():
    """
    Return the linearised state-space matrices and equilibrium values.

    Returns
    -------
    A_num : ndarray (4×4)
    B_num : ndarray (4×1)
    C_num : ndarray (1×4)
    D_num : ndarray (1×1)
    x_eq_val : float  – equilibrium ball position [m]
    i_eq     : float  – equilibrium current [A]
    V_eq     : float  – equilibrium voltage [V]
    ss_sys   : control.StateSpace
    """
    return A_num, B_num, C_num, D_num, x_eq_val, i_eq, V_eq, ss_sys


if __name__ == "__main__":
    # All computation runs on import; this block is a no-op placeholder.
    pass
