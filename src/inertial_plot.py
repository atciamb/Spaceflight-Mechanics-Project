import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --- Simulation Parameters ---
EMP = 0.01215
EM_R = 384399 # km
orb_a = 0.01  # Normalized amplitude

# --- CR3BP Equations of Motion ---
def cr3bp_eom(t, state):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + EMP)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + EMP)**2 + y**2 + z**2)
    ax = 2*vy + x - ((1-EMP)*(x+EMP))/r1**3 - EMP*(x-1+EMP)/r2**3
    ay = y - 2*vx - (1-EMP)*y/r1**3 - EMP*y/r2**3
    az = (EMP-1)*z/r1**3 - EMP*z/r2**3
    return [vx, vy, vz, ax, ay, az]

# --- Find L1 ---
f = lambda x: x - (1-EMP)*(x+EMP)/np.abs(x+EMP)**3 - EMP*(x-1+EMP)/np.abs(x-1+EMP)**3
x_L1 = brentq(f, -EMP+EMP*.5, 1-EMP-.05*EMP)

# --- Root Finder for Half-Period Crossing ---
def y_cros(t, state):
    return state[1]
y_cros.terminal, y_cros.direction = True, -1

def vy_finder(vy_gues):
    i_state = [x_L1 - orb_a, 0, 0, 0, vy_gues, 0]
    sol = solve_ivp(cr3bp_eom, [0, 5], i_state, events=y_cros, rtol=1e-11, atol=1e-11)
    f_state = sol.y_events[0][0]
    return f_state[3] # Return x-velocity at crossing

f_vy = brentq(vy_finder, 0.01, 0.9)
f_state = [x_L1 - orb_a, 0, 0, 0, f_vy, 0]

# --- ECI Transformation ---
def cr3bp_eci_pos(t, state):
    x, y, z = state[0], state[1], state[2]
    x_eci = (x + EMP)*np.cos(t) - y*np.sin(t)
    y_eci = (x + EMP)*np.sin(t) + y*np.cos(t)
    return np.array([x_eci * EM_R, y_eci * EM_R, z * EM_R])

# --- Propagate Orbit for Plotting ---
dum_prop = solve_ivp(cr3bp_eom, [0, 5], f_state, events=y_cros, rtol=1e-11, atol=1e-11)
HP = dum_prop.t_events[0][0]
T = 2.0 * HP 

t_eval = np.linspace(0, T, 500)
sol = solve_ivp(cr3bp_eom, [0, T], f_state, t_eval=t_eval, rtol=1e-11, atol=1e-11)
eci_positions = np.array([cr3bp_eci_pos(sol.t[i], sol.y[:, i]) for i in range(len(sol.t))])

# ==========================================
# --- Plotting Figure 1 ---
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left Subplot
ax1.plot(sol.y[0, :], sol.y[1, :], 'b-', label='Lyapunov Orbit')
ax1.plot(f_state[0], f_state[1], 'go', markersize=6, label='Start')
ax1.plot(x_L1, 0, 'r+', markersize=8, label='L1')
ax1.set_title('Rotating CR3BP Frame', fontsize=14)
ax1.set_xlabel('x (norm)', fontsize=12)
ax1.set_ylabel('y (norm)', fontsize=12)
ax1.grid(True, alpha=0.4)
ax1.axis('equal') 
ax1.legend(loc='upper right')

# Right Subplot
ax2.plot(eci_positions[:, 0], eci_positions[:, 1], 'g-', label='Inertial Path')
ax2.plot(0, 0, 'bo', markersize=6, label='Earth')
ax2.set_title('Earth-Centered Inertial Frame', fontsize=14)
ax2.set_xlabel('X (km)', fontsize=12)
ax2.set_ylabel('Y (km)', fontsize=12)
ax2.grid(True, alpha=0.4)
ax2.axis('equal')
ax2.legend(loc='upper right')

# <-- FIX: Angle the long X-axis numbers to prevent them crashing into each other
ax2.tick_params(axis='x', rotation=25)

# <-- FIX: Add massive padding between the two graphs so they don't bleed together
plt.tight_layout(w_pad=4.0) 

plt.savefig('fig1_lyapunov_corrected.png', dpi=300, bbox_inches='tight')
plt.show()