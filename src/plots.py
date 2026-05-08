import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def twoB_eci_eom(t, state):#asteroid 
    EMU=398600.4418 #km^3/s^2
    r = state[0:3]
    v = state[3:6]
    r_mag=np.linalg.norm(r)
    a=-EMU*r/r_mag**3
    return [v[0],v[1],v[2],a[0],a[1],a[2]]
def plot_porkchop(val_ints):
    if not val_ints: return
    t_deps=[d['t_dep'] for d in val_ints]
    tofs=[d['tof_days'] for d in val_ints]
    dvs=[d['dv_mag'] for d in val_ints]
    plt.figure(figsize=(10,7))
    # generate contour
    contour=plt.tricontourf(t_deps,tofs,dvs,levels=100,cmap='viridis',vmax=3.0)
    cbar=plt.colorbar(contour)
    cbar.set_label('delta V (km/s)')
    best=min(val_ints,key=lambda x:x['dv_mag'])
    # mark absolute minimum
    plt.plot(best['t_dep'],best['tof_days'],'r.',markersize=15,label=f"opt: {best['dv_mag']:.3f} km/s")
    plt.title('earth-asteroid intercept porkchop plot')
    plt.xlabel('departure time (TU)')
    plt.ylabel('time of flight (days)')
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.show()

def plot_3d_geometry(val_ints, ephem, ast_ist):
    if not val_ints: return
    best = val_ints[0]
    EM_R=384399 #km
    EM_T=375200 #s
    dep_time_sec = best['t_dep'] * EM_T

    start_pos = next((st[0:3] for t, st in ephem if np.isclose(t, best['t_dep'])), None)
    if start_pos is None: return
    initial_sc_state = np.concatenate((start_pos, best['v1_vec']))
    tof_seconds = best['tof_days'] * 24 * 3600
    total_ast_time = dep_time_sec + tof_seconds
    sc_arc = solve_ivp(twoB_eci_eom, [0, tof_seconds], initial_sc_state, max_step=3600)
    ast_arc = solve_ivp(twoB_eci_eom, [0, total_ast_time], ast_ist, max_step=3600)

    lya_pos=np.array([st[0:3] for _,st in ephem])
    moon_th=np.linspace(0,2*np.pi,300)
    moon_x=EM_R*np.cos(moon_th); moon_y=EM_R*np.sin(moon_th)
    t_dep_tu=best['t_dep']
    moon_pos=np.array([EM_R*np.cos(t_dep_tu), EM_R*np.sin(t_dep_tu), 0.])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='blue', s=100, label='Earth')
    ax.plot(moon_x, moon_y, np.zeros_like(moon_x), color='gray', linewidth=0.5, alpha=0.4)
    ax.scatter(*moon_pos, color='gray', s=80, label='Moon')
    ax.plot(lya_pos[:,0], lya_pos[:,1], lya_pos[:,2], color='cyan', linewidth=0.6, alpha=0.5, label='Lyapunov Orbit')
    ax.plot(ast_arc.y[0], ast_arc.y[1], ast_arc.y[2], color='red', linestyle=':', label='Asteroid Trajectory')
    ax.scatter(ast_arc.y[0,-1], ast_arc.y[1,-1], ast_arc.y[2,-1], color='red', marker='X', s=100, label='Impact Point')
    ax.plot(sc_arc.y[0], sc_arc.y[1], sc_arc.y[2], color='green', linewidth=2, label='Spacecraft Trajectory')
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=50, label='L1 Departure')
    lim = 1e6
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_title('Orbital Intercept Geometry')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.legend()
    plt.show()

def plot_monte_carlo(mc_dvs, baseline_dv):
    if not mc_dvs:
        print("No Monte Carlo data to plot.")
        return
    plt.figure(figsize=(9, 6))
    plt.hist(mc_dvs, bins=40, color='b', edgecolor='black', alpha=0.7)
    plt.axvline(baseline_dv, color='red', linestyle='dashed', linewidth=2, 
                label=f'Nominal delta V ({baseline_dv:.3f} km/s)')
    p99 = np.percentile(mc_dvs, 99)
    plt.axvline(p99, color='orange', linestyle='dotted', linewidth=2, 
                label=f'99th Percentile ({p99:.3f} km/s)')
    plt.title('MCS: delta V spread due to tracking errors')
    plt.xlabel('Required delta V (km/s)')
    plt.ylabel('Frequency (sims)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()