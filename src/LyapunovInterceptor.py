import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from rusty_lambert import lambert_solve
from plots import plot_porkchop, plot_3d_geometry, plot_monte_carlo

EMP=.01215
Re=6378.137 #km
EM_R=384399 #km
EM_T=375200 #s
EM_V=EM_R/EM_T #km/s
E_SOI=924e3 #km
orb_a=.01 #cr3bp norm units you want the lyapunov orbit ToBeFrom L1
EMU=398600.4418
MC_Runs=500 #monte carlo sims

def cr3bp_eom(t,state):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + EMP)**2 + y**2 + z**2)
    r2=np.sqrt((x-1+EMP)**2+y**2+z**2)
    ax=2*vy+x-((1-EMP)*(x+EMP))/r1**3-EMP*(x-1+EMP)/r2**3
    ay=y-2*vx-(1-EMP)*y/r1**3-EMP*y/r2**3
    az=(EMP-1)*z/r1**3-EMP*z/r2**3
    return [vx, vy, vz, ax, ay, az]


f = lambda x:x-(1-EMP)*(x+EMP)/np.abs(x+EMP)**3-EMP*(x-1+EMP)\
    /np.abs(x-1+EMP)**3
x_L1=brentq(f,-EMP+EMP*.5,1-EMP-.05*EMP)
if orb_a>=x_L1:
    raise ValueError("Orbital radius must be less than the distance to L1")
#exactly on x-axis, 2D,perp, 2d, vy? - need vy

def y_cros(t,state): #lets us track a y=0 crossing
    return state[1]
y_cros.terminal,y_cros.direction=True,-1
def vy_finder(vy_gues): #func to feed root finder
    i_state=[x_L1-orb_a,0,0,0,vy_gues,0]
    sol = solve_ivp(cr3bp_eom,[0,5], i_state,events=y_cros,\
                rtol=1e-11,atol=1e-11)
    f_state=sol.y_events[0][0]
    f_vx=f_state[3]
    return f_vx

f_vy=brentq(vy_finder,.001,.9)
f_state=[x_L1-orb_a,0,0,0,f_vy,0]
print()
print("Initial interceptor state:", f_state)
print()
def cr3bp_eci(t,state):
    x,y,z,vx,vy,vz=state
    x_eci=(x+EMP)*np.cos(t)-y*np.sin(t)
    y_eci=(x+EMP)*np.sin(t)+y*np.cos(t)
    z_eci=z
    vx_eci=(vx-y)*np.cos(t)-(vy+x+EMP)*np.sin(t)
    vy_eci=(vx-y)*np.sin(t)+(vy+x+EMP)*np.cos(t)
    vz_eci=vz
    #Dimensionalize:
    x_eci,y_eci,z_eci=x_eci*EM_R,y_eci*EM_R,z_eci*EM_R
    vx_eci,vy_eci,vz_eci=vx_eci*EM_V,vy_eci*EM_V,vz_eci*EM_V
    return np.array([x_eci,y_eci,z_eci,vx_eci,vy_eci,vz_eci])

def lya_ephem(i_state, HPT,NP=50):#bump NP for smoother ephem
    T=2.*HPT
    t_eval=np.linspace(0,T,NP)
    sol=solve_ivp(cr3bp_eom, [0,T], i_state, t_eval=t_eval, rtol=1e-11, atol=1e-11)
    eci_st=[] # ^^ verify RK4 is good enough (switch dop853 7,8th order)
    for i in range(len(sol.t)):
        t=sol.t[i]
        st_cr3bp = sol.y[:,i]
        st_eci = cr3bp_eci(t,st_cr3bp)
        eci_st.append((t,st_eci))
    return eci_st

dum_prop=solve_ivp(cr3bp_eom,[0,5],f_state,events=y_cros,rtol=1e-11,atol=1e-11)
HP=dum_prop.t_events[0][0] #half period, need to prop once for it
dep_ephem=lya_ephem(f_state,HP)

def threat_state():
    r_ast_eci=np.array([-E_SOI,0.,0.])
    return r_ast_eci

r_tar=threat_state()
tof = 24*3600*np.linspace(0.01,10,500) # intercepts taking 0-14 days
valInts=[]

def twoB_eci_eom(t, state):#asteroid 2 bp
    r = state[0:3]
    v = state[3:6]
    r_mag=np.linalg.norm(r)
    a=-EMU*r/r_mag**3
    return [v[0],v[1],v[2],a[0],a[1],a[2]]
def hits_earth(r1,v1,tof): #reject arcs that clip through earth
    ev=lambda t,s: np.linalg.norm(s[0:3])-Re
    ev.terminal=True; ev.direction=-1
    sol=solve_ivp(twoB_eci_eom,[0,tof],np.concatenate((r1,v1)),events=ev,rtol=1e-8,atol=1e-8)
    return len(sol.t_events[0])>0
def rand_threat(speed=3.5): #used to generate a random threat targeting earth
    phi = np.random.uniform(0, 2*np.pi)
    costheta = np.random.uniform(-1, 1)
    theta=np.arccos(costheta)
    r_ast_eci=E_SOI*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi), np.cos(theta)])
    r_hat = r_ast_eci/E_SOI
    # build two axes perpendicular to the approach direction
    aux = np.array([0,0,1]) if abs(r_hat[2]) < 0.9 else np.array([1,0,0])
    e1 = np.cross(r_hat, aux);  e1 /= np.linalg.norm(e1)
    e2=np.cross(r_hat,e1)
    b=Re*np.sqrt(np.random.uniform(0,1))
    psi=np.random.uniform(0, 2*np.pi)
    target=b*(np.cos(psi)*e1+np.sin(psi)*e2) # offset earth
    v_dir=target-r_ast_eci
    v_dir/=np.linalg.norm(v_dir)
    v_ast_eci=v_dir*(speed+np.random.normal(0,0.1))
    return np.concatenate((r_ast_eci, v_ast_eci))

ast_i_st=rand_threat()
print()
print('Asteroid State:',ast_i_st)
print()

for dep_time,state_eci in dep_ephem:
    r_dep = state_eci[0:3]
    v_dep = state_eci[3:6]
    dep_time_sec = dep_time * EM_T
    for c in tof:
        tot_ast_time=dep_time_sec+c
        ast_ev=lambda t,s: np.linalg.norm(s[0:3])-Re
        ast_ev.terminal=True; ast_ev.direction=-1
        ast_sol=solve_ivp(twoB_eci_eom,[0,tot_ast_time],ast_i_st,events=ast_ev,rtol=1e-8,atol=1e-8)
        if len(ast_sol.t_events[0])>0: #asteroid already hit earth before rendezvous
            continue
        r_TF=ast_sol.y[0:3, -1]
        try:
            v1_req,v2_arr = lambert_solve(r_dep,r_TF,c,EMU)
            dv_vec = np.array(v1_req)-v_dep
            dv_mag = np.linalg.norm(dv_vec) 
            if not np.isfinite(dv_mag) or dv_mag > 50.0:
                continue
            if hits_earth(r_dep,v1_req,c):
                continue
            valInts.append({'t_dep': dep_time,'tof_days': c / (24*3600),'dv_mag': dv_mag,
                            'v1_vec': v1_req,'r_TF': r_TF,'r_dep': r_dep,'v_dep': v_dep})
        except ValueError:
            continue

valInts.sort(key=lambda x: x['dv_mag'])

best = valInts[0]
print()
print(f"Optimal solution: t={best['t_dep']}, TOF={best['tof_days']} days,\
delta V={best['dv_mag']} km/s")
print()

plot_porkchop(valInts)
plot_3d_geometry(valInts,dep_ephem,ast_i_st)

mc_dvs = []
pos_sigma = 50.0  # km - 50km uncertainty position
vel_sigma = 0.01  # km/s 10m/s vel uncert

opt_tof_sec = best['tof_days'] * 24 * 3600
tot_int_t = (best['t_dep'] * EM_T) + opt_tof_sec
start_pos = best['r_dep']
start_vel = best['v_dep']
suc_int = 0

for i in range(MC_Runs):
    pos_error = np.random.normal(0, pos_sigma, 3)
    vel_error = np.random.normal(0, vel_sigma, 3)
    perturbed_ast_st = ast_i_st + np.concatenate((pos_error, vel_error))
    ast_sol = solve_ivp(twoB_eci_eom, [0, tot_int_t], perturbed_ast_st, rtol=1e-8, atol=1e-8)
    perturbed_r_TF = ast_sol.y[0:3, -1]
    
    try:
        v1_req, v2_arr = lambert_solve(start_pos, perturbed_r_TF, opt_tof_sec, EMU)
        dv_vec = np.array(v1_req) - start_vel
        dv_mag = np.linalg.norm(dv_vec)
        if np.isfinite(dv_mag) and dv_mag < 50.0:
            mc_dvs.append(dv_mag)
            suc_int += 1
    except ValueError:
        pass

print(f"Monte Carlo results: {suc_int}/{MC_Runs} valid calculations")
if suc_int:
    mean_dv=np.mean(mc_dvs)
    std_dv=np.std(mc_dvs)
    p99_dv=np.percentile(mc_dvs, 99)
    results=[("Baseline delta V", best["dv_mag"]),("Mean delta V", mean_dv),
        ("3 sigma max error", mean_dv + 3*std_dv),("99th percentile", p99_dv),]
    print("\nDelta-V Summary:")
    for label, value in results:
        print(f"{label}: {value:.3f} km/s")
    
plot_monte_carlo(mc_dvs, best["dv_mag"])
print()