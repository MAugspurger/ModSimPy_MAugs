from ModSimPy_Functions.modsim import *

def make_system(T_init, volume, r, t_end,T_env):
    return dict(T_init=T_init,
                  T_final=T_init,
                  volume=volume,
                  r=r,
                  t_end=t_end,
                  T_env=T_env,
                  t_0=0,
                  dt=1)

def change_func(t, T, system):
    r, T_env, dt = system['r'], system['T_env'], system['dt']    
    return -r * (T - T_env) * dt

def run_simulation(system, change_func):
    t_array = np.arange(system['t_0'], system['t_end']+1, system['dt'])
    n = len(t_array)
    
    series = pd.Series(index=t_array,dtype=object)
    series.iloc[0] = system['T_init']
    
    for i in range(n-1):
        t = t_array[i]
        T = series.iloc[i]
        series.iloc[i+1] = T + change_func(t, T, system)
    
    system['T_final'] = series.iloc[-1]
    return series

