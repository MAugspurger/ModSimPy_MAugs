from ModSimPy_Functions.modsim import *
import pandas as pd

def make_system(params, data):
    G0, k1, k2, k3, dt = params
    
    t_0 = data.index[0]
    t_end = data.index[-1]
    
    Gb = data.glucose[t_0]
    Ib = data.insulin[t_0]
    
    I = interp1d(data.insulin.index,data.insulin.values)
    
    init = pd.Series(dict(G=G0, X=0),dtype=np.float64)
    
    system = dict(init=init, G0=G0,
                  k1=k1,k2=k2,
                  k3=k3,dt=dt,
                  Gb=Gb, Ib=Ib, I=I,
                  t_0=t_0, t_end=t_end)
    
    return system

def slope_func(t, state, system):
    G, X = state
    k1, k2, k3, dt = system['k1'],system['k2'], system['k3'], system['dt']
    I, Ib, Gb = system['I'], system['Ib'], system['Gb']
        
    dGdt = -k1 * (G - Gb) - X*G
    dXdt = k3 * (I(t) - Ib) - k2 * X
    
    return dGdt, dXdt