from ModSimPy_Functions.modsim import *
import pandas as pd


def make_system(G0, k1, k2, k3, dt, data):
    t_0 = data.index[0]
    t_end = data.index[-1]
    
    Gb = data.glucose[t_0]
    Ib = data.insulin[t_0]
    
    I = interp1d(data.insulin.index,data.insulin.values)

    state = pd.Series(dict(G=G0, X=0),dtype=np.float64)
    system = dict(init=state,
                  k1=k1,k2=k2,
                  k3=k3,dt=dt,
                  Gb=Gb, Ib=Ib, I=I,
                  t_0=t_0, t_end=t_end)
    
    return system, state

def slope_func(t, state, system):
    G, X = state
    k1, k2, k3, dt = system['k1'],system['k2'], system['k3'], system['dt']
    I, Ib, Gb = system['I'], system['Ib'], system['Gb']
        
    dGdt = -k1 * (G - Gb) - X*G
    dXdt = k3 * (I(t) - Ib) - k2 * X
    
    return dGdt, dXdt
