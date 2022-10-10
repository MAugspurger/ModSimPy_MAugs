from ModSimPy_Functions.modsim import *
import pandas as pd


def make_system(iS,iI,iR,tc,tr,t_end):
    ''' Makes a system for a KM model problem.

    iS: Initial susceptible population
    iI: Initial infected population
    iR: Initial recovered population
    tc: Days between potentially infecting contacts
    tr: Averay recovery time, in days
    t_end: the length of time of the simulation in days
    
    return: a dictionary holding the system parameters'''
    N = iS + iI + iR
    iS = iS/N
    iI = iI/N
    iR = iR/N
    beta = 1/tc
    gamma = 1/tr
    return dict(iS=iS, iI=iI, iR=iR, N=N, beta=beta, gamma=gamma,
                t_end=t_end)


def make_state(system):
    state = pd.Series(dict(s=system['iS'], i=system['iI'], r=system['iR']),name="State Variables")
    return state


def change_func(t, state, system):
    s, i, r = state.s, state.i, state.r

    infected = system['beta'] * i * s    
    recovered = system['gamma'] * i
    
    s -= infected
    i += infected - recovered
    r += recovered
    
    return pd.Series(dict(s=s, i=i, r=r),name="State")


def plot_results(S, I, R):
    S.plot(style='--', label='Susceptible',legend=True)
    I.plot(style='-', label='Infected',legend=True)
    R.plot(style=':', label='Resistant',
           xlabel='Time (days)',
           ylabel='Fraction of population',
          legend=True)

    
def run_simulation(system, change_func):
    state = make_state(system)
    frame = pd.DataFrame([],columns=state.index)
    frame.loc[0] = state
    
    for t in range(0, system['t_end']):
        frame.loc[t+1] = change_func(t, frame.loc[t], system)
    
    return frame


def add_immunization(system, fraction):
    system['iS'] -= fraction
    system['iR'] += fraction

    
def calc_total_infected(results, system):
    s_0 = results.s[0]
    s_end = results.s[system['t_end']]
    return s_0 - s_end


def sweep_beta(system,beta_array, gamma):
    sweep = pd.Series([],dtype=object)

    # Unpack variables from original system and return to argument state
    N = system['N']
    iS, iI, iR = system['iS']*N, system['iI']*N,system['iR']*N
    tc, tr, t_end= 1/system['beta'], 1/system['gamma'], system['t_end']

    for beta in beta_array:
        system_local = make_system(iS,iI,iR,1/beta,1/gamma,t_end)
        results = run_simulation(system_local, change_func)
        sweep[beta] = calc_total_infected(results, system_local)
    return sweep


def sweep_parameters(system,beta_array, gamma_array):
    frame = pd.DataFrame([],columns=gamma_array)
    for gamma in gamma_array:
        frame[gamma] = sweep_beta(system,beta_array, gamma)
    return frame

