from ModSimPy_Functions.modsim import *

def change_func(state, p1, p2):
    """Simulate one time step.
    
    state: bikeshare State object
    p1: probability of an Augustana->Moline ride
    p2: probability of a Moline->Augustana ride
    """
    if flip(p1):
        bike_to_moline(state)
    
    if flip(p2):
        bike_to_augie(state)

def bike_to_augie(state):
    """Move one bike from Moline to Augustana.
    
    state: bikeshare State object
    """
    if state.moline == 0:
        state.moline_empty += 1
        return
    state.moline -= 1
    state.augie += 1

def bike_to_moline(state):
    """Move one bike from Augustana to Moline.
    
    state: bikeshare State object
    """
    if state.augie == 0:
        state.augie_empty += 1
        return
    state.augie -= 1
    state.moline += 1

