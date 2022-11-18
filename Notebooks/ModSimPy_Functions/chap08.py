from ModSimPy_Functions.modsim import *
import numpy as np

def drag_force(V, system):
    rho, C_d, area = system['rho'], system['C_d'], system['area']
    
    # Find the magnitude and direction of the velocity
    vel_mag = np.sqrt(V.x**2 + V.y**2)
    dir = V/vel_mag

    # Find the magnitude of the drag force
    drag_mag = rho * vel_mag**2 * C_d * area * (1/2)

    # Define the direction of the force as opposite that of the  velocity
    # Notice that "dir" is a vector, so f_drag is vector too
    f_drag = drag_mag * -dir

    return f_drag


def slope_func(t, state, system):
    x, y, vx, vy = state
    mass, g = system['mass'], system['g']
    
    V = pd.Series(dict(x=vx, y=vy),dtype=float)
    a_drag = drag_force(V, system) / mass

    # Acceleration has to be defined as a vector too
    a_grav = pd.Series(dict(x=0,y=-g),dtype=float)
    
    A = a_grav + a_drag
    
    return V.x, V.y, A.x, A.y  

def event_func(t, state, system):
    x, y, vx, vy = state
    return y

def angle_to_components(mag,angle):
    theta = np.deg2rad(angle)
    x = mag * np.cos(theta)
    y = mag * np.sin(theta)
    return pd.Series(dict(x=x,y=y),dtype=float)

def drag_force_var(V, system, cd_func):
    rho, area = system['rho'], system['area']
    vel_mag = np.sqrt(V.x**2 + V.y**2)
    C_d = cd_func(vel_mag)
    dir = V/vel_mag
    drag_mag = rho * vel_mag**2 * C_d * area * (1/2)
    f_drag = drag_mag * -dir

    return f_drag


# Define the modified slope_function here
def slope_func_var(t, state, system):
    x, y, vx, vy = state
    mass, g, cd_func = system['mass'], system['g'], system['cd_func']
    
    V = pd.Series(dict(x=vx, y=vy),dtype=float)
    a_drag = drag_force_var(V, system, cd_func) / mass

    # Acceleration has to be defined as a vector too
    a_grav = pd.Series(dict(x=0,y=-g),dtype=float)
    
    A = a_grav + a_drag
    
    return V.x, V.y, A.x, A.y  

