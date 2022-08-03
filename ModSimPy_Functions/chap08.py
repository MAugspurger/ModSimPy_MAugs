from ModSimPy_Functions.modsim import *

def drag_force(V, system):
    rho, C_d, area = system['rho'], system['C_d'], system['area']
    
    mag = rho * vector_mag(V)**2 * C_d * area / 2
    direction = -vector_hat(V)
    f_drag = mag * direction
    return f_drag


def slope_func(t, state, system):
    x, y, vx, vy = state
    mass, g = system['mass'], system['g']
    
    V = Vector(vx, vy)
    a_drag = drag_force(V, system) / mass
    a_grav = g * Vector(0, -1)
    
    A = a_grav + a_drag
    
    return V.x, V.y, A.x, A.y

def event_func(t, state, system):
    x, y, vx, vy = state
    return y

