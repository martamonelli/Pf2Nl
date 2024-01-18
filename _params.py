import numpy as np
from math import pi

NET = 1.31494*10**(-6)
f_min = 1e-05
f_knee = 20.0e-3
alpha = 1
omega = 88*2*pi/60
f_samp = 19.1

NET = 1*10**(-6)
f_min = 1e-03
f_knee = 4e-03
alpha = 1

#omega = 2*pi*1.19375 # number = f_samp/16
omega = 4*pi

xi_dets = np.array([0, pi/2, pi/4, 3*pi/4])

f_samp = 20

nside = 32
