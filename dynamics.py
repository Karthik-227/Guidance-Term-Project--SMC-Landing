import math
import numpy as np
G=6.67e-11
la,lb,lc=1000,500,250
rho=3000
GM=G*(4/3)*math.pi*rho*la*lb*lc
C20 = (2*lc**2 - (la**2 + lb**2)) / (10*la**2)
C22 = (la**2 - lb**2) / (20*la**2)
omega=np.array([0,0,2*math.pi/(8*3600)])
m0=1400
Tmax=80
c=2206
