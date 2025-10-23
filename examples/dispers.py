# -*- coding: utf-8 -*-
"""
Helper code to calculate GVD and group refractive index from a set of 
effective refractive index versus wavelength data

x (numpy array): wavelengths in nm
y (numpy array): corresponding effective refractive index values

Created on Sun Oct  5 16:36:04 2025

@author: dch2y
"""

import numpy as np
from numpy.polynomial import Chebyshev
import matplotlib.pyplot as plt
import scipy.constants as sc

"""
in this example we use published sellemeier form for refractive index of silicon
C. D. Salzberg and J. J. Villa. Infrared refractive indexes of silicon, 
germanium and modified selenium glass. J. Opt. Soc. Am., 47, 244-246 (1957)

substitute with appropriate effective refractive index which can include 
material, structural or any other dependence
"""

si_sell = np.array([[10.6684293,0.0909121907],
                    [0.0030434748,1.28766018],
                    [1.54133408,1218816.]])
my_n2 = lambda w : 1+(si_sell[:,0]*(w/1000.)**2/((w/1000.)**2-si_sell[:,1])).sum()

# populate x, y arrays 
# here we fit to third-order dispersive term - change as required

npoints = 11
x = np.linspace(1500.,1600.,npoints)
y = np.empty(npoints)
for i, xi in enumerate(x):
    y[i] = np.sqrt(my_n2(xi)) 
   
plt.plot(x,y,'o')
p, pinfo = Chebyshev.fit(x,y,3,full=True)
xx, yy = p.linspace()
xz, zz = p.deriv(1).linspace()
zz = yy - xx*zz
xg, gvd = p.deriv(2).linspace()
gvd = -1.e12*xx*gvd/sc.c
# units of gvd are ps/nm/m
plt.plot(xx, yy, lw=2)
plt.plot(xx, zz, lw=2)
plt.show()

mid_pt = (x[0]+x[-1])/2.
print("mid-point wavelength=", mid_pt, "nm")
print("n_p=", p(mid_pt))
print("n_g=", p(mid_pt)-mid_pt*p.deriv(1)(mid_pt))
print("GVD=", -1.e12*mid_pt*p.deriv(2)(mid_pt)/sc.c, "ps/nm/m")
      