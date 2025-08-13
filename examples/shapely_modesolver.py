"""Fully vectorial finite-difference mode solver example.

Flexible mode solver example for fundamental modes for isotropic media
by David Hutchings, James Watt School of Engineering, University of Glasgow
David.Hutchings@glasgow.ac.uk

Features added over basic examples

1. Uses shapely python library for manipulation and analysis of geometric 
objects in the Cartesian plane. It allows polygon regions to be employed.

2. smooth option applies a 2x2 cell moving average to permittivity values

3. Contour plots of Stokes parameters are available. These are pure real 
values and determined from both H and E field values (E co-located using a 
2x2 cell moving average) rather than plane-wave approximation of one field 
component. S0 is the irradiance, values are scaled to S0(max) and contours 
are at -3dB intervals. Additionally, average values over modal 
cross-section are computed.
S0 = Re(E_x H_y^* -E_y H_x^*) = S.z_unit (Irradiance)
S1 = Re(E_x H_y^* +E_y H_x^*)            (modal normal birefringence)
S2 = Re(E_y H_y^* -E_x H_x^*)            (modal diagonal birefringence)
S3 = Im(E_y H_y^* +E_x H_x^*) = 0        (modal circular birefringence)

Integrated Optical Isolators (planned for: Springer Series in Optical Sciences)
chapter: Theory of Faraday effect and waveguide mode conversion
David C. Hutchings
 
"""

import numpy
import shapely
import EMpy 
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import signal
plt.rcParams.update({'font.size': 25})

"""Define cross-section geometry of simulation each element tuple contains
("label",shapely.Polygon(coordinates),eps) for isotropic media
shapely.box(xmin, ymin, xmax, ymax) is shorthand method for box polygons
units should match units of wavelength wl
"""

geom = (("base",shapely.box(-800,-800,800,800),1.0**2),
        ("substrate",shapely.box(-800,-800,800,0),1.5**2),
#        ("core",shapely.box(-240,0,240,340),3.4**2))
        ("core",shapely.Polygon(((-400,0),(400,0),(160,340),(-160,340),(-400,0))),3.4**2))

wl = 1550
nx, ny = 161, 161
bbox = geom[0][1].bounds
x = numpy.linspace(bbox[0], bbox[2], nx)
y = numpy.linspace(bbox[1], bbox[3], ny)

""" smooth option applies a convolution to dielectric tensor such that 
it uses a 2x2-point moving average value. this will help smooth out 
staircase profiles 
"""
smooth = True
if smooth:
    x = signal.convolve(x,[0.5,0.5],mode='valid')
    y = signal.convolve(y,[0.5,0.5],mode='valid')

def epsfunc(x_, y_): 
    """Return a dielectric tensor matrix describing a 2d material.
    :param x_: x values
    :param y_: y values
    :return: 2d-matrix
    """
    
    if smooth:
        xw = signal.convolve(x_,[0.5,0.5],mode='valid')
        xw = numpy.concatenate(([(3*x_[0]-x_[1])/2.], xw, [(3*x_[-1]-x_[-2])/2.]))
        yw = signal.convolve(y_,[0.5,0.5],mode='valid')
        yw = numpy.concatenate(([(3*y_[0]-y_[1])/2.], yw, [(3*y_[-1]-y_[-2])/2.]))
    else:   
        xw = x_
        yw = y_
        
#assume base medium is isotropic
    working = numpy.zeros((xw.size,yw.size),dtype='complex')
    working[:,:] = geom[0][2]*numpy.ones((xw.size,yw.size))
    for i in range(1,len(geom)):
        xybounds = geom[i][1].bounds
        for ix in range(xw.size):
            if (xybounds[0] <= xw[ix] <= xybounds[2]): 
                for iy in range(yw.size):
                    if (xybounds[1] <= yw[iy] <= xybounds[3]):
                        if shapely.covers(geom[i][1],shapely.Point(xw[ix],yw[iy])):
                            working[ix,iy] = numpy.array(geom[i][2])
                            
    if smooth:
        newworking = numpy.empty((x_.size,y_.size),dtype='complex')     
        newworking[:,:] = signal.convolve2d( working[:,:], [[0.25,0.25],[0.25,0.25]],mode='valid')

        working = newworking
        
    return working
  
neigs = 2
tol = 1e-6
boundary = '0000'
solver = EMpy.modesolvers.FD.VFDModeSolver(wl, x, y, epsfunc, boundary).solve(
    neigs, tol)

levls = numpy.geomspace(1./32.,1.,num=11)
levls2 = numpy.geomspace(1./1024.,1.,num=11)
xe = signal.convolve(x,[0.5,0.5],mode='valid')
ye = signal.convolve(y,[0.5,0.5],mode='valid')
label_loc = [(3*bbox[2]+bbox[0])/4.,(5*bbox[3]+bbox[1])/6.]

def plot_coords(coords):
    pts = list(coords)
    x,y = zip(*pts)

    plt.plot(x,y)
    
    return
    
def geom_outline():
    ax.set_box_aspect(1)
    ax.set_aspect('equal')
    ax.set_xlabel('distance/nm')
    ax.set_ylabel('height/nm')
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.margins(0)
    [plot_coords(geom[i][1].exterior.coords) for i in range(1,len(geom))]
    return

print(solver.modes[0].neff)
fmax=abs(solver.modes[0].Ex).max()
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
plt.contour(xe,ye,abs(solver.modes[0].Ex.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$E_x$")
geom_outline()
ax = fig.add_subplot(2, 3, 2)
plt.contour(xe,ye,abs(solver.modes[0].Ey.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$E_y$")
geom_outline()
ax = fig.add_subplot(2, 3, 3)
plt.contour(xe,ye,abs(solver.modes[0].Ez.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$E_z$")
geom_outline()
fmax=abs(solver.modes[0].Hy).max()
ax = fig.add_subplot(2, 3, 4)
plt.contour(x,y,abs(solver.modes[0].Hx.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$H_x$")
geom_outline()
ax = fig.add_subplot(2, 3, 5)
plt.contour(x,y,abs(solver.modes[0].Hy.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$H_y$")
geom_outline()
ax = fig.add_subplot(2, 3, 6)
plt.contour(x,y,abs(solver.modes[0].Hz.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$H_z$")
geom_outline()
fig.tight_layout()
plt.show()

ExatH = signal.convolve2d(solver.modes[0].Ex,[[0.25,0.25],[0.25,0.25]])
EyatH = signal.convolve2d(solver.modes[0].Ey,[[0.25,0.25],[0.25,0.25]])
EzatH = signal.convolve2d(solver.modes[0].Ez,[[0.25,0.25],[0.25,0.25]])
#Stokes parameters
q1 = ExatH*numpy.conjugate(solver.modes[0].Hy)
q2 = EyatH*numpy.conjugate(solver.modes[0].Hx)
q3 = EyatH*numpy.conjugate(solver.modes[0].Hy)
q4 = ExatH*numpy.conjugate(solver.modes[0].Hx)

S0 = q1.real-q2.real
S1 = q1.real+q2.real
S2 = q3.real-q4.real
S3 = q3.imag+q4.imag
denom = S0.sum()
print("ave S1=",S1.sum()/denom)
print("ave S2=",S2.sum()/denom)
print("ave S3=",S3.sum()/denom)

fmax=abs(S0).max()
fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
plt.contour(x,y,abs(S0.T), fmax*levls2, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$S_0$")
geom_outline()
ax = fig.add_subplot(2, 2, 2)
plt.contour(x,y,abs(S1.T), fmax*levls2, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$|S_1|$")
geom_outline()
ax = fig.add_subplot(2, 2, 3)
plt.contour(x,y,abs(S2.T), fmax*levls2, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$|S_2|$")
geom_outline()
ax = fig.add_subplot(2, 2, 4)
plt.contour(x,y,abs(S3.T), fmax*levls2, cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$|S_3|$")
geom_outline()
fig.tight_layout()
plt.show()

print(solver.modes[1].neff)
fmax=abs(solver.modes[1].Ey).max()
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
plt.contour(xe,ye,abs(solver.modes[1].Ex.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$E_x$")
geom_outline()
ax = fig.add_subplot(2, 3, 2)
plt.contour(xe,ye,abs(solver.modes[1].Ey.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$E_y$")
geom_outline()
ax = fig.add_subplot(2, 3, 3)
plt.contour(xe,ye,abs(solver.modes[1].Ez.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$E_z$")
geom_outline()
fmax=abs(solver.modes[1].Hx).max()
ax = fig.add_subplot(2, 3, 4)
plt.contour(x,y,abs(solver.modes[1].Hx.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$H_x$")
geom_outline()
ax = fig.add_subplot(2, 3, 5)
plt.contour(x,y,abs(solver.modes[1].Hy.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$H_y$")
geom_outline()
ax = fig.add_subplot(2, 3, 6)
plt.contour(x,y,abs(solver.modes[1].Hz.T), fmax*levls, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$H_z$")
geom_outline()
fig.tight_layout()
plt.show()

ExatH = signal.convolve2d(solver.modes[1].Ex,[[0.25,0.25],[0.25,0.25]])
EyatH = signal.convolve2d(solver.modes[1].Ey,[[0.25,0.25],[0.25,0.25]])
EzatH = signal.convolve2d(solver.modes[1].Ez,[[0.25,0.25],[0.25,0.25]])
#Stokes parameters
q1 = ExatH*numpy.conjugate(solver.modes[1].Hy)
q2 = EyatH*numpy.conjugate(solver.modes[1].Hx)
q3 = EyatH*numpy.conjugate(solver.modes[1].Hy)
q4 = ExatH*numpy.conjugate(solver.modes[1].Hx)

S0 = q1.real-q2.real
S1 = q1.real+q2.real
S2 = q3.real-q4.real
S3 = q3.imag+q4.imag
denom = S0.sum()
print("ave S1=",S1.sum()/denom)
print("ave S2=",S2.sum()/denom)
print("ave S3=",S3.sum()/denom)

fmax=abs(S0).max()
fig = plt.figure(tight_layout=True)

ax = fig.add_subplot(2, 2, 1)
plt.contour(x,y,abs(S0.T), fmax*levls2, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$S_0$")
geom_outline()
ax = fig.add_subplot(2, 2, 2)
plt.contour(x,y,abs(S1.T), fmax*levls2, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$|S_1|$")
geom_outline()
ax = fig.add_subplot(2, 2, 3)
plt.contour(x,y,abs(S2.T), fmax*levls2, 
            cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$|S_2|$")
geom_outline()
ax = fig.add_subplot(2, 2, 4)
plt.contour(x,y,abs(S3.T), fmax*levls2, cmap='jet', locator=ticker.LogLocator())
ax.text(label_loc[0],label_loc[1],r"$|S_3|$")
geom_outline()
fig.tight_layout()
plt.show()


