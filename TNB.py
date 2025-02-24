import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rt = lambda t: np.array([np.sin(t), np.cos(t), np.sin(t)*np.cos(t)])
rt1 = lambda t: np.array([np.cos(t), -np.sin(t), -np.sin(t)**2 + np.cos(t)**2])
rt2 = lambda t: np.array([-np.sin(t), -np.cos(t), -2*np.sin(t)*-np.cos(t) + 2*np.cos(t)*-np.sin(t)])

norm = lambda x: np.sqrt(np.sum([i**2 for i in x]))

angle = lambda x, y: np.arccos(np.dot(x, y)/(norm(x)*norm(y)))*(180/np.pi)


r = np.arange(0, 2*np.pi+np.pi/32, np.pi/32)

R = rt(r).T

def T(t):
    h = rt1(t)
    return h / norm(h)

def N(t):
    h = rt2(t) - (np.dot(rt1(t), rt2(t))/norm(rt1(t)))*rt1(t)
    return h / norm(h)

def B(t):
    cs = np.cross(rt1(t), rt2(t))
    return cs / norm(cs)

def line(x0, x1, x2, y0, y1, y2, n=20):
    d0 = (y0 - x0)/(n - 1)
    d1 = (y1 - x1)/(n - 1)
    d2 = (y2 - x2)/(n - 1)
    cx = lambda a, dt: np.array([a + i*dt for i in range(n)])
    return cx(x0, d0), cx(x1, d1), cx(x2, d2)

scale = 0.5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for t in r:
    tx, ty, tz = scale*T(t)
    nx, ny, nz = scale*N(t)
    bx, by, bz = scale*B(t)

    ax.cla()
    ax.plot(R[:, 0], R[:, 1], R[:, 2], color='black')

    r0, r1, r2 = rt(t)

    tsx, tsy, tsz = line(r0, r1, r2, tx, ty, tz)
    nsx, nsy, nsz = line(r0, r1, r2, nx, ny, nz)
    bsx, bsy, bsz = line(r0, r1, r2, bx, by, bz)

    ax.plot(tsx, tsy, tsz, color='red')
    ax.plot(nsx, nsy, nsz, color='blue')
    ax.plot(bsx, bsy, bsz, color='green')

    plt.pause(0.0004)
    


plt.show()







    
