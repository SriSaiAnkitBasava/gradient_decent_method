import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pylab import meshgrid
import matplotlib.pyplot as plt


"""
#x = np.arange(0, 0.5, 0.00001)
x = np.arange(0, 0.5, 0.00001)
y = np.arange(0, 0.5, 0.00001)
X,Y = meshgrid(x, y)
Z = np.log(-1-X-Y) - np.log(X) - np.log(Y)
Z = np.sin(2*np.abs(X-0.3)+2*np.sin(5*Y))
fig = plt.figure()
ax = fig.gca(projection='2d')
ax.plot_surface(X, Y, Z)
plt.show()
"""