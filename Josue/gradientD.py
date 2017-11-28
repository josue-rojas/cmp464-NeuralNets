from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
# 3D surface (color map)

fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')



XYMin = -2
XYMax = 2

X = np.arange(XYMin, XYMax, 0.25)
X0 = np.array([0 for x in range(len(X))])
Y = np.arange(XYMin, XYMax, 0.25)
X, Y = np.meshgrid(X, Y)
Z = (X ** 2) + (Y ** 2)
ZY0 = (X ** 2) + 1
ZX0 = 0 + (Y ** 2)
YLine = (5/4)*X + (-1/4)
print YLine
# exit()

# X, Y, Z = axes3d.get_test_data(0.05)
# surf = ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow,
                       linewidth=0, antialiased=False, alpha=.7)


# plot
# other part thingy
ax.plot_surface(0, Y, Z, color='b', linewidth=0, antialiased=False)
# ax.plot_surface(X, YLine, Z, linewidth=0, antialiased=False)

ax.set_zlim(0, 4)
fig.colorbar(surf)

plt.show()
