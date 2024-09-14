from matplotlib.animation import FuncAnimation
import matplotlib
from math import pi

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1001)
psi = np.sin(pi - t) + 2 * t
phi = 1.5 * np.cos(0.8 * t)
round = np.linspace(0, 6.29, 50)


L = 4  # длина палки
R = 2  # радиус циллиндра
l = 2 / 3 * R  # смещение центра тяжести
m = 1  # масса циллиндра
M = 1  # масса маятника
xk = 2  # точка касания поверхности и циллиндра
alpha = pi / 9  # угол наклона поверхности

k = np.tan(pi - alpha)

x2 = 10
y2 = 0

x1 = 0
y1 = y2 - k * x2 + k * x1


xOffset = R * np.cos(pi / 2 - alpha)
yOffset = R * np.sin(pi / 2 - alpha)
b = y2 - k * (x2 + xOffset) + k * (x1) + yOffset
xO = t
yO = k * (xO) + b

xCil = R * np.sin(round)
yCil = R * np.cos(round)

xC = xO + l * np.sin(psi)
yC = yO + l * np.cos(psi)

xM = xO + L * np.sin(phi)
yM = yO - L * np.cos(phi)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")
ax.set(xlim=[-0.25, 10], ylim=[-0.25, 10])


ax.plot([x1, x2], [y1, y2], color=[0, 0, 0])
# ax.plot([x1 + xOffset, x2 + xOffset], [y1 + yOffset, y2 + yOffset], color=[0, 0, 0])
O = ax.plot(xO, yO, "o", color=[0, 1, 0])[0]
Cilinder = ax.plot(xCil, yCil, color=[0, 0, 0])[0]
C = ax.plot(xC, yC, "o", color=[0, 1, 0])[0]
M = ax.plot(xM, yM, "o", markersize=20, color=[0, 1, 0])[0]
OM = ax.plot([xO[0], xM[0]], [yO[0], yM[0]], color=[0, 0, 0])[0]
OC = ax.plot([xO[0], xC[0]], [yO[0], yC[0]], color=[0, 0, 0])[0]


def kadr(i):
    O.set_data(xO[i], yO[i])
    Cilinder.set_data(xCil + xO[i], yCil + yO[i])
    C.set_data(xC[i], yC[i])
    M.set_data(xM[i], yM[i])
    OM.set_data([xO[i], xM[i]], [yO[i], yM[i]])
    OC.set_data([xO[i], xC[i]], [yO[i], yC[i]])
    return [O, Cilinder, C, M, OM, OC]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()