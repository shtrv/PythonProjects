from matplotlib.animation import FuncAnimation
import matplotlib
from math import pi

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1001)
psi = 1.2 * np.sin(pi - 1.3 * t)
phi = 1.02 * np.cos(0.8 * t)
round = np.linspace(0, 6.29, 50)


L = 4  # длина палки
R = 2  # радиус диска
m = 1  # масса диск
M = 1  # масса маятника

xO = 10 + np.cos(pi / 2 - psi)
yO = R + 5

xC = xO + R * np.sin(psi)
yC = yO + R * np.cos(psi)


xCil = R * np.sin(round)
yCil = R * np.cos(round)

xM = xO + L * np.sin(phi)
yM = yO - L * np.cos(phi)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")
ax.set(xlim=[-0.25, 20], ylim=[-0.25, 20])


ax.plot([0, 20], [5, 5], color=[0, 0.5, 0], linewidth=4)
O = ax.plot(xO[0], yO, "o", color=[0, 1, 0])[0]
Cilinder = ax.plot(xCil, yCil, color=[0, 0, 0])[0]
Mass = ax.plot(xM, yM, "o", markersize=20, color=[0, 1, 0])[0]
OM = ax.plot([xO[0], xM[0]], [yO, yM[0]], color=[0, 0, 0])[0]
OC = ax.plot([xO[0], xC[0]], [yO, yC[0]], color=[0, 0, 0])[0]


def kadr(i):
    O.set_data(xO[i], yO)
    Cilinder.set_data(xCil + xO[i], yCil + yO)
    Mass.set_data(xM[i], yM[i])
    OM.set_data([xO[i], xM[i]], [yO, yM[i]])
    OC.set_data([xO[i], xC[i]], [yO, yC[i]])
    return [O, Cilinder, OM, Mass, OC]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
