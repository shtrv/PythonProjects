from matplotlib.animation import FuncAnimation
import matplotlib
from math import pi

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1001)
phi = 1.5 * np.cos(0.8 * t)
ksi = 2.2 * np.sin(3 * t)
round = np.linspace(0, 6.29, 50)

L = 4  # длина нити
R = 4  # радиус катушки
m = 1  # масса катушки
y0 = 10

xO = 10
yO = 17

xK0 = xO
yK0 = y0 + ksi

xK = xK0 + np.sin(-phi)
yK = yK0 + np.cos(phi)

xC = xK + R * np.cos(pi / 2 - phi)
yC = yK - R * np.sin(pi / 2 - phi)

xKat = R * np.sin(round)
yKat = R * np.cos(round)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")
ax.set(xlim=[-0.25, 20], ylim=[-0.25, 20])

ax.plot([8, 12], [18, 18], color=[0, 0.5, 0], linewidth=4)
ax.plot([9, xO, 11], [18, yO, 18], color=[0, 0, 0])
O = ax.plot(xO, yO, "o", color=[0, 1, 0])
K = ax.plot(xK[0], yK[0], color=[0, 1, 0])[0]
OK = ax.plot([xO, xK[0]], [yO, yK[0]], color=[0, 0, 0])[0]
C = ax.plot(xC[0], yC[0], "o", color=[0, 1, 0])[0]
Kat = ax.plot(xKat, yKat, color=[0, 0, 0])[0]


def kadr(i):
    K.set_data(xK[i], yK[i])
    OK.set_data([xO, xK[i]], [yO, yK[i]])
    C.set_data(xC[i], yC[i])
    Kat.set_data(xKat + xC[i], yKat + yC[i])
    return [K, OK, C, Kat]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
