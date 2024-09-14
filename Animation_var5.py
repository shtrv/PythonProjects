from matplotlib.animation import FuncAnimation
import matplotlib
from math import pi

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 10, 1001)
phi = 0.5 * np.cos(2.31 * t)
round = np.linspace(0, 6.29, 50)
ksi = 0.5 * (1 + 0.3 * np.cos(0.8 * t) + 0.5 * np.cos(3.43 * t))


L = 3  # длина палки
R = 2  # радиус диска
m = 1  # масса диска
M = 1  # масса маятника
u = 1  # масса груза
y0 = 1  # положение равновесия пружинки
H = 1  # высота груза
W = 0.5  # длина груза


xO = 5
yO = 8 - R

xD = xO + R * np.sin(round)
yD = yO + R * np.cos(round)


xA = 5 + R
yA = y0 + ksi

xGruz = np.array([-W / 2, -W / 2, W / 2, W / 2, -W / 2])
yGruz = np.array([0, H, H, 0, 0])

n = 12
h = 0.05
xP = h * np.sin(np.pi / 2 * np.arange(2 * n + 1))
yP = np.linspace(0, 1, 2 * n + 1)

xK = xO - R * np.sin(pi / 2 + ksi)
yK = yO + R * np.cos(pi / 2 + ksi)

xM = xK + L * np.sin(-phi)
yM = yK - L * np.cos(phi)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")
ax.set(xlim=[-0.25, 10], ylim=[-0.25, 10])

ax.plot([4.5, xO, 5.5, 4.5], [8, yO, 8, 8], color=[0, 0, 0])
O = ax.plot(xO, yO, "o", color=[0, 1, 0])
Disk = ax.plot(xD, yD, color=[0, 0, 0])
ax.plot([5 + R, 5 + R + 2], [0, 0], color=[0, 0, 0])
Gruz = ax.plot(xGruz + xA, yGruz + yA[0], color=[0, 0, 0])[0]
A = ax.plot(xA, yA[0], "o", color=[0, 1, 0])[0]
Pruzzhina = ax.plot(xP + 5 + R, yP * (y0 + ksi[0]), color=[0.5, 0.5, 1])[0]
StringA = ax.plot([xO + R, xA], [yO, yA[0] + H], color=[0, 0, 0])[0]
K = ax.plot(xK[0], yK[0], "o", color=[0, 1, 0])[0]
M = ax.plot(xM[0], yM[0], "o", color=[0, 1, 0])[0]
KM = ax.plot([xK[0], xM[0]], [yK[0], yM[0]], color=[0, 0, 0])[0]


def kadr(i):
    Gruz.set_data(xGruz + xA, yGruz + yA[i])
    A.set_data(xA, yA[i])
    Pruzzhina.set_data(xP + 5 + R, yP * (y0 + ksi[i]))
    StringA.set_data([xO + R, xA], [yO, yA[i] + H])
    K.set_data(xK[i], yK[i])
    M.set_data(xM[i], yM[i])
    KM.set_data([xK[i], xM[i]], [yK[i], yM[i]])

    return [Gruz, A, Pruzzhina, StringA, K, M, KM]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))

plt.show()
