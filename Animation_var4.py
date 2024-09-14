from matplotlib.animation import FuncAnimation
import matplotlib
from math import pi

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1001)
x = 0.5 * (1 + 0.3 * np.sin(0.8 * t) + 0.5 * np.cos(3.43 * t))
psi = 1.2 * np.cos(2.14 * t)
phi = 1.1 * np.sin(2.53 * t)
Alpha = np.linspace(0, 6.29, 50)

R = 6  # радиус диска
l = R / 2  # длина OC
L = 8  # длина AB
xO = 14
yO = 11

xC = xO + l * np.sin(psi)
yC = yO + l * np.cos(psi)

xD = R * np.sin(Alpha)
yD = R * np.cos(Alpha)

xA = xC - 2 * l * np.cos(psi)
yA = yC + 2 * l * np.sin(psi)

xB = xA - L * np.sin(phi)
yB = yA - L * np.cos(phi)

N = 4
r1 = 0.1
r2 = 0.8
Beta0 = np.linspace(0, 1, 50 * N + 1)
Betas = Beta0
xS = (r1 + (r2 - r1) * Betas) * np.sin(Betas)
yS = (r1 + (r2 - r1) * Betas) * np.cos(Betas)


fig = plt.figure(figsize=(5, 5))  # задаем фигуру
ax = fig.add_subplot(1, 1, 1)  # график сетки
ax.axis("equal")
ax.set(xlim=[-0.25, 20], ylim=[-0.25, 20])  # задаем размеры сетки

ax.plot([12, 16], [10, 10], color=[0, 0.5, 0], linewidth=4)
O = ax.plot(xO, yO, "o", color=[0, 1, 0])[0]
C = ax.plot(xC, yC, "o", color=[0, 1, 0])[0]
OC = ax.plot([xO, xC[0]], [yO, yC[0]], color=[0, 0, 0])[0]
Disk = ax.plot(xD + xC[0], yD + yC[0])[0]
A = ax.plot(xA, yA, "o", color=[0, 1, 0])[0]
B = ax.plot(xB, yB, "o", markersize=20, color=[0, 1, 0])[0]
AB = ax.plot([xA[0], xB[0]], [yA[0], yB[0]], color=[0, 0, 0])[0]
SpiralnayaPruzzhina = ax.plot(xS + xO, yS + yO, color=[1, 0.5, 0.5])[0]


def kadr(i):
    C.set_data(xC[i], yC[i])
    OC.set_data([xO, xC[i]], [yO, yC[i]])
    Disk.set_data(xD + xC[i], yD + yC[i])
    A.set_data(xA[i], yA[i])
    B.set_data(xB[i], yB[i])
    AB.set_data([xA[i], xB[i]], [yA[i], yB[i]])
    Betas = Beta0 * (N * 2 * np.pi + psi[i])
    xS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi + psi[i])) * np.sin(Betas)
    yS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi + psi[i])) * np.cos(Betas)
    SpiralnayaPruzzhina.set_data(xS + xO, yS + yO)
    return [C, OC, A, B, AB, Disk, SpiralnayaPruzzhina]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
