from matplotlib.animation import FuncAnimation
import matplotlib
from math import pi

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1001)
phi = 0.77 * np.cos(1.52 * t)
psi = 1.36 * np.sin(2.02 * t)
x = 1.15 * np.cos(2.66 * t) * np.sin(1.2 * t + 2)
round = np.linspace(0, 6.29, 50)

R = 4
r = 3
m = 1
y0 = 1

xO = 10 + np.cos(pi / 2 - psi)
yO = y0 + R

xOb = R * np.sin(round)
yOb = R * np.cos(round)
xOb1 = r * np.sin(round)
yOb1 = r * np.cos(round)

xA = xO + (r + (R - r) / 2) * np.cos(pi / 2 - psi)
yA = yO + (r + (R - r) / 2) * np.sin(pi / 2 - psi)

xB = xO + (r + (R - r) / 2) * np.cos(pi / 2 - phi)
yB = yO - (r + (R - r) / 2) * np.sin(pi / 2 - phi)

"""
n = 12
h = 0.15
xP = h * np.sin(np.pi / 2 * np.arange(2 * n + 1))
yP = np.linspace(0, 1, 2 * n + 1)
"""


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")
ax.set(xlim=[-0.25, 20], ylim=[-0.25, 20])

ax.plot([0, 20], [y0, y0], color=[0, 0.5, 0], linewidth=4)
O = ax.plot(xO[0], yO, "o", color=[0, 1, 0])[0]
Obruch = ax.plot(xOb, yOb, color=[0, 0, 0])[0]
Obruch1 = ax.plot(xOb1, yOb1, color=[0, 0, 0])[0]
A = ax.plot(xA[0], yA[0], "o", color=[0, 1, 0], markersize=5)[0]
B = ax.plot(xB[0], yB[0], "o", color=[0, 1, 0], markersize=(R - r) * 10)[0]
# Pruzzhina = ax.plot(xB[0] + xP, yB[0] + yP * (yA[0] - yB[0]), color=[0.5, 0.5, 1])[0]
AB = ax.plot([xA[0], xB[0]], [yA[0], yB[0]], color=[0.5, 0.5, 1])[0]


def kadr(i):
    O.set_data(xO[i], yO)
    Obruch.set_data(xOb + xO[i], yOb + yO)
    Obruch1.set_data(xOb1 + xO[i], yOb1 + yO)
    A.set_data(xA[i], yA[i])
    B.set_data(xB[i], yB[i])
    # Pruzzhina.set_data(xB[i] + xP, yB[i] + yP * (yA[i] - yB[i]))
    AB.set_data([xA[i], xB[i]], [yA[i], yB[i]])
    return [O, Obruch, Obruch1, A, B, AB]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
