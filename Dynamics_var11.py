import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import pi

matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

t = np.linspace(0, 10, 1001)
x = 0.5 * (1 + 0.3 * np.sin(0.8 * t) + 0.5 * np.cos(3.43 * t))
psi = 0.4 * np.cos(1.4 * t)
phi = 1.1 * np.sin(1.53 * t)


m = 6  # масса точки
L = 8  # длина стержней
x0 = 4
y0 = 2

xO = x0
yO = y0

xA = xO + L * np.sin(psi)
yA = yO + L * np.cos(psi)

xB = xA + L * np.sin(phi)
yB = yA + L * np.cos(phi)

N = 4
r1 = 0.1
r2 = 0.8
Beta0 = np.linspace(0, 1, 50 * N + 1)
Betas = Beta0
xS1 = (r1 + (r2 - r1) * Betas) * np.sin(Betas)
yS1 = (r1 + (r2 - r1) * Betas) * np.cos(Betas)
xS2 = (r1 + (r2 - r1) * Betas) * np.sin(Betas)
yS2 = (r1 + (r2 - r1) * Betas) * np.cos(Betas)


fig0 = plt.figure(figsize=[15, 7])

ax2 = fig0.add_subplot(2, 2, 1)
ax2.plot(t, phi, color=[0, 1, 0])
ax2.set_title("Phi(t)")

ax3 = fig0.add_subplot(2, 2, 3)
ax3.plot(t, psi, color=[0, 0, 1])
ax3.set_title("Psi(t)")

ax = fig0.add_subplot(1, 2, 2)
ax.axis("equal")
ax.set(xlim=[-0.25, 20], ylim=[-0.25, 20])

ax.plot([x0 - 2, x0 + 2], [y0 - 1, y0 - 1], color=[0, 0.5, 0], linewidth=4)
ax.plot([x0 - 1, x0, x0 + 1], [y0 - 1, y0, y0 - 1], color=[0, 0, 0])
O = ax.plot(xO, yO, "o", color=[0, 1, 0])[0]
SpiralnayaPruzzhina1 = ax.plot(xS1 + xO, yS1 + yO, color=[1, 0.5, 0.5])[0]
A = ax.plot(xA, yA, "o", color=[0, 1, 0])[0]
OA = ax.plot([xO, xA[0]], [yO, yA[0]], color=[0, 0, 0])[0]
SpiralnayaPruzzhina2 = ax.plot(xS2 + xA[0], yS2 + yA[0], color=[1, 0.5, 0.5])[0]
B = ax.plot(xB, yB, "o", color=[0, 1, 0])[0]
AB = ax.plot([xA[0], xB[0]], [yA[0], yB[0]], color=[0, 0, 0])[0]
TraceBX, TraceBY = np.array([xB[0]]), np.array([yB[0]])
TraceB = ax.plot(TraceBX, TraceBY, ":", color="black")[0]


def kadr(i):
    Betas1 = Beta0 * (N * 2 * np.pi + psi[i])
    xS1 = (r1 + (r2 - r1) * Betas1 / (N * 2 * np.pi + psi[i])) * np.sin(Betas1)
    yS1 = (r1 + (r2 - r1) * Betas1 / (N * 2 * np.pi + psi[i])) * np.cos(Betas1)
    SpiralnayaPruzzhina1.set_data(xS1 + xO, yS1 + yO)
    A.set_data(xA[i], yA[i])
    OA.set_data([xO, xA[i]], [yO, yA[i]])
    Betas2 = Beta0 * (N * 2 * np.pi + phi[i])
    xS2 = (r1 + (r2 - r1) * Betas2 / (N * 2 * np.pi + phi[i])) * np.sin(Betas2)
    yS2 = (r1 + (r2 - r1) * Betas2 / (N * 2 * np.pi + phi[i])) * np.cos(Betas2)
    SpiralnayaPruzzhina2.set_data(xS2 + xA[i], yS2 + yA[i])
    B.set_data(xB[i], yB[i])
    AB.set_data([xA[i], xB[i]], [yA[i], yB[i]])
    global TraceBX, TraceBY
    TraceBX = np.append(TraceBX, xB[i])
    TraceBY = np.append(TraceBY, yB[i])
    TraceB.set_data(TraceBX, TraceBY)
    return [SpiralnayaPruzzhina1, A, OA, SpiralnayaPruzzhina2, B, AB, TraceB]


kino = FuncAnimation(fig0, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
