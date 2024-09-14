import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

t = np.linspace(0, 10, 1001)  # сетка по времени
psi = 2.2 * np.sin(t)
phi = 1.5 * np.cos(0.8 * t)
round = np.linspace(0, 6.29, 50)

R = 6
r = 1
L = R - r

x0 = 10
y0 = 10

xO = x0
yO = y0

xDisk = R * np.sin(round)
yDisk = R * np.cos(round)

xSharn = r * np.sin(round)
ySharn = r * np.cos(round)

N = 4
r1 = 0.1
r2 = 0.8
Beta0 = np.linspace(0, 1, 50 * N + 1)
Betas = Beta0
xS = (r1 + (r2 - r1) * Betas) * np.sin(Betas)
yS = (r1 + (r2 - r1) * Betas) * np.cos(Betas)

xB = xO - L * np.sin(phi)
yB = yO - L * np.cos(phi)

xA = xO + R * np.sin(psi)
yA = yO + R * np.cos(psi)

fig0 = plt.figure(figsize=[15, 7])

ax2 = fig0.add_subplot(2, 2, 1)
ax2.plot(t, phi, color=[0, 1, 0])
ax2.set_title("Phi(t)")

ax3 = fig0.add_subplot(2, 2, 3)
ax3.plot(t, psi, color=[0, 0, 1])
ax3.set_title("Psi(t)")

ax = fig0.add_subplot(1, 2, 2)
ax.axis("equal")
ax.set(xlim=[-0.25, 20], ylim=[-0.25, 20])  # задаем размеры сетки

ax.plot([x0 - 2, x0 + 2], [y0 - 2, y0 - 2], color=[0, 0.5, 0], linewidth=4)
ax.plot([x0 - 1, x0, x0 + 1], [y0 - 2, y0, y0 - 2], color=[0, 0, 0])
O = ax.plot(xO, yO, "o", color=[0, 1, 0])[0]
Disk = ax.plot(xDisk, yDisk, color=[0, 0, 0])[0]
SpiralnayaPruzzhina = ax.plot(xS + xO, yS + yO, color=[1, 0.5, 0.5])[0]
B = ax.plot(xB, yB, "o", color=[0, 1, 0])[0]
OB = ax.plot([xO, xB[0]], [yO, yB[0]], color=[0, 0, 0])[0]
Sharn = ax.plot(xSharn, ySharn, color=[0, 0, 0])[0]
A = ax.plot(xA, yA, "o", color=[0, 1, 0])[0]
OA = ax.plot([xO, xA[0]], [yO, yA[0]], color=[0, 0, 0])[0]
TraceBX, TraceBY = np.array([xB[0]]), np.array([yB[0]])
TraceB = ax.plot(TraceBX, TraceBY, ":", color="black")[0]

def kadr(i):
    Betas = Beta0 * (N * 2 * np.pi + psi[i])
    xS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi + psi[i])) * np.sin(Betas)
    yS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi + psi[i])) * np.cos(Betas)
    SpiralnayaPruzzhina.set_data(xS + xO, yS + yO)
    Disk.set_data(xDisk + xO, yDisk + yO)
    B.set_data(xB[i], yB[i])
    OB.set_data([xO, xB[i]], [yO, yB[i]])
    Sharn.set_data(xSharn + xB[i], ySharn + yB[i])
    A.set_data(xA[i], yA[i])
    OA.set_data([xO, xA[i]], [yO, yA[i]])
    global TraceBX, TraceBY
    TraceBX = np.append(TraceBX, xB[i])
    TraceBY = np.append(TraceBY, yB[i])
    TraceB.set_data(TraceBX, TraceBY)
    return [SpiralnayaPruzzhina, Disk, B, OB, Sharn, A, OA, TraceB]


kino = FuncAnimation(fig0, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
