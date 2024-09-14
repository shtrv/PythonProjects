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

t = np.linspace(0, 10, 1001)
ksi = 2 * np.sin(0.8 * t)
round = np.linspace(0, 6.29, 50)
phi = 1.5 * np.cos(0.8 * t)

r = 2  # радиус диска
R = 5  # радиус платформы
m = 1  # масса диска
M = 1  # масса платформы
H = R + 1  # высота платформы
W = 2 * R  # длина платформы


xC = 8 + ksi
yC = H

xPlatf = np.array([-R, -R - 0.5, -R - 0.5, R + 0.5, R + 0.5, R])
yPlatf = np.array([0, 0, -H, -H, 0, 0])

xRoundPlatf = R * np.sin(round)
yRoundPlatf = -R * abs(np.cos(round))

xD = xC + (R - r) * np.sin(phi)
yD = yC - (R - r) * np.cos(phi)

xDisk = r * np.sin(round)
yDisk = r * np.cos(round)

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

ax.plot([0, 20], [0, 0], color=[0, 0, 0])
C = ax.plot(xC[0], yC, "o", color=[0, 1, 0])[0]
Platf = ax.plot(xC[0] + xPlatf, yC + yPlatf, color=[0, 0, 0])[0]
RondPlatf = ax.plot(xC[0] + xRoundPlatf, yC + yRoundPlatf, color=[0, 0, 0])[0]
D = ax.plot(xD[0], yD[0], "o", color=[0, 1, 0])[0]
Disk = ax.plot(xDisk, yDisk, color=[0, 0, 0])[0]
TraceDX, TraceDY = np.array([xD[0]]), np.array([yD[0]])
TraceD = ax.plot(TraceDX, TraceDY, ":", color="black")[0]


def kadr(i):
    C.set_data(xC[i], yC)
    Platf.set_data(xC[i] + xPlatf, yC + yPlatf)
    RondPlatf.set_data(xC[i] + xRoundPlatf, yC + yRoundPlatf)
    D.set_data(xD[i], yD[i])
    Disk.set_data(xD[i] + xDisk, yD[i] + yDisk)
    global TraceDX, TraceDY
    TraceDX = np.append(TraceDX, xD[i])
    TraceDY = np.append(TraceDY, yD[i])
    TraceD.set_data(TraceDX, TraceDY)
    return [C, Platf, RondPlatf, D, Disk, TraceD]


kino = FuncAnimation(fig0, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
