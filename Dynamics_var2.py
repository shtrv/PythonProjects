import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import pi

matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

t = np.linspace(0, 10, 1001)
phi = 1.62 * np.cos(4.33 * t)
ksi = 1.47 * np.sin(1.2 * t) + 1.5
round = np.linspace(0, 6.29, 50)

L = 7  # длина нити
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


fig0 = plt.figure(figsize=[15, 7])
ax1 = fig0.add_subplot(2, 4, 1)
ax1.plot(t, phi, color=[1, 0, 0])
ax1.set_title("X(t)")

ax2 = fig0.add_subplot(2, 4, 2)
ax2.plot(t, phi, color=[0, 1, 0])
ax2.set_title("Phi(t)")

ax3 = fig0.add_subplot(2, 2, 3)
ax3.plot(t, ksi, color=[0, 0, 1])
ax3.set_title("Ksi(t)")

ax = fig0.add_subplot(1, 2, 2)
ax.axis("equal")
ax.set(xlim=[-0.25, 20], ylim=[-0.25, 20])  # задаем размеры сетки

TraceCX, TraceCY = np.array([xC[0]]), np.array([yC[0]])[0]
TraceC = ax.plot(TraceCX, TraceCY, ":", color="black")[0]
ax.plot([8, 12], [18, 18], color=[0, 0.5, 0], linewidth=4)
ax.plot([9, xO, 11], [18, yO, 18], color=[0, 0, 0])
O = ax.plot(xO, yO, "o", color=[0, 1, 0])
K = ax.plot(xK[0], yK[0], color=[0, 1, 0])[0]
OK = ax.plot([xO, xK[0]], [yO, yK[0]], color=[0, 0, 0])[0]
C = ax.plot(xC[0], yC[0], "o", color=[0, 1, 0])[0]
Kat = ax.plot(xKat, yKat, color=[0, 0, 0])[0]


def kadr(i):
    global TraceCX, TraceCY
    K.set_data(xK[i], yK[i])
    OK.set_data([xO, xK[i]], [yO, yK[i]])
    C.set_data([xC[i], yC[i]])
    Kat.set_data([xKat + xC[i], yKat + yC[i]])
    TraceCX = np.append(TraceCX, xC[i])
    TraceCY = np.append(TraceCY, yC[i])
    TraceC.set_data(TraceCX, TraceCY)
    return [K, OK, C, Kat, TraceC]


kino = FuncAnimation(fig0, kadr, interval=t[1] - t[0], frames=len(t))

plt.show()
