import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


t = np.linspace(0, 10, 1001)  # сетка по времени
x = 0.5 * (1 + 0.3 * np.sin(0.8 * t) + 0.5 * np.cos(3.43 * t))
psi = 2.2 * np.sin(t)
phi = 1.5 * np.cos(0.8 * t)
Alpha = np.linspace(0, 6.29, 50)

L = 3  # длина палки
R = 2  # радиус циллиндра
r = 0.5  # радиус диска
m = 1  # масса диска
M = 1  # масса материальной точки
x0 = 5  # положение центра системы
y0 = 5  # положение центра системы

# описание циллиндра
xC = x0 + R * np.sin(Alpha)
yC = y0 + R * np.cos(Alpha)

# описание поведения точки A
xA = x0 + (R - r) * np.sin(psi)
yA = y0 - (R - r) * np.cos(psi)

# описание поведения материальной точки B
xB = xA + L * np.sin(phi)
yB = yA - L * np.cos(phi)


fig0 = plt.figure(figsize=[15, 7])
ax1 = fig0.add_subplot(2, 4, 1)
ax1.plot(t, x, color=[1, 0, 0])
ax1.set_title("X(t)")

ax2 = fig0.add_subplot(2, 4, 2)
ax2.plot(t, phi, color=[0, 1, 0])
ax2.set_title("Phi(t)")

ax3 = fig0.add_subplot(2, 2, 3)
ax3.plot(t, psi, color=[0, 0, 1])
ax3.set_title("Psi(t)")

ax = fig0.add_subplot(1, 2, 2)
ax.axis("equal")
ax.set(xlim=[-0.25, 10], ylim=[-0.25, 10])  # задаем размеры сетки

TraceBX, TraceBY = np.array([xB[0]]), np.array([yB[0]])
TraceB = ax.plot(TraceBX, TraceBY, ":", color="black")[0]
Cilinder = ax.plot(xC, yC, color=[0, 0, 0])[0]
Disk = ax.plot(xA, yA, "o", color=[0, 0, 0], markersize=r * 50)[0]
A = ax.plot(xA, yA, "o", color=[0, 1, 0])[0]
B = ax.plot(xB[0], yB[0], "o", color=[0, 0, 0], markersize=20)[0]
AB = ax.plot([xA[0], xB[0]], [yA[0], yB[0]], color=[1, 0, 0])[0]


def kadr(i):
    global TraceBX, TraceBY
    Disk.set_data(xA[i], yA[i])
    A.set_data(xA[i], yA[i])
    B.set_data(xB[i], yB[i])
    AB.set_data([xA[i], xB[i]], [yA[i], yB[i]])
    TraceBX = np.append(TraceBX, xB[i])
    TraceBY = np.append(TraceBY, yB[i])
    TraceB.set_data(TraceBX, TraceBY)
    return [Disk, A, B, AB, TraceB]


kino = FuncAnimation(fig0, kadr, interval=t[1] - t[0], frames=len(t))

plt.show()
