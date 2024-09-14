from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

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


fig = plt.figure(figsize=(5, 5))  # задаем фигуру
ax = fig.add_subplot(1, 1, 1)  # график сетки
ax.axis("equal")
ax.set(xlim=[-0.25, 10], ylim=[-0.25, 10])  # задаем размеры сетки

Cilinder = ax.plot(xC, yC, color=[0, 0, 0])[0]
Disk = ax.plot(xA, yA, "o", color=[0, 0, 0], markersize=r * 50)[0]
A = ax.plot(xA, yA, "o", color=[0, 1, 0])[0]
B = ax.plot(xB[0], yB[0], "o", color=[0, 0, 0], markersize=20)[0]
AB = ax.plot([xA[0], xB[0]], [yA[0], yB[0]], color=[1, 0, 0])[0]


def kadr(i):
    Disk.set_data(xA[i], yA[i])
    A.set_data(xA[i], yA[i])
    B.set_data(xB[i], yB[i])
    AB.set_data([xA[i], xB[i]], [yA[i], yB[i]])

    return [Disk, A, B, AB]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))

plt.show()
