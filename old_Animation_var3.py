from matplotlib.animation import FuncAnimation
import matplotlib
from math import pi

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1001)
psi = 2.2 * np.sin(t)
phi = 1.5 * np.cos(0.8 * t)
round = np.linspace(0, 6.29, 50)


L = 3  # длина палки
R = 2  # радиус циллиндра
m = 1  # масса циллиндра
M = 1  # масса маятника
xk = 2  # точка касания поверхности и циллиндра
alpha = pi / 5  # угол наклона поверхности

k = np.tan(pi - alpha)

x2 = 10
y2 = 0

x1 = 0
y1 = y2 - k * x2 + k * x1
b = y2 - k * x2 + k * x1

xOffset = R * np.cos(pi / 2 - alpha)
yOffset = R * np.sin(pi / 2 - alpha)

xO = t
yO = k * xO + b

xC = R * np.sin(round) + xOffset
yC = R * np.cos(round) + yOffset


fig = plt.figure(figsize=(5, 5))  # задаем фигуру
ax = fig.add_subplot(1, 1, 1)  # график сетки
ax.axis("equal")
ax.set(xlim=[-0.25, 10], ylim=[-0.25, 10])  # задаем размеры сетки


ax.plot([x1, x2], [y1, y2], color=[0, 0, 0])
# ax.plot([x1 + xOffset, x2 + xOffset], [y1 + yOffset, y2 + yOffset], color=[0, 0, 0])
O = ax.plot(xO + xOffset, yO + yOffset, "o", color=[0, 1, 0])[0]
Cilinder = ax.plot(xC, yC, color=[0, 0, 0])[0]


def kadr(i):
    O.set_data(xO[i] + xOffset, yO[i] + yOffset)
    Cilinder.set_data(xC + xO[i], yC + yO[i])
    return [O, Cilinder]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
