import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Функция изменения угла во времени
def phi_of_t(t):
    return 0.5 * np.sin(0.5 * t)

# Параметры диска, пружины и стержня
R = 0.5  # Радиус диска
r = 0.1  # Радиус маленького диска на конце стержня
M = 1  # Масса диска
k = 1  # Жесткость пружины
A = 0.5  # Амплитуда колебаний пружины
beta = 0.1  # Коэффициент затухания
omega = np.sqrt(k / M)  # Угловая частота с учетом массы

# Углы поворота
phi = np.pi / 2  # Начальное смещение всей системы от вертикали (угол Фи)

# Временной ряд
t = np.linspace(0, 10, 1001)
theta = A * np.exp(-beta * t) * np.sin(omega * t + phi_of_t(t))  # Угловое положение диска во времени

# Параметры для спиральной пружины
N = 4
r1 = 0.03
r2 = 0.1
Beta0 = np.linspace(0, 1, 50 * N + 1)

# Фигура и оси
fig, ax = plt.subplots(figsize=[13, 9])
ax.axis('equal')
ax.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5])

# Создание диска и отметки на нем
circle = plt.Circle((0, 0), R, color='blue', fill=False, linewidth=2)
ax.add_artist(circle)

# Создание спиральной пружины
spiral_spring, = ax.plot([], [], color=[1, 0.5, 0.5])

# Создание стержня
rod, = ax.plot([], [], color='black', linewidth=2)

# Создание маленького диска на конце стержня
small_circle = plt.Circle((0, 0), r, color='black', fill=False, edgecolor='black', linewidth=2)
ax.add_artist(small_circle)

def update_frame(i):
    # Положение пружины
    Betas = Beta0 * (N * 2 * np.pi + theta[i])
    xS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi + theta[i])) * np.sin(Betas)
    yS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi + theta[i])) * np.cos(Betas)
    spiral_spring.set_data(xS, yS)

    # Положение стержня и маленького диска
    x_rod_end = (R - r) * np.cos(theta[i])
    y_rod_end = (R - r) * np.sin(theta[i])
    rod.set_data([0, x_rod_end], [0, y_rod_end])
    small_circle.center = x_rod_end, y_rod_end

    return circle, spiral_spring, rod, small_circle

# Создание и запуск анимации с обновленными параметрами
anim = FuncAnimation(fig, update_frame, frames=len(t), interval=20, blit=True)

plt.show()
