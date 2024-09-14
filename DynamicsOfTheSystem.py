import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


def EqOfMovement(y, t, M, m, L, k, c, g):
    # y = x,phi,x',phi'
    # dy = x',phi',x'',phi''
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = M + m
    a12 = m * L * np.cos(y[1])
    b1 = m * L * np.sin(y[1]) * y[3] ** 2 - k * y[0]  # + Fk - s*y[2]

    a21 = m * L * np.cos(y[1])
    a22 = m * L**2
    b2 = m * g * L * np.sin(y[1]) - c * y[1]  # + Fk*l*np.cos(y[1])

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a21 * a12)
    dy[3] = (a11 * b2 - a21 * b1) / (a11 * a22 - a21 * a12)
    return dy


t_fin = 20
Nt = 2001
t = np.linspace(0, t_fin, Nt)

# разные коэффициенты
M = 0.1
m = 50
L = 1
k = 0
c = 0
g = 9.81

# нач положение
x0 = 0
phi0 = 0.3
dx0 = 0
dphi0 = 0

y0 = [x0, phi0, dx0, dphi0]

Y = odeint(EqOfMovement, y0, t, (M, m, L, k, c, g))
print("Интегрирование завершено")

x = Y[:, 0]
phi = Y[:, 1]
dx = Y[:, 2]
dphi = Y[:, 3]

W = 0.8  # длина тележки
H = 0.2  # высота тележки
rk = 0.08  # радиус колёс
x0 = 0.7  # положение равновесия пружинки

xA = x0 + W / 2 + x
yA = 2 * rk + H / 2
xB = xA + L * np.sin(phi)
yB = yA + L * np.cos(phi)
xT = np.array(
    [
        -W / 2,
        -W / 2,
        W / 2,
        W / 2,
        0.28 * W,
        0.3 * W,
        0.32 * W,
        -0.32 * W,
        -0.3 * W,
        -0.28 * W,
        -W / 2,
    ]
)
yT = np.array(
    [
        -H / 2,
        H / 2,
        H / 2,
        -H / 2,
        -H / 2,
        -H / 2 - rk,
        -H / 2,
        -H / 2,
        -H / 2 - rk,
        -H / 2,
        -H / 2,
    ]
)
Alpha = np.linspace(0, 6.29, 50)
xK = rk * np.sin(Alpha)
yK = rk * np.cos(Alpha)

n = 13
h = 0.05
xP = np.linspace(0, 1, 2 * n + 1)
yP = np.zeros(2 * n + 1)
ss = 0
for i in range(2 * n + 1):
    yP[i] = h * np.sin(ss)
    ss += np.pi / 2

N = 4
r1 = 0.03
r2 = 0.1
Beta0 = np.linspace(0, 1, 50 * N + 1)
Betas = Beta0 * (N * 2 * np.pi - phi[0])
xS = -(r1 + (r2 - r1) * Betas / (N * 2 * np.pi - phi[0])) * np.sin(Betas)
yS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi - phi[0])) * np.cos(Betas)

T = (
    M * dx**2 / 2
    + m * (dx**2 + (dphi * L) ** 2 + 2 * dx * dphi * L * np.cos(phi)) / 2
)
P = k * x**2 / 2 + c * phi**2 / 2 + m * g * L * np.cos(phi)
E = T + P

fig0 = plt.figure(figsize=[15, 7])
ax1 = fig0.add_subplot(2, 4, 1)
ax1.plot(t, x, color=[1, 0, 0])
ax1.set_title("X(t)")

ax2 = fig0.add_subplot(2, 4, 2)
ax2.plot(t, phi, color=[0, 1, 0])
ax2.set_title("Phi(t)")

ax3 = fig0.add_subplot(2, 2, 3)
ax3.plot(t, E, color=[0, 0, 1])
ax3.set_title("Energy")

# fig = plt.figure(figsize=[9,7])
ax = fig0.add_subplot(1, 2, 2)
ax.axis("equal")
ax.set(xlim=[-0.25, 3], ylim=[-0.25, 2])

ax.plot([0, 0, 2.75], [1.75, 0, 0], color=[0, 0.5, 0], linewidth=4)

SpiralnayaPruzzhina = ax.plot(xS + xA[0], yS + yA, color=[1, 0.5, 0.5])[0]
Pruzzhina = ax.plot(xP * (x0 + x[0]), yP + yA, color=[0.5, 0.5, 1])[0]

TraceBX, TraceBY = np.array([xB[0]]), np.array([yB[0]])
TraceB = ax.plot(TraceBX, TraceBY, ":", color="black")[0]
Telega = ax.plot(xA[0] + xT, yA + yT, color=[0, 0, 0])[0]
AB = ax.plot([xA[0], xB[0]], [yA, yB[0]], color=[1, 0, 0])[0]
A = ax.plot(xA[0], yA, "o", color=[1, 0, 0])[0]
B = ax.plot(
    xB[0],
    yB[0],
    "o",
    color=[0, 0.75, 0],
    markersize=20,
    markerfacecolor=[0, 1, 0],
    linewidth=2,
)[0]
K1 = ax.plot(xK + xA[0] - W * 0.3, yK + rk, color=[0, 0, 0])[0]
K2 = ax.plot(xK + xA[0] + W * 0.3, yK + rk, color=[0, 0, 0])[0]


def kadr(i):
    global TraceBX, TraceBY
    A.set_data(xA[i], yA)
    B.set_data(xB[i], yB[i])
    AB.set_data([xA[i], xB[i]], [yA, yB[i]])
    Telega.set_data(xA[i] + xT, yA + yT)
    K1.set_data(xK + xA[i] - W * 0.3, yK + rk)
    K2.set_data(xK + xA[i] + W * 0.3, yK + rk)
    Pruzzhina.set_data(xP * (x0 + x[i]), yP + yA)

    Betas = Beta0 * (N * 2 * np.pi - phi[i])
    xS = -(r1 + (r2 - r1) * Betas / (N * 2 * np.pi - phi[i])) * np.sin(Betas)
    yS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi - phi[i])) * np.cos(Betas)
    SpiralnayaPruzzhina.set_data(xS + xA[i], yS + yA)

    TraceBX = np.append(TraceBX, xB[i])
    TraceBY = np.append(TraceBY, yB[i])
    TraceB.set_data(TraceBX, TraceBY)
    return [A, B, AB, Telega, K1, K2, Pruzzhina, SpiralnayaPruzzhina, TraceB]


kino = FuncAnimation(fig0, kadr, interval=t[1] - t[0], frames=len(t))

plt.show()
