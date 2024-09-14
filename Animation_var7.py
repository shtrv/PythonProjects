from matplotlib.animation import FuncAnimation
import matplotlib
from math import pi

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1001)
ksi = 1.8 * np.sin(0.73 * t)
phi = 0.6 * np.sin(1.53 * t)
Alpha = np.linspace(0, 6.29, 50)

H = 4
W = 8
h = 0.5
w = 1
x0 = 10
y0 = 5
offs = np.arctan(H / W)


xO = x0
yO = y0

xC = xO - H / 2 * np.sin(phi)
yC = yO + H / 2 * np.cos(phi)

diag = ((H / 2) ** 2 + (W / 2) ** 2) ** 0.5
diag2 = ((h / 2) ** 2 + (w / 2) ** 2) ** 0.5

xp1 = xC - (diag) * np.sin(pi / 2 - phi - offs)
yp1 = yC - (diag) * np.cos(pi / 2 - phi - offs)
xp2 = xC + (diag) * np.sin(pi / 2 - phi + offs)
yp2 = yC + (diag) * np.cos(pi / 2 - phi + offs)
xp3 = xC + (diag) * np.sin(pi / 2 - phi - offs)
yp3 = yC + (diag) * np.cos(pi / 2 - phi - offs)
xp4 = xC - (diag) * np.sin(pi / 2 - phi + offs)
yp4 = yC - (diag) * np.cos(pi / 2 - phi + offs)

xf = xO - (H + h / 2) * np.sin(phi)
yf = yO + (H + h / 2) * np.cos(phi)

xM = xf + (W / 2 * np.cos(ksi)) * np.sin(pi / 2 - phi)
yM = yf + (W / 2 * np.cos(ksi)) * np.cos(pi / 2 - phi)

xm1 = xM - (diag2) * np.sin(pi / 2 - phi - offs)
ym1 = yM - (diag2) * np.cos(pi / 2 - phi - offs)
xm2 = xM + (diag2) * np.sin(pi / 2 - phi + offs)
ym2 = yM + (diag2) * np.cos(pi / 2 - phi + offs)
xm3 = xM + (diag2) * np.sin(pi / 2 - phi - offs)
ym3 = yM + (diag2) * np.cos(pi / 2 - phi - offs)
xm4 = xM - (diag2) * np.sin(pi / 2 - phi + offs)
ym4 = yM - (diag2) * np.cos(pi / 2 - phi + offs)

N = 4
r1 = 0.1
r2 = 0.8
Beta0 = np.linspace(0, 1, 50 * N + 1)
Betas = Beta0
xS = (r1 + (r2 - r1) * Betas) * np.sin(Betas)
yS = (r1 + (r2 - r1) * Betas) * np.cos(Betas)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")
ax.set(xlim=[-0.25, 20], ylim=[-0.25, 20])

ax.plot([8, 12], [y0 - 2, y0 - 2], color=[0, 0.5, 0], linewidth=4)
ax.plot([x0 - 1, x0, x0 + 1], [y0 - 2, y0, y0 - 2], color=[0, 0, 0])
O = ax.plot(xO, yO, "o", color=[0, 1, 0])[0]
C = ax.plot(xC, yC, "o", color=[0, 1, 0])[0]
Platforma = ax.plot(
    [xp1[0], xp2[0], xp3[0], xp4[0], xp1[0]],
    [yp1[0], yp2[0], yp3[0], yp4[0], yp1[0]],
    color=[0, 0, 0],
)[0]
# M = ax.plot(xM, yM, "o", color=[0, 1, 0])[0]
Shaiba = ax.plot(
    [xm1[0], xm2[0], xm3[0], xm4[0], xm1[0]],
    [ym1[0], ym2[0], ym3[0], ym4[0], ym1[0]],
    color=[0, 0, 0],
)[0]
SpiralnayaPruzzhina = ax.plot(xS + xO, yS + yO, color=[1, 0.5, 0.5])[0]


def kadr(i):
    C.set_data(xC[i], yC[i])
    Platforma.set_data(
        [xp1[i], xp2[i], xp3[i], xp4[i], xp1[i]],
        [yp1[i], yp2[i], yp3[i], yp4[i], yp1[i]],
    )
    # M.set_data(xM[i], yM[i])
    Shaiba.set_data(
        [xm1[i], xm2[i], xm3[i], xm4[i], xm1[i]],
        [ym1[i], ym2[i], ym3[i], ym4[i], ym1[i]],
    )
    Betas = Beta0 * (N * 2 * np.pi + phi[i])
    xS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi + phi[i])) * np.sin(Betas)
    yS = (r1 + (r2 - r1) * Betas / (N * 2 * np.pi + phi[i])) * np.cos(Betas)
    SpiralnayaPruzzhina.set_data(-xS + xO, yS + yO)
    return [C, Platforma, Shaiba, SpiralnayaPruzzhina]


kino = FuncAnimation(fig, kadr, interval=t[1] - t[0], frames=len(t))
plt.show()
