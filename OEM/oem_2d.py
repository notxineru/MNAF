#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time

# datos del sistema (S.I)
c = 3e8                         # velocidad de la luz (m/s)
dx = 10e-9                      # tamaño del paso espacial
dt = 0.5*dx/c                   # tamaño del paso temporal
puntos_x = 401                  # número de puntos en x
max_x = (puntos_x-1)*dx
x = np.linspace(0, max_x, puntos_x)
y = np.linspace(0, max_x, puntos_x)
y, x = np.meshgrid(y, x)
epsilon0 = 8.85e-12
epsilon_1 = 1
epsilon_2 = 2
conductividad_1 = 0
conductividad_2 = 4000

# datos pulso temporal
E0 = 1                          # intensidad del pulso
delta_x = 40e-9
delta_t = delta_x/c             # anchura del pulso
tp = 5*delta_t
pos_pulso = (100, 200)          # punto de genración del pulso


# datos de la simulacion
n = 10000                       # nº de pasos temporales
n_plot = 10                     # numero de pasos entre representaciones
t_pausa = 0.01

Ez = np.zeros((puntos_x, puntos_x))
Hy = np.zeros((puntos_x, puntos_x))
Hx = np.zeros((puntos_x, puntos_x))

# arrays pertinentes
epsilon = np.ones((puntos_x, puntos_x))
epsilon[200:, :] = epsilon_2
conductividad = np.zeros((puntos_x, puntos_x))
conductividad[200:, :] = conductividad_2
a = conductividad*dt/2/epsilon0/epsilon


fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('t = 0')
levels = np.linspace(-0.1, 0.1, 30)
cs = ax.contourf(x*1e6, y*1e6, np.clip(Ez, -0.1, 0.1), levels, cmap='seismic')
bar = plt.colorbar(cs)
fig.tight_layout()

# Condiciones de contorno
contorno_izda = np.zeros((np.round(2*np.sqrt(epsilon_1)).astype(int), puntos_x))
contorno_dcha = np.zeros((np.round(2*np.sqrt(epsilon_2)).astype(int), puntos_x))
contorno_sup = np.zeros((puntos_x, np.round(2*np.sqrt(epsilon_1)).astype(int)))
contorno_inf = np.zeros((puntos_x, np.round(2*np.sqrt(epsilon_1)).astype(int)))

# k pulos = 100
# lop = 200
#
# pared dielectrica en 200
# e1 = 1, e2 = 2

# añadir conductividad :)

t = 0
for i in range(n):
    Ez[1:, 1:] = ((1-a[1:, 1:])/(1+a[1:, 1:]))*Ez[1:, 1:] + 0.5/epsilon[1:, 1:]/(1+a[1:, 1:])*(Hy[1:, 1:] - Hy[:-1, 1:]) - 0.5/epsilon[1:, 1:]/(1+a[1:, 1:])*(Hx[1:, 1:] - Hx[1:, :-1])
    Ez[pos_pulso] = E0*np.exp(-0.5*(t-tp)**2/delta_t**2)
    # C.C IZDA
    Ez[0, :] = contorno_izda[0, :]
    contorno_izda = np.roll(contorno_izda, -1, axis=0)
    contorno_izda[-1, :] = Ez[1, :]

    # C.C DCHA
    Ez[-1, :] = contorno_dcha[-1, :]
    contorno_dcha = np.roll(contorno_dcha, 1, axis=0)
    contorno_dcha[0, :] = Ez[-2, :]

    # C.C INF
    Ez[:, 0] = contorno_inf[:, 0]
    contorno_inf = np.roll(contorno_inf, -1, axis=1)
    contorno_inf[:, -1] = Ez[:, 1]

    # C.C SUP
    Ez[:, -1] = contorno_sup[:, -1]
    contorno_sup = np.roll(contorno_sup, 1, axis=1)
    contorno_sup[:, 0] = Ez[:, -2]

    Hx[:, :-1] = Hx[:, :-1] - 0.5*(Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] = Hy[:-1, :] + 0.5*(Ez[1:, :] - Ez[:-1, :])

    if i % n_plot == 0:
        ax.clear()
        cs = ax.contourf(x*1e6, y*1e6, np.clip(Ez, -0.1, 0.1), levels, cmap='seismic')
        ax.set_title('t = %.1e s' % (t))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.pause(t_pausa)
    t += dt
plt.show()
