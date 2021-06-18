#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time

# datos del sistema (S.I)
c = 3e8                         # velocidad de la luz (m/s)
dx = 10e-9                      # tamaño del paso espacial
dt = 0.5*dx/c                   # tamaño del paso temporal
puntos_x = 1001                 # número de puntos en x
max_x = (puntos_x-1)*dx
x = np.arange(0, max_x, dx)     # array de posiciones
epsilon0 = 8.85e-12
epsilon_1 = 1
epsilon_2 = 4
conductividad_1 = 0
conductividad_2 = 4000

# datos pulso temporal
E0 = 1                          # intensidad del pulso
delta_x = 400e-9
delta_t = delta_x/c             # anchura del pulso
tp = 5*delta_t
x_pulso = 250                   # punto de genración del pulso


# datos de la simulacion
n = 10000                       # nº de pasos temporales
n_plot = 10                     # numero de pasos entre representaciones
t_pausa = 0.01
Ey = np.zeros(puntos_x)
Hz = np.zeros(puntos_x)

# arrays pertinentes
epsilon = np.ones(puntos_x)
epsilon[500:] = epsilon_2
conductividad = np.zeros(puntos_x)
conductividad[500:] = conductividad_2
a = conductividad*dt/2/epsilon0/epsilon


fig, ax = plt.subplots()
eyplot, = ax.plot([], [], label=r'E_y')
hzplot, = ax.plot([], [], label=r'H_z', c='red')
ax.set_xlim(0, max_x)
ax.axvline(x[500], c='black', ls='dashed')
ax.set_ylim(-1.4*E0, 1.4*E0)
ax.set_xlabel('Posición (m)')
ax.set_ylabel('Intensidad')
ax.set_title('t = 0')
ax.annotate('$\epsilon_r = %i$\n$\sigma = %i$' % (epsilon_1, conductividad_1), (x[30], 1.1*E0))
ax.annotate('$\epsilon_r = %i$\n$\sigma = %i$' % (epsilon_2, conductividad_2), (x[870], 1.1*E0))
fig.tight_layout()

# Condiciones de contorno
contorno_izda = np.zeros(np.round(2*np.sqrt(epsilon_1)).astype(int))
contorno_dcha = np.zeros(np.round(2*np.sqrt(epsilon_2)).astype(int))


t = 0
for i in range(n):
    Ey[x_pulso] = Ey[x_pulso] + E0*np.exp(-0.5*(t-tp)**2/delta_t**2)
    Ey[1:] = ((1-a[1:])/(1+a[1:]))*Ey[1:] - 0.5*1/epsilon[1:]/(1+a[1:])*(Hz[1:] - Hz[:-1])

    # C.C IZDA
    Ey[0] = contorno_izda[0]
    contorno_izda = np.roll(contorno_izda, -1)
    contorno_izda[-1] = Ey[1]

    # C.C DCHA
    Ey[-1] = contorno_dcha[-1]
    contorno_dcha = np.roll(contorno_dcha, 1)
    contorno_dcha[0] = Ey[-2]

    Hz[:-1] = Hz[:-1] - 0.5*(Ey[1:] - Ey[:-1])

    if i % n_plot == 0:
        eyplot.set_data(x, Ey)
        hzplot.set_data(x, Hz)
        ax.set_title('t = %.1e s' % (t))
        plt.pause(t_pausa)

    t += dt
plt.show()
