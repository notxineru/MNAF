#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


### DATOS DEL SISTEMA ###

lado_caja = 10
masa = 1
r_disco = 0.125
n = 100                                                  # número de partículas
x = np.random.rand(n, 2)*lado_caja                       # posición inicial
theta0 = np.random.rand(n)*2*np.pi
direccion = np.array([np.cos(theta0), np.sin(theta0)])   # dirección inicial

###

Kb = 0.01                                                # constante de boltzman
temperatura_inicial = 50
energia_inicial = np.random.exponential(Kb*temperatura_inicial, n)
v0 = np.sqrt(2*energia_inicial/masa)
v = (v0*direccion).T                                     # velocidad inicial


### DATOS DE LA SIMULACIÓN ###

dt = 0.01          # Intervalo temporal
max_iter = 10000   # Número de pasos
n_plot = 10        # Número de pasos entre representaciones
t_stop = 0.01      # Pausa entre representaciones
tiempo = np.array([i*dt for i in range(max_iter)])  # array de tiempos


def choque2(v1, v2, r12):
    uc = r12 / np.linalg.norm(r12)
    up = np.array([uc[1], -uc[0]])

    v1c = np.dot(v1, uc)
    v1p = np.dot(v1, up)
    v2c = np.dot(v2, uc)
    v2p = np.dot(v2, up)

    v1f = v2c*uc + v1p*up
    v2f = v1c*uc + v2p*up

    return v1f, v2f


def main(x, PLOT=True):

    presion = np.zeros(max_iter)

    if PLOT:
        fig, ax = plt.subplots(figsize=(6, 5.6))
        ax.set_xlim((0, lado_caja))
        ax.set_ylim((0, lado_caja))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Evolución del sistema. T = 0')
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        punto, = ax.plot([], [], 'o', c='black')

    for i in range(max_iter):
        x += v*dt
        tree = cKDTree(x)
        pares = tree.query_pairs(2*r_disco)

        for j in range(n):
            if (x[j, 0] > lado_caja and v[j, 0] > 0) or (x[j, 0] < 0 and v[j, 0] < 0):
                presion[i] += masa/4/lado_caja*2*abs(v[j, 0])/dt
                v[j, 0] = -v[j, 0]
            if (x[j, 1] > lado_caja and v[j, 1] > 0) or (x[j, 1] < 0 and v[j, 1] < 0):
                presion[i] += masa/4/lado_caja*2*abs(v[j, 1])/dt
                v[j, 1] = -v[j, 1]

        for par in pares:
            v1 = v[par[0], :]
            v2 = v[par[1], :]
            r12 = x[par[0], :] - x[par[1], :]
            if np.dot(v1, r12) - np.dot(v2, r12) < 0:
                (v[par[0], :], v[par[1], :]) = choque2(v1, v2, r12)

        if PLOT and i % n_plot == 0:
            punto.set_data(x[:, 0], x[:, 1])
            ax.set_title('Evolución del sistema. T = %.1f' % (i*dt))
            plt.pause(t_stop)

    if PLOT:
        plt.show()

    return presion

def representacion_presion(presion, tiempo, intervalo=200):

    presion = np.mean(presion.reshape(-1, intervalo), axis=1)
    tiempo = tiempo[::intervalo]

    fig, ax = plt.subplots()
    ax.plot(tiempo, presion, c='black')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Presión')
    ax.set_title('Presión en la caja en función del tiempo')
    plt.show()


def histograma_energia():

    energia_final = np.array([0.5*masa*(v[k, 0]**2+v[k, 1]**2) for k in range(n)])
    energia_final_media = np.mean(energia_final)

    fig, ax = plt.subplots()
    ax.hist(x=energia_final, bins=25, range=(0, 2*round(energia_final_media)))
    ax.set_xlabel('Energía')
    ax.set_ylabel('nº de partículas')
    ax.set_title('Distribución de energías')
    fig.tight_layout()
    plt.show()

    return energia_final_media


def analisis(energia_final_media):
    temperatura_final = energia_final_media/Kb
    presion_media = np.mean(presion)

    valor1_ideal = presion_media*lado_caja**2/n/Kb/temperatura_final  # comparación con el gas ideal

    area_reducida = lado_caja**2 - n*np.pi*r_disco**2
    valor2_real = presion_media*area_reducida/n/Kb/temperatura_final  # comparación con el gas real

    print('P_simulación/P_ideal = {:.2f}, P_simulación/P_real = {:.2f}. Cuantos más cercanos a 1 \
        estos valores, más realista la simulación.'.format(valor1_ideal, valor2_real))
    return(valor1_ideal, valor2_real)

presion = main(x)
representacion_presion(presion, tiempo)
energia_final_media = histograma_energia()
valor1_ideal, valor2_ideal = analisis(energia_final_media)
