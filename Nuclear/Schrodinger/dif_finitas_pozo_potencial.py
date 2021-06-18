#!/usr/bin/env python3

from time import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12.5})  # cambiar el tamaño de la fuente en las figuras

### CONTROL ###

directorio_figuras = ''

###

V_0 = 244                       # eV
anchura_caja = 1e-10            # m
masa_electron = 0.511e6         # ev
h_barra = 6.582e-16             # eV·s
c = 3e8                         # m/s
k = 2*masa_electron*anchura_caja**2/h_barra**2*V_0/c**2

###

du = 0.01                       # resolución del mallado espacial
umin = -2                       # valor mínimo del mallado
umax = 2 + du                   # valor máximo del mallado
u = np.arange(umin, umax, du)   # mallado espacial
n = len(u)                      # número de puntos en el mallado
centro = int(n/2)               # índice del punto medio del mallado

### FUNCIONES ###

def calculo_psi(alfa, paridad: str):
    '''
    Implementación del método de diferencias finitas para el cálculo de soluciones a la ecuación
    de Schrodinger en un pozo de potencial finito

    Parámetros:
        alfa (float): valor de energía (E = alfa*V_0)
        paridad (string): paridad de la solución a calcular. par o impar
    '''
    # arrays para almacenar las soluciones
    psi = np.zeros_like(u)
    psi_prima = np.zeros_like(u)

    # condiciones si la solución es par
    if paridad == 'par':
        psi_prima[centro] = 0
        psi[centro] = 1

    # condiciones si la solución es impar
    elif paridad == 'impar':
        psi_prima[centro] = 1
        psi[centro] = 0

    else:
        raise ValueError('Paridad no válida: la función debe ser par o impar')

    # array para almacenar el valor de la functión f en todo el mallado
    cte = np.zeros_like(u)
    for j in range(centro, n):
        if abs(u[j]) > 0.5:
            cte[j] = k*(1 - alfa)
        if abs(u[j]) <= 0.5:
            cte[j] = - k*alfa

    # comienzo de iteración recursiva
    for i in range(centro, n-1):
        psi[i+1] = psi[i] + psi_prima[i]*du               # cálculo de la función
        psi_prima[i+1] = psi_prima[i] + cte[i]*psi[i]*du  # cálculo de la derivada

    return psi


def shooting_pozo(alfa_1, paridad, num_sols, delta_alfa=0.01, tol=1e-10, max_iter=1000):
    '''
    Implementación del shooting method para soluciones a la ecuación de Schrodinger en el
    pozo de potencial.

    Parámetros:
        alfa_1 (float): valor de energía inicial (E = alfa*V_0)
        paridad (str): paridad de la función de onda a calcular. par o impar 
        num_sols (int): número de soluciones a calcular. No introducir más de 2
        tol (float): tolerancia del método
        delta_alfa (float): separación entre valores de alfa para los que calcular la función de onda
        max_iter (int): número máximo de iteraciones

    Devuelve:
        alfas (array): contiene los valores de la energía para cada solución calculada
        psis (array): contiene las funciones de onda calculadas

    '''
    # arrays para almacenar las soluciones calculadas
    alfas = np.zeros(num_sols)
    psis = np.zeros((num_sols, n))

    # aplicar el shooting method num_sols veces
    for i in range(num_sols):
        alfa_2 = alfa_1 + delta_alfa
        psi_1 = calculo_psi(alfa_1, paridad)
        psi_2 = calculo_psi(alfa_2, paridad)

        # buscar dos valores de alfa tales que se pueda aplicar bisección
        while psi_1[-1]*psi_2[-1] > 0:
            alfa_2 += delta_alfa
            psi_2 = calculo_psi(alfa_2, paridad)

        # comienzo bisección
        alfa_n = 0.5*(alfa_1+alfa_2)
        psi_n = calculo_psi(alfa_n, paridad)

        for k in range(max_iter):
            # chequear si se a alcanzado la tolerancia para finalizar bisección
            if abs(psi_n[-1]) <= tol:
                break

            if np.sign(psi_n[-1]) == np.sign(psi_1[-1]):
                alfa_1 = alfa_n
                psi_1 = psi_n

            elif np.sign(psi_n[-1]) == np.sign(psi_2[-1]):
                alfa_2 = alfa_n
                psi_2 = psi_n

            alfa_n = 0.5*(alfa_1 + alfa_2)
            psi_n = calculo_psi(alfa_n, paridad)

        alfas[i] = alfa_n           # almacenar energía en el array de soluciones

        # reconstrucción de la función en todo el mallado en función de su paridad
        if paridad == 'par':
            psi_n[:centro] = + psi_n[:centro:-1]
        elif paridad[:centro] == 'impar':
            psi_n[:centro] = - psi_n[:centro:-1]

        psi_n = psi_n/np.sqrt(np.sum(psi_n**2*du))  # normalizar función de onda
        psis[i] = psi_n                             # almacenar función en el array de soluciones
        alfa_1 = alfa_n + delta_alfa            # nuevo alfa inicial para encontrar más soluciones

    return alfas, psis

def representacion_individual(alfas, psis, save=False):
    '''
    Función que representa de una en una las funciones de onda en la caja de potencial.

    Parámetros:
        alfas (array): valores de energía de las funciones de onda
        psis (array): funciones de onda a representar
        save (bool): True para guardar las figuras en el directorio indicado por directorio_figuras
    '''
    for i in range(len(alfas)):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(u, psis[i])
        ax.set_xlabel('u=r/a')
        ax.set_title(r'$\alpha = {:.3f}$'.format(alfas[i]))
        ax.axvline(-0.5, ls='--', lw=1, c='grey')
        ax.axvline(0.5, ls='--', lw=1, c='grey')
        ax.set_ylabel(r'$\Psi(u)$')
        fig.tight_layout()
        if save:
            fig.savefig(directorio_figuras + 'dif_finitas_alfa_{:.3f}.pdf'.format(alfas[i]))
        plt.show()

###

alfa_0 = 0.5*np.pi**2/k

t_1 = time()
alfas_par, psis_par = shooting_pozo(alfa_0, 'par', 2)
alfa_impar, psi_impar = shooting_pozo(alfa_0, 'impar', 1)
print('Tiempo transcurrido: {:.2f} segundos'.format(time() - t_1))
representacion_individual(alfas_par, psis_par, False)
representacion_individual(alfa_impar, psi_impar, False)
