#!/usr/bin/env python3

from time import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12.5})  # cambiar el tamaño de la fuente de las figuras

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
umax = 2 + du                       # valor máximo del mallado
u = np.arange(umin, umax, du)   # mallado espacial
n = len(u)                      # número de puntos en el mallado

### FUNCIONES ###

def numerov_pozo(alfa):
    '''
    Función que calcula la solución a la ecuación de schrdodinger para el método de numerov
    para una caja de potencial.

    Parámetros:
        alfa (float): valor de energía (E = alfa*V_0), con V_0 el potencial de la caja.

    Devuelve:
        phi (array): función de onda calculada para el dominio de u

    '''
    # arrays para almacenar las soluciones
    phi = np.zeros_like(u)
    psi = np.zeros_like(u)

    # array para almacenar el valor de la functión f en todo el mallado
    cte = np.zeros_like(u)
    for j in range(n):
        if abs(u[j]) > 0.5:
            cte[j] = k*(1 - alfa)
        if abs(u[j]) <= 0.5:
            cte[j] = - k*alfa

    phi[0] = 0
    phi[1] = 1e-4

    # comienzo de iteración recursiva
    for i in range(n-2):
        phi[i+2] = 2*phi[i+1] - phi[i] + du**2*cte[i+1]*phi[i+1]/(1 - du**2*cte[i+1]/12)

    psi = phi/(1-du**2*cte/12)
    return psi


def shooting_pozo(alfa_1, delta_alfa=0.01, num_sols=3, tol=1e-10, max_iter=1000):
    '''
    Implementación del shooting method para soluciones a la ecuación de schrodinger en el
    pozo de potencial.

    Parámetros:
        alfa_1 (float): valor de energía inicial (E = alfa*V_0)
        num_sols (int): número de soluciones a calcular. No introducir más de 3
        tol (float): tolerancia del método
        delta_alfa (float): separación entre valores de alfa para los que calcular la función de onda
        max_iter (int): número máximo de iteraciones

    Devuelve:
        alfas (array): contiene los valores de
        psis (array): contiene las soluciones calculadas

    '''
    # arrays para almacenar las soluciones calculadas
    alfas = np.zeros(num_sols)
    psis = np.zeros((num_sols, n))

    # aplicar el shooting method num_sols veces
    for i in range(num_sols):
        alfa_2 = alfa_1 + delta_alfa
        psi_1 = numerov_pozo(alfa_1)
        psi_2 = numerov_pozo(alfa_2)

        # buscar dos valores de alfa tales que se pueda aplicar bisección
        while psi_1[-1]*psi_2[-1] > 0:
            alfa_2 += delta_alfa
            psi_2 = numerov_pozo(alfa_2)

        alfa_n = 0.5*(alfa_1+alfa_2)
        psi_n = numerov_pozo(alfa_n)

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
            psi_n = numerov_pozo(alfa_n)

        alfas[i] = alfa_n                       # almacenar energía en el array de soluciones
        psi_n = psi_n/np.sqrt(sum(psi_n**2*du))  # normalizar función de onda
        psis[i] = psi_n                         # almacenar función en el array de soluciones
        alfa_1 = alfa_n + delta_alfa

    return alfas, psis

def representacion(alfas, psis, save=False):
    '''
    Función que representa las funciones de onda en la caja de potencial.

    Parámetros:
        alfas (array): contiene los valores de energía de las funciones de onda
        psis (array): funciones de onda a representar
        save (bool): True para guardar las figuras en el directorio indicado por directorio_figuras
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(alfas)):
        ax.plot(u, psis[i], label=r'$\alpha = {:.3f}$'.format(alfas[i]))
    ax.set_xlabel('u = r/a')
    ax.axvline(-0.5, ls='--', lw=1, c='grey')
    ax.axvline(0.5, ls='--', lw=1, c='grey')
    ax.set_ylabel(r'$\Psi(r/a)$')
    ax.legend()
    if save:
        fig.savefig(directorio_figuras + 'pozo_numerov.pdf')
    else:
        ax.set_title('Autofunciones de la caja de potencial finito 1D')
    fig.tight_layout()
    plt.show()

###

alfa_0 = 0.5*np.pi**2/k         # valor inicial de alfa para comenzar el shooting method

t0 = time()
alfas, psis = shooting_pozo(alfa_0)
print('Tiempo transcurrido: {:2f}'.format(time()-t0))
representacion(alfas, psis, False)
plt.show()
