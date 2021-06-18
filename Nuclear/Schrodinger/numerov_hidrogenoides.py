#!/usr/bin/env python3

from time import time
import numpy as np
import matplotlib.pyplot as plt
# from numba import jit


plt.rcParams.update({'font.size': 12.5})  # cambiar el tamaño de la fuente de las figuras

### CONTROL ###
 
directorio_figuras = ''  # ruta relativa del directorio donde guardar las figuras

LITHIUM = False     # if True, se calculan las energías y autofunciones para el átomo de litio

L = 0               # número del momento angular
Z = 2               # número de protones en el nucleo. if LITHIUM = True, Z no tiene efecto
orbitales = {'0': 's', '1': 'p', '2': 'd', '3': 'f'}  # relacionar L con los orbitales
atomos = {'1': 'H', '2': 'He', '3': 'Li', '4': 'Be'}  # relacionar Z con los átomos

###

du = 0.01                       # resolución espacial del mallado
umin = 1e-6                     # mínimo valor del mallado. != 0 para que que el potencial no diverga
umax = 70                       # máximo valor del mallado. A más alto mayor precisión.
u = np.arange(umin, umax, du)   # mallado espacial
n = len(u)                      # número de puntos en el mallado

### FUNCIONES ###

# @jit(nopython=True)             # para compilar la función y lograr ejecuciones mucho más rapidas
def numerov(alfa):
    '''
    Función que calcula la solución a la ecuación de schrodinger por el método de numerov para
    el potencial de un átomo hidrogenoide o para el átomo de litio.

    Parámetros:
        alfa (float): valor de energía (E = alfa*E_r), con E_r = -13.6 eV la energía de rydberg

    Devuelve:
        phi (array): función de onda calculada para el dominio de u

    '''
    # arrays para almacenar los valores
    phi = np.zeros_like(u)
    psi = np.zeros_like(u)

    # calculo del valor de la función f sobre todo el mallado en función del parámetro LITHIUM
    f = np.zeros_like(u)
    if LITHIUM:
        for i in range(n):
            if u[i] <= 1:
                f[i] = L*(L + 1)/u[i]**2. - 2.*3/u[i] + 4 + alfa
            else:
                f[i] = L*(L + 1)/u[i]**2. - 2./u[i] + alfa
    else:
        for i in range(n):
            f[i] = L*(L + 1)/u[i]**2. - 2.*Z/u[i] + alfa

    phi[-1] = 0               # valor de la función en el último punto del mallado
    phi[-2] = 1e-4          # valor de la función en el penúltimo punto del mallado

    # numerov para phi por iteracion recursiva
    for i in range(n-1, 1, -1):
        phi[i-2] = 2*phi[i-1] - phi[i] + du**2*f[i-1]*phi[i-1]/(1 - du**2*f[i-1]/12)

    psi = phi/(1 - du**2*f/12)  # cáluclo de psi a partir de phi
    return psi


def shooting(alfa_0, num_sols=5, delta_alfa=0.01, tol=1e-10, max_iter=1000):
    '''
    Implementación del shooting method para soluciones a la parte radial de la función de onda

    Parámetros:
        alfa_0 (float): valor de energía (E = alfa*E_r) sobre el que comenzar a iterar
        delta_alfa (float): separación entre valores de alfa sobre los que iterar
        num_sols (int): número de soluciones a calcular
        tol (float): tolerancia del método
        max_iter (int): número máximo de iteraciones al aplicar bisección

    Devuelve:
        alfas (array): array con num_sols energías del sistema
        psis (array): array con num_sols soluciones a la parte radial de la ecuación de onda

    '''
    # array de valores de alfas sobre los que iterar para encontrar soluciones
    array_alfas = np.arange(alfa_0, 0, -delta_alfa)

    # arrays para almacenar las soluciones calculadas
    alfas = np.zeros(num_sols)
    psis = np.zeros((num_sols, n))

    index = 0

    # aplicar el shooting method num_sols veces
    for k in range(num_sols):
        alfa_1 = array_alfas[index]
        psi_1 = numerov(alfa_1)

        # buscar dos valores de alfa tales que se pueda aplicar bisección
        for i in range(index + 1, len(array_alfas)):
            alfa_2 = array_alfas[i]
            psi_2 = numerov(alfa_2)
            if np.sign(psi_1[0]) != np.sign(psi_2[0]):
                index = i
                break

        alfa_n = 0.5*(alfa_1 + alfa_2)
        psi_n = numerov(alfa_n)

        # comienzo bisección
        for i in range(max_iter):
            # chequear si se a alcanzado la tolerancia para finalizar bisección
            if abs(psi_n[0]) <= tol:
                break

            elif np.sign(psi_n[0]) == np.sign(psi_1[0]):
                alfa_1 = alfa_n
                psi_1 = psi_n

            elif np.sign(psi_n[0]) == np.sign(psi_2[0]):
                alfa_2 = alfa_n
                psi_2 = psi_n

            alfa_n = 0.5*(alfa_1 + alfa_2)
            psi_n = numerov(alfa_n)

        alfas[k] = alfa_n                             # almacenar energía en el array de soluciones
        psi_n = psi_n/np.sqrt(np.sum((psi_n**2)*du))  # normalizar función de onda
        psis[k] = psi_n                               # almacenar función en el array de soluciones

    return alfas, psis


def representacion(psis, alfas, save=False):
    '''
    Función que representa la parte radial de las funciones de onda.

    Parámetros:
        alfas (array): contiene los valores de energía de las funciones de onda
        psis (array): funciones de onda a representar
        save (bool): True para guardar las figuras en el directorio indicado por directorio_figuras

    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(alfas)):
        ax.plot(u, psis[i], label=r'n = {:d}, $\alpha = {:.3f}$'.format(i+1+L, alfas[i]))
    ax.set_xlabel(r'$r/a_0$')
    ax.set_ylabel(r'$r/a_0 \cdot R(r/a_0)$')
    ax.legend()
    if save:
        ax.set_title('Orbitales {:s}'.format(orbitales[str(L)]))
        if LITHIUM:
            fig.savefig(directorio_figuras + 'Li_{:s}.png'.format((orbitales[str(L)])))

        elif Z == 1:
            fig.savefig(directorio_figuras + 'H_{:s}.png'.format((orbitales[str(L)])))

        else:
            fig.savefig(directorio_figuras + '{:s}_hidrogenoide_{:s}.png'.format(atomos[str(Z)],
                                                                                 orbitales[str(L)]))
    else:
        if LITHIUM:
            titulo = 'Parte radial de la función de onda del átomo de Li. Orbitales ' \
                '{:s}'.format(orbitales[str(L)])
        elif Z == 1:
            titulo = 'Parte radial de la función de onda del átomo de H. Orbitales ' \
                '{:s}'.format(orbitales[str(L)])
        else:
            titulo = 'Parte radial de la función de onda del {:s} ' \
                '(hidrogenoide). Orbitales {:s}'.format(atomos[str(Z)], orbitales[str(L)])

        ax.set_title(titulo)
    fig.tight_layout()

###

# valor inicial de alfa para el shooting method en función del tipo de átomo
if LITHIUM:
    alfa_0 = 1

else:
    alfa_0 = Z**2 + 0.1

t0 = time()
alfas, psis = shooting(alfa_0, num_sols=5)
print('Tiempo de ejecución: {:.2f} segundos'.format(time() - t0))
representacion(psis, alfas)
plt.show()
