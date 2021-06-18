#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import integrate

def derivacion(fun, x, h, fórmula):
    n = fórmula[0]
    a = fórmula[1]
    fun_prima = 0
    if n == 2:
        if a == 1:
            h = -h
        c = 1 / h
        alfas = [-1, 1]
        xi = [x, x+h]
    elif n == 3 and (a == 0 or a == 2):
        if a == 2:
            h = -h
        c = 1/2/h
        alfas = [-3, 4, -1]
        xi = [x, x + h, x + 2*h]
    elif n == 3 and a == 1:
        c = 1/2/h
        alfas = [-1, 0, 1]
        xi = [x - h, x, x + h]
    else:
        print('Fórmula no conocida')
    for i in range(n):
        fun_prima += c * alfas[i] * fun(xi[i])
    return fun_prima


def f(x):
    return (1 + x)**0.5

# método analítico para f1 con sympy
x = sp.symbols('x', real=True)
fun = (x+1)**0.5
prima_sp = fun.diff(x, 1)
prima_np = sp.lambdify(x, prima_sp, 'numpy')

# métodos numéricos para f1
x = np.linspace(0, 1, 11)
prima_2p_0 = derivacion(f, x, 0.1, [2, 0])
prima_2p_1 = derivacion(f, x, 0.1, [2, 1])
prima_3p_0 = derivacion(f, x, 0.1, [3, 0])
prima_3p_1 = derivacion(f, x, 0.1, [3, 1])
prima_3p_2 = derivacion(f, x, 0.1, [3, 2])


prima_analitica = prima_np(x)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x, prima_2p_0, label='2pts. adel')
ax1.plot(x, prima_2p_1, label='2pts. retr')
ax1.plot(x, prima_3p_0, label='3pts. adel')
ax1.plot(x, prima_3p_1, label='3pts. cent')
ax1.plot(x, prima_3p_2, label='3pts. retr')
ax1.plot(x, prima_analitica, label='analítica')
ax1.grid()
ax1.legend(loc='best', ncol=2)
ax1.set_title(r'Derivada de $ f(x) =\sqrt{1+x}$')
ax2.plot(x, prima_2p_0-prima_analitica, label='2pts. adel')
ax2.plot(x, prima_2p_1-prima_analitica, label='2pts. retr')
ax2.plot(x, prima_3p_0-prima_analitica, label='3pts. adel')
ax2.plot(x, prima_3p_1-prima_analitica, label='3pts. cent')
ax2.plot(x, prima_3p_2-prima_analitica, label='3pts. retr')
ax2.grid()
ax2.legend(loc='best', ncol=2)
ax2.set_title('Error en los métodos numéricos')
plt.tight_layout()
plt.show()


### INTEGRACIÓN ###
x = sp.symbols('x', real=True)


# resolver 4 funciones con las intruccions de python para integracion

def f1(x): return np.sin(x)

def f2(x): return 1 / (1 + x**2)

def f3(x): return np.exp(-x**2)

def f4(x): return (1 + x**2)**(1 / 2)


fq1, err_fq1 = integrate.fixed_quad(f1, 0, np.pi)
quad1, err_quad1 = integrate.quadrature(f1, 0, np.pi)
s1 = sp.integrate(sp.sin(x), (x, 0, sp.pi))
print('Función 1 -> Exacta: %.1f, quad: %.16f, fixed_quad: %.16f' %
      (s1, quad1, fq1))

fq2, err_fq2 = integrate.fixed_quad(f2, 0, 5)
quad2, err_quad2 = integrate.quadrature(f2, 0, 5)
s2 = sp.integrate(1 / (1 + x**2), (x, 0, 5))
print('Función 2 -> Exacta: %.16f, quad: %.16f, fixed_quad: %.16f' %
      (s2, quad2, fq2))


fq3, err_fq3 = integrate.fixed_quad(f3, 0, 4)
quad3, err_quad3 = integrate.quadrature(f3, 0, 4)
s3 = sp.integrate(sp.exp(-x**2), (x, 0, 4))
print('Función 3 -> Exacta: %.16f, quad: %.16f, fixed_quad: %.16f' %
      (s3, quad3, fq3))

fq4, err_fq4 = integrate.fixed_quad(f4, -1, 1)
quad4, err_quad4 = integrate.quadrature(f4, -1, 1)
s4 = sp.integrate((1 + x**2)**(1 / 2), (x, -1, 1))
print('Función 4 -> Exacta: %.16f, quad: %.16f, fixed_quad: %.16f' %
      (s4, quad4, fq4))


# simps, y otro y ver como convergen
l = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

simps1 = []
trapz1 = []
simps2 = []
trapz2 = []
simps3 = []
trapz3 = []
simps4 = []
trapz4 = []

for i in l:
    puntos = np.linspace(0, np.pi, i)
    simps1.append(integrate.simps(f1(puntos), puntos))
    trapz1.append(integrate.trapz(f1(puntos), puntos))

    puntos = np.linspace(0, 5, i)
    simps2.append(integrate.simps(f2(puntos), puntos))
    trapz2.append(integrate.trapz(f2(puntos), puntos))

    puntos = np.linspace(0, 4, i)
    simps3.append(integrate.simps(f3(puntos), puntos))
    trapz3.append(integrate.trapz(f3(puntos), puntos))

    puntos = np.linspace(-1, 1, i)
    simps4.append(integrate.simps(f4(puntos), puntos))
    trapz4.append(integrate.trapz(f4(puntos), puntos))

# representación
fig, ax = plt.subplots(2, 2, figsize=(7, 6))
ax[0, 0].scatter(l, simps1, label='Simpson', s=12)
ax[0, 0].scatter(l, trapz1, label='Trapecio', s=12)
ax[0, 0].plot((l[0], l[-1]), (s1, s1), ls='--', lw=1, label='Exacta')
ax[0, 0].set_xlabel('nº de puntos')
ax[0, 0].set_title(r'$\sin(x)$')
ax[0, 0].legend()
ax[0, 1].scatter(l, simps2, label='Simpson', s=12)
ax[0, 1].scatter(l, trapz2, label='Trapecio', s=12)
ax[0, 1].plot((l[0], l[-1]), (s2, s2), ls='--', lw=1, label='Exacta')
ax[0, 1].set_xlabel('nº de puntos')
ax[0, 1].set_title(r'$\frac{1}{x^2+1}$')
ax[0, 1].legend()
ax[1, 0].scatter(l, simps3, label='Simpson', s=12)
ax[1, 0].scatter(l, trapz3, label='Trapecio', s=12)
ax[1, 0].plot((l[0], l[-1]), (s3, s3), ls='--', lw=1, label='Exacta')
ax[1, 0].set_xlabel('nº de puntos')
ax[1, 0].set_title(r'$e^-x^2"$')
ax[1, 0].legend()
ax[1, 1].scatter(l, simps4, label='Simpson', s=12)
ax[1, 1].scatter(l, trapz4, label='Trapecio', s=12)
ax[1, 1].plot((l[0], l[-1]), (s4, s4), ls='--', lw=1, label='Exacta')
ax[1, 1].set_xlabel('nº de puntos')
ax[1, 1].set_title(r'$\sqrt{x^2+1}$')
ax[1, 1].legend()
plt.tight_layout()
plt.show()

# COEFICIENTES INDETERMINADOS
def coefind(x, tipo: str, data):
    n = len(x)
    F = np.zeros_like(x)
    A = np.ones((n, n))
    for i in range(1, n):
        for j in range(n):
            A[i, j] = x[j]*A[i-1, j]
    if tipo[0:4] == 'inte':
        a, b = data
        for i in range(1, n+1):
            F[i-1] = 1/i*(b**i-a**i)
    elif tipo[0:4] == 'deri':
        orden, punto = data
        assert punto <= n, 'Punto donde calcular la derivada fuera del intervalo de puntos!'
        assert n > orden, 'El número de puntos ha de seyer mayor al orden de la derivada a calcular!'
        a = x[punto]
        for i in range(n):
            if i < orden:
                F[i] = 0
            else:
                F[i] = np.math.factorial(
                    i)/np.math.factorial(i-orden)*a**(i-orden)
    else:
        print('Introduzca tipo válido: integración o derivación')
        return None
    coef = np.linalg.solve(A, F)
    coef = np.where(abs(coef) < 10e-10, 0, coef)
    return coef

# COMPROBACIÓN DERIVACIÓN
x1 = np.linspace(1, 2, 2)
c1 = coefind(x1, 'derivación', (1, 0))
print('1ª derivada, 2pts, adelantada, h = 1. Coefs: ' + str(c1))

x2 = np.linspace(1, 4, 4)
c2 = coefind(x2, 'derivación', (2, 1))
print('2ª derivada, 4pts, a = 2º punto, h = 1. Coefs: ' + str(c2))

# h = 1 para comparar más facilmente con las fórmulas. Los resultados concuerdan con los
# valores de las tablas

x3 = np.linspace(1, 2, 2)
c3 = coefind(x3, 'integración', (1, 2))
print(c3)

x4 = np.linspace(1, 3, 3)
c4 = coefind(x4, 'integración', (1, 3))
print(c4)

# los coeficientes coinciden con los esperados
