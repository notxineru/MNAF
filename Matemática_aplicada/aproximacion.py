#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

np.seterr(divide='ignore')


def apxdimsc(base, X, Y):
    n1 = len(base)
    n2 = len(X)
    A = np.zeros((n2, n1))
    for i in range(n2):
        for j in range(n1):
            A[i, j] = base[j](X[i])
    G = np.dot(A.T, A)
    f = np.dot(A.T, Y)
    c = np.linalg.solve(G, f)
    return c

base = np.array([lambda x: 1, lambda x: np.sin(x), lambda x: np.cos(x)])
x = np.array([-np.pi/2, -np.pi/4, np.pi/4, np.pi/2])
y = np.array([0.2079, 0.4559, 2.1933, 4.8105])
c = apxdimsc(base, x, y)
print(c)

def mceval(base, c, z):
    n = len(base)
    x = 0
    for i in range(n):
        x += c[i]*base[i](z)
    return x


def aproxconmc(base, inter, f):
    F_array = []
    n = len(base)
    G = np.zeros((n, n))
    for i in range(n):
        def F(x): return f(x) * base[i](x)
        F = integrate.quad(F, *inter)[0]
        F_array.append(F)
        for j in range(i, n):
            def aux(x): return base[i](x) * base[j](x)
            G[i, j] = integrate.quad(aux, *inter)[0]
            G[j, i] = G[i, j]

    c = np.linalg.solve(G, F_array)
    return c


B1 = np.array([lambda x: 1, lambda x: 1 / x, lambda x: x**(1 / 2)])
B2 = np.array([lambda x: 1, lambda x: x, lambda x: x**2])
B3 = np.array([lambda x: 1, lambda x: np.sin(x), lambda x: np.cos(x)])
X = np.array([1, 2, 3, 4, 5])

def f1(x): return 1 / (1 + x**2)

x = np.linspace(1, 5, 300)
Y = f1(X)
y = f1(x)

c1d = apxdimsc(B1, X, Y)
y_aprox1d = mceval(B1, c1d, x)
c2d = apxdimsc(B2, X, Y)
y_aprox2d = mceval(B2, c2d, x)
c3d = apxdimsc(B3, X, Y)
y_aprox3d = mceval(B3, c3d, x)

# representación caso discreto
fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 5.5))
ax1.scatter(X, Y, label='datos', c='slategrey')
ax1.plot(x, y_aprox1d, label=r'$\left[1, \sqrt{1}, \frac{1}{x}\right]$')
ax1.plot(x, y_aprox2d, label=r'$\left[1, x, x^2\right]$')
ax1.plot(x, y_aprox3d, label=r'$\left[1, \sin{x}, \cos{x}\right]$')
ax1.plot(x, y, label=r'$f(x) = \frac{1}{1+x^2}$')
ax1.set_title('Función vs aproximaciones')
ax1.legend()
ax2.plot(x, y_aprox1d - y,
         label=r'$\left[1, \sqrt{1}, \frac{1}{x}\right]$')
ax2.plot(x, y_aprox2d - y, label=r'$\left[1, x, x^2\right]$')
ax2.plot(x, y_aprox3d - y, label=r'$\left[1, \sin{x}, \cos{x}\right]$')
ax2.set_title('Eror en las aproximaciones')
ax2.legend()
fig.suptitle('Caso discreto')
plt.tight_layout()
plt.show()

inter = (1, 5)
c1c = aproxconmc(B1, inter, f1)
y_aprox1c = mceval(B1, c1c, x)
c2c = aproxconmc(B2, inter, f1)
y_aprox2c = mceval(B2, c2c, x)
c3c = aproxconmc(B3, inter, f1)
y_aprox3c = mceval(B3, c3c, x)

# representación caso continuo
fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 5.5))
ax1.plot(x, y_aprox1c, label=r'$\left[1, \sqrt{1}, \frac{1}{x}\right]$')
ax1.plot(x, y_aprox2c, label=r'$\left[1, x, x^2\right]$')
ax1.plot(x, y_aprox3c, label=r'$\left[1, \sin{x}, \cos{x}\right]$')
ax1.plot(x, y, label=r'$f(x) = \frac{1}{1+x^2}$')
ax1.set_title('Función vs aproximaciones')
ax1.legend()
ax2.plot(x, y_aprox1c - y,
         label=r'$\left[1, \sqrt{1}, \frac{1}{x}\right]$')
ax2.plot(x, y_aprox2c - y, label=r'$\left[1, x, x^2\right]$')
ax2.plot(x, y_aprox3c - y, label=r'$\left[1, \sin{x}, \cos{x}\right]$')
ax2.set_title('Eror en las aproximaciones')
ax2.legend()
fig.suptitle('Caso continuo')
plt.tight_layout()
plt.show()


### LINEALIZACIÓN ###
x = np.array([0.75, 2., 2.5, 4., 6., 8., 8.5])
y = np.array([0.8, 1.3, 1.2, 1.6, 1.7, 1.8, 1.7])
z = np.linspace(0, 9, 300)

a = 2

#y = a*exp(bx)
c = np.polyfit(x, np.log(y), 1)  # ajuste lineal
def exp(x, a, b): return a * np.exp(b * x)
c1, cov = optimize.curve_fit(exp, x, y)  # ajuste curve_fit
f_l_exp = np.exp(np.polyval(c, z))
f_c_exp = exp(z, *c1)


# y = a*x^(b)
c = np.polyfit(np.log(x), np.log(y), 1)
def potencial(x, a, b): return a * x**b
c1, cov = optimize.curve_fit(potencial, x, y)
f_l_pot = np.exp(np.polyval(c, np.log(z)))
f_c_pot = potencial(z, *c1)


# crecimiento, y = \frac{x}{ax+b}
c = np.polyfit(1 / x, 1 / y, 1)
def crec(x, a, b): return x / (a * x + b)
c1, cov = optimize.curve_fit(crec, x, y)
f_l_crec = 1 / np.polyval(c, 1 / z)
f_c_crec = crec(z, *c1)


# logistica y = \frac{a}{1+e^{bx-c}}+d
def logis(x, a, b, c, d): return a / (1 + np.exp(b * x - c)) + d
c1, cov = optimize.curve_fit(logis, x, y)
f_c_logis = logis(z, *c1)


# arcotangente
def arctan(x, a, b, c, d): return a * np.arctan(b * x - c) + d
c1, cov = optimize.curve_fit(arctan, x, y)
f_c_artan = arctan(z, *c1)


fig = plt.figure()
plt.scatter(x, y, c='r')
plt.plot(z, f_l_crec, label='crecimiento')
plt.plot(z, f_l_exp, label='exponencial')
plt.plot(z, f_l_pot, label='potencial')
plt.legend(loc='best')
plt.title('Linealizadas (polyfit)')
plt.tight_layout()
plt.show()


fig1 = plt.figure()
plt.scatter(x, y, c='r')
plt.plot(z, f_c_crec, label='crecimiento')
plt.plot(z, f_c_exp, label='exponencial')
plt.plot(z, f_c_pot, label='potencial')
plt.legend(loc='best')
plt.title('Ajuste no lineal (curvefit)')
plt.tight_layout()
plt.show()


fig2 = plt.figure()
plt.scatter(x, y, c='r')
plt.plot(z, f_c_artan, label='arctan')
plt.plot(z, f_c_logis, label='logística')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
