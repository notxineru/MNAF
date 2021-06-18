#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, BarycentricInterpolator, KroghInterpolator, interp1d

# MÉTODO DE LAGRANGE #
# Interpolando x^2 en [1,2,3,4] con las funciones seno, coseno, 1, log(1+x)

def f1(x):
    return np.power(x, 2)


def base1(x):
    return np.array([np.sin(x), np.cos(x), np.ones(np.size(x)), np.log(1+x)])


x = np.array([1, 2, 3, 4])


y = f1(x)
A = base1(x).T
C = np.linalg.solve(A, y)

# Representación
X = np.linspace(0, 5)
Y = base1(X)
Y = np.dot(C, base1(X))


# USO DE FUNCIONES DE NUMPY
# e^-xsin(x) en el soporte s=[0,1,2,3,4]

s = np.array([0, 1, 2, 3, 4])


def f2(x):
    return np.exp(-x)*np.sin(x)


y_func = f2(s)

y_polifit = np.polyfit(s, y_func, deg=4)
y_lagrange = lagrange(s, y_func)
y_baricentric = BarycentricInterpolator(s, y_func)


X = np.linspace(0, 4)
Y_polifit = np.polyval(y_polifit, X)
Y_bar = y_baricentric(X)
Y_lag = y_lagrange(X)
Y_func = f2(X)

fig = plt.figure()
plt.scatter(s, y_func, label='datos', c='red')
plt.plot(X, Y_polifit, label="Polyfit", ls='-.')
plt.plot(X, Y_bar, label="Barycentric", ls='--')
plt.plot(X, Y_lag, label="Lagrange", ls=':')
plt.plot(X, Y_func, label="Funcion original")
plt.legend(loc="best")
plt.xlabel('f(x)')
plt.ylabel("f(x)")
plt.grid()
plt.tight_layout()
plt.show()

# inerp1d tipos: (linear, quadratic, cubic, slinear...)
y_interp1d = interp1d(s, y_func, 'cubic')
Y_interp1d = y_interp1d(X)
plt.figure()
plt.plot(X, Y_func)
plt.plot(X, Y_interp1d)
plt.tight_layout()
plt.show()
### POLINOMIOS INTERPOLADORES DE LAGRANGE ###

# def lagrange1(s):
#     L = np.zeros(len(s)+1)
#     print(L)
#     for i in range(len(L)):
#         for k in np.where(range(len(s)) != i):
#             L[i] *= (x-s[k])/(s[i]-s[k])
#     return L

s = np.random.rand(3)
s = np.insert(s, [0, 3], [0, 1])

lagrange = []
for i in range(len(s)):
    y = np.zeros_like(s)
    z = y.copy()
    z[i] = 1
    p = np.polyfit(s, z, len(s))
    lagrange.append(p)

print(lagrange)
x = np.linspace(0, 1, 200)

l0 = np.polyval(lagrange[0], x)
l1 = np.polyval(lagrange[1], x)
l2 = np.polyval(lagrange[2], x)
l3 = np.polyval(lagrange[3], x)
l4 = np.polyval(lagrange[4], x)


fig2 = plt.figure()
plt.scatter(s, np.zeros_like(s), c='red', label='soporte')
plt.plot(x, l0, label=r'$L_4^0(x)$')
plt.plot(x, l1, label=r'$L_4^1(x)$')
plt.plot(x, l2, label=r'$L_4^2(x)$')
plt.plot(x, l3, label=r'$L_4^3(x)$')
plt.plot(x, l4, label=r'$L_4^4(x)$')
plt.plot(x, l0+l1+l2+l3+l4, label=r'$\sum_{i = 0}^{4}$', ls='--')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


### DIFERENCIAS DIVIDIDAS ###

x = np.linspace(-5, 5, 7)
def f3(x): return x**4-2*x+1

y = f3(x)

def ddcoef(x, y):
    n = len(x)
    T = np.zeros([n, n])
    T[:, 0] = y
    for i in range(1, n):
        for j in range(i, n):
            T[j, i] = (T[j, i-1]-T[j-1, i-1])/(x[j]-x[j-i])
    c = np.diag(T)[:: -1]
    return c, T


c, T = ddcoef(x, y)
z = np.array([-4, 3, -2, 1])


def ddeval(x, c, z):
    n = len(c)
    s = np.zeros_like(z)
    s[s == 0] = c[0]
    for i in range(1, n):
        s = s*(z-x[n-i-1])+c[i]
    return s


s = ddeval(x, c, z)
with np.printoptions(suppress=True, linewidth=200):
    print(T)
    print(s)
print(f3(z))


### POLINOMIOS DE HERMITE ###


# x = np.linspace(0, 5, 5)
# y = np.zeros_like(x)

# p1 = KroghInterpolator(x, y)
# Y1 = p1(x)
