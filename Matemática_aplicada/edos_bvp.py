#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import sympy as sp

# Programa que reuelve problema medianye le problema de la secante
# Meterle para resolver la edo como pvi (anterior práctica)
# verificar con y''(x) = f(x,y,y')

# método de euler creado en el módulo de edos_pvi
def euler(f, inter, k_0, N):
    h = (inter[1] - inter[0]) / N
    wk = k_0
    tk = 0
    t = []
    w = []
    for k in range(N + 1):
        w.append(wk)
        t.append(tk)
        wk1 = wk + h * f(tk, wk)
        tk1 = tk + h
        tk = tk1
        wk = wk1
    return t, w

# método de disparo
def disparo(F, ab, cc, iter=50, tol=1E-6, ni=[0, 1], N=100):
    g0 = ni[0]
    g1 = ni[1]

    x, F0 = euler(F, ab, (cc[0], g0), N)
    y0 = [i[0] for i in F0]

    x, F1 = euler(F, ab, (cc[0], g1), N)
    y1 = [i[0] for i in F1]

    wn0 = y0[-1]
    wn1 = y1[-1]

    for i in range(iter):
        gamma = g1 + (cc[1] - wn1)*(g1 - g0)/(wn1 - wn0)
        x, F3 = euler(F, ab, (cc[0], gamma), N)
        y = [i[0] for i in F3]
        wn0 = wn1
        wn1 = y[-1]
        if abs(cc[1]-wn1) < tol:
            break
        g0 = g1
        g1 = gamma
    return x, y, gamma

# resolución de y -2y'+y'' = 0 por disparo
def F(t, f): return np.array([f[1], f[1] + 2*f[0]]).T
cc = (0, np.exp(2))
x_disparo, y_disparo, gamma = disparo(F, (0, 1), cc)

# resolución de  y -2y'+y'' = 0 simbólica con sympy
cc = (0, sp.exp(2))
x = sp.symbols('x', real=True)
y = sp.Function('y')
edo = sp.Eq(y(x).diff(x, x) - 2*y(x).diff(x) + y(x), 0)
y_sp = sp.dsolve(edo, y(x), ics={y(0): cc[0], y(1): cc[1]}).rhs
y_analitica = sp.lambdify(x, y_sp, 'numpy')
x_representacion = np.linspace(0, 1)

# resolución numérica con solve_bvp
def bc(ya, yb): return np.array([ya[0]-cc[0], yb[0] - np.exp(2)])

def F(t, f): return np.vstack((f[1], f[1]+2*f[0]))
x_bvp = np.linspace(0, 1, 5)
y_inicial_bvp = np.zeros((2, len(x_bvp)))
sol = solve_bvp(F, bc, x_bvp, y_inicial_bvp)

# representaciones
fig = plt.figure()
plt.plot(x_disparo, y_disparo, label='disparo')
plt.plot(x_representacion, y_analitica(x_representacion), label='analítica')
plt.scatter(sol.x, sol.y[0], c='red', label='solve_bvp')
plt.title(r'$y(x)-2y^\prime(x)+y^{\prime\prime}(x)$')
plt.legend()
plt.show()


### DIFERENCIAS FINITAS ###

def bvpfdm(F, ab, cc, n=3):
    a, b = ab[0], ab[1]
    y_a, y_b = cc[0], cc[1]
    h = (b - a)/(n - 1)
    x = np.linspace(a, b, n)
    f = F(x)
    p, q, f = f[0], f[1], f[2]
    A = np.zeros((n, n))
    B = np.zeros(n)
    for i in range(1, n-1):
        B[i] = f[i]
        A[i, i-1] = 1/h**2 - p[i]/(2*h)
        A[i, i] = q[i] - 2/h**2
        A[i, i+1] = 1/h**2 + p[i]/(2*h)
    B[0] = y_a
    B[-1] = y_b
    A[0, 0] = 1
    A[-1, -1] = 1
    y = np.linalg.solve(A, B)
    return x, y

