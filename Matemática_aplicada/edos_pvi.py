#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

### EULER Y RESOLUCION SIMBÓLICA ###

def euler(f, inter, wk, N):
    h = (inter[1] - inter[0]) / N
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

def f1(t, y): return -y
def f2(t, F): return np.array([F[1], -F[0]]).T

# resolución simbólica z' = y, y' = -z
t = sp.symbols('t', real=True)
y = sp.Function('y')
z = sp.Function('z')
edo2 = (sp.Eq(z(t).diff(t), y(t)), sp.Eq(y(t).diff(t), -z(t)))
ci2 = {y(0): 1, z(0): 2}
F2_sp = sp.dsolve(edo2, ics=ci2)
y2_sp = F2_sp[0].rhs
z2_sp = F2_sp[1].rhs
y2_np = sp.lambdify(t, y2_sp, 'numpy')
z2_np = sp.lambdify(t, z2_sp, 'numpy')
t2_representacion = np.linspace(0, 2*np.pi, 300)

# resolución numérica z' = y, y' = -z
t2_11pts, F2_11pts = euler(f2, (0, 2 * np.pi), np.array([1, 2]).T, 11)
z2_11pts = [i[0] for i in F2_11pts]
y2_11pts = [i[1] for i in F2_11pts]
t2_21pts, F2_21pts = euler(f2, (0, 2 * np.pi), np.array([1, 2]).T, 21)
z2_21pts = [i[0] for i in F2_21pts]
y2_21pts = [i[1] for i in F2_21pts]

# resolución y' = -y simbólica
ci1 = {y(0): 1}
edo1 = sp.Eq(y(t).diff(t), -y(t))
y1_sp = sp.dsolve(edo1, y(t), ics=ci1).rhs
y1_analitica = sp.lambdify(t, y1_sp, 'numpy')
t1_representacion = np.linspace(0, 5, 300)

# resolución y' = -y numérica
t1_11pts, y1_11pts = euler(f1, (0, 5), 1, 11)
t1_21pts, y1_21pts = euler(f1, (0, 5), 1, 21)

# representacion soluciones y' = -y
plt.figure()
plt.scatter(t1_11pts, y1_11pts, label='euler 11 pts')
plt.scatter(t1_21pts, y1_21pts, label='euler 21 pts')
plt.plot(t1_representacion, y1_analitica(
    t1_representacion), label='analítica')
plt.title(r'$y^\prime = -y$')
plt.legend()
plt.tight_layout()
plt.show()

# representacion soluciones z' = y, y' = -z
plt.figure()
plt.scatter(t2_11pts, z2_11pts, marker='o', c='b', label='z euler 11 pts')
plt.scatter(t2_11pts, y2_11pts, marker='o',
            c='darkorange', label='y euler 11 pts')
plt.scatter(t2_21pts, z2_21pts, marker='x', c='b', label='z euler 21 pts')
plt.scatter(t2_21pts, y2_21pts, marker='x',
            c='darkorange', label='y euler 21 pts')
plt.plot(t2_representacion, y2_np(t2_representacion), label='y analítica')
plt.plot(t2_representacion, z2_np(t2_representacion), label='z analítica')
plt.title(r'$z^\prime = y$ & $y^\prime = -z$')
plt.legend()
plt.tight_layout()
plt.show()


### RUNGE-KUTTA ###

def eulermod(f, inter, w0, N):
    h = (inter[1] - inter[0]) / N
    wi = w0
    ti = 0
    t = []
    w = []
    for k in range(N+1):
        t.append(ti)
        w.append(wi)
        k1 = f(ti, wi)
        k2 = f(ti + h, wi + h*k1)
        wi1 = wi + 0.5*h*(k1 + k2)
        ti1 = ti + h
        ti = ti1
        wi = wi1
    return t, w

# euler modificado para resolver y' = -y con 50 puntos
t1_rq, y1_rq = eulermod(f1, (0, 5), 1, 50)

fig3 = plt.figure()
plt.scatter(t1_rq, y1_rq, s=18, label='Euler modificado 50 pts')
plt.plot(t1_representacion, y1_analitica(
    t1_representacion), ls='--', label='Analítica')
plt.title(r'$y^\prime = -y$')
plt.legend()
plt.tight_layout()
plt.show()


### MÉTODOS DE SCIPY.INTEGRATE ####

# resolución de la edo1 mediante métodos de solve_ivp
sol1RK45 = solve_ivp(f1, (0, 5), [1], method='RK45')
sol1RK23 = solve_ivp(f1, (0, 5), [1], method='RK23')
sol1Radau = solve_ivp(f1, (0, 5), [1], method='Radau')
sol1BDF = solve_ivp(f1, (0, 5), [1], method='BDF')
sol1LSODA = solve_ivp(f1, (0, 5), [1], method='LSODA')

# representación
fig4 = plt.figure()
plt.scatter(sol1RK45.t, sol1RK45.y, label='RK45')
plt.scatter(sol1RK23.t, sol1RK23.y, label='RK23')
plt.scatter(sol1Radau.t, sol1Radau.y, label='Radau')
plt.scatter(sol1BDF.t, sol1BDF.y, label='BDF')
plt.scatter(sol1LSODA.t, sol1LSODA.y, label='LSODA')
plt.legend()
plt.title(r'$y^\prime = -y$. Métodos de scipy.integrate.solve_ivp')
plt.tight_layout()
plt.show()

# sistema ejemplo PDF PL
# t = sp.symbols('t', real=True)
# y = sp.Function('y')
# edo = sp.Eq(y(t).diff(t, t)-y(t).diff(t)+2*y(t), sp.exp(2*t)*sp.sin(t))
# ci = {y(0): -0.4, y(t).diff(t).subs(t, 0): -0.6}
# y_sp = sp.dsolve(edo, ics=ci).rhs
# y_np = sp.lambdify(t, y_sp, 'numpy')
# t = np.linspace(0, 5, 300)


# def f(t, F):
#     f1 = F[1]
#     f2 = 2*F[1]-2*F[0] + np.exp(2*t)*np.sin(t)
#     return np.array([f1, f2])

# t_euler, Z = euler(f, (0, 5), np.array([-0.4, -0.6]), 300)

# y_euler = [i[0] for i in Z]

# plt.figure()
# plt.plot(t_euler, y_euler)
# plt.plot(t, y_np(t))
# plt.show()
