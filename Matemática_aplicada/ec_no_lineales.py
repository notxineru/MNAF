import sympy as sym
from sympy.abc import x
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, brentq, brenth, ridder

### MÉTODOS SÍMBOLICOS CON SYMPY ###

# función ejemplo
f = x**3-2*x**2+3*x-4
#sym.plot(f, (x, -2.5, 3))
s = sym.solve(f, x)
sn_ej = np.array(s).astype(complex)

# función 1
f = x*sym.exp(x)-4
#sym.plot(f, (x, 1, 2))
s = sym.solve(f, x)
sn_1 = np.array(s).astype(float)


# función 2
f = sym.sin(x) + 0.8*sym.cos(x)
#sym.plot(f, (x, 5, 6))
s = sym.solve(f, x)  # solve encuentra una solución fuera del intervalo [5,6]
sn_2 = np.array(s).astype(float)

print("--> f(x) = x**3-2*x**2+3*x - 4 (solve)\n", sn_ej)
print("--> f(x) = x*exp(x) - 4 (solve)\n %0.4f" % (sn_1))
print("--> f(x) = sin(x) + 0.8cos(x) (solve)\n %0.4f\n" % (sn_2))


### MÉTODOS DE INTERVALO ###

def bisec(f, a, b, maxiter=1000, tolx=10**(-6), tolf=10**(-9)):
    '''
    Obtiene una solución para f=0 entre a y b traves del método de bisección

        Devuelve:
            xn (float): Solución para f=0
            iteraciones (int): Número de iteraciones realizadas
            suc_xn (lista): Sucesión de valores de xn usados para hallar la solución
    '''
    xn = 1/2*(a+b)
    fa = f(a)
    fb = f(b)
    suc_xn = []  # lista para almacenar la sucesión xn

    iteraciones = 0

    if fa*fb > 0:
        raise ValueError('No cumple bolzano, cálculo abortado')

    for i in range(maxiter):
        iteraciones += 1

        suc_xn.append(xn)
        fn = f(xn)

        if abs(fn) < tolf:
            return xn, np.array(iteraciones), suc_xn

        if fn < 0:
            a = xn
            fa = fn

        elif fn > 0:
            b = xn
            fb = fn

        elif fn == 0:
            return xn, np.array(iteraciones), suc_xn

        elif abs(b-a) < tolx:
            return xn, np.array(iteraciones), suc_xn

        xn = 1/2*(a+b)

    print('Finalizado tras terminar todas las iteraciones')
    return xn, iteraciones, suc_xn


def falsi(f, a, b, maxiter=1000, tolx=10**(-6), tolf=10**(-9)):
    '''
    Obtiene una solución para f=0 entre a y b traves del método de Régula Flasi

        Devuelve:
            xn (float): Solución para f=0
            iteraciones (int): Número de iteraciones realizadas
            suc_xn (lista): Sucesión de valores de xn usados para hallar la solución
    '''

    fa = f(a)
    fb = f(b)
    xn = (fb*a - fa*b)/(fb-fa)
    iteraciones = 0
    suc_xn = []  # lista para almacenar la sucesión xn

    if fa*fb > 0:
        raise ValueError('No cumple bolzano')

    for i in range(maxiter):
        iteraciones += 1

        suc_xn.append(xn)
        fn = f(xn)

        if abs(fn) < tolf:
            return xn, iteraciones, suc_xn

        if fn < 0:
            a = xn
            fa = fn

        elif fn > 0:
            b = xn
            fb = fn

        elif fn == 0:
            return xn, iteraciones, suc_xn

        elif abs(b-a) < tolx:
            return xn, iteraciones, suc_xn

        xn = (fb*a - fa*b)/(fb-fa)

    print('Finalizado tras terminar todas las iteraciones')
    return xn, iteraciones, suc_xn


# función 1
def f1(x): return x*np.exp(x) - 4

x = np.linspace(0, 2, 100)

fig = plt.figure()
plt.grid()
plt.plot(x, f1(x))
plt.xlabel('x')
plt.title(r'$f_1(x) = xe^{x}-4$')
plt.tight_layout()
plt.show()

sol_bisec, itera_bisec, suc_bisec = bisec(f1, 0, 2)

sol_falsi, itera_falsi, suc_falsi = falsi(f1, 0, 2)
print("--> f(x)=x*exp(x) - 4 (métodos intervalo)")

print("Método de bisección: x0 = %0.8f, %2d iteraciones " %
      (sol_bisec, itera_bisec))
print("Método de Regula Falsi: x0 = %0.8f, %2d iteraciones " %
      (sol_falsi, itera_falsi))
print("Bisectf: x0 = %0.8f, Brentq: x0 = %0.8f\n" %  # métodos de scipy.optimize
      (bisect(f1, 0, 2), brentq(f1, 0, 2)))


# función 2
def f2(x): return np.sin(x) + 0.8*np.cos(x)

x = np.linspace(5, 6, 100)

fig = plt.figure()
plt.grid()
plt.plot(x, f2(x))
plt.xlabel('x')
plt.title(r'$f_2(x) = \sin{x}+0.8\cos{x}$')
plt.tight_layout()
plt.show()

sol_bisec, itera_bisec, suc_bisec = bisec(f2, 5, 6)

sol_falsi, itera_falsi, suc_falsi = falsi(f2, 5, 6)
print("--> f(x)=sin(x)+0.8cos(x) (métodos intervalo)")
print("Método de bisección: x0 = %0.8f, %2d iteraciones " %
      (sol_bisec, itera_bisec))
print("Método de Regula Falsi: x0 = %0.8f, %2d iteraciones " %
      (sol_falsi, itera_falsi))
print("Bisectf: x0 = %0.8f, Brentq: x0 = %0.8f\n" %  # métodos de scipy.optimize
      (bisect(f2, 5, 6), brentq(f2, 5, 6)))


### MÉTODOS DE PUNTO FIJO ###
def representación_ptofijo(g_sym, a, b):
    '''
    Representa g(x) y su derivada de manera que visualmente se pueda observar si g(x)
    cumple los criterios de convergencia.

        Devuelve:
                g_np (función): g(x) "lambdificada" para trabajar con ella en numérico

    '''
    dg_sym = g_sym.diff((x, 1))
    g_np = sym.lambdify(x, g_sym, "numpy")
    dg_np = sym.lambdify(x, dg_sym, "numpy")

    x_array = np.linspace(a, b, 150)

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.gca().set_title("g(x): mapeo")
    plt.plot(x_array, g_np(x_array), label='g(x)')
    plt.plot([a, b], [a, b], ls='--', label='x')
    plt.xlim(a, b)
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x_array, dg_np(x_array), label=r'$g^\prime(x)$')
    plt.ylim(-2, 2)
    plt.xlim(a, b)
    plt.axhline(-1, color='orange', linestyle='--')
    plt.axhline(1, color='orange', linestyle='--', label=r'$|y| = 1$')
    plt.legend()
    plt.grid()
    plt.gca().set_title("g(x): contractividad")
    plt.tight_layout()
    plt.show()

    return g_np


def ptofijo(g_np, x0, maxiter=10000, tol_x=10e-6):
    '''
    Obtiene el punto fijo de g(x), la solución a f(x) = 0

        Devuelve:
            xn (float): punto fijo de g(x)
    '''
    for i in range(maxiter):
        xn = g_np(x0)
        if abs(xn-x0) < tol_x:
            return xn
        x0 = xn
    return xn


# Cálculo de diferentes funciones de punto fijo para f(x) = x*exp(x)-4

x = sym.Symbol("x", float=True)
g1_sym = (x**2*sym.exp(x)+4)/(sym.exp(x)*(x+1))
g1_np = representación_ptofijo(g1_sym, 0, 2)
x_1 = ptofijo(g1_np, 1)

g2_sym = sym.log(4/x)
g2_np = representación_ptofijo(g2_sym, 0.001, 2)
x_2 = ptofijo(g1_np, 1)

g3_sym = 4*sym.exp(-x)
g3_np = representación_ptofijo(g3_sym, 0, 2)

# Se observa que para g_3(x) el punto fijo no está en el rango de valores para los que
# la derivada de g_3(x) está contenida entre (-1,1). El criterio de convergencia
# no se cumple, y el método de punto fijo no da una solución correcta.

print("--> f(x)=x*exp(x) - 4 (métodos pto fijo)")
print("Pto fijo g1 = %0.6f\nPto fijo g2 = %0.6f" % (x_1, x_2))
