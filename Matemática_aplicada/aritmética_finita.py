#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import sympy as S


e = sys.float_info.epsilon
x = 1
suma1 = x

for i in range(10000):
    suma1 += 1/2*e

print(suma1)

suma2 = 0

for i in range(10000):
    suma2 += 1/2*e

print(suma2 + x)


N = 10000
serie = np.zeros(N)
serie[0] = 1
r = 98/99

for i in range(1, N):
    serie[i] = serie[i-1]*r

s1 = sum(serie)
s2 = sum(serie[::-1])
print(s1, s2)


#-----FUNCIONES DEF------#

def poligono(lado1, lado2, *resto_lados):
    perímetro = lado1 + lado2 + sum(resto_lados)

    return perímetro

def evalpol(p, z):
    '''
    Evalúa un polinomio (p) de grado n en distintos puntos (z)
    '''
    y = p[0]
    for i in range(1, len(p)):
        y = y*z + p[i]
    return y


p = np.array([2, -9, 6, 11, -6])
z = np.array([0, 1/3, 2/3])

y = evalpol(p, z)

print(y)


#-----FUNCIONES LAMBDA-----#

def invertir(x): return x[::-1]


print(invertir('alita'))


#-----FUNCIONES SIMBÓLICAS-----#

x = S.symbol('x')
y, z = S.symbols('y z')
f = x**3-2*x+S.sin(x)
k = f.subs(x, 1)
x = np.linspace(-1, 1)  # 50 puntos por defecto
xdat = np.linspace(-1, 1, 11)
ydat = np.random.rand(11)*2-1




fig = plt.figure(1, figsize=(8, 6))
plt.plot(x, np.sin(np.pi*x), label='seno')
plt.plot(x, np.cos(np.pi*x), ':', label='coseno')
plt.plot(x, np.exp(x), 'o-', label='Exponencial')
plt.plot(xdat, ydat, '*', label='aleatorio')
plt.legend()
plt.grid()
plt.show()
fig.savefig('imagen.png')


r = 1
c = 1
a = np.linspace(1-1, 1+1)
z = a
R = r**2 - (z-x)**2
x = R*np.cos(4*np.pi*a)
y = R*np.sin(4*np.pi*a)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)
