#!/usr/bin/env python3
import numpy as np

###SISTEMAS_LINEALES###
np.random.seed(1000)
###MÉTODOS_DIRECTOS###

print(np.arange(4 - 1, -1, -1))
def solve_tri(A, b):
    filas_A, columnas_A = A.shape
    x = np.zeros_like(b, dtype=float)
    for i in np.arange(filas_A - 1, -1, -1):
        suma = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i]-suma)/A[i, i]
    return x


def triangulación_gauss(A, b):
    filas_A, columnas_A = A.shape
    for i in range(filas_A):
        if A[i, i] == 0:
            raise ValueError('El pivote es nulo')
        for j in range(i+1, filas_A):
            m = A[j, i]/A[i, i]
            A[j, i+1:] = A[j, i+1:]-m*A[i, i+1:]
            b[j] = b[j] - m*b[i]
            A[j, i] = 0
            print(b)

# def triangulacion_gauss_superior(A, b): # PROBABLEMENTE MAL
#     filas_A, columnas_A = A.shape
#     for i in np.arange(filas_A - 1, 0, -1):
#         if A[i, i] == 0:
#             raise ValueError('El pivote es nulo')
#         for j in np.arange(i):
#             m = A[j, i]/A[i, i]
#             A[j, :i+1] = A[j, :i+1]-m*A[i, :i+1]
#             b[j] = b[j] - m*b[i]
#             A[j, i] = 0


A = np.random.rand(4, 4)
b = np.random.rand(4).T

def gauss(A, b):
    diagonal = np.diag(A)
    filas_A, columnas_A = A.shape

    if filas_A != columnas_A:
        raise ValueError('A no es matriz cuadrada')

    if not diagonal.all():
        raise ValueError('A es singular')

    if len(b.shape) != 1:
        raise ValueError('B no es un vector')

    if filas_A != b.shape[0]:
        raise ValueError('El vector y la matriz no son compatibles')
    A_trabajo = A.copy()
    b_trabajo = b.copy()			# para no cambiar las matrices orginales
    triangulación_gauss(A_trabajo, b_trabajo)
    return solve_tri(A_trabajo, b_trabajo)

# def gauss_jordan(A):            # MAL
#     n = len(A)
#     Inv = np.eye(n)
#     triangulación_gauss(A, Inv)
#     triangulacion_gauss_superior(A, Inv)
#     for i in range(n):
#         A[i, i] = A[i, i]/A[i, i]
#         Inv[i, i] = Inv[i, i]/A[i, i]
#     return A, Inv

def gaussjordan(A):  # maría
    fA, cA = A.shape
    C = np.eye(fA)
    D = A  # para no modificar la matriz original
    for i in range(fA):  # columnas
        for j in range(fA):
            if j == i:
                continue
            mji = -D[j, i]/D[i, i]
            D[j, :] = D[j, :]+mji*D[i, :]
            C[j, :] = C[j, :]+mji*C[i, :]
    diagonal = np.diag(D)
    diagonali = diagonal**(-1)
    Di = np.zeros_like(A)
    for k in range(fA):
        Di[k, k] = diagonali[k]
    return D, C, np.dot(Di, C)

def triangulación_pivoteo(A, b):
    diagonal = np.diag(A)
    filas_A, columnas_A = A.shape

    for i in range(filas_A):
        for j in range(i+1, filas_A):
            vector = abs(A[i:, i])
            pos_max = vector.argmax()
            A[[i, i+pos_max], :] = A[[i+pos_max, i], :]
            b[[i, i+pos_max]] = b[[i+pos_max, i]]
            m = A[j, i]/A[i, i]
            A[j, i+1:] = A[j, i+1:]-m*A[i, i+1:]
            b[j] = b[j] - m*b[i]
            A[j, i] = 0


def gauss_pivoteo(A, b):
    diagonal = np.diag(A)
    filas_A, columnas_A = A.shape
    A_trabajo = A.copy()
    b_trabajo = b.copy()
    triangulación_pivoteo(A_trabajo, b_trabajo)
    return solve_tri(A_trabajo, b_trabajo)


print(gauss(A, b))
print(np.linalg.solve(A, b))
print(gauss_pivoteo(A, b))
# los algoritmos dan la solución correcta

def doodlite(A):
    filas_A, columnas_A = A.shape
    U = A.copy()
    L = np.eye(filas_A, columnas_A)
    for i in range(filas_A):
        if A[i, i] == 0:
            raise ValueError('El pivote es nulo')
        for j in range(i+1, filas_A):
            m = U[j, i]/U[i, i]
            L[j, i] = m
            U[j, i+1:] = U[j, i+1:]-m*U[i, i+1:]
            U[j, i] = 0
    return L, U

L, U = doodlite(A)

def jacobi(A, b):
    D = np.diag(np.diag(A))
    L = - np.tril(A, -1)
    U = - np.triu(A, 1)
    K = np.dot(np.linalg.inv(D), (L+U))
    d = np.dot(np.linalg.inv(D), b)
    return K, d

def gauss_seidel(A, b):
    D = np.diag(np.diag(A))
    L = - np.tril(A, -1)
    U = - np.triu(A, 1)
    K = np.dot(np.linalg.inv(D-L), U)
    d = np.dot(np.linalg.inv(D-L), b)
    return K, d

def relajación(A, b, omega):
    D = np.diag(np.diag(A))
    L = - np.tril(A, -1)
    U = - np.triu(A, 1)
    K = np.dot(np.linalg.inv(D - omega*L), (1 - omega)*D + omega*U)
    d = omega*np.dot(np.linalg.inv(D - omega*L), b)
    return K, d

def iter_solve(max_iter, A, b):
    método = str(input(
        'Método de resolución iterativo? Jacobi, gauss_seidel o relajación: ')).lower()
    if método == 'relajación':
        omega = float(input('Introduzca constante de relajación: '))
        K, d = relajación(A, b, omega)

    if método == 'gauss_seidel':
        K, d = gauss_seidel(A, b)

    if método == 'jacobi':
        K, d = jacobi(A, b)

    else:
        print('Método no válido, seleccione un método válido: ')
        iter_solve(max_iter, A, b)

    K, d = gauss_seidel(A, b)
    x0 = b.copy()
    for i in range(max_iter):
        x0 = np.dot(K, x0) + d
    return x0

### AUTOVALORES ###

A = np.array([[1, 2, 1, 2], [2, 1, 1, 1], [3, 2, 1, 2], [2, 1, 1, 4]])

def potencia_iter(A, maxiter=100):
    w0 = np.random.rand(4)
    w0 = w0/np.linalg.norm(w0)
    for k in range(maxiter):
        uk = np.dot(A, w0)
        wk = uk/np.linalg.norm(uk)
        w0 = wk
    eigenvalue = np.dot(np.dot(wk.T, A), wk)
    return eigenvalue


def pot_inver(A, maxiter=100):
    w0 = np.random.rand(4)
    w0 = w0/np.linalg.norm(w0)
    for k in range(maxiter):
        uk = np.dot(np.linalg.inv(A), w0)
        wk = uk/np.linalg.norm(uk)
        w0 = wk
    eigenvalue = (np.dot(np.dot(wk.T, A), wk))
    return eigenvalue


def pot_inver_decalado(A, d, maxiter=1000):
    n = len(A)
    w0 = np.random.rand(n)
    w0 = w0/np.linalg.norm(w0)
    I = np.eye(n)
    Ai = A-d*I
    for k in range(maxiter):
        uk = np.dot(np.linalg.inv(Ai), w0)
        wk = uk/np.linalg.norm(uk)
        w0 = wk
    eigenvalue = (np.dot(np.dot(wk.T, Ai), wk))
    return eigenvalue

print(potencia_iter(A))
print(pot_inver(A))
# print(pot_inver_decalado(A, 1.7))
print(np.linalg.eig(A)[0])
# los algoritmos dan los autovalores esperados
