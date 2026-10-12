"""
oc_poisson.py
"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import matplotlib.pyplot as plt

def slap(N):
    h = 1/(N+1) #paso de la malla
    diagonales = [4*np.ones(N), -1*np.ones(N-1),-1*np.ones(N-1)]
    D = sp.diags(diagonales,[0,-1,1], format='csr')
    A = sp.kron(sp.eye(N, format ='csr'),D) + sp.diags([-1*np.ones(N**2-N),-1*np.ones(N**2-N)],[-N,N],format='csr' )
    return (A / (h**2)).tocsr()

def int_om(N):
    h = 1/(N+1)
    M = sp.diags(h**2*np.ones(N**2),0,format='csr')
    return M

def evalf(N):
    f=np.zeros([N,N])
    for i in range(0,N):
        for j in range(0,N):
            f[i,j]= i+j
            
    return f

def malla(N):
    """Nodos internos del cuadrado unitario, N por dirección."""
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")   # X[i, j] = x_i, Y[i, j] = y_j
    return X, Y, h

def evaluar(f, N):
    """Evalúa f(x, y) en los nodos internos y devuelve el vector en R^{N^2}."""
    X, Y, h = malla(N)
    return np.asarray(f(X, Y)).reshape(N*N,1)

def control_to_state(u,A, N):
    "Resuelve el problema de Poisson con control"
    N = u.size
    #A= splan(N)
    y = sp.linalg.spsolve(A,u)
    return y

def mass_matrix(N):
    "Matriz d masa para el problema de Poisson"
    h=1/(N+1)
    M = sp.diags(h**2 * np.ones(N**2), 0, format='csr')
    return M
    
def graficar(y,N, name= ""):
    "Grafica yd sobre la malla interna como un mapa de colores"
    X, Y, _ = malla(N)
    Z = np.asarray(y).reshape(N,N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap = "viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("yd")
    