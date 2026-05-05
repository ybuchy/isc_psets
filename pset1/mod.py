from mpl_toolkits import mplot3d
import numpy as np
import scipy
import matplotlib.pyplot as plt

def grid(n):
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    # Make grid without boundary points
    X, Y = np.meshgrid(x[1:-1], y[1:-1])
    return X, Y

def matrix(n):
    # Matrix with diagonal -2 and off diagonals 1
    M =  np.pow(n, 2) * scipy.sparse.diags([np.full(n-1, -2), np.ones(n-2), np.ones(n-2)], [0, -1, 1]).toarray()
    I = np.eye(n-1)
    # Discretization matrix formula given in lecture
    discr_matrix = np.kron(I, M) + np.kron(M, I)
    return discr_matrix

def f(x, y):
    return np.pow(np.pi, 2) * np.sin(np.pi * x * y) * (np.pow(x, 2) + np.pow(y, 2))

def g(x, y):
    if x == 0 or y == 0:
        return 0
    elif y == 1:
        return np.sin(np.pi * x)
    else:
        return np.sin(np.pi * y)


def sol(x, y):
    return np.sin(np.pi * x * y)


def right_side_LSE(f, g, grid_x, grid_y):
    # 5 point stencil values known from boundary points
    G = np.zeros(grid_x.shape)
    x_vec = grid_x[0, :]
    y_vec = grid_y[:, 0]
    G[0, :] += list(map(lambda x: g(0, x), y_vec))
    G[-1, :] += list(map(lambda x: g(1, x), y_vec))
    G[:, 0] += list(map(lambda x: g(x, 0), x_vec))
    G[:, -1] += list(map(lambda x: g(x, 1), x_vec))

    beta = f(grid_x, grid_y) + np.pow(grid_x.shape[0]+1, 2) * G

    # Order F for writing into vector like in lecture
    b = beta.flatten(order='F')
    return b

if __name__ == "__main__":
    errors = []
    Ns = [10, 14, 20, 28, 40, 56, 80]

    for N in Ns:
        X, Y = grid(N)

        A = matrix(N)
        b = right_side_LSE(f, g, X, Y)
        soln = np.linalg.solve(-A, b)
        exact = sol(X, Y)

        errors.append(np.abs(soln.reshape(X.shape) - exact).max())

    ax = plt.axes()
    ax.loglog(Ns, errors, label="error")
    ax.loglog(Ns, list(map(lambda x: x**-1, Ns)), label="linear")    
    ax.loglog(Ns, list(map(lambda x: x**-2, Ns)), label="quadratic")
    ax.loglog(Ns, list(map(lambda x: x**-3, Ns)), label="cubic")
    ax.legend()
    plt.show()

# d) 1: Größter Aufwand ist np.linalg.solve, benutzt LR-Zerlegung ~> O(N^3), also ca 1000 mla so lange.

# d) 2: diskretisierungsmatrix ist sparse ~> benutze scipy.sparse.linalg.spsolve oder band LR-zerlegung
