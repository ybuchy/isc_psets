from dataclasses import dataclass
import numpy as np
import meshio
import matplotlib.pyplot as plt
import itertools

@dataclass
class Triangulation:
    """
    Hier sollte Ihre Klassendefinition stehen. Achten Sie darauf, dass Ihre 
    Klasse tatsächlich den angegebenen Namen "Triangulation" hat!
    """

    # Hier steht Ihr Code
    points: list
    triangles: list
    edges_dir: list
    edges_neu: list

    def plot(self):
        fig, ax = plt.subplots()
        ax.triplot(list(map(lambda x: x[0], self.points)), list(map(lambda x: x[1], self.points)), self.triangles)
        plt.show()

    def get_hmax(self):
        # double counts!
        h_max = 0
        for triangle in self.triangles:
            for edge in zip(triangle, [*triangle[1:], triangle[0]]):
                p1 = np.asarray(self.points[edge[0]], dtype=np.float32)
                p2 = np.asarray(self.points[edge[1]], dtype=np.float32)
                h_max = max(h_max, np.linalg.norm(p2 - p1))
        return h_max

    def inner_points(self):
        edges = [set(edge) for triang in self.triangles for edge in itertools.combinations(triang, 2)]
        edge_counts = {tuple(edge):edges.count(edge) for edge in edges}
        outer_edges = [edge for edge in edges if edge_counts[tuple(edge)] == 1]
        outer_points = set(point for edge in outer_edges for point in edge)
        return [k not in outer_points for k in range(len(self.points))]
        

def funct(x, y):
    return 15 * x * y * (1 - 1/4 * x ** 2 - y ** 2) - 1/2 * x ** 3 * y - 8 * x * y ** 3
    
def exact(x, y):
    return x * y * (1 - 1/4 * x ** 2 - y ** 2) ** 2
    
def poisson(triangulation, f):
    areas = []
    N = len(triangulation.points)
    A = np.zeros((N, N))
    b = np.zeros((N))
    for triangle in triangulation.triangles:

        # needed for formulas given in lecture for linear F.E.
        M = np.array(tuple([1, *triangulation.points[point]] for point in triangle))
        M_inv = np.linalg.inv(M)
        area = 1/2 * abs(np.linalg.det(M))
        areas.append(area)

        # Calculate A
        for (k1, point), (k2, point2) in itertools.product(enumerate(triangle), enumerate(triangle)):
            # A symmetric, so don't have to waste computations
            if point < point2:
                a = area * (M_inv[1][k1] * M_inv[1][k2] + M_inv[2][k1] * M_inv[2][k2])
                A[point, point2] += a
                A[point2, point] += a
            elif point == point2:
                A[point, point] += area * (M_inv[1][k1] * M_inv[1][k2] + M_inv[2][k1] * M_inv[2][k2])

        # Calculate b
        points = [triangulation.points[point] for point in triangle]
        midpoint = 1/3 * np.fromiter(map(sum, zip(*points)), dtype='f')
        for point in triangle:
            b[point] += 1/3 * area * f(*midpoint)

    is_inner = triangulation.inner_points()
    # Use mask to get only inner rows/cols
    A_inner = A[np.ix_(is_inner, is_inner)]
    b_inner = b[is_inner]

    return A_inner, b_inner, areas
        
def read_msh(filename):
    """
    Lese msh-File ein, lese alle notwendigen Daten aus und erstelle daraus eine
    Instanz der selbst definierten Klasse 'Triangulation'
    
    Parameters
    ----------
    filename : str
        filename der msh-Datei, in der die Triangulierung gespeichert ist.

    Returns
    -------
    Triangulierung
        Triangulierung als Instanz der selbst definierten Klasse 
        'Triangulierung'
    """
    
    # Endung .msh ergänzen, falls noch nicht der Fall
    if not filename.endswith('.msh'):
        filename += '.msh'
        
    mesh = meshio.read(filename)
    
    # relevante Daten auslesen:
    
    # 1.) Punkteliste mit Koordinaten aller Punkte
    points_3d = mesh.points
    points = points_3d[:, :2]
    
    # 2.) Dreiecksliste
    triangles = mesh.cells_dict['triangle']
    
    # 3.) Dirichlet-Rand-Kanten
    if 'DirichletRand' in mesh.field_data: # Checke, ob entsprechender Key existiert
        dir_tag = mesh.field_data['DirichletRand'][0]
        edge_dir_YN = ( mesh.cell_data_dict['gmsh:physical']['line'] == dir_tag )
        edges_dir = mesh.cells_dict['line'][edge_dir_YN]
    else:
        edges_dir = np.zeros([0, 2], dtype=int) 
    
    # 4.) Neumann-Rand-Kanten
    if 'NeumannRand' in mesh.field_data: # Checke, ob entsprechender Key existiert
        neu_tag = mesh.field_data['NeumannRand'][0]
        edge_neu_YN = ( mesh.cell_data_dict['gmsh:physical']['line'] == neu_tag )
        edges_neu = mesh.cells_dict['line'][edge_neu_YN]
    else:
        edges_neu = np.zeros([0, 2], dtype=int)
    
    return Triangulation(points, triangles, edges_dir, edges_neu)


if __name__ == "__main__":
    t = Triangulation([(0,0), (3,0), (1,1), (2,1), (1.5, 2), (0,3), (3,3)], [[0,2,5],[0,2,3],[0,1,3],[5,2,4],[2,3,4],[4,5,6],[3,4,6],[1,3,6]], [0,1,5,6], [])
    #t.plot()
    t.get_hmax()
    t.inner_points()
    poisson(t, funct)
    filenames = [f"meshes/elliplse{k : 03d}.msh" for k in range(11)]
    tri = read_msh("meshes/ellipse07.msh")
    #tri.plot()
    A, b, areas = poisson(tri, funct)
    approx_inner = np.linalg.solve(A, b)
    Px = list(map(lambda x: x[0], tri.points))
    Py = list(map(lambda x: x[1], tri.points))
    N = len(Px)
    approx = np.zeros((N))
    approx[tri.inner_points()] = approx_inner
    sol = [exact(pt[0], pt[1]) for pt in tri.points]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1, projection = "3d")
    ax1.plot_trisurf(Px, Py, tri.triangles, approx, cmap='viridis')
    ax1.set_title("approx")

    ax2 = fig.add_subplot(2, 2, 2, projection = "3d")
    ax2.plot_trisurf(Px, Py, tri.triangles, sol, cmap='viridis')
    ax2.set_title("sol")
    #plt.show()

    eps = lambda pt: abs(exact(*tri.points[pt]) - approx[pt]) ** 2
    error = 0
    for triangle, area in zip(tri.triangles, areas):
        error += area * 1/3 * (sum(eps(pt) for pt in triangle))
    error = np.sqrt(error)
    
    filenames = [f"meshes/ellipse{k:02d}.msh" for k in range(11)]
    triangulations = [read_msh(filename) for filename in filenames]
    errors = []
    h_max = []

    for k, tri in enumerate(triangulations):
        print(k)
        A, b, areas = poisson(tri, funct)
        approx_inner = np.linalg.solve(A, b)
        Px = list(map(lambda x: x[0], tri.points))
        Py = list(map(lambda x: x[1], tri.points))
        N = len(Px)
        approx = np.zeros((N))
        approx[tri.inner_points()] = approx_inner
        sol = [exact(pt[0], pt[1]) for pt in tri.points]

        eps = lambda pt: abs(exact(*tri.points[pt]) - approx[pt]) ** 2
        error = 0
        for triangle, area in zip(tri.triangles, areas):
            error += area * 1/3 * (sum(eps(pt) for pt in triangle))
        errors.append(np.sqrt(error))
        h_max.append(tri.get_hmax())

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.loglog(h_max, errors, label="error")
    ax3.loglog(h_max, list(map(lambda x: .1 * x ** 1, h_max)), label="linear")
    ax3.loglog(h_max, list(map(lambda x: .1 * x ** 2, h_max)), label="quadratic")
    ax3.loglog(h_max, list(map(lambda x: .05 * x ** 3, h_max)), label="cubic")
    ax3.invert_xaxis()
    ax3.legend()
    plt.tight_layout()
    plt.show()
