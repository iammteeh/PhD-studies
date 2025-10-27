import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla
import numpy as np
import networkx as nx


def compute_laplacian_spectrum(graph, k=6):
    """Compute the Laplacian spectrum of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.
    k : int, optional
        The number of eigenvalues and eigenvectors to compute, by default 6.

    Returns
    -------
    eigenvalues : np.ndarray
        The computed eigenvalues.
    eigenvectors : np.ndarray
        The computed eigenvectors.
    """
    L = nx.laplacian_matrix(graph).astype(float)
    eigenvalues, eigenvectors = spla.eigsh(L, k=k, which='SM')
    # Sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

def compute_laplace_distance(graph, p=2):
    """Compute the Laplace distance matrix of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.
    p : float, optional
        The power to which the Laplacian is raised, by default 2.

    Returns
    -------
    distance_matrix : np.ndarray
        The computed Laplace distance matrix.
    """
    L = nx.laplacian_matrix(graph).astype(float)
    L_pinv = spla.pinv(spla.fractional_matrix_power(L.toarray(), p))
    n = L.shape[0]
    diag = np.diag(L_pinv)
    distance_matrix = np.add.outer(diag, diag) - 2 * L_pinv
    return distance_matrix