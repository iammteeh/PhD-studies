import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla
import numpy as np
import networkx as nx

from src.graphstudies.compute_laplacian import compute_laplace_distance, compute_laplacian_spectrum

def spectral_weight_matrix(graph, k=6):
    """Compute the spectral weight matrix of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.
    k : int, optional
        The number of eigenvalues and eigenvectors to compute, by default 6.

    Returns
    -------
    weight_matrix : np.ndarray
        The computed spectral weight matrix.
    """
    L = nx.laplacian_matrix(graph).astype(float)
    eigenvalues, eigenvectors = spla.eigsh(L, k=k, which='SM')
    # Sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Compute the spectral weight matrix
    weight_matrix = np.zeros((L.shape[0], L.shape[0]))
    for i in range(k):
        weight_matrix += (1.0 / (eigenvalues[i] + 1e-10)) * np.outer(eigenvectors[:, i], eigenvectors[:, i])
    return weight_matrix

def matern_spectral_weight_matrix(graph, k=6, alpha=1.0):
    """Compute the Matern spectral weight matrix of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The input graph.
    k : int, optional
        The number of eigenvalues and eigenvectors to compute, by default 6.
    alpha : float, optional
        The smoothness parameter for the Matern kernel, by default 1.0.

    Returns
    -------
    weight_matrix : np.ndarray
        The computed Matern spectral weight matrix.
    """
    weight_matrix = spectral_weight_matrix(graph, k=k)
    # Apply the Matern kernel
    distance_matrix = compute_laplace_distance(graph)
    weight_matrix *= np.exp(-alpha * distance_matrix)
    return weight_matrix