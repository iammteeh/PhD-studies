import math
import numpy as np
import networkx as nx


from pgmpy.models import BayesianModel, DynamicBayesianNetwork as DBN, MarkovModel, FactorGraph
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.sampling import BayesianModelSampling, GibbsSampling

# we basically just need any structure over which we can sample a generative model of joint probabilities
# using graphs to model heterogeneous mass functions is simple to understand yet versatile to cover a wide range of applications
# to generate this data, any random graph describes an unknown domain function, which we want to analyze later
# deterministic graphs serve as knowledge graphs over domains to model specific dependencies 

def generate_random_graph(type="geometric", nodes=50):
    match type:
        case "geometric":
            graph = nx.random_geometric_graph(n=nodes, radius=0.25, seed=42)
        case "cograph":
            graph = nx.random_cograph(n=nodes, seed=42)
        case "kernel":
            # Generate an Erdős–Rényi random graph with c as the mean expected degree
            def integral(u, w, z, c=1):
                return c * (z - w)
            def root(u, w, r, c=1):
                return r / c + w
            graph = nx.random_kernel_graph(n=nodes, kernel_integral=integral, kernel_root=root)
            graph = nx.erdos_renyi_graph(n=nodes, p=0.1)
        case "cluster":
            deg = [(0, 1)] * nodes
            graph = nx.random_clustered_graph(joint_degree_sequence=deg, seed=42)
    
    return graph

def generate_graph(type="ring_of_cliques", nodes=50):
    match type:
        case "ring_of_cliques":
            graph = nx.ring_of_cliques(num_cliques=nodes/10, clique_size=nodes/5)
        case "balanced_tree":
            graph = nx.balanced_tree(r=nodes//10, h=nodes//5)
        case "binomial_tree":
            graph = nx.binomial_tree(n=nodes)
        case "binomial_graph":
            graph = nx.gnp_random_graph(n=nodes, p=nodes//10) # probability p of edge creation, expected degree nodes/10, is ideal for sparse graphs and scales well but may create disconnected graphs as well as dense graphs but prone to overfitting because of many edges (exponential growth of cliques)
        case "path":
            graph = nx.path_graph(n=nodes)
        case "complete":
            graph = nx.complete_graph(n=nodes)
        case "bipartite":
            graph = nx.complete_bipartite_graph(n=nodes)
        case "turan":
            graph = nx.turan_graph(n=nodes, r=nodes//10)
        case "star":
            graph = nx.star_graph(n=nodes)
        case "barbell":
            graph = nx.barbell_graph(m1=(nodes//2)-1,m2=(nodes//2)-1)
    
    return graph

