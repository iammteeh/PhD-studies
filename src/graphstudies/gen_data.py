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

def markov_from_skeleton(G, bias_range=(0.2, 0.8), coupling_range=(0.5, 2.0), seed=42):
    """
    Binary Markov network:
      - unary factors phi_i(x_i) ~ biased Bernoulli
      - pairwise factors phi_ij(x_i, x_j) ~ Ising-like couplings
    Deterministic subgraphs can be encoded by strong couplings or degenerate tables.
    Graphs can be cyclic and contain cliques, representing complex dependencies and are usually undirected for Markov networks,
    because they represent conditional independence relationships.

    bias_range: tuple(float, float)
        Range for sampling unary bias probabilities. Each variable x_i has P(x_i=1) ~ Uniform(bias_range[0], bias_range[1]).
        default is (0.2, 0.8) to avoid extreme biases. Interpretation depends on coupling strengths, e.g., for strong positive couplings, a bias towards 1 increases the likelihood of connected variables being 1.
    
    coupling_range: tuple(float, float)
        Range for sampling pairwise coupling strengths. Each edge (i, j) has coupling strength J_ij ~ Uniform(coupling_range[0], coupling_range[1]).
        default is (0.5, 2.0) to allow for both weak and strong couplings. Higher values favor agreement (if same=True) or disagreement (if same=False) between connected variables.

    The mechanics of how these parameters influence the joint distribution are rooted in the Ising model framework, where unary factors represent individual variable tendencies and pairwise factors capture interactions between variables.
    Numerically, this results in a complex landscape of dependencies that can be explored through sampling methods. Strong couplings can create near-deterministic relationships, while biases can skew marginal distributions and influence edge connectivity patterns.
    """
    rng = np.random.default_rng(seed)
    M = MarkovModel()
    M.add_nodes_from(G.nodes())
    M.add_edges_from(G.edges())

    factors = []

    # unary
    for v in M.nodes():
        p1 = rng.uniform(*bias_range)
        phi = DiscreteFactor(variables=[v], cardinality=[2], values=[1-p1, p1])
        factors.append(phi)

    # pairwise
    for u, v in M.edges():
        # favor equality or inequality randomly
        J = rng.uniform(*coupling_range)
        same = rng.random() < 0.5 
        table = np.array([
            [J if same else 1, 1],
            [1, J if same else 1]
        ], dtype=float)
        phi = DiscreteFactor(variables=[u, v], cardinality=[2, 2], values=table)
        factors.append(phi)

    M.add_factors(*factors)
    return M

def gibbs_sample_markov(M, size=5000, burn_in=500, seed=123):
    gibbs = GibbsSampling(M)
    return gibbs.sample(size=size, burn_in=burn_in, seed=seed)

def factorgraph_from_skeleton(G, bias_range=(0.2, 0.8), coupling_range=(0.5, 2.0), seed=42):
    """
    Binary Factor Graph:
      - unary factors phi_i(x_i) ~ biased Bernoulli
      - pairwise factors phi_ij(x_i, x_j) ~ Ising-like couplings
    Deterministic subgraphs can be encoded by strong couplings or degenerate tables.

    Is similar to markov_from_skeleton but uses FactorGraph instead of MarkovModel.
    The main difference is that FactorGraph explicitly represents factors as nodes, such that
    the resulting graph is bipartite between variable nodes and factor nodes.
    """
    rng = np.random.default_rng(seed)
    FG = FactorGraph()
    FG.add_nodes_from(G.nodes())
    FG.add_edges_from(G.edges())

    factors = []

    # unary
    for v in FG.nodes():
        p1 = rng.uniform(*bias_range)
        phi = DiscreteFactor(variables=[v], cardinality=[2], values=[1-p1, p1])
        factors.append(phi)

    # pairwise
    for u, v in FG.edges():
        # favor equality or inequality randomly
        J = rng.uniform(*coupling_range)
        same = rng.random() < 0.5
        table = np.array([
            [J if same else 1, 1],
            [1, J if same else 1]
        ], dtype=float)
        phi = DiscreteFactor(variables=[u, v], cardinality=[2, 2], values=table)
        factors.append(phi)

    FG.add_factors(*factors)
    return FG

def gen_data():
    base_graph = generate_random_graph(type="geometric", nodes=50)
    # apply constraints to base_graph to get knowledge graph
    constrained_graph = generate_graph(type="ring_of_cliques", nodes=50)
    G = nx.compose(base_graph, constrained_graph) # symmetric difference to combine edges
    M = markov_from_skeleton(G, bias_range=(0.2, 0.8), coupling_range=(0.5, 2.0), seed=42)
    data = gibbs_sample_markov(M, size=1000, burn_in=500, seed=123)
    return G, M, data