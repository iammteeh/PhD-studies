import math
import numpy as np
import networkx as nx


from pgmpy.models import DiscreteBayesianNetwork, DynamicBayesianNetwork as DBN, MarkovNetwork, FactorGraph, LinearGaussianBayesianNetwork, FunctionalBayesianNetwork, ClusterGraph
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor, NoisyORCPD, JointProbabilityDistribution
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.sampling import BayesianModelSampling, GibbsSampling

# we basically just need any structure (R,F,G,A) over which we can sample a generative model of joint probabilities
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

# Mor specifically, it is useful to model binary Markov networks and factor graphs over random graph structures
# to generate synthetic data for testing inference algorithms and learning algorithms in probabilistic graphical models.
# These models can capture complex dependencies and interactions between binary variables, thus are easy to interpret, making them suitable for various applications
# because once we have a stationary distribution defined by the graph structure and associated factors, we can sample from this distribution to generate synthetic data.
# This synthetic data can then be used to evaluate the performance of inference algorithms (e.g., Gibbs sampling, belief propagation) and learning algorithms (e.g., parameter estimation, structure learning) in probabilistic graphical models.

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
    M = MarkovNetwork()
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

# To represent more complex interactions and higher-order dependencies between variables,
# we can use Factor Graphs, which explicitly represent factors as nodes in the graph.
# This allows us to model not only pairwise interactions but also higher-order interactions among multiple variables.
# Factor Graphs are bipartite graphs consisting of variable nodes and factor nodes. Normally, Factor Graphs are undirected graphs, as they represent joint distributions without inherent directionality.
# They can also contain cycles and cliques, similar to Markov networks, allowing for the representation of complex dependencies among variables.
# It is particularly useful for representing models where factors involve multiple variables, such as in error-correcting codes, constraint satisfaction problems, and certain types of probabilistic graphical models.
# For example, in a Factor Graph, we can have factors that represent interactions among three or more variables, capturing more intricate relationships than pairwise factors alone.
# Especially if the factorization represents a function, such that we can use FunctionalCPD to define deterministic relationships between variables over which we can sample random processes.

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

# To induce conditional dynamics to control when there is stable flow in the network,
# we can use Dynamic Bayesian Networks (DBNs) to model temporal dependencies between variables across time
# This allows us to capture how the state of variables evolves over time based on their previous states
# and the influence of other variables in the network.
# By defining transition probabilities and temporal dependencies,
# we can simulate the dynamic behavior of the system and generate time-series data that reflects these dynamics
# For the start, we assume that the random CPD reflect potential temporal dependencies between variables across time steps,
# while we construct static structures and explore their dynamics through sampling.

def generate_random_CPD(G):
    """Generate random CPDs for a DiscreteBayesianNetwork defined by the DAG G."""
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            p = rng.uniform(0.2, 0.8)
            cpd = TabularCPD(variable=node, variable_card=2, values=[[1-p], [p]])
        else:
            # child node
            parent_card = [2] * len(parents)
            num_rows = 2
            num_cols = 2 ** len(parents)
            values = np.zeros((num_rows, num_cols))
            for col in range(num_cols):
                p = rng.uniform(0.2, 0.8)
                values[0, col] = 1 - p
                values[1, col] = p
            cpd = TabularCPD(variable=node, variable_card=2, evidence=parents, evidence_card=parent_card, values=values)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_random_functional_CPD(G):
    """Generate functional CPDs for a Functional Bayesian Network defined by skeleton G."""
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            func = lambda : rng.integers(0, 2)  # random binary value
            cpd = FunctionalCPD(variable=node, function=func, evidence=[])
        else:
            # child node
            def func(*args):
                return int(sum(args) % 2)  # simple parity function
            cpd = FunctionalCPD(variable=node, function=func, evidence=parents)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_random_linear_gaussian_CPD(G):
    """Generate linear Gaussian CPDs for a Linear Gaussian Bayesian Network defined by skeleton G."""
    model = LinearGaussianBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            mean = rng.normal(0, 1)
            variance = rng.uniform(0.5, 2.0)
            cpd = LinearGaussianCPD(variable=node, mean=mean, variance=variance, evidence=[])
        else:
            # child node
            coefficients = {parent: rng.uniform(0.5, 1.5) for parent in parents}
            variance = rng.uniform(0.5, 2.0)
            cpd = LinearGaussianCPD(variable=node, coefficients=coefficients, variance=variance, evidence=parents)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_complex_family_functional_CPD(G):
    """Generate complex functional CPDs for a Functional Bayesian Network defined by skeleton G.
    this function creates CPDs that combine multiple functional forms to model complex dependencies. 
     For example, a node's value could depend on a combination of parity, growth, and decay functions of its parents.
     1. Parity Function: The node's value is determined by the parity (even or odd sum) of its parents' values.
     2. Growth Function: The node's value increases with the sum of its parents' values, capped at a maximum.
     3. Decay Function: The node's value decreases with the sum of its parents' values, floored at a minimum.
    """
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []

    # Example for simple product of functions:
    def func(*args):
        prod = 1
        for arg in args:
            prod *= arg
        return int(prod % 2)  # product mod 2
    
    #To model an experimentally useful complex family of functional CPDs, we can define a composite function that combines these behaviors:
    def complex_func(*args):
        parity = sum(args) % 2
        growth = min(sum(args) + 1, 1)
        decay = max(sum(args) - 1, 0)
        # Combine the effects (this is just an example; the combination logic can be adjusted)
        combined_value = (parity + growth + decay) / 3
        return int(combined_value > 0.5)  # thresholding for binary output
    
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            func = lambda : rng.integers(0, 2)  # random binary value
            cpd = FunctionalCPD(variable=node, function=func, evidence=[])
        else:
            # child node
            cpd = FunctionalCPD(variable=node, function=complex_func, evidence=parents)
        cpds.append(cpd)
    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_growth_functional_CPD(G):
    """Generate growth functional CPDs for a Functional Bayesian Network defined by skeleton G."""
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            func = lambda : rng.integers(0, 2)  # random binary value
            cpd = FunctionalCPD(variable=node, function=func, evidence=[])
        else:
            # child node
            def func(*args):
                return int(min(sum(args) + 1, 1))  # growth function capped at 1
            cpd = FunctionalCPD(variable=node, function=func, evidence=parents)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_decay_functional_CPD(G):
    """Generate decay functional CPDs for a Functional Bayesian Network defined by skeleton G."""
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            func = lambda : rng.integers(0, 2)  # random binary value
            cpd = FunctionalCPD(variable=node, function=func, evidence=[])
        else:
            # child node
            def func(*args):
                return int(max(sum(args) - 1, 0))  # decay function floored at 0
            cpd = FunctionalCPD(variable=node, function=func, evidence=parents)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_matern_functional_CPD(G, alpha=1.0):
    """Generate Matern functional CPDs for a Functional Bayesian Network defined by skeleton G."""
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            func = lambda : rng.integers(0, 2)  # random binary value
            cpd = FunctionalCPD(variable=node, function=func, evidence=[])
        else:
            # child node
            def func(*args):
                distance = sum(args)
                matern_value = (1 + (math.sqrt(3) * distance) / alpha) * math.exp(-(math.sqrt(3) * distance) / alpha)
                return int(matern_value > 0.5)  # thresholding for binary output
            cpd = FunctionalCPD(variable=node, function=func, evidence=parents)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_regular_functional_CPD(G):
    """
    Generate regular functional CPDs for a Functional Bayesian Network defined by skeleton G.
    Regular functions could be simple deterministic mappings, such as parity functions or threshold functions,
    such that we get partial conditional dependencies between variables, which we can control via parameters.
    We shape the partial conditional dependence as noise over the functional CPD, such that we can switch between
    confounding or mediating effects in the network structure.

    This can be done by defining a function that maps parent values to child values with some added randomness.
    """
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            func = lambda : rng.integers(0, 2)  # random binary value
            cpd = FunctionalCPD(variable=node, function=func, evidence=[])
        else:
            # child node
            def func(*args):
                base_value = sum(args) % 2  # simple parity function
                noise = rng.random() < 0.1  # 10% chance to flip the value
                if noise:
                    return int(1 - base_value)
                return int(base_value)
            cpd = FunctionalCPD(variable=node, function=func, evidence=parents)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def elastic_functional_CPD(G, noise_level=0.1):
    """Generate elastic functional CPDs for a Functional Bayesian Network defined by skeleton G.
    This way we have random functional CPDs with controllable noise to model partial conditional dependencies,
    which change the likelihood of certain configurations in the joint distribution, such that we can explore confounding or mediating effects in the network structure,
    or estimate the robustness of inference algorithms to noise in the functional relationships.
    If we use a graphical elastic net regularization approach, we can control the sparsity and smoothness of the functional relationships in the network.
    Sparse functions can represent strong, direct dependencies between variables, while smooth functions can capture more gradual changes and indirect dependencies, to
    get a feeling of how robust inference is to different types of functional relationships and their (factorial) complexities.
    The noise_level parameter controls the amount of randomness introduced into the functional relationships.
    """
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            func = lambda : rng.integers(0, 2)  # random binary value
            cpd = FunctionalCPD(variable=node, function=func, evidence=[])
        else:
            # child node
            def func(*args):
                base_value = sum(args) % 2  # simple parity function
                noise = rng.normal(loc=0.0, scale=noise_level)
                return int(base_value + noise > 0.5)  # thresholding for binary output
            cpd = FunctionalCPD(variable=node, function=func, evidence=parents)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_factored_noise_functional_CPD(G, noise_level=0.1):
    """
    this is similar to elastic_functional_CPD but uses a factored noise model to introduce randomness into the functional relationships, which 
    is essentially a way to model partial conditional dependencies with structured noise, a.k.a. causal noise as structured equations.
    """
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(G.nodes())
    model.add_edges_from(G.edges())

    rng = np.random.default_rng(42)
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        if not parents:
            # root node
            func = lambda : rng.integers(0, 2)  # random binary value
            cpd = FunctionalCPD(variable=node, function=func, evidence=[])
        else:
            # child node
            def func(*args):
                base_value = sum(args) % 2  # simple parity function
                # factored noise: each parent contributes some noise
                total_noise = sum(rng.normal(loc=0.0, scale=noise_level) for _ in args)
                return int(base_value + total_noise > 0.5)  # thresholding for binary output
            cpd = FunctionalCPD(variable=node, function=func, evidence=parents)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def generate_functional_bayesian_data(G, num_samples=1000, functional_type="complex", seed=42, sampling_method="forward"):
    """Generate data for a Functional Bayesian Network defined by the DAG G."""
    match functional_type:
        case "linear_gaussian":
            model = generate_random_linear_gaussian_CPD(G)
        case "complex":
            model = generate_complex_family_functional_CPD(G)
        case "growth":
            model = generate_growth_functional_CPD(G)
        case "decay":
            model = generate_decay_functional_CPD(G)
        case "matern":
            model = generate_matern_functional_CPD(G)
        case _:
            model = generate_random_functional_CPD(G)

    match sampling_method:
        case "forward":
            sampler = BayesianModelSampling(model)
            data = sampler.forward_sample(size=num_samples, seed=seed) # is essentially a gaussian process over the functional CPDs
        case "Gibbs":
            gibbs = GibbsSampling(model)
            data = gibbs.sample(size=num_samples, burn_in=500, seed=seed)
        case "simulate":
            data = model.simulate(num_samples, seed=seed)
        case _:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    return data

def linear_structural_equation_model(G, noise_scale=0.1, seed=42):
    """Generate data from a linear Gaussian structural equation model defined by the DAG G."""
    rng = np.random.default_rng(seed)
    data = {}
    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        if not parents:
            # root node, sample from standard normal
            data[node] = rng.normal(loc=0.0, scale=noise_scale)
        else:
            # linear combination of parents + noise
            parent_values = np.array([data[p] for p in parents])
            weights = rng.uniform(0.5, 1.5, size=len(parents))
            noise = rng.normal(loc=0.0, scale=noise_scale)
            data[node] = np.dot(weights, parent_values) + noise
    return data

def hierarchical_bayesian_model(G, group_var, noise_scale=0.1, seed=42):
    """Generate data from a hierarchical Bayesian model defined by the DAG G with a grouping variable."""
    rng = np.random.default_rng(seed)
    data = {}
    group_levels = rng.integers(low=0, high=3, size=10)  # assume 3 groups for simplicity, could be a Gaussian process as well
    group_params = {level: rng.normal(loc=0.0, scale=1.0) for level in group_levels}

    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        if node == group_var:
            # grouping variable
            data[node] = rng.choice(group_levels)
        elif not parents:
            # root node, sample from group-specific distribution
            group_level = data[group_var]
            mu = group_params[group_level]
            data[node] = rng.normal(loc=mu, scale=noise_scale)
        else:
            # linear combination of parents + noise
            parent_values = np.array([data[p] for p in parents])
            weights = rng.uniform(0.5, 1.5, size=len(parents))
            noise = rng.normal(loc=0.0, scale=noise_scale)
            data[node] = np.dot(weights, parent_values) + noise
    return data

def gibbs_sample_markov(M, size=5000, burn_in=500, seed=123):
    gibbs = GibbsSampling(M)
    return gibbs.sample(size=size, burn_in=burn_in, seed=seed)

def generate_random_data(G, num_samples=1000):
    """Generate random data for a DiscreteBayesianNetwork defined by the DAG G."""
    model = generate_random_CPD(G)
    data = model.simulate(num_samples, seed=42)
    return data

def gen_random_fields_graph_data():
    base_graph = generate_random_graph(type="geometric", nodes=50)
    # apply constraints to base_graph to get knowledge graph
    constrained_graph = generate_graph(type="ring_of_cliques", nodes=50)
    G = nx.compose(base_graph, constrained_graph) # symmetric difference to combine edges
    M = markov_from_skeleton(G, bias_range=(0.2, 0.8), coupling_range=(0.5, 2.0), seed=42)
    data = gibbs_sample_markov(M, size=1000, burn_in=500, seed=123)
    return G, M, data

def gen_structural_equation_data():
    """
    First, we create a directed acyclic graph (DAG) to represent the causal structure of the variables.
    Then, we generate data from a linear Gaussian structural equation model defined by this DAG.
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ('X1', 'Y1'),
        ('X2', 'Y1'),
        ('Y1', 'Y2'),
        ('Y2', 'Y3')
    ])
    data = linear_structural_equation_model(G, noise_scale=0.1, seed=42)
    return G, data

def gen_hierarchical_bayesian_data():
    G = nx.DiGraph()
    G.add_edges_from([
        ('Group', 'A'),
        ('Group', 'B'),
        ('A', 'C'),
        ('B', 'C'),
        ('C', 'D')
    ])
    data = hierarchical_bayesian_model(G, group_var='Group', noise_scale=0.1, seed=42)
    return G, data

def generate_random_cluster_graph(G, bias_range=(0.2, 0.8), coupling_range=(0.5, 2.0), seed=42):
    """
    Generate a Cluster Graph from the skeleton G with binary variables.
    Each cluster corresponds to a clique in the original graph.
    Factors are defined similarly to markov_from_skeleton.

    Cluster Graphs are useful for representing complex dependencies and performing inference using the junction tree algorithm.
    """
    rng = np.random.default_rng(seed)
    CG = ClusterGraph()

    # Find cliques in the graph
    cliques = list(nx.find_cliques(G))
    for clique in cliques:
        CG.add_cluster(clique)

    factors = []

    # unary factors for each variable
    for v in G.nodes():
        p1 = rng.uniform(*bias_range)
        phi = DiscreteFactor(variables=[v], cardinality=[2], values=[1-p1, p1])
        factors.append(phi)

    # pairwise factors for each edge
    for u, v in G.edges():
        J = rng.uniform(*coupling_range)
        same = rng.random() < 0.5 
        table = np.array([
            [J if same else 1, 1],
            [1, J if same else 1]
        ], dtype=float)
        phi = DiscreteFactor(variables=[u, v], cardinality=[2, 2], values=table)
        factors.append(phi)

    CG.add_factors(*factors)
    return CG

def gibbs_sample_cluster_graph(CG, size=5000, burn_in=500, seed=123):
    gibbs = GibbsSampling(CG)
    return gibbs.sample(size=size, burn_in=burn_in, seed=seed)

def generate_covariance_data(G, C=1.0, noise_scale=0.1, num_samples=1000, seed=42):
    """Generate data from a Gaussian graphical model defined by the undirected graph G."""
    rng = np.random.default_rng(seed)
    n = len(G.nodes())
    adj_matrix = nx.to_numpy_array(G)
    precision_matrix = C * adj_matrix + np.eye(n)  # simple precision matrix
    covariance_matrix = np.linalg.inv(precision_matrix)

    data = rng.multivariate_normal(mean=np.zeros(n), cov=covariance_matrix + noise_scale * np.eye(n), size=num_samples)
    return data

def gaussian_from_skeleton(G, noise_scale=0.1, seed=42):
    """Generate a Gaussian graphical model from the skeleton G."""
    rng = np.random.default_rng(seed)
    M = MarkovNetwork()
    M.add_nodes_from(G.nodes())
    M.add_edges_from(G.edges())

    factors = []

    # unary
    for v in M.nodes():
        mean = rng.normal(0, 1)
        variance = rng.uniform(0.5, 2.0)
        phi = DiscreteFactor(variables=[v], cardinality=[2], values=[mean - math.sqrt(variance), mean + math.sqrt(variance)])
        factors.append(phi)

    # pairwise
    for u, v in M.edges():
        covariance = rng.uniform(0.5, 2.0)
        table = np.array([
            [covariance, 0],
            [0, covariance]
        ], dtype=float)
        phi = DiscreteFactor(variables=[u, v], cardinality=[2, 2], values=table)
        factors.append(phi)

    M.add_factors(*factors)
    return M

def gen_covariance_graph_data(graph_type="geometric", structure_type="markov", functional_type="complex"):
    match graph_type:
        case "geometric":
            G = generate_random_graph(type="geometric", nodes=50)
        case "cograph":
            G = generate_random_graph(type="cograph", nodes=50)
        case "kernel":
            G = generate_random_graph(type="kernel", nodes=50)
        case "cluster":
            G = generate_random_graph(type="cluster", nodes=50)
        case _:
            raise ValueError(f"Unknown graph type: {graph_type}")
    # induce CPD structure via covariance
    if structure_type == "markov":
        M = markov_from_skeleton(G, bias_range=(0.2, 0.8), coupling_range=(0.5, 2.0), seed=42)
    elif structure_type == "gaussian":
        M = gaussian_from_skeleton(G, noise_scale=0.1, seed=42)
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")
    # add functional CPDs to model complex dependencies
    #M.add_functional_cpd("f1", [0, 1], [2], lambda x: x[0] + x[1])
    #M.add_functional_cpd("f2", [1, 2], [3], lambda x: x[0] * x[1])
    #M.add_functional_cpd("f3", [0, 2], [3], lambda x: x[0] - x[1])
    # get functional CPD by model type
    match functional_type:
        case "linear_gaussian":
            model = generate_random_linear_gaussian_CPD(G)
            functional_cpd = model.get_cpds()
        case "complex":
            model = generate_complex_family_functional_CPD(G)
            functional_cpd = model.get_cpds()
        case "growth":
            model = generate_growth_functional_CPD(G)
            functional_cpd = model.get_cpds()
        case "decay":
            model = generate_decay_functional_CPD(G)
            functional_cpd = model.get_cpds()
        case "matern":
            model = generate_matern_functional_CPD(G)
            functional_cpd = model.get_cpds()
        case _:
            model = generate_random_functional_CPD(G)
            functional_cpd = model.get_cpds()
    # we now have to convert the markov model to a functional bayesian network
    model = FunctionalBayesianNetwork()
    model.add_nodes_from(M.nodes())
    model.add_edges_from(M.edges())
    model.add_cpds(*functional_cpd)
    model.check_model()
    # finally, sample data from the combined model
    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=1000, seed=42)
    # compute empirical covariance
    covariance = np.cov(data.T) 
    return G, M, data, covariance
    # alternatively, generate data directly from covariance
    # first get graph from model
    G = model.moralize()
    data = generate_covariance_data(G, C=1.0, noise_scale=0.1, num_samples=1000, seed=42)

