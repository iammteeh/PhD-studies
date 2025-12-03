# adapter module for ggmpy
from pgmpy.models import DiscreteBayesianNetwork, MarkovNetwork, DynamicBayesianNetwork
from pgmpy.inference import BayesianModelSampling

def get_property(model, property_name):
    """Get a property from the DiscreteBayesianNetwork model."""
    if property_name == "cpds":
        return model.get_cpds()
    elif property_name == "nodes":
        return model.nodes()
    elif property_name == "edges":
        return model.edges()
    elif property_name == "moral_graph":
        return model.moralize()
    elif property_name == "skeleton":
        return model.get_skeleton()
    elif property_name == "independencies":
        return model.get_independencies()
    elif property_name == "is_dag":
        return model.is_dag()
    elif property_name == "is_complete":
        return model.is_complete()
    elif property_name == "is_fully_connected":
        return model.is_fully_connected()
    elif property_name == "is_bayesian":
        return model.is_bayesian()
    elif property_name == "blanket":
        return {node: model.get_markov_blanket(node) for node in model.nodes()}
    elif property_name == "parents":
        return {node: model.get_parents(node) for node in model.nodes()}
    elif property_name == "children":
        return {node: model.get_children(node) for node in model.nodes()}
    elif property_name == "roots":
        return model.get_roots()
    elif property_name == "leaves":
        return model.get_leaves()
    elif property_name == "neighbors":
        return {node: model.get_neighbors(node) for node in model.nodes()}
    elif property_name == "paths":
        return {node: {target: list(model.get_all_paths(node, target)) for target in model.nodes() if target != node} for node in model.nodes()}
    elif property_name == "descendants":
        return {node: model.get_descendants(node) for node in model.nodes()}
    elif property_name == "ancestors":
        return {node: model.get_ancestors(node) for node in model.nodes()}
    elif property_name == "topological_sort":
        return list(model.topological_sort())
    elif property_name == "marginals":
        sampler = BayesianModelSampling(model)
        data = sampler.forward_sample(size=1000, seed=42)
        marginals = {}
        for node in model.nodes():
            marginals[node] = data[node].value_counts(normalize=True).to_dict()
        return marginals
    elif property_name == "random":
        return model.get_random()
    else:
        raise ValueError(f"Unknown property: {property_name}")
    
def to(model, to_type):
    """Convert the DiscreteBayesianNetwork model to another type."""
    match to_type:
        case "markov":
            return model.to_markov_model()
        case "factor_graph":
            return model.to_factor_graph()
        case _:
            raise ValueError(f"Unknown conversion type: {to_type}")