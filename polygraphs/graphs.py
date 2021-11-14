"""
Graphs
"""
import sys
import inspect
import math
import networkx as nx
import dgl
import numpy as np

from .hyperparameters import HyperParameters
from . import datasets


def _isconnected(graph):
    return nx.is_strongly_connected(dgl.to_networkx(graph))


def _buckleup(graph, exist_ok=False):
    """
    Adds self loops to given graph.
    """
    if not exist_ok:
        # The number of edges in the given graph
        count = len(graph.edges())
        # Remove all self-loops in the graph, if present
        graph = dgl.remove_self_loop(graph)
        # Assert |E'| = |E|
        assert len(graph.edges()) == count
    # Add self-loops for each node in the graph and return a new graph
    return dgl.transform.add_self_loop(graph)


def sample_(selfloop=True):
    """
    Returns a sample 6-node graph.
    """
    src = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5])
    dst = np.array([2, 5, 2, 3, 4, 5, 0, 1, 3, 5, 1, 2, 4, 5, 1, 3, 0, 1, 2, 3])

    graph = dgl.graph((src, dst))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def sample(params):
    """
    Returns a sample 6-node graph from hyper-parameters.
    """
    # Update network size
    params.size = 6
    return sample_(selfloop=params.selfloop)


def wheel_(size, selfloop=True):
    """
    Returns an undirected wheel graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx
    graph = dgl.from_networkx(nx.wheel_graph(size))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def wheel(params):
    """
    Returns an undirected wheel graph from hyper-parameters.
    """
    return wheel_(params.size, selfloop=params.selfloop)


def cycle_(size, directed=False, selfloop=True):
    """
    Returns a cycle graph.
    """
    # Check network size
    assert size > 1
    # Get networkx constructor
    constructor = nx.DiGraph if directed else nx.Graph
    # Get graph from networkx
    graph = dgl.from_networkx(nx.cycle_graph(size, create_using=constructor))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def cycle(params):
    """
    Returns a cycle graph from hyper-parameters.
    """
    return cycle_(params.size, directed=params.directed, selfloop=params.selfloop)


def star_(size, selfloop=True):
    """
    Returns an undirected star graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx. The graph has n + 1 nodes for integer n, so substract 1
    graph = dgl.from_networkx(nx.star_graph(size - 1))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def star(params):
    """
    Returns an undirected star graph.
    """
    return star_(params.size, selfloop=params.selfloop)


def line_(size, directed=False, selfloop=True):
    """
    Return a line graph.
    """
    # Check network size
    assert size > 1
    # Get networkx constructor
    constructor = nx.DiGraph if directed else nx.Graph
    # Get graph from networkx
    graph = dgl.from_networkx(nx.path_graph(size, create_using=constructor))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def line(params):
    """
    Return a line graph from hyper-parameters.
    """
    return line_(params.size, directed=params.directed, selfloop=params.selfloop)


def grid_(size, selfloop=True):
    """
    Returns a 2-D square grid graph.
    """
    # Check network size
    assert size > 1
    # Check network size is a perfect square (approximate solution)
    assert size == math.pow(int(math.sqrt(size) + 0.5), 2)
    rows = columns = int(math.sqrt(size))
    # Get graph from networkx
    graph = dgl.from_networkx(nx.grid_2d_graph(rows, columns))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def grid(params):
    """
    Returns a 2-D square grid graph from hyper-parameters.
    """
    return grid_(params.size, selfloop=params.selfloop)


def random_(size, probability, seed=None, directed=False, selfloop=True):
    """
    Returns an Erdos-Renyi graph.
    """
    # Check network size
    assert size > 1
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    # Get graph from networkx
    graph = dgl.from_networkx(
        nx.erdos_renyi_graph(size, probability, seed=seed, directed=directed)
    )
    if not _isconnected(graph):
        # Probability p should be greater than ((1 + e)ln(size))/size
        _p = math.log(size) / size
        msg = f"Graph G({size, probability}) is disconnected. Try p > {_p}"
        raise Exception(msg)
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def random(params):
    """
    Returns an Erdos-Renyi graph from hyper-parameters
    """
    return random_(
        params.size,
        params.random.probability,
        seed=params.random.seed,
        directed=params.directed,
        selfloop=params.selfloop,
    )


def complete_(size, selfloop=True):
    """
    Returns an undirected fully-connected graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx
    graph = dgl.from_networkx(nx.complete_graph(size))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def complete(params):
    """
    Returns an undirected fully-connected graph from hyper-parameters.
    """
    return complete_(params.size, selfloop=params.selfloop)


def karate_(selfloop=True):
    """
    Returns Zachary's Karate club social network.
    """
    # Get graph from networkx
    graph = dgl.from_networkx(nx.karate_club_graph())
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def karate(params):
    """
    Returns Zachary's Karate club social network from hyper-parameters.
    """
    # Update network size
    params.size = 34
    return karate_(selfloop=params.selfloop)


def wattsstrogatz_(
    size, knn, probability, tries=100, seed=None, selfloop=True
):  # pylint: disable=too-many-arguments
    """
    Returns a connected Watts–Strogatz small-world graph.
    """
    # Check network size
    assert size > 1
    # Check neighbourhood size
    assert knn > 1
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    # Get graph from networkx
    graph = dgl.from_networkx(
        nx.connected_watts_strogatz_graph(
            size, knn, probability, tries=tries, seed=seed
        )
    )
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def wattsstrogatz(params):
    """
    Returns a connected Watts–Strogatz small-world graph from hyper-parameters.
    """
    return wattsstrogatz_(
        params.size,
        params.wattsstrogatz.knn,
        params.wattsstrogatz.probability,
        tries=params.wattsstrogatz.tries,
        seed=params.wattsstrogatz.seed,
        selfloop=params.selfloop,
    )


def barabasialbert_(size, attachments, seed=None, selfloop=True):
    """
    Returns a random graph according to the Barabasi–Albert preferential attachment model.
    """
    # Check network size
    assert size > 1
    # Check neighbourhood size
    assert attachments > 0
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    # Get graph from networkx
    graph = dgl.from_networkx(nx.barabasi_albert_graph(size, attachments, seed=seed))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def barabasialbert(params):
    """
    Returns a random graph according to the Barabasi–Albert
    preferential attachment model from hyper-parameters.
    """
    return barabasialbert_(
        params.size,
        params.barabasialbert.attachments,
        seed=params.barabasialbert.seed,
        selfloop=params.selfloop,
    )


def snap(params):
    """
    Returns a SNAP dataset, identified by `params.snap.name`.
    """
    graph = datasets.snap.getbyname(params.snap.name).read()
    # Update network size
    params.size = graph.num_nodes()
    return graph


def ogb(params):
    """
    Returns an OGB dataset, identified by `params.ogb.name`.
    Current only one dataset is supported, ogbl-collab.
    """
    assert params.ogb.name and isinstance(params.ogb.name, str)
    assert params.ogb.name.lower == "collab"
    dataset = datasets.ogb.Collab()
    graph = dataset.read()
    # Update network size
    params.size = graph.num_nodes()
    return graph


def create(params):
    """
    Returns a GDL graph of given type and size.
    """
    assert isinstance(params, HyperParameters)
    # Create friendly dictionary from list of (name, function) tuples
    members = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
    constructor = members.get(params.kind)
    if constructor is None:
        raise Exception(f"Invalid graph type: {params.kind}")
    # Construct DGL graph
    return constructor(params=params)
