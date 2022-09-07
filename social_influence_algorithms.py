import random
from copy import deepcopy
from numba import jit
from operator import itemgetter

import networkx as nx
import numpy as np


def find_linear_directed_acyclic_graph(G: nx.digraph, v: int, theta=1/380):
    DAG = nx.DiGraph()

    influence = {(v, v): 1.0}

    # Select the node which has the most influence on v, and add to DAG. The first
    # selection will be v.
    max_influence = 1.0

    while max_influence > theta:

        most_influential_edge = max(influence, key=influence.get)
        x = most_influential_edge[0]

        # Add the most influential node x to DAG. Then remove edges from nodes in
        # DAG to x, to ensure the resulting graph is acyclic.
        DAG.add_node(x)
        DAG.add_weighted_edges_from(
            [(x, u, G.edges[(x, u)]["weight"]) for u in G.successors(x) if u in DAG.nodes and u != x])

        # Update the influence for nodes preceeding x. Note that this method of
        # updating, and the inclusion of a threshold weight theta ensures that DAG
        # remains connected.
        for u in (u for u in G.predecessors(x) if u not in DAG.nodes):
            if (u, v) not in influence:
                influence[(u, v)] = 0.0
            influence[(u, v)] += G.edges[(u, x)]["weight"] * influence[(x, v)]

        del influence[most_influential_edge]

        try:
            max_influence = influence[max(influence, key=influence.get)]
        except ValueError:
            max_influence = 0.0
    return DAG

# compute the linear coefficient by which activation of nodes in the DAG influence node v
def compute_influence_scalars(D: nx.digraph, v: int, S = []):

    DAG = nx.subgraph(D, list(nx.ancestors(D, v)) + [v])
    assert nx.is_directed_acyclic_graph(
        DAG), "D must be a directed acyclic graph"

    nx.set_node_attributes(D, {u: float(u == v)
                           for u in D.nodes}, "influence_on_v")

    node_sequence = [u for u in nx.topological_sort(DAG)]
    for u in (u for u in reversed(node_sequence) if u not in S and u != v):
        D.nodes[u]["influence_on_v"] = 0.0
        for x in (x for x in D.successors(u) if x in node_sequence):
            D.nodes[u]["influence_on_v"] += D.edges[(u, x)]["weight"] * D.nodes[x]["influence_on_v"]

            
    return D

def seed_selection(G: nx.DiGraph, k: int):
    assert G.number_of_nodes() >= k, "k must be less than or equal to n"

    print("initialization")
    S = []
    G = deepcopy(G)

    nx.set_node_attributes(
        G, {v: 0.0 for v in G.nodes}, "incremental_influence")
    nx.set_node_attributes(G, {v: set([]) for v in G.nodes}, "influence_set")

    # For each node v in D, we compute a local directed acyclic graph
    # containing nodes whch have a significant influence on v using a
    # greedy algorithm.
    LDAG = {v: find_linear_directed_acyclic_graph(G, v) for v in G.nodes}

    for v in G.nodes:
        # For each node u in D, we derive an influence set containing nodes
        # which have been deemed to be significantly influenced by u.
        for u in LDAG[v].nodes:
            G.nodes[u]["influence_set"].add(v)

        # The activation of a node in a directed acyclic graph influences
        # its successors linearly. Since a DAG contains no loops, it follows
        # inductively that a node's influence on decendants is linear.
        #
        # For each local directed acyclic digraph we compute the linear
        # scalars for each node's influence on v.
        LDAG[v] = compute_influence_scalars(LDAG[v], v)
        for u in LDAG[v].nodes:
            LDAG[v].nodes[u]["activation_probability"] = 0.0
            G.nodes[u]["incremental_influence"] += LDAG[v].nodes[u]["influence_on_v"]

    # We select nodes greedily based on incremental influence, an estimate for
    # the net influence of a node if it was activated, discounting the influence of
    # nodes which have already been selected.
    print("main loop")
    for i in range(k):

        s = max(
            [(v, G.nodes[v]["incremental_influence"])
             for v in G.nodes if v not in S],
            key=itemgetter(1)
        )[0]

        # The influence of s must be discounted for future calculations.
        for v in (v for v in G.nodes[s]["influence_set"] if v not in S):
            # Ancestors of an activated node no longer have an influence on that
            # node, influences in each LDAG must be adjusted accordingly.

            nx.set_node_attributes(
                LDAG[v], {u: 0.0 for u in LDAG[v].nodes}, "change_in_influence_on_v")
            LDAG[v].nodes[s]["change_in_influence_on_v"] = -LDAG[v].nodes[s]["influence_on_v"]

            # First, the LDAG is traversed backwards, and linearity is used to backpropogate
            # changes in influence.
            node_sequence = [u for u in nx.topological_sort(
                LDAG[v]) if u in nx.ancestors(LDAG[v], s) or u == s]
            for u in (u for u in reversed(node_sequence) if u not in S and u != s):
                for x in (x for x in LDAG[v].successors(u) if x in node_sequence):

                    change_in_influence = LDAG[v].nodes[x]["change_in_influence_on_v"]
                    change_in_influence *= LDAG[v].edges[(u, x)]["weight"]
                    LDAG[v].nodes[u]["change_in_influence_on_v"] += change_in_influence

            # Next, changes in influence are used to recalculate total influence,
            # and incremental influence.
            for u in node_sequence:

                LDAG[v].nodes[u]["influence_on_v"] += LDAG[v].nodes[u]["change_in_influence_on_v"]

                incremental_influence_on_v = LDAG[v].nodes[u]["change_in_influence_on_v"]
                # The probability that u is activated by other nodes is discounted when
                # calculating influence because it doesn't result from u's activation.
                incremental_influence_on_v *= (1 -
                                               LDAG[v].nodes[u]["activation_probability"])

                G.nodes[u]["incremental_influence"] += incremental_influence_on_v

            nx.set_node_attributes(
                LDAG[v], {u: 0.0 for u in LDAG[v].nodes}, "change_in_activation_probability")
            LDAG[v].nodes[s]["change_in_activation_probability"] = 1.0 - LDAG[v].nodes[s]["activation_probability"]

            # The activation probabilitity of a descendant of an activated node has
            # changed and must be updated, this also affects its influence.

            node_sequence = [u for u in nx.topological_sort(
                LDAG[v]) if u in nx.descendants(LDAG[v], s) or u == s]

            # First, the LDAG is traversed forwards, and linearity is used to propogate
            # changed is activation probability.
            for u in (u for u in node_sequence if u not in S and u != s):
                for x in (x for x in LDAG[v].predecessors(u) if x in node_sequence):

                    change_in_activation_probability = LDAG[v].nodes[x]["change_in_activation_probability"]
                    change_in_activation_probability *= LDAG[v].edges[(
                        x, u)]["weight"]
                    LDAG[v].nodes[u]["change_in_activation_probability"] += change_in_activation_probability

            # Next, changes in activation probability are used to calculate total activation
            # probabilities.
            for u in node_sequence:

                LDAG[v].nodes[u]["activation_probability"] += LDAG[v].nodes[u]["change_in_activation_probability"]

                incremental_influence_on_v = LDAG[v].nodes[u]["influence_on_v"]
                incremental_influence_on_v *= LDAG[v].nodes[u]["change_in_activation_probability"]
                # Since the probability of descendants of s being activated has been updated, the
                # influence gained from activating them must be updated.

                G.nodes[u]["incremental_influence"] -= incremental_influence_on_v

        S.append(s)

    return S

# using a monte-carlo algorithm
def expected_influence(G: nx.graph, S: np.array, num_iterations=10000):
    G = deepcopy(G)

    influence_spread_sum = 0.0
    for i in range(num_iterations):
        # For each vertex, select an activation threshold uniformly at random from [0, 1],
        # which reflects our lack of knowledge of users' true activation thresholds.
        nx.set_node_attributes(G, {v: random.uniform(0.0, 1.0)
                               for v in G.nodes}, "activation_threshold")

        activated_nodes = S
        def activation_state(x): return float(x in activated_nodes)

        # Influence is propogated. The process stops when no additional nodes are activated
        # during a time-step
        nodes_activated_at_t = S
        while nodes_activated_at_t.size != 0:
            # Compute the nodes being activated at the next time step
            nodes_activated_at_t = np.array([])

            non_activated_nodes = (
                v for v in G.nodes if v not in activated_nodes)
            for v in non_activated_nodes:
                influence_on_v = sum(
                    [activation_state(u) * G.edges[(u, v)]["weight"] for u in G.predecessors(v)])
                if influence_on_v > G.nodes[v]["activation_threshold"]:
                    nodes_activated_at_t = np.append(nodes_activated_at_t, v)

            activated_nodes = np.concatenate(
                [activated_nodes, nodes_activated_at_t])

        influence_spread_sum += activated_nodes.size

    return influence_spread_sum / num_iterations


def compute_influence(G: nx.graph, theta=1/380):
    influence = {v: 0.0 for v in G.nodes}

    # For each node v in D, we compute a local directed acyclic graph
    # containing nodes whch have a significant influence on v using a
    # greedy algorithm.
    LDAG = {v: find_linear_directed_acyclic_graph(G, v, theta) for v in G.nodes}

    for v in G.nodes:
        for u in LDAG[v].nodes:

        # The activation of a node in a directed acyclic graph influences
        # its successors linearly. Since a DAG contains no loops, it follows
        # inductively that a node's influence on decendants is linear.
        #
        # For each local directed acyclic digraph we compute the linear
        # scalars for each node's influence on v.
            LDAG[v] = compute_influence_scalars(LDAG[v], v)
            for u in LDAG[v].nodes:
                LDAG[v].nodes[u]["activation_probability"] = 0.0
                influence[u] += LDAG[v].nodes[u]["influence_on_v"]
    return influence