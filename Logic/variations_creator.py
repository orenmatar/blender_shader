from collections import defaultdict
from typing import List
from uuid import uuid4

import networkx as nx
import numpy as np
from dataclasses import dataclass
from enum import Enum
import uuid

from Logic.network_manager import NetworkManager
from Logic.node_readers_writers import ParamRequestType, ParamType


class VariationType(Enum):
    SEED = 1
    NUMERIC = 2
    CAT_AND_NUMERIC = 3
    ADD_NODE = 4
    REMOVE_NODE = 5
    ADD_EDGE = 6
    REMOVE_EDGE = 7


@dataclass
class VariationDescriptor:
    variation_type: VariationType
    step: dict


@dataclass
class TwoWayVariationDescriptor:
    steps_forward: List[VariationDescriptor]
    steps_backward: List[VariationDescriptor]


NODES_TO_SAMPLE_ADD = [
    # "CombineXYZ",
    # "Mapping",
    # "Math",
    "MixFloat",
    "MixVector",
    # "SeparateXYZ",
    "TexGabor",
    "TexGradient",
    "TexNoise",
    "TexVoronoiF",
    "TexWave",
    # "ValToRGB",
    # "VectorMath",
]

# TODO: make a class for edges: node from, node to, input, output, replace many of the steps with this class
# so many area in the code are confusing with "in node" "out node" which are not clear, and the steps sometimes are
# defined where the "in" and "out" are on a separate dictionary, and sometimes they are in the same dictionary.
"""
Intro to this file:
The whole logic here is to create variations to the network. Some variations that are possible are not sampled at the
moment - e.g. adding a new node to connect to a numeric input of a node (ATM we only add node on an edge).
Every function that creates a variation returns a TwoWayVariationDescriptor - which is a forward step and a backward step.
The forward step is the change to the network, and the backward step is the change to revert the network back to the original.
E.g. if the forward step is to add the node, and the backward step is to remove it.
This basically means each variation that is created creates two steps, so more data for the model to learn from.
It also means we can take a network, slowly remove nodes from it until there is nothing, and the backsteps of this process
will be the process to generate the network from scratch.
Sometimes more then one backsteps is needed to revert the network to the original state - e.g. when removing a node
that had several connections - we need to re-add the node and then re-add all the connections to it.
The functions only generate a variation, they do not apply it to the network. The apply_variation function does that.
This means we can save the instructions on a super-network (where each node represents a whole network). The
edges are the operations we need to take between the networks to get from one to the other - they are the variations' steps.
"""

def param_request_type_to_variation_type(param_request_type: ParamRequestType) -> VariationType:
    """
    Convert the ParamRequestType to the VariationType
    """
    if param_request_type == ParamRequestType.SEED:
        return VariationType.SEED
    if param_request_type == ParamRequestType.NUMERIC:
        return VariationType.NUMERIC
    if param_request_type == ParamRequestType.NON_SEED:
        return VariationType.CAT_AND_NUMERIC


def non_structural_changes(
    nm: NetworkManager, max_to_change: int, change_types: ParamRequestType
) -> TwoWayVariationDescriptor:
    """
    Make a change to the network that is not structural - i.e. changing seeds, categorical or numeric values
    Sorry about the mess - it was a long day, it should be easy to simplify the first half of this function
    """
    # get distributions for all the nodes and parameters that can be changed
    dist_vals = nm.get_all_nodes_values(change_types, return_ranges=True, not_input_values=True)
    nm.apply_distribution_limitations(dist_vals)
    # make a list of the nodes and the parameters that can be changed
    # we're making this flat list so we can sample all params equally from all nodes
    change_options = [(node, param_name) for node, values in dist_vals.items() for param_name, _ in values.items()]
    # TODO: can i use dist_vals directly instead of creating a new dict? get_nodes_properties takes a list...
    old_vals = nm.get_nodes_properties(change_options)

    # remove the current category values from options to choose from (while technically the numeric value can be re-sampled...)
    # TODO: maybe i can do this directly on dist_vals instead of creating a new dict
    # result: a dictionary of dictionaries, where the inner dictionary is the parameters that can be changed for each node
    new_dist_vals_forced_changed = defaultdict(dict)
    for node_name, param_name in change_options:
        vals, param_type = dist_vals[node_name][param_name]
        if param_type == ParamType.CATEGORICAL:
            new_vals = tuple([v for v in vals if v != old_vals[node_name][param_name]])
        else:
            new_vals = vals
        if len(new_vals) > 0:
            new_dist_vals_forced_changed[node_name][param_name] = (new_vals, param_type)

    # get the change options, but make sure we only pick params where a real change is possible
    # for numerics make sure that there is actually a distribution allowed after the limitations - it's not (0,0)
    # again making a flat list for easy sampling
    change_options = []
    for node, values in new_dist_vals_forced_changed.items():
        for param_name, (dist, param_type) in values.items():
            if param_type != ParamType.CATEGORICAL:
                if dist[0] != dist[1]:
                    change_options.append((node, param_name))
            else:
                if len(dist) > 0:
                    change_options.append((node, param_name))

    # finally, we have a list of params that can be changed, and we pick a random number of them to change
    n_changes = min(max_to_change, len(change_options))
    to_change_idx = np.random.choice(range(len(change_options)), n_changes, replace=False)
    to_change = [change_options[i] for i in to_change_idx]
    selection_dist = defaultdict(dict)  # a dict of options to select from, from the params we decided to change
    backwards_step = defaultdict(dict)  # the old values to return to on the backward step
    for node, param in to_change:
        selection_dist[node][param] = new_dist_vals_forced_changed[node][param]
        backwards_step[node][param] = old_vals[node][param]

    new_vals = nm.pick_random_values_from_dict(selection_dist)
    variation_type = param_request_type_to_variation_type(change_types)
    forward = VariationDescriptor(variation_type, new_vals)
    backward = VariationDescriptor(variation_type, backwards_step)
    return TwoWayVariationDescriptor([forward], [backward])


def add_random_node_on_edge(nm: NetworkManager) -> TwoWayVariationDescriptor:
    """
    Add a random node on a random edge in the network
    """
    edges = list(nm.network.edges(data=True))
    edge = edges[np.random.randint(0, len(edges))]  # pick some edge
    new_node_type_name = np.random.choice(NODES_TO_SAMPLE_ADD)  # pick a random node type
    node_type = nm.NODE_TYPES[new_node_type_name]
    new_node_out = node_type.get_random_output()
    new_node_in = node_type.get_random_input()
    new_node_name = new_node_type_name + f"_{str(uuid4())}" # create a unique name for the new node

    step = {
        "edge": edge,
        "new_node_type": new_node_type_name,
        "new_node_in": new_node_in,
        "new_node_out": new_node_out,
        "new_node_name": new_node_name,
    }
    forward = VariationDescriptor(VariationType.ADD_NODE, step)

    step = {
        "remove_node_name": new_node_name,
        "replacement_edge": edge,
    }
    backward = VariationDescriptor(VariationType.REMOVE_NODE, step)
    return TwoWayVariationDescriptor([forward], [backward])


def add_random_edge(nm: NetworkManager) -> TwoWayVariationDescriptor:
    """
    Add a random edge to the network.
    The edge can be between any two nodes, even if they are connected (so it will do nothing)
    """
    # choose from all nodes that have an input to them - not value node
    # TODO: probably best to make this not hard coded, have a method for nodes with no inputs
    nodes_to_sample = [node for node in nm.network.nodes() if node != "InputNode_1" and "Value" not in node]
    in_node = np.random.choice(nodes_to_sample)  # select a node to connect to
    in_node_type = nm.node_name_to_node_type(in_node)
    inputs = list(in_node_type.get_inputs())
    input_to_connect = np.random.choice(inputs)  # select a random input to connect to, even if already connected
    descendants = nx.descendants(nm.network, in_node)
    # do not connect to self, or to descendants (that creates a cycle)
    non_legit_nodes = descendants | {in_node, "OutputNode_1"}
    possible_connections = [node for node in nm.network.nodes() if node not in non_legit_nodes]
    out_node = np.random.choice(possible_connections)
    out_node_type = nm.node_name_to_node_type(out_node)
    outputs = list(out_node_type.get_outputs())
    out_to_connect = np.random.choice(outputs)

    step = {"in_node": in_node, "out_node": out_node, "in": input_to_connect, "out": out_to_connect}

    forward = VariationDescriptor(VariationType.ADD_EDGE, step)

    # check if there was already a connection to that node's input
    connected_edge = [edge for edge in nm.network.in_edges(in_node, data=True) if edge[2]["in"] == input_to_connect]
    if len(connected_edge) > 0:
        assert len(connected_edge) == 1, "This really should not happen"  # can only be 0 or 1
        connected_edge = connected_edge[0]
        step = {
            "in_node": connected_edge[1],
            "out_node": connected_edge[0],
            "in": connected_edge[2]["in"],
            "out": connected_edge[2]["out"],
        }
        # if there was an edge there already the backwards step is to add it back
        backwards = VariationDescriptor(VariationType.ADD_EDGE, step)
    else:
        # if there was no edge there before, the backwards step is to remove the edge we just added
        step = {"in_node": in_node, "out_node": out_node, "in": input_to_connect, "out": out_to_connect}
        backwards = VariationDescriptor(VariationType.REMOVE_EDGE, step)
    return TwoWayVariationDescriptor([forward], [backwards])


def remove_random_edge(nm: NetworkManager) -> TwoWayVariationDescriptor:
    """
    Remove a random edge from the network
    """
    edges = list(nm.network.edges(data=True))
    selected_edge = edges[np.random.randint(0, len(edges))]
    step = {
        "out_node": selected_edge[0],
        "in_node": selected_edge[1],
        "in": selected_edge[2]["in"],
        "out": selected_edge[2]["out"],
    }
    forward = VariationDescriptor(VariationType.REMOVE_EDGE, step)
    backwards = VariationDescriptor(VariationType.ADD_EDGE, step)
    return TwoWayVariationDescriptor([forward], [backwards])


def remove_random_node(nm: NetworkManager) -> TwoWayVariationDescriptor:
    """
    Remove a random node from the network
    """
    nodes = [node for node in nm.network.nodes() if node not in ["InputNode_1", "OutputNode_1"]]
    node_to_remove = np.random.choice(nodes)
    return create_remove_node_variation(nm, node_to_remove)

def create_remove_node_variation(nm: NetworkManager, node_to_remove: str) -> TwoWayVariationDescriptor:
    """
    Creates the variation descriptor for removing a node.
    If we remove a node that is in the middle of two other nodes, we need to connect them instead. If it has several
    inputs or outputs - select the first input and connect to the first output.
    Additionally for the backward step, we need to create steps for every other edge that was connected to this node.
    And finally for the backward step, we need to create a step to change its attributes back to what they were
    (numeric and categorical values).
    """
    node_type = nm.node_name_to_node_type(node_to_remove)
    all_in_edges = nm.network.in_edges(node_to_remove, data=True)
    all_out_edges = nm.network.out_edges(node_to_remove, data=True)
    in_order = {name: i for i, name in enumerate(node_type.get_inputs_names_list())}
    out_order = {name: i for i, name in enumerate(node_type.get_outputs())}
    # order the in and out edges by the order given by the node definition. so we can easily take the first input
    all_in_edges = sorted(all_in_edges, key=lambda x: in_order[x[2]["in"]])
    all_out_edges = sorted(all_out_edges, key=lambda x: out_order[x[2]["out"]])

    # get the node attributes that are not the default values (only the ones we need to change when we add the node back)
    node_attributes = nm.get_node_non_default_vals(node_to_remove)
    forward_step = {"remove_node_name": node_to_remove}
    backward_step = {
        "new_node_type": nm.node_name_to_node_type_name(node_to_remove),
        "new_node_name": node_to_remove,
    }

    # if this node is in the middle of two other nodes - if it has in and out - connect them instead
    if len(all_in_edges) > 0 and len(all_out_edges) > 0:
        # take the first input and first output and remember to connect them
        first_in = all_in_edges[0]
        first_out = all_out_edges[0]
        # the new edge will be from the input of the in edge to the output of the out edge
        replacement_edge = (first_in[0], first_out[1], {"in": first_out[2]["in"], "out": first_in[2]["out"]})
        forward_step["replacement_edge"] = replacement_edge

        # in the backwards step - adding a node - this is the edge we add the node on
        backward_step["edge"] = replacement_edge
        # define the in and out into the new node
        backward_step["new_node_in"] = first_in[2]["in"]
        backward_step["new_node_out"] = first_out[2]["out"]
        # now remove those edges, so we are only left with other edges, that the backward step's node is not directly
        # placed on - edges that are not part of the backstep
        all_in_edges.remove(first_in)
        all_out_edges.remove(first_out)

    forward = VariationDescriptor(VariationType.REMOVE_NODE, forward_step)
    backwards_steps = [VariationDescriptor(VariationType.ADD_NODE, backward_step)]
    # next backstep is to change the node params back to what they were
    if len(node_attributes) > 0:
        backwards_steps.append(VariationDescriptor(VariationType.CAT_AND_NUMERIC, {node_to_remove: node_attributes}))

    # next backsteps - adding edges for each connection that was removed
    for edge in all_in_edges + all_out_edges:
        # we don't know what the node will actually be called. this can only be determined when running the whole sequence of steps together
        # when we first run the addition of the node, we get the actual name, and then replace the name here before applying this step
        step = {
            "in_node": edge[1],
            "out_node": edge[0],
            "in": edge[2]["in"],
            "out": edge[2]["out"],
        }
        backwards_steps.append(VariationDescriptor(VariationType.ADD_EDGE, step))

    return TwoWayVariationDescriptor([forward], backwards_steps)


def apply_variation(nm: NetworkManager, variation: VariationDescriptor):
    """
    Apply the variation to the network
    """
    step = variation.step
    if variation.variation_type in [VariationType.SEED, VariationType.NUMERIC, VariationType.CAT_AND_NUMERIC]:
        nm.set_nodes_attributes(step)

    elif variation.variation_type == VariationType.ADD_NODE:
        new_node_type = step["new_node_type"]
        new_node_name = step["new_node_name"]
        nm.add_node_by_type_and_name(new_node_type, new_node_name)
        # if the node is in the middle of two other nodes - disconnect them, and connect them to the new node
        if "edge" in step:
            node_before, node_after, edge_in_out = step["edge"]
            nm.remove_edge(node_before, node_after, edge_in_out["in"])
            nm.add_edge(node_before, new_node_name, edge_in_out["out"], variation.step["new_node_in"])
            nm.add_edge(new_node_name, node_after, step["new_node_out"], edge_in_out["in"])

    elif variation.variation_type == VariationType.REMOVE_NODE:
        node_to_remove = step["remove_node_name"]
        nm.remove_node(node_to_remove)
        # if there is a replacement edge - add it back (if the node was in the middle of two other nodes)
        if "replacement_edge" in step:
            replacement_edge = step["replacement_edge"]
            # from the output of what used to go into this node, to the input of what used to go out of it
            nm.add_edge(replacement_edge[0], replacement_edge[1], replacement_edge[2]["out"], replacement_edge[2]["in"])

    elif variation.variation_type == VariationType.ADD_EDGE:
        in_node, out_node, input_to_connect, out_to_connect = step["in_node"], step["out_node"], step["in"], step["out"]
        connected_edge = [edge for edge in nm.network.in_edges(in_node, data=True) if edge[2]["in"] == input_to_connect]
        if len(connected_edge) > 0:
            assert len(connected_edge) == 1, "This really should not happen"
            old_out_node, old_in_node, old_in_out_names = connected_edge[0]
            # if there is a connection - remove it first
            nm.remove_edge(old_out_node, old_in_node, old_in_out_names["in"])
        nm.add_edge(out_node, in_node, out_to_connect, input_to_connect)

    elif variation.variation_type == VariationType.REMOVE_EDGE:
        in_node, out_node, input_to_connect, out_to_connect = step["in_node"], step["out_node"], step["in"], step["out"]
        nm.remove_edge(out_node, in_node, input_to_connect)

    # re calc layers anc connect vector inputs that must be connected, if something happened to them
    nm.finish_network()
