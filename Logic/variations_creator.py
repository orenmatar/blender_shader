from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
from dataclasses import dataclass
from enum import Enum

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


def param_request_type_to_variation_type(param_request_type: ParamRequestType) -> VariationType:
    if param_request_type == ParamRequestType.SEED:
        return VariationType.SEED
    if param_request_type == ParamRequestType.NUMERIC:
        return VariationType.NUMERIC
    if param_request_type == ParamRequestType.NON_SEED:
        return VariationType.CAT_AND_NUMERIC


def non_structural_changes(
    nm: NetworkManager, max_to_change: int, change_types: ParamRequestType
) -> TwoWayVariationDescriptor:
    dist_vals = nm.get_all_nodes_values(change_types, return_ranges=True, not_input_values=True)
    nm.apply_distribution_limitations(dist_vals)

    # get the change options, but make sure we only pick params where a real change is possible
    # for numerics make sure that there is actually a distribution allowed after the limitations - it's not (0,0)
    change_options = []
    for node, values in dist_vals.items():
        for param_name, (dist, param_type) in values.items():
            if param_type != ParamType.CATEGORICAL:
                if dist[0] != dist[1]:
                    change_options.append((node, param_name))
            else:
                change_options.append((node, param_name))

    n_changes = min(max_to_change, len(change_options))
    to_change_idx = np.random.choice(range(len(change_options)), n_changes, replace=False)
    to_change = [change_options[i] for i in to_change_idx]

    old_vals = nm.get_nodes_properties(to_change)

    # remove the current category values from options to choose from (while technically the numeric value can be re-sampled...)
    new_dist_vals_forced_changed = defaultdict(dict)
    for node_name, param_name in to_change:
        vals, param_type = dist_vals[node_name][param_name]
        if param_type == ParamType.CATEGORICAL:
            new_vals = tuple([v for v in vals if v != old_vals[node_name][param_name]])
        else:
            new_vals = vals
        new_dist_vals_forced_changed[node_name][param_name] = (new_vals, param_type)

    new_vals = nm.pick_random_values_from_dict(new_dist_vals_forced_changed)
    variation_type = param_request_type_to_variation_type(change_types)
    forward = VariationDescriptor(variation_type, new_vals)
    backward = VariationDescriptor(variation_type, old_vals)
    return TwoWayVariationDescriptor([forward], [backward])


def add_random_node_on_edge(nm: NetworkManager) -> TwoWayVariationDescriptor:
    edges = list(nm.network.edges(data=True))
    edge = edges[np.random.randint(0, len(edges))]
    new_node_type_name = np.random.choice(NODES_TO_SAMPLE_ADD)
    node_type = nm.NODE_TYPES[new_node_type_name]
    new_node_out = node_type.get_random_output()
    new_node_in = node_type.get_random_input()

    step = {"edge": edge, "new_node": new_node_type_name, "new_node_in": new_node_in, "new_node_out": new_node_out}
    forward = VariationDescriptor(VariationType.ADD_NODE, step)

    step = {
        "remove_node_type": new_node_type_name,
        "new_edge": edge,
    }
    backward = VariationDescriptor(VariationType.REMOVE_NODE, step)
    return TwoWayVariationDescriptor([forward], [backward])


def add_random_edge(nm: NetworkManager) -> TwoWayVariationDescriptor:
    # all nodes that have an input to them - not value node
    nodes = [node for node in nm.network.nodes() if node != "InputNode_1" and "Value" not in node]
    in_node = np.random.choice(nodes)
    in_node_type = nm.node_name_to_node_type(in_node)
    inputs = list(in_node_type.get_inputs())
    input_to_connect = np.random.choice(inputs)
    descendants = nx.descendants(nm.network, in_node)
    possible_connections = [node for node in nodes if node not in descendants and node != in_node]
    out_node = np.random.choice(possible_connections)
    out_node_type = nm.node_name_to_node_type(out_node)
    outputs = list(out_node_type.get_outputs())
    out_to_connect = np.random.choice(outputs)

    step = {"in_node": in_node, "out_node": out_node, "in": input_to_connect, "out": out_to_connect}

    forward = VariationDescriptor(VariationType.ADD_EDGE, step)

    connected_edge = [edge for edge in nm.network.in_edges(in_node, data=True) if edge[2]["in"] == input_to_connect]
    if len(connected_edge) > 0:
        assert len(connected_edge) == 1, "This really should not happen"
        connected_edge = connected_edge[0]
        step = {
            "in_node": connected_edge[1],
            "out_node": connected_edge[0],
            "in": connected_edge[2]["in"],
            "out": connected_edge[2]["out"],
        }
        backwards = VariationDescriptor(VariationType.ADD_EDGE, step)
    else:
        step = {"in_node": in_node, "out_node": out_node, "in": input_to_connect, "out": out_to_connect}
        backwards = VariationDescriptor(VariationType.REMOVE_EDGE, step)
    return TwoWayVariationDescriptor([forward], [backwards])


def remove_random_edge(nm: NetworkManager) -> TwoWayVariationDescriptor:
    edges = list(nm.network.edges(data=True))
    selected_edge = edges[np.random.randint(0, len(edges))]
    step = {
        "out_node": selected_edge[1],
        "in_node": selected_edge[0],
        "in": selected_edge[2]["in"],
        "out": selected_edge[2]["out"],
    }
    forward = VariationDescriptor(VariationType.REMOVE_EDGE, step)
    backwards = VariationDescriptor(VariationType.ADD_EDGE, step)
    return TwoWayVariationDescriptor([forward], [backwards])


def remove_random_node(nm: NetworkManager) -> TwoWayVariationDescriptor:
    nodes = [node for node in nm.network.nodes() if node not in ["InputNode_1", "OutputNode_1"]]
    node_to_remove = np.random.choice(nodes)
    node_type = nm.node_name_to_node_type(node_to_remove)
    all_in_edges = nm.network.in_edges(node_to_remove, data=True)
    all_out_edges = nm.network.out_edges(node_to_remove, data=True)
    in_order = {name: i for i, name in enumerate(node_type.get_inputs_names_list())}
    out_order = {name: i for i, name in enumerate(node_type.get_outputs())}
    # order the in and out edges by the order given by the node definition. so we can easily take the first input
    all_in_edges = sorted(all_in_edges, key=lambda x: in_order[x[2]['in']])
    all_out_edges = sorted(all_out_edges, key=lambda x: out_order[x[2]['out']])

    forward_step = {
        'node': node_to_remove
    }
    backward_step = {
        'new_node': nm.node_name_to_node_type_name(node_to_remove),
    }

    # if this node is in the middle of two other nodes - if it has in and out - connect them instead
    if len(all_in_edges) > 0 and len(all_out_edges) > 0:
        # take the first input and first output and remember to connect them
        first_in = all_in_edges[0]
        first_out = all_out_edges[0]
        forward_step['replacement_edge'] = first_in, first_out

        # backwards - adding a new node in the middle of an edge
        # the edge to add the node on is from out of the input to the new node to the in of the output of the new node
        edge_to_add_on = (first_in[0], first_out[0], {'in': first_out[2]['in'], 'out': first_in[2]['out']})
        backward_step['edge'] = edge_to_add_on
        # define the in and out into the new node
        backward_step["new_node_in"] = first_in[2]['in']
        backward_step["new_node_out"] = first_out[2]['out']
        # now remove those edges, so we are only left with other edges, that the backward step's node is not directly placed on, to add
        # i.e. other edges to this node but that are not part of the add node process of the backstep
        all_in_edges.remove(first_in)
        all_out_edges.remove(first_out)

    forward = VariationDescriptor(VariationType.REMOVE_NODE, forward_step)
    backwards_steps = [VariationDescriptor(VariationType.ADD_NODE, backward_step)]

    # all extra edges that were connected to this node
    for edge in all_in_edges + all_out_edges:
        # we don't know what the node will actually be called. this can only be determined when running the whole sequence of steps together
        # when we first run the addition of the node, we get the actual name, and then replace the name here before applying this step
        step = {
            'in_node': edge[1].replace(node_to_remove, 'NODE_NAME_MISSING'),
            'out_node': edge[0].replace(node_to_remove, 'NODE_NAME_MISSING'),
            'in': edge[2]['in'],
            'out': edge[2]['out'],
        }
        backwards_steps.append(VariationDescriptor(VariationType.ADD_EDGE, step))

    return TwoWayVariationDescriptor([forward], backwards_steps)


def apply_variation(nm: NetworkManager, variation: VariationDescriptor):
    step = variation.step
    if variation.variation_type in [VariationType.SEED, VariationType.NUMERIC, VariationType.CAT_AND_NUMERIC]:
        nm.set_nodes_attributes(step)

    elif variation.variation_type == VariationType.ADD_NODE:
        new_node_name = nm.add_node_by_type_name(step["new_node"])
        if "edge" in step:
            node_before, node_after, edge_in_out = step["edge"]
            nm.remove_edge(node_before, node_after, edge_in_out["in"])
            nm.add_edge(node_before, new_node_name, edge_in_out["out"], variation.step["new_node_in"])
            nm.add_edge(new_node_name, node_after, step["new_node_out"], edge_in_out["in"])
        # for the case of adding a new node we need to return the node name so it may be used if it is part of a sequence
        # of steps, and the name is needed for the next step - like adding an edge to it
        nm.finish_network()
        return new_node_name

    elif variation.variation_type == VariationType.REMOVE_NODE:
        # this means it's a variations created as a reverse of adding node, and we don't know the node actual name, just type
        if "remove_node_type" in step:
            out_node, in_node, out_in_names = step["new_edge"]
            # finding the node that precedes it, going to the known input, verifying it's the right type
            edge_to_node = [x for x in nm.network.in_edges(in_node, data=True) if x[2]["in"] == out_in_names["in"]]
            assert len(edge_to_node) == 1
            edge_to_node = edge_to_node[0]
            node_to_remove = edge_to_node[0]
            assert nm.node_name_to_node_type_name(node_to_remove) == variation.step["remove_node_type"]
            nm.remove_node(node_to_remove)
            nm.add_edge(out_node, in_node, out_in_names["out"], out_in_names["in"])
        else:  #if we know the node name
            node_to_remove = step["node"]
            nm.remove_node(node_to_remove)
            if "replacement_edge" in step:
                first_in, first_out = step["replacement_edge"]
                # from the output of what used to go into this node, to the input of what used to go out of it
                nm.add_edge(first_in[0], first_out[1], first_in[2]['out'], first_out[2]['in'])

    # TODO: this whole step structure is very confusing, and should be refactored. in the step i have directly keys of
    # 'in' and 'out' whereas the edge on the network has three elements, the third is the dictionary with the keys.
    # ideally: define a class for an edge and then read it directly.
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

    nm.finish_network()
