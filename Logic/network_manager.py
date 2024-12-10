from collections import defaultdict
from typing import Tuple

from Logic.node_readers_writers import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NetworkManager(object):
    NODE_TYPES = {
        "CombineXYZ": CombineXYZ,
        "Mapping": Mapping,
        "Math": Math,
        "MixFloat": MixFloat,
        "MixVector": MixVector,
        "SeparateXYZ": SeparateXYZ,
        "TexGabor": TexGabor,
        "TexGradient": TexGradient,
        "TexNoise": TexNoise,
        "TexVoronoiF": TexVoronoiF,
        "TexWave": TexWave,
        "ValToRGB": ValToRGB,
        "Value": Value,
        "VectorMath": VectorMath,
        "OutputNode": OutputNode,
        "InputNode": InputNode,
    }
    NODE_TYPES_FOR_GENERATION = [node_name for node_name in NODE_TYPES if node_name not in ["OutputNode", "InputNode"]]

    def __init__(self):
        self.network = nx.MultiDiGraph()
        self.node_counts = defaultdict(int)
        self.free_inputs = {}
        self.must_connect_to_inputs = {}
        self.names_to_types = {}
        for node_name, node_type in self.NODE_TYPES.items():
            node_vector_inputs = {
                input_name
                for input_name, input_type in node_type.get_inputs().items()
                if input_type.param_type == ParamType.VECTOR_INPUT
            }
            self.must_connect_to_inputs[node_name] = node_vector_inputs

    def initialize_network(self):
        self.in_node_name, self.out_node_name = "InputNode", "OutputNode"
        self.add_node(OutputNode, self.out_node_name, {"layer": 1})
        in_properties = InputNode.get_node_type_params(ParamRequestType.ALL, default_values=True)
        in_properties["layer"] = 1  # input layer incase nothing is connected to it
        self.add_node(InputNode, self.in_node_name, in_properties)
        self.in_node_output_name = list(InputNode.get_outputs())[0]

    def get_random_param_values(self, param_type: ParamRequestType):
        dist_vals = self.get_all_nodes_values(param_type, return_ranges=True, not_input_values=True)
        return self.pick_random_value_from_dict(dist_vals)

    def set_nodes_attributes(self, update_attributes: dict):
        for node, attributes in update_attributes.items():
            for attr_name, value in attributes.items():
                self.network.nodes[node][attr_name] = value

    def generate_random_network(self, n_additions=5):
        # only after initialization (add assert)
        for i in range(n_additions):
            free_inputs = [node_name for node_name, inputs in self.free_inputs.items() if len(inputs) > 0]
            if len(free_inputs) == 0:
                break
            random_node = np.random.choice(free_inputs)
            random_in = np.random.choice(list(self.free_inputs[random_node]))
            new_node_type_name = np.random.choice(self.NODE_TYPES_FOR_GENERATION)
            out = self.NODE_TYPES[new_node_type_name].get_random_output()
            new_node_name = self.add_node_by_type(new_node_type_name)
            self.add_edge(new_node_name, random_node, out, random_in)

        for node_name, free_inputs in self.free_inputs.items():
            for free_input in list(free_inputs):
                if free_input in self.must_connect_to_inputs[self.node_name_to_node_type_name(node_name)]:
                    self.add_edge(
                        self.in_node_name,
                        node_name,
                        self.in_node_output_name,
                        free_input,
                    )

    def add_node_by_type(self, node_type_name):
        self.node_counts[node_type_name] += 1
        node_name = node_type_name + f"_{self.node_counts[node_type_name]}"
        node_type: Node = self.NODE_TYPES[node_type_name]
        properties = node_type.get_node_type_params(ParamRequestType.ALL, default_values=True)
        self.add_node(node_type, node_name, properties)
        return node_name

    def get_node_connected_inputs(self, node):
        in_edges = self.network.in_edges(node, data=True)
        return [edge_properties["in"] for previous_node, self_node, edge_properties in in_edges]

    def get_all_nodes_values(self, param_type: ParamRequestType, return_ranges=False, not_input_values=True):
        nodes = self.network.nodes(data=True)
        all_values = {}
        for node, values in nodes:
            if not_input_values:
                input_taken = self.get_node_connected_inputs(node)
                values = {k: v for k, v in values.items() if k not in input_taken}
            node_type_name = self.node_name_to_node_type_name(node)
            node_type = self.NODE_TYPES[node_type_name]
            properties = node_type.get_node_type_params(param_type, default_values=False)
            if return_ranges:
                node_vals = {k: properties[k] for k in values if k in properties}
            else:
                node_vals = {k: v for k, v in values.items() if k in properties}
            if len(node_vals) > 0:
                all_values[node] = node_vals
        return all_values

    @staticmethod
    def pick_random_value_from_dict(nodes_and_dist: Dict[str, Dict[str, Tuple[tuple, ParamType]]]):
        result = {}
        for node, values in nodes_and_dist.items():
            node_rand_vals = {}
            for attr_name, (dist, attr_type) in values.items():
                if attr_type in [ParamType.FLOAT, ParamType.SEED]:
                    value = np.random.uniform(low=dist[0], high=dist[1])
                elif attr_type == ParamType.VECTOR:
                    value = tuple(np.random.uniform(low=dist[0], high=dist[1], size=3))
                elif attr_type == ParamType.CATEGORICAL:
                    value = np.random.choice(dist)
                else:
                    raise ValueError
                node_rand_vals[attr_name] = value
            result[node] = node_rand_vals
        return result

    def add_node(self, node_type, node_name, properties):
        self.network.add_node(node_name, **properties)
        free_inputs = node_type.get_node_type_free_inputs()
        self.free_inputs[node_name] = free_inputs
        self.names_to_types[node_name] = node_type

    def remove_node(self, node1):
        # TODO: make sure to remove inputs from free input dict, must_connect_to_inputs...
        raise NotImplementedError

    def add_edge(self, node1, node2, out1, in2):
        assert in2 in self.free_inputs[node2], "Input must be free"
        self.network.add_edge(node1, node2, **{"out": out1, "in": in2}, key=in2)
        self.network.nodes[node1]["layer"] = self.network.nodes[node2]["layer"] + 1
        self.free_inputs[node2].remove(in2)

    def remove_edge(self, node1, node2, in2):
        self.network.remove_edge(node1, node2, key=in2)
        self.free_inputs[node2].add(in2)

    def to_node_instance(self, node_name):
        node_type_name = self.node_name_to_node_type_name(node_name)
        node_type = self.NODE_TYPES[node_type_name]
        node_data = self.network.nodes()[node_name]
        input_data = [x[2]["in"] for x in self.network.in_edges(node_name, data=True)]
        return node_type.properties_to_node_instance(input_data, node_data, node_name)

    @staticmethod
    def node_name_to_node_type_name(node_name):
        if node_name in ["OutputNode", "InputNode"]:
            return node_name
        return node_name[:-2]  # remove _i from the name

    def draw_network(self):
        labels = {
            node: node + "".join(f"\n{key}: {value}" for key, value in attrs.items() if key != "layer")
            for node, attrs in self.network.nodes(data=True)
        }
        edge_labels = defaultdict(str)
        for out_node, in_node, vals in self.network.edges(data=True):
            edge_labels[(out_node, in_node)] += f"{vals['out']} -> {vals['in']},"

        # try draw_spectral, draw_planar, draw_spring, draw_networkx_labels, draw_networkx
        plt.figure(figsize=(10, 18))
        pos = nx.multipartite_layout(self.network, subset_key="layer", align="horizontal")

        nx.draw(
            self.network,
            pos,
            with_labels=True,
            node_shape="s",
            labels=labels,
            bbox=dict(facecolor="skyblue", boxstyle="round", ec="silver", pad=0.3),
            edge_color="gray",
            arrowsize=100,
        )
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=edge_labels, font_color="red")
        plt.show()


if __name__ == "__main__":
    self = NetworkManager()
    self.initialize_network()
    self.generate_random_network(n_additions=4)
    self.get_all_nodes_values(ParamRequestType.ALL, return_ranges=True)
