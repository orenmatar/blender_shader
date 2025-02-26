import copy
import json
from collections import defaultdict
from typing import Tuple

from Logic.bpy_connector import generate_image
from Logic.node_readers_writers import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Logic.utils import sample_uniform, is_empty_image


class NetworkManager(object):
    """
    This class is responsible for managing the network (using networkx) of nodes.
    """

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
        "TexVoronoiDistance": TexVoronoiDistance,
        "TexWave": TexWave,
        "ValToRGB": ValToRGB,
        "Value": Value,
        "VectorMath": VectorMath,
        "OutputNode": OutputNode,
        "InputNode": InputNode,
    }
    OutputNodeNAME = "OutputNode"
    InputNodeNAME = "InputNode"
    NODE_TYPES_FOR_GENERATION = [
        node_name for node_name in NODE_TYPES if node_name not in ["OutputNode", "InputNode", "Value", "MixFloat"]
    ]
    INITIALIZATION_CODE = """import bpy
import sys
sys.path.append('/Users/orenm/BlenderShaderProject/project_files/')

from Logic.bpy_connector import set_file_path, clean_scene, set_for_texture_generation, settings_for_texture_generation, NodesAdder

path = '/Users/orenm/Desktop/test.png'
clean_scene()
set_for_texture_generation()
settings_for_texture_generation(resolution=512)
set_file_path(path)

material = bpy.data.materials.new(name='my_material')
material.use_nodes = True
bpy.data.objects['Plane'].data.materials.append(material)
nodes = material.node_tree.nodes
links = material.node_tree.links
[nodes.remove(n) for n in nodes]
node_tree = material.node_tree
nodes_adder = NodesAdder(material.node_tree)
    """

    def __init__(self):
        self.network = nx.MultiDiGraph()
        self.node_counts = defaultdict(int)  # node_name -> count of nodes of this type, so we can set unique names
        self.free_inputs = {}  # node_name -> set of free inputs we can connect to at the moment
        # TODO: these should probably just be constants
        self.input_node_name, self.output_node_name = "InputNode_1", "OutputNode_1"
        self.node_value_ranges_name = "node_value_ranges"

    def copy(self):
        """
        Copy the network manager
        """
        new_manager = NetworkManager()
        new_manager.network = self.network.copy()
        new_manager.node_counts = copy.deepcopy(self.node_counts)
        new_manager.free_inputs = copy.deepcopy(self.free_inputs)
        new_manager.input_node_name = self.input_node_name
        new_manager.output_node_name = self.output_node_name
        new_manager.node_value_ranges_name = self.node_value_ranges_name
        return new_manager

    def to_dict(self):
        """
        Convert the network to a dict
        """
        network_data = nx.node_link_data(self.network, edges="edges")
        # convert the free inputs to a list, so it can be saved to json
        free_inputs_as_list = {key: list(value) for key, value in self.free_inputs.items()}
        manager_data = {
            "node_counts": self.node_counts,
            "free_inputs": free_inputs_as_list,
            "in_node_name": self.input_node_name,
            "out_node_name": self.output_node_name,
            "node_value_ranges_name": self.node_value_ranges_name,
            "network_data": network_data,
        }
        return manager_data

    def dump_to_json(self, filename):
        """
        Dump the network to a json file
        """
        as_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(as_dict, f)

    @staticmethod
    def from_dict(manager_data):
        """
        Create a network manager from a dict
        """
        new_manager = NetworkManager()
        new_manager.node_counts = defaultdict(int, manager_data["node_counts"])
        new_manager.free_inputs = {k: set(v) for k, v in manager_data["free_inputs"].items()}
        new_manager.input_node_name = manager_data["in_node_name"]
        new_manager.output_node_name = manager_data["out_node_name"]
        new_manager.node_value_ranges_name = manager_data["node_value_ranges_name"]
        network = nx.node_link_graph(manager_data["network_data"], edges="edges")
        new_manager.network = network
        return new_manager

    @staticmethod
    def load_from_json(filename):
        """
        Load the network from a json file
        """
        with open(filename, "r") as f:
            manager_data = json.load(f)
        return NetworkManager.from_dict(manager_data)

    def is_empty_network(self):
        """Returns True if the network is meaningless - the input connected directly to the output"""
        return self.network.has_edge("InputNode_1", "OutputNode_1")

    def network_data_for_comparison(self):
        """
        Get the network data for comparison - only the important properties, sorted so it is easy to compare
        :return:
        """
        all_nodes = copy.deepcopy(dict(self.network.nodes(data=True)))
        # make a dict of nodes and their attributes, excluding the layer and value ranges which are not important
        for node_name, node_props in all_nodes.items():
            if "layer" in node_props:
                del node_props["layer"]
            if self.node_value_ranges_name in node_props:
                del node_props[self.node_value_ranges_name]
        # make a sorted list of edges, so it is easy to compare, with the dict of attributes as a frozenset
        all_edges = sorted([(node1, node2, frozenset(data)) for node1, node2, data in self.network.edges(data=True)])
        return all_nodes, all_edges

    @staticmethod
    def find_nodes_differences(network1, network2):
        differences = defaultdict(dict)
        nodes1, edges1 = network1.network_data_for_comparison()
        nodes2, edges2 = network2.network_data_for_comparison()
        for node_name, vals in nodes1.items():
            for val_name, val in vals.items():
                if val != nodes2[node_name][val_name]:
                    differences[node_name][val_name] = (val, nodes2[node_name][val_name])
        return differences

    @staticmethod
    def compare_networks(network1, network2, compare_node_properties=True, tuple_list_hack=False):
        """
        Compare two networks, given their nodes and edges
        If compare_node_properties is False, only compare the node types, not the attribute values (i.e., doesn't matter
        if scale or some param is different)
        """
        nodes1, edges1 = network1.network_data_for_comparison()
        nodes2, edges2 = network2.network_data_for_comparison()
        if compare_node_properties:
            if not tuple_list_hack:
                return nodes1 == nodes2 and edges1 == edges2
            # OMG i'm so sorry for this hack. Don't know why exactly but sometimes we can node attributes with the same
            # values but one is a tuple and another is a list, so it returns False, as the networks are not equal.
            # instead of fixing it - i'm checking here if that's the case, and they still have the same values - then we
            # can still say the networks are equal.
            if nodes1 != nodes2:
                nodes_diffs = NetworkManager.find_nodes_differences(network1, network2)
                for node_name, vals in nodes_diffs.items():
                    for val_name, (val_1, val_2) in vals.items():
                        if tuple(val_2) != tuple(val_1):  # if even cases as tuple they are not the same - then false
                            return False
            # if we didn't find any differences, then just check the edges
            return edges1 == edges2
        node_types1 = {NetworkManager.node_name_to_node_type_name(node) for node in nodes1}
        node_types2 = {NetworkManager.node_name_to_node_type_name(node) for node in nodes2}
        return node_types1 == node_types2 and edges1 == edges2

    def __eq__(self, other):
        return self.compare_networks(self, other)

    def initialize_network(self):
        """
        Initialize the network with input and output nodes
        """
        self.output_node_name = self.add_node_by_type_name(self.OutputNodeNAME)
        self.input_node_name = self.add_node_by_type_name(self.InputNodeNAME)

    def apply_distribution_limitations(self, nodes_and_dist: Dict[str, Dict[str, Tuple[tuple, ParamType]]]):
        """
        Given a dict of nodes and their distribution for sample - apply their limitations (if any are saved on them),
        narrowing the distribution.
        """
        for node_name, distributions in nodes_and_dist.items():
            # take the limitations from the node on the network
            node_distribution_limitations = self.network.nodes(self.node_value_ranges_name)[node_name]
            if node_distribution_limitations is not None:
                for attr_name, new_dist in node_distribution_limitations.items():
                    if attr_name in distributions:
                        distributions[attr_name] = (new_dist, distributions[attr_name][1])

    def get_random_param_values(self, param_type: ParamRequestType, with_limitations=True):
        """
        Get random values for all the nodes in the network, given a param type - seeds, numeric only, categorical...
        """
        dist_vals = self.get_all_nodes_values(param_type, return_ranges=True, not_input_values=True)
        if with_limitations:
            self.apply_distribution_limitations(dist_vals)
        return self.pick_random_values_from_dict(dist_vals)

    def set_nodes_attributes(self, update_attributes: dict):
        """
        Set the attributes of the nodes in the network
        """
        for node, attributes in update_attributes.items():
            for attr_name, value in attributes.items():
                self.network.nodes[node][attr_name] = value

    def get_node_non_default_vals(self, node_name):
        """
        Get the param values only if they are not the default values of that node type, for a specific node
        """
        node_instance = self.to_node_instance(node_name)
        default_vals = node_instance.get_node_type_params(ParamRequestType.ALL, default_values=True)
        current_cals = node_instance.numeric | node_instance.categorical | node_instance.seeds
        non_default_vals = {k: v for k, v in current_cals.items() if v != default_vals[k]}
        return non_default_vals

    def generate_random_network(self, n_additions=5):
        """
        Generate a random network by adding nodes and connecting them randomly
        ONLY after initialization (consider adding assert)
        """
        assert self.output_node_name in self.network.nodes, "Network must be initialized"
        for i in range(n_additions):
            # pick a free input
            free_inputs = [node_name for node_name, inputs in self.free_inputs.items() if len(inputs) > 0]
            if len(free_inputs) == 0:
                break
            random_node = np.random.choice(free_inputs)
            random_in = np.random.choice(list(self.free_inputs[random_node]))
            # pick a random node to connect to it
            new_node_type_name = np.random.choice(self.NODE_TYPES_FOR_GENERATION)
            out = self.NODE_TYPES[new_node_type_name].get_random_output()
            new_node_name = self.add_node_by_type_name(new_node_type_name)
            self.add_edge(new_node_name, random_node, out, random_in)
        self.finish_network()

    def finish_network(self):
        """
        Finish the network by connecting all the nodes that must be connected to the input node
        """
        assert self.output_node_name in self.network.nodes, "Network must be initialized"
        # node_name -> set of inputs that must be connected to - (vector inputs)
        must_connect_to_inputs = {}
        for node_name, node_type in self.NODE_TYPES.items():
            node_vector_inputs = {
                input_name
                for input_name, input_type in node_type.get_inputs().items()
                if input_type.param_type == ParamType.VECTOR_INPUT  # VECTOR_INPUT must be connected to something
            }
            must_connect_to_inputs[node_name] = node_vector_inputs

        # connect free inputs that need a connection to the input node
        input_node_output_name = list(InputNode.get_outputs())[0]
        for node_name, free_inputs in self.free_inputs.items():
            for free_input in list(free_inputs):
                if free_input in must_connect_to_inputs[self.node_name_to_node_type_name(node_name)]:
                    self.add_edge(self.input_node_name, node_name, input_node_output_name, free_input)

        self.calc_layers()

    def set_node_distribution_limitations(self, node_name, dist):
        """
        Set the distribution limitations for a specific node attribute
        """
        self.network.nodes[node_name][self.node_value_ranges_name] = dist

    def add_node_by_type_name(self, node_type_name):
        """
        Add a node of a specific type to the network. Node name is unique - by the count of that type
        """
        node_name = node_type_name + f"_{self.node_counts[node_type_name]+1}"
        self.add_node_by_type_and_name(node_type_name, node_name)
        return node_name

    def add_node_by_type_and_name(self, node_type_name, node_name):
        """
        Add a node of a specific type to the network. Node name is unique - by the count of that type
        """
        if node_type_name in {self.OutputNodeNAME, self.InputNodeNAME}:
            assert self.node_counts[node_type_name] == 0, "Only one input and output node allowed"
        self.node_counts[node_type_name] += 1
        node_type = self.NODE_TYPES[node_type_name]
        properties = node_type.get_node_type_params(ParamRequestType.ALL, default_values=True)
        self._add_node(node_type, node_name, properties)
        return node_name

    def get_node_connected_inputs(self, node):
        """
        Get the inputs that are connected to a specific node
        """
        in_edges = self.network.in_edges(node, data=True)
        return [edge_properties["in"] for previous_node, self_node, edge_properties in in_edges]

    def get_nodes_properties(self, nodes_properties: list):
        """
        Get the properties of the nodes in the network
        """
        nodes = self.network.nodes(data=True)
        res = defaultdict(dict)
        for node, prop in nodes_properties:
            res[node][prop] = nodes[node][prop]
        return res

    def get_all_nodes_values(self, param_type: ParamRequestType, return_ranges=False, not_input_values=True):
        """
        Get all the values of the nodes in the network, given a param type - seeds, numeric only, categorical...
        """
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
    def pick_random_values_from_dict(nodes_and_dist: Dict[str, Dict[str, Tuple[tuple, ParamType]]]):
        """
        Pick a random value for each node param, from a dictionary of node params and their distributions
        """
        result = {}
        for node, values in nodes_and_dist.items():
            node_rand_vals = {}
            for attr_name, (dist, attr_type) in values.items():
                if attr_type in [ParamType.FLOAT, ParamType.SEED]:
                    value = float(sample_uniform(low=dist[0], high=dist[1], size=1).round(1)[0])
                elif attr_type == ParamType.VECTOR:
                    value = list(sample_uniform(low=dist[0], high=dist[1], size=3).round(1))
                elif attr_type == ParamType.CATEGORICAL:
                    value = np.random.choice(dist)
                else:
                    raise ValueError
                node_rand_vals[attr_name] = value
            result[node] = node_rand_vals
        return result

    def _add_node(self, node_type, node_name, properties):
        """
        Adds a note to the network, and keeps track of its inputs
        """
        assert node_name not in self.network.nodes, "Node name must be unique"
        self.network.add_node(node_name, **properties)
        free_inputs = node_type.get_node_type_free_inputs()
        self.free_inputs[node_name] = free_inputs

    def remove_node(self, node1):
        # first remove all edges
        incoming_edges = self.network.in_edges(node1, data=True)
        outgoing_edges = self.network.out_edges(node1, data=True)
        all_edges = list(incoming_edges) + list(outgoing_edges)
        for edge in all_edges:
            self.remove_edge(edge[0], edge[1], edge[2]["in"])
        # remove from free inputs, and remove node
        if node1 in self.free_inputs:
            del self.free_inputs[node1]
        self.network.remove_node(node1)

    def add_edge(self, node1, node2, out1, in2):
        assert in2 in self.free_inputs[node2], "Input must be free"
        self.network.add_edge(node1, node2, **{"out": out1, "in": in2}, key=in2)
        self.free_inputs[node2].remove(in2)

    def remove_edge(self, node1, node2, in2):
        self.network.remove_edge(node1, node2, key=in2)
        self.free_inputs[node2].add(in2)

    def calc_layers(self):
        """
        Calculates the layers of the nodes - their distance from output. Layer is important for visualization
        """
        assert self.output_node_name in self.network.nodes, "Network must be initialized"
        lengths = nx.shortest_path_length(self.network.reverse(copy=False), self.output_node_name)
        for node, length in lengths.items():
            self.network.nodes[node]["layer"] = length + 1
        for node in self.network.nodes:
            if node not in lengths:  # in case it wasn't connected
                self.network.nodes[node]["layer"] = 15

    def to_node_instance(self, node_name: str) -> Node:
        """
        Convert a node in the network to a node instance (from node_readers_writers.py)
        """
        node_type_name = self.node_name_to_node_type_name(node_name)
        node_type = self.NODE_TYPES[node_type_name]
        node_data = self.network.nodes[node_name]
        input_data = [x[2]["in"] for x in self.network.in_edges(node_name, data=True)]
        return node_type.properties_to_node_instance(input_data, node_data, node_name)

    def node_name_to_node_type(self, node_name: str) -> type(Node):
        """
        Convert a node name to a node type name - simply remove the _i from the name
        """
        type_name = self.node_name_to_node_type_name(node_name)
        return self.NODE_TYPES[type_name]

    @staticmethod
    def node_name_to_node_type_name(node_name: str) -> str:
        """
        Convert a node name to a node type name - simply remove the _i from the name
        """
        return node_name.split("_")[0]  # remove _i from the name

    def draw_network(self):
        """
        Draw the network using networkx
        """
        # set the labels to include all the params of the node (otherwise it's just the name)
        dont_include_params = {"layer", self.node_value_ranges_name}
        labels = {
            node: node + "".join(f"\n{key}: {value}" for key, value in attrs.items() if key not in dont_include_params)
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

    def calculate_true_free_inputs(self):
        """
        Calculate the true free inputs of the network, given the current connections
        Useful for debugging and checking the network, if the free inputs are incorrect
        """
        true_free_inputs = {}
        for node in self.network.nodes():
            node_type = self.node_name_to_node_type(node)
            true_free_inputs[node] = node_type.get_node_type_free_inputs()

        for edge in self.network.edges(data=True):
            in_node = edge[1]
            in_name = edge[2]["in"]
            true_free_inputs[in_node].remove(in_name)
        return true_free_inputs

    def generate_code(self, with_initialization_code=False):
        """
        Generate the code for the network
        """
        # generating by order of layer, so output is last
        layers = defaultdict(list)
        for node_name, data in self.network.nodes(data=True):
            layers[data["layer"]].append(node_name)

        code = ""
        for i in range(max(layers), 0, -1):
            for node_name in layers[i]:
                node_instance = self.to_node_instance(node_name)
                code += node_instance.to_code("nodes_adder.create_node")

        for out1, in2, data in self.network.edges(data=True):
            input_mapping = self.node_name_to_node_type(in2).INPUT_MAPPING
            code += f"\nnode_tree.links.new({out1}.outputs['{data['out']}'], {in2}.inputs[{input_mapping[data['in']]}])"
        if with_initialization_code:
            code = self.INITIALIZATION_CODE + code
        return code


def check_nm_not_empty(nm):
    generate_image(nm, "/tmp/tmp.png")
    return is_empty_image("/tmp/tmp.png")


if __name__ == "__main__":
    self = NetworkManager()
    self.initialize_network()
    self.generate_random_network(n_additions=4)
    self.get_all_nodes_values(ParamRequestType.ALL, return_ranges=True)
