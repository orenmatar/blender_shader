from collections import defaultdict

from Logic.node_readers_writers import OutputNode, InputNode, NumericType
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NetworkManager(object):
    NODE_TYPES = ['CombineXYZ', 'Mapping',
                  'Math', 'MixFloat', 'MixVector', 'SeparateXYZ', 'TexGabor', 'TexGradient',
                  'TexNoise', 'TexVoronoiF', 'TexWave', 'ValToRGB', 'Value', 'VectorMath']
    NODE_TYPES = {name: globals()[name] for name in NODE_TYPES if name in globals()}

    def __init__(self):
        self.network = nx.MultiDiGraph()
        self.node_counts = defaultdict(int)
        self.free_inputs = {}
        self.must_connect_to_inputs = {}
        for node_name, node_type in self.NODE_TYPES.items():
            node_vector_inputs = {input_name for input_name, input_type in node_type.get_inputs().items() if
                                  input_type.param_type == NumericType.VECTOR_INPUT}
            self.must_connect_to_inputs[node_name] = node_vector_inputs

    def initialize_network(self):
        self.in_node_name, self.out_node_name = "InputNode", "OutputNode"
        self.add_node(OutputNode, self.out_node_name, {'layer': 1})
        self.add_node(InputNode, self.in_node_name, {'layer': 1})  # input layer incase nothing is connected to it
        self.in_node_output_name = list(InputNode.get_outputs())[0]

    def generate_random_network(self, n_additions=5):
        # only after initialization (add assert)
        for i in range(n_additions):
            random_node = np.random.choice(
                [node_name for node_name, inputs in self.free_inputs.items() if len(inputs) > 0])
            random_in = np.random.choice(list(self.free_inputs[random_node]))
            new_node_type_name = np.random.choice(list(self.NODE_TYPES))
            out = self.NODE_TYPES[new_node_type_name].get_random_output()
            new_node_name = self.add_node_by_type(new_node_type_name)
            self.add_edge(new_node_name, random_node, out, random_in)

        for node_name, free_inputs in self.free_inputs.items():
            for free_input in list(free_inputs):
                if free_input in self.must_connect_to_inputs[self.node_name_to_node_type_name(node_name)]:
                    self.add_edge(self.in_node_name, node_name, self.in_node_output_name, free_input)

    def add_node_by_type(self, node_type_name):
        self.node_counts[node_type_name] += 1
        node_name = node_type_name + f'_{self.node_counts[node_type_name]}'
        node_type = self.NODE_TYPES[node_type_name]
        properties = node_type.get_node_type_params()
        self.add_node(node_type, node_name, properties)
        return node_name

    def add_node(self, node_type, node_name, properties):
        self.network.add_node(node_name, **properties)
        free_inputs = node_type.get_node_type_free_inputs()
        self.free_inputs[node_name] = free_inputs

    def remove_node(self, node1):
        # TODO: make sure to remove inputs from free input dict
        raise NotImplementedError

    def add_edge(self, node1, node2, out1, in2):
        assert in2 in self.free_inputs[node2], 'Input must be free'
        self.network.add_edge(node1, node2, **{'out': out1, 'in': in2}, key=in2)
        self.network.nodes[node1]['layer'] = self.network.nodes[node2]['layer'] + 1
        self.free_inputs[node2].remove(in2)

    def remove_edge(self, node1, node2, in2):
        self.network.remove_edge(node1, node2, key=in2)
        self.free_inputs[node2].add(in2)

    def to_node_instance(self, node_name):
        node_type_name = self.node_name_to_node_type_name(node_name)
        node_type = self.NODE_TYPES[node_type_name]
        node_data = self.network.nodes()[node_name]
        input_data = [x[2]['in'] for x in self.network.in_edges(node_name, data=True)]
        return node_type.properties_to_node_instance(input_data, node_data)

    @staticmethod
    def node_name_to_node_type_name(node_name):
        if node_name in ['OutputNode', 'InputNode']:
            return node_name
        return node_name[:-2]  # remove _i from the name

    def draw_network(self):
        labels = {
            node: node + "".join(f"\n{key}: {value}" for key, value in attrs.items() if key != 'layer')
            for node, attrs in self.network.nodes(data=True)
        }
        edge_labels = defaultdict(str)
        for out_node, in_node, vals in self.network.edges(data=True):
            edge_labels[(out_node, in_node)] += f"{vals['out']} -> {vals['in']},"

        # try draw_spectral, draw_planar, draw_spring, draw_networkx_labels, draw_networkx
        plt.figure(figsize=(10, 18))
        pos = nx.multipartite_layout(self.network, subset_key="layer", align="horizontal")

        nx.draw(self.network, pos, with_labels=True, node_shape='s',
                labels=labels,
                bbox=dict(facecolor="skyblue",
                          boxstyle="round", ec="silver", pad=0.3),
                edge_color="gray",
                arrowsize=100)
        nx.draw_networkx_edge_labels(
            self.network, pos,
            edge_labels=edge_labels,
            font_color='red'
        )
        plt.show()