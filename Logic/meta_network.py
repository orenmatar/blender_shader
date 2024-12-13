from typing import Tuple, Optional
from dataclasses import dataclass

from black.trans import defaultdict

from Logic.network_manager import NetworkManager
from Logic.node_readers_writers import *
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum

IN = "IN"
OUT = "OUT"
textures = ("TexNoise", "TexWave", "TexGabor", "TexGradient", "TexVoronoiF")
mix_vector = ("MixVector",)
mapping = ("Mapping",)
sep = ("SeparateXYZ",)
math_float = ("Math",)
vector_math = ("VectorMath",)
either_math = ("Math", "VectorMath")
ramp = ("ValToRGB",)
combine = ("CombineXYZ",)
value = ("Value",)


class InOutType(Enum):
    VECTOR = 1
    FLOAT = 2
    ANY = 3
    OUTPUT = 4


@dataclass
class SubMetaNode:
    sample_group: tuple
    allowed_params: Optional[dict] = None


@dataclass
class Con:  # Connection for short
    from_node: str  # name of the node to connect from
    to_node: str  # names of node to connect to
    # names of possible inputs in to_node. If None then random from all free inputs.
    in_names: Optional[List[str]] = None
    # names of possible output to use from the from_node. If None then random.
    out_names: Optional[List[str]] = None


@dataclass
class MetaNode(object):
    name: str
    sub_meta_nodes: Dict[str, SubMetaNode]
    connections: List[Con]
    output_type: InOutType = InOutType.VECTOR
    input_type: InOutType = InOutType.VECTOR
    required_output_type: InOutType = InOutType.ANY
    required_input_type: InOutType = InOutType.ANY

    def get_in_connections(self):
        return [con for con in self.connections if con.from_node == IN]

    def get_out_connection(self):
        return [con for con in self.connections if con.from_node == OUT][0]

    def get_n_inputs(self):
        return len(self.get_in_connections())


class MetaNetworkManager(object):
    """
    The manager of the meta network takes a list of possible meta nodes and creates a network of them.
    The generation process is random, starting from an output node and then randomly picks a metanode that can connect
    to it based on the meta-node's defined "required_output_type". Each meta-node can define the required_input_type,
    and at each sample it verifies this matches the output_type and input_type of the connected nodes.
    The layer of each meta-node is it's distance from the output node. The manager will also not generate a node that
    is further than max_layers from the output node.
    """

    OUTPUT_NODE_NAME = "OUTPUT"
    OUTPUT_NODE = MetaNode(OUTPUT_NODE_NAME, {"main_node": SubMetaNode(("OutputNode",))}, [Con(IN, "main_node")], input_type=InOutType.OUTPUT)

    def __init__(self, meta_nodes: List[MetaNode], max_layers: int = 3, n_additions: int = 5):
        self.network = nx.DiGraph()
        self.meta_nodes_for_sample = {meta_node.name: meta_node for meta_node in meta_nodes}
        self.max_layers = max_layers
        self.n_additions = n_additions
        self.node_counts = defaultdict(int)  # node_name -> count of nodes of this type, so we can set unique names
        self.meta_nodes_names = {**self.meta_nodes_for_sample, **{"OUTPUT": self.OUTPUT_NODE}}
        self.nodes_by_output = defaultdict(list)
        for meta_node in meta_nodes:
            self.nodes_by_output[meta_node.output_type].append(meta_node.name)
            # make sure that the list contains inputs to fill each required input
        for meta_node in meta_nodes:
            if meta_node.required_input_type not in [InOutType.ANY, InOutType.OUTPUT]:
                assert (
                    len(self.nodes_by_output[meta_node.required_input_type]) > 0
                ), f"Node {meta_node.name} has no input nodes of type {meta_node.required_input_type}"

    def generate_network(self):
        """
        Generate a network starting from the output node.
        """
        self.network.add_node(f"{self.OUTPUT_NODE_NAME}__1", layer=0, n_inputs=1)

        for i in range(self.n_additions):
            # Get nodes with open inputs - fewer inputs than what they have to offer
            nodes_open_connections = [
                node
                for node in self.network.nodes
                if self.network.in_degree(node) < self.network.nodes[node]["n_inputs"]
            ]
            # Get nodes that are not too deep
            nodes_not_too_deep = [
                node for node in nodes_open_connections if self.network.nodes[node]["layer"] < self.max_layers
            ]
            if len(nodes_not_too_deep) == 0:
                break
            # Get a node to add inputs to
            node_to_add_inputs = np.random.choice(nodes_not_too_deep)
            node_data = self.meta_nodes_names[node_to_add_inputs.split("__")[0]]  # remove the count from the name
            required_input = node_data.required_input_type
            input_type = node_data.input_type

            # if the node has input requirements, filter the possible inputs
            possible_inputs = list(self.meta_nodes_for_sample)
            if required_input != InOutType.ANY:
                possible_inputs = self.nodes_by_output[required_input]
            # filter the possible inputs by their required output type
            final_possibilities = []
            for possible_input in possible_inputs:
                meta_node = self.meta_nodes_for_sample[possible_input]
                # take only nodes that have the required output type, or those that don't care
                if meta_node.required_output_type in [input_type, InOutType.ANY]:
                    final_possibilities.append(possible_input)
            node_to_add = self.meta_nodes_for_sample[np.random.choice(final_possibilities)]

            new_layer = self.network.nodes[node_to_add_inputs]["layer"] + 1
            # make sure each node has a unique name by giving it an index after the name "node__i"
            new_node_name = f"{node_to_add.name}__{self.node_counts[node_to_add.name]}"
            self.node_counts[node_to_add.name] += 1
            self.network.add_node(new_node_name, layer=new_layer, n_inputs=node_to_add.get_n_inputs())
            self.network.add_edge(new_node_name, node_to_add_inputs)

    def meta_network_to_flat_network(self):
        """
        Convert the meta network to the regular network type.
        Create a regular (non meta) node for every node that is in the meta structure, keeping track of their real names
        (as opposed to the internal names used by the meta structure definition)
        Then create all the edges between them - using a mapping between internal and real names.
        Meanwhile, collect the connection data between IN and OUT connections
        Finally make those connections between the meta nodes.
        """
        nm = NetworkManager()
        nm.initialize_network()

        # create a node for every internal node of a meta structure, and collect connection names
        all_connections = {}
        for meta_node_name in self.network.nodes:
            # get the meta node type (remove index)
            meta_node_data = self.meta_nodes_names[meta_node_name.split("__")[0]]
            node_names_mapping = {}
            # create nodes by the internal nodes and keep track of their real names and internal names
            for internal_name, sub_meta_node in meta_node_data.sub_meta_nodes.items():
                possible_types_names = sub_meta_node.sample_group
                # sample the actual node from the sample group
                node_type_name = np.random.choice(possible_types_names)
                if node_type_name == nm.OutputNodeNAME:  # output node is added in initialization
                    node_name = nm.out_node_name
                else:
                    node_name = nm.add_node_by_type_name(node_type_name)
                node_names_mapping[internal_name] = node_name
                # if there are limitations on the distribution of params for the node - write them to the real node
                if sub_meta_node.allowed_params is not None:
                    nm.set_node_distribution_limitations(node_name, sub_meta_node.allowed_params)
            all_connections[meta_node_name] = (meta_node_data, node_names_mapping)

        # Create all the connections between the nodes
        meta_nodes_connections = defaultdict(lambda: defaultdict(list))
        for meta_node_name, (meta_node_data, node_names_mapping) in all_connections.items():
            for connection in meta_node_data.connections:
                internal_name_from = connection.from_node
                internal_name_to = connection.to_node
                # if the connection is with another meta-node - just add it to the dict and we'll connect them later
                if internal_name_from == IN:
                    to_node = node_names_mapping[internal_name_to]
                    meta_nodes_connections[meta_node_name][IN].append((to_node, connection))
                    continue
                if internal_name_to == OUT:
                    # take the name of the meta node the out is directed to and add the data to it
                    successor_meta_node = list(self.network.successors(meta_node_name))[0]
                    from_node = node_names_mapping[internal_name_from]
                    meta_nodes_connections[successor_meta_node][OUT].append((from_node, connection))
                    continue

                from_node = node_names_mapping[internal_name_from]
                to_node = node_names_mapping[internal_name_to]
                to_input = self.choose_to_name(nm, connection, to_node)
                from_output = self.choose_from_name(nm, connection, from_node)
                nm.add_edge(from_node, to_node, from_output, to_input)

        # after we collected the meta nodes connections, actually connect them
        for meta_node_to, connections in meta_nodes_connections.items():
            for out_connection in connections[OUT]:
                from_node, from_connection = out_connection
                in_connections = connections[IN]
                # randomly pick one of the in connections and remove from list
                to_node, to_connection = in_connections.pop(np.random.randint(len(in_connections)))
                to_input = self.choose_to_name(nm, to_connection, to_node)
                from_output = self.choose_from_name(nm, from_connection, from_node)
                # not calculating the layers since the generation doesn't have to be from out to in, which will
                # mess up the layer calculation. It will be calculated at the end.
                nm.add_edge(from_node, to_node, from_output, to_input)

        nm.finish_network()
        return nm

    @staticmethod
    def choose_from_name(network_manager: NetworkManager, connection: Con, from_node: str):
        """
        Chooses the name of the output to take from the from_node, given options and limitations of the connection
        """
        from_type = network_manager.node_name_to_node_type(from_node)
        from_options = list(from_type.get_outputs())
        if connection.out_names is not None:
            # if defined - choose randomly from the limited output names
            from_options = [x for x in from_options if x in connection.out_names]
        from_output = np.random.choice(from_options)
        return from_output

    @staticmethod
    def choose_to_name(network_manager: NetworkManager, connection: Con, to_node: str):
        """
        Chooses the name of the input to the to_node, given what is free and the limitations of the connection
        """
        to_options = list(network_manager.free_inputs[to_node])
        if connection.in_names is not None:
            # if defined - select from the possible inputs, otherwise select from all free inputs
            to_options = [x for x in to_options if x in connection.in_names]
        to_input = np.random.choice(to_options)
        return to_input

    def draw_network(self):
        """
        Draw the network using networkx
        """
        # try draw_spectral, draw_planar, draw_spring, draw_networkx_labels, draw_networkx
        plt.figure(figsize=(10, 18))
        pos = nx.multipartite_layout(self.network, subset_key="layer", align="horizontal")
        nx.draw(
            self.network,
            pos,
            with_labels=True,
            node_shape="s",
            bbox=dict(facecolor="skyblue", boxstyle="round", ec="silver", pad=0.3),
            edge_color="gray",
            arrowsize=100,
        )
        plt.show()
