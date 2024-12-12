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
ramp = ("ValToRGB",)


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
    to_nodes: List[str]  # names of nodes to connect to
    # names of inputs to choose randomly from. If None then random from all free inputs.
    in_names: Optional[List[str]] = None
    # names of outputs to use from the first node. If None then random. Must be the same length as to_nodes
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

    def get_in_connection(self):
        return [con for con in self.connections if con.from_node == IN][0]

    def get_out_connection(self):
        return [con for con in self.connections if con.from_node == OUT][0]

    def get_n_inputs(self):
        return len(self.get_in_connection().to_nodes)


class MetaNetworkManager(object):
    """
    The manager of the meta network takes a list of possible meta nodes and creates a network of them.
    The generation process is random, starting from an output node and then randomly picks a metanode that can connect
    to it based on the meta-node's defined "required_output_type". Each meta-node can define the required_input_type,
    and at each sample it verifies this matches the output_type and input_type of the connected nodes.
    The layer of each meta-node is it's distance from the output node. The manager will also not generate a node that
    is further than max_layers from the output node.
    If the "IN" gives a list of nodes, then a new connection can be created for either of them.
    TODO: below are some issues that should be improved:
    I am leaving a bit of a mess here. And without full functionality, each metanode can have multiple inputs but not
    outputs. The type of inputs is defined per meta node and not per input to it - which is a bit limiting.
    The fact that a connection can define several to_nodes is also a bit limiting - because it can't define which
    names are relevant to which one. Maybe should have one connection defined for each input and output.
    """
    OUTPUT_NODE_NAME = "OUTPUT"
    OUTPUT_NODE = MetaNode(OUTPUT_NODE_NAME, {'main_node': SubMetaNode(('OutputNode',))}, [Con(IN, ['main_node'])])

    def __init__(self, meta_nodes: List[MetaNode], max_layers: int = 3, n_additions: int = 5):
        self.network = nx.DiGraph()
        self.meta_nodes = meta_nodes
        self.max_layers = max_layers
        self.n_additions = n_additions
        self.node_counts = defaultdict(int)  # node_name -> count of nodes of this type, so we can set unique names
        self.meta_nodes_names = {**{meta_node.name: meta_node for meta_node in meta_nodes}, **{'OUTPUT': self.OUTPUT_NODE}}
        self.nodes_by_input = defaultdict(list)
        self.nodes_by_output = defaultdict(list)
        for meta_node in meta_nodes:
            self.nodes_by_input[meta_node.input_type].append(meta_node.name)
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
        :param output_node_name: The name of the output node to start the generation from.
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
            node_data = self.meta_nodes_names[node_to_add_inputs.split('__')[0]]  # remove the count from the name
            required_input = node_data.required_input_type
            input_type = node_data.input_type

            # if the node has input requirements, filter the possible inputs
            possible_inputs = list(self.meta_nodes_names)
            if required_input != InOutType.ANY:
                possible_inputs = self.nodes_by_output[required_input]
            # filter the possible inputs by their required output type
            final_possibilities = []
            for possible_input in possible_inputs:
                meta_node = self.meta_nodes_names[possible_input]
                # take only nodes that have the required output type, or those that don't care
                if meta_node.required_output_type == input_type or meta_node.required_output_type == InOutType.ANY:
                    final_possibilities.append(possible_input)
            node_to_add = self.meta_nodes_names[np.random.choice(final_possibilities)]

            new_layer = self.network.nodes[node_to_add_inputs]["layer"] + 1
            new_node_name = f"{node_to_add.name}__{self.node_counts[node_to_add.name]}"
            self.node_counts[node_to_add.name] += 1
            self.network.add_node(new_node_name, layer=new_layer, n_inputs=node_to_add.get_n_inputs())
            self.network.add_edge(new_node_name, node_to_add_inputs)

    def meta_network_to_flat_network(self):
        """
        Convert the meta network to the regular network type.
        Visiting every node in the network, we create the nodes in the structure and connect them.
        Then randomize their parameters within the defined limitations.
        """
        nm = NetworkManager()
        nm.initialize_network()

        all_connections = {}
        for meta_node_name in self.network.nodes:
            meta_node_data = self.meta_nodes_names[meta_node_name.split('__')[0]]
            node_names_mapping = {}
            for internal_name, sub_meta_node in meta_node_data.sub_meta_nodes.items():
                possible_types_names = sub_meta_node.sample_group
                node_type_name = np.random.choice(possible_types_names)
                node_name = nm.add_node_by_type_name(node_type_name)
                node_names_mapping[internal_name] = node_name
            all_connections[meta_node_name] = (meta_node_data, node_names_mapping)

        for meta_node_name, (meta_node_data, node_names_mapping) in all_connections.items():
            for connection in meta_node_data.connections:
                internal_name_from = connection.from_node
                for i, internal_name_to in enumerate(connection.to_nodes):
                    if internal_name_from == IN:
                        continue # skip the input node, we will connect them using the output of the predecessor
                    if internal_name_to == OUT:
                        # TODO: consider there could be multiple inputs
                        # TODO: I only need one - not in and out
                        # TODO: from out, get the successor, then pick one of the "ins" it has and connect.
                        # TODO: sample the right node - not using nm.free_inputs because it can have more then one
                        # TODO: make sure I can limit which input to take for that node, or leave todo
                        successor_meta_node = list(self.network.successors(meta_node_name))[0]
                        successor_meta_node_data, successor_name_mapping = all_connections[successor_meta_node]
                        successor_in_connection = successor_meta_node_data.get_in_connection()
                        node_names = [successor_name_mapping[node_internal] for node_internal in successor_in_connection.to_nodes]
                    # get the name from the internal MetaNode definition ("tex1") to the name in the network ("TexNoise_1")
                    from_node = node_names_mapping[internal_name_from]
                    to_node = node_names_mapping[internal_name_to]

                    from_type = nm.node_name_to_node_type(from_node)
                    if connection.out_names is not None:
                        # if defined - get the specific output for this connection
                        # e.g. the "X" output of the "SeparateXYZ" node
                        from_output = connection.out_names[i]
                    else:
                        from_output = from_type.get_random_output()

                    if connection.in_names is not None:
                        # if defined - select from the possible inputs, otherwise select from all free inputs
                        to_possibilities = connection.in_names
                    else:
                        to_possibilities = list(nm.free_inputs[to_node])
                    to_input = np.random.choice(to_possibilities)
                    # not calculating the layers since the generation doesn't have to be from out to in, which will
                    # mess up the layer calculation. It will be calculated at the end.
                    nm.add_edge(from_node, to_node, from_output, to_input, calc_layer=False)

        # TODO: calc layers
        nm.finish_network()
        # TODO: add randomization and limitations of params
        return nm

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


separated_textures = MetaNode(
    "separated_textures",
    {
        "mix": SubMetaNode(mix_vector),
        "tex1": SubMetaNode(textures),
        "tex2": SubMetaNode(textures),
        "mapping": SubMetaNode(mapping),
    },
    [
        Con(IN, ["mapping"]),
        Con("tex1", ["mix"]),
        Con("tex2", ["mix"]),
        Con("mapping", ["tex1", "tex2"]),
        Con("mix", [OUT]),
    ]
)

sep_math_add = MetaNode(
    "sep_math_add",
    {
        "sep": SubMetaNode(sep),
        "math1": SubMetaNode(math_float, {"operation": ["ABSOLUTE", "POWER"]}),
        "math2": SubMetaNode(math_float, {"operation": ["ABSOLUTE", "POWER"]}),
        "math3": SubMetaNode(math_float, {"operation": ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"]}),
    },
    [
        Con(IN, ["sep"]),
        Con("math1", ["math3"]),
        Con("math2",["math3"]),
        Con("sep", ["math1", "math2"], out_names=["X", "Y"]),
        Con("math3",[OUT]),
    ],
    required_input_type=InOutType.VECTOR,
    output_type=InOutType.FLOAT,
)


burn_dodge = MetaNode(
    "burn_dodge",
    {
        "burn": SubMetaNode(mix_vector, {"blend_type": ["Burn"], "B": (0, 0, 0)}),
        "dodge": SubMetaNode(mix_vector, {"blend_type": ["Dodge"], "B": (1, 1, 1)}),
    },

    [Con(IN, ["burn"], in_names=["A"]), Con('burn', ["dodge"], in_names=["A"]), Con('dodge', [OUT])],
    required_output_type=InOutType.OUTPUT,
)

combine_textures = MetaNode(
    "combine_textures",
    {
        "tex_fac": SubMetaNode(textures),
        "mix": SubMetaNode(mix_vector),
        "ramp": SubMetaNode(ramp),
        "tex1": SubMetaNode(textures),
        "tex2": SubMetaNode(textures),
    },
    [
        Con(IN, ["tex_fac", "tex1", "tex2"], in_names=[VECTOR]),
        Con("mix", [OUT]),
        Con("tex_fac", ["ramp"]),
        Con("ramp", ["mix"]),
        Con("tex1", ["mix"]),
        Con("tex2", ["mix"]),
    ],
)

if __name__ == "__main__":
    meta_nodes = [separated_textures, sep_math_add, burn_dodge, combine_textures]
    manager = MetaNetworkManager(meta_nodes, max_layers=3, n_additions=5)
    manager.generate_network()
    flat_network = manager.meta_network_to_flat_network()
