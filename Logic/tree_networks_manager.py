import json
import time
from collections import defaultdict
from typing import Dict, Optional

import networkx as nx
import os
from datetime import datetime
import matplotlib.pyplot as plt

from Logic.bpy_connector import generate_image, check_nm_not_empty
from Logic.blender_tree_manager import BlenderTreeManager
from Logic.node_readers_writers import ParamRequestType
from Logic.utils import (
    deep_unfreeze,
    deep_freeze,
    is_empty_image,
    is_uuid_name,
    force_mutable_key_change,
    force_mutable,
)
from Logic.variations_creator import (
    TwoWayVariationDescriptor,
    apply_variation,
    VariationDescriptor,
    non_structural_changes,
    to_nothing_variation,
    VariationType,
    add_two_variations,
    replace_name_on_edge,
)

IS_CLUSTER_BASE = "is_cluster_base"
HAS_IMAGE = "has_image"
FAILED_IMAGE_GENERATION = "failed_image_generation"
HAD_POST_IMAGE_EXPANSION = "had_post_image_expansion"
ACTION_PREDICTED_VALUE = "action_predicted_value"
CODE_PREDICTED_VALUE = "code_predicted_value"
VISITS = "visits"
TOTAL_VALUE = "total_value"
TEXTURE_SIMILARITY_VALUE = "texture_similarity_value"
IS_EMPTY_IMAGE = "is_empty_image"
IS_EMPTY_NETWORK = "is_empty_network"
ON_PATH_TO_EMPTY = "on_path_to_empty"
MEGA_NODES_GENERATION = "mega_nodes_generation"
META_NODES_GENERATION = "meta_nodes_generation"
RANDOM_GENERATION = "random_generation"


def get_labels_set(labels_set=None):
    if labels_set is None:
        labels_set = set()
    return set(labels_set)  # copy the initial set so we don't override it


class TreesNetworkManager:
    """
    Manages a network of BlenderTreeManager instances using a directed graph. Holds the relationships between different
    BlenderTreeManager instances as nodes and the variations applied to transition between them as edges
    (i.e. what is  the "edit" required to go from one to the other).
    """
    def __init__(self, folder: Optional[str]):
        self.network = nx.DiGraph()
        self.blender_tree_managers: Dict[str, BlenderTreeManager] = {}
        self.cluster_starts = set()
        self.max_id = 0
        self.folder = folder

    def copy(self) -> "TreesNetworkManager":
        """
        Returns a deep copy of the TreesNetworkManager instance.
        Ensures that mutable attributes like network, network_managers, and cluster_starts
        are independently copied.
        """
        new_tnm = TreesNetworkManager(self.folder)
        new_tnm.network = self.network.copy()

        # Deep copy of the blender tree managers dictionary and its values
        new_tnm.blender_tree_managers = {
            key: manager.copy()  # Assumes NetworkManager has a .copy() method
            for key, manager in self.blender_tree_managers.items()
        }
        new_tnm.cluster_starts = self.cluster_starts.copy()
        new_tnm.max_id = self.max_id
        return new_tnm

    def get_sorted_nodes(self, sort_by: str) -> list[tuple]:
        all_nodes = self.network.nodes
        nodes_scores = [(all_nodes[node_id].get(sort_by, float("-inf")), node_id) for node_id in all_nodes]
        return sorted(nodes_scores, reverse=True)

    def _add_node(self, node_id: str, tree_manager: BlenderTreeManager, labels_set=None):
        """Add a new node (a BlenderTreeManager instance) to the network."""
        labels_set = get_labels_set(labels_set)
        assert node_id not in self.network, "Node already exists."
        assert node_id not in self.blender_tree_managers, "Node already exists."
        is_empty_network = tree_manager.is_empty_network()
        if is_empty_network:
            labels_set.add(IS_EMPTY_NETWORK)
        self.network.add_node(node_id, labels=labels_set)
        self.blender_tree_managers[node_id] = tree_manager

    def add_cluster(self, tree_manager: BlenderTreeManager, labels_set=None):
        """Add a new cluster, starting with a given NetworkManager instance."""
        labels_set = get_labels_set(labels_set)
        labels_set.add(IS_CLUSTER_BASE)
        node_id = self._generate_unique_id()
        self._add_node(node_id, tree_manager, labels_set=labels_set)
        self.cluster_starts.add(node_id)
        return node_id

    def _add_edge(self, source: str, target: str, variation_descriptor: VariationDescriptor):
        """Add a new edge to the network."""
        assert source in self.network, "Source node does not exist."
        assert target in self.network, "Target node does not exist."
        variation_type = variation_descriptor.variation_type.name  # convert from enum to string
        self.network.add_edge(source, target, variation_type=variation_type, step=variation_descriptor.step)

    def connect_existing_nodes(self, node1: str, node2: str, connection: TwoWayVariationDescriptor):
        """
        Connect two existing nodes in the network with a given two-way variation descriptor,
        Creating edges in both directions (what is the "edit action" required to go from one to the other and vice versa).
        """
        steps_forwards = connection.steps_forward
        steps_backwards = connection.steps_backward
        assert len(steps_forwards) == 1, "Can only connect existing nodes with one step"
        assert len(steps_backwards) == 1, "Can only connect existing nodes with one step"
        self._add_edge(node1, node2, steps_forwards[0])
        self._add_edge(node2, node1, steps_backwards[0])

    def add_step(self, source_node_id: str, variation_descriptor: VariationDescriptor, labels_set=None):
        """Add a new edit to the network, applying it to a given source node."""
        labels_set = get_labels_set(labels_set)
        new_node_id = self._generate_unique_id()
        new_btm = self.apply_variation(self.blender_tree_managers[source_node_id], variation_descriptor)
        self._add_node(new_node_id, new_btm, labels_set=labels_set)
        self._add_edge(source_node_id, new_node_id, variation_descriptor)
        return new_node_id

    def add_sequence(
        self, start_node_id: str, two_way_variation_descriptor: TwoWayVariationDescriptor, node_labels=None, contract=True
    ):
        """Add a sequence of nodes starting from a given node, given a sequence, adding edges in both directions."""
        assert start_node_id in self.network, "Start node does not exist."
        added_nodes = []
        current_node_id = start_node_id
        for step in two_way_variation_descriptor.steps_forward:
            current_node_id = self.add_step(current_node_id, step, labels_set=node_labels)
            added_nodes.append(current_node_id)

        # Add the backward steps, except for the last one - which will connect back to the start node
        for step in two_way_variation_descriptor.steps_backward[:-1]:
            current_node_id = self.add_step(current_node_id, step, labels_set=node_labels)
            added_nodes.append(current_node_id)

        final_variation = two_way_variation_descriptor.steps_backward[-1]
        self._add_edge(current_node_id, start_node_id, final_variation)

        # Contract nodes that represent identical blender trees
        if contract:
            self._contract_nodes(added_nodes)

        return added_nodes

    def _contract_nodes(self, nodes_to_check: list[str]):
        # group to identical groups of nodes
        node_groups = self._group_identical(nodes_to_check)
        for group in node_groups:
            if len(group) > 1:
                # pick the first in the group as the main node, the rest will merge into it
                main_node = group[0]
                for node in group[1:]:
                    # add the others connections to the main node, and delete from network managers
                    self.network = nx.contracted_nodes(self.network, main_node, node, self_loops=False)
                    # as default, it adds data from the node to the main one under "contraction", we don't want it
                    del self.network.nodes[main_node]["contraction"]
                    del self.blender_tree_managers[node]

    def add_node_label(self, node_id, attr_name):
        """Set an attribute for a given node."""
        assert node_id in self.network, "Node does not exist."
        self.network.nodes[node_id]["labels"].add(attr_name)

    def set_node_value(self, node_id, key, value):
        """Set a key-value for a given node."""
        assert node_id in self.network, "Node does not exist."
        self.network.nodes[node_id][key] = value

    def get_node_value(self, node_id, key, default=None):
        """Get a key-value for a given node."""
        assert node_id in self.network, "Node does not exist."
        return self.network.nodes[node_id].get(key, default)

    def get_nodes_with_label(self, attr_name):
        """Get all nodes that have a certain attribute."""
        return [node for node, node_data in self.network.nodes(data=True) if attr_name in node_data["labels"]]

    def get_nodes_without_label(self, attr_name):
        """Get all nodes that have a certain attribute."""
        return [node for node, node_data in self.network.nodes(data=True) if attr_name not in node_data["labels"]]

    def get_nodes_with_edge_type(self, edge_type):
        nodes = set()
        for u, v, data in self.network.edges(data=True):
            if data["variation_type"] == edge_type:
                nodes.add(u)
        return list(nodes)

    def delete_nodes(self, nodes, images_path=None):
        for node in nodes:
            del self.blender_tree_managers[node]
            self.network.remove_node(node)
            if images_path:
                img_path = self.make_image_path(node, images_path)
                if os.path.exists(img_path):
                    os.remove(img_path)

    def draw_network(self, with_values=False):
        """Draw the network."""
        plt.figure(figsize=(10, 18))
        pos = nx.spring_layout(self.network)
        labels = None
        if with_values:
            labels = {}
            for node, attrs in self.network.nodes(data=True):
                code_value = attrs.get(CODE_PREDICTED_VALUE)
                action_value = attrs.get(ACTION_PREDICTED_VALUE)
                labels[node] = (
                    f"Code %: {round(code_value, 2) if code_value else None}\n"
                    f"Action %: {round(action_value, 2) if action_value else None}\n"
                    f"Image: {HAS_IMAGE in attrs['labels']}"
                )
        nx.draw(
            self.network,
            pos,
            with_labels=True,
            labels=labels,
            bbox=dict(facecolor="skyblue", boxstyle="round", ec="silver", pad=0.3),
            node_shape="s",
        )
        edge_labels = nx.get_edge_attributes(self.network, "weight")
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=edge_labels)
        plt.show()

    def _group_identical(self, nodes: list[str]) -> list[list[str]]:
        """
        Group nodes that represent the same blender tree, so they are identical.
        Returns a list of lists, where each sublist contains the node IDs of identical nodes.
        """
        groups = []
        for node in nodes:
            # Check if the object belongs to an existing group
            for group in groups:
                if (
                    self.blender_tree_managers[node] == self.blender_tree_managers[group[0]]
                ):  # Compare with a representative of the group
                    group.append(node)
                    break
            else:
                # If no group matches, create a new group
                groups.append([node])
        return groups

    def save(self, overwrite=False):
        """Save the current state of the network and network managers."""
        assert self.folder is not None, "Folder path is not set."
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        network_file = os.path.join(self.folder, f"network_{timestamp}.json")
        managers_file = os.path.join(self.folder, f"managers_{timestamp}.json")
        if not overwrite:
            assert not os.path.exists(network_file), "Network file already exists."
            assert not os.path.exists(managers_file), "Managers file already exists."
        network = nx.node_link_data(self.network, edges="edges")
        # convert the set to list for json
        nodes_data = []
        for node_data in network["nodes"]:
            node_data["labels"] = list(node_data["labels"])
            nodes_data.append(node_data)
        network["nodes"] = nodes_data
        network["edges"] = deep_unfreeze(network["edges"])
        network_data = {"network": network, "cluster_starts": list(self.cluster_starts), "max_id": self.max_id}
        with open(network_file, "w") as f:
            json.dump(network_data, f)
        with open(managers_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.blender_tree_managers.items()}, f)

    @staticmethod
    def load(folder, load_networks_managers=True):
        """Load the most recent state of the network and network managers."""
        files = sorted(os.listdir(folder), reverse=True)
        network_file = next(f for f in files if f.startswith("network_"))
        managers_file = next(f for f in files if f.startswith("managers_"))
        new_tnm = TreesNetworkManager(folder)
        with open(os.path.join(folder, network_file), "r") as f:
            network_data = json.load(f)
        new_tnm.cluster_starts = set(network_data["cluster_starts"])
        new_tnm.max_id = network_data["max_id"]
        network = nx.node_link_graph(network_data["network"], edges="edges")
        for _, node_data in network.nodes(data=True):
            # convert back from list to set (saved as list for json)
            node_data["labels"] = set(node_data["labels"])
        for _, _, edge_data in network.edges(data=True):
            edge_data["step"] = deep_freeze(edge_data["step"])
        new_tnm.network = network
        if load_networks_managers:
            with open(os.path.join(folder, managers_file), "r") as f:
                new_tnm.blender_tree_managers = {k: BlenderTreeManager.from_dict(v) for k, v in json.load(f).items()}
        return new_tnm

    def _generate_unique_id(self):
        """Generate a unique ID for a new node."""
        self.max_id += 1
        return str(self.max_id)

    def node_has_label(self, node_name, label):
        return label in self.network.nodes[node_name]["labels"]

    @staticmethod
    def apply_variation(btm: BlenderTreeManager, variation: VariationDescriptor):
        """Apply a variation to a NetworkManager instance and return a new instance."""
        new_network_manager = btm.copy()
        apply_variation(new_network_manager, variation)
        return new_network_manager

    @staticmethod
    def make_image_path(node_id, images_path):
        return os.path.join(images_path, f"{node_id}.png")

    def generate_images_for_nodes(
        self,
        node_ids_to_generate: list[str],
        images_path: str,
        override_images=False,
        save_every: Optional[int]=None,
        print_progress=False,
        resolution=512,
    ):
        """
        Generate images for a list of node IDs and save them to the specified path.
        """
        empty_count = 0
        failed = []
        for i, node_id in enumerate(node_ids_to_generate, start=1):
            if print_progress:
                from IPython.core.display_functions import clear_output

                clear_output(wait=True)
                print(f"working on image {i}/{len(node_ids_to_generate)}")
            nm = self.blender_tree_managers[node_id]
            img_path = self.make_image_path(node_id, images_path)
            if not override_images:
                assert not os.path.exists(img_path), "Image already exists!"
            try:
                generate_image(nm, img_path, resolution=resolution)
                assert os.path.exists(img_path)
                self.add_node_label(node_id, HAS_IMAGE)
                is_empty = bool(is_empty_image(img_path))  # bool to convert from np.bool_
                if is_empty:
                    self.add_node_label(node_id, IS_EMPTY_IMAGE)
                    empty_count += 1
            except Exception as e:
                print("Failed!")
                failed.append((node_id, e))
            if save_every is not None and i % save_every == 0:
                print("Saving...")
                self.save()
        return failed, empty_count

    def generate_images(self, images_path: str, override_images=False, save_every=None, resolution=512):
        images_to_generate = self.get_nodes_without_label(HAS_IMAGE)
        now = time.time()
        failed, empty_count = self.generate_images_for_nodes(
            images_to_generate,
            images_path,
            override_images=override_images,
            save_every=save_every,
            resolution=resolution,
        )
        time_required = time.time() - now
        print(
            f"Generated {len(images_to_generate)} images, including {empty_count} empty images, "
            f"in {round(time_required, 2)} seconds, failed on: {len(failed)}"
        )
        return failed

    def connect_new_node_to_existing_connections(
        self, new_node_id: str, main_node_id: str, new_connection: TwoWayVariationDescriptor
    ):
        """
        This function can be used in the case when we just created a Seed or Param change on a node.
        If that node already had param change variations - so it is connected to other nodes with an edge type of
        Seed or Param - then we can connect the new node to all the old nodes of the same type,
        since they will always be valid variations or the new node as well (because param and seed changes
        don't change the structure of the tree).
        Terminology used here:
        We created a new variation from a "main node". The new variation is called the new node. Other variations
        coming from the main node are called side nodes.
        """
        assert len(new_connection.steps_forward) == 1, "for these types it should only be one step in either direction"
        step_backward = new_connection.steps_backward[0]
        step_forward = new_connection.steps_forward[0]
        variation_type = step_backward.variation_type
        # find all edges going to the main node - the node we just connected to - that have the same type
        edges = self.network.edges(main_node_id, data=True)
        edges_to_connect = [
            edge for edge in edges if edge[2]["variation_type"] == variation_type.name and edge[1] != new_node_id
        ]
        # for each of these edges:
        for _, side_connection_id, step_data in edges_to_connect:
            from_main_node_to_side_node = VariationDescriptor(
                VariationType[step_data["variation_type"]], step=step_data["step"]
            )
            # get the data of the backwards step - from the side node to the main
            backwards_data = self.network.get_edge_data(side_connection_id, main_node_id)
            if backwards_data is None:  # if no backwards connection was created
                continue
            from_side_node_to_main_node = VariationDescriptor(
                VariationType[backwards_data["variation_type"]], step=backwards_data["step"]
            )
            # add the variations - from new to main to side, and the opposite
            from_new_to_side = add_two_variations(step_backward, from_main_node_to_side_node)
            from_side_to_new = add_two_variations(from_side_node_to_main_node, step_forward)
            connection = TwoWayVariationDescriptor(steps_forward=[from_new_to_side], steps_backward=[from_side_to_new])
            self.connect_existing_nodes(new_node_id, side_connection_id, connection)

    def fix_uuid_names(self):
        """
        This is made to fix the fact that in random node adding we give the new nodes a UUID so its name can be
        'TexGradient_5665b166_d9cc_481c_9421_f7ebea95ca95', and not 'TexGradient_2'
        Find all the nodes with long names, find a new name for them and replace everywhere
        The UUID is given originally to avoid name conflicts when adding new nodes.
        """
        # finds all the uuid-based nodes and finds in which nms they participate
        uuid_to_nm_ids = defaultdict(list)
        for nm_id, nm in self.blender_tree_managers.items():
            for node in nm.network.nodes:
                if is_uuid_name(node):
                    uuid_to_nm_ids[node].append(nm_id)
        print(f"Found {len(uuid_to_nm_ids)} uuid-based names to replace")

        # for each node name, get a new name, then replace in all nms and on all edges it appears in
        for node_name, nm_ids in uuid_to_nm_ids.items():
            uuid_node_type = BlenderTreeManager.node_name_to_node_type_name(node_name)
            all_similar_nodes_nums = []
            for nm_id in nm_ids:
                nm = self.blender_tree_managers[nm_id]
                # getting the numbers of all nodes of the same type from all nms that have this node in them
                similar_nodes = [
                    node
                    for node in nm.network.nodes
                    if BlenderTreeManager.node_name_to_node_type_name(node) == uuid_node_type and node != node_name
                ]
                all_similar_nodes_nums.extend([int(s.split("_", maxsplit=1)[1]) for s in similar_nodes])
            new_num = next(
                n for n in range(1, 1000) if n not in all_similar_nodes_nums
            )  # get the first number not used
            new_name = uuid_node_type + f"_{new_num}"
            for nm_id in nm_ids:
                nm = self.blender_tree_managers[nm_id]
                nm.rename_node(node_name, new_name)

            # get all edges coming or going to these nodes
            connected_edges = []  # all edges connected to any nm
            for nm_id in nm_ids:
                connected_edges.extend(self.network.in_edges(nm_id, data=True))
                connected_edges.extend(self.network.out_edges(nm_id, data=True))

            # for each case of edge type if the node name is there, replace it with the new name
            # goes through every place on the edge the name may appear
            for _, _, edge_data in connected_edges:
                new_step = edge_data["step"]
                if edge_data["variation_type"] == VariationType.ADD_NODE.name:
                    if edge_data["step"]["new_node_name"] == node_name:
                        new_step = force_mutable(edge_data["step"], "new_node_name", new_name)
                    elif "edge" in edge_data["step"]:
                        edge = edge_data["step"]["edge"]
                        new_edge = replace_name_on_edge(edge, node_name, new_name)
                        new_step = force_mutable(edge_data["step"], "edge", new_edge)

                elif edge_data["variation_type"] == VariationType.REMOVE_NODE.name:
                    if edge_data["step"]["remove_node_name"] == node_name:
                        new_step = force_mutable(edge_data["step"], "remove_node_name", new_name)
                    elif "replacement_edge" in edge_data["step"]:
                        edge = edge_data["step"]["replacement_edge"]
                        new_edge = replace_name_on_edge(edge, node_name, new_name)
                        new_step = force_mutable(edge_data["step"], "replacement_edge", new_edge)

                elif edge_data["variation_type"] in [VariationType.ADD_EDGE.name, VariationType.REMOVE_EDGE.name]:
                    if edge_data["step"]["in_node"] == node_name:
                        new_step = force_mutable(edge_data["step"], "in_node", new_name)
                    elif edge_data["step"]["out_node"] == node_name:
                        new_step = force_mutable(edge_data["step"], "out_node", new_name)

                elif edge_data["variation_type"] in [
                    VariationType.SEED.name,
                    VariationType.NUMERIC.name,
                    VariationType.CAT_AND_NUMERIC.name,
                ]:
                    if node_name in edge_data["step"]:
                        new_step = force_mutable_key_change(edge_data["step"], node_name, new_name)

                else:
                    raise

                edge_data["step"] = new_step

        # just verify we removed all
        remained_uuids = set()
        for nm_id, nm in self.blender_tree_managers.items():
            for node in nm.network.nodes:
                if is_uuid_name(node):
                    remained_uuids.add(node)
        assert len(remained_uuids) == 0

    def create_partial_str_representation(self, starting_node: str, k_steps: int) -> tuple[str, list[str]]:
        """
        Create a string representation of a subgraph around a starting node within k steps.
        The representation does not include the full details of each node, just their IDs and the types of variations on the edges,
        Mostly useful for presentation purposes.
        """
        network = k_hop_subgraph_both_directions(self.network, starting_node, k_steps)

        nodes_text = '\n'.join([f'node_{x}' for x in network.nodes])
        edges_text = ''
        for edge in nx.node_link_data(network, edges="edges")['edges']:
            if edge['variation_type'] == 'ADD_NODE':
                change_name = f"Add {edge['step']['new_node_type']}"
            elif edge['variation_type'] == 'ADD_EDGE':
                change_name = f"Connect {edge['step']['out_node']} to {edge['step']['in_node']}"
            elif edge['variation_type'] == 'CAT_AND_NUMERIC':
                inner_node_name, changes = list(edge['step'].items())[0]
                attr, change_type = list(changes.items())[0]
                if type(change_type) == float:
                    change_type = "INCREASE"  # due to a bug
                change_name = f"{change_type} {inner_node_name} {attr}"
            elif edge['variation_type'] == 'REMOVE_EDGE':
                change_name = f"Disconnect {edge['step']['out_node']} from {edge['step']['in_node']}"
            elif edge['variation_type'] == 'REMOVE_NODE':
                change_name = f"Remove {edge['step']['remove_node_name']}"
            else:
                raise ValueError()
            edges_text += f'node_{edge["source"]} -- {change_name} --> node_{edge["target"]}\n'

        all_text = f'NODES:\n{nodes_text} \n\n EDGES:\n{edges_text}'
        return all_text, list(network.nodes)

def k_hop_subgraph_both_directions(G, source, k):
    out_nodes = nx.single_source_shortest_path_length(
        G, source, cutoff=k
    ).keys()

    in_nodes = nx.single_source_shortest_path_length(
        G.reverse(copy=False), source, cutoff=k
    ).keys()

    nodes = set(out_nodes) | set(in_nodes)
    return G.subgraph(nodes).copy()

def change_seed(nm, n_changes=5, **kwargs):
    return non_structural_changes(nm, n_changes, ParamRequestType.SEED, **kwargs)


def change_numeric(nm, n_changes=4, **kwargs):
    return non_structural_changes(nm, n_changes, ParamRequestType.NUMERIC, **kwargs)


def change_params(btm: BlenderTreeManager, n_changes=3, **kwargs):
    return non_structural_changes(btm, n_changes, ParamRequestType.NON_SEED, **kwargs)


def completely_random_generation(n_additions=6, n_change_params=5, **kwargs) -> BlenderTreeManager:
    """
    Generate a completely random Blender tree network, applying a number of random node additions and parameter changes.
    """
    btm = BlenderTreeManager()
    btm.initialize_network()
    btm.generate_random_tree(n_additions=n_additions)
    params_change = change_params(btm, n_changes=n_change_params)
    apply_variation(btm, params_change.steps_forward[0])
    return btm


def make_cluster_base(kwargs: dict) -> BlenderTreeManager:
    """
    Make a cluster base that is guaranteed to have an empty network image.
    """
    cluster_base = completely_random_generation(**kwargs)
    for i in range(10):  # just to avoid while True
        if not check_nm_not_empty(cluster_base):
            return cluster_base
        cluster_base = completely_random_generation(**kwargs)
    return cluster_base  # return an empty one if somehow it didn't work after 10 attempts, should be very rare


def make_cluster(tree_network_manager: TreesNetworkManager, concat_param_change=True, cluster_kwargs=None):
    """
    Start a new cluster, and make all the variations leading from an empty blender tree to the cluster base.
    """
    if cluster_kwargs is None:
        cluster_kwargs = {}
    cluster_base = make_cluster_base(cluster_kwargs)
    new_cluster_id = tree_network_manager.add_cluster(cluster_base, labels_set=frozenset({RANDOM_GENERATION}))
    empty_network_variation = to_nothing_variation(cluster_base, concat_param_change=concat_param_change)
    new_nodes = tree_network_manager.add_sequence(
        new_cluster_id, empty_network_variation, node_labels=frozenset({ON_PATH_TO_EMPTY}), contract=True
    )
    return new_nodes


def make_variations(tree_network_manager: TreesNetworkManager, selected_node: str, variation_func, **kwargs):
    nm = tree_network_manager.blender_tree_managers[selected_node]
    two_way_variations: TwoWayVariationDescriptor = variation_func(nm, **kwargs)

    if two_way_variations is None or len(two_way_variations.steps_forward[0].step) == 0:
        fail_msg = (
            f"Attempted to create variation: {variation_func.__name__} on {selected_node} but no variation was created"
        )
        return fail_msg

    new_nodes = tree_network_manager.add_sequence(selected_node, two_way_variations)

    # connect to other nodes that are connected to the same node
    non_structural_variation_types = [VariationType.SEED, VariationType.NUMERIC, VariationType.CAT_AND_NUMERIC]
    if (
        len(two_way_variations.steps_backward) == 1
        and two_way_variations.steps_backward[0].variation_type in non_structural_variation_types
    ):
        assert len(new_nodes) == 1, "If steps backwards is 1 there should only have been one new node"
        tree_network_manager.connect_new_node_to_existing_connections(new_nodes[0], selected_node, two_way_variations)
