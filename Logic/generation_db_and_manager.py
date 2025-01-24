import json
import time
from typing import Dict, List

import networkx as nx
import os
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.core.display_functions import clear_output

from Logic.bpy_connector import generate_image
from Logic.meta_network import MetaNetworkManager
from Logic.network_manager import NetworkManager, check_nm_not_empty
from Logic.node_readers_writers import ParamRequestType
from Logic.structures_definitions import ALL_META_NODES, MEGA_STRUCTURES
from Logic.utils import deep_unfreeze, deep_freeze, is_empty_image
from Logic.variations_creator import (
    TwoWayVariationDescriptor,
    apply_variation,
    VariationDescriptor,
    non_structural_changes,
    to_nothing_variation,
    VariationType,
    add_two_variations,
)

IS_CLUSTER_BASE = "is_cluster_base"
HAS_IMAGE = "has_image"
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


class DBManager:
    def __init__(self, folder):
        self.network = nx.DiGraph()
        self.network_managers: Dict[str, NetworkManager] = {}
        self.cluster_starts = set()
        self.max_id = 0
        self.folder = folder

    def _add_node(self, node_id, network_manager: NetworkManager, labels_set=None):
        """Add a new node to the network."""
        labels_set = get_labels_set(labels_set)
        assert node_id not in self.network, "Node already exists."
        assert node_id not in self.network_managers, "Node already exists."
        is_empty_network = network_manager.is_empty_network()
        if is_empty_network:
            labels_set.add(IS_EMPTY_NETWORK)
        self.network.add_node(node_id, labels=labels_set)
        self.network_managers[node_id] = network_manager

    def add_cluster(self, network_manager: NetworkManager, labels_set=None):
        """Add a new cluster with a given NetworkManager instance."""
        labels_set = get_labels_set(labels_set)
        labels_set.add(IS_CLUSTER_BASE)
        node_id = self._generate_unique_id()
        self._add_node(node_id, network_manager, labels_set=labels_set)
        self.cluster_starts.add(node_id)
        return node_id

    def _add_edge(self, source, target, variation_descriptor: VariationDescriptor):
        """Add a new edge to the network."""
        assert source in self.network, "Source node does not exist."
        assert target in self.network, "Target node does not exist."
        variation_type = variation_descriptor.variation_type.name  # convert from enum to string
        self.network.add_edge(source, target, variation_type=variation_type, step=variation_descriptor.step)

    def connect_existing_nodes(self, node1, node2, connection: TwoWayVariationDescriptor):
        steps_forwards = connection.steps_forward
        steps_backwards = connection.steps_backward
        assert len(steps_forwards) == 1, "Can only connect existing nodes with one step"
        assert len(steps_backwards) == 1, "Can only connect existing nodes with one step"
        self._add_edge(node1, node2, steps_forwards[0])
        self._add_edge(node2, node1, steps_backwards[0])

    def add_step(self, source_node_id, variation_descriptor: VariationDescriptor, labels_set=None):
        """Add a new step to the network."""
        labels_set = get_labels_set(labels_set)
        new_node_id = self._generate_unique_id()
        new_network_manager = self._apply_variation(self.network_managers[source_node_id], variation_descriptor)
        self._add_node(new_node_id, new_network_manager, labels_set=labels_set)
        self._add_edge(source_node_id, new_node_id, variation_descriptor)
        return new_node_id

    def add_sequence(
        self, start_node_id, two_way_variation_descriptor: TwoWayVariationDescriptor, node_labels=None, contract=True
    ):
        """Add a sequence of nodes starting from a given node."""
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

        # Contract nodes that have the same network manager
        # This can happen as we remove nodes and then add them back and get the same network
        # and also sometimes we "add edge" from input to a node that already has that edge
        if contract:
            # group to identical groups of nodes
            # TODO: the labels of the contracted node are only the labels of the main one, not sure if that's ok
            node_groups = self.group_identical(added_nodes)
            for group in node_groups:
                if len(group) > 1:
                    # pick the first in the group as the main node, the rest will merge into it
                    main_node = group[0]
                    for node in group[1:]:
                        # add the others connections to the main node, and delete from network managers
                        self.network = nx.contracted_nodes(self.network, main_node, node, self_loops=False)
                        # as default it adds data from the node to the main one under "contraction", we don't want it
                        del self.network.nodes[main_node]["contraction"]
                        del self.network_managers[node]
        return added_nodes

    def add_node_label(self, node_id, attr_name):
        """Set an attribute for a given node."""
        assert node_id in self.network, "Node does not exist."
        self.network.nodes[node_id]["labels"].add(attr_name)

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
            del self.network_managers[node]
            self.network.remove_node(node)
            if images_path:
                img_path = self.make_image_path(node, images_path)
                if os.path.exists(img_path):
                    os.remove(img_path)

    def draw_network(self):
        """Draw the network."""
        pos = nx.spring_layout(self.network)
        nx.draw(self.network, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(self.network, "weight")
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=edge_labels)
        plt.show()

    def group_identical(self, nodes: List[str]):
        """
        Group nodes that have the same network manager.
        Returns a list of lists, where each inner list contains the nodes that belong to the same group.
        """
        groups = []
        for node in nodes:
            # Check if the object belongs to an existing group
            for group in groups:
                if (
                    self.network_managers[node] == self.network_managers[group[0]]
                ):  # Compare with a representative of the group
                    group.append(node)
                    break
            else:
                # If no group matches, create a new group
                groups.append([node])
        return groups

    def save(self, overwrite=False):
        """Save the current state of the network and network managers."""
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
            json.dump({k: v.to_dict() for k, v in self.network_managers.items()}, f)

    @staticmethod
    def load(folder, load_networks_managers=True):
        """Load the most recent state of the network and network managers."""
        files = sorted(os.listdir(folder), reverse=True)
        network_file = next(f for f in files if f.startswith("network_"))
        managers_file = next(f for f in files if f.startswith("managers_"))
        db_manager = DBManager(folder)
        with open(os.path.join(folder, network_file), "r") as f:
            network_data = json.load(f)
        db_manager.cluster_starts = set(network_data["cluster_starts"])
        db_manager.max_id = network_data["max_id"]
        network = nx.node_link_graph(network_data["network"], edges="edges")
        for _, node_data in network.nodes(data=True):
            # convert back from list to set (saved as list for json)
            node_data["labels"] = set(node_data["labels"])
        for _, _, edge_data in network.edges(data=True):
            edge_data["step"] = deep_freeze(edge_data["step"])
        db_manager.network = network
        if load_networks_managers:
            with open(os.path.join(folder, managers_file), "r") as f:
                db_manager.network_managers = {k: NetworkManager.from_dict(v) for k, v in json.load(f).items()}
        return db_manager

    def _generate_unique_id(self):
        """Generate a unique ID for a new node."""
        self.max_id += 1
        return str(self.max_id)

    @staticmethod
    def _apply_variation(network_manager: NetworkManager, variation: VariationDescriptor):
        """Apply a variation to a NetworkManager instance and return a new instance."""
        new_network_manager = network_manager.copy()
        apply_variation(new_network_manager, variation)
        return new_network_manager

    @staticmethod
    def make_image_path(node_id, images_path):
        return os.path.join(images_path, f"{node_id}.png")

    def generate_images(self, images_path, override_images=False, save_every=None):
        empty_count = 0
        failed = []
        images_to_generate = self.get_nodes_without_label(HAS_IMAGE)
        now = time.time()
        for i, node_id in enumerate(images_to_generate, start=1):
            clear_output(wait=True)
            print(f"working on image {i}/{len(images_to_generate)}")
            nm = self.network_managers[node_id]
            img_path = self.make_image_path(node_id, images_path)
            if not override_images:
                assert not os.path.exists(img_path), "Image already exists!"
            try:
                generate_image(nm, img_path)
                assert os.path.exists(img_path)
                self.add_node_label(node_id, HAS_IMAGE)
                is_empty = bool(is_empty_image(img_path))  # bool to convert from np.bool_
                if is_empty:
                    self.add_node_label(node_id, IS_EMPTY_IMAGE)
                    empty_count += 1
            except Exception as e:
                print('Failed!')
                failed.append((node_id, e))
            if save_every is not None and i % save_every == 0:
                print('Saving...')
                self.save()
        time_required = time.time() - now
        print(
            f"Generated {len(images_to_generate)} images, including {empty_count} empty images, "
            f"in {round(time_required, 2)} seconds, failed on: {len(failed)}"
        )
        return failed

    def connect_new_node_to_existing_connections(
        self, new_node_id, main_node_id, new_connection: TwoWayVariationDescriptor
    ):
        """
        This function can be used in the case when we just created a Seed or Param change on a node.
        If that node already had param change variations - so it is connected to other nodes with an edge type of
        Seed or Param - then we can connect the new node to all the old nodes of the same type
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
        new_connections_count = 0
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
            new_connections_count += 1
        # if new_connections_count > 0:
        #     print(f"Created {new_connections_count} new connections between existing nodes")


def change_seed(nm, n_changes=5, **kwargs):
    return non_structural_changes(nm, n_changes, ParamRequestType.SEED, **kwargs)


def change_numeric(nm, n_changes=4, **kwargs):
    return non_structural_changes(nm, n_changes, ParamRequestType.NUMERIC, **kwargs)


def change_params(nm, n_changes=3, **kwargs):
    return non_structural_changes(nm, n_changes, ParamRequestType.NON_SEED, **kwargs)


def completely_random_generation(n_additions=6, n_change_params=5, **kwargs):
    nm = NetworkManager()
    nm.initialize_network()
    nm.generate_random_network(n_additions=n_additions)
    nm.finish_network()
    params_change = change_params(nm, n_changes=n_change_params)
    apply_variation(nm, params_change.steps_forward[0])
    return nm


def regular_meta_nodes(max_layers=2, n_additions=3, n_change_params=5):
    manager = MetaNetworkManager(ALL_META_NODES, max_layers=max_layers, n_additions=n_additions)
    manager.generate_network()
    nm = manager.meta_network_to_flat_network()
    params_change = change_params(nm, n_changes=n_change_params)
    apply_variation(nm, params_change.steps_forward[0])
    return nm


def mega_nodes(max_layers=1, n_additions=1, n_change_params=5):
    manager = MetaNetworkManager(MEGA_STRUCTURES, max_layers=max_layers, n_additions=n_additions)
    manager.generate_network()
    nm = manager.meta_network_to_flat_network()
    params_change = change_params(nm, n_changes=n_change_params)
    apply_variation(nm, params_change.steps_forward[0])
    return nm


def make_cluster_base(make_initial_func, kwargs):
    cluster_base = make_initial_func(**kwargs)
    for i in range(10):  # just to avoid while True
        if not check_nm_not_empty(cluster_base):
            return cluster_base
        cluster_base = make_initial_func(**kwargs)
    return cluster_base  # return an empty one if somehow it didn't work after 10 attempts


def make_cluster(db_manager: DBManager, cluster_func=regular_meta_nodes, concat_param_change=True, cluster_kwargs=None):
    if cluster_kwargs is None:
        cluster_kwargs = {}
    generation_label = base_func_type_label[cluster_func]
    cluster_base = make_cluster_base(cluster_func, cluster_kwargs)
    new_cluster_id = db_manager.add_cluster(cluster_base, labels_set=frozenset({generation_label}))
    empty_network_variation = to_nothing_variation(cluster_base, concat_param_change=concat_param_change)
    new_nodes = db_manager.add_sequence(
        new_cluster_id, empty_network_variation, node_labels=frozenset({ON_PATH_TO_EMPTY}), contract=True
    )
    return new_nodes


def make_variations(db_manager: DBManager, selected_node, variation_func, **kwargs):
    nm = db_manager.network_managers[selected_node]
    two_way_variations: TwoWayVariationDescriptor = variation_func(nm, **kwargs)

    if two_way_variations is None or len(two_way_variations.steps_forward[0].step) == 0:
        fail_msg = f"Attempted to create variation: {variation_func.__name__} on {selected_node} but no variation was created"
        return fail_msg

    new_nodes = db_manager.add_sequence(selected_node, two_way_variations)

    # connect to other nodes that are connected to the same node
    non_structural_variation_types = [VariationType.SEED, VariationType.NUMERIC, VariationType.CAT_AND_NUMERIC]
    if (
        len(two_way_variations.steps_backward) == 1
        and two_way_variations.steps_backward[0].variation_type in non_structural_variation_types
    ):
        assert len(new_nodes) == 1, "If steps backwards is 1 there should only have been one new node"
        db_manager.connect_new_node_to_existing_connections(new_nodes[0], selected_node, two_way_variations)


base_func_type_label = {
    completely_random_generation: RANDOM_GENERATION,
    regular_meta_nodes: META_NODES_GENERATION,
    mega_nodes: MEGA_NODES_GENERATION,
}
