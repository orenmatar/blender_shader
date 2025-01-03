import json
from typing import Dict

import networkx as nx
import os
from datetime import datetime
import matplotlib.pyplot as plt

from Logic.bpy_connector import check_nm_not_empty
from Logic.meta_network import MetaNetworkManager
from Logic.network_manager import NetworkManager
from Logic.node_readers_writers import ParamRequestType
from Logic.structures_definitions import ALL_META_NODES, MEGA_STRUCTURES
from Logic.variations_creator import (
    TwoWayVariationDescriptor,
    apply_variation,
    VariationDescriptor,
    non_structural_changes,
)


class DBManager:
    def __init__(self):
        self.network = nx.DiGraph()
        self.network_managers: Dict[int, NetworkManager] = {}
        self.cluster_starts = set()

    def _add_node(self, node_id, network_manager, **attr):
        """Add a new node to the network."""
        assert node_id not in self.network, "Node already exists."
        assert node_id not in self.network_managers, "Node already exists."
        self.network.add_node(node_id, **attr)
        self.network_managers[node_id] = network_manager

    def add_cluster(self, network_manager: NetworkManager):
        """Add a new cluster with a given NetworkManager instance."""
        node_id = self._generate_unique_id()
        self._add_node(node_id, network_manager, has_image=False, is_cluster_start=True)
        self.cluster_starts.add(node_id)
        return node_id

    def _add_edge(self, source, target, variation_descriptor: VariationDescriptor):
        """Add a new edge to the network."""
        assert source in self.network, "Source node does not exist."
        assert target in self.network, "Target node does not exist."
        variation_type = variation_descriptor.variation_type.name  # convert from enum to string
        self.network.add_edge(source, target, variation_type=variation_type, step=variation_descriptor.step)

    def add_step(self, source_node_id, variation_descriptor: VariationDescriptor):
        """Add a new step to the network."""
        new_node_id = self._generate_unique_id()
        new_network_manager = self._apply_variation(self.network_managers[source_node_id], variation_descriptor)
        self._add_node(new_node_id, new_network_manager, has_image=False, is_cluster_start=False)
        self._add_edge(source_node_id, new_node_id, variation_descriptor)
        return new_node_id

    def add_sequence(self, start_node_id, two_way_variation_descriptor: TwoWayVariationDescriptor):
        """Add a sequence of nodes starting from a given node."""
        assert start_node_id in self.network, "Start node does not exist."
        current_node_id = start_node_id
        for step in two_way_variation_descriptor.steps_forward:
            current_node_id = self.add_step(current_node_id, step)

        # Add the backward steps, except for the last one - which will connect back to the start node
        for step in two_way_variation_descriptor.steps_backward[:-1]:
            current_node_id = self.add_step(current_node_id, step)

        final_variation = two_way_variation_descriptor.steps_backward[-1]
        self._add_edge(current_node_id, start_node_id, final_variation)

    def set_node_attribute(self, node_id, attr_name, attr_value):
        """Set an attribute for a given node."""
        assert node_id in self.network, "Node does not exist."
        self.network.nodes[node_id][attr_name] = attr_value

    def get_nodes_with_attribute(self, attr_name, attr_value):
        """Get all nodes that have a certain attribute."""
        return [node for node, attrs in self.network.nodes(data=True) if attrs.get(attr_name) == attr_value]

    def draw_network(self):
        """Draw the network."""
        pos = nx.spring_layout(self.network)
        nx.draw(self.network, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(self.network, "weight")
        nx.draw_networkx_edge_labels(self.network, pos, edge_labels=edge_labels)
        plt.show()

    def save(self, folder, overwrite=False):
        """Save the current state of the network and network managers."""
        timestamp = datetime.now().strftime("%Y%m%d%H")
        network_file = os.path.join(folder, f"network_{timestamp}.json")
        managers_file = os.path.join(folder, f"managers_{timestamp}.json")
        if not overwrite:
            assert not os.path.exists(network_file), "Network file already exists."
            assert not os.path.exists(managers_file), "Managers file already exists."
        network = nx.node_link_data(self.network, edges="edges")
        network_data = {"network": network, "cluster_starts": list(self.cluster_starts)}
        with open(network_file, "w") as f:
            json.dump(network_data, f)
        with open(managers_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.network_managers.items()}, f)

    @staticmethod
    def load(folder):
        """Load the most recent state of the network and network managers."""
        files = sorted(os.listdir(folder), reverse=True)
        network_file = next(f for f in files if f.startswith("network_"))
        managers_file = next(f for f in files if f.startswith("managers_"))
        db_manager = DBManager()
        with open(os.path.join(folder, network_file), "r") as f:
            network_data = json.load(f)
        db_manager.cluster_starts = set(network_data["cluster_starts"])
        db_manager.network = nx.node_link_graph(network_data["network"], edges="edges")
        with open(os.path.join(folder, managers_file), "r") as f:
            db_manager.network_managers = {k: NetworkManager.from_dict(v) for k, v in json.load(f).items()}
        return db_manager

    def _generate_unique_id(self):
        """Generate a unique ID for a new node."""
        return str(len(self.network.nodes))

    @staticmethod
    def _apply_variation(network_manager: NetworkManager, variation: VariationDescriptor):
        """Apply a variation to a NetworkManager instance and return a new instance."""
        new_network_manager = network_manager.copy()
        apply_variation(new_network_manager, variation)
        return new_network_manager


def change_seed(nm, n_changes=3):
    return non_structural_changes(nm, n_changes, ParamRequestType.SEED)


def change_numeric(nm, n_changes=3):
    return non_structural_changes(nm, n_changes, ParamRequestType.NUMERIC)


def change_params(nm, n_changes=3):
    return non_structural_changes(nm, n_changes, ParamRequestType.NON_SEED)


def completely_random_generation(n_additions=6):
    nm = NetworkManager()
    nm.initialize_network()
    nm.generate_random_network(n_additions=n_additions)
    nm.finish_network()
    params_change = change_params(nm, n_changes=4)
    apply_variation(nm, params_change.steps_forward[0])
    return nm


def regular_meta_nodes(max_layers=2, n_additions=3):
    manager = MetaNetworkManager(ALL_META_NODES, max_layers=max_layers, n_additions=n_additions)
    manager.generate_network()
    nm = manager.meta_network_to_flat_network()
    params_change = change_params(nm, n_changes=4)
    apply_variation(nm, params_change.steps_forward[0])
    return nm


def mega_nodes(max_layers=1, n_additions=1):
    manager = MetaNetworkManager(MEGA_STRUCTURES, max_layers=max_layers, n_additions=n_additions)
    manager.generate_network()
    nm = manager.meta_network_to_flat_network()
    params_change = change_params(nm, n_changes=4)
    apply_variation(nm, params_change.steps_forward[0])
    return nm


def make_cluster_base(make_initial_func, kwargs):
    cluster_base = make_initial_func(**kwargs)
    for i in range(10):  # just to avoid while True
        if not check_nm_not_empty(cluster_base):
            return cluster_base
        cluster_base = make_initial_func(**kwargs)
    return cluster_base  # return an empty one if somehow it didn't work after 10 attempts
