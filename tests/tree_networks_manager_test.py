import pytest
import numpy as np

from Logic.blender_tree_manager import BlenderTreeManager
from Logic.tree_networks_manager import TreesNetworkManager, completely_random_generation
from Logic.variations_creator import (
    add_random_node_on_edge,
    add_random_edge,
    remove_random_edge,
    remove_random_node,
    VariationDescriptor,
    TwoWayVariationDescriptor,
    VariationType,
)


def _create_simple_tnm(seed=0, n_additions=6) -> tuple[TreesNetworkManager, str, BlenderTreeManager]:
    """
    Create a simple TreesNetworkManager with a single cluster base generated deterministically.
    Returns (tnm, node_id, base_btm)
    """
    np.random.seed(seed)
    base_btm = completely_random_generation(n_additions=n_additions, n_change_params=10)
    tnm = TreesNetworkManager(folder=None)
    node_id = tnm.add_cluster(base_btm, labels_set=frozenset({"test_cluster"}))
    return tnm, node_id, base_btm


def test_add_cluster_creates_node_and_registers_manager():
    tnm, node_id, base = _create_simple_tnm(seed=1, n_additions=4)
    assert node_id in tnm.network
    assert node_id in tnm.cluster_starts
    assert node_id in tnm.blender_tree_managers
    # the stored manager should be equal (structurally) to the one we passed
    assert tnm.blender_tree_managers[node_id] == base


def test_add_sequence_with_structural_variation_adds_nodes():
    # deterministic pick of structural changes - ensure we test them all
    structural_changes = [add_random_node_on_edge, add_random_edge, remove_random_edge, remove_random_node]
    np.random.seed(2)
    tnm, node_id, base = _create_simple_tnm(seed=2, n_additions=6)

    # For each structural change function, try multiple deterministic attempts until one produces a variation.
    for i, func in enumerate(structural_changes):
        succeeded = False
        for attempt in range(10):  # try a few different random seeds/attempts
            candidate = base.copy()
            np.random.seed(1000 + i * 10 + attempt)
            maybe = func(candidate)
            if maybe is None:
                continue
            # if we got a variation, add the sequence and verify nodes were created/registered
            added = tnm.add_sequence(node_id, maybe, node_labels=frozenset({"struct_test"}), contract=False)
            assert isinstance(added, list)
            assert len(added) > 0
            for new_node in added:
                assert new_node in tnm.network
                assert new_node in tnm.blender_tree_managers
            succeeded = True
            break
        assert succeeded, f"{func.__name__} failed to produce a variation after multiple attempts"


def test_connect_existing_nodes_creates_bidirectional_edges():
    # create two separate cluster bases
    tnm1, node1, base1 = _create_simple_tnm(seed=3, n_additions=5)
    tnm = tnm1  # single manager for tests
    tnm2_base = completely_random_generation(n_additions=4, n_change_params=10)
    node2 = tnm._generate_unique_id()
    tnm._add_node(node2, tnm2_base, labels_set=frozenset({"second"}))

    # prepare a simple TwoWayVariationDescriptor with single-step forward/backwards
    forward = VariationDescriptor(VariationType.SEED, {"dummy": "x"})
    backward = VariationDescriptor(VariationType.SEED, {"dummy": "y"})
    two_way = TwoWayVariationDescriptor(steps_forward=[forward], steps_backward=[backward])

    # connect the nodes; should add edges in both directions
    tnm.connect_existing_nodes(node1, node2, two_way)
    assert tnm.network.has_edge(node1, node2)
    assert tnm.network.has_edge(node2, node1)
    assert tnm.network.edges[node1, node2]["variation_type"] == VariationType.SEED.name
    assert tnm.network.edges[node2, node1]["variation_type"] == VariationType.SEED.name


def test_group_identical_finds_duplicate_managers():
    tnm, node_id, base = _create_simple_tnm(seed=4, n_additions=4)
    # create another node with an identical BlenderTreeManager (copy of base)
    another_id = tnm._generate_unique_id()
    duplicate_btm = base.copy()
    tnm._add_node(another_id, duplicate_btm, labels_set=frozenset({"dup"}))

    groups = tnm._group_identical([node_id, another_id])
    # both nodes should be grouped together
    assert any(set(group) == {node_id, another_id} for group in groups)
