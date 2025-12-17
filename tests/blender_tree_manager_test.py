import pytest
import numpy as np

from Logic.blender_tree_manager import BlenderTreeManager
from Logic.node_readers_writers import ParamRequestType  # for value queries
from Logic.constants import INCREASE, DECREASE, NO_ACTION_LABEL, NODES_EDGES_SEP

# ...existing code...

def _create_random_manager(seed=0, n_additions=5):
    """Helper to create a deterministic random manager with a few nodes."""
    np.random.seed(seed)
    manager = BlenderTreeManager()
    manager.initialize_network()
    manager.generate_random_tree(n_additions=n_additions)
    return manager


def test_copy_preserves_network():
    manager = _create_random_manager(seed=1, n_additions=6)
    mgr_copy = manager.copy()
    assert manager == mgr_copy


def test_to_dict_and_from_dict_roundtrip():
    manager = _create_random_manager(seed=2, n_additions=6)
    data = manager.to_dict()
    restored = BlenderTreeManager.from_dict(data)
    assert BlenderTreeManager.compare_networks(manager, restored, compare_node_properties=True, tuple_list_hack=True)


def test_to_str_and_from_str_roundtrip():
    manager = _create_random_manager(seed=3, n_additions=6)
    s = manager.to_str(with_seeds=True)
    restored = BlenderTreeManager.from_str(s)
    assert BlenderTreeManager.compare_networks(manager, restored, compare_node_properties=True, tuple_list_hack=True)


def test_to_str_with_image_tokens_includes_tokens():
    manager = _create_random_manager(seed=4, n_additions=3)
    s_with_tokens = manager.to_str(with_seeds=False, add_image_tokens=True, with_cur_image=False)
    # string should contain the image token sequence defined by BlenderTreeManager (A N I N)
    assert manager.WITHOUT_IMAGE_TOKENS in s_with_tokens


def test_calculate_true_free_inputs_matches_internal_free_inputs():
    manager = _create_random_manager(seed=5, n_additions=8)
    true_free = manager.calculate_true_free_inputs()
    # both are dicts mapping node -> set of free inputs
    assert {k: set(v) for k, v in manager.free_inputs.items()} == true_free


def test_rename_node_updates_free_inputs_key():
    manager = BlenderTreeManager()
    manager.initialize_network()
    # add a single node to be renamed
    node_name = manager.add_node_by_type_name(next(iter(manager.NODE_TYPES_FOR_GENERATION)))
    # ensure it's present in free_inputs
    assert node_name in manager.free_inputs
    new_name = node_name + "_new"
    manager.rename_node(node_name, new_name)
    assert new_name in manager.free_inputs
    assert new_name in manager.network.nodes


def test_set_nodes_attributes_increase_scalar_and_vector():
    manager = BlenderTreeManager()
    manager.initialize_network()
    # add a node type that typically has numeric parameters (choose any available node type)
    node_type_name = next(iter(manager.NODE_TYPES))  # safe generic pick
    # avoid picking Input/Output as they are special; ensure we pick a real param-bearing node
    if node_type_name in {manager.OutputNodeNAME, manager.InputNodeNAME}:
        node_type_name = "Value" if "Value" in manager.NODE_TYPES else next(n for n in manager.NODE_TYPES if n not in {manager.OutputNodeNAME, manager.InputNodeNAME})
    node_name = manager.add_node_by_type_name(node_type_name)

    # find at least one numeric attribute currently on the node (scalar or vector)
    attrs = manager.network.nodes[node_name]
    numeric_attr = None
    is_vector = False
    for k, v in attrs.items():
        if k in {"layer", manager.node_value_ranges_name}:
            continue
        if isinstance(v, (int, float)):
            numeric_attr = k
            is_vector = False
            break
        if isinstance(v, (list, tuple)):
            # ensure it's numeric vector
            if all(isinstance(x, (int, float)) for x in v):
                numeric_attr = k
                is_vector = True
                break

    if numeric_attr is None:
        pytest.skip("No numeric attribute available on the chosen node to test increases/decreases")

    old_val = manager.network.nodes[node_name][numeric_attr]
    # Test INCREASE for scalar or vector
    if not is_vector:
        manager.set_nodes_attributes({node_name: {numeric_attr: INCREASE}})
        new_val = manager.network.nodes[node_name][numeric_attr]
        assert isinstance(new_val, (int, float))
        assert new_val != old_val
    else:
        # build a vector of INCREASE directives matching length
        directive = tuple([INCREASE] * len(old_val))
        manager.set_nodes_attributes({node_name: {numeric_attr: directive}})
        new_val = manager.network.nodes[node_name][numeric_attr]
        assert isinstance(new_val, tuple)
        assert any(a != b for a, b in zip(old_val, new_val))


def test_set_nodes_attributes_decrease_and_no_action():
    manager = BlenderTreeManager()
    manager.initialize_network()
    node_name = manager.add_node_by_type_name("Value") if "Value" in manager.NODE_TYPES else manager.add_node_by_type_name(next(iter(manager.NODE_TYPES)))
    attrs = manager.network.nodes[node_name]
    # pick any numeric scalar attribute
    numeric_attr = None
    for k, v in attrs.items():
        if isinstance(v, (int, float)):
            numeric_attr = k
            break
    if numeric_attr is None:
        pytest.skip("No scalar numeric attribute to test decrease/no-action")

    old = manager.network.nodes[node_name][numeric_attr]
    manager.set_nodes_attributes({node_name: {numeric_attr: NO_ACTION_LABEL}})
    assert manager.network.nodes[node_name][numeric_attr] == old
    manager.set_nodes_attributes({node_name: {numeric_attr: DECREASE}})
    assert manager.network.nodes[node_name][numeric_attr] != old
