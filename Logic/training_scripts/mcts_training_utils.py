import torch
import networkx as nx

from Logic.constants import MAIN_HEAD_LABEL_TO_ID_MAP, SECONDARY_HEAD_LABEL_TO_ID_MAP
from Logic.mcts_operator import MCTSOperator
from Logic.tree_networks_manager import TEXTURE_SIMILARITY_VALUE, ACTION_PREDICTED_VALUE, CODE_PREDICTED_VALUE, \
    TreesNetworkManager
from Logic.tokens_to_labels_and_back import edge_to_example


def filter_nodes_by_distance(G: nx.DiGraph, root, min_depth):
    # Get the shortest path lengths from root to all reachable nodes
    path_lengths = nx.single_source_shortest_path_length(G, root)

    # Identify target nodes that are at least distance k from the root
    distant_nodes = {node for node, dist in path_lengths.items() if dist >= min_depth}

    # Find all ancestors (including the nodes themselves) of distant nodes
    # These are the ones we want to keep
    nodes_to_keep = set()
    for node in distant_nodes:
        try:
            path = nx.shortest_path(G, source=root, target=node)
            nodes_to_keep.update(path)
        except nx.NetworkXNoPath:
            continue

    # get only those nodes that have a branch out of them that gets deep enough
    return nodes_to_keep


def texture_to_reward(texture_value):
    # power to make the lower values closer to zero (we really want to reward for 0.7-1)
    return max(0, texture_value) ** 2


def node_to_target_value(graph_manager: TreesNetworkManager, node_id: str, reward_prediction_weight: float):
    node_descendants = nx.descendants(graph_manager.network, node_id) | {node_id}
    descendants_values = [graph_manager.get_node_value(n, TEXTURE_SIMILARITY_VALUE, 0) for n in node_descendants]
    best_value_on_path = max(descendants_values)
    best_value_on_path = texture_to_reward(best_value_on_path)
    current_prediction = graph_manager.get_node_value(node_id, CODE_PREDICTED_VALUE, 0)
    combined_value = reward_prediction_weight * best_value_on_path + (1 - reward_prediction_weight) * current_prediction
    return combined_value


def node_to_mcts_training_example(
    graph_manager: TreesNetworkManager,
    node_id: str,
    current_dir: str,
    target_img_path: str,
    mcts_operator: MCTSOperator,
    add_cur_image=True,
    reward_prediction_weight=0.8,
):
    assert graph_manager.network.out_degree(node_id) > 0, "Should only have nodes with outgoing edges"
    target_value = node_to_target_value(graph_manager, node_id, reward_prediction_weight=reward_prediction_weight)

    code = graph_manager.blender_tree_managers[node_id].to_str(
        with_seeds=False, add_image_tokens=True, with_cur_image=add_cur_image
    )

    best_value = -100
    best_action = None
    for _, neighbor, edge_data in graph_manager.network.out_edges(node_id, data=True):
        # with numbers mode it just takes that value and does not try to convert to increase/decrease, name is confusing
        # no need to convert to increase/decrease here because it is already converted
        example_data = edge_to_example(edge_data, code, mcts_operator.tokenizer, numbers_mode=True)
        if example_data is None:
            continue
        del example_data["attention_mask"]
        value = graph_manager.get_node_value(neighbor, CODE_PREDICTED_VALUE, 0)
        # TODO: fallback to action predicted if there is no code predicted value for anyone
        if value > best_value:
            best_value = value
            best_action = example_data["main_head_ids"], example_data["secondary_head_ids"]

    main_head_labels, secondary_head_labels = best_action

    node_image_path = graph_manager.make_image_path(node_id, current_dir)

    final_example_data = {
        "input_ids": example_data["input_ids"],
        "main_head_labels": main_head_labels,
        "secondary_head_labels": secondary_head_labels,
        "target_value": target_value,
        "target_img_path": target_img_path,
        "node_image_path": node_image_path,
        "add_cur_image": add_cur_image,
    }
    return final_example_data
