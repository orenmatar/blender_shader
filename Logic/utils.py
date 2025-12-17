import json
import math
import os
import random
import time
from types import MappingProxyType
from typing import Any

from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def sample_uniform(low=0, high=10, size=1):
    return np.random.uniform(low=low, high=high, size=size)


def sample_log_scale(low=0, high=10, power=3, size=1):
    x = np.random.uniform(0, 1, size=size)
    return low + (high - low) * (x**power)


def save_graph(graph, filename):
    with open(filename, "w") as f:
        json.dump(nx.node_link_data(graph, edges="edges"), f)


def load_graph(filename):
    with open(filename, "r") as f:
        return nx.node_link_graph(json.load(f), edges="edges")


def normalize(obj):
    """
    Recursively normalize a structure by converting tuples to lists.
    """
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize(v) for v in obj]
    elif isinstance(obj, tuple):
        return [normalize(v) for v in obj]  # Convert tuple to list
    else:
        return obj  # Return the object as is for other types


def compare_dicts(d1, d2):
    """
    Compare two dictionaries, ignoring list/tuple differences.
    """
    return normalize(d1) == normalize(d2)


def deep_freeze(obj: Any) -> Any:
    """
    Recursively converts a dictionary (and its nested structures) into immutable forms.
    - Dicts are wrapped in MappingProxyType.
    - Lists are converted to tuples.
    - Other mutable types can be handled as needed.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: deep_freeze(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return tuple(deep_freeze(v) for v in obj)
    elif isinstance(obj, set):
        return frozenset(deep_freeze(v) for v in obj)
    # Add more cases here if necessary
    else:
        return obj  # Immutable types (e.g., int, float, str, tuple) are returned as-is


def deep_unfreeze(obj: Any) -> Any:
    """
    Recursively converts immutable objects (like MappingProxyType, tuple, frozenset)
    into their mutable counterparts (dict, list, set).
    """
    if isinstance(obj, (MappingProxyType, dict)):
        return {k: deep_unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, (tuple, list)):
        return [deep_unfreeze(v) for v in obj]
    elif isinstance(obj, (frozenset, set)):
        return {deep_unfreeze(v) for v in obj}
    else:
        return obj  # Immutable types (e.g., int, float, str) are returned as-is


def show_image_grid(image_data):
    """
    Display a grid of up to 4x4 grayscale images with text above each image.

    Parameters:
        image_data (list of tuples): A list of tuples where each tuple contains:
            - image_path (str): Path to the image file.
            - title (str): Text to display above the image.
    """
    # Limit to the first 16 images
    image_data = image_data[:16]

    # Calculate the grid size (smallest square that fits all images)
    num_images = len(image_data)
    grid_size = math.ceil(num_images**0.5)  # Smallest square grid

    # Create the figure and axes
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()  # Flatten for easy iteration

    # Loop through the images and add them to the grid
    for i, ax in enumerate(axes):
        if i < num_images:
            image_path, title = image_data[i]
            img = mpimg.imread(image_path)
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10, color="black")
        else:
            # Leave unused slots empty with no image or title
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def is_empty_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array.std() < 1


def lc(iterable, print_every=1, clear=True):
    t = time.time()
    for index, item in enumerate(iterable, start=1):
        if index % print_every == 0:
            if clear:
                from IPython.core.display_functions import clear_output

                clear_output(wait=True)
            print(f"{index}/{len(iterable)} in {round(time.time() - t, 2)} seconds")
        yield item


def force_mutable(mappingproxy_obj, key, value):
    """
    This function takes a mappingproxy object, makes it mutable (as a dictionary),
    and then converts it back to a mappingproxy object after modifications.
    """
    mutable_dict = dict(mappingproxy_obj)
    mutable_dict[key] = value
    return MappingProxyType(mutable_dict)


def force_mutable_key_change(mappingproxy_obj, key, new_key_name):
    mutable_dict = dict(mappingproxy_obj)
    mutable_dict[new_key_name] = mutable_dict[key]
    del mutable_dict[key]
    return MappingProxyType(mutable_dict)


def is_uuid_name(node_name):
    return len(node_name) > 25


def are_identical_images(path1, path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    # Convert images to numpy arrays for easy comparison
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # Compare the shape and content of the images
    return np.array_equal(img1_array, img2_array)


def custom_shortest_paths(network, node1, cutoff=12, edges_to_ignore=("SEED",)):
    """
    Finds shortest distance to node for all nodes in the cluster but not counting SEED as distance
    """
    distances = {node1: 0}  # Store distances from node1
    visited = set()

    def dfs(node, current_distance):
        if current_distance > cutoff:
            return

        for neighbor, edge_data in network[node].items():
            step = 0 if edge_data.get("variation_type") in edges_to_ignore else 1
            new_distance = current_distance + step

            # Update if the node hasn't been visited or we found a shorter path
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                dfs(neighbor, new_distance)

    dfs(node1, 0)
    return distances


def edge_weight_function(from_node, to_node, attributes):
    return attributes["variation_type"] != "SEED"


def find_edge_for_target_label(db_manager, from_node, target_label=None, target_distance=None, cutoff=25):
    distances = custom_shortest_paths(db_manager.network, from_node, cutoff=cutoff)
    legit_nodes = list(distances)
    if target_label is not None:
        legit_nodes = [node for node in legit_nodes if db_manager.node_has_label(node, target_label)]
    if target_distance is not None:
        legit_nodes = [node for node in legit_nodes if distances[node] == target_distance]
    if len(legit_nodes) == 0:
        return None, None, None
    target_node = random.choice(legit_nodes)
    distance = distances[target_node]
    neighbors = list(db_manager.network.neighbors(from_node))
    random.shuffle(neighbors)
    for neighbor in neighbors:
        neigh_distance = nx.shortest_path_length(
            db_manager.network, source=neighbor, target=target_node, weight=edge_weight_function
        )
        if neigh_distance == distance - 1 and db_manager.network[from_node][neighbor]["variation_type"] != "SEED":
            return distance, target_node, neighbor  # This edge leads one step closer to the target

    return None, None, None  # If no such edge is found


def create_unique_subdir(base_dir):
    os.makedirs(base_dir, exist_ok=True)  # Ensure base directory exists

    existing = [d for d in os.listdir(base_dir) if d.isdigit()]
    existing_nums = sorted([int(d) for d in existing])

    next_num = 1
    if existing_nums:
        next_num = existing_nums[-1] + 1

    new_dir = os.path.join(base_dir, str(next_num))
    os.makedirs(new_dir)

    return new_dir
