import json
import networkx as nx
from PIL import Image
import numpy as np


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


def is_empty_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array.std() < 1


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
