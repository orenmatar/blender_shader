import json
import math
from types import MappingProxyType
from typing import Any

from IPython.core.display_functions import clear_output
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
    for index, item in enumerate(iterable, start=1):
        if index % print_every == 0:
            if clear:
                clear_output(wait=True)
            print(f"{index}/{len(iterable)}")
        yield item


