import bpy
from bpy.types import bpy_prop_array
import mathutils


def get_shader_node_by_name(name):
    return bpy.data.node_groups.new("TempGroup", "ShaderNodeTree").to_nodes.new(
        "ShaderNode" + name
    )


unnecessary_props = {
    "location",
    "width",
    "height",
    "warning_propagation",
    "use_custom_color",
    "show_texture",
    "color",
    "select",
    "show_options",
    "show_preview",
    "hide",
    "mute",
}
vector_type = (bpy_prop_array, mathutils.Vector, mathutils.Euler)
relevant_nodes = [
    "CombineXYZ",
    "Math",
    "Mix",
    "MixRGB",
    "SeparateXYZ",
    "Mapping",
    "TexBrick",
    "TexChecker",
    "TexCoord",
    "TexGabor",
    "TexGradient",
    "TexNoise",
    "TexVoronoi",
    "TexWave",
    "ValToRGB",
    "Value",
    "VectorMath",
]


def get_shader_node_data():
    # Initialize the dictionary to hold all shader nodes and their properties
    shader_nodes_data = {}

    # Get all node types
    for node in dir(bpy.types):
        if node.startswith("ShaderNode"):  # Filter only shader nodes
            node_class = getattr(bpy.types, node)
            node_name = node.replace("ShaderNode", "")
            if node_name not in relevant_nodes:
                continue

            # Temporary node instance to inspect its properties
            try:
                temp_node = bpy.data.node_groups.new(
                    "TempGroup", "ShaderNodeTree"
                ).to_nodes.new(node)
            except:
                continue

            params = {}
            # Iterate over all node properties
            for prop_name, prop in temp_node.bl_rna.properties.items():
                if prop_name.startswith("bl_") or prop_name in unnecessary_props:
                    continue
                if not prop.is_readonly:  # Skip readonly properties
                    if prop.type == "ENUM":  # Categorical options
                        params[prop_name] = {
                            "type": "enum",
                            "options": [item.identifier for item in prop.enum_items],
                            "default": getattr(temp_node, prop_name),
                        }
                    elif prop.type in {"INT", "FLOAT"}:  # Numeric options
                        params[prop_name] = {
                            "type": prop.type.lower(),
                            "default": getattr(temp_node, prop_name),
                            "min": prop.hard_min,
                            "max": prop.hard_max,
                        }
                    elif prop.type == "BOOLEAN":  # Boolean
                        params[prop_name] = {
                            "type": "boolean",
                            "default": getattr(temp_node, prop_name),
                        }
            if node_name == "ValToRGB":
                for i, element in enumerate(temp_node.color_ramp.elements):
                    # default is 0 for first element and 1 for the second, we will only have these two
                    params[f"element_{i}"] = {
                        "type": "float",
                        "default": i,
                        "min": 0,
                        "max": 1,
                    }

            inputs = []
            # Iterate over node inputs
            for i, input_socket in enumerate(temp_node.inputs):
                input_name = input_socket.name
                if hasattr(input_socket, "default_value"):
                    default_value = input_socket.default_value
                    vals = {"name": input_name, "input_id": i}
                    if isinstance(default_value, float):  # Float values
                        vals.update({"type": "float", "default": default_value})
                    elif isinstance(default_value, int):  # Integer values
                        vals.update({"type": "int", "default": default_value})
                    elif isinstance(
                        default_value, vector_type
                    ):  # Vector or Color values
                        vals.update({"type": "vector", "default": tuple(default_value)})
                    inputs.append(vals)

            outputs = {}
            for out_socket in temp_node.outputs:
                if isinstance(default_value, int):
                    socket_type = "int"
                elif isinstance(default_value, vector_type):
                    socket_type = "vector"
                else:
                    socket_type = out_socket.name
                outputs[out_socket.name] = {"type": socket_type}

            # Add node data to the shader_nodes_data dictionary
            shader_nodes_data[node_name] = {
                "inputs": inputs,
                "outputs": outputs,
                "params": params,
            }
            # Remove the temporary node
            bpy.data.node_groups.remove(temp_node.id_data)
    return shader_nodes_data


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_shader_node_data())
