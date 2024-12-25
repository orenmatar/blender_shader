from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Dict, Set
import numpy as np

VECTOR = "Vector"
VALUE = "Value"
COLOR = "Color"
RESULT = "Result"

seed_value_range = (0, 1000)

# TODO: remove the \n from the beginning of every code, have a func to add line?
# TODO: deliver the method for creating links between nodes to the node instances
# TODO: allow outputs to depend on params? e.g. vector math output is "Value" not "Vector" if it is dot product


class ParamType(Enum):
    VECTOR = 1
    FLOAT = 2
    VECTOR_INPUT = 3  # those are the vector inputs to textures usually - if not connected they default to coordinates
    CATEGORICAL = 4
    SEED = 5


class ParamRequestType(Enum):
    NUMERIC = 1
    CATEGORICAL = 2
    SEED = 3
    ALL = 4
    NON_VECTOR_INPUT = 5
    NON_SEED = 6


@dataclass
class Param:
    name: Union[str, int]
    options_range: tuple
    default: Union[tuple, str, float]
    param_type: ParamType


@dataclass
class NumericInput(Param):
    """
    A numeric input can be connected to another node or set to a value. In Blender all of these parameters can be
    connected to another node, but here I limit some of them to be set to a value, for simplicity of the generation.
    """

    as_input: bool


@dataclass
class Output:
    name: str
    param_type: ParamType


def _convert_list_to_dict(some_list) -> dict:
    """
    For any list of Params or Outputs, convert it to a dictionary with the name as the key.
    """
    return {input_.name: input_ for input_ in some_list}


def _is_string_or_np_str(obj) -> bool:
    return isinstance(obj, str) or isinstance(obj, np.str_)


def _dict_to_string_params(some_dict) -> str:
    """
    Convert a dictionary to a string of parameters for a function call.
    """
    param_list = []
    for key, value in some_dict.items():
        param_list.append(f"{key} = '{value}'" if _is_string_or_np_str(value) else f"{key} = {value}")
    return ",".join(param_list)


class Node(object):
    """
    Base class for all nodes in the shader network. A converter between the networkx graph and Blender nodes.
    Each subclass refers to a single node in the network, but can create several nodes and links in Blender.
    e.g. the Output node creates an Emission node in Blender and links it to the Output node, the Input node creates
    a Texture Coordinate node and a Mapping node and links them together.
    Each subclass has a set of NUMERIC, CATEGORICAL, SEEDs and OUTPUTS parameters that define the node.
    NUMERIC are parameters that can either get a value (e.g. "scale" for noise texture) or be connected to another node
    like a vector input to a texture (This may be a poor design choice and they may need to be separated, but the mix is
    complicated since numerics can always also be an input, but some vector inputs can't really be numeric - i.e.
    can't be set to a value or stay unconnected).
    CATEGORICAL are parameters that can take a value from a set of options (e.g. "operation" for math node).
    Seeds are parameters that are used to generate random values (e.g. "W" for noise texture), the assumption is that
    playing with these values will not change the texture in a meaningful way.
    Each code is responsible for converting the node to Blender code that can be used to create the node in Blender.
    Every node needs its own implementation of to_code since some need to change the names of the parameters to convert
    to the Blender code.
    """

    NAME = "GenericNode"
    NUMERIC: Dict = {}
    CATEGORICAL: Dict = {}
    OUTPUTS: Dict = {}
    SEED: Dict = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically creates all required dictionaries from the lists of Params and Outputs.
        """
        super().__init_subclass__(**kwargs)
        cls.NUMERIC = _convert_list_to_dict(getattr(cls, "NUMERIC", []))
        cls.CATEGORICAL = _convert_list_to_dict(getattr(cls, "CATEGORICAL", []))
        cls.OUTPUTS = _convert_list_to_dict(getattr(cls, "OUTPUTS", []))
        cls.SEED = _convert_list_to_dict(getattr(cls, "SEED", []))
        cls.INPUT_MAPPING = cls.get_input_names_mapping()
        assert all(
            [x not in cls.NUMERIC for x in cls.CATEGORICAL]
        ), "Cannot use the same name in numeric and categorical"

    def __init__(self, inputs: list, numeric: dict, categorical: dict, node_name: str, seeds: dict = None):
        self.inputs = inputs
        self.numeric = numeric
        self.categorical = categorical
        self.node_name = node_name
        if seeds is None:
            seeds = {}
        self.seeds = seeds

    @classmethod
    def get_input_names_mapping(cls):
        return {name: f"'{name}'" for name in cls.NUMERIC}  # place name in "" as it is used to generate code later

    def to_code(self, func_name: str) -> str:
        """
        Convert the node to code that can be used to create the node in Blender.
        func_name is the name of the function that creates the node, e.g. "nodes_adder.create_node"
        """
        params = {**self.numeric, **self.categorical}
        string_params = _dict_to_string_params(params)
        code = f'\n{self.node_name} = {func_name}("ShaderNode{self.NAME}", {string_params})'
        return code

    @classmethod
    def get_node_type_params(cls, param_type: ParamRequestType, default_values: bool = True):
        """
        Get the parameters of the node type.
        """
        if param_type == ParamRequestType.SEED:
            relevant_dict = cls.SEED
        elif param_type == ParamRequestType.NUMERIC:
            relevant_dict = cls.NUMERIC
        elif param_type == ParamRequestType.CATEGORICAL:
            relevant_dict = cls.CATEGORICAL
        elif param_type == ParamRequestType.ALL:
            relevant_dict = {**cls.SEED, **cls.NUMERIC, **cls.CATEGORICAL}
        elif param_type == ParamRequestType.NON_SEED:
            relevant_dict = {**cls.NUMERIC, **cls.CATEGORICAL}
        elif param_type == ParamRequestType.NON_VECTOR_INPUT:
            all_params = {**cls.SEED, **cls.NUMERIC, **cls.CATEGORICAL}
            relevant_dict = {
                key: value for key, value in all_params.items() if value.param_type != ParamType.VECTOR_INPUT
            }
        else:
            raise ValueError
        if default_values:
            return {key: value.default for key, value in relevant_dict.items()}
        # if you should not return default values - then return ranges for random generation
        return {key: (value.options_range, value.param_type) for key, value in relevant_dict.items()}

    @classmethod
    def properties_to_node_instance(cls, inputs: list, properties: dict, node_name: str) -> "Node":
        """
        Takes properties as stored by the networkx graph and creates a node instance.
        """
        numeric = {key: val for key, val in properties.items() if key in cls.NUMERIC}
        categorical = {key: val for key, val in properties.items() if key in cls.CATEGORICAL}
        seeds = {key: val for key, val in properties.items() if key in cls.SEED}
        if len(seeds):
            return cls(inputs, numeric, categorical, node_name, seeds)
        return cls(inputs, numeric, categorical, node_name)

    @classmethod
    def get_inputs(cls) -> Dict:
        """
        Get the possible numeric inputs of the node type.
        """
        return {key: val for key, val in cls.NUMERIC.items() if val.as_input}

    @classmethod
    def get_inputs_names_list(cls) -> List:
        """
        Used if we need to make sure the order is constant
        """
        return [input_name for input_name, val in cls.NUMERIC.items() if val.as_input]

    @classmethod
    def get_random_input(cls) -> str:
        return np.random.choice(list(cls.get_node_type_free_inputs()))

    @classmethod
    def get_outputs(cls) -> Dict:
        return cls.OUTPUTS

    @classmethod
    def get_node_type_free_inputs(cls) -> Set:
        """
        Get the inputs that can be connected to another node
        """
        return {key for key, param in cls.NUMERIC.items() if param.as_input}

    @classmethod
    def get_random_output(cls) -> str:
        return np.random.choice(list(cls.OUTPUTS))

    @staticmethod
    def set_vector_code(node_name: str, vector_name: str, values: list, filter_zeros=False) -> str:
        """
        Write code for Blender to set the values of a whole vector.
        e.g.
        node.inputs['Vector'].default_value[0] = 0.4
        node.inputs['Vector'].default_value[1] = 0  # this will be skipped if filter_zeros is True
        node.inputs['Vector'].default_value[2] = 2
        """
        code = ""
        for i, val in enumerate(values):
            if val == 0 and filter_zeros:
                continue
            code += f"\n{node_name}.inputs[{vector_name}].default_value[{i}] = {val}"
        return code


class CombineXYZ(Node):
    NAME = "CombineXYZ"
    NUMERIC = [
        NumericInput("X", (-10, 10), 0, ParamType.FLOAT, True),
        NumericInput("Y", (-10, 10), 0, ParamType.FLOAT, True),
        NumericInput("Z", (-10, 10), 0, ParamType.FLOAT, True),
    ]
    OUTPUTS = [Output(VECTOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeCombineXYZ")'
        for key, val in self.numeric.items():
            if val != 0:  # the default is 0 so no need to create a line for it
                code += f'\n{self.node_name}.inputs["{key}"].default_value = {val}'
        return code


class Mapping(Node):
    NAME = "Mapping"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Location", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
        NumericInput("Scale", (0, 10), (0, 0, 0), ParamType.VECTOR, True),
    ]
    OUTPUTS = [Output(VECTOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeMapping")'
        for key, val in self.numeric.items():
            code += self.set_vector_code(self.node_name, f"'{key}'", val, filter_zeros=True)
        return code


class Math(Node):
    NAME = "Math"
    NUMERIC = [
        NumericInput("value_0", (-10, 10), 0, ParamType.FLOAT, True),
        NumericInput("value_1", (-10, 10), 0, ParamType.FLOAT, True),
    ]
    CATEGORICAL = [
        Param(
            "operation",
            ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "POWER", "SQRT", "ABSOLUTE"),
            "ADD",
            ParamType.CATEGORICAL,
        )
    ]
    OUTPUTS = [Output(VALUE, ParamType.FLOAT)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    @classmethod
    def get_input_names_mapping(cls):
        """
        Converts the names of the inputs to the Blender names - simple 0, 1 in Blender.
        """
        return {"value_0": 0, "value_1": 1}

    def to_code(self, func_name):
        string_params = _dict_to_string_params(self.categorical)
        code = f'\n{self.node_name} = {func_name}("ShaderNodeMath", {string_params})'
        for key, value in self.numeric.items():
            if value != 0:
                in_name = self.INPUT_MAPPING[key]  # get the Blender name of the input
                code += f"\n{self.node_name}.inputs[{in_name}].default_value = {value}"
        return code


class MixFloat(Node):
    NAME = "MixFloat"
    NUMERIC = [
        NumericInput("Factor", (0, 1), 0.5, ParamType.FLOAT, True),
        NumericInput("A", (0, 1), 0, ParamType.FLOAT, True),
        NumericInput("B", (0, 1), 0, ParamType.FLOAT, True),
    ]
    OUTPUTS = [Output(RESULT, ParamType.FLOAT)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeMix")'
        for key, value in self.numeric.items():
            code += f'\n{self.node_name}.inputs["{key}"].default_value = {value}'
        return code


class MixVector(Node):
    NAME = "MixVector"
    NUMERIC = [
        NumericInput("Factor", (0, 1), 0, ParamType.FLOAT, True),
        NumericInput("A", (0, 1), (0, 0, 0), ParamType.VECTOR, True),
        NumericInput("B", (0, 1), (0, 0, 0), ParamType.VECTOR, True),
    ]
    CATEGORICAL = [
        Param(
            "blend_type",
            ("MIX", "MULTIPLY", "BURN", "DODGE", "ADD", "OVERLAY", "SUBTRACT"),
            "ADD",
            ParamType.CATEGORICAL,
        )
    ]
    OUTPUTS = [Output(RESULT, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name):
        string_params = _dict_to_string_params(self.categorical)
        code = f'\n{self.node_name} = {func_name}("ShaderNodeMix", {string_params})'
        # set to RGBA since it should receive a vector, otherwise it's the same node as MixFloat
        code += f'\n{self.node_name}.data_type = "RGBA"'
        for key, value in self.numeric.items():
            if key == "Factor":
                code += f'\n{self.node_name}.inputs["Factor"].default_value = {value}'
            else:
                code += self.set_vector_code(self.node_name, f"'{key}'", value, filter_zeros=False)
        return code


class SeparateXYZ(Node):
    NAME = "SeparateXYZ"
    NUMERIC = [NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True)]
    OUTPUTS = [
        Output("X", ParamType.FLOAT),
        Output("Y", ParamType.FLOAT),
        Output("Z", ParamType.FLOAT),
    ]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeSeparateXYZ")'
        # removed this - we are not using the params, it will always be connected
        # code += self.set_vector_code(self.node_name, f"'Vector'", self.numeric["Vector"], filter_zeros=True)
        return code


class InputNode(Node):
    NAME = "InputNode"
    OUTPUTS = [Output("Vector", ParamType.VECTOR)]
    NUMERIC = [NumericInput("X location", (-1, 1), 0, ParamType.FLOAT, False)]
    # just rotating the whole texture counts as a "seed" since it creates a non-meaningful change to the texture
    SEED = [Param("Z Rotation", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical, node_name, seeds):
        super().__init__(inputs, numeric, categorical, node_name, seeds)

    def to_code(self, func_name, node_tree_name="node_tree"):
        code = f'\ninput_coor = {func_name}("ShaderNodeTexCoord")'
        code += f'\n{self.node_name} = {func_name}("ShaderNodeMapping")'
        code += f'\n{self.node_name}.inputs["Location"].default_value[0] = {self.numeric["X location"]}'
        code += f'\n{self.node_name}.inputs["Rotation"].default_value[2] = {self.seeds["Z Rotation"]}'
        code += f'\n{node_tree_name}.links.new(input_coor.outputs["Object"], {self.node_name}.inputs["Vector"])'
        return code


class TexGabor(Node):
    NAME = "TexGabor"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 10), 5, ParamType.FLOAT, False),
        NumericInput("Frequency", (0, 10), 2, ParamType.FLOAT, False),
    ]
    OUTPUTS = [Output(VALUE, ParamType.FLOAT)]
    SEED = [Param("Orientation", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical, node_name, seeds):
        super().__init__(inputs, numeric, categorical, node_name, seeds)

    def to_code(self, func_name):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeTexGabor")'
        set_params = {**self.seeds, **{key: val for key, val in self.numeric.items() if key != VECTOR}}
        for key, value in set_params.items():
            code += f'\n{self.node_name}.inputs["{key}"].default_value = {value}'
        return code


class TexGradient(Node):
    NAME = "TexGradient"
    NUMERIC = [NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True)]
    CATEGORICAL = [
        Param(
            "gradient_type",
            ("LINEAR", "DIAGONAL", "SPHERICAL", "RADIAL"),
            "LINEAR",
            ParamType.CATEGORICAL,
        )
    ]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name):
        string_params = _dict_to_string_params(self.categorical)
        code = f'\n{self.node_name} = {func_name}("ShaderNodeTexGradient", {string_params})'
        return code


class TexNoise(Node):
    NAME = "TexNoise"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 10), 5, ParamType.FLOAT, False),
        NumericInput("Lacunarity", (0, 10), 2, ParamType.FLOAT, False),
        NumericInput("Distortion", (0, 5), 0, ParamType.FLOAT, False),
    ]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]
    SEED = [Param("W", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical, node_name, seeds):
        super().__init__(inputs, numeric, categorical, node_name, seeds)

    def to_code(self, func_name):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeTexNoise", noise_dimensions="4D")'
        # params - seeds and numeric except for the vector input
        set_params = {**self.seeds, **{key: val for key, val in self.numeric.items() if key != VECTOR}}
        for key, value in set_params.items():
            code += f'\n{self.node_name}.inputs["{key}"].default_value = {value}'
        return code


class TexVoronoiF(Node):
    NAME = "TexVoronoiF"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 10), 5, ParamType.FLOAT, True),
        NumericInput("Randomness", (0, 1), 1, ParamType.FLOAT, False),
    ]
    CATEGORICAL = [Param("distance", ("EUCLIDEAN", "CHEBYCHEV"), "EUCLIDEAN", ParamType.CATEGORICAL)]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]
    SEED = [Param("W", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical, node_name, seeds):
        super().__init__(inputs, numeric, categorical, node_name, seeds)

    def to_code(self, func_name):
        string_params = _dict_to_string_params(self.categorical)
        code = f'\n{self.node_name} = {func_name}("ShaderNodeTexVoronoi", {string_params}, voronoi_dimensions="4D")'
        set_params = {
            **self.seeds,
            **{key: val for key, val in self.numeric.items() if key != VECTOR},
        }
        for key, value in set_params.items():
            code += f'\n{self.node_name}.inputs["{key}"].default_value = {value}'
        return code


class TexWave(Node):
    NAME = "TexWave"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 10), 5, ParamType.FLOAT, False),
        NumericInput("Distortion", (0, 5), 0, ParamType.FLOAT, False),
    ]
    CATEGORICAL = [Param("wave_profile", ("SIN", "SAW"), "SIN", ParamType.CATEGORICAL)]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]
    SEED = [Param("Phase Offset", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical, node_name, seeds):
        super().__init__(inputs, numeric, categorical, node_name, seeds)

    def to_code(self, func_name):
        string_params = _dict_to_string_params(self.categorical)
        code = f'\n{self.node_name} = {func_name}("ShaderNodeTexWave", {string_params})'
        set_params = {
            **self.seeds,
            **{key: val for key, val in self.numeric.items() if key != VECTOR},
        }
        for key, value in set_params.items():
            code += f'\n{self.node_name}.inputs["{key}"].default_value = {value}'
        return code


class ValToRGB(Node):
    NAME = "ValToRGB"
    NUMERIC = [
        NumericInput("Fac", (0, 1), 0, ParamType.FLOAT, True),
        NumericInput("element_0", (0, 1), 0, ParamType.FLOAT, False),
        NumericInput("element_1", (0, 1), 1, ParamType.FLOAT, False),
    ]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeValToRGB")'
        code += f'\n{self.node_name}.inputs[0].default_value = {self.numeric["Fac"]}'
        code += f'\n{self.node_name}.color_ramp.elements[0].position = {self.numeric["element_0"]}'
        code += f'\n{self.node_name}.color_ramp.elements[1].position = {self.numeric["element_1"]}'
        return code


class Value(Node):
    NAME = "Value"
    NUMERIC = [
        NumericInput(VALUE, (-10, 10), 0, ParamType.FLOAT, False),
    ]
    OUTPUTS = [Output(VALUE, ParamType.FLOAT)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeValue")'
        code += f"\n{self.node_name}.outputs[0].default_value = {self.numeric[VALUE]}"
        return code


class VectorMath(Node):
    NAME = "VectorMath"
    NUMERIC = [
        NumericInput("vector_0", (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("vector_1", (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
    ]
    CATEGORICAL = [
        Param(
            "operation",
            (
                "ADD",
                "SUBTRACT",
                "MULTIPLY",
                "DIVIDE",
                "CROSS_PRODUCT",
                "ABSOLUTE",
                "FRACTION"
            ),
            "ADD",
            ParamType.CATEGORICAL,
        )
    ]
    OUTPUTS = [Output(VECTOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    @classmethod
    def get_input_names_mapping(cls):
        return {"vector_0": 0, "vector_1": 1}

    def to_code(self, func_name):
        string_params = _dict_to_string_params(self.categorical)
        code = f'\n{self.node_name} = {func_name}("ShaderNodeVectorMath", {string_params})'
        # removed this part since we're only using vector math as vector input - it has no params
        # for key, value in self.numeric.items():
        #     if value != 0:
        #         in_name = self.INPUT_MAPPING[key]
        #         code += self.set_vector_code(self.node_name, in_name, value, filter_zeros=True)
        return code


class OutputNode(Node):
    NAME = "Output"
    NUMERIC = [
        NumericInput("Color", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
    ]

    def __init__(self, inputs, numeric, categorical, node_name):
        super().__init__(inputs, numeric, categorical, node_name)

    def to_code(self, func_name, node_tree_name="node_tree"):
        code = f'\n{self.node_name} = {func_name}("ShaderNodeEmission")'
        code += f"\n{self.node_name}.inputs['Strength'].default_value = 1"
        code += f'\noutput = {func_name}("ShaderNodeOutputMaterial")'
        code += f'\n{node_tree_name}.links.new({self.node_name}.outputs["Emission"], output.inputs["Surface"])'
        return code
