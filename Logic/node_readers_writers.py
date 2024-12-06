from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Dict, Set
import numpy as np

VECTOR = "Vector"
VALUE = "Value"
COLOR = "Color"
RESULT = "Result"

seed_value_range = (0, 1000)


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


@dataclass
class Param:
    name: str
    options_range: tuple
    default: Union[tuple, str, float]
    param_type: ParamType


@dataclass
class NumericInput(Param):
    as_input: bool


@dataclass
class Output:
    name: str
    param_type: ParamType


def _convert_list_to_dict(some_list):
    """
    For any list of Params or Outputs
    :return:
    """
    return {input_.name: input_ for input_ in some_list}


class Node(object):
    NUMERIC: Dict = {}
    CATEGORICAL: Dict = {}
    OUTPUTS: Dict = {}
    SEED: Dict = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically creates INPUT_DICT for any subclass based on its NUMERIC.
        """
        super().__init_subclass__(**kwargs)
        cls.NUMERIC = _convert_list_to_dict(getattr(cls, "NUMERIC", []))
        cls.CATEGORICAL = _convert_list_to_dict(getattr(cls, "CATEGORICAL", []))
        cls.OUTPUTS = _convert_list_to_dict(getattr(cls, "OUTPUTS", []))
        cls.SEED = _convert_list_to_dict(getattr(cls, "SEED", []))
        assert all(
            [x not in cls.NUMERIC for x in cls.CATEGORICAL]
        ), "Cannot use the same name in numeric and categorical"

    def __init__(self, inputs, numeric, categorical, seeds=None):
        self.inputs = inputs
        self.numeric = numeric
        self.categorical = categorical
        if seeds is None:
            seeds = {}
        self.seeds = seeds

    @classmethod
    def get_node_type_params(cls, param_type: ParamRequestType, default_values=True):
        if param_type == ParamRequestType.SEED:
            relevant_dict = cls.SEED
        elif param_type == ParamRequestType.NUMERIC:
            relevant_dict = cls.NUMERIC
        elif param_type == ParamRequestType.CATEGORICAL:
            relevant_dict = cls.CATEGORICAL
        elif param_type == ParamRequestType.ALL:
            relevant_dict = {**cls.SEED, **cls.NUMERIC, **cls.CATEGORICAL}
        elif param_type == ParamRequestType.NON_VECTOR_INPUT:
            all_params = {**cls.SEED, **cls.NUMERIC, **cls.CATEGORICAL}
            relevant_dict = {
                key: value
                for key, value in all_params.items()
                if value.param_type != ParamType.VECTOR_INPUT
            }
        else:
            raise ValueError
        if default_values:
            return {key: value.default for key, value in relevant_dict.items()}
        # if do not return default values - then return ranges
        return {
            key: (value.options_range, value.param_type)
            for key, value in relevant_dict.items()
        }

    @classmethod
    def properties_to_node_instance(cls, inputs, properties):
        numeric = {key: val for key, val in properties.items() if key in cls.NUMERIC}
        categorical = {
            key: val for key, val in properties.items() if key in cls.CATEGORICAL
        }
        return cls(inputs, numeric, categorical)

    @classmethod
    def get_inputs(cls):
        return {key: val for key, val in cls.NUMERIC.items() if val.as_input}

    @classmethod
    def get_outputs(cls):
        return cls.OUTPUTS

    @classmethod
    def get_node_type_free_inputs(cls):
        return {key for key, param in cls.NUMERIC.items() if param.as_input}

    @classmethod
    def get_random_output(cls):
        return np.random.choice(list(cls.OUTPUTS))


class CombineXYZ(Node):
    NAME = "CombineXYZ"
    NUMERIC = [
        NumericInput("X", (-10, 10), 0, ParamType.FLOAT, True),
        NumericInput("Y", (-10, 10), 0, ParamType.FLOAT, True),
        NumericInput("Z", (-10, 10), 0, ParamType.FLOAT, True),
    ]
    OUTPUTS = [Output(VECTOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class Mapping(Node):
    NAME = "Mapping"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Location", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
        NumericInput("Scale", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
    ]
    OUTPUTS = [Output(VECTOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


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

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class MixFloat(Node):
    NAME = "MixFloat"
    NUMERIC = [
        NumericInput("Factor", (-10, 10), 0, ParamType.FLOAT, False),
        NumericInput("A", (-10, 10), 0, ParamType.FLOAT, True),
        NumericInput("B", (-10, 10), 0, ParamType.FLOAT, True),
    ]
    CATEGORICAL = [
        Param(
            "operation",
            ("MIX", "MULTIPLY", "BURN", "DODGE", "ADD", "OVERLAY", "SUBTRACT"),
            "ADD",
            ParamType.CATEGORICAL,
        )
    ]
    OUTPUTS = [Output(RESULT, ParamType.FLOAT)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class MixVector(Node):
    NAME = "MixVector"
    NUMERIC = [
        NumericInput("Factor", (-10, 10), 0, ParamType.FLOAT, False),
        NumericInput("A", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
        NumericInput("B", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
    ]
    OUTPUTS = [Output(RESULT, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class SeparateXYZ(Node):
    NAME = "SeparateXYZ"
    NUMERIC = [NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR, True)]
    OUTPUTS = [
        Output("X", ParamType.FLOAT),
        Output("Y", ParamType.FLOAT),
        Output("Z", ParamType.FLOAT),
    ]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class InputNode(Node):
    NAME = "InputNode"
    OUTPUTS = [Output("Object", ParamType.VECTOR)]
    SEED = [Param("Z Rotation", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexGabor(Node):
    NAME = "TexGabor"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 20), 5, ParamType.FLOAT, False),
        NumericInput("Frequency", (0, 10), 2, ParamType.FLOAT, False),
    ]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]
    SEED = [Param("Orientation", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexGradient(Node):
    NAME = "TexGradient"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
    ]
    CATEGORICAL = [
        Param(
            "gradient_type",
            ("LINEAR", "DIAGONAL", "SPHERICAL", "RADIAL"),
            "LINEAR",
            ParamType.CATEGORICAL,
        )
    ]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexNoise(Node):
    NAME = "TexNoise"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 20), 5, ParamType.FLOAT, False),
        NumericInput("Lacunarity", (0, 10), 2, ParamType.FLOAT, False),
        NumericInput("Distortion", (0, 5), 0, ParamType.FLOAT, False),
    ]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]
    SEED = [Param("W", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexVoronoiF(Node):
    NAME = "TexVoronoiF"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 20), 5, ParamType.FLOAT, False),
        NumericInput("Randomness", (0, 1), 1, ParamType.FLOAT, False),
    ]
    CATEGORICAL = [
        Param(
            "distance", ("EUCLIDEAN", "CHEBYCHEV"), "EUCLIDEAN", ParamType.CATEGORICAL
        )
    ]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]
    SEED = [Param("W", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexWave(Node):
    NAME = "TexWave"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), ParamType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 20), 5, ParamType.FLOAT, False),
        NumericInput("Distortion", (0, 5), 0, ParamType.FLOAT, False),
    ]
    CATEGORICAL = [Param("wave_profile", ("SIN", "SAW"), "SIN", ParamType.CATEGORICAL)]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]
    SEED = [Param("Phase Offset", seed_value_range, 0, ParamType.SEED)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class ValToRGB(Node):
    NAME = "ValToRGB"
    NUMERIC = [
        NumericInput("Fac", (-10, 10), 0, ParamType.FLOAT, True),
        NumericInput("element_0", (0, 1), 0, ParamType.FLOAT, False),
        NumericInput("element_1", (0, 1), 1, ParamType.FLOAT, False),
    ]
    OUTPUTS = [Output(COLOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class Value(Node):
    NAME = "Value"
    NUMERIC = [
        NumericInput(VALUE, (-10, 10), 0, ParamType.FLOAT, True),
    ]
    OUTPUTS = [Output(VALUE, ParamType.FLOAT)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class VectorMath(Node):
    NAME = "VectorMath"
    NUMERIC = [
        NumericInput("vector_0", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
        NumericInput("vector_1", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
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
                "DOT_PRODUCT",
                "ABSOLUTE",
            ),
            "ADD",
            ParamType.CATEGORICAL,
        )
    ]
    OUTPUTS = [Output(VECTOR, ParamType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class OutputNode(Node):
    NAME = "Output"
    NUMERIC = [
        NumericInput("Color", (-10, 10), (0, 0, 0), ParamType.VECTOR, True),
    ]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)
