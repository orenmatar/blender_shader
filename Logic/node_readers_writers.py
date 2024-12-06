from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Dict
import numpy as np

VECTOR = "Vector"
VALUE = "Value"
COLOR = "Color"
RESULT = "Result"


class NumericType(Enum):
    VECTOR = 1
    FLOAT = 2
    VECTOR_INPUT = 3


@dataclass
class Param:
    name: str
    options_range: tuple
    default: Union[tuple, str, float]


@dataclass
class NumericInput(Param):
    param_type: NumericType
    as_input: bool


@dataclass
class Output:
    name: str
    param_type: NumericType


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

    def __init_subclass__(cls, **kwargs):
        """
        Automatically creates INPUT_DICT for any subclass based on its NUMERIC.
        """
        super().__init_subclass__(**kwargs)
        cls.NUMERIC = _convert_list_to_dict(
            getattr(cls, "NUMERIC", [])
        )
        cls.CATEGORICAL = _convert_list_to_dict(getattr(cls, "CATEGORICAL", []))
        cls.OUTPUTS = _convert_list_to_dict(getattr(cls, "OUTPUTS", []))
        assert all([x not in cls.NUMERIC for x in cls.CATEGORICAL]), "Cannot use the same name in numeric and categorical"

    def __init__(self, inputs, numeric, categorical):
        self.inputs = inputs
        self.numeric = numeric
        self.categorical = categorical

    @classmethod
    def properties_to_node_instance(cls, inputs, properties):
        numeric = {key: val for key, val in properties.items() if key in cls.NUMERIC}
        categorical = {key: val for key, val in properties.items() if key in cls.CATEGORICAL}
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
    def get_node_type_params(cls, default_values = True):
        all_properties = {}
        for key, param in cls.NUMERIC.items():
            all_properties[key] = param.default
        for key, param in cls.CATEGORICAL.items():
            all_properties[key] = param.default
        return all_properties

    @classmethod
    def get_random_output(cls):
        return np.random.choice(list(cls.OUTPUTS))


class CombineXYZ(Node):
    NAME = "CombineXYZ"
    NUMERIC = [
        NumericInput("X", (-10, 10), 0, NumericType.FLOAT, True),
        NumericInput("Y", (-10, 10), 0, NumericType.FLOAT, True),
        NumericInput("Z", (-10, 10), 0, NumericType.FLOAT, True),
    ]
    OUTPUTS = [Output(VECTOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class Mapping(Node):
    NAME = "Mapping"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), NumericType.VECTOR_INPUT, True),
        NumericInput("Location", (-10, 10), (0, 0, 0), NumericType.VECTOR, True),
        NumericInput("Scale", (-10, 10), (0, 0, 0), NumericType.VECTOR, True),
    ]
    OUTPUTS = [Output(VECTOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class Math(Node):
    NAME = "Math"
    NUMERIC = [
        NumericInput("value_0", (-10, 10), 0, NumericType.FLOAT, True),
        NumericInput("value_1", (-10, 10), 0, NumericType.FLOAT, True),
    ]
    CATEGORICAL = [
        Param(
            "operation",
            ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "POWER", "SQRT", "ABSOLUTE"),
            "ADD",
        )
    ]
    OUTPUTS = [Output(VALUE, NumericType.FLOAT)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class MixFloat(Node):
    NAME = "MixFloat"
    NUMERIC = [
        NumericInput("Factor", (-10, 10), 0, NumericType.FLOAT, False),
        NumericInput("A", (-10, 10), 0, NumericType.FLOAT, True),
        NumericInput("B", (-10, 10), 0, NumericType.FLOAT, True),
    ]
    CATEGORICAL = [
        Param(
            "operation",
            ("MIX", "MULTIPLY", "BURN", "DODGE", "ADD", "OVERLAY", "SUBTRACT"),
            "ADD",
        )
    ]
    OUTPUTS = [Output(RESULT, NumericType.FLOAT)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class MixVector(Node):
    NAME = "MixVector"
    NUMERIC = [
        NumericInput("Factor", (-10, 10), 0, NumericType.FLOAT, False),
        NumericInput("A", (-10, 10), (0, 0, 0), NumericType.VECTOR, True),
        NumericInput("B", (-10, 10), (0, 0, 0), NumericType.VECTOR, True),
    ]
    OUTPUTS = [Output(RESULT, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class SeparateXYZ(Node):
    NAME = "SeparateXYZ"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), NumericType.VECTOR, True)
    ]
    OUTPUTS = [
        Output("X", NumericType.FLOAT),
        Output("Y", NumericType.FLOAT),
        Output("Z", NumericType.FLOAT),
    ]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class InputNode(Node):
    NAME = "InputNode"
    OUTPUTS = [Output("Object", NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexGabor(Node):
    NAME = "TexGabor"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), NumericType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 20), 5, NumericType.FLOAT, False),
        NumericInput("Frequency", (0, 10), 2, NumericType.FLOAT, False),
    ]
    OUTPUTS = [Output(COLOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexGradient(Node):
    NAME = "TexGradient"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), NumericType.VECTOR_INPUT, True),
    ]
    CATEGORICAL = [
        Param(
            "gradient_type",
            ("LINEAR", "DIAGONAL", "SPHERICAL", "RADIAL"),
            "LINEAR",
        )
    ]
    OUTPUTS = [Output(COLOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexNoise(Node):
    NAME = "TexNoise"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), NumericType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 20), 5, NumericType.FLOAT, False),
        NumericInput("Lacunarity", (0, 10), 2, NumericType.FLOAT, False),
        NumericInput("Distortion", (0, 5), 0, NumericType.FLOAT, False),
    ]
    OUTPUTS = [Output(COLOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexVoronoiF(Node):
    NAME = "TexVoronoiF"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), NumericType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 20), 5, NumericType.FLOAT, False),
        NumericInput("Randomness", (0, 1), 1, NumericType.FLOAT, False),
    ]
    CATEGORICAL = [Param("distance", ("EUCLIDEAN", "CHEBYCHEV"), "EUCLIDEAN")]
    OUTPUTS = [Output(COLOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class TexWave(Node):
    NAME = "TexWave"
    NUMERIC = [
        NumericInput(VECTOR, (-10, 10), (0, 0, 0), NumericType.VECTOR_INPUT, True),
        NumericInput("Scale", (0, 20), 5, NumericType.FLOAT, False),
        NumericInput("Distortion", (0, 5), 0, NumericType.FLOAT, False),
    ]
    CATEGORICAL = [Param("wave_profile", ("SIN", "SAW"), "SIN")]
    OUTPUTS = [Output(COLOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class ValToRGB(Node):
    NAME = "ValToRGB"
    NUMERIC = [
        NumericInput("Fac", (-10, 10), 0, NumericType.FLOAT, True),
        NumericInput("element_0", (0, 1), 0, NumericType.FLOAT, False),
        NumericInput("element_1", (0, 1), 1, NumericType.FLOAT, False),
    ]
    OUTPUTS = [Output(COLOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class Value(Node):
    NAME = "Value"
    NUMERIC = [
        NumericInput(VALUE, (-10, 10), 0, NumericType.FLOAT, True),
    ]
    OUTPUTS = [Output(VALUE, NumericType.FLOAT)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class VectorMath(Node):
    NAME = "VectorMath"
    NUMERIC = [
        NumericInput("vector_0", (-10, 10), (0, 0, 0), NumericType.VECTOR, True),
        NumericInput("vector_1", (-10, 10), (0, 0, 0), NumericType.VECTOR, True),
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
        )
    ]
    OUTPUTS = [Output(VECTOR, NumericType.VECTOR)]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)


class OutputNode(Node):
    NAME = "Output"
    NUMERIC = [
        NumericInput("Color", (-10, 10), (0, 0, 0), NumericType.VECTOR, True),
    ]

    def __init__(self, inputs, numeric, categorical):
        super().__init__(inputs, numeric, categorical)
