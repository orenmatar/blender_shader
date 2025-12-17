VECTOR_VALUES_SEP = " | "
VECTOR_VALUES_OPEN = "[ "
VECTOR_VALUES_CLOSE = " ]"
NODE_OPEN = " ( "
NODE_CLOSE = " )"
NAME_VALUE_SEP = " : "
PARAMETERS_SEP = " , "
NODES_EDGES_SEP = " ; "
NODES_START = "NODES: "
EDGES_START = "EDGES: "
FROM_TO = " -> "
NO_ACTION_ID = 0


NO_ACTION_LABEL = "0__"
DECREASE = "DECREASE"
INCREASE = "INCREASE"
VECTOR = "VECTOR"


MAIN_HEAD_LABEL_TO_ID_MAP = {
    "0": NO_ACTION_ID,
    "REMOVE_NODE": 1,
    "NUMERIC": 2,
    "CAT_AND_NUMERIC": 3,
    "REMOVE_EDGE": 4,
    "ADD_EDGE": 5,
    "ADD_NODE": 6,
    "VECTOR": 7,
}

SECONDARY_HEAD_LABEL_TO_ID_MAP = {
    "": NO_ACTION_ID,
    "INCREASE": 1,
    "DECREASE": 2,
    "SAW": 3,
    "BURN": 4,
    "DIAGONAL": 5,
    "SPHERICAL": 6,
    "from__out": 7,
    "to__Y": 8,
    "to__Vector": 9,
    "TexGradient__Vector__Color": 10,
    "MixVector__A__Result": 11,
    "OVERLAY": 12,
    "to__X": 13,
    "TexNoise__Scale__Color": 14,
    "TexWave__Vector__Color": 15,
    "TexWave__Scale__Color": 16,
    "MixVector__Factor__Result": 17,
    "Mapping__Vector__Vector": 18,
    "to__Scale": 19,
    "TexGabor__Vector__Value": 20,
    "TexNoise__Vector__Color": 21,
    "DODGE": 22,
    "LINEAR": 23,
    "0__": 24,
    "SUBTRACT": 25,
    "Value": 26,
    "Math": 27,
    "TexVoronoiDistance__Vector__Distance": 28,
    "TexVoronoiF__Vector__Color": 29,
    "MixVector__B__Result": 30,
    "MULTIPLY": 31,
    "RADIAL": 32,
    "Math__value_1__Value": 33,
    "ValToRGB__Fac__Color": 34,
    "VectorMath__vector_0__Vector": 35,
    "FRACTION": 36,
    "to__Location": 37,
    "MIX": 38,
    "ADD": 39,
    "VectorMath": 40,
    "CROSS_PRODUCT": 41,
    "DIVIDE": 42,
    "ABSOLUTE": 43,
    "EUCLIDEAN": 44,
    "CHEBYCHEV": 45,
    "SIN": 46,
    "Math__value_0__Value": 47,
    "POWER": 48,
    "from__X": 49,
    "SeparateXYZ__Vector__X": 50,
    "from__Y": 51,
    "SQRT": 52,
    "CombineXYZ__X__Vector": 53,
    "ValToRGB": 54,
    "CombineXYZ": 55,
    "CombineXYZ__Y__Vector": 56,
    "MixVector": 57,
    "SeparateXYZ__Vector__Y": 58,
}

MAIN_HEAD_ID_TO_LABEL_MAP = {value: key for key, value in MAIN_HEAD_LABEL_TO_ID_MAP.items()}
SECONDARY_HEAD_ID_TO_LABEL_MAP = {value: key for key, value in SECONDARY_HEAD_LABEL_TO_ID_MAP.items()}
SECONDARY_HEAD_WEIGHTS = {"INCREASE": 0.2, "from__out": 0.2, "DECREASE": 0.2, "to__Vector": 0.3, "": 0.07}

HEAD2_OPTIONS_BY_HEAD1 = {
    "0": {""},
    "REMOVE_NODE": {""},
    "NUMERIC": {"0__", "DECREASE", "INCREASE"},
    "CAT_AND_NUMERIC": {
        "ABSOLUTE",
        "ADD",
        "BURN",
        "CHEBYCHEV",
        "CROSS_PRODUCT",
        "DIAGONAL",
        "DIVIDE",
        "DODGE",
        "EUCLIDEAN",
        "FRACTION",
        "LINEAR",
        "MIX",
        "MULTIPLY",
        "OVERLAY",
        "POWER",
        "RADIAL",
        "SAW",
        "SIN",
        "SPHERICAL",
        "SQRT",
        "SUBTRACT",
    },
    "REMOVE_EDGE": {""},
    "ADD_EDGE": {"from__X", "from__Y", "from__out", "to__Location", "to__Scale", "to__Vector", "to__X", "to__Y"},
    "ADD_NODE": {
        "CombineXYZ",
        "CombineXYZ__X__Vector",
        "CombineXYZ__Y__Vector",
        "Mapping",
        "Mapping__Vector__Vector",
        "Math",
        "Math__value_0__Value",
        "Math__value_1__Value",
        "MixVector",
        "MixVector__A__Result",
        "MixVector__B__Result",
        "MixVector__Factor__Result",
        "SeparateXYZ",
        "SeparateXYZ__Vector__X",
        "SeparateXYZ__Vector__Y",
        "TexGabor",
        "TexGabor__Vector__Value",
        "TexGradient",
        "TexGradient__Vector__Color",
        "TexNoise",
        "TexNoise__Scale__Color",
        "TexNoise__Vector__Color",
        "TexVoronoiDistance",
        "TexVoronoiDistance__Vector__Distance",
        "TexVoronoiF",
        "TexVoronoiF__Vector__Color",
        "TexWave",
        "TexWave__Scale__Color",
        "TexWave__Vector__Color",
        "ValToRGB",
        "ValToRGB__Fac__Color",
        "Value",
        "VectorMath",
        "VectorMath__vector_0__Vector",
    },
    "VECTOR": {"0__", "DECREASE", "INCREASE"},
}
