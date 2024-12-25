from Logic.meta_network import *


mix_textures = MetaNode(
    "mix_textures",
    {
        "mix": SubMetaNode(NODE_LIST_mix_vector),
        "tex1": SubMetaNode(NODE_LIST_textures),
        "tex2": SubMetaNode(NODE_LIST_textures),
    },
    [
        Con("tex1", "mix"),
        Con("tex2", "mix"),
        Con(IN, "tex1"),
        Con(IN, "tex2"),
        Con("mix", OUT),
    ],
    input_type=InOutType.VECTOR_OR_COLOR,
    output_type=InOutType.COLOR,
)

math_on_x_y_separately = MetaNode(
    "math_on_x_y_separately",
    {
        "sep": SubMetaNode(NODE_LIST_sep),
        "math1": SubMetaNode(NODE_LIST_math_float, allowed_params={"operation": ("ABSOLUTE", "POWER")}),
        "math2": SubMetaNode(NODE_LIST_math_float, allowed_params={"operation": ("ABSOLUTE", "POWER")}),
        "math3": SubMetaNode(NODE_LIST_math_float, allowed_params={"operation": ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE")}),
    },
    [
        Con(IN, "sep"),
        Con("math1", "math3"),
        Con("math2", "math3"),
        Con("sep", "math2", out_names=["X"]),
        Con("sep", "math1", out_names=["Y"]),
        Con("math3", OUT),
    ],
    required_input_type=InOutType.VECTOR_OR_COLOR,
    output_type=InOutType.FLOAT,
    input_type=InOutType.VECTOR_OR_COLOR,
)

tex_on_x_y_separately_then_math = MetaNode(
    "tex_on_x_y_separately_then_math",
    {
        "sep": SubMetaNode(NODE_LIST_sep),
        "tex1": SubMetaNode(NODE_LIST_textures),
        "tex2": SubMetaNode(NODE_LIST_textures),
        "math1": SubMetaNode(NODE_LIST_math_float, allowed_params={"operation": ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE")}),
    },
    [
        Con(IN, "sep"),
        Con("tex1", "math1"),
        Con("tex2", "math1"),
        Con("sep", "tex1", out_names=["X"]),
        Con("sep", "tex2", out_names=["Y"]),
        Con("math1", OUT),
    ],
    required_input_type=InOutType.VECTOR_OR_COLOR,
    output_type=InOutType.FLOAT,
    input_type=InOutType.VECTOR_OR_COLOR,
)

tex_on_x_y_separately_then_combine = MetaNode(
    "tex_on_x_y_separately_then_combine",
    {
        "sep": SubMetaNode(NODE_LIST_sep),
        "tex1": SubMetaNode(NODE_LIST_textures),
        "tex2": SubMetaNode(NODE_LIST_textures),
        "comb1": SubMetaNode(NODE_LIST_combine),
    },
    [
        Con(IN, "sep"),
        Con("tex1", "comb1", in_names=["X"]),
        Con("tex2", "comb1", in_names=["Y"]),
        Con("sep", "tex1", out_names=["X"]),
        Con("sep", "tex2", out_names=["Y"]),
        Con("comb1", OUT),
    ],
    required_input_type=InOutType.VECTOR_OR_COLOR,
    output_type=InOutType.COLOR,
    input_type=InOutType.VECTOR_OR_COLOR,
)


burn_dodge = MetaNode(
    "burn_dodge",
    {
        "burn": SubMetaNode(NODE_LIST_mix_vector, allowed_params={"blend_type": ("BURN",), "B": (0, 0)}),
        "dodge": SubMetaNode(NODE_LIST_mix_vector, allowed_params={"blend_type": ("DODGE",), "B": (1, 1)}),
    },
    [Con(IN, "burn", in_names=["A"]), Con("burn", "dodge", in_names=["A"]), Con("dodge", OUT)],
    required_output_type=InOutType.OUTPUT,
    input_type=InOutType.COLOR,
)

combine_textures = MetaNode(
    "combine_textures",
    {
        "tex_fac": SubMetaNode(NODE_LIST_textures),
        "mix": SubMetaNode(NODE_LIST_mix_vector),
        "ramp": SubMetaNode(NODE_LIST_ramp),
        "tex1": SubMetaNode(NODE_LIST_textures),
        "tex2": SubMetaNode(NODE_LIST_textures),
    },
    [
        Con(IN, "tex_fac", in_names=[VECTOR]),
        Con(IN, "tex1", in_names=[VECTOR]),
        Con(IN, "tex2", in_names=[VECTOR]),
        Con("mix", OUT),
        Con("tex_fac", "ramp"),
        Con("ramp", "mix", in_names=["Factor"]),
        Con("tex1", "mix", in_names=["A", "B"]),
        Con("tex2", "mix", in_names=["A", "B"]),
    ],
    output_type=InOutType.COLOR,
    input_type=InOutType.VECTOR_OR_COLOR,
)

duplicate_texture_to_two_textures_then_mix = MetaNode(
    "duplicate_texture_to_two_textures_then_mix",
    {
        "mix": SubMetaNode(NODE_LIST_mix_vector),
        "tex0": SubMetaNode(NODE_LIST_textures),
        "tex1": SubMetaNode(NODE_LIST_textures),
        "tex2": SubMetaNode(NODE_LIST_textures),
    },
    [
        Con("tex1", "mix"),
        Con("tex2", "mix"),
        Con(IN, "tex0"),
        Con("tex0", "tex2"),
        Con("tex0", "tex1"),
        Con("mix", OUT),
    ],
    output_type=InOutType.COLOR,
    input_type=InOutType.VECTOR_OR_COLOR,
)

texture_and_mix_with_self = MetaNode(
    "texture_and_mix_with_self",
    {
        "mix": SubMetaNode(NODE_LIST_mix_vector),
        "tex1": SubMetaNode(NODE_LIST_textures),
        "mapping": SubMetaNode(NODE_LIST_mapping, allowed_params={"Location": (0, 0), "Scale": (1, 1)}),
    },
    [
        Con(IN, "mapping", in_names=[VECTOR]),
        Con("mapping", "tex1"),
        Con("mapping", "mix", in_names=['A']),
        Con("tex1", "mix", in_names=['B']),
        Con("mix", OUT),
    ],
    output_type=InOutType.COLOR,
    required_output_type=InOutType.VECTOR,
    input_type=InOutType.VECTOR,
)

combineXyz = MetaNode(
    "combineXyz",
    {
        "comb": SubMetaNode(NODE_LIST_combine),
    },
    [
        Con(IN, "comb", in_names=["X"]),
        Con(IN, "comb", in_names=["Y"]),
        Con(IN, "comb", in_names=["Z"]),
        Con("comb", OUT),
    ],
    required_input_type=InOutType.FLOAT,
    output_type=InOutType.VECTOR_OR_COLOR,
)

texture_ramp = MetaNode(
    "texture_ramp",
    {
        "tex1": SubMetaNode(NODE_LIST_textures),
        "ramp": SubMetaNode(NODE_LIST_ramp),
    },
    [
        Con(IN, "tex1"),
        Con("tex1", "ramp"),
        Con("ramp", OUT),
    ],
    output_type=InOutType.FLOAT,
)

texture_on_mapping = MetaNode(
    "texture_on_mapping",
    {"mapping": SubMetaNode(NODE_LIST_mapping), "tex1": SubMetaNode(NODE_LIST_textures), "math1": SubMetaNode(NODE_LIST_either_math)},
    [
        Con(IN, "tex1"),
        Con(IN, "mapping", in_names=[VECTOR]),
        Con("tex1", "math1"),
        Con("math1", "mapping", in_names=["Location", "Scale"]),
        Con("mapping", OUT),
    ],
    input_type=InOutType.VECTOR,
    output_type=InOutType.VECTOR,
    required_output_type=InOutType.VECTOR,
)

tex_on_frac_on_scale = MetaNode(
    "tex_on_frac_on_scale",
    {
        "value": SubMetaNode(NODE_LIST_value),
        "tex1": SubMetaNode(NODE_LIST_textures),
        "vec_math1": SubMetaNode(NODE_LIST_vector_math, allowed_params={"operation": ("FRACTION",)}),
        "vec_math2": SubMetaNode(NODE_LIST_vector_math, allowed_params={"operation": ("MULTIPLY", "DIVIDE")}),
    },
    [
        Con(IN, "vec_math2"),
        Con('value', "vec_math2"),  # value used to set all xyz to the same value
        Con("vec_math2", "vec_math1", in_names=['vector_0']),
        Con("vec_math1", "tex1"),
        Con("tex1", OUT),
    ],
    output_type=InOutType.COLOR,
    input_type=InOutType.VECTOR_OR_COLOR,
)


mega_structure1 = MetaNode(
    "mega_structure1",
    {
        "mapping": SubMetaNode(NODE_LIST_mapping),
        "tex0": SubMetaNode(NODE_LIST_gradient),
        "tex1": SubMetaNode(NODE_LIST_voronoi),
        "tex2": SubMetaNode(NODE_LIST_gradient),
        'value1': SubMetaNode(NODE_LIST_value),
        'value2': SubMetaNode(NODE_LIST_value, allowed_params={"Value": (-1, 0)}),
        'ramp': SubMetaNode(NODE_LIST_ramp),
        "math": SubMetaNode(NODE_LIST_math_float, allowed_params={"operation": ("MULTIPLY", "ADD")}),
        "vec_math1": SubMetaNode(NODE_LIST_vector_math, allowed_params={"operation": ("MULTIPLY",)}),
        "vec_math2": SubMetaNode(NODE_LIST_vector_math, allowed_params={"operation": ("FRACTION",)}),
        "vec_math3": SubMetaNode(NODE_LIST_vector_math, allowed_params={"operation": ("ADD",)}),
    },
    [
        Con(IN, "mapping", in_names=[VECTOR]),
        Con('mapping', "tex0"),
        Con("tex0", "vec_math1", in_names=['vector_0']),
        Con("vec_math1", "vec_math2"),
        Con("vec_math2", "vec_math3"),
        Con("vec_math3", "tex2"),
        Con("value1", "vec_math1"),
        Con("value2", "vec_math3"),
        Con("tex2", "ramp"),
        Con("ramp", "math"),
        Con("math", "tex1", in_names=['Scale']),
        Con("tex1", OUT),
    ],
    output_type=InOutType.COLOR,
    input_type=InOutType.VECTOR_OR_COLOR,
    required_output_type=InOutType.COLOR,
)

mega_structure2 = MetaNode(
    "mega_structure2",
    {
        "mapping": SubMetaNode(NODE_LIST_mapping, allowed_params={"Location": (0, 0), "Scale": (1, 1)}),
        "tex0": SubMetaNode(NODE_LIST_textures),
        "mix1": SubMetaNode(NODE_LIST_mix_vector),
        "mix2": SubMetaNode(NODE_LIST_mix_vector, allowed_params={"blend_type": ("SUBTRACT",), "B": (1, 1)}),
        'math1': SubMetaNode(NODE_LIST_math_float, allowed_params={'value_0': (0, 1), 'operation': ('SUBTRACT',)}),
        'value1': SubMetaNode(NODE_LIST_value, allowed_params={"Value": (0,1)}),
        'gradient1': SubMetaNode(NODE_LIST_gradient),
        'gradient2': SubMetaNode(NODE_LIST_gradient),
        "tex1": SubMetaNode(NODE_LIST_textures),
        "tex2": SubMetaNode(NODE_LIST_textures),
        "mix3": SubMetaNode(NODE_LIST_mix_vector),
    },
    [
        Con(IN, "mapping", in_names=[VECTOR]),
        Con('mapping', "tex0"),
        Con("mapping", "mix1", in_names=['A', 'B']),
        Con("tex0", "mix1", in_names=['A', 'B']),
        Con("mix1", "mix2", in_names=['A']),
        Con("value1", "mix1", in_names=['Factor']),
        Con("value1", "math1", in_names=['value_1']),
        Con("math1", "mix2", in_names=['Factor']),
        Con("mix2", "gradient1"),
        Con("mix2", "gradient2"),
        Con("gradient1", "tex1"),
        Con("gradient2", "tex2"),
        Con("tex1", "mix3", in_names=['A']),
        Con("tex2", "mix3", in_names=['B']),
        Con("mix3", OUT),
    ],
    output_type=InOutType.COLOR,
    input_type=InOutType.VECTOR_OR_COLOR,
    required_output_type=InOutType.COLOR,
)


ALL_META_NODES = [
    mix_textures,
    math_on_x_y_separately,
    tex_on_x_y_separately_then_math,
    tex_on_x_y_separately_then_combine,
    burn_dodge,
    combine_textures,
    duplicate_texture_to_two_textures_then_mix,
    texture_and_mix_with_self,
    combineXyz,
    texture_ramp,
    texture_on_mapping,
    tex_on_frac_on_scale,
]

MEGA_STRUCTURES = [
mega_structure1,
mega_structure2
]

if __name__ == "__main__":
    meta_nodes = [mega_structure2, burn_dodge]
    manager = MetaNetworkManager(meta_nodes, max_layers=2, n_additions=2)
    manager.generate_network()
    nm = manager.meta_network_to_flat_network()
