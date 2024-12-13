from Logic.meta_network import *


mix_textures = MetaNode(
    "mix_textures",
    {
        "mix": SubMetaNode(mix_vector),
        "tex1": SubMetaNode(textures),
        "tex2": SubMetaNode(textures),
    },
    [
        Con("tex1", "mix"),
        Con("tex2", "mix"),
        Con(IN, "tex1"),
        Con(IN, "tex2"),
        Con("mix", OUT),
    ],
    input_type=InOutType.VECTOR,
    output_type=InOutType.VECTOR,
)

math_on_x_y_separately = MetaNode(
    "math_on_x_y_separately",
    {
        "sep": SubMetaNode(sep),
        "math1": SubMetaNode(math_float, allowed_params={"operation": ("ABSOLUTE", "POWER")}),
        "math2": SubMetaNode(math_float, allowed_params={"operation": ("ABSOLUTE", "POWER")}),
        "math3": SubMetaNode(math_float, allowed_params={"operation": ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE")}),
    },
    [
        Con(IN, "sep"),
        Con("math1", "math3"),
        Con("math2", "math3"),
        Con("sep", "math2", out_names=["X"]),
        Con("sep", "math1", out_names=["Y"]),
        Con("math3", OUT),
    ],
    required_input_type=InOutType.VECTOR,
    output_type=InOutType.FLOAT,
    input_type=InOutType.VECTOR,
)

tex_on_x_y_separately_then_math = MetaNode(
    "tex_on_x_y_separately_then_math",
    {
        "sep": SubMetaNode(sep),
        "tex1": SubMetaNode(textures),
        "tex2": SubMetaNode(textures),
        "math1": SubMetaNode(math_float, allowed_params={"operation": ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE")}),
    },
    [
        Con(IN, "sep"),
        Con("tex1", "math1"),
        Con("tex2", "math1"),
        Con("sep", "tex1", out_names=["X"]),
        Con("sep", "tex2", out_names=["Y"]),
        Con("math1", OUT),
    ],
    required_input_type=InOutType.VECTOR,
    output_type=InOutType.FLOAT,
    input_type=InOutType.VECTOR,
)

tex_on_x_y_separately_then_combine = MetaNode(
    "tex_on_x_y_separately_then_combine",
    {
        "sep": SubMetaNode(sep),
        "tex1": SubMetaNode(textures),
        "tex2": SubMetaNode(textures),
        "comb1": SubMetaNode(combine),
    },
    [
        Con(IN, "sep"),
        Con("tex1", "comb1", in_names=["X"]),
        Con("tex2", "comb1", in_names=["Y"]),
        Con("sep", "tex1", out_names=["X"]),
        Con("sep", "tex2", out_names=["Y"]),
        Con("comb1", OUT),
    ],
    required_input_type=InOutType.VECTOR,
    output_type=InOutType.VECTOR,
    input_type=InOutType.VECTOR,
)


burn_dodge = MetaNode(
    "burn_dodge",
    {
        "burn": SubMetaNode(mix_vector, allowed_params={"blend_type": ("Burn",), "B": (0, 0)}),
        "dodge": SubMetaNode(mix_vector, allowed_params={"blend_type": ("Dodge",), "B": (1, 1)}),
    },
    [Con(IN, "burn", in_names=["A"]), Con("burn", "dodge", in_names=["A"]), Con("dodge", OUT)],
    required_output_type=InOutType.OUTPUT,
)

combine_textures = MetaNode(
    "combine_textures",
    {
        "tex_fac": SubMetaNode(textures),
        "mix": SubMetaNode(mix_vector),
        "ramp": SubMetaNode(ramp),
        "tex1": SubMetaNode(textures),
        "tex2": SubMetaNode(textures),
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
    output_type=InOutType.VECTOR,
)

duplicate_texture_to_two_textures_then_mix = MetaNode(
    "duplicate_texture_to_two_textures_then_mix",
    {
        "mix": SubMetaNode(mix_vector),
        "tex0": SubMetaNode(textures),
        "tex1": SubMetaNode(textures),
        "tex2": SubMetaNode(textures),
    },
    [
        Con("tex1", "mix"),
        Con("tex2", "mix"),
        Con(IN, "tex0"),
        Con("tex0", "tex2"),
        Con("tex0", "tex1"),
        Con("mix", OUT),
    ],
    output_type=InOutType.VECTOR,
)

texture_and_mix_with_self = MetaNode(
    "texture_and_mix_with_self",
    {
        "mix": SubMetaNode(mix_vector),
        "tex1": SubMetaNode(textures),
        "mapping": SubMetaNode(mapping, allowed_params={"Location": (0, 0)}),
    },
    [
        Con(IN, "mapping"),
        Con("mapping", "tex1"),
        Con("mapping", "mix", in_names=['A']),
        Con("tex1", "mix", in_names=['B']),
        Con("mix", OUT),
    ],
    output_type=InOutType.VECTOR,
)

combineXyz = MetaNode(
    "combineXyz",
    {
        "comb": SubMetaNode(combine),
    },
    [
        Con(IN, "comb", in_names=["X"]),
        Con(IN, "comb", in_names=["Y"]),
        Con(IN, "comb", in_names=["Z"]),
        Con("comb", OUT),
    ],
    required_input_type=InOutType.FLOAT,
    output_type=InOutType.VECTOR,
)

texture_ramp = MetaNode(
    "texture_ramp",
    {
        "tex1": SubMetaNode(textures),
        "ramp": SubMetaNode(ramp),
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
    {"mapping": SubMetaNode(mapping), "tex1": SubMetaNode(textures), "math1": SubMetaNode(either_math)},
    [
        Con(IN, "tex1"),
        Con(IN, "mapping", in_names=[VECTOR]),
        Con("tex1", "math1"),
        Con("math1", "mapping", in_names=["Location", "Scale"]),
        Con("mapping", OUT),
    ],
    output_type=InOutType.VECTOR,
)

tex_on_frac_on_scale = MetaNode(
    "tex_on_frac_on_scale",
    {
        "value": SubMetaNode(value),
        "tex1": SubMetaNode(textures),
        "vec_math1": SubMetaNode(vector_math, allowed_params={"operation": ("FRACTION")}),
        "vec_math2": SubMetaNode(vector_math, allowed_params={"operation": ("MULTIPLY", "DIVIDE")}),
    },
    [
        Con(IN, "vec_math2"),
        Con(value, "vec_math2"),  # value used to set all xyz to the same value
        Con("vec_math2", "vec_math1"),
        Con("vec_math1", "tex1"),
        Con("tex1", OUT),
    ],
    output_type=InOutType.VECTOR,
)

if __name__ == "__main__":
    meta_nodes = [
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
    meta_nodes = [math_on_x_y_separately, tex_on_x_y_separately_then_math, burn_dodge, texture_and_mix_with_self]
    meta_nodes = [burn_dodge]
    manager = MetaNetworkManager(meta_nodes, max_layers=2, n_additions=3)
    manager.generate_network()
    nm = manager.meta_network_to_flat_network()
