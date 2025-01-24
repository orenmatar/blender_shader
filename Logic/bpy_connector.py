import bpy


def purge_orphans():
    """
    Remove all orphan data blocks

    see this from more info:
    https://youtu.be/3rNqVPtbhzc?t=149
    """
    if bpy.app.version >= (3, 0, 0):
        # run this only for Blender versions 3.0 and higher
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    else:
        # run this only for Blender versions lower than 3.0
        # call purge_orphans() recursively until there are no more orphan data blocks to purge
        result = bpy.ops.outliner.orphans_purge()
        if result.pop() != "CANCELLED":
            purge_orphans()


def clean_scene():
    """
    Removing all of the objects, collection, materials, particles,
    textures, images, curves, meshes, actions, nodes, and worlds from the scene

    Checkout this video explanation with example

    "How to clean the scene with Python in Blender (with examples)"
    https://youtu.be/3rNqVPtbhzc
    """
    # make sure the active object is not in Edit Mode
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.editmode_toggle()

    # make sure non of the objects are hidden from the viewport, selection, or disabled
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False

    # select all the object and delete them (just like pressing A + X + D in the viewport)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # find all the collections and remove them
    collection_names = [col.name for col in bpy.data.collections]
    for name in collection_names:
        bpy.data.collections.remove(bpy.data.collections[name])

    # in the case when you modify the world shader
    # delete and recreate the world object
    world_names = [world.name for world in bpy.data.worlds]
    for name in world_names:
        bpy.data.worlds.remove(bpy.data.worlds[name])
    # create a new world data block
    bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds["World"]

    purge_orphans()


def set_for_texture_generation():
    """
    Set the scene for texture generation
    Add a plane, set the camera to be above it looking down, and set the background to be black
    """
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.camera_add(location=(0, 0, 2.75))  # at 2.75 we see the entire plane
    bpy.context.scene.camera = bpy.context.object
    bpy.data.objects["Camera"].rotation_euler = (0, 0, 0)
    # Ensure the World is using nodes and making it dark
    bpy.context.scene.world.use_nodes = True
    node_tree = bpy.context.scene.world.node_tree
    background_node = node_tree.nodes["Background"]
    background_node.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # RGBA (black color)
    # Set world strength to 0 to make it fully dark
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0  # Strength


def set_file_path(file_path):
    """
    Set the file path for the render
    :param file_path:
    """
    bpy.context.scene.render.filepath = file_path


def settings_for_texture_generation(resolution=512):
    """
    Set the render settings for texture generation
    :param resolution:
    """
    scene = bpy.data.scenes["Scene"]
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.color_mode = "BW"  # black and white for now
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.use_simplify = True  # to speed up the rendering
    bpy.context.scene.eevee.use_shadows = False  # to speed up the rendering
    bpy.context.scene.eevee.taa_render_samples = 1  # to speed up the rendering
    bpy.context.scene.view_settings.view_transform = "Standard"  # to speed up the rendering


class NodesAdder(object):
    def __init__(self, tree, node_distance=250):
        self.x_location = 0  # keeps track of the x location of the node, so they don't overlap
        self.tree = tree
        self.node_distance = node_distance

    def create_node(self, node_type: str, **kwargs):
        node = self.tree.nodes.new(type=node_type)
        node.location.x = self.x_location
        self.x_location += self.node_distance
        for attr, value in kwargs.items():
            setattr(node, attr, value)
        return node


def generate_image(nm, image_path):
    code = nm.generate_code(with_initialization_code=False)
    clean_scene()
    set_for_texture_generation()
    settings_for_texture_generation(resolution=512)
    set_file_path(image_path)
    material = bpy.data.materials.new(name="my_material")
    material.use_nodes = True
    bpy.data.objects["Plane"].data.materials.append(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    [nodes.remove(n) for n in nodes]
    node_tree = material.node_tree
    nodes_adder = NodesAdder(material.node_tree)
    exec(code)
    bpy.ops.render.render(write_still=True)

