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
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.object.camera_add(location=(0, 0, 2.5))
    bpy.context.scene.camera = bpy.context.object
    bpy.data.objects["Camera"].rotation_euler = (0, 0, 0)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 0)


def settings_for_texture_generation(path, resolution=512):
    scene = bpy.data.scenes["Scene"]
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.color_mode = "BW"
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"  # ('BLENDER_EEVEE_NEXT', 'BLENDER_WORKBENCH', 'CYCLES')
    bpy.context.scene.render.filepath = path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.eevee.use_shadows = False
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.view_settings.view_transform = "Standard"


class NodesAdder(object):
    def __init__(self, tree, node_distance=250):
        self.x_location = 0
        self.tree = tree
        self.node_distance = node_distance

    def create_node(self, node_type, **kwargs):
        node = self.tree.nodes.new(type=node_type)
        node.location.x = self.x_location
        self.x_location += self.node_distance
        for attr, value in kwargs.items():
            setattr(node, attr, value)
        return node
