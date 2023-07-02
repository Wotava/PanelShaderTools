import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from math import pi
from .bit_encoding import pack_manual, map_float_to_int_range, as_float, as_float_denormalized, check_mask, \
    ultra_generic_packer, encode_by_rule
from .utils import get_rotator

MAX_LAYERS = 8
PIXELS_PER_LAYER = 2
verbose = 0


def auto_update(self, context) -> None:
    if context.scene.panel_manager.use_auto_update:
        manager: LayerManager = context.scene.panel_manager
        manager.write_preset(manager.active_preset)


GLOBAL_RULESET = {
    # Special: calculated
    "distance_sum": dict(type="float", min_value=0, max_value=1000, raw=False, bits=32),
    "remap": dict(type="float", min_value=0, max_value=1, raw=False, bits=16),

    "plane_offset": dict(type="float", min_value=0, max_value=1, raw=False, bits=16),
    "decal_thickness": dict(type="float", min_value=0, max_value=4, raw=False, bits=12),
    "use_FG_mask": dict(type="bool", min_value=0, max_value=1, raw=True, bits=1),
    "fg_sectors": dict(type="int", min_value=0, max_value=31, raw=True, bits=5),
    "bg_sectors": dict(type="int", min_value=0, max_value=31, raw=True, bits=5),
    "sector_offset": dict(type="int", min_value=0, max_value=63, raw=True, bits=6),
    "panel_type": dict(type="int", min_value=0, max_value=10, raw=True, bits=4),

    # Special: Vec3 position
    "3D_pos.x": dict(type="vec3", min_value=-128, max_value=128, raw=False, bits=32),
    "3D_pos.y": dict(type="vec3", min_value=-128, max_value=128, raw=False, bits=32),
    "3D_pos.z": dict(type="vec3", min_value=-128, max_value=128, raw=False, bits=32),

    # Special: Vec2 position (two values input)
    "2D_pos.x": dict(type="vec2", min_value=-128, max_value=128, raw=False, bits=32),
    "2D_pos.y": dict(type="vec2", min_value=-128, max_value=128, raw=False, bits=32),

    # Special: directionless normal rotator
    "normal_yaw": dict(type="float", min_value=-(pi / 2), max_value=(pi / 2), raw=False, bits=24),
    "normal_pitch": dict(type="float", min_value=-(pi / 2), max_value=(pi / 2), raw=False, bits=24),

    # Special: direction vector rotator
    "tile_dir_yaw": dict(type="float", min_value=-pi, max_value=pi, raw=False, bits=25),
    "tile_dir_pitch": dict(type="float", min_value=(-pi / 2), max_value=(pi / 2), raw=False, bits=23),

    "fan_divisions": dict(type="int", min_value=2, max_value=64, raw=True, bits=6),
    "angle_offset": dict(type="float", min_value=0, max_value=(pi / 2), raw=False, bits=12),
    "remap_angular": dict(type="float", min_value=0, max_value=(pi / 2), raw=False, bits=12),

    "tile_direction_2d": dict(type="float", min_value=0, max_value=1, raw=False, bits=32),
    "tile_distance": dict(type="float", min_value=0, max_value=1000, raw=False, bits=16),
    "line_direction": dict(type="float", min_value=0, max_value=1, raw=False, bits=32),

    # Utility
    "flip_block": dict(type="bool", min_value=0, max_value=1, raw=True, bits=1)
}


class PanelLayer(bpy.types.PropertyGroup):
    # Mandatory
    plane_normal: bpy.props.FloatVectorProperty(
        name="Plane Normal",
        subtype='XYZ',
        min=-1.0,
        max=1.0,
        update=auto_update,
        default=[0.0, 0.0, 1.0]
    )
    plane_offset: bpy.props.FloatProperty(
        name="Plane Offset",
        min=0.0,
        max=1.0,
        update=auto_update,
        default=0
    )
    plane_dist_A: bpy.props.FloatProperty(
        name="Plane Distance A",
        min=0.0,
        max=50.0,
        update=auto_update,
        default=1
    )
    plane_dist_B: bpy.props.FloatProperty(
        name="Plane Distance B",
        min=0.0,
        max=50.0,
        update=auto_update,
        default=1
    )
    decal_length: bpy.props.FloatProperty(
        name="Decal Length",
        min=0.0,
        max=32.0,
        update=auto_update,
        default=1
    )
    decal_thickness: bpy.props.FloatProperty(
        name="Decal Thickness",
        min=0.0,
        max=4.0,
        update=auto_update,
        default=0.5
    )

    # Skip on first layer
    use_FG_mask: bpy.props.BoolProperty(
        name="Use Previous FG Mask",
        update=auto_update,
        default=False
    )
    sector_offset: bpy.props.IntProperty(
        name="Sector Offset",
        min=0,
        max=63,
        update=auto_update,
        default=0
    )
    fg_sectors: bpy.props.IntProperty(
        name="FG Sectors",
        min=0,
        max=31,
        update=auto_update,
        default=0
    )
    bg_sectors: bpy.props.IntProperty(
        name="BG Sectors",
        min=0,
        max=31,
        update=auto_update,
        default=0
    )

    fan_divisions: bpy.props.IntProperty(
        name="Fan Divisions",
        min=2,
        max=64,
        update=auto_update,
        default=0
    )
    angle_offset: bpy.props.FloatProperty(
        name="Angle Offset",
        min=0.0,
        max=1.0,
        update=auto_update,
        default=0.0
    )
    remap_angular: bpy.props.FloatProperty(
        name="Remap Angular",
        min=0.0,
        max=1.0,
        update=auto_update,
        default=0.0
    )

    position_2d: bpy.props.FloatVectorProperty(
        name="2D Position",
        subtype='XYZ',
        min=-128000.0,
        max=128000.0,
        size=2,
        update=auto_update,
        default=[0.0, 1.0]
    )
    position_3d: bpy.props.FloatVectorProperty(
        name="3D Position",
        subtype='XYZ',
        min=-128000.0,
        max=128000.0,
        update=auto_update,
        default=[0.0, 0.0, 1.0]
    )

    tile_direction_2d: bpy.props.FloatProperty(
        name="2D Tile Direction (Angle)",
        min=0.0,
        max=1.0,
        update=auto_update,
        default=0.0
    )
    tile_direction_3d: bpy.props.FloatVectorProperty(
        name="3D Tile Direction",
        subtype='XYZ',
        min=-1.0,
        max=1.0,
        update=auto_update,
        default=[0.0, 0.0, 1.0]
    )
    tile_distance: bpy.props.FloatProperty(
        name="Tile Distance",
        min=0.0,
        max=1000.0,
        update=auto_update,
        default=0.0
    )

    line_direction: bpy.props.FloatProperty(
        name="Line Direction (Angle)",
        min=0.0,
        max=1.0,
        update=auto_update,
        default=0.0
    )

    panel_type: bpy.props.EnumProperty(
        name="Panel Type",
        items=[
            ('Planar', 'Planar Panel', ''),
            ('Fan', 'Fan Panel', ''),
            ('UV Fan', 'UV Fan Panel', ''),
            ('Normalized Fan', 'Normalized Fan Panel', ''),
            ('UV Normalized Fan', 'UV Normalized Fan Panel', ''),
            ('UV Lines', 'UV Lines Panel', ''),
            ('UV Circles', 'UV Circles Panel', ''),
            ('UV Circles Tiled', 'UV Circles Tiled Panel', ''),
            ('Spherical', 'Spherical Panel', ''),
            ('Spherical Tiled', 'Spherical Tiled Panel', ''),
            ('Cylinder', 'Cylinder Panel', ''),
        ],
        update=auto_update,
        default='UV Lines'
    )

    # Internal usage
    use_layer: bpy.props.BoolProperty(
        name="Use This Layer",
        description="If false, this layer will be skipped when writing to image",
        update=auto_update,
        default=True
    )

    # This dictionary defines which values are used for UI and packing
    sets = {
        'Planar': ["plane_normal", "distance_sum", "plane_offset", "decal_thickness", "use_FG_mask", "fg_sectors",
                   "bg_sectors", "sector_offset", "panel_type"],
        'Fan': ["decal_thickness", "use_FG_mask", "fg_sectors", "bg_sectors", "sector_offset", "position_2d",
                "plane_normal", "fan_divisions", "angle_offset", "remap_angular", "panel_type"],
        'Normalized Fan': ["distance_sum", "plane_offset", "decal_thickness", "use_FG_mask",
                           "fg_sectors", "bg_sectors", "sector_offset", "position_2d", "plane_normal",
                           "fan_divisions", "angle_offset", "remap_angular", "panel_type"],
        'UV Normalized Fan': ["distance_sum", "plane_offset", "decal_thickness", "use_FG_mask", "fg_sectors",
                              "bg_sectors", "sector_offset", "position_2d", "fan_divisions", "angle_offset",
                              "remap_angular", "panel_type"],
        'UV Lines': ["distance_sum", "plane_offset", "decal_thickness", "use_FG_mask", "fg_sectors", "bg_sectors",
                     "sector_offset", "line_direction", "panel_type"],
        'UV Circles': ["distance_sum", "plane_offset", "decal_thickness", "use_FG_mask", "fg_sectors", "bg_sectors",
                       "sector_offset", "position_2d", "panel_type"],
        'UV Circles Tiled': ["distance_sum", "decal_thickness", "use_FG_mask",
                             "fg_sectors", "bg_sectors", "sector_offset", "position_2d", "tile_direction_2d",
                             "tile_distance", "panel_type"],
        'UV Fan': ["decal_thickness", "use_FG_mask", "fg_sectors", "bg_sectors", "sector_offset", "position_2d",
                   "fan_divisions", "angle_offset", "remap_angular", "panel_type"],
        'Spherical': ["distance_sum", "plane_offset", "decal_thickness", "use_FG_mask", "fg_sectors", "bg_sectors",
                      "sector_offset", "position_3d", "panel_type"],
        'Spherical Tiled': ["distance_sum", "decal_thickness", "use_FG_mask", "fg_sectors", "bg_sectors",
                            "sector_offset", "tile_direction_3d", "tile_distance", "position_3d", "panel_type"],
        'Cylinder': ["distance_sum", "plane_offset", "decal_thickness", "use_FG_mask",
                     "fg_sectors", "bg_sectors", "sector_offset", "position_2d", "plane_normal", "panel_type"],
    }

    def __len__(self):
        return len(self.__annotations__) - 1

    def set_values(self, values: [float]):
        # TODO fix this mess
        if len(values) < len(self) + 2:
            values.extend([0] * ((len(self) + 2) - len(values)))
        targets = list(self.__annotations__)
        i = 0
        for item in targets:
            if item == 'plane_normal':
                self.plane_normal = values[i:i + 3]
                i += 3
            else:
                setattr(self, item, values[i])
                i += 1

    def match(self, target: 'LayerPreset'):
        """Copy all parameters from provided layer, thus matching it"""
        targets = list(self.__annotations__)
        i = 0
        for item in targets:
            if item == 'plane_normal':
                self.plane_normal = getattr(target, item)
                self.plane_normal = self.plane_normal.copy()
            else:
                setattr(self, item, getattr(target, item))
            i += 1
        self.name = target.name + " copy"

    def get_values(self) -> []:
        """Returns a list of layer values matching current panel type paired with their names
         in format [value, value_name]"""
        values = []
        for val in self.sets[self.panel_type]:
            if val == 'distance_sum' or val == 'remap':
                distance_sum = self.plane_dist_A + self.plane_dist_B
                if distance_sum != 0:
                    distance_remap = self.plane_dist_B / distance_sum
                else:
                    distance_remap = 0

                if val == "distance_sum":
                    values.append([distance_sum, "distance_sum"])
                    values.append([distance_remap, "remap"])
                else:
                    values.append([distance_remap, "remap"])
            elif val == "position_3d":
                values.append([self.position_3d.x, "3D_pos.x"])
                values.append([self.position_3d.y, "3D_pos.y"])
                values.append([self.position_3d.z, "3D_pos.z"])
            elif val == "position_2d":
                if self.panel_type in ['Cylinder', 'Fan', 'Normalized Fan']:
                    position: Vector = self.position_2d
                    normal: Vector = self.plane_normal.normalized()

                    if abs(Vector((0.0, 0.0, 1.0)).dot(normal)) >= 1 - 0.001:
                        up_vector = Vector((0.0, 1.0, 0.0))
                    else:
                        up_vector = Vector((0.0, 0.0, 1.0))

                    local_y = ((normal.cross(up_vector)).cross(normal)).normalized()
                    local_x = normal.cross(local_y)
                    local_pos = Vector((position.dot(local_x), position.dot(local_y)))

                    values.append([local_pos.x, "2D_pos.x"])
                    values.append([local_pos.y, "2D_pos.y"])
                else:
                    values.append([self.position_2d.x, "2D_pos.x"])
                    values.append([self.position_2d.y, "2D_pos.y"])
            elif val == "plane_normal":
                yaw, pitch = get_rotator(self.plane_normal.normalized())
                values.append([yaw, "normal_yaw"])
                values.append([pitch, "normal_pitch"])
            elif val == "tile_direction_3d":
                yaw, pitch = get_rotator(self.tile_direction_3d.normalized(), limit_range=False)
                values.append([yaw, "tile_dir_yaw"])
                values.append([pitch, "tile_dir_pitch"])
            elif val == "use_FG_mask":
                values.append([int(self.use_FG_mask), val])
            elif val == "panel_type":
                values.append([list(self.sets.keys()).index(self.panel_type), val])
            else:
                values.append([getattr(self, val), val])
        return values

    def get_pixel(self) -> [float]:
        """Returns a list of floats with encoded layer parameters. Encoding rules are taken from GLOBAL_RULESET
         global variable"""
        values = self.get_values()
        values = encode_by_rule(values, GLOBAL_RULESET)
        return ultra_generic_packer(values, validate=True)

    def print_conversion_code(self, all_types=False) -> None:
        """Prints generated GLSL float decoding code for current panel preset. Pretty much the same as get_pixel(), but
         passes generate_code=True to the generic packer. A bit hacky approach for now"""
        init_panel_type = self.panel_type
        if all_types:
            for p_type in self.sets:
                self.panel_type = p_type
                print(f"\n// GENERATED CODE FOR PANEl TYPE {p_type}")
                values = encode_by_rule(self.get_values(), GLOBAL_RULESET)
                ultra_generic_packer(values, generate_code=True)
        else:
            print(f"\n// GENERATED CODE FOR PANEl TYPE {self.panel_type}")
            values = encode_by_rule(self.get_values(), GLOBAL_RULESET)
            ultra_generic_packer(values, generate_code=True)

        self.panel_type = init_panel_type

    def print_values(self):
        for item in list(self.__annotations__):
            attr = getattr(self, item)
            print(item, attr)

    def draw_panel(self, layout, show_operators=True):
        box = layout.box()
        blend_box = layout.box()
        row = box.row()
        row.prop(self, "use_layer", text="Enable Layer")

        row = box.row()
        row.prop(self, "panel_type")

        i = 0
        for prop in self.sets[self.panel_type]:
            if i == 0 or i >= 3:
                row = box.row(align=True)
                i = 0

            if prop == 'panel_type':
                continue
            elif prop in ["distance_sum", "remap"]:
                row.prop(self, "plane_dist_A")
                row.prop(self, "plane_dist_B")
                i += 2
            elif prop in ["use_FG_mask", "fg_sectors", "bg_sectors", "sector_offset"]:
                blend_box.prop(self, prop)
            elif prop in ["plane_normal", "position_2d", "position_3d", "tile_direction_2d", "tile_direction_3d"]:
                row = box.row(align=True)
                row.prop(self, prop)
                i += 3
            else:
                row.prop(self, prop)
                i += 1
        return

    def reset_to_default(self):
        targets = list(self.__annotations__)
        for item in targets:
            self.property_unset(item)


class LayerPreset(bpy.types.PropertyGroup):
    # This class handles writing layers to images and swapping their order
    object_ID: bpy.props.IntProperty(
        name="Owning Object ID",
        description="Used to determine pixel offset from the start of the image for current object",
        default=0
    )

    layers: bpy.props.CollectionProperty(
        type=PanelLayer,
        name="Panel Layers",
        description="Layers associated with this object"
    )
    active_layer: bpy.props.IntProperty(
        name="Active Layer Index",
        default=0
    )

    def add_layer(self, match_target: PanelLayer = None):
        if len(self.layers) < MAX_LAYERS:
            new_layer = self.layers.add()
            if match_target:
                new_layer.match(match_target)
            else:
                new_layer.name = "Layer Preset №" + str(len(self.layers) - 1)
            return new_layer
        else:
            print(f"Layer cap at {MAX_LAYERS} reached")
            return None

    def remove_layer(self, index: int = -1):
        if index > -1:
            self.layers.remove(index)
        else:
            self.layers.remove(self.active_layer)

    def get_pixel_strip(self) -> [float]:
        """Returns a list of floats to compose a pixel from. List is filled with empty values until it matches the
         target length for a preset specified by MAX_LAYERS and PIXELS_PER_LAYER global variables."""
        pixels = []
        target_length = MAX_LAYERS * PIXELS_PER_LAYER * 4
        for layer in self.layers:
            if layer.use_layer:
                pixels.extend(layer.get_pixel())
        pixels.extend([0] * (target_length - len(pixels)))
        return pixels

    def match(self, target: 'LayerPreset'):
        """Copies all layers from target"""
        target_len = len(target.layers)
        for i in range(0, target_len):
            self.add_layer(target.layers[i])
        self.name = target.name + " copy"
        self.active_layer = target.active_layer

    def get_active(self) -> PanelLayer:
        return self.layers[self.active_layer]

    def clean(self):
        self.active_layer = -1
        for i in range(len(self.layers), -1, -1):
            self.remove_layer(i)

    def draw_panel(self, layout, show_operators=True):
        # LAYERS
        row = layout.row()
        row.template_list("DATA_UL_PanelLayer", "", self, "layers", self,
                          "active_layer")
        col = row.column(align=True)
        col.operator("panels.add_layer", icon='ADD', text="")
        col.operator("panels.remove_layer", icon='REMOVE', text="")
        col.separator()

        col.operator("panels.duplicate_layer", icon='DUPLICATE', text="")
        col.separator()

        if len(self.layers) > 2 and self.active_layer > 0:
            op = col.operator("panels.move_layer", icon='TRIA_UP', text="")
            op.move_up = True
        if len(self.layers) > 2 and self.active_layer < (len(self.layers) - 1):
            op = col.operator("panels.move_layer", icon='TRIA_DOWN', text="")
            op.move_up = False

        if len(self.layers) > 0:
            current_layer = self.get_active()
            current_layer.draw_panel(layout, show_operators)


class LayerManager(bpy.types.PropertyGroup):
    """Controls all layer presets in the scene and handles images"""
    scene_presets: bpy.props.CollectionProperty(
        type=LayerPreset,
        name="Scene Presets",
        description="All Panel Presets in this Scene"
    )
    active_preset: bpy.props.IntProperty(
        name="Active Preset Index",
        default=0
    )
    target_image: bpy.props.PointerProperty(
        type=bpy.types.Image,
        name="Image to write data to"
    )
    max_layers: bpy.props.IntProperty(
        name="Max Amount of Layers per Preset",
        default=8
    )
    use_auto_update: bpy.props.BoolProperty(
        name="Enable Auto-Update",
        description="Enables auto update of target image when any layer parameter is changed",
        default=False
    )
    use_auto_offset: bpy.props.BoolProperty(
        name="Update Offset on Normal-set",
        description="Enables auto update of plane offset to match selected target vertices when setting panel normal",
        default=True
    )

    def new_preset(self):
        """Create new preset with given name"""
        return self.scene_presets.add()

    def remove_preset(self, index: int = None, destroy=False):
        """Remove and do something with linked objects"""
        if index:
            target = index
        else:
            target = self.active_preset

        if destroy:
            attrib_array = np.zeros((1), int)
            for obj in bpy.data.objects:
                if obj.type != 'MESH':
                    continue

                attrib = obj.data.attributes.get('panel_preset_index')
                if not attrib:
                    continue

                attrib = attrib.data
                attrib_array.resize((len(obj.data.polygons)))
                attrib.foreach_get('value', attrib_array)
                attrib_array[attrib_array == target] = 0
                attrib_array[attrib_array > target] -= 1
                attrib.foreach_set('value', attrib_array.tolist())
                obj.data.update()
            self.scene_presets.remove(target)
        else:
            self.scene_presets[target].clean()

    def duplicate_preset(self, index: int):
        """Duplicate preset at the given index"""
        pass

    def write_preset(self, preset_id: int, image: bpy.types.Image = None) -> None:
        """Used update some preset on-the-go when values are changed"""
        if not image:
            image = self.target_image

        preset: LayerPreset = self.scene_presets[preset_id]
        pixels = preset.get_pixel_strip()
        start_pos = preset_id * len(pixels)
        end_pos = start_pos + len(pixels)
        image.pixels[start_pos: end_pos] = pixels
        image.update()

    def write_image(self, image: bpy.types.Image = None) -> None:
        """Write pixels to specified image"""
        if not image:
            image = self.target_image

        for i in range(0, len(self.scene_presets)):
            self.write_preset(i, image)

    def clean_image(self, image: bpy.types.Image = None) -> None:
        if not image:
            image = self.target_image
        for i in range(0, len(image.pixels)):
            image.pixels[i] = 0.0

    def read_image(self):
        """Reads presets from specified image and appends them to Scene"""
        pass

    def check_image(self, layout=None) -> bool:
        box = None

        check_passed = True
        if self.target_image.use_half_precision:
            check_passed = False
            if layout:
                if not box:
                    box = layout.box()
                box.row(align=True).label(text="Using half-precision", icon='ERROR')
        if self.target_image.colorspace_settings.name != 'Non-Color':
            check_passed = False
            if layout:
                if not box:
                    box = layout.box()
                box.row(align=True).label(text="Incorrect color space", icon='ERROR')

        if pow(self.target_image.size[0], 2) < (len(self.scene_presets) * 8):
            check_passed = False
            if layout:
                if not box:
                    box = layout.box()
                if len(self.target_image.pixels) / 4 > 256:
                    box.row(align=True).label(text="256 presets exceeded!", icon='ERROR')
                else:
                    box.row(align=True).label(text="Image size is too small", icon='ERROR')

        if layout and not check_passed:
            box.row(align=True).operator("panels.adjust_image", text="Adjust", icon='MODIFIER_DATA')

        return check_passed

    def adjust_image(self) -> str:
        changes = "Adjusted: "
        if self.target_image.use_half_precision:
            self.target_image.use_half_precision = False
            changes += " half precision,"
        if self.target_image.colorspace_settings.name != 'Non-Color':
            self.target_image.colorspace_settings.name = 'Non-Color'
            changes += " color space,"

        if pow(self.target_image.size[0], 2) < (len(self.scene_presets) * 8):
            for i in range(0, 9):
                if pow(2, i) > pow(self.target_image.size[0], 2):
                    changes += f" image size {self.target_image.size[0]}x{self.target_image.size[0]} " \
                               f"-> {pow(2, i)}x{pow(2, i)}"
                    self.target_image.scale(1, 1)
                    self.target_image.pixels[0:4] = [0, 0, 0, 1]
                    self.target_image.scale(pow(2, i), pow(2, i))
                    break

        return changes

    def get_active(self) -> LayerPreset:
        """Return ref to active preset"""
        return self.scene_presets[self.active_preset]

    def draw_panel(self, layout):
        active_preset = self.get_active()

        # Target Image selection
        row = layout.row()
        col = row.column(align=True)
        col.scale_x = 0.6
        col.label(text="Target Image")
        col = row.column(align=True)
        col.template_ID(self, "target_image", new="image.new", open="image.open")
        if self.target_image:
            self.check_image(layout)
        layout.row().operator("panels.bake_presets")
        layout.row().operator("panels.assign_preset")

        row = layout.row(align=True)
        row.operator("panels.select_by_preset")
        op = row.operator("panels.select_by_face")
        op.call_edit = False
        op = row.operator("panels.select_by_face", text="E")
        op.call_edit = True
        layout.row().prop(self, "use_auto_update", icon='FILE_REFRESH')
        layout.row().prop(self, "use_auto_offset", icon='MOD_LENGTH')

        active_preset.draw_panel(layout)
