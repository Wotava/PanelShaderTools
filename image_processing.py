import bpy
from mathutils import Vector
from math import pi
from .bit_encoding import pack_manual, map_float_to_int_range, as_float, as_float_denormalized, check_mask, rotator_unpack_test
from .utils import get_rotator

MAX_LAYERS = 8
verbose = 0


def auto_update(self, context) -> None:
    if context.scene.panel_manager.use_auto_update:
        manager: LayerManager = context.scene.panel_manager
        manager.write_preset(manager.active_preset)


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
        max=31,
        update=auto_update,
        default=0
    )
    fg_sectors: bpy.props.IntProperty(
        name="FG Sectors",
        min=0,
        max=15,
        update=auto_update,
        default=0
    )
    bg_sectors: bpy.props.IntProperty(
        name="BG Sectors",
        min=0,
        max=15,
        update=auto_update,
        default=0
    )

    # Internal usage
    use_layer: bpy.props.BoolProperty(
        name="Use This Layer",
        description="If false, this layer will be skipped when writing to image",
        update=auto_update,
        default=True
    )

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

    def get_pixel(self) -> [float]:
        """Returns RGBA pixel with encoded values"""
        # Prepare normal vector as a Yaw-Pitch rotator with signed ints
        # Encoded as follows:
        # 1 bit Yaw-sign, 15 bit Yaw rotation in radians in range [-Pi/2, Pi/2]
        # 1 bit Pitch-sign, 15 bit Pitch rotation in radians in range [-Pi/2, Pi/2]
        yaw, pitch = get_rotator(self.plane_normal.normalized())
        r_channel = [yaw < 0, map_float_to_int_range(abs(yaw), 0, pi / 2, 15),
                     pitch < 0, map_float_to_int_range(abs(pitch), 0, pi / 2, 15)]
        r_packed = pack_manual(r_channel, [1, 15, 1, 15])
        r_channel, r_flip = as_float_denormalized(r_packed)
        if verbose > 2:
            print(f"rch bits {bin(r_packed)} ({len(bin(r_packed)) - 2} bit)")

        distance_sum = self.plane_dist_A + self.plane_dist_B
        if distance_sum != 0:
            distance_remap = self.plane_dist_B / distance_sum
        else:
            distance_remap = 0
        g_channel = [map_float_to_int_range(distance_sum, 0, 100, 16),
                     map_float_to_int_range(distance_remap, 0, 1, 10),
                     map_float_to_int_range(self.decal_length, 0, 32, 6)]

        g_packed = pack_manual(g_channel, [16, 10, 6])
        if verbose > 2:
            print(f"gch bits {bin(g_packed)} ({len(bin(g_packed)) - 2} bit)")
        g_channel, g_flip = as_float_denormalized(g_packed)

        b_channel = [int(g_flip), int(r_flip),
                     map_float_to_int_range(self.plane_offset, 0, 1, 10),
                     map_float_to_int_range(self.decal_thickness, 0, 4, 6),
                     self.fg_sectors, self.bg_sectors, self.sector_offset, 0]

        # Test if this value has non-zero (000000) exponent before packing
        # and write a flip-bit, otherwise blender will simply round it all
        # down to zero when reading texture
        b_packed = pack_manual(b_channel, [1, 1, 10, 6, 4, 4, 5, 1])
        if check_mask(b_packed):
            b_flip = True
            b_packed ^= 1
            b_channel = as_float_denormalized(b_packed)[0]
        else:
            b_flip = False
            b_channel = as_float(b_packed)

        if verbose > 2:
            print(f"bch bits {bin(b_packed)} ({len(bin(b_packed)) - 2} bit)")

        a_channel = [int(self.use_FG_mask), 0]
        a_packed = pack_manual(a_channel, [1, 1])

        if check_mask(a_packed):
            a_packed ^= 1
            a_channel = as_float_denormalized(a_packed)[0]
        else:
            a_channel = as_float(a_packed)

        if verbose > 2:
            print(f"ach bits {bin(a_packed)} ({len(bin(a_packed)) - 2} bit)")

        if verbose > 1:
            print(f"flips r{r_flip} g{g_flip} b{b_flip}")
            print(f"writing {r_channel}, {g_channel}, {b_channel}, {a_channel}")
        return [r_channel, g_channel, b_channel, a_channel]

    def get_values(self):
        # TODO fix this mess too
        pixels = []
        for item in list(self.__annotations__):
            attr = getattr(self, item)
            print(attr)
            if type(attr) is Vector:
                pixels.extend(attr)
            else:
                pixels.append(float(attr))
        return pixels

    def print_values(self):
        for item in list(self.__annotations__):
            attr = getattr(self, item)
            print(item, attr)


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

    def remove_layer(self, index: int = None):
        if index:
            self.layers.remove(index)
        else:
            self.layers.remove(self.active_layer)

    def get_pixel_strip(self) -> [float]:
        # TODO adjust
        pixels = []
        target_length = 32
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

    def remove_preset(self, index: int = None):
        """Remove  and do something with linked objects"""
        if index:
            self.scene_presets.remove(index)
        else:
            self.scene_presets.remove(self.active_preset)

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
