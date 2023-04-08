import bpy
from mathutils import Vector
MAX_LAYERS = 8


class PanelLayer(bpy.types.PropertyGroup):
    # Mandatory
    plane_normal: bpy.props.FloatVectorProperty(
        name="Plane Normal",
        subtype='XYZ',
        min=-1.0,
        max=1.0,
        default=[0.0, 0.0, 0.0]
    )
    plane_offset: bpy.props.FloatProperty(
        name="Plane Offset",
        default=0
    )
    plane_dist_A: bpy.props.FloatProperty(
        name="Plane Distance A",
        default=0
    )
    plane_dist_B: bpy.props.FloatProperty(
        name="Plane Distance B",
        default=0
    )
    decal_length: bpy.props.FloatProperty(
        name="Decal Length",
        default=0
    )
    decal_thickness: bpy.props.FloatProperty(
        name="Decal Thickness",
        default=0
    )
    leak_length: bpy.props.FloatProperty(
        name="Leak Length",
        default=0
    )

    # Skip on first layer
    use_FG_mask: bpy.props.BoolProperty(
        name="Use Previous FG Mask",
        default=False
    )
    sector_offset: bpy.props.FloatProperty(
        name="Sector Offset",
        default=0
    )
    fg_sectors: bpy.props.FloatProperty(
        name="FG Sectors",
        default=0
    )
    bg_sectors: bpy.props.FloatProperty(
        name="BG Sectors",
        default=0
    )

    # Internal usage
    use_layer: bpy.props.BoolProperty(
        name="Use This Layer",
        description="If false, this layer will be skipped when writing to image",
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
                self.plane_normal = values[i:i+3]
                i += 3
            else:
                setattr(self, item, values[i])
                i += 1

    def get_values(self):
        # TODO fix this mess too
        pixels = []
        for item in list(self.__annotations__):
            attr = getattr(self, item)
            if type(attr) is Vector:
                pixels.extend(attr)
            else:
                pixels.append(float(attr))
        return pixels


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

    def add_layer(self, values: [float] = None):
        if len(self.layers) < MAX_LAYERS:
            new_layer = self.layers.add()
            if values:
                new_layer.set_values(values)
            return new_layer
        else:
            print(f"Layer cap at {MAX_LAYERS} reached")
            return None

    def remove_layer(self, index: int = None):
        if index:
            self.layers.remove(index)
        else:
            self.layers.remove(self.active_layer)

    def get_combined_values(self) -> [float]:
        # TODO adjust
        pixels = []
        skips = 0
        for layer in self.layers:
            if layer.use_layer:
                pixels.extend(layer.get_values())
            else:
                skips += 1
        pixels.extend([0] * skips)
        return pixels

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

    def write_image(self, image: bpy.types.Image = None):
        """Write pixels to specified image"""
        if not image:
            image = self.target_image

        pos = 0
        for preset in self.scene_presets:
            for layer in preset.layers:
                pixels = layer.get_values()
                print(f"writing {pixels}")
                image.pixels[pos:pos + len(pixels)] = pixels
                pos += len(pixels)

    def read_image(self):
        """Reads presets from specified image and appends them to Scene"""
        pass

    def get_active(self) -> LayerPreset:
        """Return ref to active preset"""
        return self.scene_presets[self.active_preset]