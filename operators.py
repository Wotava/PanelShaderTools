import math

import bpy
import bmesh
from mathutils import Vector, Euler, Matrix, Quaternion
from . utils import Transform
import numpy as np
ATTRIBUTE_NAME = "panel_preset_index"


def update_objects(targets):
    for obj in targets:
        if obj.type == 'MESH':
            obj.data.update()

class DebugOperator(bpy.types.Operator):
    """Debug operator"""
    bl_idname = "panels.debug"
    bl_label = "Debug"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        context.scene.panel_manager.active_preset.layers[0].print_conversion_code()
        return {'FINISHED'}


class PANELS_OP_AddPreset(bpy.types.Operator):
    """Adds new panel preset, used in UI"""
    bl_label = "Add Panel Preset"
    bl_idname = "panels.add_preset"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        context.scene.panel_manager.new_preset()
        return {'FINISHED'}


class PANELS_OP_RemovePreset(bpy.types.Operator):
    """Resets selected preset to defaults or deletes them completely if prompted"""
    bl_label = "Remove Panel Preset"
    bl_idname = "panels.remove_preset"

    destructive: bpy.props.BoolProperty(
        name="Delete from Stack",
        description="Deletes this preset from stack and updates references on all objects. "
                    "WARNING: can be slow in large scenes",
        default=False
    )

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.presets) > 0

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        manager = context.scene.panel_manager
        manager.remove_preset(destroy=self.destructive)
        if context.scene.panel_manager.target_image:
            manager.clean_image()
            manager.write_image()
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class PANELS_OP_AddLayer(bpy.types.Operator):
    """Adds new layer to active preset, used in UI"""
    bl_label = "Add Preset Layer"
    bl_idname = "panels.add_layer"

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.active_preset.layers) < context.scene.panel_manager.max_layers

    def execute(self, context):
        manager = context.scene.panel_manager
        manager.active_preset.add_layer()
        if manager.target_image:
            manager.write_image()
        else:
            self.report({'INFO'}, "No target image provided")
        update_objects(context.visible_objects)
        return {'FINISHED'}


class PANELS_OP_RemoveLayer(bpy.types.Operator):
    """Removes active layer from active preset, used in UI"""
    bl_label = "Remove Preset Layer"
    bl_idname = "panels.remove_layer"

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.active_preset.layers) > 0

    def execute(self, context):
        manager = context.scene.panel_manager
        manager.active_preset.remove_layer()
        if manager.target_image:
            manager.write_image()
        else:
            self.report({'INFO'}, "No target image provided")
        update_objects(context.visible_objects)
        return {'FINISHED'}


class PANELS_OP_MoveLayer(bpy.types.Operator):
    """Moves active layer in stack"""
    bl_label = "Move Preset Layer"
    bl_idname = "panels.move_layer"

    move_up: bpy.props.BoolProperty(
        name="Move Selection Up",
        default=True
    )

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.active_preset.layers) > 1

    def execute(self, context):
        manager = context.scene.panel_manager
        active_preset = manager.active_preset

        current_index = active_preset.active_layer_index
        target_index = current_index

        if self.move_up:
            target_index -= 1
        else:
            target_index += 1

        # Just to make sure
        if target_index < 0 or target_index > (len(active_preset.layers) - 1):
            self.report({'ERROR'}, "Index is out of range")
            return {'CANCELLED'}

        active_preset.layers.move(current_index, target_index)
        active_preset.active_layer_index = target_index

        if manager.target_image:
            manager.write_image()
        else:
            self.report({'INFO'}, "No target image provided")

        update_objects(context.visible_objects)
        return {'FINISHED'}


class PANELS_OP_DuplicateLayer(bpy.types.Operator):
    """Duplicates active layer"""
    bl_label = "Duplicate Preset Layer"
    bl_idname = "panels.duplicate_layer"

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.active_preset.layers) > 0

    def execute(self, context):
        manager = context.scene.panel_manager
        active_preset = manager.active_preset
        current_layer = active_preset.active_layer

        new_layer = active_preset.layers.add()
        new_layer.match(current_layer)

        active_preset.active_layer_index = len(active_preset.layers) - 1

        if manager.target_image:
            manager.write_image()
        else:
            self.report({'INFO'}, "No target image provided")

        update_objects(context.visible_objects)
        return {'FINISHED'}


class PANELS_OP_AssignPreset(bpy.types.Operator):
    """Assign current preset to selected faces"""
    bl_label = "Assign Preset"
    bl_idname = "panels.assign_preset"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' \
            and context.mode == 'EDIT_MESH' and context.object.data.total_face_sel > 0

    def execute(self, context):
        target = context.object.data.attributes
        active_preset_index = context.scene.panel_manager.active_preset_index

        bm = bmesh.from_edit_mesh(context.object.data)
        selected = [f.index for f in bm.faces if f.select]

        # A hacky way of writing attributes and getting selected faces
        # from OBJECT mode because in EDIT attribute data is empty
        bpy.ops.object.mode_set(mode='OBJECT')
        name = ATTRIBUTE_NAME
        if target.find(name) < 0:
            attrib = target.new(name, 'INT', 'FACE')
        else:
            attrib = target[target.find(name)]
        attrib_data = attrib.data.values()
        for index in selected:
            attrib_data[index].value = active_preset_index

        self.report({'INFO'}, f"Set preset {active_preset_index} on {len(selected)} faces")
        bm.free()
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class PANELS_OP_SelectFacesFromPreset(bpy.types.Operator):
    """Select faces that have current preset assigned to them"""
    bl_label = "Select Faces by Preset"
    bl_idname = "panels.select_by_preset"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' \
            and len(context.scene.panel_manager.presets) > 0

    def execute(self, context):
        # Switch
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')

        target_id = context.scene.panel_manager.active_preset_index

        if len(context.selected_objects) == 0:
            targets = [context.object]
        else:
            targets = context.selected_objects

        for obj in targets:
            mesh = obj.data
            bm = bmesh.from_edit_mesh(mesh)
            attrib = bm.faces.layers.int.get(ATTRIBUTE_NAME)
            if attrib is None:
                continue

            for f in bm.faces:
                if f[attrib] == target_id:
                    f.select = True
            bmesh.update_edit_mesh(mesh)

        return {'FINISHED'}


class PANELS_OP_SelectPresetFromFace(bpy.types.Operator):
    """Select preset that is assigned to the active(selected) face"""
    bl_label = "Select Preset by Face"
    bl_idname = "panels.select_by_face"
    bl_options = {'REGISTER', 'UNDO'}

    call_edit: bpy.props.BoolProperty(
        name="Call Edit Modal",
        default=False
    )

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH' \
            and context.object.data.total_face_sel == 1 and len(context.scene.panel_manager.presets) > 0

    def get_active_preset(self, context):
        mesh = context.object.data
        bm = bmesh.from_edit_mesh(mesh)
        target_id = -1

        attrib = bm.faces.layers.int.get(ATTRIBUTE_NAME)
        if attrib is None:
            self.report({'ERROR'}, "No presets assigned to this obj")
            return {'CANCELLED'}

        for f in bm.faces:
            if f.select:
                target_id = f[attrib]
                break

        return target_id

    def execute(self, context):
        if not self.call_edit:
            target_id = self.get_active_preset(context)
            if len(context.scene.panel_manager.presets) - 1 >= target_id >= 0:
                context.scene.panel_manager.active_preset_index = target_id
                self.report({'INFO'}, f"Selected preset #{target_id}")
            else:
                self.report({'ERROR'}, f"Requested preset #{target_id} is out of range")
                return {'CANCELLED'}
            bpy.ops.mesh.select_all(action='DESELECT')
        else:
            if self.start_preset:
                context.scene.panel_manager.active_preset_index = self.start_preset
                self.report({'INFO'}, f"Restoring preset {self.start_preset}, operator exit")
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        if not self.active_preset:
            self.active_preset = self.get_active_preset(context)
        context.scene.panel_manager.presets[self.active_preset].draw_panel(layout, False)

    def invoke(self, context, event):
        self.cancelled = False
        self.finished = False
        start_preset = None
        if self.call_edit:
            self.start_preset = context.scene.panel_manager.active_preset_index
            self.active_preset = self.get_active_preset(context)
            context.scene.panel_manager.active_preset_index = self.active_preset
            return context.window_manager.invoke_props_dialog(self)
        else:
            self.execute(context)
            return {'FINISHED'}


class PANELS_OP_DuplicatePreset(bpy.types.Operator):
    """Duplicates active preset and all its layers"""
    bl_label = "Duplicate Preset"
    bl_idname = "panels.duplicate_preset"

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.presets) > 0

    def execute(self, context):
        context.scene.panel_manager.duplicate_preset()
        return {'FINISHED'}


class PANELS_OP_BakePresets(bpy.types.Operator):
    """Bakes presets to the specified image"""
    bl_label = "Bake Presets"
    bl_idname = "panels.bake_presets"

    @classmethod
    def poll(cls, context):
        return context.scene.panel_manager.target_image

    def execute(self, context):
        context.scene.panel_manager.write_image()
        return {'FINISHED'}


class PANELS_OP_AdjustImage(bpy.types.Operator):
    """Adjusts image parameters"""
    bl_label = "Adjust Target Image"
    bl_idname = "panels.adjust_image"

    @classmethod
    def poll(cls, context):
        return context.scene.panel_manager.target_image

    def execute(self, context):
        res = context.scene.panel_manager.adjust_image()
        self.report({"INFO"}, res)
        return {'FINISHED'}


class PANELS_OP_DefinePlaneNormal(bpy.types.Operator):
    """Define plane normal for a layer by selecting either an edge or two vertices"""
    bl_label = "Define Plane Normal"
    bl_idname = "panels.define_plane_normal"

    target: bpy.props.StringProperty(
        name="Property to Set",
        default="plane_normal"
    )
    
    @classmethod
    def poll(cls, context):
        return context.object

    def execute(self, context):
        # find selected vertices

        selection = [vert.co for vert in self.bm.verts if vert.select]
        target_layer = context.scene.panel_manager.active_preset.active_layer
        world_matrix = context.active_object.matrix_world

        if self.target in ["plane_normal", "tile_direction_3d"]:
            selection = []
            for elem in reversed(self.bm.select_history):
                if isinstance(elem, bmesh.types.BMVert):
                    selection.append(elem.co)
                    if len(selection) == 2:
                        break

            target_location: Vector = (selection[0] - selection[1])
            target_location.normalize()
            setattr(target_layer, self.target, target_location)

            if self.target == "plane_normal" and context.scene.panel_manager.use_auto_offset:
                target_vert_co = selection[0]

                target_vert_co = world_matrix @ target_vert_co
                x = target_vert_co.dot(target_layer.plane_normal)
                y = target_layer.plane_dist_A + target_layer.plane_dist_B
                target_layer.plane_offset = (x % y) / y
                if target_layer.plane_normal.z >= 0:
                    target_layer.plane_offset = 1 - target_layer.plane_offset
                self.report({"INFO"}, f"{self.target} value set with auto-offset, operator exit")
            else:
                self.report({"INFO"}, f"{self.target} value set as normal, operator exit")
        else:
            # calculate median location
            center = sum(selection, Vector()) / len(selection)
            center = world_matrix @ center
            setattr(target_layer, self.target, center)
            self.report({"INFO"}, f"{self.target} value set, operator exit")

        self.finished = True
        return {'FINISHED'}

    def modal(self, context, event):
        if self.cancelled:
            bpy.ops.object.mode_set(mode=self.start_mode)
            return {'CANCELLED'}

        if self.finished:
            bpy.ops.object.mode_set(mode=self.start_mode)
            return {'FINISHED'}

        if event.type in {'ESC'}:
            self.cancelled = True
            return {'PASS_THROUGH'}

        if context.object.data.total_vert_sel == 2:
            self.execute(context)
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        self.start_mode = context.mode
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
        else:
            self.start_mode = 'EDIT'

        if self.target in ["plane_normal", "tile_direction_3d"] and context.object.data.total_vert_sel != 2:
            bpy.ops.mesh.select_all(action='DESELECT')

        self.cancelled = False
        self.finished = False
        self.bm = bmesh.from_edit_mesh(context.edit_object.data)

        if self.target in ["plane_normal", "tile_direction_3d"] and context.object.data.total_vert_sel == 2:
            self.execute(context)
            return {'FINISHED'}
        elif self.target != "plane_normal" and context.object.data.total_vert_sel >= 1:
            self.execute(context)
            return {'FINISHED'}

        bpy.ops.transform.translate('INVOKE_DEFAULT')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


# Preset-binding operators
class PANELS_OP_MarkAsOrigin(bpy.types.Operator):
    """Marks this object as a Preset Origin by simply adding original_transform property"""
    bl_label = "Mark as Preset Origin"
    bl_idname = "panels.mark_origin"

    @classmethod
    def poll(cls, context):
        return context.object and len(context.selected_objects) > 0

    def execute(self, context):
        obj = context.object
        obj.rotation_mode = 'QUATERNION'
        if obj.type == 'EMPTY':
            obj.empty_display_type = 'SPHERE'
        obj.lock_scale = [True, True, True]

        obj.origin_transform.from_obj(obj)
        return {'FINISHED'}


class PANELS_OP_UpdateTransform(bpy.types.Operator):
    """"""
    bl_label = "Update Transform from Origin"
    bl_idname = "panels.update_transform"

    @classmethod
    def poll(cls, context):
        return context.object and context.mode == "OBJECT" and len(context.selected_objects) > 0

    def execute(self, context):
        obj = context.object
        tr = obj.origin_transform
        if tr.scale == Vector((0.0, 0.0, 0.0)):
            self.report({'ERROR'}, "This object was not initialized as Preset Origin")

        if len(obj.children) == 0:
            self.report({'ERROR'}, "No children objects to transform")

        # Calculate transform deltas and create delta matrix
        # For some reason, rotation only works consistently from
        # (1, 0, 0, 0) quaternion and world origin position at (0, 0, 0)
        # We need to reverse location and rotation adjustment
        # before doing any changes.
        # It's not the most elegant solution, but it works
        loc_delta = Matrix.Translation(obj.location)
        loc_delta_reverse = Matrix.Translation(Vector((0.0, 0.0, 0.0)) - tr.location)

        origin_quaternion: Quaternion = tr.rotation
        current_rotation = obj.rotation_quaternion
        ref_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        rot_delta_reverse = origin_quaternion.rotation_difference(ref_quaternion).to_matrix()
        rot_delta = ref_quaternion.rotation_difference(current_rotation).to_matrix()

        # get all presets used in this hierarchy
        children = obj.children
        attrib_array = np.zeros((1), int)
        attrib_unique = np.zeros((1), int)
        for child in children:
            attrib = child.data.attributes.get(ATTRIBUTE_NAME)
            if attrib:
                attrib_array.resize((len(attrib.data)))
                attrib.data.foreach_get('value', attrib_array)
                attrib_unique = np.union1d(attrib_array, attrib_unique)

        # transform used presets
        all_presets = context.scene.panel_manager.presets
        presets = [all_presets[i] for i in attrib_unique]

        position = ['position_2d', 'position_3d']
        direction = ['tile_direction_3d', 'plane_normal']
        for preset in presets:
            for layer in preset.layers:
                if layer.panel_type.find('UV') != -1:
                    continue

                for attribute in position:
                    cur_attr = getattr(layer, attribute)
                    cur_attr = loc_delta_reverse @ cur_attr
                    cur_attr = rot_delta_reverse @ cur_attr
                    cur_attr = rot_delta @ cur_attr
                    cur_attr = loc_delta @ cur_attr
                    setattr(layer, attribute, cur_attr)

                for attribute in direction:
                    cur_attr = getattr(layer, attribute)
                    cur_attr = rot_delta_reverse @ cur_attr
                    cur_attr = rot_delta @ cur_attr
                    setattr(layer, attribute, cur_attr)

        context.scene.panel_manager.write_image()
        obj.origin_transform.from_obj(obj)

        return {'FINISHED'}


class PANELS_OP_StorageIO(bpy.types.Operator):
    """"""
    bl_label = "Storage IO"
    bl_idname = "panels.storage_io"

    action_type: bpy.props.EnumProperty(
        name="Action",
        description="",
        items=[
            ('READ_PREF_TO_SCENE', 'Load Prefs to Scene', 'Copy Presets from Addon Preferences to Scene'),
            ('WRITE_SCENE_TO_PREF', 'Write Scene to Prefs', 'Write Scene Presets to Addon Preferences'),
        ],
        default='READ_PREF_TO_SCENE',
    )

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        manager = context.scene.panel_manager
        pref_storage = context.preferences.addons[__package__].preferences.panel_presets
        scene_storage = manager.panel_presets
        if self.action_type == 'READ_PREF_TO_SCENE':
            scene_storage.clear()
            for preset in pref_storage:
                local_new = scene_storage.add()
                local_new.match(preset, name_postfix=False)
        elif self.action_type == 'WRITE_SCENE_TO_PREF':
            pref_storage.clear()
            for preset in scene_storage:
                local_new = pref_storage.add()
                local_new.match(preset, name_postfix=False)

        return {'FINISHED'}
