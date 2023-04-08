import bpy
import bmesh
from mathutils import Vector
import numpy as np

class DebugOperator(bpy.types.Operator):
    """Debug operator"""
    bl_idname = "material.debug_panel_shader"
    bl_label = "Scale Oversize UVs"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.view.show_developer_ui

    def execute(self, context):
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
        context.scene.panel_manager.active_preset = len(context.scene.panel_manager.scene_presets) - 1
        return {'FINISHED'}


class PANELS_OP_RemovePreset(bpy.types.Operator):
    """Adds new panel preset, used in UI"""
    bl_label = "Remove Panel Preset"
    bl_idname = "panels.remove_preset"

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.scene_presets) > 0

    def execute(self, context):
        context.scene.panel_manager.remove_preset()
        if context.scene.panel_manager.active_preset > 0:
            context.scene.panel_manager.active_preset -= 1
        return {'FINISHED'}


class PANELS_OP_AddLayer(bpy.types.Operator):
    """Adds new layer to active preset, used in UI"""
    bl_label = "Add Preset Layer"
    bl_idname = "panels.add_layer"

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.get_active().layers) < context.scene.panel_manager.max_layers

    def execute(self, context):
        # TODO make it insert layer next to active
        context.scene.panel_manager.get_active().add_layer()
        context.scene.panel_manager.get_active().active_layer = len(context.scene.panel_manager.get_active().layers) - 1
        return {'FINISHED'}


class PANELS_OP_RemoveLayer(bpy.types.Operator):
    """Removes active layer from active preset, used in UI"""
    bl_label = "Remove Preset Layer"
    bl_idname = "panels.remove_layer"

    @classmethod
    def poll(cls, context):
        return len(context.scene.panel_manager.get_active().layers) > 0

    def execute(self, context):
        # TODO make it insert layer next to active
        context.scene.panel_manager.get_active().remove_layer()
        context.scene.panel_manager.get_active().active_layer = len(context.scene.panel_manager.get_active().layers) - 1
        return {'FINISHED'}


class PANELS_OP_AssignPreset(bpy.types.Operator):
    """Assigns preset to the selected faces"""
    bl_label = "Assign Preset"
    bl_idname = "panels.assign_preset"

    @classmethod
    def poll(cls, context):
        return context.active_object

    def execute(self, context):
        pass


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


class PANELS_OP_DefinePlaneNormal(bpy.types.Operator):
    """Define plane normal for a layer by selecting either an edge or two vertices"""
    bl_label = "Define Plane Normal"
    bl_idname = "panels.define_plane_normal"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.object

    def execute(self, context):
        # find selected vertices
        verts = np.asarray(self.bm.verts)
        selection = np.asarray([vert.select for vert in verts], dtype=bool)

        selection = np.nonzero(selection)
        target_location: Vector = self.bm.verts[selection[0][1]].co - self.bm.verts[selection[0][0]].co
        target_location.normalize()

        target_layer = context.scene.panel_manager.get_active().get_active()
        target_layer.plane_normal = target_location
        self.finished = True
        self.report({"INFO"}, "Value set, operator exit")
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
            bpy.ops.mesh.select_all(action='DESELECT')
        else:
            self.start_mode = 'EDIT'

        self.cancelled = False
        self.finished = False
        self.bm = bmesh.from_edit_mesh(context.edit_object.data)

        if context.object.data.total_vert_sel == 2:
            self.execute(context)
            return {'FINISHED'}


        bpy.ops.transform.translate('INVOKE_DEFAULT')
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}