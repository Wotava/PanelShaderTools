import bpy

class DebugOperator(bpy.types.Operator):
    """Debug operator"""
    bl_idname = "material.debug_panel_shader"
    bl_label = "Scale Oversize UVs"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.view.show_developer_ui

    @staticmethod
    def execute(self, context):
        return {'FINISHED'}


class PANELS_OP_AddPreset(bpy.types.Operator):
    """Adds new panel preset, used in UI"""
    bl_label = "Add Panel Preset"
    bl_idname = "panels.add_preset"

    @classmethod
    def poll(cls, context):
        return True

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def execute(self, context):
        pass


class PANELS_OP_BakePresets(bpy.types.Operator):
    """Bakes presets to the specified image"""
    bl_label = "Bake Presets"
    bl_idname = "panels.bake_presets"

    @classmethod
    def poll(cls, context):
        return context.scene.panel_manager.target_image

    @staticmethod
    def execute(self, context):
        context.scene.panel_manager.write_image()
        return {'FINISHED'}
