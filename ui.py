import bpy
from . image_processing import PanelLayer, LayerManager


class DATA_UL_PanelPreset(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        preset = item
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row()
            col = row.column(align=True)
            col.scale_x = 0.35
            col.label(text=f"#{index}")

            col = row.column(align=True)
            col.scale_x = 0.5
            if preset.name != "":
                col.label(text="Preset")
                col = row.column(align=True)
                col.prop(preset, "name", text="", emboss=False, icon='PRESET')
            else:
                col.label(text="Enter Name")
                col = row.column(align=True)
                col.prop(preset, "name", text="", emboss=True, icon="RIGHTARROW")

            col = row.column(align=True)
            col.label(text=f"ID: {preset.id}")
            return

        # TODO clean up
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            if preset.name != "":
                layout.prop(preset, "name", text="", emboss=False, icon='PRESET')


class DATA_UL_PanelLayer(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layer: PanelLayer = item
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row()
            col = row.column(align=True)
            col.scale_x = 0.25
            col.label(text=f"#{index}")

            col = row.column(align=True)
            col.scale_x = 0.5

            if layer.name != "":
                col.prop(layer, "name", text="", emboss=False, icon='LAYER_ACTIVE')
            else:
                col.prop(layer, "name", text="", emboss=False, icon='LAYER_USED')
            col = row.column(align=True)
            col.scale_x = 0.5
            col.prop(layer, "panel_type", text="", emboss=False)

            col = row.column(align=True)
            col.prop(layer, "use_layer", text="")
            return

        # TODO clean up
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layer.label(text="Layer", translate=False, icon='QUESTION')


class DATA_PT_PanelShader(bpy.types.Panel):
    bl_label = "Panel Shader Presets"
    bl_idname = "DATA_PT_panel_controls"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout

        # Active preset index is stored in scene panel manager
        manager: LayerManager = context.scene.panel_manager
        # Presets are stored in addon's preferences
        preset_storage = manager.preset_storage

        row = layout.row()
        row.prop(manager, "storage_type", expand=True)

        row = layout.row()
        row.template_list("DATA_UL_PanelPreset", "", preset_storage, "panel_presets", manager,
                          "active_preset_index")

        # PRESET MANIPULATORS
        col = row.column(align=True)
        op = col.operator("panels.preset_manipulator", icon='ADD', text="")
        op.action_type = 'ADD_PRESET'

        col = col.column(align=True)
        op = col.operator("panels.preset_manipulator", icon='REMOVE', text="")
        op.action_type = 'REMOVE_PRESET'
        col.enabled = len(context.scene.panel_manager.presets) > 0

        col.separator()
        col = col.column(align=True)
        op = col.operator("panels.preset_manipulator", icon='DUPLICATE', text="")
        op.action_type = 'DUPLICATE_PRESET'
        col.enabled = manager.active_preset is not None

        op = col.operator("panels.storage_io", icon='FILE_REFRESH', text="")
        op.action_type = 'SYNC_SELECTED'

        row = layout.row(align=True)
        op = row.operator("panels.storage_io", text="Read from Addon", icon='FILE')
        op.action_type = 'READ_PREF_TO_SCENE'
        op = row.operator("panels.storage_io", text="Write to Addon", icon='FILE_TICK')
        op.action_type = 'WRITE_SCENE_TO_PREF'
        row = layout.row(align=True)
        op = row.operator("panels.storage_io", text="Push Selected", icon='EXPORT')
        op.action_type = 'PUSH_FROM_OBJECT'
        op = row.operator("panels.storage_io", text="Pull Selected", icon='IMPORT')
        op.action_type = 'PULL_FROM_OBJECT'

        # Display layer props
        if len(manager.presets) > 0:
            manager.draw_panel(layout)
