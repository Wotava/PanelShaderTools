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
        col = row.column(align=True)
        col.operator("panels.add_preset", icon='ADD', text="")
        col.operator("panels.remove_preset", icon='REMOVE', text="")
        col.separator()
        col.operator("panels.duplicate_preset", icon='DUPLICATE', text="")

        if manager.storage_type == 'SCENE':
            row = layout.row()
            if len(manager.presets) == 0:
                op = row.operator("panels.storage_io", text="Read Presets from Addon", icon='FILE')
                op.action_type = 'READ_PREF_TO_SCENE'
            else:
                op = row.operator("panels.storage_io", text="Write Presets to Addon", icon='FILE_TICK')
                op.action_type = 'WRITE_SCENE_TO_PREF'

        # Display layer props
        if len(manager.presets) > 0:
            manager.draw_panel(layout)
