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
            col.label(text="Panel Layer", translate=False, icon='QUESTION')
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
        manager: LayerManager = context.scene.panel_manager
        row = layout.row()
        row.template_list("DATA_UL_PanelPreset", "", manager, "scene_presets", manager,
                          "active_preset")
        col = row.column(align=True)
        col.operator("panels.add_preset", icon='ADD', text="")
        col.operator("panels.remove_preset", icon='REMOVE', text="")

        col.separator()
        # Add ops here
        # col.operator("panels.move_preset_up", icon='ADD', text="")
        # col.operator("panels.remove_preset", icon='REMOVE', text="")

        # Display layer props
        if len(manager.scene_presets) > 0:
            preset = manager.get_active()

            # Target Image selection
            # TODO calculate optimal image size and auto-adjust it with one click
            row = layout.row()
            col = row.column(align=True)
            col.scale_x = 0.6
            col.label(text="Target Image")
            col = row.column(align=True)
            col.template_ID(manager, "target_image", new="image.new", open="image.open")
            if manager.target_image:
                manager.check_image(layout)
            layout.row().operator("panels.bake_presets")
            layout.row().operator("panels.assign_preset")
            layout.row().prop(manager, "use_auto_update", icon='FILE_REFRESH')
            layout.row().prop(manager, "use_auto_offset", icon='MOD_LENGTH')

            row = layout.row()
            row.template_list("DATA_UL_PanelLayer", "", preset, "layers", preset,
                              "active_layer")
            col = row.column(align=True)
            col.operator("panels.add_layer", icon='ADD', text="")
            col.operator("panels.remove_layer", icon='REMOVE', text="")
            col.separator()

            col.operator("panels.duplicate_layer", icon='DUPLICATE', text="")
            col.separator()

            if len(preset.layers) > 2 and preset.active_layer > 0:
                op = col.operator("panels.move_layer", icon='TRIA_UP', text="")
                op.move_up = True
            if len(preset.layers) > 2 and preset.active_layer < (len(preset.layers) - 1):
                op = col.operator("panels.move_layer", icon='TRIA_DOWN', text="")
                op.move_up = False


            if len(preset.layers) > 0:
                current_layer = preset.get_active()
                box = layout.box()
                row = box.row()
                row.prop(current_layer, "use_layer", text="Enable Layer")

                row = box.row(align=True)
                row.prop(current_layer, "plane_normal")
                row.operator("panels.define_plane_normal", icon='ORIENTATION_NORMAL', text="Set", emboss=True)
                row = box.row(align=True)
                row.label(text="Plane Data")
                row.prop(current_layer, "plane_offset", text="Offset")
                row.prop(current_layer, "plane_dist_A", text="Dist A")
                row.prop(current_layer, "plane_dist_B", text="Dist B")

                row = box.row(align=True)
                row.label(text="Decals")
                row.prop(current_layer, "decal_length", text="Length")
                row.prop(current_layer, "decal_thickness", text="Thickness")

                row = box.row(align=True)
                row.label(text="Sectors")
                row.prop(current_layer, "use_FG_mask", icon='FILE_PARENT', text="Use Previous")
                row.prop(current_layer, "sector_offset", text="Offset")
                row.prop(current_layer, "fg_sectors", text="FG")
                row.prop(current_layer, "bg_sectors", text="BG")

