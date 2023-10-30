import math

import bpy
import bmesh
from mathutils import Vector, Euler, Matrix, Quaternion
from . utils import has_custom_attrib
import numpy as np
ATTRIBUTE_NAME = "panel_preset_index"
ID_ATTRIBUTE_NAME = "panel_preset_id"


def update_objects(targets):
    for obj in targets:
        if obj.type == 'MESH':
            obj.data.update()


def vector_angle(v1, v2, normalized):
    if v1 == v2:
        return 0

    if not normalized:
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    else:
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


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
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        context.scene.panel_manager.new_preset()
        update_objects(context.visible_objects)
        return {'FINISHED'}


class PANELS_OP_RemovePreset(bpy.types.Operator):
    """Resets selected preset to defaults or deletes them completely if prompted"""
    bl_label = "Remove Panel Preset"
    bl_idname = "panels.remove_preset"
    bl_options = {'REGISTER', 'UNDO'}

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
        update_objects(context.visible_objects)
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class PANELS_OP_AddLayer(bpy.types.Operator):
    """Adds new layer to active preset, used in UI"""
    bl_label = "Add Preset Layer"
    bl_idname = "panels.add_layer"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.scene.panel_manager.active_preset
                and len(context.scene.panel_manager.active_preset.layers) < context.scene.panel_manager.max_layers)

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
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.panel_manager.active_preset and len(context.scene.panel_manager.active_preset.layers) > 0

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
    bl_options = {'REGISTER', 'UNDO'}

    move_up: bpy.props.BoolProperty(
        name="Move Selection Up",
        default=True
    )

    @classmethod
    def poll(cls, context):
        return context.scene.panel_manager.active_preset and len(context.scene.panel_manager.active_preset.layers) > 1

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
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.panel_manager.active_preset and len(context.scene.panel_manager.active_preset.layers) > 0

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
            and (context.mode == 'EDIT_MESH' and context.object.data.total_face_sel > 0 or context.mode == 'OBJECT')

    def execute(self, context):
        manager = context.scene.panel_manager
        active_preset_index = manager.active_preset_index
        active_preset_id = manager.active_preset.id
        if active_preset_id == -1:
            manager.generate_id(manager.active_preset)

        # Do everything in object mode, because writing UVs crashes Blender on large objects
        start_mode = context.mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Ensure we have all the necessary UV maps in the right order
        target_uv_names = ['UVMap', 'UVMap_Aligned', 'UVMap_SlopePreset', 'UVMap_Inset']
        target_preset_uv = 2
        for obj in context.selected_objects:
            if len(obj.data.uv_layers) < 4:
                for layer in target_uv_names[len(obj.data.uv_layers):]:
                    obj.data.uv_layers.new().name = layer

        for obj in context.selected_objects:
            target = obj.data.attributes

            if start_mode == 'OBJECT':
                selected = [f.index for f in obj.data.polygons]
            else:
                selected = [f.index for f in obj.data.polygons if f.select]

            uv_layer = obj.data.uv_layers[target_preset_uv]

            name = ATTRIBUTE_NAME
            if target.find(name) < 0:
                attrib = target.new(name, 'INT', 'FACE')
            else:
                attrib = target[target.find(name)]

            # Process everything through separate lists to avoid slowdowns and segfaults
            attributes = np.zeros((len(attrib.data.values())), dtype=int)
            attrib.data.foreach_get("value", attributes)
            for i in range(len(attributes)):
                if i in selected:
                    attributes[i] = active_preset_index
            attrib.data.foreach_set("value", attributes)

            uv_all = np.zeros((len(uv_layer.data) * 2))
            uv_layer.data.foreach_get("uv", uv_all)

            # get a list of selected
            if start_mode == 'OBJECT':
                target_loops = [[*x.loop_indices] for x in obj.data.polygons]
            else:
                target_loops = [[*x.loop_indices] for x in obj.data.polygons if x.select]

            target_loops = [y for x in target_loops for y in x]  # extend a list of lists with nested comprehension
            for i in target_loops:
                if math.isnan(uv_all[i*2]):
                    uv_all[i*2] = 0.0
                uv_all[i*2+1] = float(active_preset_index)
            uv_layer.data.foreach_set("uv", uv_all)

            # assign a dictionary as a custom property to hold scene_id = preset_id pairs

            if not has_custom_attrib(obj, ID_ATTRIBUTE_NAME):
                print("new dict")
                obj[ID_ATTRIBUTE_NAME] = dict()
            pair_attribute = obj[ID_ATTRIBUTE_NAME]
            tdict = pair_attribute.to_dict()
            tdict[str(active_preset_index)] = active_preset_id
            print(tdict)

            # Keep only used pairs
            uniques = np.unique(attributes)
            for key, value in zip(tdict.keys(), tdict.values()):
                if int(key) not in uniques:
                    tdict.pop(key)

            pair_attribute.update(tdict)

            self.report({'INFO'}, f"Set preset {active_preset_index} on {len(selected)} faces")

        if start_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class PANELS_OP_ActualizePresets(bpy.types.Operator):
    """Assign current preset to selected faces"""
    bl_label = "Actualize Presets"
    bl_idname = "panels.actualize_presets"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object

    def draw(self, context):
        layout = self.layout
        layout.label(icon='ERROR',
                     text="No objects selected, operator will be run on ALL objects in this file. Proceed?")

    def execute(self, context):
        manager = context.scene.panel_manager
        bpy.ops.object.mode_set(mode='OBJECT')

        targets = []
        if len(context.selected_objects) > 0:
            targets = [x for x in context.selected_objects if x.type == 'MESH']

        if len(targets) == 0:
            targets = [x for x in bpy.data.objects if x.type == 'MESH']

        for obj in targets:
            if not has_custom_attrib(obj, ID_ATTRIBUTE_NAME):
                print(f"No preset pair data found on {obj.name}, skip")
                continue

            target = obj.data.attributes
            uv_layer = obj.data.uv_layers['UVMap_SlopePreset']

            name = ATTRIBUTE_NAME
            if target.find(name) < 0:
                print(f"No preset data found on {obj.name}, skip")
                continue
            else:
                attrib = target[name]

            preset_pos = np.zeros((len(attrib.data.values())), dtype=int)
            attrib.data.foreach_get("value", preset_pos)

            preset_uv = np.zeros((len(uv_layer.data) * 2))
            uv_layer.data.foreach_get("uv", preset_uv)

            # Initialize "new" lists with negative ones, so we can replace unaffected values with original data
            preset_pos_new = np.full((len(attrib.data.values())), fill_value=-1, dtype=int)
            preset_uv_new = np.full((len(uv_layer.data) * 2), fill_value=-1.0, dtype=float)

            pair_dict = obj[ID_ATTRIBUTE_NAME].to_dict()
            for key, value in zip(list(pair_dict.keys()), list(pair_dict.values())):
                if manager.panel_presets[int(key)].id != value:
                    new_pos = manager.find_pos_by_id(value)
                    if new_pos == -1:
                        # TODO Look for this id in prefs container and copy it if found
                        pass
                    print(key, value)
                    pair_dict.pop(key)
                    pair_dict[str(new_pos)] = value
                    preset_pos_new[np.where(preset_pos == int(key))] = new_pos
                    preset_uv_new[np.where(preset_pos == int(key))] = float(new_pos)
            obj[ID_ATTRIBUTE_NAME].clear()
            obj[ID_ATTRIBUTE_NAME].update(pair_dict)
            # Replace "-1" unaffected values with original data
            preset_pos_new[preset_pos_new == -1] = preset_pos[preset_pos_new == -1]
            preset_uv_new[preset_uv_new == -1] = preset_uv[preset_uv_new == -1]

            # Since we write slope mask to the U channel, fill all uneven indices with original values
            checker = np.tile([True, False], len(uv_layer.data))
            preset_uv_new[checker] = preset_uv[checker]

            # Actualize data on the mesh
            attrib.data.foreach_set("value", preset_pos_new)
            uv_layer.data.foreach_get("uv", preset_uv)

        return {'FINISHED'}

    def invoke(self, context, event):
        targets = []
        if context.selected_objects > 0:
            targets = [x for x in context.selected_objects if x.type == 'MESH']

        if len(targets) == 0:
            return context.window_manager.invoke_props_dialog(self)
        else:
            self.execute(context)
            return {'FINISHED'}


class PANELS_OP_WriteSlope(bpy.types.Operator):
    """Writes slope-marker to UV"""
    bl_label = "Write Slope to UV"
    bl_idname = "panels.write_slope"
    bl_options = {'REGISTER', 'UNDO'}

    slope_epsilon: bpy.props.FloatProperty(
        name="Slope Angle Tolerance",
        default=1,
    )

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'MESH'

    def select_smooth_faces(self, context, obj):
        mesh = obj.data

        # Auto-smooth should always be enabled when we work with slope mapping
        if not mesh.use_auto_smooth:
            # Test if this mesh uses uniform smooth shading and set the auto-smooth angle to 180 to keep it
            if obj.data.polygons[0].use_smooth:
                mesh.auto_smooth_angle = 180
            mesh.use_auto_smooth = True

        # Update normals
        mesh.calc_normals_split()

        # Use evaluated variant of the mesh as reference if the object uses modifiers that don't affect geometry
        if len(obj.modifiers) != 0:
            unsupported_modifiers = False

            for modifier in obj.modifiers:
                if modifier.type in ['WEIGHTED_NORMAL', 'DATA_TRANSFER', 'ARRAY', 'MIRROR']:
                    continue
                else:
                    unsupported_modifiers = True
                    break

            if unsupported_modifiers:
                # Can't select anything when we have geometry-altering modifiers
                self.report({'ERROR'}, f"Some objects have unsupported geometry-altering modifiers")
                return
            else:
                depsgraph = context.evaluated_depsgraph_get()
                object_eval = obj.evaluated_get(depsgraph)
                ref_mesh = object_eval.to_mesh()
                ref_mesh.calc_normals_split()
        else:
            ref_mesh = mesh

        # Since normals are stored in face-corner domain in every polygon loop,
        # check every loop of a polygon against the first loop's normal,
        # and if the angle between them is >0deg - set-select them to utilize
        # "Select Similar -> delimit UV" operator to mark the entire island
        for poly in ref_mesh.polygons:
            ref_normal = ref_mesh.loops[poly.loop_indices[0]].normal

            for index in poly.loop_indices:
                if vector_angle(ref_mesh.loops[index].normal, ref_normal, True) > (0 + self.slope_epsilon):
                    break
            else:
                continue

            # If we have found non-flat-shaded polygon - select it,
            # if the index is in range of original mesh polygon count
            # (so we won't try to select faces generated by array or mirror),
            # otherwise break the loop to stop calculation for this mesh
            if poly.index < len(mesh.polygons):
                mesh.polygons[poly.index].select = True
            else:
                break

        # Destroy evaluated mesh
        if ref_mesh != mesh:
            object_eval.to_mesh_clear()
        return

    def execute(self, context):
        # Deselect everything
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')

        # Do everything in object mode, because writing UVs crashes Blender on large objects
        bpy.ops.object.mode_set(mode='OBJECT')

        # Ensure we have all the necessary UV maps in the right order
        target_uv_names = ['UVMap', 'UVMap_Aligned', 'UVMap_SlopePreset', 'UVMap_Inset']
        target_preset_uv = 2
        for obj in context.selected_objects:
            if len(obj.data.uv_layers) < 4:
                for layer in target_uv_names[len(obj.data.uv_layers):]:
                    obj.data.uv_layers.new().name = layer
            self.select_smooth_faces(context, obj)

        # Extend selection to fill UV islands
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_linked(delimit={'UV'})
        bpy.ops.object.mode_set(mode='OBJECT')

        for obj in context.selected_objects:
            uv_layer = obj.data.uv_layers[target_preset_uv]

            # Process everything through a separate list to avoid slowdowns and segfaults
            uv_all = np.zeros((len(uv_layer.data) * 2))
            uv_layer.data.foreach_get("uv", uv_all)

            # get a list of selected
            slope_loops = [[*x.loop_indices] for x in obj.data.polygons if x.select]
            slope_loops = [y for x in slope_loops for y in x]  # extend a list of lists with nested comprehension

            clean_loops = [[*x.loop_indices] for x in obj.data.polygons if not x.select]
            clean_loops = [y for x in clean_loops for y in x]  # extend a list of lists with nested comprehension

            for i in slope_loops:
                uv_all[i*2] = -1

            for i in clean_loops:
                uv_all[i*2] = +1

            uv_layer.data.foreach_set("uv", uv_all)

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
    bl_options = {'REGISTER', 'UNDO'}

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
        update_objects(context.visible_objects)
        return {'FINISHED'}


class PANELS_OP_AdjustImage(bpy.types.Operator):
    """Adjusts image parameters"""
    bl_label = "Adjust Target Image"
    bl_idname = "panels.adjust_image"
    bl_options = {'REGISTER', 'UNDO'}

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
    bl_options = {'REGISTER', 'UNDO'}

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
            if len(selection) < 2:
                self.report({'ERROR'}, "Not enough data for selection. Maybe you have non-mesh selection?")
                return {'CANCELLED'}
            target_location: Vector = (selection[0] - selection[1])
            target_location.normalize()
            setattr(target_layer, self.target, target_location)

            if self.target == "plane_normal" and context.scene.panel_manager.use_auto_offset:
                target_vert_co = selection[0]

                target_vert_co = world_matrix @ target_vert_co
                target_layer.plane_offset_internal = target_vert_co
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
    bl_options = {'REGISTER', 'UNDO'}

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
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object and context.mode == "OBJECT" and len(context.selected_objects) > 0

    def execute(self, context):
        obj = context.object
        tr = obj.origin_transform
        if tr.scale == Vector((0.0, 0.0, 0.0)):
            self.report({'ERROR'}, "This object was not initialized as Preset Origin")
            return {'CANCELLED'}

        if len(obj.children) == 0:
            self.report({'ERROR'}, "No children objects to transform")
            return {'CANCELLED'}

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

        position = ['position_2d', 'position_3d', 'plane_offset_internal']
        direction = ['tile_direction_3d', 'plane_normal']
        for preset in presets:
            for layer in preset.layers:
                if layer.panel_type.find('UV') != -1:
                    continue

                for attribute in direction:
                    cur_attr = getattr(layer, attribute)
                    cur_attr = rot_delta_reverse @ cur_attr
                    cur_attr = rot_delta @ cur_attr
                    setattr(layer, attribute, cur_attr)

                for attribute in position:
                    cur_attr = getattr(layer, attribute)
                    cur_attr = loc_delta_reverse @ cur_attr
                    cur_attr = rot_delta_reverse @ cur_attr
                    cur_attr = rot_delta @ cur_attr
                    cur_attr = loc_delta @ cur_attr
                    setattr(layer, attribute, cur_attr)


        context.scene.panel_manager.write_image()
        obj.origin_transform.from_obj(obj)

        return {'FINISHED'}


class PANELS_OP_StorageIO(bpy.types.Operator):
    """"""
    bl_label = "Storage IO"
    bl_idname = "panels.storage_io"
    bl_options = {'REGISTER', 'UNDO'}

    action_type: bpy.props.EnumProperty(
        name="Action",
        description="",
        items=[
            ('READ_PREF_TO_SCENE', 'Load Prefs to Scene', 'Copy Presets from Addon Preferences to Scene'),
            ('WRITE_SCENE_TO_PREF', 'Write Scene to Prefs', 'Write Scene Presets to Addon Preferences'),
            ('SYNC_SELECTED', 'Sync Selected', 'Sync selected presets with addon storage'),
            ('PUSH_FROM_OBJECT', 'Push from Object', 'Push presets used in selected objects to addon storage'),
            ('PULL_FROM_OBJECT', 'Pull from Object', 'Pull presets used in selected objects from addon storage'),
        ],
        default='READ_PREF_TO_SCENE',
    )

    @classmethod
    def poll(cls, context):
        return True

    @classmethod
    def description(cls, context, props):
        option = getattr(props, "action_type")
        if option == 'READ_PREF_TO_SCENE':
            desc = "Copy All Presets from Addon Preferences to Scene"
        elif option == 'WRITE_SCENE_TO_PREF':
            desc = "Write Scene Presets to Addon Preferences"
        elif option == 'SYNC_SELECTED':
            desc = "Sync selected preset with addon storage"
        elif option == 'PUSH_FROM_OBJECT':
            desc = "Push presets used in selected objects to addon storage"
        else:
            desc = "Pull presets used in selected objects from addon storage"
        return desc

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
        elif self.action_type == 'SYNC_SELECTED':
            if manager.active_preset:
                manager.sync_preset(manager.active_preset.id)
        else:
            if len(context.selected_objects) < 0:
                self.report({'INFO'}, "No object selected")

            used_presets = []
            for obj in context.selected_objects:
                pair_dict = obj[ID_ATTRIBUTE_NAME].to_dict()
                used_presets.extend(pair_dict.values())
            used_presets = np.unique(used_presets)

            pull_from_prefs = self.action_type == 'PULL_FROM_OBJECT'

            for preset_id in used_presets:
                manager.sync_preset(preset_id, pull_from_prefs=pull_from_prefs)

            if pull_from_prefs:
                bpy.ops.panels.actualize_presets()

        return {'FINISHED'}
