if "bpy" in locals():
    import importlib
    importlib.reload(utils)
    importlib.reload(image_processing)
    importlib.reload(ui)
    importlib.reload(operators)
    importlib.reload(bit_encoding)
    print("[SHTOOLS] Addon reload")
else:
    import bpy
    from . import utils
    from . import image_processing
    from . import ui
    from . import operators
    from . import bit_encoding


bl_info = {
    'name': 'Tools for Panel Shader',
    'description': 'Custom tools for Panel Material Shader',
    'location': '3D View > Toolbox',
    'author': 'wotava',
    'version': (0, 1),
    'blender': (3, 40, 0),
    'category': 'Material'
}

classes = [
    utils.Transform,
    image_processing.IDContainer,
    image_processing.PanelLayer,
    image_processing.LayerPreset,
    image_processing.AddonPresetStorage,
    image_processing.LayerManager,
    ui.DATA_UL_PanelPreset,
    ui.DATA_UL_PanelLayer,
    ui.DATA_PT_PanelShader,
    operators.DebugOperator,
    operators.PANELS_OP_PresetManipulation,
    operators.PANELS_OP_AssignPreset,
    operators.PANELS_OP_ActualizePresets,
    operators.PANELS_OP_WriteSlope,
    operators.PANELS_OP_SelectFacesFromPreset,
    operators.PANELS_OP_SelectPresetFromFace,
    operators.PANELS_OP_BakePresets,
    operators.PANELS_OP_AdjustImage,
    operators.PANELS_OP_DefinePlaneNormal,
    operators.PANELS_OP_MarkAsOrigin,
    operators.PANELS_OP_UpdateTransform,
    operators.PANELS_OP_StorageIO
]


def register():
    print(bit_encoding.verbose)
    print(utils.verbose)
    bpy.context.preferences.use_preferences_save = True
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.panel_manager = bpy.props.PointerProperty(type=image_processing.LayerManager)
    bpy.types.Object.origin_transform = bpy.props.PointerProperty(type=utils.Transform)


def unregister():
    # Unregister this addon
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.panel_manager