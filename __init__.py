if "bpy" in locals():
    import importlib
    importlib.reload(image_processing)
    importlib.reload(ui)
    importlib.reload(operators)
    importlib.reload(bit_encoding)
    importlib.reload(utils  )
    print("[SHTOOLS] Addon reload")
else:
    import bpy
    from . import image_processing
    from . import ui
    from . import operators
    from . import bit_encoding
    from . import utils


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
    image_processing.PanelLayer,
    image_processing.LayerPreset,
    image_processing.LayerManager,
    ui.DATA_UL_PanelPreset,
    ui.DATA_UL_PanelLayer,
    ui.DATA_PT_PanelShader,
    operators.DebugOperator,
    operators.PANELS_OP_AddPreset,
    operators.PANELS_OP_RemovePreset,
    operators.PANELS_OP_AddLayer,
    operators.PANELS_OP_RemoveLayer,
    operators.PANELS_OP_AssignPreset,
    operators.PANELS_OP_BakePresets,
    operators.PANELS_OP_AdjustImage,
    operators.PANELS_OP_DefinePlaneNormal
]


def register():
    print(bit_encoding.verbose)
    print(utils.verbose)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.panel_manager = bpy.props.PointerProperty(type=image_processing.LayerManager)


def unregister():
    # Unregister this addon
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.panel_manager