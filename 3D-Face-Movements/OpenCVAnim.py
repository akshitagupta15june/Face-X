import bpy

class OBJECT_MT_OpenCVPanel(bpy.types.WorkSpaceTool):
    bl_label = "OpenCV Animation"
    bl_space_type = 'VIEW_3D'
    bl_context_mode='OBJECT'
    bl_idname = "ui_plus.opencv"
    bl_options = {'REGISTER'}
    bl_icon = "ops.generic.select_circle"
        
    def draw_settings(context, layout, tool):

        row = layout.row()
        op = row.operator("wm.opencv_operator", text="Capture", icon="OUTLINER_OB_CAMERA")
        
def register():
    bpy.utils.register_tool(OBJECT_MT_OpenCVPanel, separator=True, group=True)

def unregister():
     bpy.utils.unregister_tool(OBJECT_MT_OpenCVPanel)

if __name__ == "__main__":
    register()