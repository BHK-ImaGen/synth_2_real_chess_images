import shutil
import math
import os
import bpy

def render_views(board_info, *, view, angle, out_dir, also_overhead, primary_name="synthetic_raw.png"):
    """
    Render exactly ONCE for the requested angle, writing to primary_name.
    Then copy that PNG to a named filename (synthetic_{angle}_{view}.png).
    Overhead is rendered only if requested via also_overhead.
    """

    center = board_info["center"]
    scale_factor = board_info["scale_factor"]
    scene = bpy.context.scene

    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    horizontal_offset = camera_height * math.tan(angle_radians)
    camera_z = center.z + camera_height

    _remove_all_cameras()

    # Lighting
    light_height = center.z + camera_height * 2.0
    reset_lights_and_add_sun(center, light_height, energy=3.0, sun_angle_deg=0.1)

    def z_rot_for_view():
        # Keep your convention: view=white rotates 180 degrees
        return math.radians(180) if view == "white" else 0.0

    def render_overhead(filepath):
        bpy.ops.object.camera_add(location=(center.x, center.y, camera_z))
        cam = bpy.context.active_object

        cam.rotation_euler = (0.0, 0.0, 0.0)
        cam.rotation_euler.z += z_rot_for_view()

        cam.data.lens = LENS
        if hasattr(cam.data, "dof"):
            cam.data.dof.use_dof = False

        _render_with_camera(scene, cam, filepath)
        bpy.data.objects.remove(cam, do_unlink=True)

    def render_side(which, filepath):
        # IMPORTANT FIX:
        # When we rotate the camera 180° for view="white", left/right swap.
        # Compensate by flipping the sign so "east/west" remain consistent.
        sign = +1 if which == "east" else -1
        if view == "white":
            sign *= -1

        cam_location = (center.x + sign * horizontal_offset, center.y, camera_z)

        bpy.ops.object.camera_add(location=cam_location)
        cam = bpy.context.active_object

        cam.rotation_euler = (0.0, 0.0, 0.0)
        cam.rotation_euler.z += z_rot_for_view()

        cam.data.lens = LENS
        if hasattr(cam.data, "dof"):
            cam.data.dof.use_dof = False

        _render_with_camera(scene, cam, filepath)
        bpy.data.objects.remove(cam, do_unlink=True)

    primary_fp = os.path.join(out_dir, primary_name)

    if angle == "overhead":
        render_overhead(primary_fp)
        named_fp = os.path.join(out_dir, f"synthetic_overhead_{view}.png")
        shutil.copyfile(primary_fp, named_fp)
        print(f"Saved (primary): {os.path.basename(primary_fp)}")
        print(f"Saved (copied):  {os.path.basename(named_fp)}")
        return

    # side angle
    render_side(angle, primary_fp)
    named_fp = os.path.join(out_dir, f"synthetic_{angle}_{view}.png")
    shutil.copyfile(primary_fp, named_fp)
    print(f"Saved (primary): {os.path.basename(primary_fp)}")
    print(f"Saved (copied):  {os.path.basename(named_fp)}")

    if also_overhead:
        oh_fp = os.path.join(out_dir, f"synthetic_overhead_{view}.png")
        render_overhead(oh_fp)
        print(f"Saved: {os.path.basename(oh_fp)}")
