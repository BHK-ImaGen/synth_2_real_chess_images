import bpy
import math
import os
import sys
import argparse
import shutil
from pathlib import Path
from mathutils import Vector, Matrix

REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 2.0
DESIRED_ANGLE_DEGREES = 25.0
LENS = 26.0


def enable_cycles_cuda_gpu():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()

    for d in prefs.devices:
        d.use = (d.type == "CUDA")

    scene.cycles.device = "GPU"
    print("Cycles CUDA devices:", [(d.name, d.type, d.use) for d in prefs.devices])


def get_board_info():
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")

    if plane is None or frame is None:
        raise RuntimeError("Could not find objects 'Black & white' and/or 'Outer frame' in the .blend")

    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8.0

    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2.0
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)

    scale_factor = board_size / REAL_BOARD_SIZE

    return {
        "square_size": square_size,
        "plane_min": plane_min,
        "plane_max": plane_max,
        "center": center,
        "scale_factor": scale_factor,
    }


def position_to_square(pos, board_info):
    square_size = board_info["square_size"]
    plane_min = board_info["plane_min"]
    plane_max = board_info["plane_max"]

    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord("a") + file_idx)

    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1

    return f"{file_letter}{rank_number}"


def detect_starting_positions(board_info):
    pieces = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue

        name = obj.name
        piece_type = None

        if name in ["B", "C", "D", "E", "F", "G", "H", "A(texture)"]:
            piece_type = "P"
        elif name in ["B.001", "C.001", "D.001", "E.001", "F.001", "G.001", "H.001", "A(textures)"]:
            piece_type = "p"
        elif "rook" in name.lower():
            piece_type = "R" if "white" in name.lower() else "r"
        elif "knight" in name.lower():
            piece_type = "N" if "white" in name.lower() else "n"
        elif "bitshop" in name.lower() or "bishop" in name.lower():
            piece_type = "B" if "white" in name.lower() else "b"
        elif "queen" in name.lower():
            piece_type = "Q" if "white" in name.lower() else "q"
        elif "king" in name.lower():
            piece_type = "K" if "white" in name.lower() else "k"

        if piece_type:
            square = position_to_square(obj.location, board_info)
            pieces[name] = {
                "square": square,
                "piece_type": piece_type,
                "start_pos": obj.location.copy(),
                "start_rot": obj.rotation_euler.copy(),
                "start_hide_render": bool(obj.hide_render),
                "start_hide_viewport": bool(obj.hide_viewport),
            }

    print(f"✓ Detected {len(pieces)} pieces")
    return pieces


def reset_pieces(starting_pieces):
    for piece_name, info in starting_pieces.items():
        obj = bpy.data.objects.get(piece_name)
        if not obj:
            continue
        obj.location = info["start_pos"].copy()
        obj.rotation_euler = info["start_rot"].copy()
        obj.hide_render = False
        obj.hide_viewport = False


def parse_fen(fen):
    board_fen = fen.split()[0]
    ranks = board_fen.split("/")
    position = {}

    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        board_rank = 8 - rank_idx
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                file_letter = chr(ord("a") + file_idx)
                square = f"{file_letter}{board_rank}"
                position[square] = char
                file_idx += 1
    return position


def apply_fen(fen, starting_pieces, board_info):
    target_position = parse_fen(fen)
    square_size = board_info["square_size"]

    pieces_used = set()

    for target_square, piece_type in target_position.items():
        candidates = []
        for piece_name, info in starting_pieces.items():
            if info["piece_type"] == piece_type and piece_name not in pieces_used:
                from_square = info["square"]
                from_file = ord(from_square[0]) - ord("a")
                from_rank = int(from_square[1]) - 1
                to_file = ord(target_square[0]) - ord("a")
                to_rank = int(target_square[1]) - 1
                distance = abs(to_file - from_file) + abs(to_rank - from_rank)
                candidates.append((distance, piece_name, from_square))

        if not candidates:
            print(f"Warning: No piece of type '{piece_type}' available for {target_square}")
            continue

        candidates.sort()
        _, piece_name, from_square = candidates[0]

        obj = bpy.data.objects.get(piece_name)
        if obj:
            from_file = ord(from_square[0]) - ord("a")
            from_rank = int(from_square[1]) - 1
            to_file = ord(target_square[0]) - ord("a")
            to_rank = int(target_square[1]) - 1

            file_diff = to_file - from_file
            rank_diff = to_rank - from_rank

            obj.location.x -= file_diff * square_size
            obj.location.y -= rank_diff * square_size

            obj.hide_render = False
            obj.hide_viewport = False
            pieces_used.add(piece_name)

    for piece_name in starting_pieces.keys():
        if piece_name not in pieces_used:
            obj = bpy.data.objects.get(piece_name)
            if obj:
                obj.hide_render = True
                obj.hide_viewport = True

    print(f"✓ Position set ({len(pieces_used)} pieces visible)")


def configure_render(scene, *, res, samples, supersample, denoise_mode, adaptive_threshold):
    scene.render.engine = "CYCLES"

    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = supersample

    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = adaptive_threshold

    if denoise_mode.lower() == "off":
        scene.cycles.use_denoising = False
    else:
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "denoiser"):
            try:
                scene.cycles.denoiser = "OPENIMAGEDENOISE"
            except Exception:
                pass

    if hasattr(scene.render, "filter_size"):
        scene.render.filter_size = 0.5

    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 0

    try:
        scene.cycles.device = "GPU"
    except Exception:
        pass


def reset_lights_and_add_sun(center, light_height, energy=3.0, sun_angle_deg=0.1):
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, light_height))
    sun_obj = bpy.context.active_object
    sun = sun_obj.data
    sun.energy = energy
    if hasattr(sun, "angle"):
        sun.angle = math.radians(sun_angle_deg)


def _remove_all_cameras():
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)


def maybe_rotate_plane_90deg_once():
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")
    if not (plane and frame):
        return

    if abs((plane.rotation_euler.z % (2 * math.pi)) - math.radians(90)) < 1e-3:
        return
    if abs(plane.rotation_euler.z) > 1e-3:
        return

    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2.0

    original_pos = plane.location.copy()
    offset = original_pos - center
    plane.rotation_euler.z = math.radians(90)

    rot_matrix = Matrix.Rotation(math.radians(90), 3, "Z")
    rotated_offset = rot_matrix @ offset
    plane.location = center + rotated_offset


def render_one(board_info, *, view: str, angle: str, out_dir: str, primary_name="synthetic_raw.png"):
    center = board_info["center"]
    scale_factor = board_info["scale_factor"]

    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    horizontal_offset = camera_height * math.tan(angle_radians)
    camera_z = center.z + camera_height

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    _remove_all_cameras()

    light_height = center.z + camera_height * 2.0
    reset_lights_and_add_sun(center, light_height, energy=3.0, sun_angle_deg=0.1)

    scene = bpy.context.scene
    z_rotation_offset = math.radians(180) if view == "white" else 0.0

    primary_fp = os.path.join(out_dir, primary_name)

    def render_overhead(filepath):
        bpy.ops.object.camera_add(location=(center.x, center.y, camera_z))
        cam = bpy.context.active_object

        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.rotation_euler.z += z_rotation_offset

        cam.data.lens = LENS
        if hasattr(cam.data, "dof"):
            cam.data.dof.use_dof = False

        scene.camera = cam
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam, do_unlink=True)

    def render_side(which, filepath):
        sign = +1 if which == "east" else -1
        cam_loc = (center.x + sign * horizontal_offset, center.y, camera_z)

        bpy.ops.object.camera_add(location=cam_loc)
        cam = bpy.context.active_object

        cam.rotation_euler = (0.0, 0.0, 0.0)
        cam.rotation_euler.z += z_rotation_offset

        cam.data.lens = LENS
        if hasattr(cam.data, "dof"):
            cam.data.dof.use_dof = False

        scene.camera = cam
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam, do_unlink=True)

    if angle == "overhead":
        render_overhead(primary_fp)
        named_fp = os.path.join(out_dir, f"synthetic_overhead_{view}.png")
        shutil.copyfile(primary_fp, named_fp)
        return primary_fp, named_fp

    render_side(angle, primary_fp)
    named_fp = os.path.join(out_dir, f"synthetic_{angle}_{view}.png")
    shutil.copyfile(primary_fp, named_fp)
    return primary_fp, named_fp


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    p = argparse.ArgumentParser()
    p.add_argument("--fen", required=True)
    p.add_argument("--view", required=True, choices=["white", "black"])
    p.add_argument("--angle", required=True, choices=["overhead", "east", "west"])
    p.add_argument("--out_dir", required=True)

    p.add_argument("--resolution", type=int, default=800)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--supersample", type=int, default=200)
    p.add_argument("--denoise", type=str, default="off", choices=["off", "on"])

    # generate.py passes this even if you don't use it; accept it to avoid argparse failure
    p.add_argument("--also_overhead", type=str, default="off", choices=["on", "off"])

    p.add_argument("--adaptive_threshold", type=float, default=0.015)
    p.add_argument("--plane_rot_fix", type=str, default="on", choices=["on", "off"])
    return p.parse_args(argv)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    board_info = get_board_info()
    enable_cycles_cuda_gpu()

    if args.plane_rot_fix == "on":
        maybe_rotate_plane_90deg_once()

    configure_render(
        bpy.context.scene,
        res=args.resolution,
        samples=args.samples,
        supersample=args.supersample,
        denoise_mode=args.denoise,
        adaptive_threshold=args.adaptive_threshold,
    )

    starting_pieces = detect_starting_positions(board_info)
    reset_pieces(starting_pieces)
    apply_fen(args.fen, starting_pieces, board_info)

    # render requested angle
    primary_fp, named_fp = render_one(board_info, view=args.view, angle=args.angle, out_dir=args.out_dir)
    print(f"Saved (primary): {os.path.basename(primary_fp)}")
    print(f"Saved (copied):  {os.path.basename(named_fp)}")

    # optionally render overhead too (only if side requested)
    if args.also_overhead == "on" and args.angle != "overhead":
        primary_fp2, named_fp2 = render_one(board_info, view=args.view, angle="overhead", out_dir=args.out_dir)
        print(f"Saved (extra):   {os.path.basename(named_fp2)}")


if __name__ == "__main__":
    main()