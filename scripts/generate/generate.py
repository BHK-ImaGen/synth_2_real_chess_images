import os
import json
import argparse
import subprocess

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from generator import UNetGenerator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def preprocess_chess_board_training_style(img_bgr, output_size=512):
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect)
        pts = np.intp(pts)

    pts = pts.astype("float32")

    rect_ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect_ordered[0] = pts[np.argmin(s)]
    rect_ordered[2] = pts[np.argmax(s)]
    rect_ordered[1] = pts[np.argmin(diff)]
    rect_ordered[3] = pts[np.argmax(diff)]

    dst = np.array(
        [[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect_ordered, dst)
    warped = cv2.warpPerspective(img_rgb, M, (output_size, output_size))
    return warped


def load_norm_stats(norm_json_path):
    if norm_json_path and os.path.exists(norm_json_path):
        with open(norm_json_path, "r") as f:
            data = json.load(f)
        mean = data.get("mean", None)
        std = data.get("std", None)
        if mean is not None and std is not None and len(mean) == 3 and len(std) == 3:
            print(f"[Norm] Loaded mean/std from: {norm_json_path}")
            return mean, std

    print("[Norm] No norm stats JSON found; proceeding WITHOUT mean/std normalization.")
    if norm_json_path:
        print(f"       (Expected at: {norm_json_path})")
    return None, None


def img_rgb_to_model_input(img_rgb_uint8, device, mean=None, std=None):
    x = torch.from_numpy(img_rgb_uint8).float() / 255.0
    x = x.permute(2, 0, 1)

    if mean is not None and std is not None:
        x = T.Normalize(mean=mean, std=std)(x)

    x = x * 2.0 - 1.0
    return x.unsqueeze(0).to(device)


def model_output_to_bgr_uint8(y):
    y = y.squeeze(0).detach().cpu()
    y = (y + 1.0) / 2.0
    y = y.clamp(0, 1)
    y = (y.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def load_generator(weights_path, device):
    gen = UNetGenerator()
    sd = torch.load(weights_path, map_location=device)
    gen.load_state_dict(sd)
    gen.to(device)
    gen.eval()
    return gen


def read_image_bgr(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fen", required=True)
    p.add_argument("--view", choices=["white", "black"], required=True)
    p.add_argument("--angle", choices=["overhead", "east", "west"], required=True)

    p.add_argument("--blender_path", required=True)
    p.add_argument("--blend_file", required=True)
    p.add_argument("--generator_weights", required=True)

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # keep your local render defaults
    p.add_argument("--resolution", type=int, default=800)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--supersample", type=int, default=200)
    p.add_argument("--denoise", choices=["off", "on"], default="off")

    args = p.parse_args()

    renders_dir = os.path.join(BASE_DIR, "renders")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    blender_script = os.path.join(BASE_DIR, "chess_position_api_v2_sharp.py")
    synthetic_raw = os.path.join(renders_dir, "synthetic_raw.png")

    # IMPORTANT: delete stale image so you can't accidentally reuse old renders
    if os.path.exists(synthetic_raw):
        os.remove(synthetic_raw)

    # Norm json path next to weights by convention
    norm_json = os.path.splitext(args.generator_weights)[0] + ".norm.json"

    for path, name in [
        (args.blend_file, "blend_file"),
        (blender_script, "blender_script"),
        (args.generator_weights, "generator_weights"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # 1) Blender
    blender_cmd = [
        args.blender_path,
        args.blend_file,
        "--background",
        "--python", blender_script,
        "--",
        "--fen", args.fen,
        "--view", args.view,
        "--out_dir", renders_dir,
        "--resolution", str(args.resolution),
        "--samples", str(args.samples),
        "--supersample", str(args.supersample),
        "--denoise", args.denoise,
        "--angle", args.angle,
        "--also_overhead", "off",
    ]
    subprocess.run(blender_cmd, check=True, cwd=BASE_DIR)

    if not os.path.exists(synthetic_raw):
        raise RuntimeError("Blender did not produce synthetic_raw.png")

    # 2) preprocess
    raw = read_image_bgr(synthetic_raw)
    cropped_rgb = preprocess_chess_board_training_style(raw, output_size=512)
    if cropped_rgb is None:
        cropped_rgb = cv2.cvtColor(
            cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )

    cv2.imwrite(os.path.join(results_dir, "synthetic.png"), cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))

    # 3) inference
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = load_generator(args.generator_weights, device)
    mean, std = load_norm_stats(norm_json)
    x = img_rgb_to_model_input(cropped_rgb, device=device, mean=mean, std=std)

    with torch.no_grad():
        fake = gen(x)

    fake_bgr = model_output_to_bgr_uint8(fake)
    cv2.imwrite(os.path.join(results_dir, "realistic.png"), fake_bgr)

    side = np.hstack([cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR), fake_bgr])
    cv2.imwrite(os.path.join(results_dir, "side_by_side.png"), side)

    print("Results saved to:")
    print(" -", os.path.join(results_dir, "synthetic.png"))
    print(" -", os.path.join(results_dir, "realistic.png"))
    print(" -", os.path.join(results_dir, "side_by_side.png"))


if __name__ == "__main__":
    main()