import os
import json
import subprocess
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from generator import UNetGenerator

# =========================
# CONFIG
# =========================
BLENDER_PATH = "/home/guykou/apps/blender-4.2.0-linux-x64/blender"
BLEND_FILE = "/home/guykou/chess/chess-set.blend"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDERS_DIR = os.path.join(BASE_DIR, "renders")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

BLENDER_SCRIPT = os.path.join(BASE_DIR, "chess_position_api_v2_sharp.py")

GENERATOR_WEIGHTS = "/home/guykou/chess/final_models/run10_10_BCE_adv_weakerD_20260118_065207/checkpoints/gen_epoch_150.pth"
SYNTHETIC_RAW = os.path.join(RENDERS_DIR, "synthetic_raw.png")

NORM_STATS_JSON = os.path.splitext(GENERATOR_WEIGHTS)[0] + ".norm.json"

# =========================
# Board preprocessing (MATCH TRAINING)
# =========================
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
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect_ordered, dst)
    warped = cv2.warpPerspective(img_rgb, M, (output_size, output_size))
    return warped

# =========================
# Normalization helpers (MATCH TRAINING)
# =========================
def load_norm_stats():
    if os.path.exists(NORM_STATS_JSON):
        with open(NORM_STATS_JSON, "r") as f:
            data = json.load(f)
        mean = data.get("mean", None)
        std = data.get("std", None)
        if mean is not None and std is not None and len(mean) == 3 and len(std) == 3:
            print(f"[Norm] Loaded mean/std from: {NORM_STATS_JSON}")
            return mean, std

    print("[Norm] No norm stats JSON found; proceeding WITHOUT mean/std normalization.")
    print(f"       (Expected at: {NORM_STATS_JSON})")
    return None, None

def img_rgb_to_model_input(img_rgb_uint8, device, mean=None, std=None):
    x = torch.from_numpy(img_rgb_uint8).float() / 255.0
    x = x.permute(2, 0, 1)

    if mean is not None and std is not None:
        norm = T.Normalize(mean=mean, std=std)
        x = norm(x)

    x = x * 2.0 - 1.0
    x = x.unsqueeze(0).to(device)
    return x

def model_output_to_bgr_uint8(y):
    y = y.squeeze(0).detach().cpu()
    y = (y + 1.0) / 2.0
    y = y.clamp(0, 1)
    y = (y.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)

# =========================
# Model loading
# =========================
def load_generator(device):
    gen = UNetGenerator()
    sd = torch.load(GENERATOR_WEIGHTS, map_location=device)
    gen.load_state_dict(sd)
    gen.to(device)
    gen.eval()
    return gen

# =========================
# Core required function
# =========================
def generate_chessboard_image(fen: str, view: str, angle: str = "overhead") -> None:
    """
    view:  'white' or 'black'
    angle: 'overhead' or 'east' or 'west'
    """
    os.makedirs(RENDERS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if angle not in {"overhead", "east", "west"}:
        raise ValueError("angle must be one of: overhead/east/west")
    if view not in {"white", "black"}:
        raise ValueError("view must be one of: white/black")

    # sanity checks
    for p, name in [
        (BLENDER_PATH, "BLENDER_PATH"),
        (BLEND_FILE, "BLEND_FILE"),
        (BLENDER_SCRIPT, "BLENDER_SCRIPT"),
        (GENERATOR_WEIGHTS, "GENERATOR_WEIGHTS"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    # -------------------------
    # 1. Run Blender (synthetic)
    # -------------------------
    blender_cmd = [
        BLENDER_PATH,
        BLEND_FILE,
        "--background",
        "--python", BLENDER_SCRIPT,
        "--",
        "--fen", fen,
        "--view", view,
        "--out_dir", RENDERS_DIR,
        "--resolution", "800",
        "--samples", "256",
        "--supersample", "200",
        "--denoise", "off",
        "--angle", angle,
        "--also_overhead", "off",
    ]
    subprocess.run(blender_cmd, check=True, cwd=BASE_DIR)

    # -------------------------
    # 2. Load & preprocess synthetic image
    # -------------------------
    if not os.path.exists(SYNTHETIC_RAW):
        raise RuntimeError("Blender did not produce synthetic_raw.png")

    raw = cv2.imdecode(np.fromfile(SYNTHETIC_RAW, dtype=np.uint8), cv2.IMREAD_COLOR)
    cropped_rgb = preprocess_chess_board_training_style(raw, output_size=512)

    if cropped_rgb is None:
        cropped_rgb = cv2.cvtColor(
            cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )

    synthetic_path = os.path.join(RESULTS_DIR, "synthetic.png")
    cv2.imwrite(synthetic_path, cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))

    # -------------------------
    # 3. Generator inference
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = load_generator(device)

    mean, std = load_norm_stats()
    x = img_rgb_to_model_input(cropped_rgb, device=device, mean=mean, std=std)

    with torch.no_grad():
        fake = gen(x)

    fake_bgr = model_output_to_bgr_uint8(fake)

    realistic_path = os.path.join(RESULTS_DIR, "realistic.png")
    cv2.imwrite(realistic_path, fake_bgr)

    # -------------------------
    # 4. Side-by-side
    # -------------------------
    syn_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    side_by_side = np.hstack([syn_bgr, fake_bgr])
    side_path = os.path.join(RESULTS_DIR, "side_by_side.png")
    cv2.imwrite(side_path, side_by_side)

    print("Results saved to ./results/")
    print(f" - {synthetic_path}")
    print(f" - {realistic_path}")
    print(f" - {side_path}")

if __name__ == "__main__":
    test_fen = "8/p4p2/7p/2kp2p1/P5P1/2K1P3/5P1P/8"
    generate_chessboard_image(test_fen, view="white", angle="west")
