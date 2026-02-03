import os
import re
import cv2
import numpy as np

# Optional (recommended). If not installed, we fall back to a tiny FEN parser below.
try:
    import chess
    _HAVE_CHESS = True
except Exception:
    _HAVE_CHESS = False

import argparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_FENS_DIR = os.path.join(BASE_DIR, "dataset_root", "fens")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "real_from_crop_data")
DEFAULT_CROPS_DIR = os.path.join(BASE_DIR, "dataset_root", "crops", "overhead")

parser = argparse.ArgumentParser()
parser.add_argument("--fens_dir", type=str, default=DEFAULT_FENS_DIR)
parser.add_argument("--skip_files", type=list, default=[
    os.path.join(DEFAULT_FENS_DIR, "all_original_fens.txt"),
    os.path.join(DEFAULT_FENS_DIR, "new_fens_randomPlaced_n20000_seed0.txt"),
])
parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
parser.add_argument("--crops_dir", type=str, default=DEFAULT_CROPS_DIR)
args = parser.parse_args()


# -----------------------------
# CONFIG (dataset generation)
# -----------------------------
FENS_DIR = args.fens_dir
SKIP_FILE = args.skip_files
OUT_DIR = args.out_dir
CROPS_DIR = args.crops_dir
IMG_SIZE = 512

# If True: skip generating if output image already exists
SKIP_IF_EXISTS = True


# -----------------------------
# Helpers
# -----------------------------
def _square_color_name(file_idx: int, rank_idx_from_white: int) -> str:
    """
    file_idx: 0..7 for a..h
    rank_idx_from_white: 0..7 for rank1..rank8 (white side at bottom)

    a1 is a dark square.
    Dark squares -> "black", light squares -> "white" (to match crop names).
    """
    # a1: (0+0)=0 -> dark
    is_dark = ((file_idx + rank_idx_from_white) % 2 == 0)
    return "black" if is_dark else "white"


def _fen_piece_placement_to_dict(fen: str):
    """
    Fallback parser: returns dict {(file_idx, rank_idx_from_white): (color, piece)}
    color in {"white","black"}, piece in {"pawn","knight","bishop","rook","queen","king"}
    """
    piece_map = {}
    placement = fen.split()[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"Bad FEN placement: {placement}")

    letter_to_piece = {
        "p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"
    }

    # FEN goes rank8 -> rank1
    for fen_rank_idx, row in enumerate(ranks):
        rank_from_white = 7 - fen_rank_idx  # rank8->7 ... rank1->0
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                color = "white" if ch.isupper() else "black"
                piece = letter_to_piece[ch.lower()]
                piece_map[(file_idx, rank_from_white)] = (color, piece)
                file_idx += 1
        if file_idx != 8:
            raise ValueError(f"Bad FEN row: {row}")
    return piece_map


def _load_crop_rgb(path: str):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not load crop: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def soften_square_boundaries(board_rgb, tile, edge_width=3, blur_ksize=5):
    """
    Softens square boundaries to reduce crop seams.

    edge_width: how many pixels around each square edge to soften
    blur_ksize: Gaussian blur kernel (odd)
    """
    h, w, _ = board_rgb.shape
    mask = np.zeros((h, w), dtype=np.float32)

    # Vertical edges
    for i in range(1, 8):
        x = i * tile
        x0 = max(0, x - edge_width)
        x1 = min(w, x + edge_width)
        mask[:, x0:x1] = 1.0

    # Horizontal edges
    for i in range(1, 8):
        y = i * tile
        y0 = max(0, y - edge_width)
        y1 = min(h, y + edge_width)
        mask[y0:y1, :] = 1.0

    blurred = cv2.GaussianBlur(board_rgb, (blur_ksize, blur_ksize), 0)

    mask = np.repeat(mask[:, :, None], 3, axis=2)
    out = board_rgb * (1.0 - mask) + blurred * mask
    return out.astype(np.uint8)


def fen_to_safe_filename(fen: str, max_len: int = 200) -> str:
    """
    Use the full FEN as the filename (as requested), but sanitize it so it's a valid path.
    Replaces spaces and forbidden characters.
    """
    fen = fen.strip()
    s = fen.replace(" ", "__")
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_.=\-]+", "-", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s + ".png"


# -----------------------------
# Main function
# -----------------------------
def fen_to_overhead_real_image(
    fen: str,
    crops_dir: str = CROPS_DIR,
    size: int = 512,
    visualize: bool = False,
    cache: dict = None,
):
    if size % 8 != 0:
        raise ValueError(f"size must be divisible by 8, got {size}")
    tile = size // 8

    if cache is None:
        cache = {}

    piece_at = {}
    if _HAVE_CHESS:
        b = chess.Board(fen)
        for sq, pc in b.piece_map().items():
            file_idx = chess.square_file(sq)
            rank_idx_from_white = chess.square_rank(sq)
            color = "white" if pc.color == chess.WHITE else "black"
            piece = {
                chess.PAWN: "pawn",
                chess.KNIGHT: "knight",
                chess.BISHOP: "bishop",
                chess.ROOK: "rook",
                chess.QUEEN: "queen",
                chess.KING: "king",
            }[pc.piece_type]
            piece_at[(file_idx, rank_idx_from_white)] = (color, piece)
    else:
        piece_at = _fen_piece_placement_to_dict(fen)

    board_rgb = np.zeros((size, size, 3), dtype=np.uint8)

    for fen_rank_idx in range(8):  # rank8..rank1 (top->bottom)
        rank_idx_from_white = 7 - fen_rank_idx
        y0 = fen_rank_idx * tile
        y1 = y0 + tile

        for file_idx in range(8):  # a..h (left->right)
            x0 = file_idx * tile
            x1 = x0 + tile

            sq_color = _square_color_name(file_idx, rank_idx_from_white)

            if (file_idx, rank_idx_from_white) in piece_at:
                pcolor, pname = piece_at[(file_idx, rank_idx_from_white)]
                fname = f"{pcolor}_{pname}_on_{sq_color}.png"
            else:
                fname = f"empty_{sq_color}.png"

            fpath = os.path.join(crops_dir, fname)

            key = (fpath, tile)
            if key not in cache:
                crop_rgb = _load_crop_rgb(fpath)
                crop_rgb = cv2.resize(crop_rgb, (tile, tile), interpolation=cv2.INTER_AREA)
                cache[key] = crop_rgb

            board_rgb[y0:y1, x0:x1] = cache[key]

    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(board_rgb)
        plt.title("Generated overhead board (RGB)")
        plt.axis("off")
        plt.show()

    board_rgb = cv2.GaussianBlur(board_rgb, (3, 3), 0)
    board_rgb = soften_square_boundaries(board_rgb, tile=tile, edge_width=3, blur_ksize=5)
    return board_rgb


# -----------------------------
# Dataset generation
# -----------------------------
PRINT_EVERY = 100
PRINT_FILE_START = True
PRINT_FILE_END = True


def _normalize_skip_files(skip_files):
    """
    Accepts:
      - None
      - a string path
      - a list/tuple/set of string paths
    Returns:
      - set of absolute paths
    """
    if skip_files is None:
        return set()
    if isinstance(skip_files, str):
        return {os.path.abspath(skip_files)}
    if isinstance(skip_files, (list, tuple, set)):
        return {os.path.abspath(p) for p in skip_files}
    raise TypeError(f"SKIP_FILE must be str or list/tuple/set[str], got: {type(skip_files)}")


def generate_dataset_from_fen_folder(
    fens_dir: str = FENS_DIR,
    out_dir: str = OUT_DIR,
    skip_file=SKIP_FILE,  # <- can be list now
    crops_dir: str = CROPS_DIR,
    size: int = IMG_SIZE,
    skip_if_exists: bool = SKIP_IF_EXISTS,
):
    os.makedirs(out_dir, exist_ok=True)
    cache = {}

    skip_set = _normalize_skip_files(skip_file)

    # Collect fen files
    fen_files = []
    for fn in sorted(os.listdir(fens_dir)):
        path = os.path.join(fens_dir, fn)
        if not os.path.isfile(path):
            continue
        if not fn.lower().endswith(".txt"):
            continue
        if os.path.abspath(path) in skip_set:
            continue
        fen_files.append(path)

    if not fen_files:
        print(f"[warn] No fen txt files found in: {fens_dir}")
        print(f"[warn] Skip set ({len(skip_set)}):")
        for p in sorted(skip_set):
            print(f"  - {p}")
        return

    print(f"[start] Found {len(fen_files)} fen files")
    print(f"[start] Output directory: {out_dir}")
    print(f"[start] Skip if exists: {skip_if_exists}")
    print(f"[start] Skip files ({len(skip_set)}):")
    for p in sorted(skip_set):
        print(f"  - {p}")
    print("")

    total_fens = 0
    written = 0
    skipped = 0
    failed = 0

    for file_idx, fpath in enumerate(fen_files, 1):
        if PRINT_FILE_START:
            print(f"[file {file_idx}/{len(fen_files)}] Processing: {os.path.basename(fpath)}")

        with open(fpath, "r") as f:
            for line_idx, line in enumerate(f, 1):
                fen = line.strip()
                if not fen:
                    continue

                total_fens += 1
                out_name = fen_to_safe_filename(fen)
                out_path = os.path.join(out_dir, out_name)

                if skip_if_exists and os.path.exists(out_path):
                    skipped += 1
                else:
                    try:
                        img_rgb = fen_to_overhead_real_image(
                            fen,
                            crops_dir=crops_dir,
                            size=size,
                            visualize=False,
                            cache=cache,
                        )
                        ok = cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                        if not ok:
                            raise RuntimeError("cv2.imwrite returned False")
                        written += 1
                    except Exception as e:
                        failed += 1
                        print(f"[fail] file={os.path.basename(fpath)} line={line_idx}")
                        print(f"       fen: {fen}")
                        print(f"       err: {e}")

                if total_fens % PRINT_EVERY == 0:
                    print(f"[prog] fens={total_fens} | written={written} | skipped={skipped} | failed={failed}")

        if PRINT_FILE_END:
            print(
                f"[file done] {os.path.basename(fpath)} | "
                f"total so far: fens={total_fens}, written={written}, skipped={skipped}, failed={failed}"
            )
            print("")

    print("==== DONE ====")
    print(f"Fen files processed: {len(fen_files)}")
    print(f"Total fens read:      {total_fens}")
    print(f"Images written:       {written}")
    print(f"Skipped (exists):     {skipped}")
    print(f"Failed:              {failed}")
    print(f"Output dir:           {out_dir}")


if __name__ == "__main__":
    generate_dataset_from_fen_folder()
