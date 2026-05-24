#!/usr/bin/env python3
"""
wheel_compositor.py  —  CusToMatch 高品質ホイール合成ツール
=====================================================================
GPU不要。OpenCV Poisson Blending (seamlessClone) + 接地シャドウで
実写感のある車×ホイール合成画像を一括生成します。

使い方:
  cd Customatch/tools
  pip install -r requirements.txt
  python wheel_compositor.py                      # 全組み合わせ
  python wheel_compositor.py --car fd3s --wheel te37 ce28
  python wheel_compositor.py --no-poisson          # 軽量アルファブレンド
  python wheel_compositor.py --preview fd3s te37   # 1枚だけ確認用に表示
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys
import argparse

# ─── パス設定 ─────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
IMAGES    = ROOT / 'images'
COMPOSITE = IMAGES / 'composite'
COMPOSITE.mkdir(exist_ok=True)

# ─── 車種設定（app.html の CAR_IMAGES と同期して更新する） ─────────────────
CAR_PRESETS = {
    'fd3s': {
        'image': 'car_side.jpg',
        'wheels': [
            {'cx': 0.196, 'cy': 0.675, 'r': 0.064, 'ry': 0.064},
            {'cx': 0.775, 'cy': 0.682, 'r': 0.065, 'ry': 0.064},
        ]
    },
    'gr86': {
        'image': 'gr86_zn8_side.png',
        'wheels': [
            {'cx': 0.216, 'cy': 0.672, 'r': 0.060},
            {'cx': 0.786, 'cy': 0.669, 'r': 0.060},
        ]
    },
    'brz': {
        'image': 'brz_zd8_side.png',
        'wheels': [
            {'cx': 0.216, 'cy': 0.625, 'r': 0.063},
            {'cx': 0.798, 'cy': 0.624, 'r': 0.063},
        ]
    },
    'civic': {
        'image': 'civic_fl5_side.png',
        'wheels': [
            {'cx': 0.282, 'cy': 0.673, 'r': 0.060},
            {'cx': 0.814, 'cy': 0.668, 'r': 0.058},
        ]
    },
}

WHEEL_FILES = {
    'te37':  'wheel_te37.png',
    'ce28':  'wheel_ce28.png',
    'rpf1':  'wheel_rpf1.png',
    'bbs':   'wheel_bbs.png',
    'work':  'wheel_work.png',
    'advan': 'wheel_advan.png',
    'gram':  'wheel_gram.png',
    'ssr':   'wheel_ssr.png',
}

# ─── ① ホイール背景除去（BFS エッジ白飛ばし） ────────────────────────────
def remove_white_bg(img: np.ndarray) -> np.ndarray:
    """
    エッジから連結する白画素 (> 228) を透明化。
    内部のシルバー/クロームは保護。
    JSの processWheelBg と同等処理。
    """
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result = img.copy()
    H, W   = img.shape[:2]
    gray   = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    visited = np.zeros((H, W), dtype=bool)
    stack   = []

    def seed(x, y):
        if 0 <= x < W and 0 <= y < H and not visited[y, x] and gray[y, x] > 228:
            visited[y, x] = True
            stack.append((x, y))

    for x in range(W): seed(x, 0);   seed(x, H-1)
    for y in range(H): seed(0, y);   seed(W-1, y)

    while stack:
        x, y = stack.pop()
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
            seed(x+dx, y+dy)

    result[visited, 3] = 0

    # オートクロップ（中心 80% の非背景領域を正方形でくり抜く）
    mx0, mx1 = int(W * 0.10), int(W * 0.90)
    my0, my1 = int(H * 0.10), int(H * 0.90)
    roi_mask = (~visited)[my0:my1, mx0:mx1]
    ys, xs   = np.where(roi_mask)
    if xs.size > 0:
        minx, maxx = xs.min() + mx0, xs.max() + mx0
        miny, maxy = ys.min() + my0, ys.max() + my0
        sz   = max(maxx - minx, maxy - miny)
        cxc  = (minx + maxx) // 2
        cyc  = (miny + maxy) // 2
        half = int(sz * 0.54)
        x0, y0 = max(0, cxc-half), max(0, cyc-half)
        x1, y1 = min(W, cxc+half), min(H, cyc+half)
        result  = result[y0:y1, x0:x1]

    return result


# ─── ② ホイール画像を楕円に変形 ──────────────────────────────────────────
def warp_wheel(wheel_bgra: np.ndarray, rx: int, ry: int) -> tuple[np.ndarray, np.ndarray]:
    """
    ホイール画像を rx × ry の楕円サイズに変形する。
    縦圧縮 (ry/rx < 1.0) でパース感を表現。
    Returns: (bgr_image, alpha_mask) — どちらも正方形 (size × size)
    """
    size = max(rx, ry) * 2 + 6
    # Lanczosで高品質リサイズ
    wh = cv2.resize(wheel_bgra, (size, size), interpolation=cv2.INTER_LANCZOS4)

    # 縦方向をアフィン圧縮 (rx→ry に合わせてパース)
    scale_y = ry / rx if rx > 0 else 1.0
    cy_c    = size / 2
    M = np.float32([[1, 0, 0], [0, scale_y, cy_c * (1 - scale_y)]])
    warped  = cv2.warpAffine(wh, M, (size, size), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 楕円マスク（feathering あり）
    ell_mask = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    cv2.ellipse(ell_mask, (c, c), (rx, ry), 0, 0, 360, 255, -1)
    ell_mask = cv2.GaussianBlur(ell_mask, (9, 9), 3)

    bgr   = warped[:, :, :3]
    alpha = warped[:, :, 3] if warped.shape[2] == 4 else np.full((size,size), 255, np.uint8)
    combined_alpha = (ell_mask.astype(np.float32) / 255 *
                      alpha.astype(np.float32)    / 255 * 255).astype(np.uint8)
    return bgr, combined_alpha


# ─── ③ Poisson Blending で1輪を合成 ──────────────────────────────────────
def blend_one_wheel(base: np.ndarray, wheel_bgra: np.ndarray, pos: dict,
                    use_poisson: bool = True) -> np.ndarray:
    H, W = base.shape[:2]
    cx   = int(pos['cx'] * W)
    cy   = int(pos['cy'] * H)
    rx   = int(pos['r']  * W)
    ry   = int(pos.get('ry', pos['r']) * W)

    warped_bgr, alpha = warp_wheel(wheel_bgra, rx, ry)
    size = warped_bgr.shape[0]
    half = size // 2

    # キャンバス上の配置矩形
    x0 = max(0, cx - half);  y0 = max(0, cy - half)
    x1 = min(W, cx + half);  y1 = min(H, cy + half)
    wx0 = x0 - (cx - half);  wy0 = y0 - (cy - half)
    wx1 = wx0 + (x1 - x0);   wy1 = wy0 + (y1 - y0)

    result = base.copy()

    if use_poisson:
        # ── Poisson Blending ──────────────────────────────────────────
        wheel_canvas = np.zeros_like(base)
        mask_canvas  = np.zeros((H, W), dtype=np.uint8)

        wheel_canvas[y0:y1, x0:x1] = warped_bgr[wy0:wy1, wx0:wx1]
        _, bin_mask  = cv2.threshold(alpha, 15, 255, cv2.THRESH_BINARY)
        mask_canvas[y0:y1, x0:x1]  = bin_mask[wy0:wy1, wx0:wx1]

        margin = 5
        if (cx > rx + margin and cy > ry + margin and
                cx < W - rx - margin and cy < H - ry - margin and
                np.count_nonzero(mask_canvas) > 200):
            try:
                result = cv2.seamlessClone(
                    wheel_canvas, base, mask_canvas,
                    (cx, cy), cv2.NORMAL_CLONE
                )
            except Exception as e:
                print(f"    Poisson failed ({e}), fallback → alpha blend")
                result = _alpha_blend(base, warped_bgr, alpha, x0, y0, x1, y1, wx0, wy0, wx1, wy1)
        else:
            result = _alpha_blend(base, warped_bgr, alpha, x0, y0, x1, y1, wx0, wy0, wx1, wy1)
    else:
        result = _alpha_blend(base, warped_bgr, alpha, x0, y0, x1, y1, wx0, wy0, wx1, wy1)

    return result


def _alpha_blend(base, overlay, alpha, x0, y0, x1, y1, wx0, wy0, wx1, wy1):
    result = base.astype(np.float32)
    a   = alpha[wy0:wy1, wx0:wx1].astype(np.float32)[:, :, None] / 255.0
    src = overlay[wy0:wy1, wx0:wx1].astype(np.float32)
    result[y0:y1, x0:x1] = a * src + (1 - a) * result[y0:y1, x0:x1]
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── ④ 接地シャドウ追加 ─────────────────────────────────────────────────
def add_contact_shadow(img: np.ndarray, pos: dict) -> np.ndarray:
    """
    タイヤ底端（接地点）に扁平楕円グラデーションのシャドウを描画。
    乗算ブレンドで自然な暗さを表現。
    """
    H, W = img.shape[:2]
    cx   = int(pos['cx'] * W)
    cy   = int(pos['cy'] * H)
    rx   = int(pos['r']  * W)
    ry   = int(pos.get('ry', pos['r']) * W)

    # 接地点：タイヤ楕円の底端
    gnd_y  = cy + ry
    sh_rx  = int(rx * 0.90)
    sh_ry  = int(rx * 0.13)   # 縦13% → 扁平

    # シャドウマスク（ガウシアンぼかし付き楕円）
    shadow = np.zeros((H, W), dtype=np.float32)
    shadow_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(shadow_mask, (cx, gnd_y), (sh_rx, sh_ry), 0, 0, 360, 255, -1)
    shadow = cv2.GaussianBlur(shadow_mask.astype(np.float32) / 255.0, (31, 31), 10)

    # 乗算ブレンド（影の強さ 0.42）
    result = img.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= (1.0 - shadow * 0.42)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─── ⑤ 1組合せを処理 ────────────────────────────────────────────────────
def composite_one(car_key: str, wheel_key: str,
                  use_poisson: bool = True,
                  preview: bool = False) -> Path | None:
    preset     = CAR_PRESETS.get(car_key)
    wheel_file = WHEEL_FILES.get(wheel_key)
    if not preset or not wheel_file:
        print(f"[ERROR] 不明な car={car_key} / wheel={wheel_key}")
        return None

    car_path   = IMAGES / preset['image']
    wheel_path = IMAGES / wheel_file
    if not car_path.exists():
        print(f"[SKIP] {car_path.name} が見つかりません")
        return None
    if not wheel_path.exists():
        print(f"[SKIP] {wheel_path.name} が見つかりません")
        return None

    # 車画像
    car_bgr = cv2.imread(str(car_path))

    # ホイール画像（背景除去）
    wheel_raw = cv2.imread(str(wheel_path), cv2.IMREAD_UNCHANGED)
    if wheel_raw is None:
        print(f"[ERROR] {wheel_path.name} を読み込めません")
        return None
    wheel_bgra = remove_white_bg(wheel_raw)

    # 全ホイール位置を合成
    result = car_bgr.copy()
    for pos in preset['wheels']:
        result = blend_one_wheel(result, wheel_bgra, pos, use_poisson=use_poisson)
    # シャドウは全ホイール合成後に追加（ホイールの上に被らないように）
    for pos in preset['wheels']:
        result = add_contact_shadow(result, pos)

    if preview:
        cv2.imshow(f"{car_key} + {wheel_key}", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    out_path = COMPOSITE / f"{car_key}_{wheel_key}.jpg"
    cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return out_path


# ─── ⑥ 全組合せバッチ処理 ───────────────────────────────────────────────
def run_batch(car_keys=None, wheel_keys=None, use_poisson=True):
    car_keys   = car_keys   or list(CAR_PRESETS.keys())
    wheel_keys = wheel_keys or list(WHEEL_FILES.keys())
    total      = len(car_keys) * len(wheel_keys)
    done       = 0
    url_map    = {}

    for ck in car_keys:
        for wk in wheel_keys:
            done += 1
            label = f"[{done}/{total}] {ck} + {wk}"
            print(f"{label} ... ", end='', flush=True)
            out = composite_one(ck, wk, use_poisson=use_poisson)
            if out:
                url_map[f"{ck}_{wk}"] = f"images/composite/{out.name}"
                print("OK")
            else:
                print("SKIP")

    # app.html 用スニペット出力
    print("\n" + "="*60)
    print("// app.html に追加する COMPOSITE_IMAGES 定義:")
    print("// (この辞書をキャッシュとして使い、")
    print("//  存在するキーは高品質画像を優先表示)")
    print("var COMPOSITE_IMAGES =", json.dumps(url_map, ensure_ascii=False, indent=2) + ";")
    print("="*60)
    return url_map


# ─── エントリポイント ────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='CusToMatch — ホイール高品質合成ツール'
    )
    ap.add_argument('--car',   nargs='+', choices=list(CAR_PRESETS.keys()),
                    metavar='CAR',   help='対象車種 (省略=全車種)')
    ap.add_argument('--wheel', nargs='+', choices=list(WHEEL_FILES.keys()),
                    metavar='WHL',   help='対象ホイール (省略=全種)')
    ap.add_argument('--no-poisson', action='store_true',
                    help='Poisson blendingを無効にしてアルファブレンドのみ使用')
    ap.add_argument('--preview', nargs=2, metavar=('CAR', 'WHEEL'),
                    help='1組だけ合成してウィンドウ表示 (保存しない)')
    args = ap.parse_args()

    if args.preview:
        composite_one(args.preview[0], args.preview[1],
                      use_poisson=not args.no_poisson, preview=True)
    else:
        run_batch(
            car_keys=args.car,
            wheel_keys=args.wheel,
            use_poisson=not args.no_poisson,
        )
