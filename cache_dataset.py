"""预缓存数据集: 把 22.8MB 原始 BMP 缩放到 DETR 目标分辨率, 存成小 JPEG。

之后 dataset.py 从缓存读, 训练/续训跳步都快 ~12x。
用 cv2 (与训练一致, 比 PIL 快 ~10x)。
用法:
  python cache_dataset.py --split train
  python cache_dataset.py --split val
断点续传: 已存在的缓存自动跳过。
"""
import argparse
import os
import cv2
from pathlib import Path

DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT") or (
    r"E:\Renming" if os.name == "nt" else "/root/autodl-tmp/Renming"
)
SHORTEST = 1024
LONGEST = 2048


def target_size(w, h):
    scale = min(LONGEST / max(w, h), SHORTEST / min(w, h))
    return round(w * scale), round(h * scale)  # (new_w, new_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT,
                    help="数据集根目录 (含 images/cache)")
    args = ap.parse_args()

    img_dir = Path(args.data_root) / "images" / args.split
    out_dir = Path(args.data_root) / "cache" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p.stem for p in img_dir.glob("*.bmp")])
    todo = [f for f in files if not (out_dir / f"{f}.jpg").exists()]
    print(f"{args.split}: {len(files)} total, {len(todo)} to cache")

    for i, name in enumerate(todo):
        src = img_dir / f"{name}.bmp"
        dst = out_dir / f"{name}.jpg"
        try:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)  # BGR
            h, w = img.shape[:2]
            nw, nh = target_size(w, h)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        except Exception as e:
            print(f"ERR {name}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  {args.split}: {i + 1}/{len(todo)}")

    print(f"done {args.split}: {len(todo)} cached -> {out_dir}")


if __name__ == "__main__":
    main()
