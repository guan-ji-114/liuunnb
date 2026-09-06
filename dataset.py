"""DETR 数据集: 读 YOLO 格式 → DETR labels 格式。

YOLO 格式: class cx cy w h (归一化)
DETR labels: {"class_labels": LongTensor(N), "boxes": FloatTensor(N,4) cxcywh 归一化}
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import DetrImageProcessor

CLASS_NAMES = {0: "Hip", 1: "Knee", 2: "Ankle"}
N_CLASSES = 3


class DetrDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, processor, limit=0, cache_dir=None):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.processor = processor
        self.cache_dir = Path(cache_dir) if cache_dir else None
        files = sorted([p.stem for p in self.img_dir.glob("*.bmp")])
        # 过滤空标注(0框)图: 否则 boxes 张量变 1D, DETR 匈牙利匹配 cdist 崩溃
        files = [f for f in files
                 if any(l.strip() for l in (self.lbl_dir / f"{f}.txt").read_text().splitlines())]
        self.files = files[:limit] if limit else files

    def __len__(self):
        return len(self.files)

    def _load_label(self, name):
        txt = self.lbl_dir / f"{name}.txt"
        classes, boxes = [], []
        for line in Path(txt).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            c, cx, cy, w, h = map(float, line.split())
            classes.append(int(c))
            boxes.append([cx, cy, w, h])
        return classes, boxes

    def __getitem__(self, idx):
        name = self.files[idx]
        cache = self.cache_dir / f"{name}.jpg" if self.cache_dir else None
        if cache and cache.exists():
            img = cv2.imread(str(cache))
        else:
            img = cv2.imread(str(self.img_dir / f"{name}.bmp"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        classes, boxes = self._load_label(name)

        # DETR labels: cxcywh 归一化 (已是归一化, 直接转 tensor)
        labels = {
            "class_labels": torch.tensor(classes, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }

        # 预处理 (resize + normalize)
        encoding = self.processor(images=img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)  # (3,H,W)
        pixel_mask = encoding.get("pixel_mask", None)
        if pixel_mask is not None:
            pixel_mask = pixel_mask.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
            "name": name,
        }


def collate(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    labels = [b["labels"] for b in batch]
    names = [b["name"] for b in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask,
            "labels": labels, "name": names}


if __name__ == "__main__":
    import os
    ROOT = os.environ.get("DATA_ROOT") or (
        r"E:\Renming" if os.name == "nt" else "/root/autodl-tmp/Renming"
    )
    WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "facebook_detr_resnet50")
    processor = DetrImageProcessor.from_pretrained(
        WEIGHTS,
        size={"shortest_edge": 1024, "longest_edge": 2048},
    )
    ds = DetrDataset(
        os.path.join(ROOT, "images", "train"),
        os.path.join(ROOT, "labels", "train"),
        processor,
        limit=4,
    )
    print("n =", len(ds))
    s = ds[0]
    print("pixel_values", s["pixel_values"].shape)
    print("pixel_mask", s["pixel_mask"].shape)
    print("class_labels", s["labels"]["class_labels"])
    print("boxes", s["labels"]["boxes"].shape, s["labels"]["boxes"][:2])
