"""DETR 评估: COCO 风格 mAP (mAP50, mAP50-95), 逐类 Hip/Knee/Ankle + 可视化。

评判标准沿用 E:\Renming\新建文本文档.txt 的 mAP 口径。
YOLO 基线: all mAP50=0.991, mAP50-95=0.591;
  Hip 0.995/0.563, Knee 0.995/0.709, Ankle 0.982/0.501
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import DetrForObjectDetection, DetrImageProcessor
from dataset import DetrDataset, collate, N_CLASSES, CLASS_NAMES

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights", "facebook_detr_resnet50")
DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT") or (
    r"E:\Renming" if os.name == "nt" else "/root/autodl-tmp/Renming"
)
COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}  # BGR


@torch.no_grad()
def predict_batch(model, pixel_values, pixel_mask, processor, target_sizes):
    """target_sizes: list of (H, W) 原图尺寸。返回 list of {boxes(绝对xyxy), scores, labels}。"""
    out = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    results = processor.post_process_object_detection(
        out, target_sizes=target_sizes, threshold=0.0)
    preds = []
    for r in results:
        preds.append({
            "boxes": r["boxes"],
            "scores": r["scores"],
            "labels": r["labels"],
        })
    return preds


def load_gt(img_name, lbl_dir, orig_shape):
    """读 YOLO 格式 GT, 转绝对 xyxy。orig_shape=(H,W)。"""
    txt = Path(lbl_dir) / f"{img_name}.txt"
    H, W = orig_shape
    boxes, labels = [], []
    for line in Path(txt).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        c, cx, cy, w, h = map(float, line.split())
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        boxes.append([x1, y1, x2, y2])
        labels.append(int(c))
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shortest-edge", type=int, default=1024)
    ap.add_argument("--longest-edge", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 张 (0=全部)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT,
                    help="数据集根目录 (含 images/labels/cache)")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader 进程数 (AutoDL 上可设 4-8)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "eval"))
    ap.add_argument("--viz", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DetrImageProcessor.from_pretrained(
        WEIGHTS_DIR,
        size={"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge},
    )
    model = DetrForObjectDetection.from_pretrained(
        WEIGHTS_DIR, num_labels=N_CLASSES, ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ds = DetrDataset(os.path.join(args.data_root, "images", args.split),
                     os.path.join(args.data_root, "labels", args.split),
                     processor, limit=args.limit,
                     cache_dir=os.path.join(args.data_root, "cache", args.split))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate,
                        num_workers=args.workers)
    print(f"eval images: {len(ds)}")

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True, backend="faster_coco_eval")
    viz_count = 0
    os.makedirs(args.out, exist_ok=True)

    for i, batch in enumerate(loader):
        pv = batch["pixel_values"].to(device)
        pm = batch["pixel_mask"].to(device)
        name = batch["name"][0]

        # 读原图尺寸
        img = cv2.imread(os.path.join(args.data_root, "images", args.split, f"{name}.bmp"))
        H, W = img.shape[:2]

        preds = predict_batch(model, pv, pm, processor,
                              target_sizes=[torch.tensor([H, W])])

        gt = load_gt(name, os.path.join(args.data_root, "labels", args.split), (H, W))

        preds[0]["boxes"] = preds[0]["boxes"].float().cpu()
        preds[0]["scores"] = preds[0]["scores"].cpu()
        preds[0]["labels"] = preds[0]["labels"].cpu()
        gt["boxes"] = gt["boxes"].float().cpu()
        gt["labels"] = gt["labels"].cpu()
        metric.update(preds, [gt])

        # 可视化
        if viz_count < args.viz:
            vis = img.copy()
            for box, label, score in zip(preds[0]["boxes"], preds[0]["labels"], preds[0]["scores"]):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                c = int(label.item())
                cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS.get(c, (0, 255, 255)), 8)
                cv2.putText(vis, f"{CLASS_NAMES.get(c)}:{score.item():.2f}",
                            (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS.get(c), 4)
            for box, label in zip(gt["boxes"], gt["labels"]):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                c = int(label.item())
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 3)
            s = 800 / vis.shape[0]
            vis = cv2.resize(vis, (int(vis.shape[1] * s), 800))
            cv2.imwrite(os.path.join(args.out, f"vis_{viz_count}.jpg"), vis)
            viz_count += 1

    result = metric.compute()
    summary = {
        "map50-95": float(result["map"].item()),
        "map50": float(result["map_50"].item()),
        "map75": float(result["map_75"].item()),
        "per_class_map50-95": {CLASS_NAMES[i]: float(v.item())
                                for i, v in enumerate(result["map_per_class"]) if i < N_CLASSES},
    }
    print("=" * 70)
    print(f"mAP50-95: {summary['map50-95']:.4f}   mAP50: {summary['map50']:.4f}   mAP75: {summary['map75']:.4f}")
    print("per-class mAP50-95:", summary["per_class_map50-95"])
    print("YOLO 基线: mAP50=0.991, mAP50-95=0.591 (Hip 0.563 / Knee 0.709 / Ankle 0.501)")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("metrics saved to", os.path.join(args.out, "metrics.json"))


if __name__ == "__main__":
    main()
