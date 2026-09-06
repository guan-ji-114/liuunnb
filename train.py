"""DETR 微调训练 + 解剖先验(创新点)。

总损失 = DETR 标准损失(匈牙利匹配) + w_layout * L_layout + w_sym * L_sym
- L_layout: 髋<膝<踝 垂直顺序先验
- L_sym: 左右对称(翻转等变)

用法:
  python train.py --smoke --epochs 1 --limit 50      # 快速验证
  python train.py --epochs 40 --w-layout 0.5 --w-sym 0.5 --out runs/full
"""
import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import DetrForObjectDetection, DetrImageProcessor
from dataset import DetrDataset, collate, N_CLASSES
from anatomy_loss import layout_loss, symmetry_loss

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights", "facebook_detr_resnet50")
DATA_ROOT_DEFAULT = os.environ.get("DATA_ROOT") or (
    r"E:\Renming" if os.name == "nt" else "/root/autodl-tmp/Renming"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.02,
                    help="余弦学习率预热的步数占总步数比例")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT,
                    help="数据集根目录 (含 images/labels/cache)")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader 进程数 (AutoDL 上可设 4-8 加速数据加载)")
    ap.add_argument("--w-layout", type=float, default=0.5)
    ap.add_argument("--w-sym", type=float, default=0.5)
    ap.add_argument("--shortest-edge", type=int, default=1024)
    ap.add_argument("--longest-edge", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="只训练前 N 张 (0=全部)")
    ap.add_argument("--smoke", action="store_true", help="只跑 2 个 batch")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=500, help="每 N 步存 checkpoint (断点续训)")
    ap.add_argument("--clip-max-norm", type=float, default=0.1, help="梯度裁剪 (DETR 稳定必需)")
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--amp", default=False, action=argparse.BooleanOptionalAction,
                    help="混合精度 bf16 加速 (默认 fp32, 因 bf16 的 GIoU 梯度会 NaN)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"
    print(f"device={device} amp={use_amp}")

    # processor + 模型
    processor = DetrImageProcessor.from_pretrained(
        WEIGHTS_DIR,
        size={"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge},
    )
    model = DetrForObjectDetection.from_pretrained(
        WEIGHTS_DIR,
        num_labels=N_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)
    if args.freeze_backbone:
        model.model.freeze_backbone()

    # 数据 (优先读缓存 JPEG, 快 ~12x)
    ds = DetrDataset(os.path.join(args.data_root, "images", "train"),
                     os.path.join(args.data_root, "labels", "train"),
                     processor, limit=args.limit,
                     cache_dir=os.path.join(args.data_root, "cache", "train"))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                        num_workers=args.workers)
    print(f"train images: {len(ds)}")
    total_steps = args.epochs * len(ds)
    warmup_steps = int(args.warmup_frac * total_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 断点续训
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    resume_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        resume_step = ck.get("global_step", 0)
        print(f"[RESUME] from global_step {resume_step}")

    global_step = 0
    nan_streak = 0
    diverged = False
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        ep_loss = 0.0
        n_batch = 0
        for batch in loader:
            # 断点续训: 跳过已完成的 step
            if global_step < resume_step:
                global_step += 1
                continue

            # 余弦+预热 LR 调度 (每步更新, 防止训练后期发散)
            if global_step < warmup_steps:
                s = max(global_step / warmup_steps, 1e-6)
            else:
                p = min((global_step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
                s = 0.5 * (1 + math.cos(math.pi * p))
            for g in optimizer.param_groups:
                g["lr"] = args.lr * s

            pv = batch["pixel_values"].to(device)
            pm = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]

            optimizer.zero_grad()
            step_ok = False
            step_loss = step_detr = step_layout = step_sym = float("nan")
            try:
                # 原图前向 (DETR 主损失) + 先验, 混合精度 bf16
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(pixel_values=pv, pixel_mask=pm, labels=labels)
                    total = out.loss

                    # 布局先验 (单前向)
                    l_layout = layout_loss(out.logits, out.pred_boxes)

                    # 对称先验 (翻转图双前向, 无 labels)
                    l_sym = torch.tensor(0.0, device=device)
                    if args.w_sym > 0:
                        pv_flip = torch.flip(pv, dims=[-1])
                        pm_flip = torch.flip(pm, dims=[-1])
                        out_flip = model(pixel_values=pv_flip, pixel_mask=pm_flip)
                        l_sym = symmetry_loss(out.pred_boxes, out_flip.pred_boxes)

                    loss = total + args.w_layout * l_layout + args.w_sym * l_sym

                loss.backward()
                if torch.isfinite(loss):
                    norm = clip_grad_norm_(model.parameters(), args.clip_max_norm)
                    if torch.isfinite(norm):
                        optimizer.step()
                        step_ok = True
                        step_loss = loss.item()
                        step_detr = total.item()
                        step_layout = l_layout.item()
                        step_sym = l_sym.item()
                    else:
                        print(f"WARNING: NaN grad at step{global_step}, skip")
                else:
                    print(f"WARNING: non-finite loss at step{global_step}, skip")
            except (ValueError, RuntimeError) as e:
                print(f"WARNING: step{global_step} error, skip: {str(e)[:80]}")

            if step_ok:
                ep_loss += step_loss
                n_batch += 1
                nan_streak = 0
            else:
                nan_streak += 1
                if nan_streak >= 200:
                    print(f"FATAL: 连续 {nan_streak} 步 NaN, 模型已发散, 停止训练 (不覆盖干净 checkpoint)")
                    diverged = True
                    break
            global_step += 1

            if global_step % 10 == 0:
                if step_ok:
                    print(f"ep{epoch} step{global_step} loss={step_loss:.4f} "
                          f"(detr={step_detr:.3f} layout={step_layout:.4f} "
                          f"sym={step_sym:.4f})")
                else:
                    print(f"ep{epoch} step{global_step} SKIPPED")

            if global_step % args.save_every == 0 and nan_streak == 0:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "global_step": global_step}, ckpt_path)

            if args.smoke and global_step >= 2:
                break

        if n_batch > 0:
            print(f"=== epoch {epoch} (global_step {global_step}) avg_loss={ep_loss/max(n_batch,1):.4f} "
                  f"{time.time()-t0:.0f}s ===")

        if args.smoke:
            break
        if diverged:
            break

    if not diverged:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "global_step": global_step}, ckpt_path)
        torch.save(model.state_dict(), os.path.join(args.out, "model_final.pt"))
        print("saved to", args.out)
    else:
        print("训练因发散停止, 未覆盖 checkpoint.pt (可 --resume 从最近干净权重继续)")


if __name__ == "__main__":
    main()
