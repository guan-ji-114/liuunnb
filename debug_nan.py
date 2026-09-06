"""定位 NaN 梯度来源: 加载 33210 权重, 单步 forward+backward + detect_anomaly."""
import os
import torch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from dataset import DetrDataset, collate, N_CLASSES
from anatomy_loss import layout_loss, symmetry_loss

DATA_ROOT = os.environ.get("DATA_ROOT", "/root/autodl-tmp/Renming")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "weights", "facebook_detr_resnet50")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device, "| torch", torch.__version__)

processor = DetrImageProcessor.from_pretrained(
    WEIGHTS_DIR, size={"shortest_edge": 1024, "longest_edge": 2048})
model = DetrForObjectDetection.from_pretrained(
    WEIGHTS_DIR, num_labels=N_CLASSES, ignore_mismatched_sizes=True).to(device)
ck = torch.load("runs/full/checkpoint.pt", map_location=device)
model.load_state_dict(ck["model"])
print("loaded checkpoint step", ck["global_step"])

ds = DetrDataset(os.path.join(DATA_ROOT, "images", "train"),
                 os.path.join(DATA_ROOT, "labels", "train"),
                 processor, cache_dir=os.path.join(DATA_ROOT, "cache", "train"))
loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate)

model.train()
torch.autograd.set_detect_anomaly(True)

for i, batch in enumerate(loader):
    if i >= 1:
        break
    pv = batch["pixel_values"].to(device)
    pm = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]

    out = model(pixel_values=pv, pixel_mask=pm, labels=labels)
    total = out.loss
    l_layout = layout_loss(out.logits, out.pred_boxes)

    # 预测框统计 (检查是否退化)
    with torch.no_grad():
        w = out.pred_boxes[..., 2]
        h = out.pred_boxes[..., 3]
        area = w * h
        print(f"pred box: w.min={w.min().item():.3e} h.min={h.min().item():.3e} "
              f"area.min={area.min().item():.3e} area.nan={bool(torch.isnan(area).any())}")

    pv_flip = torch.flip(pv, dims=[-1])
    pm_flip = torch.flip(pm, dims=[-1])
    out_flip = model(pixel_values=pv_flip, pixel_mask=pm_flip)
    l_sym = symmetry_loss(out.pred_boxes, out_flip.pred_boxes)

    loss = total + 0.5 * l_layout + 0.5 * l_sym
    print(f"loss: total={total.item():.4f} layout={l_layout.item():.4f} "
          f"sym={l_sym.item():.4f} sum={loss.item():.4f}")

    print("backward ...")
    loss.backward()
    print("backward OK (no NaN)")
