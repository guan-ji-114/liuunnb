# DETR 检测器项目 — 进度存档

> 最后更新: 2026-08-21。跨会话存档, 恢复进程先读这里。

## 一、目标

- **非 YOLO** 检测器 + 解剖先验创新点, 提升下肢 X 光检测的 **mAP50-95**
- 输出仍是框回归 (Hip/Knee/Ankle 3 类)
- YOLO12n 基线: mAP50=0.991, mAP50-95=0.591 (Hip 0.563 / Knee 0.709 / Ankle 0.501)

## 二、方案 (已批准)

**DETR (transformers DetrForObjectDetection, facebook/detr-resnet-50 预训练) + 解剖先验损失 + 提高分辨率**

创新点 (下肢全长 X 光独有, 通用检测器没有):
1. **L_layout**: 髋(y) < 膝(y) < 踝(y) 垂直顺序先验 (soft, 分类概率加权)
2. **L_sym**: 左右对称 (翻转等变, x→1-x 镜像)

分辨率: 640 (YOLO) → 1024/2048 (DETR), 踝目标从 ~50px 提到 ~150px, 是 mAP50-95 提升主杠杆。

## 三、代码 (E:\Renming\experiments\detr\, 全部完成 + 验证)

| 文件 | 状态 |
|---|---|
| `dataset.py` | ✅ YOLO → DETR labels 格式 |
| `anatomy_loss.py` | ✅ layout_loss + symmetry_loss (创新点) |
| `train.py` | ✅ DETR 微调 + 先验 loss, 断点续训, smoke 通过 |
| `evaluate.py` | ✅ COCO mAP (mAP50-95/mAP50/mAP75/逐类), 流程跑通 |

权重: `weights/facebook_detr_resnet50/` (已下载, 从 hf-mirror, 165MB)

## 四、环境依赖 (已装)

- transformers 5.15.1 (DETR 实现)
- **timm 1.0.28** (DETR backbone 需要, 必装)
- **torchmetrics 1.9.0 + faster-coco-eval 1.7.2** (评估 MAP 需要)
- 关键: torchmetrics MAP 需 `backend="faster_coco_eval"` (Windows 装不了 pycocotools)

## 五、训练命令

```bash
cd E:\Renming\experiments\detr
# 小规模验证 (当前后台运行中)
python -u train.py --epochs 10 --limit 200 --w-layout 0.5 --w-sym 0.5 --out runs/verify

# 全量训练 (30-50 epoch, 8GB 约 30-50 小时)
python -u train.py --epochs 40 --w-layout 0.5 --w-sym 0.5 --out runs/full

# 消融
python -u train.py --epochs 40 --w-layout 0 --w-sym 0 --out runs/base      # 基线无先验
python -u train.py --epochs 40 --w-layout 0.5 --w-sym 0 --out runs/layout  # 只布局
python -u train.py --epochs 40 --w-layout 0 --w-sym 0.5 --out runs/sym     # 只对称
```

## 六、评估命令

```bash
python -u evaluate.py --ckpt runs/full/model_final.pt --out runs/eval
# 输出 metrics.json: mAP50-95 / mAP50 / mAP75 / 逐类 mAP50-95
```

## 七、关键参数

- 分辨率: `--shortest-edge 1024 --longest-edge 2048` (8GB 显存 batch=1 可行)
- 学习率: `--lr 5e-5` (微调)
- `--freeze-backbone` 可选 (冻结 resnet backbone 加速, 有 mAP 风险)
- **`--amp` 混合精度 bf16** (默认**关**=fp32; bf16 的 GIoU 梯度会 NaN, 不可用于此任务)
- **`--save-every 500`** 每 N 步存 checkpoint (断点续训, 防笔记本睡眠/中断丢进度)
- **`--clip-max-norm 0.1`** 梯度裁剪 (DETR 稳定必需, 否则 box 头爆炸 NaN)
- **预缓存** `E:\Renming\cache\{split}\*.jpg`: 原始 22.8MB BMP 缩放成小 JPEG, dataset.py 自动读缓存, 训练提速 ~28%, 续训跳步提速 ~12x
- num_labels=3 (class_labels_classifier 自动 92→4 重初始化)

## 八、踩坑记录

1. **timm 未装**: DETR backbone 用 TimmBackbone, 必须 `pip install timm`
2. **torchmetrics MAP 需要 coco backend**: Windows 装不了 pycocotools → 用 `faster-coco-eval` + `backend="faster_coco_eval"`
3. **设备不匹配**: 预测在 cuda、GT 在 cpu, metric.update 前统一 `.cpu()`
4. **逐类 map50 不存在**: torchmetrics 只有 `map_per_class` (mAP50-95), 没有 `map_50_per_class`
5. **权重下载**: 从 `hf-mirror.com/facebook/detr-resnet-50/resolve/main/` 用 urllib 下载 (绕过 xet)
6. **笔记本 GPU 发热降频**: 200 张/epoch 从 183s 涨到 308s, fp16 可缓解(功耗更低)
7. **空标注图崩溃**: 数据集有 0 框的空 txt (train 2 张, val 4 张), 空 boxes → `torch.tensor([])` 变 1D `(0,)`, DETR 匈牙利匹配 `torch.cdist` 报 "X2 got: 1D"。修复: dataset.py 过滤空标注图 + boxes `reshape(-1,4)`
8. **训练 NaN (最终根因)**: **bf16 精度太低** — GIoU 梯度算 `1/union` 时 union 下溢成 0 → `inf×0=NaN`, 损失有限但梯度 NaN, 一旦进入持续 NaN(卡死, 99% 步跳过)。fp16 也有(10位mantissa, 但 LayerNorm 溢出)。**修复: 用 fp32**(默认, 24位mantissa)。曾误判为 fp16 溢出/缺梯度裁剪, 梯度裁剪(0.1)+NaN步跳过+try/except 是必要保护但治标不治本。
9. **bf16 卡死教训**: NaN grad 步跳过只是防止权重污染, 但如果 NaN 持续(bf16 固有问题), 模型冻结不更新, 白跑 9 小时。诊断方法: 统计日志里 WARNING 占比。
10. **fp32 全量 40ep 也发散 (2026-09-06 AutoDL)**: 验证集 200×10ep 干净, 但全量 4673×40ep 在 **ep15 step71450 起持续 NaN**(永久发散, 非单样本触发)。根因 = **train.py 无 LR 调度**, 恒定 lr=5e-5 跑满 40ep, DETR box 头(GIoU 含 `1/union`)后期梯度爆炸。验证只跑 10ep 恰好没越过发散点(~ep15)。**修复: cosine+预热调度 + 连续 200 NaN 自动停机 + NaN 时不覆盖 checkpoint**。结论: bf16 是必要非充分, fp32 + 无调度照样会发散。

## 九、进度 (2026-08-23 更新)

1. ✅ verify (200 张 × 10 ep) 完成: loss 2.8→0.68 无 NaN; **val mAP50-95=0.620 已超 YOLO 基线 0.591** (Ankle 0.579 vs 0.501)
2. 关键发现: **L_layout 恒为 0.0000** — 预训练 DETR 天然满足髋<膝<踝顺序, 布局先验失效 → 消融砍成 base vs sym
3. ✅ AMP 已加 + smoke 验证无 NaN (loss 3.9→2.3)
4. 🔄 全量 `full` (4868 张 × 40 ep) 训练中, AMP 开, --resume 断点续训
5. 待做: 全量完 → 评估 → 消融 `base` + `sym` → 对比基线

### 实测耗时
- fp32: 200 张/epoch = 183s(冷) → 308s(降频); 4868 张 ≈ 74-125 分钟/epoch
- AMP: 冷启动 ~0.83s/step, 稳态待全量首 epoch 实测
- 40 ep 全量 fp32 ≈ 49h; AMP 预计 ~25-30h

## 十、评判标准 (用户指定)

沿用 E:\Renming\新建文本文档.txt 的 mAP 口径:
- 主指标 mAP50-95 (目标 > 0.591), 副指标 mAP50 (基线 0.991)
- 逐类 Hip/Knee/Ankle

## 十一、AutoDL 部署 (2026-09-06, 代码已改跨平台)

- 路径已参数化: 所有 `E:\Renming` 硬编码改成 `--data-root` / 环境变量 `DATA_ROOT`,
  Linux 默认 `/root/autodl-tmp/Renming`, Windows 默认仍 `E:\Renming`
- 新增 `--workers` (DataLoader 进程数, 默认 0, AutoDL 可 4-8)
- 新增脚本: `setup_autodl.sh` (装依赖+下权重), `run_train.sh` (nohup 后台训练+续训)
- **上传清单 (训练只这些, 不必传 128GB BMP)**:
  - `labels/` (小), `cache/` (缩放 JPEG, 几 GB), `experiments/detr/` (代码+权重)
  - 若要在 AutoDL 上评估则还需 `images/*.bmp` (evaluate 需原图 H/W 做 GT 反归一化)
  - 全量 checkpoint `runs/full/checkpoint.pt` (498MB, step 33210) 若要续训一并传
- 上机顺序: `bash setup_autodl.sh` → 小样本验证 → `bash run_train.sh`
- 评估可下回本地跑 `evaluate.py` (本地有 BMP), 或传 BMP 上 AutoDL 跑
- **2026-09-06 进度**: 4090 全量 40ep 已跑到 ep15 step71450 发散(见踩坑10)。干净 checkpoint 在 **step71000** (step71495 手动 Ctrl+C 未跨过 71500 保存点)。续训流程: ① git pull 最新 train.py(已加调度+停机) → ② 校验 `checkpoint.pt` 权重非 NaN(命令见下) → ③ `bash run_train.sh`(自动 `--resume`)。

