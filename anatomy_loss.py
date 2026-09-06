"""解剖先验损失(创新点): 下肢全长 X 光独有。

1. layout_loss: 约束预测框的类别垂直质心满足 髋(y) < 膝(y) < 踝(y)。
   用分类概率软加权, 不依赖硬 argmax, 可微。

2. symmetry_loss: 左右腿镜像对称。约束原图预测 ≈ 翻转图预测的水平镜像
   (x → 1-x, y/h/w 不变)。
"""
import torch
import torch.nn.functional as F


def layout_loss(logits, pred_boxes):
    """logits: (B, Q, C+1) class logits (最后一维是 no-object 类)
    pred_boxes: (B, Q, 4) cxcywh 归一化 (sigmoid 输出, 0-1)
    返回标量 hinge loss, 约束 hip_y < knee_y < ankle_y。"""
    prob = F.softmax(logits, dim=-1)[:, :, :3]  # (B, Q, 3) 前3类前景概率
    y_center = pred_boxes[:, :, 1]              # (B, Q) 归一化 y 中心

    # 每类加权 y 中心 (按分类概率加权)
    weighted_y = prob * y_center.unsqueeze(-1)  # (B, Q, 3)
    denom = prob.sum(dim=1).clamp(min=1e-6)     # (B, 3)
    class_y = weighted_y.sum(dim=1) / denom     # (B, 3)

    # 类索引: 0=Hip(应最上), 1=Knee(中), 2=Ankle(最下)
    loss = F.relu(class_y[:, 0] - class_y[:, 1]).mean() + \
           F.relu(class_y[:, 1] - class_y[:, 2]).mean()
    return loss


def symmetry_loss(pred_boxes_orig, pred_boxes_flip):
    """pred_boxes_orig: 原图预测 (B, Q, 4) cxcywh 归一化
    pred_boxes_flip: 水平翻转图预测 (B, Q, 4)
    约束: 原图预测 ≈ 翻转图预测的 x 镜像 (x→1-x)。
    用 L1 距离。"""
    flipped = pred_boxes_flip.clone()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]  # x 中心镜像
    # y/h/w 不变
    return F.l1_loss(pred_boxes_orig, flipped)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, Q, C = 2, 100, 4
    logits = torch.randn(B, Q, C)
    boxes = torch.rand(B, Q, 4)
    print("layout_loss =", layout_loss(logits, boxes).item())

    b1 = torch.rand(B, Q, 4)
    b2 = torch.rand(B, Q, 4)
    print("symmetry_loss =", symmetry_loss(b1, b2).item())
