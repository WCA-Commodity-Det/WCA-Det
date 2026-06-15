import argparse
import inspect
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import ultralytics.nn.modules as nn_modules
import ultralytics.nn.modules.head as nn_head
import ultralytics.nn.tasks as tasks
import ultralytics.utils.loss as yolo_loss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


DATA_YAML = "configs/dataset_merged_weak_aug.yaml"
PRETRAINED_WEIGHTS = "yolo11m.pt"
SIMAM_C3K2_LAYERS = set()
SIMAM_ALPHA = 0.0
SIMAM_MODE = "boost"
SIMAM_GATE_INIT = 0.0
ECA_C3K2_LAYERS = set()
ECA_ALPHA = 0.10
ECA_LAYER_ALPHAS = {}
ECA_K_SIZE = 3
ECA_HEAD_CLS = False
ECA_HEAD_LEVELS = {0, 1, 2}
TRAIN_SCOPE = "all"
SIMAM_HEAD_CLS = False
SIMAM_HEAD_LEVELS = {0, 1, 2}
CLASS_WEIGHTS = None
CLS_LOSS_MODE = "bce"
FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 1.0
_ORIGINAL_C3K2_FORWARD = nn_modules.C3k2.forward
_ORIGINAL_DETECT_FORWARD_HEAD = nn_head.Detect.forward_head
_ORIGINAL_V8_DETECTION_LOSS_INIT = yolo_loss.v8DetectionLoss.__init__


def _infer_channels_from_parse_model():
    frame = inspect.currentframe().f_back
    while frame:
        local_vars = frame.f_locals
        if "ch" in local_vars and "f" in local_vars:
            ch = local_vars["ch"]
            f = local_vars["f"]
            if isinstance(f, int):
                return ch[f]
            return sum(ch[i] for i in f)
        frame = frame.f_back
    return None


class EMA(nn.Module):
    """Efficient Multi-scale Attention used as the A module."""

    def __init__(self, channels=None, factor=8):
        super().__init__()
        channels = channels or _infer_channels_from_parse_model()
        if channels is None:
            raise ValueError("EMA channels could not be inferred. Pass channels in the YAML.")

        groups = min(factor, channels)
        while channels % groups != 0:
            groups -= 1

        self.groups = groups
        group_channels = channels // groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        if hasattr(self, "eca_attn"):
            out = self.eca_attn(out)
        return out


class SimAM(nn.Module):
    """Parameter-free spatial attention used as the C module."""

    def __init__(self, channels=None, e_lambda=1e-4, alpha=None, mode=None):
        super().__init__()
        self.e_lambda = e_lambda
        self.alpha = alpha
        self.mode = mode

    def forward(self, x):
        return simam_tensor(x, self.e_lambda, alpha=self.alpha, mode=self.mode)


class ECA(nn.Module):
    """Efficient channel attention for fine-grained product recognition."""

    def __init__(self, channels=None, k_size=3, alpha=0.10):
        super().__init__()
        self.alpha = alpha
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * (1.0 + self.alpha * (self.sigmoid(y) - 0.5))


def simam_tensor(x, e_lambda=1e-4, alpha=None, mode=None):
    alpha = SIMAM_ALPHA if alpha is None else alpha
    mode = SIMAM_MODE if mode is None else mode
    _, _, h, w = x.size()
    n = h * w - 1
    x_minus_mu_square = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
    y = x_minus_mu_square / (
        4 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / n + e_lambda)
    ) + 0.5
    attention = torch.sigmoid(y)

    if mode == "direct":
        return x * attention
    if mode == "blend":
        return x + alpha * (x * attention - x)
    if mode == "boost":
        return x * (1.0 + alpha * (attention - 0.5))
    if mode == "learnable":
        return x * (1.0 + alpha * (attention - 0.5))
    raise ValueError(f"Unsupported SimAM mode: {mode}")


def _c3k2_forward_with_selective_simam(self, x):
    out = _ORIGINAL_C3K2_FORWARD(self, x)
    if getattr(self, "i", None) in ECA_C3K2_LAYERS:
        if hasattr(self, "eca_attn"):
            out = self.eca_attn(out)
    if getattr(self, "i", None) in SIMAM_C3K2_LAYERS:
        if SIMAM_MODE == "learnable":
            if hasattr(self, "simam_gate"):
                gate_alpha = SIMAM_ALPHA * torch.tanh(self.simam_gate)
                return simam_tensor(out, alpha=gate_alpha, mode="learnable")
            return out
        return simam_tensor(out)
    return out


def enable_selective_simam():
    nn_modules.C3k2.forward = _c3k2_forward_with_selective_simam
    tasks.C3k2.forward = _c3k2_forward_with_selective_simam


def _detect_forward_head_with_cls_simam(self, x, box_head=None, cls_head=None):
    if not SIMAM_HEAD_CLS and not ECA_HEAD_CLS:
        return _ORIGINAL_DETECT_FORWARD_HEAD(self, x, box_head=box_head, cls_head=cls_head)
    if box_head is None or cls_head is None:
        return dict()

    bs = x[0].shape[0]
    boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
    cls_feats = []
    for i in range(self.nl):
        feat = x[i]
        if ECA_HEAD_CLS and i in ECA_HEAD_LEVELS and hasattr(self, "eca_cls_attn"):
            feat = self.eca_cls_attn[i](feat)
        if SIMAM_HEAD_CLS and i in SIMAM_HEAD_LEVELS:
            feat = simam_tensor(feat)
        cls_feats.append(feat)
    scores = torch.cat(
        [cls_head[i](cls_feats[i]).view(bs, self.nc, -1) for i in range(self.nl)],
        dim=-1,
    )
    return dict(boxes=boxes, scores=scores, feats=x)


def enable_detect_cls_simam():
    nn_head.Detect.forward_head = _detect_forward_head_with_cls_simam
    if hasattr(nn_modules, "Detect"):
        nn_modules.Detect.forward_head = _detect_forward_head_with_cls_simam
    if hasattr(tasks, "Detect"):
        tasks.Detect.forward_head = _detect_forward_head_with_cls_simam


def official_bbox_loss_forward(
    self,
    pred_dist,
    pred_bboxes,
    anchor_points,
    target_bboxes,
    target_scores,
    target_scores_sum,
    fg_mask,
    imgsz=None,
    stride=None,
):
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        loss_dfl = torch.zeros_like(loss_iou)

    return loss_iou, loss_dfl


def patch_official_bbox_loss():
    yolo_loss.BboxLoss.forward = official_bbox_loss_forward


class ElementwiseFocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.5, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, pred, label):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1.0 - label) * (1.0 - pred_prob)
        loss = loss * (1.0 - p_t).pow(self.gamma)
        if self.alpha >= 0:
            alpha_factor = label * self.alpha + (1.0 - label) * (1.0 - self.alpha)
            loss = loss * alpha_factor
        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, 1, -1).to(loss.device)
        return loss


class ElementwiseWeightedBCELoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, pred, label):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, 1, -1).to(loss.device)
        return loss


def patch_detection_cls_loss():
    if CLS_LOSS_MODE == "bce":
        if CLASS_WEIGHTS is None:
            yolo_loss.v8DetectionLoss.__init__ = _ORIGINAL_V8_DETECTION_LOSS_INIT
            return

        def weighted_bce_init(self, *args, **kwargs):
            _ORIGINAL_V8_DETECTION_LOSS_INIT(self, *args, **kwargs)
            self.bce = ElementwiseWeightedBCELoss(class_weights=CLASS_WEIGHTS).to(self.device)

        yolo_loss.v8DetectionLoss.__init__ = weighted_bce_init
        return

    if CLS_LOSS_MODE != "focal":
        raise ValueError(f"Unsupported cls loss: {CLS_LOSS_MODE}")

    def focal_init(self, *args, **kwargs):
        _ORIGINAL_V8_DETECTION_LOSS_INIT(self, *args, **kwargs)
        self.bce = ElementwiseFocalLoss(
            gamma=FOCAL_GAMMA,
            alpha=FOCAL_ALPHA,
            class_weights=CLASS_WEIGHTS,
        ).to(self.device)

    yolo_loss.v8DetectionLoss.__init__ = focal_init


def register_custom_modules():
    for module in (EMA, SimAM, ECA):
        setattr(tasks, module.__name__, module)
        setattr(nn_modules, module.__name__, module)
        setattr(sys.modules["__main__"], module.__name__, module)
    enable_selective_simam()
    enable_detect_cls_simam()
    patch_detection_cls_loss()


def attach_learnable_simam_gates(model, gate_init=0.0):
    attached = []
    net = getattr(model, "model", model)
    for module in net.modules():
        if getattr(module, "i", None) in SIMAM_C3K2_LAYERS:
            if not hasattr(module, "simam_gate"):
                module.simam_gate = nn.Parameter(torch.tensor(float(gate_init)))
            attached.append(getattr(module, "i", None))
    print(f"Attached learnable SimAM gates to C3k2 layers: {attached}")
    return attached


def attach_eca_modules(model, alpha=0.10, k_size=None):
    k_size = ECA_K_SIZE if k_size is None else k_size
    attached = []
    net = getattr(model, "model", model)
    for module in net.modules():
        layer_id = getattr(module, "i", None)
        if layer_id in ECA_C3K2_LAYERS:
            layer_alpha = ECA_LAYER_ALPHAS.get(layer_id, alpha)
            if not hasattr(module, "eca_attn"):
                module.eca_attn = ECA(k_size=k_size, alpha=layer_alpha)
            else:
                module.eca_attn.alpha = layer_alpha
            attached.append((layer_id, layer_alpha))
    if attached:
        print(f"Attached ECA modules to C3k2 layers: {attached}, default_alpha={alpha}, k_size={k_size}")
    return attached


def attach_detect_eca_modules(model, alpha=0.10, k_size=None):
    k_size = ECA_K_SIZE if k_size is None else k_size
    if not ECA_HEAD_CLS:
        return []
    attached = []
    net = getattr(model, "model", model)
    for module in net.modules():
        if isinstance(module, nn_head.Detect):
            if not hasattr(module, "eca_cls_attn"):
                module.eca_cls_attn = nn.ModuleList([ECA(k_size=k_size, alpha=alpha) for _ in range(module.nl)])
            attached.append(getattr(module, "i", None))
    if attached:
        print(f"Attached Detect cls ECA modules to heads: {attached}, levels={sorted(ECA_HEAD_LEVELS)}, alpha={alpha}, k_size={k_size}")
    return attached


class SimAMDetectionTrainer(DetectionTrainer):
    def setup_model(self):
        ckpt = super().setup_model()
        if SIMAM_MODE == "learnable":
            attach_learnable_simam_gates(self.model, SIMAM_GATE_INIT)
        attach_eca_modules(self.model, ECA_ALPHA, ECA_K_SIZE)
        attach_detect_eca_modules(self.model, ECA_ALPHA, ECA_K_SIZE)
        attach_class_weights(self.model, CLASS_WEIGHTS)
        if TRAIN_SCOPE not in {"all", "eca"}:
            configure_trainable_params(self.model, TRAIN_SCOPE)
        return ckpt

    def _setup_train(self):
        super()._setup_train()
        if TRAIN_SCOPE == "all":
            return

        configure_trainable_params(self.model, TRAIN_SCOPE)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()
        self.scheduler.last_epoch = self.start_epoch - 1


def freeze_arg_for_train_scope(train_scope):
    if train_scope in {"head", "gates_head"}:
        if any(layer < 11 for layer in ECA_C3K2_LAYERS):
            return [layer for layer in range(11) if layer not in ECA_C3K2_LAYERS]
        return 11
    if train_scope == "gates":
        raise ValueError("train-scope=gates is not reliable in Ultralytics rebuilds; use gates_head.")
    return None


def configure_trainable_params(model, train_scope="all"):
    net = getattr(model, "model", model)
    layers = getattr(net, "model", net)
    if train_scope == "all":
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Train scope: all ({trainable:,} trainable parameters)")
        return

    for param in net.parameters():
        param.requires_grad = False

    for module in net.modules():
        if hasattr(module, "simam_gate"):
            module.simam_gate.requires_grad = True
        if train_scope in {"eca", "head", "gates_head"}:
            if hasattr(module, "eca_attn"):
                for param in module.eca_attn.parameters():
                    param.requires_grad = True
            if hasattr(module, "eca_cls_attn"):
                for param in module.eca_cls_attn.parameters():
                    param.requires_grad = True

    if train_scope in {"head", "gates_head"}:
        for module in layers[11:]:
            for param in module.parameters():
                param.requires_grad = True

    if train_scope == "head":
        for module in net.modules():
            if hasattr(module, "simam_gate"):
                module.simam_gate.requires_grad = False

    if train_scope not in {"eca", "gates", "head", "gates_head"}:
        raise ValueError(f"Unsupported train scope: {train_scope}")

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Train scope: {train_scope} ({trainable:,} trainable parameters)")
    for name, param in net.named_parameters():
        if "simam_gate" in name or "eca_attn" in name or "eca_cls_attn" in name:
            init_value = param.detach().float().mean().cpu().item()
            print(f"  {name}: requires_grad={param.requires_grad}, init_mean={init_value:.6f}")


def parse_simam_layers(value):
    if value.lower() == "none":
        return set()
    if value.lower() == "all":
        return {2, 4, 6, 8, 13, 16, 19, 22}
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_head_levels(value):
    if value.lower() == "none":
        return set()
    if value.lower() == "all":
        return {0, 1, 2}
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_class_weights(value):
    if value is None or value.lower() == "none":
        return None
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != 12:
        raise ValueError(f"Expected 12 class weights, got {len(weights)}: {weights}")
    return weights


def parse_eca_layer_alphas(value):
    if value is None or value.lower() == "none":
        return {}
    result = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layer, layer_alpha = item.split(":", 1)
        result[int(layer.strip())] = float(layer_alpha.strip())
    return result


def attach_class_weights(model, weights):
    if weights is None:
        return
    net = getattr(model, "model", model)
    net.class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Using class weights: {weights}")


def train_perfect(
    data=DATA_YAML,
    model_yaml="configs/yolo11-ema-eca-perfect.yaml",
    epochs=200,
    batch=8,
    workers=4,
    name="wca_det_yolo11_ema_eca",
    imgsz=640,
    simam_layers="none",
    simam_alpha=0.0,
    simam_mode="boost",
    simam_gate_init=0.0,
    simam_head_cls=False,
    simam_head_levels="all",
    eca_layers="none",
    eca_alpha=0.10,
    eca_layer_alphas=None,
    eca_k_size=3,
    eca_head_cls=False,
    eca_head_levels="all",
    train_scope="all",
    weights=PRETRAINED_WEIGHTS,
    optimizer="SGD",  # 🌟 修改 1：修改默认优化器为 SGD，封杀带有 Bug 的 Muon
    lr0=None,
    weight_decay=None,
    loss="official",
    mixup=0.15,
    copy_paste=0.15,
    mosaic=1.0,
    hsv_v=0.4,
    auto_augment="randaugment",
    close_mosaic=15,
    warmup_epochs=None,
    warmup_bias_lr=None,
    box_gain=None,
    cls_gain=None,
    dfl_gain=None,
    class_weights=None,
    cls_loss="bce",
    focal_gamma=1.0,
    focal_alpha=0.5,
):
    global CLASS_WEIGHTS, CLS_LOSS_MODE, ECA_ALPHA, ECA_C3K2_LAYERS, ECA_HEAD_CLS, ECA_HEAD_LEVELS, ECA_K_SIZE, ECA_LAYER_ALPHAS, FOCAL_ALPHA, FOCAL_GAMMA, SIMAM_ALPHA, SIMAM_C3K2_LAYERS, SIMAM_GATE_INIT, SIMAM_HEAD_CLS, SIMAM_HEAD_LEVELS, SIMAM_MODE, TRAIN_SCOPE
    SIMAM_C3K2_LAYERS = parse_simam_layers(simam_layers)
    SIMAM_ALPHA = simam_alpha
    SIMAM_MODE = simam_mode
    SIMAM_GATE_INIT = simam_gate_init
    SIMAM_HEAD_CLS = simam_head_cls
    SIMAM_HEAD_LEVELS = parse_head_levels(simam_head_levels)
    ECA_C3K2_LAYERS = parse_simam_layers(eca_layers)
    ECA_ALPHA = eca_alpha
    ECA_LAYER_ALPHAS = parse_eca_layer_alphas(eca_layer_alphas)
    ECA_K_SIZE = eca_k_size
    ECA_HEAD_CLS = eca_head_cls
    ECA_HEAD_LEVELS = parse_head_levels(eca_head_levels)
    TRAIN_SCOPE = train_scope
    CLASS_WEIGHTS = parse_class_weights(class_weights)
    CLS_LOSS_MODE = cls_loss
    FOCAL_GAMMA = focal_gamma
    FOCAL_ALPHA = focal_alpha
    register_custom_modules()
    if loss == "official":
        patch_official_bbox_loss()
    print(f"Using SimAM on C3k2 layers: {sorted(SIMAM_C3K2_LAYERS)}")
    print(f"Using Detect cls SimAM: {SIMAM_HEAD_CLS}, levels: {sorted(SIMAM_HEAD_LEVELS)}")
    print(f"Using SimAM mode: {SIMAM_MODE}, alpha: {SIMAM_ALPHA}")
    print(f"Using SimAM gate init: {SIMAM_GATE_INIT}")
    print(f"Using ECA on C3k2 layers: {sorted(ECA_C3K2_LAYERS)}, alpha: {ECA_ALPHA}, layer_alphas: {ECA_LAYER_ALPHAS}, k_size: {ECA_K_SIZE}")
    print(f"Using Detect cls ECA: {ECA_HEAD_CLS}, levels: {sorted(ECA_HEAD_LEVELS)}, k_size: {ECA_K_SIZE}")
    print(f"Using bbox loss: {loss}")
    print(f"Using cls loss: {CLS_LOSS_MODE}, focal_gamma: {FOCAL_GAMMA}, focal_alpha: {FOCAL_ALPHA}")

    model = YOLO(model_yaml, task="detect")
    if SIMAM_MODE == "learnable":
        attach_learnable_simam_gates(model, SIMAM_GATE_INIT)
    attach_eca_modules(model, ECA_ALPHA, ECA_K_SIZE)
    attach_detect_eca_modules(model, ECA_ALPHA, ECA_K_SIZE)
    model.load(weights)
    attach_class_weights(model, CLASS_WEIGHTS)
    configure_trainable_params(model, train_scope)
    freeze_arg = freeze_arg_for_train_scope(train_scope)

    train_args = dict(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="0",
        workers=workers,
        project="runs/train",
        name=name,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        hsv_v=hsv_v,
        auto_augment=auto_augment,
        cos_lr=True,
        close_mosaic=close_mosaic,
        patience=0,           # 🌟 修改 2：强制关闭早停机制，榨干模型性能！
        amp=True,
        optimizer=optimizer,
        seed=0,
        deterministic=True,
    )
    if freeze_arg is not None:
        train_args["freeze"] = freeze_arg
    if lr0 is not None:
        train_args["lr0"] = lr0
    if weight_decay is not None:
        train_args["weight_decay"] = weight_decay
    if box_gain is not None:
        train_args["box"] = box_gain
    if cls_gain is not None:
        train_args["cls"] = cls_gain
    if dfl_gain is not None:
        train_args["dfl"] = dfl_gain
    if warmup_epochs is not None:
        train_args["warmup_epochs"] = warmup_epochs
    if warmup_bias_lr is not None:
        train_args["warmup_bias_lr"] = warmup_bias_lr
    return model.train(trainer=SimAMDetectionTrainer, **train_args)


def val_perfect(
    data=DATA_YAML,
    model_yaml="configs/yolo11-ema-eca-perfect.yaml",
    batch=8,
    workers=4,
    name="wca_det_yolo11_ema_eca_val",
    imgsz=640,
    simam_layers="none",
    simam_alpha=0.0,
    simam_mode="boost",
    simam_gate_init=0.0,
    simam_head_cls=False,
    simam_head_levels="all",
    eca_layers="none",
    eca_alpha=0.10,
    eca_layer_alphas=None,
    eca_k_size=3,
    eca_head_cls=False,
    eca_head_levels="all",
    weights=PRETRAINED_WEIGHTS,
    loss="official",
    augment=False,
):
    global ECA_ALPHA, ECA_C3K2_LAYERS, ECA_HEAD_CLS, ECA_HEAD_LEVELS, ECA_K_SIZE, ECA_LAYER_ALPHAS, SIMAM_ALPHA, SIMAM_C3K2_LAYERS, SIMAM_GATE_INIT, SIMAM_HEAD_CLS, SIMAM_HEAD_LEVELS, SIMAM_MODE
    SIMAM_C3K2_LAYERS = parse_simam_layers(simam_layers)
    SIMAM_ALPHA = simam_alpha
    SIMAM_MODE = simam_mode
    SIMAM_GATE_INIT = simam_gate_init
    SIMAM_HEAD_CLS = simam_head_cls
    SIMAM_HEAD_LEVELS = parse_head_levels(simam_head_levels)
    ECA_C3K2_LAYERS = parse_simam_layers(eca_layers)
    ECA_ALPHA = eca_alpha
    ECA_LAYER_ALPHAS = parse_eca_layer_alphas(eca_layer_alphas)
    ECA_K_SIZE = eca_k_size
    ECA_HEAD_CLS = eca_head_cls
    ECA_HEAD_LEVELS = parse_head_levels(eca_head_levels)
    register_custom_modules()
    if loss == "official":
        patch_official_bbox_loss()
    print(f"Validating SimAM on C3k2 layers: {sorted(SIMAM_C3K2_LAYERS)}")
    print(f"Validating Detect cls SimAM: {SIMAM_HEAD_CLS}, levels: {sorted(SIMAM_HEAD_LEVELS)}")
    print(f"Using SimAM mode: {SIMAM_MODE}, alpha: {SIMAM_ALPHA}")
    print(f"Using SimAM gate init: {SIMAM_GATE_INIT}")
    print(f"Validating ECA on C3k2 layers: {sorted(ECA_C3K2_LAYERS)}, alpha: {ECA_ALPHA}, layer_alphas: {ECA_LAYER_ALPHAS}, k_size: {ECA_K_SIZE}")
    print(f"Validating Detect cls ECA: {ECA_HEAD_CLS}, levels: {sorted(ECA_HEAD_LEVELS)}, k_size: {ECA_K_SIZE}")
    print(f"Using weights: {weights}")

    model = YOLO(model_yaml, task="detect")
    if SIMAM_MODE == "learnable":
        attach_learnable_simam_gates(model, SIMAM_GATE_INIT)
    attach_eca_modules(model, ECA_ALPHA, ECA_K_SIZE)
    attach_detect_eca_modules(model, ECA_ALPHA, ECA_K_SIZE)
    model.load(weights)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device="0",
        workers=workers,
        project="runs/val",
        name=name,
        plots=True,
        augment=augment,
    )
    print(f"Final val mAP50: {metrics.box.map50}")
    print(f"Final val mAP50-95: {metrics.box.map}")

    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        summary_path = Path(save_dir) / "val_summary.csv"
        summary_path.write_text(
            "\n".join(
                [
                    "name,imgsz,batch,augment,map50,map50_95",
                    f"{name},{imgsz},{batch},{augment},{metrics.box.map50},{metrics.box.map}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Saved val summary to: {summary_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SPSB YOLO11 ablation models.")
    parser.add_argument(
        "--experiment",
        default="perfect",
        choices=["perfect"],
        help="Experiment to run.",
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "val"],
        help="Train or validate the selected experiment.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for train/val.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--name", default="wca_det_yolo11_ema_eca", help="Run name.")
    parser.add_argument("--weights", default=PRETRAINED_WEIGHTS, help="Initial weights.")
    parser.add_argument("--data", default=DATA_YAML, help="Dataset YAML path.")
    parser.add_argument("--model-yaml", default="configs/yolo11-ema-eca-perfect.yaml", help="Model YAML path.")
    parser.add_argument("--optimizer", default="SGD", help="Ultralytics optimizer setting.") # 🌟 修改 3：终端参数默认值也强制锁死为 SGD
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate override.")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None, help="Weight decay override.")
    parser.add_argument(
        "--cls-loss",
        default="bce",
        choices=["bce", "focal"],
        help="Classification loss for detection head training.",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.0, help="Focal loss gamma when --cls-loss focal.")
    parser.add_argument("--focal-alpha", type=float, default=0.5, help="Focal loss alpha when --cls-loss focal.")
    parser.add_argument(
        "--loss",
        default="official",
        choices=["official", "env"],
        help="Use official CIoU/DFL bbox loss or the currently modified environment loss.",
    )
    parser.add_argument(
        "--simam-layers",
        default="none",
        help="Comma-separated C3k2 layer indices, 'all', or 'none'.",
    )
    parser.add_argument(
        "--simam-alpha",
        type=float,
        default=0.0,
        help="SimAM strength for boost/blend modes. 0 disables the effect, 1 is strongest.",
    )
    parser.add_argument(
        "--simam-mode",
        default="boost",
        choices=["boost", "blend", "direct", "learnable"],
        help="boost lightly enhances features, blend interpolates toward original SimAM, direct is original SimAM, learnable trains a gate.",
    )
    parser.add_argument(
        "--simam-gate-init",
        type=float,
        default=0.0,
        help="Initial raw gate value for learnable SimAM. 0 starts from the EMA-only output.",
    )
    parser.add_argument(
        "--simam-head-cls",
        action="store_true",
        help="Apply SimAM only to the Detect classification branch inputs.",
    )
    parser.add_argument(
        "--simam-head-levels",
        default="all",
        help="Detect feature levels for classification-branch SimAM: 'all', 'none', or comma-separated 0,1,2.",
    )
    parser.add_argument(
        "--eca-layers",
        default="none",
        help="Comma-separated C3k2 layer indices for injected ECA, 'all', or 'none'.",
    )
    parser.add_argument(
        "--eca-alpha",
        type=float,
        default=0.10,
        help="Residual ECA strength. 0 keeps the injected module as identity.",
    )
    parser.add_argument(
        "--eca-layer-alphas",
        default=None,
        help="Optional per-layer ECA strengths, e.g. '22:0.05,13:0.01'. Overrides --eca-alpha for listed layers.",
    )
    parser.add_argument(
        "--eca-k-size",
        type=int,
        default=3,
        help="ECA 1D channel-convolution kernel size. Use an odd number such as 3 or 5.",
    )
    parser.add_argument(
        "--eca-head-cls",
        action="store_true",
        help="Apply ECA only to the Detect classification branch inputs.",
    )
    parser.add_argument(
        "--eca-head-levels",
        default="all",
        help="Detect feature levels for classification-branch ECA: 'all', 'none', or comma-separated 0,1,2.",
    )
    parser.add_argument(
        "--train-scope",
        default="all",
        choices=["all", "eca", "gates", "head", "gates_head"],
        help="Which parameters to train. eca trains only ECA modules; gates_head freezes the backbone and trains gates plus detection head.",
    )
    parser.add_argument("--mixup", type=float, default=0.15, help="MixUp augmentation probability.")
    parser.add_argument("--copy-paste", dest="copy_paste", type=float, default=0.15, help="Copy-paste augmentation probability.")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability.")
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.4, help="HSV value augmentation.")
    parser.add_argument("--auto-augment", default="randaugment", help="Ultralytics auto augment policy.")
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=15, help="Epochs before the end to disable mosaic.")
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=None, help="Warmup epochs override.")
    parser.add_argument("--warmup-bias-lr", dest="warmup_bias_lr", type=float, default=None, help="Warmup bias learning rate override.")
    parser.add_argument("--box-gain", dest="box_gain", type=float, default=None, help="Box loss gain override.")
    parser.add_argument("--cls-gain", dest="cls_gain", type=float, default=None, help="Classification loss gain override.")
    parser.add_argument("--dfl-gain", dest="dfl_gain", type=float, default=None, help="DFL loss gain override.")
    parser.add_argument(
        "--class-weights",
        default=None,
        help="Comma-separated 12-class classification-loss weights, or 'none'.",
    )
    parser.add_argument("--augment-val", action="store_true", help="Use test-time augmentation during validation.")
    args = parser.parse_args()

    if args.experiment == "perfect":
        if args.mode == "train":
            train_perfect(
                epochs=args.epochs,
                data=args.data,
                model_yaml=args.model_yaml,
                batch=args.batch,
                workers=args.workers,
                name=args.name,
                imgsz=args.imgsz,
                simam_layers=args.simam_layers,
                simam_alpha=args.simam_alpha,
                simam_mode=args.simam_mode,
                simam_gate_init=args.simam_gate_init,
                simam_head_cls=args.simam_head_cls,
                simam_head_levels=args.simam_head_levels,
                eca_layers=args.eca_layers,
                eca_alpha=args.eca_alpha,
                eca_layer_alphas=args.eca_layer_alphas,
                eca_k_size=args.eca_k_size,
                eca_head_cls=args.eca_head_cls,
                eca_head_levels=args.eca_head_levels,
                train_scope=args.train_scope,
                weights=args.weights,
                optimizer=args.optimizer,
                lr0=args.lr0,
                weight_decay=args.weight_decay,
                loss=args.loss,
                mixup=args.mixup,
                copy_paste=args.copy_paste,
                mosaic=args.mosaic,
                hsv_v=args.hsv_v,
                auto_augment=args.auto_augment,
                close_mosaic=args.close_mosaic,
                warmup_epochs=args.warmup_epochs,
                warmup_bias_lr=args.warmup_bias_lr,
                box_gain=args.box_gain,
                cls_gain=args.cls_gain,
                dfl_gain=args.dfl_gain,
                class_weights=args.class_weights,
                cls_loss=args.cls_loss,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
            )
        else:
            val_perfect(
                data=args.data,
                model_yaml=args.model_yaml,
                batch=args.batch,
                workers=args.workers,
                name=args.name,
                imgsz=args.imgsz,
                simam_layers=args.simam_layers,
                simam_alpha=args.simam_alpha,
                simam_mode=args.simam_mode,
                simam_gate_init=args.simam_gate_init,
                simam_head_cls=args.simam_head_cls,
                simam_head_levels=args.simam_head_levels,
                eca_layers=args.eca_layers,
                eca_alpha=args.eca_alpha,
                eca_layer_alphas=args.eca_layer_alphas,
                eca_k_size=args.eca_k_size,
                eca_head_cls=args.eca_head_cls,
                eca_head_levels=args.eca_head_levels,
                weights=args.weights,
                loss=args.loss,
                augment=args.augment_val,
            )


if __name__ == "__main__":
    main()
