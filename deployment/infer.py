import json, re, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import List, Tuple, Dict

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

# --- heads & model builders (mirrors train.py) ---
def _get_feature_dim(model, arch: str) -> int:
    if arch.startswith("resnet"):          return model.fc.in_features
    if arch.startswith("convnext"):        return model.classifier[2].in_features
    if arch.startswith("efficientnet_v2"): return model.classifier[1].in_features
    if arch.startswith("mobilenet_v3"):    return model.classifier[0].in_features
    raise ValueError(f"Unsupported arch: {arch}")

def _attach_head(model, arch: str, head: nn.Module) -> nn.Module:
    if arch.startswith("resnet"):            model.fc = head
    elif arch.startswith("convnext"):        model.classifier = nn.Sequential(nn.Flatten(1), head)
    elif arch.startswith("efficientnet_v2"): model.classifier = nn.Sequential(nn.Dropout(p=0.0), head)
    elif arch.startswith("mobilenet_v3"):    model.classifier = head
    else: raise ValueError(f"Unsupported arch: {arch}")
    return model

def _make_head(in_dim: int, num_classes: int, hidden: List[int], dropout: float, norm: str):
    layers=[]; last=in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if   norm=="batchnorm": layers.append(nn.BatchNorm1d(h))
        elif norm=="layernorm": layers.append(nn.LayerNorm(h))
        layers += [nn.ReLU(inplace=True), nn.Dropout(dropout)]
        last=h
    layers.append(nn.Linear(last, num_classes))
    return nn.Sequential(*layers)

def build_model_from_config(cfg: dict, num_classes: int):
    arch   = cfg["model"]["arch"]
    pretr  = cfg["model"].get("pretrained", True)
    if   arch=="resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretr else None)
    elif arch=="convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretr else None)
    elif arch=="convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretr else None)
    elif arch=="efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretr else None)
    elif arch=="mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretr else None)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    in_dim = _get_feature_dim(model, arch)
    head = _make_head(
        in_dim, num_classes,
        cfg["model"]["head_hidden"],
        cfg["model"]["head_dropout"],
        cfg["model"]["head_norm"]
    )
    return _attach_head(model, arch, head), arch

def load_predict_fn(
    config_path: str,
    ckpt_path: str,
    classes_path: str
):
    with open(config_path, "r") as f: cfg = json.load(f)
    with open(classes_path, "r") as f:
        classes_raw = json.load(f)
    # classes_raw is [{"id":..,"name":..}] from your run
    id2name = {int(c["id"]): c["name"] for c in classes_raw}
    name2id = {v:k for k,v in id2name.items()}
    num_classes = len(id2name)

    model, arch = build_model_from_config(cfg, num_classes)
    dev = _device()
    sd = torch.load(ckpt_path, map_location="cpu")
    state_dict = sd.get("model", sd)  # supports either format
    model.load_state_dict(state_dict, strict=True)
    model.to(dev).eval()

    # preprocess (letterbox like training)
    size = int(cfg["preprocess"].get("image_size", 224))
    tfm = transforms.Compose([
        transforms.Resize((size, size)),   # simple square resize for GUI crops
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    @torch.no_grad()
    def predict_pils(pils: List[Image.Image], topk: int = 3):
        if not pils: return []
        xb = torch.stack([tfm(im.convert("RGB")) for im in pils], 0).to(dev)
        with torch.amp.autocast(device_type=("mps" if dev.type=="mps" else dev.type), enabled=True):
            logits = model(xb)
            probs  = F.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()
        topk_ids = probs.topk(k=min(topk, num_classes), dim=1).indices.cpu().numpy()
        out = []
        for i, ids in enumerate(topk_ids):
            out.append([(id2name[int(j)], float(probs_np[i, int(j)])) for j in ids])
        return out

    return predict_pils, list(name2id.keys())
