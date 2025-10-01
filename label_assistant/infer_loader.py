import os, json
from typing import List, Callable
import torch, torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224

# ---- Basic letterbox to 224x224 (keeps aspect) ----
class Letterbox224:
    def __init__(self, size=INPUT_SIZE, fill=0):
        self.size=size; self.fill=fill
    def __call__(self, img: Image.Image) -> Image.Image:
        w,h = img.size
        s = min(self.size/w, self.size/h)
        nw, nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
        img = img.resize((nw,nh), Image.BICUBIC)
        canvas = Image.new("RGB",(self.size,self.size),self.fill)
        left = (self.size-nw)//2; top=(self.size-nh)//2
        canvas.paste(img,(left,top))
        return canvas

# ---- Minimal head that fits common training setups ----
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, p=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(p),
            nn.Linear(1024, 512),    nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, 256),     nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained=False):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # up to pool
        self.head = MLPHead(2048, num_classes)
    def forward(self, x):
        f = self.features(x)                 # B, 2048, 1, 1
        f = torch.flatten(f, 1)              # B, 2048
        return self.head(f)                  # B, C

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "classes" in raw:
        return raw["classes"] if isinstance(raw["classes"], list) else [c["name"] for c in raw["classes"]]
    # allow {"car":0, ...}
    if isinstance(raw, dict):
        id_to_name = {v:k for k,v in raw.items()}
        return [id_to_name[i] for i in range(len(id_to_name))]
    raise ValueError("Unsupported class_map.json format")

def _build_model(num_classes: int) -> nn.Module:
    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    return model

def _try_load_checkpoint(model: nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    # support either {"model": state_dict} or a plain state_dict
    state_dict = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(state_dict, strict=False)

def load_predict_fn(ckpt_path: str, labels: List[str]) -> Callable:
    """
    Returns a function: predict(pil_crops: List[Image]) -> List[{top1, topk}]
    """
    # If running on HF Spaces and the checkpoint isn't present, allow pulling from Hub
    if not os.path.exists(ckpt_path):
        from huggingface_hub import hf_hub_download
        repo_id = os.getenv("HF_MODEL_ID")  # set this in Space Secrets if needed
        filename = os.getenv("HF_MODEL_FILE", "best.pth")
        if not repo_id:
            raise FileNotFoundError(f"{ckpt_path} not found and HF_MODEL_ID not set.")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

    device = get_device()
    model = _build_model(num_classes=len(labels)).to(device).eval()
    _try_load_checkpoint(model, ckpt_path)

    tf = transforms.Compose([
        Letterbox224(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    @torch.no_grad()
    def predict(pil_crops: List[Image.Image], top_k: int = 3):
        x = torch.stack([tf(im) for im in pil_crops]).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.topk(top_k, dim=1)
        out = []
        for c, i in zip(conf.cpu(), idx.cpu()):
            topk = [{"label": labels[j], "prob": float(c[k])} for k, j in enumerate(i)]
            out.append({"top1": topk[0], "topk": topk})
        return out

    return predict

def crop_from_bbox(image: Image.Image, b: dict) -> Image.Image:
    x1, y1, x2, y2 = [int(b[k]) for k in ("x1","y1","x2","y2")]
    if x2 <= x1 or y2 <= y1: raise ValueError("Invalid box dimensions")
    w, h = image.size
    x1 = max(0, min(x1, w-1)); x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h-1)); y2 = max(1, min(y2, h))
    return image.crop((x1, y1, x2, y2))
