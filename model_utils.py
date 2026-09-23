"""
model_utils.py — DenseNet121 (CC + MLO) BI-RADS classifier.

Averages the softmax probabilities from a CC-view DenseNet121 and an
MLO-view DenseNet121 (each with CBAM attention) to produce a final
prediction. Both app.py (Gradio) and streamlit_app.py (Streamlit) import
from this file.
"""

import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms, models
from huggingface_hub import hf_hub_download

# ==========================================
# 0) Configuration — EDIT THESE
# ==========================================
USE_LOCAL_CHECKPOINTS = False
CHECKPOINT_REPO_ID = "min331/birads-densenet-weights"
CHECKPOINT_REPO_REVISION = "main"
HF_TOKEN = os.environ.get("HF_TOKEN")

class_names = ["BI-RADS 1", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]
num_classes = len(class_names)

# Short, plain-language descriptions shown next to each class in the UI.
# NOTE: these are general/standard BI-RADS descriptions for reference only —
# review and adjust wording with your supervisor/clinical advisor before
# showing this to any real end users.
class_descriptions = {
    "BI-RADS 1": "Negative — no significant abnormality found.",
    "BI-RADS 3": "Probably benign — likely non-cancerous; short-interval follow-up is often recommended.",
    "BI-RADS 4": "Suspicious abnormality — biopsy may be considered.",
    "BI-RADS 5": "Highly suggestive of malignancy — biopsy strongly recommended.",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_FILES = {
    "densenet_cc":  "densenet121_noaugment2cc_newaveraging_cbamattention_fine_tuned_model.pth",
    "densenet_mlo": "densenet121_noaugment2mlo_newaveraging_cbamattention_fine_tuned_model.pth",
}

# ==========================================
# 1) Preprocessing (must match training)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ==========================================
# 2) Shared CBAM module (identical to training script)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_channels = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_map = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn_map


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ==========================================
# 3) DenseNet121 + CBAM (exact match to training script)
# ==========================================
class DenseNetWithCBAM(nn.Module):
    def __init__(self, num_classes=4, dropout=0.8, weights=None, reduction_ratio=16, kernel_size=7):
        super().__init__()
        base = models.densenet121(weights=weights)
        self.features = base.features
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(in_channels=base.classifier.in_features, reduction_ratio=reduction_ratio, kernel_size=kernel_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(base.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.relu(feat)
        feat = self.cbam(feat)
        out = self.avgpool(feat)
        out = torch.flatten(out, 1)
        return self.fc(out)

# ==========================================
# 4) Checkpoint resolution + model loading
# ==========================================
def resolve_checkpoint_path(key):
    filename = CHECKPOINT_FILES[key]
    if USE_LOCAL_CHECKPOINTS:
        return os.path.join("checkpoints", filename)
    return hf_hub_download(
        repo_id=CHECKPOINT_REPO_ID,
        filename=filename,
        revision=CHECKPOINT_REPO_REVISION,
        token=HF_TOKEN,
    )


def load_model(checkpoint_key, **kwargs):
    net = DenseNetWithCBAM(num_classes=num_classes, **kwargs)
    ckpt_path = resolve_checkpoint_path(checkpoint_key)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net = net.to(device).eval()
    return net


_models_cache = None


def get_models():
    """Loads both checkpoints once and caches them for the process lifetime."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    densenet_cc = load_model("densenet_cc", dropout=0.8, weights=None)
    densenet_mlo = load_model("densenet_mlo", dropout=0.8, weights=None)

    _models_cache = (densenet_cc, densenet_mlo)
    return _models_cache

# ==========================================
# 5) Inference
# ==========================================
def run_classification(cc_image, mlo_image):
    """
    cc_image, mlo_image: PIL.Image (RGB or convertible to RGB)

    Returns:
        final_probs: dict[class_name -> float] — CC+MLO averaged probabilities
        cc_probs:    dict[class_name -> float] — CC view alone
        mlo_probs:   dict[class_name -> float] — MLO view alone
    """
    densenet_cc, densenet_mlo = get_models()

    x_cc = transform(cc_image.convert("RGB")).unsqueeze(0).to(device)
    x_mlo = transform(mlo_image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        prob_cc = F.softmax(densenet_cc(x_cc), dim=1)
        prob_mlo = F.softmax(densenet_mlo(x_mlo), dim=1)
        avg_probs = (prob_cc + prob_mlo) / 2.0

    cc_probs_np = prob_cc[0].cpu().numpy()
    mlo_probs_np = prob_mlo[0].cpu().numpy()
    avg_probs_np = avg_probs[0].cpu().numpy()

    cc_probs = {class_names[i]: float(cc_probs_np[i]) for i in range(num_classes)}
    mlo_probs = {class_names[i]: float(mlo_probs_np[i]) for i in range(num_classes)}
    final_probs = {class_names[i]: float(avg_probs_np[i]) for i in range(num_classes)}

    return final_probs, cc_probs, mlo_probs


# Kept as an alias so any existing code importing run_ensemble still works.
run_ensemble = run_classification
