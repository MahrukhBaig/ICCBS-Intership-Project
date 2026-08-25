# ============================================================
# ORAL AI — PRODUCTION STREAMLIT UI
# ============================================================

import os
import glob
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import open_clip
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Set page config for a premium wide layout
st.set_page_config(
    page_title="ORAL AI — Intelligent Oral Histopathology Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = ["mdoscc", "normal", "osmf", "pdoscc", "wdoscc"]

CLASS_DESCRIPTIONS = {
    "mdoscc": "Moderately Differentiated Oral Squamous Cell Carcinoma",
    "normal": "Normal Oral Tissue",
    "osmf": "Oral Submucous Fibrosis",
    "pdoscc": "Poorly Differentiated Oral Squamous Cell Carcinoma",
    "wdoscc": "Well Differentiated Oral Squamous Cell Carcinoma",
}

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = "https://huggingface.co/MahrukhB29/oral-ai-model/resolve/main/best_orchid_hybrid_stage2.pth"
MODEL_PATH = os.path.join(MODEL_DIR, "best_orchid_hybrid_stage2.pth")

# Automatic downloading from Hugging Face if the file is missing
if not os.path.exists(MODEL_PATH):
    import urllib.request
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    st.info("📥 Downloading model checkpoint from Hugging Face... Please wait. This will only happen once.")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    def download_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(read_so_far / total_size, 1.0)
            progress_bar.progress(percent)
            status_text.text(f"Downloaded {read_so_far / (1024*1024):.1f} MB of {total_size / (1024*1024):.1f} MB ({percent*100:.1f}%)")
        else:
            status_text.text(f"Downloaded {read_so_far / (1024*1024):.1f} MB")
            
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, download_progress)
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Model checkpoint downloaded successfully!")
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Failed to download model checkpoint: {str(e)}")
        st.stop()

# Fallback checkpoint discovery
if not os.path.exists(MODEL_PATH):
    pth_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pth")))
    if len(pth_files) == 1:
        MODEL_PATH = pth_files[0]

if not os.path.exists(MODEL_PATH):
    st.error("### ⚠️ Model Checkpoint Not Found")
    st.markdown(
        """
        The model checkpoint file was not found in the `models/` directory and download failed.
        """
    )
    st.stop()

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SwinMultiStageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0
        )
        self.patch_embed = swin.patch_embed
        self.layers = swin.layers
        self.norm = swin.norm

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers[0](x)
        s1 = x
        x = self.layers[1](x)
        s2 = x
        x = self.layers[2](x)
        s3 = x
        x = self.layers[3](x)
        s4 = x
        return s1, s2, s3, s4


class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, text_dim=512):
        super().__init__()
        self.visual_projection = nn.Linear(visual_dim, text_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(512)
        self.ffn = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )
        self.norm2 = nn.LayerNorm(512)

    def forward(self, visual_features, text_embeddings):
        if visual_features.dim() == 4:
            B, H, W, C = visual_features.shape
            visual_features = visual_features.reshape(B, H * W, C)
        visual_features = self.visual_projection(visual_features)
        B = visual_features.shape[0]
        text_features = text_embeddings.unsqueeze(0).expand(B, -1, -1)
        attention_output, _ = self.cross_attention(
            query=visual_features,
            key=text_features,
            value=text_features
        )
        x = self.norm1(visual_features + attention_output)
        x = self.norm2(x + self.ffn(x))
        return x


class MultiScaleSemanticFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = CrossAttentionFusion(visual_dim=96)
        self.stage2 = CrossAttentionFusion(visual_dim=192)
        self.stage3 = CrossAttentionFusion(visual_dim=384)
        self.stage4 = CrossAttentionFusion(visual_dim=768)

    def forward(self, s1, s2, s3, s4, text_embeddings):
        f1 = self.stage1(s1, text_embeddings)
        f2 = self.stage2(s2, text_embeddings)
        f3 = self.stage3(s3, text_embeddings)
        f4 = self.stage4(s4, text_embeddings)
        return f1, f2, f3, f4


class ResidualConvBlock(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.block(x)
        x = x + residual
        return self.activation(x)


class HybridDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder3 = ResidualConvBlock(channels=512)
        self.decoder2 = ResidualConvBlock(channels=512)
        self.decoder1 = ResidualConvBlock(channels=512)

    def forward(self, fused_features):
        x = self.decoder3(fused_features)
        x = self.decoder2(x)
        x = self.decoder1(x)
        return x


class SwinCLIPHybrid(nn.Module):
    def __init__(self, text_embeddings, num_classes=5):
        super().__init__()
        self.encoder = SwinMultiStageEncoder()
        self.text_embeddings = nn.Parameter(text_embeddings.clone(), requires_grad=False)
        self.semantic_fusion = MultiScaleSemanticFusion()
        self.decoder = HybridDecoder()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        _, _, _, f4 = self.semantic_fusion(s1, s2, s3, s4, self.text_embeddings)
        x = f4
        B, N, C = x.shape
        H = int(N ** 0.5)
        W = H
        x = x.transpose(1, 2)
        x = x.reshape(B, C, H, W)
        x = self.decoder(x)
        features = F.adaptive_avg_pool2d(x, 1)
        features = features.flatten(1)
        logits = self.classifier(features)
        return {"logits": logits, "features": features}


# ============================================================
# CLIP PROMPTS
# ============================================================

CLIP_PROMPTS = {
    "mdoscc": [
        "a histopathological image of moderately differentiated oral squamous cell carcinoma",
        "a microscopic image of moderately differentiated oral squamous cell carcinoma",
        "a histology slide showing moderately differentiated OSCC",
        "moderately differentiated oral squamous cell carcinoma under a microscope",
        "a pathology image of moderately differentiated oral cancer",
        "oral squamous cell carcinoma with moderate differentiation",
    ],
    "normal": [
        "a histopathological image of normal oral tissue",
        "a microscopic image of normal oral mucosa",
        "a histology image showing healthy oral tissue",
        "a pathology slide of normal oral mucosa",
        "a microscopic view of healthy oral epithelium",
        "normal oral tissue under a microscope",
    ],
    "osmf": [
        "a histopathological image of oral submucous fibrosis",
        "a microscopic image showing oral submucous fibrosis",
        "a histology slide of oral submucous fibrosis",
        "oral mucosa affected by oral submucous fibrosis",
        "a pathology image showing fibrosis of the oral mucosa",
        "a microscopic view of fibrotic oral tissue",
    ],
    "pdoscc": [
        "a histopathological image of poorly differentiated oral squamous cell carcinoma",
        "a microscopic image of poorly differentiated oral squamous cell carcinoma",
        "a histology slide showing poorly differentiated OSCC",
        "poorly differentiated oral squamous cell carcinoma under a microscope",
        "a pathology image of poorly differentiated oral cancer",
        "oral squamous cell carcinoma with poor differentiation",
    ],
    "wdoscc": [
        "a histopathological image of well differentiated oral squamous cell carcinoma",
        "a microscopic image of well differentiated oral squamous cell carcinoma",
        "a histology slide showing well differentiated OSCC",
        "a pathology image of well differentiated oral cancer",
        "a microscopic view of well differentiated oral tissue",
        "oral squamous cell carcinoma with high differentiation",
    ]
}


# ============================================================
# LOAD MODEL & CLIP EMBEDDINGS (Cached for Performance)
# ============================================================

@st.cache_resource
def load_models_and_embeddings():
    print(f"[OK] Loading models on: {device}")
    
    # Load CLIP semantic encoder
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # Extract text embeddings
    text_embeddings_list = []
    with torch.no_grad():
        for class_name in CLASS_NAMES:
            prompts = CLIP_PROMPTS[class_name]
            tokens = clip_tokenizer(prompts).to(device)
            embeddings = clip_model.encode_text(tokens)
            embeddings = F.normalize(embeddings, dim=-1)
            class_embedding = embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embedding, dim=0)
            text_embeddings_list.append(class_embedding)

    text_embeddings_tensor = torch.stack(text_embeddings_list)

    # Load hybrid classification model
    hybrid_model = SwinCLIPHybrid(
        text_embeddings=text_embeddings_tensor,
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    fixed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder.swin."):
            key = key.replace("encoder.swin.", "encoder.", 1)
        fixed_state_dict[key] = value

    hybrid_model.load_state_dict(fixed_state_dict, strict=True)
    hybrid_model.eval()

    # Wrap model for Grad-CAM
    class HybridCAMWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)["logits"]

    cam_model = HybridCAMWrapper(hybrid_model).to(device)
    cam_model.eval()
    
    target_layer = hybrid_model.decoder.decoder1.block[3]

    print(f"[OK] ORAL AI model loaded: {os.path.basename(MODEL_PATH)}")
    return hybrid_model, cam_model, target_layer


hybrid_model, cam_model, target_layer = load_models_and_embeddings()


# ============================================================
# IMAGE PREPROCESSING & PREDICTION
# ============================================================

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CAM_MEAN = np.array([0.485, 0.456, 0.406])
CAM_STD = np.array([0.229, 0.224, 0.225])


def predict_orchid(image):
    image = image.convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    hybrid_model.eval()
    with torch.no_grad():
        output = hybrid_model(input_tensor)
        logits = output["logits"]
        probabilities_tensor = torch.softmax(logits, dim=1)[0]

    probabilities = probabilities_tensor.detach().cpu().numpy()
    prediction_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[prediction_index]
    confidence = float(probabilities[prediction_index])
    
    class_probabilities = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)
    }

    return predicted_class, confidence, class_probabilities, prediction_index


def generate_gradcam(image, prediction_index):
    image = image.convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(prediction_index)]

    with GradCAM(model=cam_model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_image = input_tensor[0].detach().cpu().numpy()
    rgb_image = np.transpose(rgb_image, (1, 2, 0))
    rgb_image = rgb_image * CAM_STD + CAM_MEAN
    rgb_image = np.clip(rgb_image, 0, 1)

    heatmap = show_cam_on_image(np.zeros_like(rgb_image), grayscale_cam, use_rgb=True)
    overlay = show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)

    original_array = (rgb_image * 255).astype(np.uint8)
    heatmap_array = heatmap.astype(np.uint8)
    overlay_array = overlay.astype(np.uint8)

    return original_array, heatmap_array, overlay_array


# ============================================================
# PREMIUM CUSTOM CSS INJECTION
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global styles override */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #f4f7fb !important;
        color: #1e293b !important;
    }
    
    /* Header/Navbar Area */
    .navbar {
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 25px;
        border-radius: 20px;
        background: linear-gradient(135deg, #172033, #263b5c);
        box-shadow: 0 12px 35px rgba(15,23,42,0.18);
        margin-bottom: 30px;
    }
    .nav-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .logo-box {
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 15px;
        font-size: 23px;
        background: linear-gradient(135deg, #2563eb, #38bdf8);
    }
    .logo-title {
        color: white;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 1px;
        line-height: 1.2;
    }
    .logo-subtitle {
        color: #a5b4c8;
        font-size: 11px;
    }
    .system-status {
        padding: 10px 16px;
        border-radius: 14px;
        color: #cbd5e1;
        font-size: 11px;
        background: rgba(255, 255, 255, 0.08);
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 7px;
        animation: pulse 2s infinite;
    }

    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 30px 10px 40px 10px;
    }
    .hero-title {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(135deg, #1e293b 30%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-description {
        max-width: 800px;
        margin: 12px auto 0 auto;
        line-height: 1.8;
        font-size: 15px;
        color: #64748b;
    }

    /* Prediction Card */
    .prediction-card {
        min-height: 380px;
        padding: 40px 35px;
        border-radius: 22px;
        text-align: center;
        background: linear-gradient(145deg, #16213a, #202f50);
        box-shadow: 0 15px 40px rgba(15,23,42,0.20);
        color: #f8fafc !important;
        margin-bottom: 20px;
    }
    .prediction-label {
        color: #a5b4c8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
    }
    .diagnosis-name {
        margin-top: 20px;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #7dd3fc;
    }
    .full-diagnosis {
        max-width: 520px;
        margin: 12px auto 0;
        font-size: 14px;
        line-height: 1.6;
        color: #e2e8f0;
    }
    .confidence-box {
        margin-top: 25px;
        padding: 18px;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.55);
    }
    .confidence-label {
        color: #94a3b8;
        font-size: 12px;
    }
    .confidence-value {
        margin-top: 7px;
        font-size: 38px;
        font-weight: 800;
    }
    .high-confidence { color: #4ade80 !important; }
    .moderate-confidence { color: #facc15 !important; }
    .low-confidence { color: #fb7185 !important; }

    .confidence-status {
        margin-top: 18px;
        padding: 16px;
        border-radius: 14px;
        text-align: left;
        background: rgba(255, 255, 255, 0.05);
    }
    .status-title {
        font-size: 13px;
        font-weight: 700;
    }
    .status-message {
        margin-top: 7px;
        color: #cbd5e1;
        font-size: 12px;
        line-height: 1.6;
    }

    /* Info card list styled */
    .probability-card {
        background: white;
        border: 1px solid #dbe3ef;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.05);
        margin-bottom: 20px;
    }
    .prob-header {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .prob-bar-container {
        margin-bottom: 14px;
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 600;
        color: #475569;
        margin-bottom: 5px;
    }
    .prob-bar-bg {
        width: 100%;
        height: 10px;
        background: #eef2f7;
        border-radius: 5px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
        border-radius: 5px;
    }

    /* Disclaimer box styling */
    .disclaimer {
        padding: 20px 24px;
        border-radius: 15px;
        background: #fffaf0;
        border-left: 5px solid #f59e0b;
        color: #92400e;
        line-height: 1.6;
        font-size: 13px;
        margin-top: 30px;
    }

    /* XAI Section style */
    .xai-section-title {
        text-align: center;
        margin-top: 50px;
        margin-bottom: 30px;
    }
    .xai-main-title {
        font-size: 26px;
        font-weight: 800;
        color: #1e293b;
    }
    .xai-desc {
        color: #64748b;
        font-size: 14px;
        margin-top: 8px;
    }

    /* XAI Display Layout */
    .xai-card-title {
        font-size: 14px;
        font-weight: 700;
        color: #1e293b;
        background: #eef4ff;
        border: 1px solid #dbe3ef;
        border-bottom: none;
        padding: 10px;
        text-align: center;
        border-top-left-radius: 15px;
        border-top-right-radius: 15px;
    }
    .xai-image-box {
        border: 1px solid #dbe3ef;
        border-radius: 0 0 15px 15px;
        background: white;
        padding: 10px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.04);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .xai-placeholder-box {
        height: 320px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: #f8fafc;
        color: #94a3b8;
        font-size: 14px;
        font-weight: 500;
        border: 1px dashed #cbd5e1;
        border-radius: 15px;
        text-align: center;
    }

    /* Section header layout */
    .sec-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 18px;
    }
    .sec-icon {
        width: 44px;
        height: 44px;
        background: #eef4ff;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .sec-title-text {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
    }
    .sec-subtitle-text {
        font-size: 11px;
        color: #64748b;
    }

    /* Model Information Block */
    .model-info-card {
        background: white;
        border: 1px solid #dbe3ef;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.05);
        margin-top: 55px;
        margin-bottom: 40px;
    }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .info-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
    }
    .info-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    .info-value {
        font-size: 14px;
        font-weight: 600;
        color: #1e293b;
        margin-top: 4px;
    }
    /* Primary button custom styles - Softer Blue */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.15) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.25) !important;
    }
    /* File uploader custom styles */
    [data-testid="stFileUploader"] > section {
        background-color: white !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"] > section:hover,
    [data-testid="stFileUploader"] > section:focus,
    [data-testid="stFileUploader"] > section:active,
    [data-testid="stFileUploader"] > section:focus-within {
        border-color: #cbd5e1 !important;
        background-color: #f8fafc !important;
    }
    [data-testid="stFileUploader"] button, 
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
        background-color: transparent !important;
        background: transparent !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover, 
    [data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }
    [data-testid="stFileUploader"] section div {
        color: #64748b !important;
    }
    
    /* Style the uploaded file chip to be white instead of black — broad override */
    [data-testid="stFileUploader"] [data-testid="stUploadedFile"],
    [data-testid="stUploadedFile"],
    .stUploadedFile,
    [class*="uploadedFile"],
    [class*="UploadedFile"],
    div[data-testid="stFileUploader"] li,
    div[data-testid="stFileUploader"] ul li {
        background-color: white !important;
        background: white !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #1e293b !important;
    }
    [data-testid="stUploadedFile"] *,
    [class*="uploadedFile"] *,
    [class*="UploadedFile"] * {
        background-color: transparent !important;
        background: transparent !important;
        color: #1e293b !important;
    }
    /* Override any small dark pill/badge inside the uploader */
    [data-testid="stFileUploader"] section ul,
    [data-testid="stFileUploader"] section li {
        background-color: white !important;
        background: white !important;
        list-style: none !important;
    }
    [data-testid="stFileUploader"] section li > div {
        background-color: white !important;
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        color: #1e293b !important;
    }
    [data-testid="stFileUploader"] section li > div * {
        background-color: transparent !important;
        color: #1e293b !important;
    }
    [data-testid="stFileUploader"] section li button {
        background-color: transparent !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        color: #64748b !important;
    }
    [data-testid="stFileUploader"] section li button:hover {
        background-color: #f1f5f9 !important;
        color: #ef4444 !important;
    }
    
    /* Uploaded Image Custom Sizing */
    [data-testid="stImage"] img {
        border-radius: 16px !important;
        max-height: 400px !important;
        object-fit: cover !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER / NAVBAR
# ============================================================

st.markdown(
    """
    <div class="navbar">
        <div class="nav-left">
            <div class="logo-box">
                <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <!-- Outer cell membrane -->
                    <ellipse cx="16" cy="16" rx="13" ry="11" stroke="white" stroke-width="2" fill="none" opacity="0.9"/>
                    <!-- Cell membrane texture bumps -->
                    <path d="M3.5 13 Q5 10 3.5 7" stroke="white" stroke-width="1.2" fill="none" opacity="0.5"/>
                    <path d="M28.5 13 Q27 10 28.5 7" stroke="white" stroke-width="1.2" fill="none" opacity="0.5"/>
                    <!-- Nucleus -->
                    <ellipse cx="16" cy="16" rx="6" ry="5" stroke="white" stroke-width="1.8" fill="rgba(255,255,255,0.15)"/>
                    <!-- Nucleolus (solid dot inside nucleus) -->
                    <circle cx="16" cy="16" r="2" fill="white" opacity="0.9"/>
                    <!-- Organelle dots scattered in cytoplasm -->
                    <circle cx="7" cy="13" r="1.2" fill="white" opacity="0.6"/>
                    <circle cx="9" cy="20" r="0.9" fill="white" opacity="0.5"/>
                    <circle cx="24" cy="19" r="1.2" fill="white" opacity="0.6"/>
                    <circle cx="23" cy="11" r="0.8" fill="white" opacity="0.4"/>
                    <circle cx="12" cy="24" r="0.9" fill="white" opacity="0.5"/>
                    <circle cx="20" cy="23" r="0.7" fill="white" opacity="0.4"/>
                </svg>
            </div>
            <div>
                <div class="logo-title">ORAL AI</div>
                <div class="logo-subtitle">Intelligent Oral Histopathology Analysis Platform</div>
            </div>
        </div>
        <div class="system-status">
            <span class="status-dot"></span>
            AI System Online &nbsp; | &nbsp; Model v1.0
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">Analyze Oral Histopathology with AI</div>
        <div class="hero-description">
            Upload a histopathology image to receive an AI-assisted classification, confidence analysis, 
            class probability distribution, and visual explanations through Grad-CAM.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MAIN APPLICATION ROWS (Columns)
# ============================================================

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(
        """
        <div class="sec-header" style="border-bottom: 2px solid #eef4ff; padding-bottom: 8px; margin-bottom: 20px;">
            <div>
                <div class="sec-title-text" style="font-size: 20px;">Histopathology Image</div>
                <div class="sec-subtitle-text">Upload an oral tissue sample</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an oral histopathology image file (PNG, JPG, JPEG)...",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    # JS via components.v1.html — can access window.parent.document (same-origin)
    components.html(
        """
        <script>
        (function() {
            function fixChip() {
                try {
                    var doc = window.parent.document;
                    var targets = doc.querySelectorAll(
                        '[data-testid="stFileUploader"] section li, ' +
                        '[data-testid="stFileUploader"] section li > div, ' +
                        '[data-testid="stFileUploader"] section ul li'
                    );
                    targets.forEach(function(el) {
                        el.style.setProperty('background', 'white', 'important');
                        el.style.setProperty('background-color', 'white', 'important');
                        el.style.setProperty('color', '#1e293b', 'important');
                        el.style.setProperty('border', '1px solid #e2e8f0', 'important');
                        el.style.setProperty('border-radius', '10px', 'important');
                    });
                    // Nuke any dark-background children
                    doc.querySelectorAll('[data-testid="stFileUploader"] section li *').forEach(function(el) {
                        var bg = window.parent.getComputedStyle(el).backgroundColor;
                        var m = bg.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
                        if (m && parseInt(m[1]) < 80 && parseInt(m[2]) < 80 && parseInt(m[3]) < 80) {
                            el.style.setProperty('background', 'transparent', 'important');
                            el.style.setProperty('background-color', 'transparent', 'important');
                            el.style.setProperty('color', '#1e293b', 'important');
                        }
                    });
                } catch(e) {}
            }
            fixChip();
            setTimeout(fixChip, 300);
            setTimeout(fixChip, 800);
            try {
                new MutationObserver(fixChip).observe(
                    window.parent.document.body,
                    { childList: true, subtree: true, attributes: true }
                );
            } catch(e) {}
        })();
        </script>
        """,
        height=0,
    )

    input_image = None
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            st.image(input_image, caption="Uploaded Histopathology Slide", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

    analyze_clicked = st.button(
        "Analyze Image with ORAL AI",
        use_container_width=True,
        type="primary",
        disabled=(uploaded_file is None),
    )

with col2:
    st.markdown(
        """
        <div class="sec-header" style="border-bottom: 2px solid #eef4ff; padding-bottom: 8px; margin-bottom: 20px;">
            <div>
                <div class="sec-title-text" style="font-size: 20px;">AI Classification Result</div>
                <div class="sec-subtitle-text">AI-assisted prediction and confidence analysis</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # We store session state variables to hold results between interactions
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
        st.session_state.predicted_class = None
        st.session_state.confidence_pct = 0.0
        st.session_state.probabilities = {}
        st.session_state.orig_img = None
        st.session_state.heat_img = None
        st.session_state.overlay_img = None

    # Handle Analysis logic
    if analyze_clicked and input_image is not None:
        with st.spinner("Analyzing tissue slide and generating explanations..."):
            try:
                # Prediction
                pred_class, conf, probs, idx = predict_orchid(input_image)
                
                # Grad-CAM
                orig, heat, overlay = generate_gradcam(input_image, idx)
                
                # Cache results in Session State
                st.session_state.prediction_made = True
                st.session_state.predicted_class = pred_class
                st.session_state.confidence_pct = conf * 100
                st.session_state.probabilities = probs
                st.session_state.orig_img = orig
                st.session_state.heat_img = heat
                st.session_state.overlay_img = overlay
            except Exception as e:
                st.error("An error occurred during classification.")
                st.exception(e)

    # Render Prediction Cards
    if st.session_state.prediction_made:
        # Confidence logic details
        pct = st.session_state.confidence_pct
        if pct >= 80:
            status_title = "High Confidence"
            status_msg = "The model shows a strong preference for this classification."
            status_class = "high-confidence"
        elif pct >= 50:
            status_title = "Moderate Confidence"
            status_msg = "The prediction shows moderate separation from the remaining classes."
            status_class = "moderate-confidence"
        else:
            status_title = "Low Confidence"
            status_msg = "Class probabilities are closely distributed. Consider expert pathological review."
            status_class = "low-confidence"

        display_class = st.session_state.predicted_class.upper()
        full_diagnosis = CLASS_DESCRIPTIONS[st.session_state.predicted_class]

        prediction_card_html = f"""
        <div class="prediction-card">
            <div class="prediction-label">AI PREDICTION</div>
            <div class="diagnosis-name">{display_class}</div>
            <div class="full-diagnosis">{full_diagnosis}</div>
            <div class="confidence-box">
                <div class="confidence-label">Prediction Confidence</div>
                <div class="confidence-value {status_class}">{pct:.2f}%</div>
            </div>
            <div class="confidence-status">
                <div class="status-title {status_class}">● {status_title}</div>
                <div class="status-message">{status_msg}</div>
            </div>
        </div>
        """
        st.html(prediction_card_html)

        # Render Probability bars card
        prob_bars_html = f"""
        <div class="probability-card">
            <div class="prob-header" style="font-size: 15px; font-weight: 700; color: #172033; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px; margin-bottom: 15px;">Class Probability Distribution</div>
        """
        class_colors = {
            "mdoscc": "linear-gradient(90deg, #ef4444, #fca5a5)",
            "normal": "linear-gradient(90deg, #22c55e, #86efac)",
            "osmf": "linear-gradient(90deg, #a855f7, #d8b4fe)",
            "pdoscc": "linear-gradient(90deg, #f97316, #ffedd5)",
            "wdoscc": "linear-gradient(90deg, #3b82f6, #93c5fd)",
        }
        for cls_name in CLASS_NAMES:
            cls_pct = st.session_state.probabilities.get(cls_name, 0.0) * 100
            cls_desc = CLASS_DESCRIPTIONS[cls_name]
            bar_color = class_colors.get(cls_name, "linear-gradient(90deg, #2563eb, #38bdf8)")
            prob_bars_html += f"""
            <div class="prob-bar-container">
                <div class="prob-labels">
                    <span>{cls_name.upper()} ({cls_desc})</span>
                    <span>{cls_pct:.2f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {cls_pct}%; background: {bar_color};"></div>
                </div>
            </div>
            """
        prob_bars_html += "</div>"
        st.html(prob_bars_html)

    else:
        # Default ready card
        st.markdown(
            """
            <div class="prediction-card">
                <div class="prediction-label">AI PREDICTION</div>
                <div class="diagnosis-name" style="margin-top: 35px;">READY FOR ANALYSIS</div>
                <div class="full-diagnosis" style="margin-top: 20px;">
                    Upload a histopathology image on the left and click <b>Analyze Image with ORAL AI</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# EXPLAINABLE AI ANALYSIS Visual Explanations
# ============================================================

st.markdown(
    """
    <div class="xai-section-title">
        <div class="xai-main-title">Explainable AI Analysis</div>
        <div class="xai-desc">Explore which regions of the histopathology image contributed most strongly to the AI model's prediction using Grad-CAM.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

xai_col1, xai_col2, xai_col3 = st.columns(3)

if st.session_state.prediction_made:
    with xai_col1:
        st.markdown('<div class="xai-card-title">Original Image</div>', unsafe_allow_html=True)
        st.image(st.session_state.orig_img, use_container_width=True)

    with xai_col2:
        st.markdown('<div class="xai-card-title">Grad-CAM Attention</div>', unsafe_allow_html=True)
        st.image(st.session_state.heat_img, use_container_width=True)

    with xai_col3:
        st.markdown('<div class="xai-card-title">Grad-CAM Overlay</div>', unsafe_allow_html=True)
        st.image(st.session_state.overlay_img, use_container_width=True)
else:
    # Render Placeholder grids
    with xai_col1:
        st.markdown('<div class="xai-placeholder-box">Upload and analyze image to view original slide</div>', unsafe_allow_html=True)
    with xai_col2:
        st.markdown('<div class="xai-placeholder-box">Upload and analyze image to view Grad-CAM attention</div>', unsafe_allow_html=True)
    with xai_col3:
        st.markdown('<div class="xai-placeholder-box">Upload and analyze image to view overlay visualization</div>', unsafe_allow_html=True)

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
         <b>Research and Educational Use Only</b><br>
        This AI system is intended for research and educational purposes. It is not designed to replace professional pathological diagnosis or clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL INFORMATION
# ============================================================

st.html(
    """
    <div class="model-info-card">
        <h3 style="margin-top:0; color:#172033; font-weight:800; font-size:22px; margin-bottom: 6px;">🧬 Model Architecture & Statistics</h3>
        <p style="color:#64748b; font-size:14px; margin-bottom: 24px;">ORAL AI is a hybrid multimodal classification model guided by semantic prompts.</p>
        
        <div style="display: flex; gap: 40px; flex-wrap: wrap;">
            <!-- Left Side: Architecture Details -->
            <div style="flex: 2; min-width: 300px;">
                <h4 style="color:#172033; font-size:15px; font-weight:700; margin-bottom: 12px; border-bottom: 2px solid #eef4ff; padding-bottom: 6px;">Component Architecture</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 10px 0; font-size: 13px; font-weight: 700; color: #64748b; width: 40%;">Visual Backbone</td>
                        <td style="padding: 10px 0; font-size: 13px; font-weight: 600; color: #1e293b;">Swin Transformer (Tiny)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 10px 0; font-size: 13px; font-weight: 700; color: #64748b;">Semantic Encoder</td>
                        <td style="padding: 10px 0; font-size: 13px; font-weight: 600; color: #1e293b;">CLIP Text Embeddings (ViT-B-16)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;">
                        <td style="padding: 10px 0; font-size: 13px; font-weight: 700; color: #64748b;">Fusion Strategy</td>
                        <td style="padding: 10px 0; font-size: 13px; font-weight: 600; color: #1e293b;">Multi-Stage Cross-Attention</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; font-size: 13px; font-weight: 700; color: #64748b;">Decoder Backbone</td>
                        <td style="padding: 10px 0; font-size: 13px; font-weight: 600; color: #1e293b;">U-Net Inspired Decoder</td>
                    </tr>
                </table>
            </div>
            
            <!-- Right Side: Key Performance Metrics -->
            <div style="flex: 1; min-width: 200px;">
                <h4 style="color:#172033; font-size:15px; font-weight:700; margin-bottom: 12px; border-bottom: 2px solid #eef4ff; padding-bottom: 6px;">Performance Metrics</h4>
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 12px; font-weight: 700; color: #166534;">Test Accuracy</span>
                        <span style="font-size: 16px; font-weight: 800; color: #15803d;">95.73%</span>
                    </div>
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 12px; font-weight: 700; color: #166534;">Macro F1-Score</span>
                        <span style="font-size: 16px; font-weight: 800; color: #15803d;">96.20%</span>
                    </div>
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 12px; font-weight: 700; color: #166534;">Macro ROC-AUC</span>
                        <span style="font-size: 16px; font-weight: 800; color: #15803d;">0.9976</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div style="margin-top:35px; border-top: 1px solid #f1f5f9; padding-top: 25px;">
            <div style="font-weight:700; font-size:14px; color:#172033; margin-bottom:14px;">ORCHID Dataset Classification Classes:</div>
            <ul style="list-style-type: none; padding-left: 0; margin: 0; display: flex; flex-direction: column; gap: 10px;">
                <li style="display: flex; align-items: center; gap: 12px; font-size: 13px; color: #475569;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #ef4444;"></span>
                    <strong>MDOSCC</strong> — Moderately Differentiated Oral Squamous Cell Carcinoma
                </li>
                <li style="display: flex; align-items: center; gap: 12px; font-size: 13px; color: #475569;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #22c55e;"></span>
                    <strong>NORMAL</strong> — Normal Oral Tissue
                </li>
                <li style="display: flex; align-items: center; gap: 12px; font-size: 13px; color: #475569;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #a855f7;"></span>
                    <strong>OSMF</strong> — Oral Submucous Fibrosis
                </li>
                <li style="display: flex; align-items: center; gap: 12px; font-size: 13px; color: #475569;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #f97316;"></span>
                    <strong>PDOSCC</strong> — Poorly Differentiated Oral Squamous Cell Carcinoma
                </li>
                <li style="display: flex; align-items: center; gap: 12px; font-size: 13px; color: #475569;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #3b82f6;"></span>
                    <strong>WDOSCC</strong> — Well Differentiated Oral Squamous Cell Carcinoma
                </li>
            </ul>
        </div>
    </div>
    """
)