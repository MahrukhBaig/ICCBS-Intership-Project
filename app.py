# ============================================================
# ORAL AI — PRODUCTION GRADIO UI
# ============================================================

import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import open_clip
import gradio as gr

from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLASS_NAMES = [
    "mdoscc",
    "normal",
    "osmf",
    "pdoscc",
    "wdoscc"
]

CLASS_DESCRIPTIONS = {
    "mdoscc":
        "Moderately Differentiated Oral Squamous Cell Carcinoma",

    "normal":
        "Normal Oral Tissue",

    "osmf":
        "Oral Submucous Fibrosis",

    "pdoscc":
        "Poorly Differentiated Oral Squamous Cell Carcinoma",

    "wdoscc":
        "Well Differentiated Oral Squamous Cell Carcinoma",
}

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_orchid_hybrid_stage2.pth"
)

# Fallback checkpoint discovery
if not os.path.exists(MODEL_PATH):

    pth_files = sorted(
        glob.glob(
            os.path.join(
                MODEL_DIR,
                "*.pth"
            )
        )
    )

    if len(pth_files) == 1:
        MODEL_PATH = pth_files[0]


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"No .pth checkpoint found in: {MODEL_DIR}\n"
        "Place best_orchid_hybrid_stage2.pth inside the models folder."
    )


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

    def __init__(
        self,
        visual_dim,
        text_dim=512
    ):

        super().__init__()

        self.visual_projection = nn.Linear(
            visual_dim,
            text_dim
        )

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


    def forward(
        self,
        visual_features,
        text_embeddings
    ):

        if visual_features.dim() == 4:

            B, H, W, C = visual_features.shape

            visual_features = visual_features.reshape(
                B,
                H * W,
                C
            )

        visual_features = self.visual_projection(
            visual_features
        )

        B = visual_features.shape[0]

        text_features = (
            text_embeddings
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        attention_output, _ = self.cross_attention(
            query=visual_features,
            key=text_features,
            value=text_features
        )

        x = self.norm1(
            visual_features +
            attention_output
        )

        x = self.norm2(
            x +
            self.ffn(x)
        )

        return x


class MultiScaleSemanticFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = CrossAttentionFusion(
            visual_dim=96
        )

        self.stage2 = CrossAttentionFusion(
            visual_dim=192
        )

        self.stage3 = CrossAttentionFusion(
            visual_dim=384
        )

        self.stage4 = CrossAttentionFusion(
            visual_dim=768
        )


    def forward(
        self,
        s1,
        s2,
        s3,
        s4,
        text_embeddings
    ):

        f1 = self.stage1(
            s1,
            text_embeddings
        )

        f2 = self.stage2(
            s2,
            text_embeddings
        )

        f3 = self.stage3(
            s3,
            text_embeddings
        )

        f4 = self.stage4(
            s4,
            text_embeddings
        )

        return f1, f2, f3, f4


class ResidualConvBlock(nn.Module):

    def __init__(
        self,
        channels=512
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            )
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

        self.decoder3 = ResidualConvBlock(
            channels=512
        )

        self.decoder2 = ResidualConvBlock(
            channels=512
        )

        self.decoder1 = ResidualConvBlock(
            channels=512
        )


    def forward(self, fused_features):

        x = self.decoder3(
            fused_features
        )

        x = self.decoder2(x)

        x = self.decoder1(x)

        return x


class SwinCLIPHybrid(nn.Module):

    def __init__(
        self,
        text_embeddings,
        num_classes=5
    ):

        super().__init__()

        self.encoder = SwinMultiStageEncoder()

        self.text_embeddings = nn.Parameter(
            text_embeddings.clone(),
            requires_grad=False
        )

        self.semantic_fusion = (
            MultiScaleSemanticFusion()
        )

        self.decoder = HybridDecoder()

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(self, x):

        s1, s2, s3, s4 = (
            self.encoder(x)
        )

        _, _, _, f4 = (
            self.semantic_fusion(
                s1,
                s2,
                s3,
                s4,
                self.text_embeddings
            )
        )

        x = f4

        B, N, C = x.shape

        H = int(
            N ** 0.5
        )

        W = H

        x = x.transpose(
            1,
            2
        )

        x = x.reshape(
            B,
            C,
            H,
            W
        )

        x = self.decoder(x)

        features = F.adaptive_avg_pool2d(
            x,
            1
        )

        features = features.flatten(
            1
        )

        logits = self.classifier(
            features
        )

        return {
            "logits": logits,
            "features": features
        }


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
# LOAD CLIP
# ============================================================

print(
    f"Running on: {device}"
)

print(
    "Loading CLIP semantic encoder..."
)

clip_model, _, _ = (
    open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        device=device
    )
)

clip_tokenizer = (
    open_clip.get_tokenizer(
        "ViT-B-16"
    )
)

clip_model.eval()

for param in clip_model.parameters():

    param.requires_grad = False


text_embeddings_list = []


with torch.no_grad():

    for class_name in CLASS_NAMES:

        prompts = CLIP_PROMPTS[
            class_name
        ]

        tokens = (
            clip_tokenizer(
                prompts
            ).to(device)
        )

        embeddings = (
            clip_model.encode_text(
                tokens
            )
        )

        embeddings = F.normalize(
            embeddings,
            dim=-1
        )

        class_embedding = (
            embeddings.mean(
                dim=0
            )
        )

        class_embedding = F.normalize(
            class_embedding,
            dim=0
        )

        text_embeddings_list.append(
            class_embedding
        )


text_embeddings = torch.stack(
    text_embeddings_list
)


# ============================================================
# LOAD HYBRID MODEL
# ============================================================

hybrid_model = SwinCLIPHybrid(
    text_embeddings=text_embeddings,
    num_classes=NUM_CLASSES
).to(device)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

state_dict = checkpoint.get(
    "model_state_dict",
    checkpoint
)


fixed_state_dict = {}


for key, value in state_dict.items():

    if key.startswith(
        "encoder.swin."
    ):

        key = key.replace(
            "encoder.swin.",
            "encoder.",
            1
        )

    fixed_state_dict[key] = value


hybrid_model.load_state_dict(
    fixed_state_dict,
    strict=True
)

hybrid_model.eval()


print(
    f"✓ ORAL AI model loaded: "
    f"{os.path.basename(MODEL_PATH)}"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

inference_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


CAM_MEAN = np.array([
    0.485,
    0.456,
    0.406
])

CAM_STD = np.array([
    0.229,
    0.224,
    0.225
])


# ============================================================
# PREDICTION
# ============================================================

def predict_orchid(image):

    image = image.convert(
        "RGB"
    )

    input_tensor = (
        inference_transform(
            image
        )
        .unsqueeze(0)
        .to(device)
    )

    hybrid_model.eval()


    with torch.no_grad():

        output = hybrid_model(
            input_tensor
        )

        logits = output[
            "logits"
        ]

        probabilities_tensor = (
            torch.softmax(
                logits,
                dim=1
            )[0]
        )


    probabilities = (
        probabilities_tensor
        .detach()
        .cpu()
        .numpy()
    )


    prediction_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_class = (
        CLASS_NAMES[
            prediction_index
        ]
    )

    confidence = float(
        probabilities[
            prediction_index
        ]
    )


    class_probabilities = {

        CLASS_NAMES[i]:
        float(
            probabilities[i]
        )

        for i in range(
            NUM_CLASSES
        )
    }


    return (
        predicted_class,
        confidence,
        class_probabilities,
        prediction_index
    )


# ============================================================
# GRAD-CAM WRAPPER
# ============================================================

class HybridCAMWrapper(
    nn.Module
):

    def __init__(
        self,
        model
    ):

        super().__init__()

        self.model = model


    def forward(self, x):

        return self.model(
            x
        )["logits"]


cam_model = HybridCAMWrapper(
    hybrid_model
).to(device)

cam_model.eval()


target_layer = (
    hybrid_model
    .decoder
    .decoder1
    .block[3]
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image,
    prediction_index
):

    image = image.convert(
        "RGB"
    )

    input_tensor = (
        inference_transform(
            image
        )
        .unsqueeze(0)
        .to(device)
    )


    targets = [
        ClassifierOutputTarget(
            prediction_index
        )
    ]


    with GradCAM(
        model=cam_model,
        target_layers=[
            target_layer
        ]
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]


    rgb_image = (
        input_tensor[0]
        .detach()
        .cpu()
        .numpy()
    )

    rgb_image = np.transpose(
        rgb_image,
        (1, 2, 0)
    )


    rgb_image = (
        rgb_image * CAM_STD
        + CAM_MEAN
    )

    rgb_image = np.clip(
        rgb_image,
        0,
        1
    )


    # Pure heatmap
    heatmap = show_cam_on_image(
        np.zeros_like(
            rgb_image
        ),
        grayscale_cam,
        use_rgb=True
    )


    # Original + Grad-CAM
    overlay = show_cam_on_image(
        rgb_image.astype(
            np.float32
        ),
        grayscale_cam,
        use_rgb=True
    )


    return (

        (
            rgb_image * 255
        ).astype(
            np.uint8
        ),

        heatmap,

        overlay
    )


# ============================================================
# PRODUCTION UI PREDICTION WRAPPER
# ============================================================

def ui_predict_production(
    image
):

    # ========================================================
    # NO IMAGE
    # ========================================================

    if image is None:

        empty_prediction = """

        <div class="prediction-card">

            <div class="prediction-label">
                🧠 AI PREDICTION
            </div>

            <div class="diagnosis-name">
                READY FOR ANALYSIS
            </div>

            <div class="full-diagnosis">
                Upload a histopathology image and click
                <b>Analyze Image with ORAL AI</b>.
            </div>

        </div>

        """

        return (
            None,
            None,
            None,
            empty_prediction,
            {},
            ""
        )


    # ========================================================
    # IMAGE FORMAT SAFETY
    # ========================================================

    if not isinstance(
        image,
        Image.Image
    ):

        image = Image.fromarray(
            image
        )


    image = image.convert(
        "RGB"
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    (
        predicted_class,
        confidence,
        probabilities,
        prediction_index

    ) = predict_orchid(
        image
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    (
        original_img,
        heatmap_img,
        overlay_img

    ) = generate_gradcam(
        image,
        prediction_index
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence_pct = (
        confidence * 100
    )


    if confidence_pct >= 80:

        confidence_status = (
            "High Confidence"
        )

        confidence_message = (
            "The model shows a strong preference "
            "for this classification."
        )

        confidence_class = (
            "high-confidence"
        )


    elif confidence_pct >= 50:

        confidence_status = (
            "Moderate Confidence"
        )

        confidence_message = (
            "The prediction shows moderate separation "
            "from the remaining classes."
        )

        confidence_class = (
            "moderate-confidence"
        )


    else:

        confidence_status = (
            "Low Confidence"
        )

        confidence_message = (
            "Class probabilities are closely distributed. "
            "Consider expert pathological review."
        )

        confidence_class = (
            "low-confidence"
        )


    # ========================================================
    # DIAGNOSIS
    # ========================================================

    full_diagnosis = (
        CLASS_DESCRIPTIONS[
            predicted_class
        ]
    )

    display_class = (
        predicted_class.upper()
    )


    # ========================================================
    # PREDICTION CARD
    # ========================================================

    prediction_html = f"""

    <div class="prediction-card">

        <div class="prediction-label">

            🧠 AI PREDICTION

        </div>


        <div class="diagnosis-name">

            {display_class}

        </div>


        <div class="full-diagnosis">

            {full_diagnosis}

        </div>


        <div class="confidence-box">

            <div class="confidence-label">

                Prediction Confidence

            </div>


            <div class="
                confidence-value
                {confidence_class}
            ">

                {confidence_pct:.2f}%

            </div>

        </div>


        <div class="confidence-status">

            <div class="
                status-title
                {confidence_class}
            ">

                ● {confidence_status}

            </div>


            <div class="status-message">

                {confidence_message}

            </div>

        </div>

    </div>

    """


    # ========================================================
    # DISCLAIMER
    # ========================================================

    disclaimer_html = """

    <div class="disclaimer">

        ⚠️ <b>Research and Educational Use Only</b><br>

        This AI system is intended for research and
        educational purposes. It is not designed to replace
        professional pathological diagnosis or clinical
        decision-making.

    </div>

    """


    # ========================================================
    # EXACTLY SIX OUTPUTS
    # ========================================================

    return (

        original_img,

        heatmap_img,

        overlay_img,

        prediction_html,

        probabilities,

        disclaimer_html

    )


# ============================================================
# PRODUCTION CSS — CELL 19 STYLE
# ============================================================

CUSTOM_CSS = """

html,
body,
#root {

    margin: 0 !important;
    padding: 0 !important;

    width: 100% !important;

    min-height: 100vh !important;

    background: #f4f7fb !important;
}


.gradio-container {

    width: 100% !important;

    max-width: none !important;

    min-height: 100vh !important;

    margin: 0 !important;

    padding: 0 !important;

    background: #f4f7fb !important;

    color: #1e293b !important;
}


/* =========================================================
   APP WRAPPER
========================================================= */

.app-wrapper {

    width: 100% !important;

    max-width: 1500px !important;

    margin: 0 auto !important;

    padding:
        28px
        45px
        70px
        45px !important;

    box-sizing: border-box !important;
}


/* =========================================================
   NAVBAR
========================================================= */

.navbar {

    width: 100%;

    box-sizing: border-box;

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding:
        18px
        25px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            #172033,
            #263b5c
        );

    box-shadow:
        0 12px 35px
        rgba(15,23,42,0.18);
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

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #38bdf8
        );
}


.logo-title {

    color: white;

    font-size: 22px;

    font-weight: 700;

    letter-spacing: 1px;
}


.logo-subtitle {

    margin-top: 3px;

    color: #a5b4c8;

    font-size: 11px;
}


.system-status {

    padding:
        10px
        16px;

    border-radius: 14px;

    color: #cbd5e1;

    font-size: 11px;

    background:
        rgba(
            255,
            255,
            255,
            0.08
        );
}


.status-dot {

    display: inline-block;

    width: 7px;

    height: 7px;

    border-radius: 50%;

    background: #22c55e;

    margin-right: 7px;
}


/* =========================================================
   HERO
========================================================= */

.hero-section {

    text-align: center;

    padding:
        70px
        20px
        75px
        20px;
}


.hero-title {

    font-size: 36px;

    font-weight: 700;

    color: #172033;
}


.hero-description {

    max-width: 800px;

    margin:
        16px
        auto
        0
        auto;

    line-height: 1.8;

    font-size: 15px;

    color: #64748b;
}


/* =========================================================
   ANALYSIS ROW
========================================================= */

.analysis-row {

    align-items:
        flex-start !important;

    gap: 35px !important;
}


.analysis-row > .column {

    align-self:
        flex-start !important;
}


/* =========================================================
   SECTION HEADERS
========================================================= */

.section-header {

    height: 72px;

    display: flex;

    align-items: center;

    gap: 13px;

    margin-bottom: 18px;
}


.section-icon {

    width: 45px;

    height: 45px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 13px;

    background: #eef4ff;

    font-size: 20px;
}


.section-title {

    font-size: 19px;

    font-weight: 700;

    color: #1e293b;
}


.section-subtitle {

    margin-top: 5px;

    font-size: 11px;

    color: #64748b;
}


/* =========================================================
   IMAGE FRAME
========================================================= */

.image-frame {

    border-radius:
        18px !important;

    overflow:
        hidden !important;

    border:
        1px solid
        #dbe3ef !important;

    background:
        white !important;

    box-shadow:
        0 8px 25px
        rgba(15,23,42,0.06);
}


/* =========================================================
   GRADIO IMAGE UPLOAD — TEXT VISIBILITY FIX
========================================================= */

/* Make the upload area text clearly visible */
.image-frame,
.image-frame * {
    opacity: 1 !important;
}

.image-frame {
    color: #475569 !important;
}

.image-frame .wrap,
.image-frame .image-container,
.image-frame .upload-container {
    color: #475569 !important;
}

/* Gradio upload placeholder */
.image-frame [data-testid="upload-text"],
.image-frame .upload-text,
.image-frame .upload-text *,
.image-frame .or,
.image-frame .or *,
.image-frame .drop-text,
.image-frame .drop-text *,
.image-frame p,
.image-frame span {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Upload icon */
.image-frame svg {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Upload label/header */
.image-frame label,
.image-frame label span {
    color: #334155 !important;
    opacity: 1 !important;
}

/* =========================================================
   PREDICTION CARD — TEXT VISIBILITY
========================================================= */

.prediction-card {
    color: #f8fafc !important;
}

.prediction-card .prediction-label {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}

.prediction-card .diagnosis-name {
    color: #7dd3fc !important;
    opacity: 1 !important;
}

.prediction-card .full-diagnosis {
    color: #f1f5f9 !important;
    opacity: 1 !important;
}

.prediction-card .confidence-label {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}

.prediction-card .status-message {
    color: #e2e8f0 !important;
    opacity: 1 !important;
}

/* =========================================================
   ANALYZE BUTTON
========================================================= */



#analyze_button {

    margin-top:
        16px !important;
}


#analyze_button button {

    width:
        100% !important;

    height:
        54px !important;

    border:
        none !important;

    border-radius:
        14px !important;

    font-size:
        14px !important;

    font-weight:
        700 !important;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #38bdf8
        ) !important;

    box-shadow:
        0 10px 25px
        rgba(37,99,235,0.25)
        !important;
}


/* =========================================================
   PREDICTION CARD
========================================================= */

.prediction-card {

    min-height:
        430px;

    box-sizing:
        border-box;

    padding:
        42px
        35px;

    border-radius:
        22px;

    text-align:
        center;

    background:
        linear-gradient(
            145deg,
            #16213a,
            #202f50
        );

    box-shadow:
        0 15px 40px
        rgba(15,23,42,0.20);
}


.prediction-label {

    color:
        #a5b4c8;

    font-size:
        11px;

    font-weight:
        700;

    letter-spacing:
        2px;
}


.diagnosis-name {

    margin-top:
        25px;

    font-size:
        34px;

    font-weight:
        800;

    letter-spacing:
        2px;

    color:
        #7dd3fc;
}


.full-diagnosis {

    max-width:
        520px;

    margin:
        12px
        auto
        0;

    font-size:
        14px;

    line-height:
        1.6;

    color:
        #e2e8f0;
}


/* =========================================================
   CONFIDENCE BOX
========================================================= */

.confidence-box {

    margin-top:
        28px;

    padding:
        18px;

    border-radius:
        14px;

    background:
        rgba(
            15,
            23,
            42,
            0.55
        );
}


.confidence-label {

    color:
        #94a3b8;

    font-size:
        12px;
}


.confidence-value {

    margin-top:
        7px;

    font-size:
        38px;

    font-weight:
        800;
}


.high-confidence {

    color:
        #4ade80 !important;
}


.moderate-confidence {

    color:
        #facc15 !important;
}


.low-confidence {

    color:
        #fb7185 !important;
}


/* =========================================================
   CONFIDENCE STATUS
========================================================= */

.confidence-status {

    margin-top:
        18px;

    padding:
        16px;

    border-radius:
        14px;

    text-align:
        left;

    background:
        rgba(
            255,
            255,
            255,
            0.05
        );
}


.status-title {

    font-size:
        13px;

    font-weight:
        700;
}


.status-message {

    margin-top:
        7px;

    color:
        #cbd5e1;

    font-size:
        12px;

    line-height:
        1.6;
}


/* =========================================================
   PROBABILITY CARD
========================================================= */

.probability-card {

    margin-top:
        18px;

    padding:
        15px;

    border-radius:
        16px;

    background:
        white;

    border:
        1px solid
        #dbe3ef;

    box-shadow:
        0 8px 25px
        rgba(15,23,42,0.05);
}


/* =========================================================
   EXPLAINABLE AI
========================================================= */

.xai-header {

    text-align:
        center;

    margin-top:
        60px !important;

    margin-bottom:
        35px;
}


.xai-header h2 {

    font-size:
        27px;

    color:
        #1e293b;
}


.xai-header p {

    max-width:
        700px;

    margin:
        12px
        auto;

    color:
        #64748b;

    font-size:
        14px;

    line-height:
        1.7;
}


/* =========================================================
   DISCLAIMER
========================================================= */

.disclaimer {

    margin-top:
        40px;

    padding:
        20px
        24px;

    border-radius:
        15px;

    background:
        #fffaf0;

    border-left:
        5px solid
        #f59e0b;

    color:
        #92400e;

    line-height:
        1.6;
}


/* =========================================================
   MODEL INFORMATION
========================================================= */

.model-info {
    margin-top: 50px !important;
    padding: 35px !important;
    border-radius: 20px !important;
    background: #ffffff !important;
    border: 1px solid #dbe3ef !important;
    box-shadow: 0 8px 25px rgba(15,23,42,0.05);
    color: #1e293b !important;
}

/* Force readable text inside Gradio Markdown */
.model-info .prose,
.model-info .markdown,
.model-info .prose *,
.model-info .markdown *,
.model-info h1,
.model-info h2,
.model-info h3,
.model-info h4,
.model-info p,
.model-info li,
.model-info th,
.model-info td,
.model-info strong,
.model-info b {
    color: #1e293b !important;
}

/* Main heading */
.model-info h2 {
    color: #172033 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Subheadings */
.model-info h3 {
    color: #1e293b !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    margin-top: 28px !important;
}

/* Table */
.model-info table {
    width: 100% !important;
    border-collapse: collapse !important;
    color: #1e293b !important;
}

.model-info th {
    color: #172033 !important;
    background: #f8fafc !important;
    font-weight: 700 !important;
    text-align: left !important;
    border: 1px solid #dbe3ef !important;
    padding: 10px !important;
}

.model-info td {
    color: #475569 !important;
    background: #ffffff !important;
    border: 1px solid #dbe3ef !important;
    padding: 10px !important;
}

/* Dataset class list */
.model-info li {
    color: #475569 !important;
    margin-bottom: 8px !important;
}

/* Normal paragraphs */
.model-info p {
    color: #475569 !important;
    line-height: 1.6 !important;
}

/* =========================================================
   RESPONSIVE DESIGN
========================================================= */

@media (max-width: 900px) {

    .app-wrapper {

        padding:
            20px !important;
    }


    .navbar {

        padding:
            15px
            18px;
    }


    .system-status {

        display:
            none;
    }


    .hero-section {

        padding:
            50px
            10px !important;
    }


    .hero-title {

        font-size:
            28px !important;
    }


    .analysis-row {

        gap:
            20px !important;
    }


    .prediction-card {

        min-height:
            auto;
    }

}


@media (max-width: 600px) {

    .logo-title {

        font-size:
            18px;
    }


    .logo-subtitle {

        font-size:
            9px;
    }


    .hero-title {

        font-size:
            25px !important;
    }


    .hero-description {

        font-size:
            13px;
    }


    .diagnosis-name {

        font-size:
            27px;
    }

}

"""


# ============================================================
# CREATE GRADIO APPLICATION
# ============================================================

with gr.Blocks(

    title=
        "ORAL AI — Intelligent Oral Histopathology Analysis",

    css=
        CUSTOM_CSS

) as demo:


    # ========================================================
    # MAIN APP WRAPPER
    # ========================================================

    with gr.Column(
        elem_classes=[
            "app-wrapper"
        ]
    ):


        # ====================================================
        # NAVBAR
        # ====================================================

        gr.HTML(
            """

            <div class="navbar">

                <div class="nav-left">

                    <div class="logo-box">
                        🧬
                    </div>

                    <div>

                        <div class="logo-title">
                            ORAL AI
                        </div>

                        <div class="logo-subtitle">
                            Intelligent Oral Histopathology Analysis Platform
                        </div>

                    </div>

                </div>


                <div class="system-status">

                    <span class="status-dot"></span>

                    AI System Online
                    &nbsp; | &nbsp;
                    Model v1.0

                </div>

            </div>

            """
        )


        # ====================================================
        # HERO
        # ====================================================

        gr.HTML(
            """

            <div class="hero-section">

                <div class="hero-title">

                    Analyze Oral Histopathology with AI

                </div>


                <div class="hero-description">

                    Upload a histopathology image to receive an
                    AI-assisted classification, confidence analysis,
                    class probability distribution, and visual
                    explanation through Grad-CAM.

                </div>

            </div>

            """
        )


        # ====================================================
        # MAIN ANALYSIS
        # ====================================================

        with gr.Row(
            elem_classes=[
                "analysis-row"
            ]
        ):


            # =================================================
            # LEFT — IMAGE
            # =================================================

            with gr.Column(
                scale=1
            ):


                gr.HTML(
                    """

                    <div class="section-header">

                        <div class="section-icon">
                            🔬
                        </div>

                        <div>

                            <div class="section-title">
                                Histopathology Image
                            </div>

                            <div class="section-subtitle">
                                Upload an oral tissue sample
                            </div>

                        </div>

                    </div>

                    """
                )


                input_image = gr.Image(

                    type="pil",

                    label=
                        "Upload Histopathology Image",

                    height=430,

                    elem_classes=[
                        "image-frame"
                    ]
                )


                predict_button = gr.Button(

                    "✨ Analyze Image with ORAL AI",

                    variant="primary",

                    elem_id=
                        "analyze_button"
                )


            # =================================================
            # RIGHT — PREDICTION
            # =================================================

            with gr.Column(
                scale=1
            ):


                gr.HTML(
                    """

                    <div class="section-header">

                        <div class="section-icon">
                            🧠
                        </div>

                        <div>

                            <div class="section-title">
                                AI Classification Result
                            </div>

                            <div class="section-subtitle">
                                AI-assisted prediction and confidence analysis
                            </div>

                        </div>

                    </div>

                    """
                )


                prediction_output = gr.HTML(

                    value="""

                    <div class="prediction-card">

                        <div class="prediction-label">
                            🧠 AI PREDICTION
                        </div>

                        <div class="diagnosis-name">
                            READY FOR ANALYSIS
                        </div>

                        <div class="full-diagnosis">

                            Upload a histopathology image and run
                            ORAL AI to begin the classification.

                        </div>

                    </div>

                    """
                )


                with gr.Column(
                    elem_classes=[
                        "probability-card"
                    ]
                ):

                    probability_output = gr.Label(

                        label=
                            "📊 Class Probability Distribution",

                        num_top_classes=5
                    )


        # ====================================================
        # EXPLAINABLE AI HEADER
        # ====================================================

        gr.HTML(
            """

            <div class="xai-header">

                <h2>
                    🔍 Explainable AI Analysis
                </h2>

                <p>

                    Explore which regions of the histopathology
                    image contributed most strongly to the AI
                    model's prediction using Grad-CAM
                    visualization.

                </p>

            </div>

            """
        )


        # ====================================================
        # GRAD-CAM OUTPUTS
        # ====================================================

        with gr.Row():


            original_output = gr.Image(

                label=
                    "Original Image",

                height=320,

                elem_classes=[
                    "image-frame"
                ]
            )


            heatmap_output = gr.Image(

                label=
                    "Grad-CAM Attention",

                height=320,

                elem_classes=[
                    "image-frame"
                ]
            )


            overlay_output = gr.Image(

                label=
                    "Grad-CAM Overlay",

                height=320,

                elem_classes=[
                    "image-frame"
                ]
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        disclaimer_output = gr.HTML()


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with gr.Column(
            elem_classes=[
                "model-info"
            ]
        ):

            gr.Markdown(
                """

## 🧠 Model Information

| Component | Description |
|---|---|
| **Visual Backbone** | Swin Transformer |
| **Semantic Guidance** | CLIP Text Embeddings |
| **Fusion Strategy** | Multi-Stage Cross Attention |
| **Decoder** | U-Net Inspired Decoder |
| **Classifier** | 5-Class Oral Histopathology Classification |

### ORCHID Dataset Classes

- 🔴 **MDOSCC** — Moderately Differentiated OSCC
- 🟢 **NORMAL** — Normal Oral Tissue
- 🟣 **OSMF** — Oral Submucous Fibrosis
- 🟠 **PDOSCC** — Poorly Differentiated OSCC
- 🔵 **WDOSCC** — Well Differentiated OSCC

### Model Performance

**Test Accuracy:** 95.73%

**Macro F1-Score:** 96.20%

**Macro ROC-AUC:** 0.9976

"""
            )


    # ========================================================
    # CONNECT BUTTON
    # ========================================================

    predict_button.click(

        fn=
            ui_predict_production,

        inputs=
            input_image,

        outputs=[

            original_output,

            heatmap_output,

            overlay_output,

            prediction_output,

            probability_output,

            disclaimer_output

        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    print(
        "✓ ORAL AI Production Interface created successfully"
    )

    demo.queue().launch()