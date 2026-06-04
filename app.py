"""
CropSense — Gradio Web App
Run: python app.py → opens at http://localhost:7860
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "cropsense_efficientnet.pth"
CLASS_JSON      = "class_names.json"
IMG_SIZE        = 224
TOP_K           = 5

# ── Load class names ──────────────────────────────────────────────────────────
with open(CLASS_JSON) as f:
    class_names = json.load(f)
num_classes = len(class_names)

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m

print("Loading CropSense model...")
model = load_model()
print("Model ready ✓")

# ── Transforms ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return {}, "Please upload a leaf image."

    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    top_probs, top_idxs = probs.topk(TOP_K)
    results = {class_names[i]: float(p) for i, p in zip(top_idxs, top_probs)}

    best_class = class_names[top_idxs[0]]
    confidence = float(top_probs[0]) * 100

    # Parse class name: "Tomato__Early_blight" → crop + condition
    parts  = best_class.replace("__", "_").split("_")
    crop   = parts[0]
    status = " ".join(parts[1:]) if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in status.lower()

    status_icon = "✅ Healthy" if is_healthy else "⚠️ Disease Detected"
    summary = (
        f"**Crop:** {crop}  \n"
        f"**Status:** {status_icon}  \n"
        f"**Condition:** {status}  \n"
        f"**Confidence:** {confidence:.1f}%"
    )
    return results, summary


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, .gradio-container, .main, .wrap, .app, .contain {
    background: #F0EDE6 !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: none !important;
    border: none !important;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 6vw !important;
}
footer { display: none !important; }

#cropsense-header {
    padding: 4rem 0 2rem;
    border-bottom: 1.5px solid #C8C3B8;
    margin-bottom: 3rem;
}
#cropsense-header .eyebrow {
    font-size: 11px; font-weight: 500;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #8A8479; margin-bottom: 0.5rem;
}
#cropsense-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.8rem, 6vw, 4.8rem) !important;
    font-weight: 800 !important; line-height: 1.0 !important;
    color: #1A1916 !important; letter-spacing: -0.03em;
    margin: 0 0 0.8rem !important;
}
#cropsense-header .sub {
    font-size: 14px; color: #6B6860;
    font-weight: 300; letter-spacing: 0.02em;
}

.upload-zone .wrap {
    border: 1.5px dashed #B8B3A8 !important;
    border-radius: 4px !important;
    background: #EAE7DF !important;
}
.upload-zone .wrap:hover {
    border-color: #1A1916 !important;
    background: #E4E1D8 !important;
}

.section-label {
    font-size: 10px; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #8A8479; margin-bottom: 1rem; display: block;
}

#predict-btn {
    background: #1A1916 !important; color: #F0EDE6 !important;
    border: none !important; border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    padding: 14px 28px !important; width: 100% !important;
    margin-top: 0.75rem !important; cursor: pointer !important;
    transition: background 0.2s;
}
#predict-btn:hover { background: #333028 !important; }

.summary-box {
    background: #FFFFFF !important;
    border: 1px solid #D8D3CB !important;
    border-radius: 4px !important;
    padding: 16px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}
.summary-box.healthy-card { border-left: 4px solid #4CAF50 !important; }

#info-strip {
    border-top: 1px solid #C8C3B8;
    margin-top: 3rem; padding: 1.5rem 0 2rem;
    display: flex; gap: 3rem; flex-wrap: wrap;
}
#info-strip .info-item { flex: 1; min-width: 140px; }
#info-strip .info-label {
    font-size: 10px; font-weight: 500;
    letter-spacing: 0.18em; text-transform: uppercase; color: #8A8479; margin-bottom: 4px;
}
#info-strip .info-val {
    font-family: 'Syne', sans-serif;
    font-size: 15px; font-weight: 600; color: #1A1916;
}
.main-row { gap: 4rem !important; align-items: flex-start !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="CropSense", css=CSS) as demo:

    gr.HTML("""
    <div id="cropsense-header">
        <p class="eyebrow">AI · Computer Vision · Agriculture</p>
        <h1>CropSense</h1>
        <p class="sub">Crop disease detection · EfficientNetB0 · 38 classes · 54K images · Grad-CAM explainability</p>
    </div>
    """)

    with gr.Row(elem_classes="main-row"):
        with gr.Column(scale=4):
            gr.HTML('<span class="section-label">Upload leaf image</span>')
            image_input = gr.Image(
                label="", type="numpy", height=340,
                elem_classes="upload-zone", show_label=False
            )
            predict_btn = gr.Button("Detect Disease", elem_id="predict-btn", variant="primary")

        with gr.Column(scale=6):
            gr.HTML('<span class="section-label">Diagnosis</span>')
            summary_out = gr.Markdown(elem_classes="summary-box")
            gr.HTML('<span class="section-label" style="margin-top:1.5rem">Top 5 predictions</span>')
            label_out = gr.Label(num_top_classes=TOP_K, show_label=False)

    gr.HTML("""
    <div id="info-strip">
        <div class="info-item"><div class="info-label">Model</div><div class="info-val">EfficientNetB0</div></div>
        <div class="info-item"><div class="info-label">Dataset</div><div class="info-val">PlantVillage</div></div>
        <div class="info-item"><div class="info-label">Classes</div><div class="info-val">38</div></div>
        <div class="info-item"><div class="info-label">Images</div><div class="info-val">54,305</div></div>
        <div class="info-item"><div class="info-label">Explainability</div><div class="info-val">Grad-CAM</div></div>
    </div>
    """)

    predict_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[label_out, summary_out]
    )

if __name__ == "__main__":
    demo.launch(share=False)
