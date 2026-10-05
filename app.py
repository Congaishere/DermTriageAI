import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import datetime
import os

st.set_page_config(page_title="DermTriage AI - Pilot", layout="centered")

st.title("🔬 Differential Triage Decision Support")
st.caption("Investigational Pilot for Fitzpatrick Types IV–VI | Tropical Dermatoses")

CLASSES = [
    "Eczema (Atopic/Allergic)",
    "Impetigo / Pyoderma (Bacterial)",
    "Psoriasis (Plaque/Scaly)",
    "Tinea (Fungal/Ringworm)"
]

@st.cache_resource
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 4)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'skin_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_input = st.file_uploader("Upload Lesion Photo", type=["jpg", "png", "jpeg"])
if not image_input:
    image_input = st.camera_input("Or take picture with camera")

if image_input:
    img = Image.open(image_input).convert("RGB")
    st.image(img, caption="Lesion Image", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fst = st.selectbox("Fitzpatrick Phototype", ["Type III", "Type IV", "Type V", "Type VI"], index=1)
    with col2:
        lesion_loc = st.selectbox("Anatomical Site", ["Face / Perioral", "Limbs / Extremities", "Trunk", "Flexural folds / Groin"])

    if st.button("Run Multi-Class Triage & Visual Deconvolution"):
        # Erythema/Vascular contrast enhancement
        np_img = np.array(img)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_a = clahe.apply(a)

        # Apply heatmap via OpenCV so Streamlit displays it cleanly
        heatmap = cv2.applyColorMap(enhanced_a, cv2.COLORMAP_MAGMA)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        st.markdown("### 🔍 Vascular Contrast Isolation (Melanin-Decoupled)")
        st.image(heatmap_rgb, caption="Sub-Visual Erythema & Border Contrast Map", use_container_width=True)

        # Neural Net Probability Outputs
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy() * 100

        st.markdown("### 📊 Differential Confidence Breakdown")
        for i, class_name in enumerate(CLASSES):
            st.write(f"**{class_name}:** {probs[i]:.1f}%")
            st.progress(int(probs[i]))

        top_idx = int(np.argmax(probs))
        if top_idx == 1:
            st.error("⚠️ **Bacterial Pyoderma Suspected:** Contraindicated for isolated topical corticosteroid monotherapy.")
        elif top_idx == 3:
            st.warning("⚠️ **Superficial Fungal Suspected:** High risk of Tinea Incognito if topical steroids are applied without antifungal coverage.")
        elif top_idx == 0:
            st.info("ℹ️ **Inflammatory Eczema:** Evaluate barrier disruption and pruritus history.")
        elif top_idx == 2:
            st.info("ℹ️ **Psoriatic Plaque:** Examine extensor surfaces for bilateral symmetry.")

        # Dermatologist Concordance Entry
        st.markdown("---")
        st.subheader("👨‍⚕️ Clinician Concordance Entry")
        doc_name = st.selectbox("Evaluating Clinician", ["Doctor A", "Doctor B", "Doctor C"])
        doc_diag = st.radio("Your Independent Clinical Assessment:", CLASSES + ["Indeterminate / Laboratory Biopsy Needed"])
        notes = st.text_input("Clinical Observations")

        if st.button("Log Case in Validation Dataset"):
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
            log_file = os.path.join(log_dir, "clinical_validation_log.csv")
            log_entry = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Doctor": doc_name,
                "Phototype": fst,
                "Site": lesion_loc,
                "Top_Model_Prediction": CLASSES[top_idx],
                "Confidence": f"{probs[top_idx]:.1f}%",
                "Doctor_Assessment": doc_diag,
                "Concordance": "Yes" if CLASSES[top_idx] == doc_diag else "No",
                "Notes": notes
            }
            df = pd.DataFrame([log_entry])
            if not os.path.exists(log_file):
                df.to_csv(log_file, index=False)
            else:
                df.to_csv(log_file, mode='a', header=False, index=False)
            st.success("Case successfully logged to clinical_validation_log.csv!")