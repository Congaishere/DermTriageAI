import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Impetus",
    page_icon="🔬",
    layout="centered"
)

# --- Force Mobile/iPad Safari Web App Name to "Impetus" ---
components.html(
    """
    <script>
        document.title = "Impetus";
        window.parent.document.title = "Impetus";
        
        let metaApple = window.parent.document.querySelector('meta[name="apple-mobile-web-app-title"]');
        if (!metaApple) {
            metaApple = window.parent.document.createElement('meta');
            metaApple.name = "apple-mobile-web-app-title";
            window.parent.document.head.appendChild(metaApple);
        }
        metaApple.content = "Impetus";

        let metaApp = window.parent.document.querySelector('meta[name="application-name"]');
        if (!metaApp) {
            metaApp = window.parent.document.createElement('meta');
            metaApp.name = "application-name";
            window.parent.document.head.appendChild(metaApp);
        }
        metaApp.content = "Impetus";
    </script>
    """,
    height=0,
    width=0,
)

# --- UI Header & Clinical Branding ---
st.title("🔬 Impetus: Tropical Dermatoses Triage")
st.caption("Clinical Decision-Support Calibrated for Fitzpatrick Types IV–VI | 4-Way Differential Analysis")

CLASSES = [
    "Eczema (Atopic/Allergic)",
    "Impetigo / Pyoderma (Bacterial)",
    "Psoriasis (Plaque/Scaly)",
    "Tinea (Fungal/Ringworm)"
]

# --- Model Loading ---
@st.cache_resource
def load_model():
    m = models.mobilenet_v3_small(weights=None)
    num_ftrs = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(num_ftrs, 4)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Search all possible directory structures where weights could exist
    possible_paths = [
        os.path.join(base_dir, "skin_model.pth"),
        os.path.join(base_dir, "..", "skin_model.pth"),
        os.path.join(base_dir, "skin_model"),
        os.path.join(base_dir, "..", "skin_model"),
        "skin_model.pth"
    ]

    loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            try:
                m.load_state_dict(torch.load(path, map_location="cpu"))
                loaded = True
                break
            except Exception:
                pass

    if not loaded:
        st.warning("⚠️ Notice: Running on uninitialized base weights. Please ensure skin_model.pth is in your GitHub repository.")

    m.eval()
    return m

model = load_model()

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Image Input ---
image_input = st.file_uploader("Upload Lesion Photo", type=["jpg", "png", "jpeg"])
if not image_input:
    image_input = st.camera_input("Or take a picture with camera")

if image_input:
    img = Image.open(image_input).convert("RGB")
    st.image(img, caption="Target Clinical Image", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fst = st.selectbox(
            "Fitzpatrick Phototype",
            ["Type III", "Type IV", "Type V", "Type VI"],
            index=1
        )
    with col2:
        lesion_loc = st.selectbox(
            "Anatomical Site",
            ["Face / Perioral", "Limbs / Extremities", "Trunk", "Flexural folds / Groin"]
        )

    if st.button("Run Multi-Class Triage & Visual Deconvolution"):
        # 1. Melanin-Decoupled Vascular Contrast Isolation (CIELAB Space)
        np_img = np.array(img)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_a = clahe.apply(a)

        # Apply Heatmap via OpenCV Magma colormap
        heatmap = cv2.applyColorMap(enhanced_a, cv2.COLORMAP_MAGMA)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        st.markdown("### 🔍 Vascular Contrast Isolation (Melanin-Decoupled)")
        st.image(heatmap_rgb, caption="Sub-Visual Erythema & Border Contrast Map", use_container_width=True)

        # 2. Multi-Class Probability Inference
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy() * 100

        top_idx = int(np.argmax(probs))
        top_confidence = float(probs[top_idx])

        # 3. Confidence Threshold Gate (Handling Out-of-Distribution / Clear Skin)
        if top_confidence < 45.0:
            st.info(
                "ℹ️ **Indeterminate / Low Contrast Detected:** Visual features do not meet the minimum diagnostic threshold for active lesion differentiation. "
                "Ensure the lesion is focused and centered, or consider in-person clinical dermoscopy."
            )
        else:
            st.markdown("### 📊 Differential Confidence Breakdown")
            for i, class_name in enumerate(CLASSES):
                st.write(f"**{class_name}:** {probs[i]:.1f}%")
                st.progress(int(probs[i]))

            # 4. Clinical Safety Flags
            if top_idx == 1:
                st.error("⚠️ **Bacterial Pyoderma Suspected:** Contraindicated for isolated topical corticosteroid monotherapy.")
            elif top_idx == 3:
                st.warning("⚠️ **Superficial Fungal Suspected:** High risk of Tinea Incognito if topical steroids are applied without antifungal coverage.")
            elif top_idx == 0:
                st.info("ℹ️ **Inflammatory Pattern (Eczema):** Evaluate barrier disruption, pruritus history, and xerosis.")
            elif top_idx == 2:
                st.info("ℹ️️ **Hyperkeratotic Plaque Pattern (Psoriasis):** Examine extensor surfaces for bilateral symmetry.")

            # 5. Clinical Management & Pharmacological Reference Pathways
            st.markdown("---")
            st.subheader("📋 Therapeutic Reference & Clinical Pathways")
            with st.expander(f"View Standard-of-Care Considerations for {CLASSES[top_idx]}", expanded=True):
                if top_idx == 0:  # Eczema
                    st.markdown("""
                    **First-Line Therapeutic Options (Physician Discretion):**
                    * **Barrier Repair:** Liberal application of unscented, ceramide- or lipid-rich emollients immediately post-bathing.
                    * **Anti-Inflammatory:** Topical corticosteroids (low-potency hydrocortisone 1–2.5% for facial/flexural areas; medium-potency for trunk/extremities) or topical calcineurin inhibitors (tacrolimus 0.03–0.1%).
                    
                    ⚠️ **Safety Precaution:** If meliceric (honey-colored) crusting or purulence develops, evaluate for secondary *Staphylococcus aureus* superinfection before escalating steroid potency.
                    """)
                elif top_idx == 1:  # Impetigo
                    st.markdown("""
                    **First-Line Therapeutic Options (Physician Discretion):**
                    * **Topical Antibacterial (Localized):** Crust debridement using warm saline soaks; topical mupirocin 2% or fusidic acid ointment 2–3 times daily for 5–7 days.
                    * **Systemic Antibacterial (Widespread / Bullous):** Oral beta-lactamase-resistant antibiotic regimen (cephalexin or amoxicillin-clavulanate).
                    
                    🛑 **CRITICAL CONTRAINDICATION:** **Avoid topical corticosteroid monotherapy.** Corticosteroids suppress local immune defenses, driving accelerated bacterial proliferation.
                    """)
                elif top_idx == 2:  # Psoriasis
                    st.markdown("""
                    **First-Line Therapeutic Options (Physician Discretion):**
                    * **Topical Therapy:** Fixed-dose combination topical corticosteroid and vitamin D3 analogue (calcipotriene/betamethasone dipropionate).
                    * **Keratolytic Agents:** Topical salicylic acid (2–5%) or urea formulations to facilitate scale debridement.
                    
                    ℹ️ **Clinical Note:** Avoid abrupt discontinuation of systemic corticosteroids to prevent precipitating generalized pustular psoriasis.
                    """)
                elif top_idx == 3:  # Tinea
                    st.markdown("""
                    **First-Line Therapeutic Options (Physician Discretion):**
                    * **Topical Antifungals:** Topical allylamines (terbinafine 1%) or azoles (clotrimazole, ketoconazole) applied daily, extending at least 2 cm beyond the active annular border for 2–4 weeks.
                    * **Extensive / Recalcitrant:** Oral terbinafine or itraconazole following potassium hydroxide (KOH) examination confirmation.
                    
                    🛑 **CRITICAL CONTRAINDICATION:** **Avoid topical corticosteroid monotherapy.** Steroids artificially blunt erythema while allowing rapid mycelial proliferation (*Tinea Incognito*).
                    """)
                st.caption("⚖️ *Investigational pilot decision-support reference only. Not a prescription or substitute for independent clinical judgment.*")

        # 6. Dermatologist Concordance Entry & Field Data Logging
        st.markdown("---")
        st.subheader("👨‍⚕️ Clinician Concordance Entry")
        doc_name = st.selectbox("Evaluating Clinician", ["Doctor A", "Doctor B", "Doctor C"])
        doc_diag = st.radio(
            "Your Independent Clinical Assessment:",
            CLASSES + ["Indeterminate / Laboratory Biopsy Needed"]
        )
        notes = st.text_input("Clinical Observations")

        if st.button("Log Case in Validation Dataset"):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(base_dir, "clinical_validation_log.csv")
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
