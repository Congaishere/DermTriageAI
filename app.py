"""Impetus research prototype. Does not provide clinical diagnoses or prescriptions."""
from pathlib import Path
import hashlib, io, json
from datetime import datetime, timezone
import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'skin_model_multilabel.pth'
DISPLAY = {'eczema':'Eczema (atopic/allergic)', 'impetigo':'Impetigo / pyoderma',
           'psoriasis':'Psoriasis (plaque/scaly)', 'tinea':'Tinea (fungal)'}
CLINICIAN_OPTIONS = list(DISPLAY.values()) + ['No visible target lesion', 'Other / uncertain']
REFERENCE = {
 'eczema': 'Review history of itch, xerosis, distribution, exposures and possible secondary infection; verify management using current local dermatology guidance.',
 'impetigo': 'Check for crusting, purulence, systemic illness and need for microbiology; confirm treatment using current local antimicrobial guidance.',
 'psoriasis': 'Assess plaque morphology, distribution, nail/joint symptoms and alternatives; follow current local dermatology guidance.',
 'tinea': 'Consider confirmatory KOH/microscopy when appropriate and steroid-modified infection; follow current local antifungal guidance.'
}

st.set_page_config(page_title='Impetus',page_icon='🔬',layout='centered')
st.title('🔬 Impetus — Dermatology Research Prototype')
st.caption('Four independent disease scores · experimental image enhancement · clinician evaluation')
st.warning('Research-only prototype: scores are UNCALIBRATED and are not probabilities of disease, a healthy-skin diagnosis, or a substitute for independent assessment. Not for treatment decisions.')

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {MODEL_PATH}')
    checkpoint = torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
    names = checkpoint['diseases']
    if checkpoint.get('architecture') != 'mobilenet_v3_small' or names != list(DISPLAY):
        raise ValueError('Unexpected architecture or disease order in checkpoint')
    if checkpoint.get('output_activation') != 'sigmoid':
        raise ValueError('Not a sigmoid multilabel checkpoint')
    if set(checkpoint['class_to_idx']) != set(names+['healthy']):
        raise ValueError('Expected five training categories including healthy')
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features,len(names))
    net.load_state_dict(checkpoint['model_state_dict'],strict=True)
    net.eval()
    pre = transforms.Compose([transforms.Resize(tuple(checkpoint['image_size'])), transforms.ToTensor(),
                              transforms.Normalize(checkpoint['mean'],checkpoint['std'])])
    return net, names, pre, checkpoint

try:
    model, diseases, preprocess, checkpoint = load_model()
except Exception as exc:
    st.error(f'Prediction disabled: {exc}')
    st.stop()

st.info('Fitzpatrick phototype and site are recorded for review only; they do not currently change the model predictions. Contrast enhancement is not validated vascular or melanin separation.')
up = st.file_uploader('Upload a photograph (avoid identifiable patient information)',type=['jpg','jpeg','png'])
cam = st.camera_input('Or take a photograph') if up is None else None
file = up if up is not None else cam
if file is None:
    st.stop()
raw = file.getvalue()
if len(raw)>12*1024*1024:
    st.error('Maximum image size is 12 MB.'); st.stop()
try:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
except Exception:
    st.error('Could not open the image.'); st.stop()
if img.width * img.height > 25_000_000:
    st.error('Image exceeds 25 megapixels.'); st.stop()
# New image clears the prior assessment and model result.
fingerprint=hashlib.sha256(raw).hexdigest()
if st.session_state.get('active_image') != fingerprint:
    st.session_state['active_image']=fingerprint
    st.session_state['locked_assessment']=None
    st.session_state['analysis']=None

st.image(img, caption='Input image',use_container_width=True)
with st.form('clinician_before_ai'):
    st.subheader('1. Independent clinician assessment')
    doctor=st.selectbox('Evaluator code',['Doctor A','Doctor B','Doctor C'])
    phototype=st.selectbox('Reported Fitzpatrick type (optional)', ['Unknown','Type I','Type II','Type III','Type IV','Type V','Type VI'])
    site=st.selectbox('Anatomical site',['Unknown','Face / perioral','Limbs / extremities','Trunk','Flexural folds / groin','Other'])
    assessment=st.selectbox('Assessment BEFORE viewing AI results',CLINICIAN_OPTIONS)
    notes=st.text_area('Observations (do not enter personal identifiers)',max_chars=1000)
    lock=st.form_submit_button('Lock assessment')
if lock:
    if st.session_state['locked_assessment'] is not None:
        st.warning('An assessment is already locked for this image. To assess independently again, upload a new image in a new session.')
    else:
        st.session_state['locked_assessment']={
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'evaluator_code':doctor,'phototype':phototype,'site':site,
            'assessment':assessment,'notes':notes}
        st.success('Assessment locked for this image. You may now run AI analysis.')
if st.session_state['locked_assessment'] is None:
    st.info('Record the independent assessment first to limit anchoring bias.')
    st.stop()

if st.button('2. Run AI analysis',disabled=st.session_state['analysis'] is not None):
    with torch.inference_mode():
        logits=model(preprocess(img).unsqueeze(0))
        scores=torch.sigmoid(logits)[0].cpu().numpy().tolist()
    a=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2LAB)[:,:,1]
    contrast=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(a)
    heat=cv2.cvtColor(cv2.applyColorMap(contrast,cv2.COLORMAP_MAGMA),cv2.COLOR_BGR2RGB)
    st.session_state['analysis']={'scores':scores,'heatmap':heat}

analysis=st.session_state['analysis']
if analysis is None:
    st.stop()
st.subheader('3. Experimental contrast map')
st.image(analysis['heatmap'],caption='CLAHE-enhanced CIELAB a-channel; not a validated vascular measurement',use_container_width=True)
st.subheader('4. Independent disease scores (NOT calibrated probabilities)')
for name,score in zip(diseases,analysis['scores']):
    st.write(f'**{DISPLAY[name]}:** {score*100:.1f} / 100 (experimental score)')
    st.progress(float(score))
st.warning('No decision threshold has been clinically validated. Low scores do NOT confirm healthy skin; high scores do NOT confirm disease. Unseen diseases may generate high scores.')

st.subheader('5. Condition-related reference notes')
st.caption('General points for trained clinicians to verify, NOT individual treatment suggestions. Not linked to a diagnostic decision.')
for name in diseases:
    with st.expander(DISPLAY[name]):
        st.write(REFERENCE[name])

st.subheader('6. Clinician feedback (research export)')
review=st.text_area('Post-AI observations / disagreement (no identifying information)',key='post_ai_notes')
feedback={
 'model_version':'skin_model_multilabel.pth',
 'model_validated_for_clinical_use':False,
 'independent_assessment':st.session_state['locked_assessment'],
 'uncalibrated_scores':dict(zip(diseases,[round(float(x),6) for x in analysis['scores']])),
 'post_ai_notes':review,
 'reference_standard':'Not established by this entry',
}
st.download_button('Download this research feedback as JSON',data=json.dumps(feedback,indent=2),
                   file_name='impetus_feedback.json',mime='application/json')
st.caption('This version does not persist photographs or feedback into a server-side clinical database. Downloads are local to the clinician. Do not use identifiable patient information. A production clinical workflow requires approved secure storage, authentication, and governance.')
