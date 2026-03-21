import cv2
import numpy as np
from PIL import Image
import torch
import timm
from torchvision import transforms
import streamlit as st

EYE_CLASSES = ['Bulging_Eyes', 'Cataracts', 'Conjunctivitis', 
                'Crossed_Eyes', 'Normal', 'Uveitis']

@st.cache_resource
def load_eye_model():
    from torchvision import models
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 6)
    m.load_state_dict(torch.load('eye_model_clean_best.pth',
                                  map_location='cpu',
                                  weights_only=False))
    m.eval()
    return m

def analyze_front_eye(image_pil):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = tf(image_pil.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(
            load_eye_model()(img_tensor), dim=1
        )[0].numpy()

    results = {}
    for cls, prob in zip(EYE_CLASSES, probs):
        if cls == 'Normal':
            continue
        # Map class names to display names
        display = {
            'Bulging_Eyes': 'Bulging Eyes',
            'Cataracts': 'Cataract',
            'Conjunctivitis': 'Redness / Conjunctivitis',
            'Crossed_Eyes': 'Crossed Eyes',
            'Uveitis': 'Uveitis'
        }.get(cls, cls)
        results[display] = float(prob)

    return results

def get_front_eye_recommendations(results):
    recommendations = []
    high_risk = []
    for condition, score in results.items():
        if score > 0.6:
            high_risk.append(condition)
        if score > 0.4:
            if condition == 'Redness / Conjunctivitis':
                recommendations.append("Possible conjunctivitis — avoid touching eyes, consult a doctor if persists beyond 2 days")
            elif condition == 'Cataract':
                recommendations.append("Possible cataract — schedule a detailed eye examination with an ophthalmologist")
            elif condition == 'Uveitis':
                recommendations.append("Possible uveitis — this needs urgent attention, see a doctor immediately")
            elif condition == 'Bulging Eyes':
                recommendations.append("Eye bulging detected — consult a specialist to rule out thyroid issues")
            elif condition == 'Crossed Eyes':
                recommendations.append("Eye misalignment detected — consult an ophthalmologist")
    needs_fundus = any(results.get(c, 0) > 0.5 for c in ['Cataract', 'Uveitis'])
    return recommendations, high_risk, needs_fundus