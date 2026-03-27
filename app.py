# --- CLOUD SAFE HARDWARE PATCH ---
import sys
import unittest.mock as mock

# Trick the cloud server into thinking it has hardware & audio libraries
try:
    import RPi.GPIO
except ImportError:
    sys.modules['RPi'] = mock.MagicMock()
    sys.modules['RPi.GPIO'] = mock.MagicMock()

try:
    import pyaudio
except (ImportError, OSError):
    sys.modules['pyaudio'] = mock.MagicMock()

try:
    import speech_recognition
except ImportError:
    sys.modules['speech_recognition'] = mock.MagicMock()

try:
    import cv2
except (ImportError, OSError):
    sys.modules['cv2'] = mock.MagicMock()

# -----------------------------------

import streamlit as st
import numpy as np
import torch
from torchvision import transforms
import timm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from report_generator import generate_report
from voice_input import record_voice
from database import load_cases, save_case, update_case, load_appointments, get_patient_visits
from auth import (register_patient, login_patient,
                  register_doctor, login_doctor, get_all_doctors,
                  login_admin, get_pending_doctors,
                  approve_doctor, reject_doctor, DOCS_DIR)
from styles import load_css
from constants import DISEASE_NAMES, DISEASE_COLORS
from disease_info import DISEASE_INFO
from symptom_check import SYMPTOMS, triage
from patient_records import (render_patient_health_record,
                              render_doctor_patient_history)
import json
from datetime import datetime
import tempfile
import os
import hashlib
from encryption import encrypt_data, decrypt_data
from chatbot_flow import render_chatbot_screening
from optical_health_scan import run_unified_scanner
from datetime import datetime
import tempfile
import os
import hashlib
from encryption import encrypt_data, decrypt_data
from chatbot_flow import render_chatbot_screening
from optical_health_scan import run_unified_scanner

# ── Appointments ───────────────────────────────────────────────────────────────
APPOINTMENTS_FILE = "appointments.json"

ALL_TIME_SLOTS = [
    "09:00 AM","09:30 AM","10:00 AM","10:30 AM",
    "11:00 AM","11:30 AM","12:00 PM","02:00 PM",
    "02:30 PM","03:00 PM","03:30 PM","04:00 PM",
    "04:30 PM","05:00 PM"
]

def get_booked_slots(doctor_email, date_str):
    """Return set of time slots already taken by this doctor on this date
    (only counts Pending, Confirmed — not Cancelled/Completed)."""
    active_statuses = {"Pending", "Confirmed"}
    return {
        a["time_slot"]
        for a in load_appointments()
        if a["doctor_email"] == doctor_email
        and a["date"] == date_str
        and a["status"] in active_statuses
    }

def get_available_slots(doctor_email, date_str):
    """Return list of slots still open for this doctor on this date."""
    booked = get_booked_slots(doctor_email, date_str)
    return [s for s in ALL_TIME_SLOTS if s not in booked]

def book_appointment(patient_email, patient_name, doctor_email,
                     doctor_name, date, time_slot, case_id, notes):
    appointments = load_appointments()

    # ── Overlap guard ──────────────────────────────────────────
    booked = get_booked_slots(doctor_email, date)
    if time_slot in booked:
        return None, f"Slot {time_slot} on {date} is already booked for Dr. {doctor_name}. Please choose another slot."

    appt_id = f"APPT-{len(appointments)+1:04d}"
    appointments.append({
        "appointment_id": appt_id,
        "patient_email":  patient_email,
        "patient_name":   patient_name,
        "doctor_email":   doctor_email,
        "doctor_name":    doctor_name,
        "date":           date,
        "time_slot":      time_slot,
        "status":         "Pending",
        "case_id":        case_id,
        "notes":          notes,
        "created_at":     datetime.now().strftime("%d %b %Y, %I:%M %p")
    })
    encrypted = encrypt_data(json.dumps(appointments, indent=2))
    with open(APPOINTMENTS_FILE, 'wb') as f:
        f.write(encrypted)
    return appt_id, None

def update_appointment_status(appt_id, status):
    if not os.path.exists(APPOINTMENTS_FILE):
        return
    with open(APPOINTMENTS_FILE, 'rb') as f:
        raw = f.read()
    try:
        appointments = json.loads(decrypt_data(raw))
    except:
        with open(APPOINTMENTS_FILE, 'r') as f:
            appointments = json.load(f)
    for a in appointments:
        if a['appointment_id'] == appt_id:
            a['status'] = status
            if status == 'Confirmed' and not a.get('meet_link'):
                code = hashlib.md5(appt_id.encode()).hexdigest()[:10]
                a['meet_link'] = f"https://meet.jit.si/nayana-{appt_id.lower()}"
            break
    encrypted = encrypt_data(json.dumps(appointments, indent=2))
    with open(APPOINTMENTS_FILE, 'wb') as f:
        f.write(encrypted)

@st.dialog("Appointment Confirmed", width="small")
def show_booking_success(details):
    st.success(f"Successfully booked with Dr. {details['doc_name']}!")
    st.markdown(f"**Date:** {details['date']}")
    st.markdown(f"**Time:** {details['time_slot']}")
    st.markdown(f"**Booking ID:** {details['appt_id']}")
    st.info("Check your 'Appointments' tab for the meeting link.")
    if st.button("Close & Continue", use_container_width=True, type="primary"):
        del st.session_state['appt_success_details']
        st.session_state['last_case_id'] = details['cid']
        st.rerun()

# ── Chat ───────────────────────────────────────────────────────
MESSAGES_FILE = "messages.json"

def load_all_messages():
    if not os.path.exists(MESSAGES_FILE):
        return {}
    with open(MESSAGES_FILE, 'rb') as f:
        raw = f.read()
    try:
        return json.loads(decrypt_data(raw))
    except:
        with open(MESSAGES_FILE, 'r') as f:
            return json.load(f)

def save_all_messages(data):
    encrypted = encrypt_data(json.dumps(data, indent=2))
    with open(MESSAGES_FILE, 'wb') as f:
        f.write(encrypted)

def load_messages(case_id):
    data = load_all_messages()
    return data.get(case_id, [])

def send_message(case_id, sender_name, sender_role, text):
    data = load_all_messages()
    if case_id not in data:
        data[case_id] = []
    data[case_id].append({
        "sender_name": sender_name,
        "sender_role": sender_role,
        "text":        text,
        "timestamp":   datetime.now().strftime("%d %b %Y, %I:%M %p")
    })
    save_all_messages(data)

def render_chat(case_id, current_role, current_name):
    messages = load_messages(case_id)
    st.markdown('<div class="section-label">Messages</div>', unsafe_allow_html=True)
    if not messages:
        st.caption("No messages yet — start the conversation below.")
    else:
        for msg in messages:
            is_me      = msg['sender_role'] == current_role
            align      = "flex-end" if is_me else "flex-start"
            bg         = "rgba(59,130,246,0.12)" if is_me else "rgba(255,255,255,0.05)"
            border_clr = "rgba(59,130,246,0.3)"  if is_me else "rgba(255,255,255,0.1)"
            name_color = "#93c5fd" if is_me else "#94a3b8"
            role_badge = msg['sender_role'].capitalize()
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin-bottom:10px;">
                <div style="max-width:75%;background:{bg};
                            border:1px solid {border_clr};
                            border-radius:10px;padding:11px 15px;">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                        <span style="font-size:11px;font-weight:700;color:{name_color};">{msg['sender_name']}</span>
                        <span style="font-size:10px;font-weight:600;padding:1px 7px;
                            border-radius:4px;background:{border_clr};opacity:0.85;">{role_badge}</span>
                        <span style="font-size:10px;color:#64748b;margin-left:auto;">{msg['timestamp']}</span>
                    </div>
                    <div style="font-size:14px;line-height:1.55;">
                        {msg['text']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    key      = f"chat_input_{case_id}_{current_role}"
    msg_text = st.text_input("Type a message",
                              placeholder="Write something...",
                              key=key,
                              label_visibility="collapsed")
    if st.button("Send", key=f"send_{case_id}_{current_role}",
                 type="primary"):
        if msg_text.strip():
            send_message(case_id, current_name,
                         current_role, msg_text.strip())
            st.rerun()
        else:
            st.warning("Please type a message first")

# ── Front Eye Analyzer ─────────────────────────────────────────
EYE_CLASSES = ['Cataracts', 'Conjunctivitis', 'Crossed_Eyes',
               'Eyelid_Conditions', 'Normal', 'Uveitis']

@st.cache_resource
def load_eye_model():
    from torchvision import models as tv_models
    m = tv_models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(
        m.classifier[1].in_features, 6)
    m.load_state_dict(torch.load(
        'eye_model_v2_best.pth',
        map_location='cpu', weights_only=False))
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
    display_map = {
        'Cataracts':        'Cataract',
        'Conjunctivitis':   'Redness / Conjunctivitis',
        'Crossed_Eyes':     'Crossed Eyes',
        'Eyelid_Conditions':'Eyelid Condition',
        'Normal':           'Normal',
        'Uveitis':          'Uveitis'
    }
    return {display_map.get(cls, cls): float(prob)
            for cls, prob in zip(EYE_CLASSES, probs)}
def capture_and_detect_eye(image_pil):
    img_np = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) == 0:
        return None, False, None
    (x, y, w, h) = sorted(eyes, key=lambda e: e[2], reverse=True)[0]
    eye_roi = img_bgr[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    roi_adj = clahe.apply(roi_gray)
    roi_final = cv2.merge([roi_adj, roi_adj, roi_adj])
    img_annotated = img_np.copy()
    cv2.rectangle(img_annotated, (x,y), (x+w, y+h), (0,255,0), 2)
    cleaned_pil = Image.fromarray(cv2.cvtColor(roi_final, cv2.COLOR_BGR2RGB))
    annotated_pil = Image.fromarray(img_annotated)
    return cleaned_pil, True, annotated_pil

def get_front_eye_recommendations(results):
    recommendations = []
    high_risk       = []
    for condition, score in results.items():
        if score > 0.6:
            high_risk.append(condition)
        if score > 0.4:
            if condition == 'Redness / Conjunctivitis':
                recommendations.append(
                    "Possible conjunctivitis — avoid touching "
                    "eyes, consult a doctor if it persists "
                    "beyond 2 days")
            elif condition == 'Cataract':
                recommendations.append(
                    "Possible cataract — schedule a detailed "
                    "eye examination with an ophthalmologist")
            elif condition == 'Uveitis':
                recommendations.append(
                    "Possible uveitis — this needs urgent "
                    "attention, see a doctor immediately")
            elif condition == 'Eyelid Condition':
                recommendations.append(
                    "Eyelid condition detected — could be a "
                    "stye or infection, consult a doctor")
            elif condition == 'Crossed Eyes':
                recommendations.append(
                    "Eye misalignment detected — consult "
                    "an ophthalmologist")
    needs_fundus = any(
        results.get(c, 0) > 0.5
        for c in ['Cataract', 'Uveitis']
    )
    return recommendations, high_risk, needs_fundus

# ── App config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Nayana — AI Eye Screening",
    page_icon="N",
    layout="centered",
    initial_sidebar_state="expanded"
)

if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

st.markdown(load_css(st.session_state['dark_mode']),
            unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────
for key, val in {
    'role':               None,
    'patient_logged_in':  False,
    'patient_user':       None,
    'doctor_logged_in':   False,
    'doctor_user':        None,
    'admin_logged_in':    False,
    'page':               'screening',
    'doctor_page':        'cases',
    'triage':             None,
    'voice_memory':       None,
    'show_front_cam':     False,
    'show_fundus_cam':    False,
    'screening_step':     1,
    # ── chatbot flow ──────────────────────────────────────────
    'chat_stage':         'greeting',
    'chat_symptoms':      [],
    'quest_index':        0,
    'chat_raw_text':      '',
    'chat_clarify_text':  '',
    # ── consent ───────────────────────────────────────────────
    'consent_accepted':   False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Dialog Triggers ────────────────────────────────────────────
if 'appt_success_details' in st.session_state:
    show_booking_success(st.session_state['appt_success_details'])

# ── Consent popup gate ─────────────────────────────────────────
if not st.session_state['consent_accepted']:
    _,_col,_ = st.columns([1, 2.2, 1])
    with _col:
        st.markdown("""
        <div class="consent-card">
          <div class="consent-brand">nayana</div>
          <div class="consent-subtitle">Terms of Use &amp; Patient Consent</div>

          <p class="consent-body">
            This app supports <strong>tele-ophthalmology services</strong>
            and assists qualified professionals in eye screening and consultation.
          </p>
          <p class="consent-body">
            <strong>&#x2695;&#xFE0F; Clinical oversight:</strong>
            Final diagnosis and treatment decisions must always be made by a
            <strong>licensed eye care professional</strong> — not by this app alone.
          </p>
          <p class="consent-body" style="margin-bottom:8px;">By using this app, you agree to:</p>
          <ul style="font-size:13px;line-height:2.1;padding-left:22px;margin-bottom:18px;">
            <li>Provide accurate and complete information</li>
            <li>Follow medical advice provided through this platform</li>
            <li>Seek <strong>immediate medical care</strong> in emergencies</li>
          </ul>
          <p style="font-size:12px;color:#64748b;line-height:1.6;margin-bottom:0;">
            &#x1F512; All data is securely encrypted. Results may vary based on image and device conditions.
          </p>
          <div class="consent-emergency">
            <strong>&#x1F6A8; Emergency Notice:</strong>
            This app is <em>not</em> for emergency situations.
            For sudden vision loss, severe pain, or injury&#8202;—&#8202;seek immediate medical attention.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        c_dec, c_acc = st.columns(2)
        with c_dec:
            if st.button("Decline", use_container_width=True,
                         key="consent_decline"):
                st.error(
                    "You must accept the terms to use Nayana. "
                    "Please close this tab or accept to continue."
                )
        with c_acc:
            if st.button("Accept & Continue", type="primary",
                         use_container_width=True,
                         key="consent_accept"):
                st.session_state['consent_accepted'] = True
                st.rerun()
    st.stop()  # halt all further rendering until consent is given


# ── ML helpers ─────────────────────────────────────────────────
def check_quality(image_np):
    gray  = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    score = 100
    tips  = []
    blur  = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < 100:
        score -= 30
        tips.append("Hold the camera steady — image is blurry")
    brightness = gray.mean()
    if brightness < 60:
        score -= 25
        tips.append("Move to a brighter spot — too dark")
    elif brightness > 200:
        score -= 20
        tips.append("Reduce the light a bit — overexposed")
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=30, maxRadius=200)
    if circles is None:
        score -= 25
        tips.append("Centre your eye in the frame")
    return max(score, 0), tips

def preprocess_retinal(image_pil):
    img = np.array(image_pil.resize((224, 224)))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return Image.fromarray(
        cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB))

@st.cache_resource
def load_model():
    m = timm.create_model('efficientnet_b0',
                           pretrained=False, num_classes=8)
    m.load_state_dict(torch.load('odir_model.pth',
                                  map_location='cpu'))
    m.eval()
    return m

def predict(image_pil):
    image_pil = preprocess_retinal(image_pil)
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    with torch.no_grad():
        probs = torch.sigmoid(
            load_model()(tf(image_pil).unsqueeze(0)))[0]
    return probs.numpy()

def get_heatmap(image_pil):
    image_pil = preprocess_retinal(image_pil)
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    tensor = tf(image_pil).unsqueeze(0)
    m      = load_model()
    cam    = GradCAM(model=m, target_layers=[m.blocks[-1][-1]])
    gcam   = cam(input_tensor=tensor)[0]
    rgb    = np.array(image_pil.resize((224,224))) / 255.0
    return show_cam_on_image(
        rgb.astype(np.float32), gcam, use_rgb=True)

def get_risk(probs):
    nn = [(DISEASE_NAMES[i], probs[i])
          for i in range(1,8) if probs[i] > 0.3]
    if not nn:
        return "Looking good! No major concerns detected.", "low"
    elif max(p for _,p in nn) > 0.6:
        top = max(nn, key=lambda x: x[1])
        return (f"Please see a specialist — "
                f"{top[0]} detected."), "high"
    else:
        return ("Some signs worth checking. "
                "Follow-up recommended."), "moderate"

def green_chart(probs):
    dark = st.session_state.get('dark_mode', True)
    bg   = '#13131f' if dark else '#ffffff'
    tc   = '#94a3b8' if dark else '#2d6a4f'
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    bars = ax.barh(DISEASE_NAMES, probs*100,
                   color=DISEASE_COLORS, height=0.55,
                   edgecolor='none')
    ax.set_xlabel("Confidence (%)", color=tc, fontsize=9)
    ax.set_xlim(0, 108)
    ax.tick_params(colors=tc, labelsize=9)
    for s in ax.spines.values():
        s.set_color('#b7e4c7')
    ax.axvline(50, color='#b7e4c7', lw=0.8, ls='--')
    for bar, p in zip(bars, probs):
        ax.text(p*100+1.5, bar.get_y()+bar.get_height()/2,
                f'{p*100:.1f}%', va='center',
                fontsize=8.5, color=tc)
    plt.tight_layout(pad=0.5)
    return fig

model = load_model()

# ── Step bar ───────────────────────────────────────────────────
def step_bar(current_step):
    steps = ["Symptoms", "Eye Photos", "Results"]
    html  = '<div class="step-bar">'
    for i, label in enumerate(steps, 1):
        if i < current_step:
            dc, lc, di = "done",    "done",    "✓"
        elif i == current_step:
            dc, lc, di = "active",  "active",  str(i)
        else:
            dc, lc, di = "pending", "pending", str(i)
        html += f"""
        <div class="step">
            <div class="step-dot {dc}">{di}</div>
            <span class="step-label {lc}">{label}</span>
        </div>"""
        if i < len(steps):
            lc2 = "done" if i < current_step else ""
            html += f'<div class="step-line {lc2}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ── Notification helpers ───────────────────────────────────────
def get_patient_notifications(patient_email):
    all_cases    = load_cases()
    my_cases     = [c for c in all_cases
                    if c.get('patient_email','') == patient_email]
    new_reviews  = sum(1 for c in my_cases
                       if c['status'] == 'Reviewed')
    unread_msgs  = sum(
        len([m for m in load_messages(c['case_id'])
             if m['sender_role'] == 'doctor'])
        for c in my_cases
    )
    my_appts     = [a for a in load_appointments()
                    if a['patient_email'] == patient_email]
    appt_updates = sum(1 for a in my_appts
                       if a['status'] in ['Confirmed','Cancelled'])
    return new_reviews, unread_msgs, appt_updates

def get_doctor_notifications(doctor_email):
    all_appts     = [a for a in load_appointments()
                     if a['doctor_email'] == doctor_email]
    my_case_ids   = [a['case_id'] for a in all_appts]
    my_cases      = [c for c in load_cases()
                     if c['case_id'] in my_case_ids]
    pending_cases = sum(1 for c in my_cases
                        if c['status'] == 'Pending')
    pending_appts = sum(1 for a in all_appts
                        if a['status'] == 'Pending')
    unread_msgs   = sum(
        len([m for m in load_messages(c['case_id'])
             if m['sender_role'] == 'patient'])
        for c in my_cases
    )
    return pending_cases, pending_appts, unread_msgs

def notif_label(label, count):
    return f"{label} ({count})" if count > 0 else label

# ── Navbars ────────────────────────────────────────────────────
def patient_navbar(user):
    new_reviews, unread_msgs, appt_updates = get_patient_notifications(user['email'])
    all_cases = load_cases()
    my_cases  = [c for c in all_cases if c.get('patient_email','') == user['email']]
    total     = len(my_cases)
    reviewed  = sum(1 for c in my_cases if c['status'] == 'Reviewed')

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="sidebar-wordmark">nayana</div>
          <div class="sidebar-tag">AI Eye Screening</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-user-name">{user['name']}</div>
            <div class="sidebar-user-meta">Patient &nbsp;·&nbsp; {user['email']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-nav-label">Navigation</div>',
                    unsafe_allow_html=True)

        if st.button("Screening",
                     type=("primary" if st.session_state['page']=='screening' else "secondary"),
                     use_container_width=True, key="nav_scr"):
            st.session_state['page'] = 'screening'
            st.session_state['screening_step'] = 1
            st.session_state['chat_stage']     = 'greeting'
            st.session_state['chat_symptoms']  = []
            st.session_state['quest_index']    = 0
            st.rerun()

        results_label = f"Results  +{new_reviews + unread_msgs} new" if (new_reviews + unread_msgs) > 0 else "Results"
        if st.button(results_label,
                     type=("primary" if st.session_state['page']=='results' else "secondary"),
                     use_container_width=True, key="nav_res"):
            st.session_state['page'] = 'results'
            st.rerun()

        if st.button("Health Record",
                     type=("primary" if st.session_state['page']=='health_record' else "secondary"),
                     use_container_width=True, key="nav_hr"):
            st.session_state['page'] = 'health_record'
            st.rerun()

        if st.button("Optical Health",
                     type=("primary" if st.session_state['page']=='optical_scan' else "secondary"),
                     use_container_width=True, key="nav_ohs"):
            st.session_state['page'] = 'optical_scan'
            st.rerun()

        st.markdown('<div class="sidebar-nav-label">My Stats</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Screenings", total)
        c2.metric("Reviewed", reviewed)

        st.divider()
        dark_label = "☀ Light Mode" if st.session_state['dark_mode'] else "◑ Dark Mode"
        if st.button(dark_label, use_container_width=True, key="nav_theme"):
            st.session_state['dark_mode'] = not st.session_state['dark_mode']
            st.rerun()
        if st.button("Sign Out", use_container_width=True, key="nav_so"):
            keys_to_clear = [k for k in st.session_state.keys() if k not in ['dark_mode']]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()

        st.write("")
        st.markdown("""
        <div class="info-banner">
          <strong>&#x1F512; Consent &amp; Terms</strong><br>
          Results are AI-assisted only. All diagnoses must be confirmed by a licensed eye care
          professional. Your data is securely encrypted. Not for emergency use.
        </div>
        """, unsafe_allow_html=True)

def doctor_navbar(doc):
    pending_cases, pending_appts, unread_msgs = get_doctor_notifications(doc['email'])
    all_appts = [a for a in load_appointments() if a['doctor_email'] == doc['email']]
    total_cases = len(load_cases())

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="sidebar-wordmark">nayana</div>
          <div class="sidebar-tag">Doctor Portal</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-user-name">Dr. {doc['name']}</div>
            <div class="sidebar-user-meta">{doc.get('specialization','Ophthalmologist')}</div>
            <div class="sidebar-user-meta" style="margin-top:2px;">{doc.get('hospital','')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-nav-label">Navigation</div>',
                    unsafe_allow_html=True)

        cases_label = f"Cases  +{pending_cases} pending" if pending_cases > 0 else "Cases"
        if st.button(cases_label,
                     type=("primary" if st.session_state['doctor_page']=='cases' else "secondary"),
                     use_container_width=True, key="dnav_cases"):
            st.session_state['doctor_page'] = 'cases'
            st.rerun()

        appt_label = f"Appointments  +{pending_appts} new" if pending_appts > 0 else "Appointments"
        if st.button(appt_label,
                     type=("primary" if st.session_state['doctor_page']=='appointments' else "secondary"),
                     use_container_width=True, key="dnav_appt"):
            st.session_state['doctor_page'] = 'appointments'
            st.rerun()

        msg_label = f"Messages  +{unread_msgs} unread" if unread_msgs > 0 else "Messages"
        if st.button(msg_label,
                     type=("primary" if st.session_state['doctor_page']=='messages' else "secondary"),
                     use_container_width=True, key="dnav_msg"):
            st.session_state['doctor_page'] = 'messages'
            st.rerun()

        st.markdown('<div class="sidebar-nav-label">Dashboard</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Cases", total_cases)
        c2.metric("Pending", pending_cases)
        c3, c4 = st.columns(2)
        c3.metric("Appts", len(all_appts))
        c4.metric("Messages", unread_msgs)

        st.divider()
        dark_label = "☀ Light Mode" if st.session_state['dark_mode'] else "◑ Dark Mode"
        if st.button(dark_label, use_container_width=True, key="dnav_theme"):
            st.session_state['dark_mode'] = not st.session_state['dark_mode']
            st.rerun()
        if st.button("Sign Out", use_container_width=True, key="dnav_so"):
            keys_to_clear = [k for k in st.session_state.keys() if k not in ['dark_mode']]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()

        st.write("")
        st.markdown("""
        <div class="info-banner">
          <strong>&#x1F512; Consent &amp; Terms</strong><br>
          Results are AI-assisted only. All diagnoses must be confirmed by a licensed professional.
          Data is securely encrypted. Not for emergency use.
        </div>
        """, unsafe_allow_html=True)

def _clear_session():
    """Clears all session state on logout to prevent data leakage."""
    keys_to_clear = [k for k in st.session_state.keys()
                     if k != 'dark_mode']
    for k in keys_to_clear:
        del st.session_state[k]

# ── Results renderer ───────────────────────────────────────────
def render_my_results(my_cases):
    if not my_cases:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">N</div>
            <div class="empty-title">No screenings yet</div>
            <div class="empty-sub">Complete a screening to see
            your results here</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("New Screening", type="primary",
                     key="empty_go_screen"):
            st.session_state['page']           = 'screening'
            st.session_state['screening_step'] = 1
            st.rerun()
        return

    total    = len(my_cases)
    reviewed = sum(1 for c in my_cases
                   if c['status']=='Reviewed')
    pending  = total - reviewed
    high     = sum(1 for c in my_cases
                   if 'High' in c['risk_level']
                   or 'specialist' in c['risk_level'])

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Scans", total)
    m2.metric("Reviewed",    reviewed)
    m3.metric("Pending",     pending)
    m4.metric("Urgent",      high)
    st.write("")

    if total > 1:
        st.markdown('<div class="section-label">'
                    'Risk Trend</div>',
                    unsafe_allow_html=True)
        risk_scores, timestamps = [], []
        for c in my_cases:
            r = c['risk_level'].lower()
            risk_scores.append(
                3 if ('high' in r or 'specialist' in r)
                else 2 if ('moderate' in r or 'follow' in r)
                else 1
            )
            timestamps.append(c['timestamp'][:6])
        dark = st.session_state.get('dark_mode', True)
        bg   = '#13131f' if dark else '#ffffff'
        tc   = '#94a3b8' if dark else '#2d6a4f'
        fig, ax = plt.subplots(figsize=(8, 2.4))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        clrs = {1:'#0d9488', 2:'#d97706', 3:'#dc2626'}
        ax.plot(timestamps, risk_scores,
                color='#3b82f6', linewidth=2, zorder=1, alpha=0.5)
        ax.scatter(timestamps, risk_scores,
                   c=[clrs[s] for s in risk_scores],
                   s=80, zorder=2)
        ax.set_yticks([1,2,3])
        ax.set_yticklabels(['Low','Moderate','High'],
                           color=tc, fontsize=9)
        ax.tick_params(axis='x', colors=tc, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#b7e4c7')
        ax.set_ylim(0.5, 3.5)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close()
        st.write("")

    for case in reversed(my_cases):
        status   = case['status']
        risk     = case['risk_level']
        icon     = "[Reviewed]" if status=="Reviewed" else "[Pending]"
        ri       = ("[High]" if "High" in risk
                    or "specialist" in risk
                    else "[Moderate]" if "Moderate" in risk
                    or "follow" in risk.lower()
                    else "[Low]")
        msgs     = load_messages(case['case_id'])
        doc_msgs = [m for m in msgs
                    if m['sender_role']=='doctor']
        badge    = (f" [{len(doc_msgs)} msg(s)]"
                    if doc_msgs else "")

        with st.expander(
            f"{icon} {case['case_id']} — "
            f"{case['timestamp']} — {ri} {status}{badge}",
            expanded=case['case_id']==st.session_state.get(
                'last_case_id')
        ):
            c1,c2 = st.columns(2)
            with c1:
                if (case.get('image_path') and
                        os.path.exists(case['image_path'])):
                    st.image(
                        Image.open(case['image_path']),
                        caption="Retinal scan",
                        width='stretch')
                if (case.get('heatmap_path') and
                        os.path.exists(case['heatmap_path'])):
                    st.image(
                        Image.open(case['heatmap_path']),
                        caption="AI heatmap",
                        width='stretch')
            with c2:
                st.markdown("**What the AI found**")
                probs = case['probs']
                for i,(name,p) in enumerate(
                    zip(DISEASE_NAMES,probs)
                ):
                    if p > 0.3:
                        pc1,pc2 = st.columns([3,1])
                        pc1.progress(float(p), text=name)
                        if p>0.7:   pc2.error(f"{p*100:.0f}%")
                        elif p>0.5: pc2.warning(f"{p*100:.0f}%")
                        else:       pc2.info(f"{p*100:.0f}%")
                if "High" in risk or "specialist" in risk:
                    st.markdown('<div class="risk-pill risk-high" style="margin-top:8px;"></div>', unsafe_allow_html=True)
                    st.error(f"▲ High Risk — {risk}")
                elif "Moderate" in risk or "follow" in risk.lower():
                    st.markdown('<div class="risk-pill risk-moderate" style="margin-top:8px;"></div>', unsafe_allow_html=True)
                    st.warning(f"◆ Moderate — {risk}")
                else:
                    st.markdown('<div class="risk-pill risk-low" style="margin-top:8px;"></div>', unsafe_allow_html=True)
                    st.success(f"● Low Risk — {risk}")

            st.divider()
            st.markdown("**Your doctor said:**")
            if status != "Reviewed":
                st.info("Awaiting review — check back soon")
            else:
                st.success(
                    f"Reviewed: {case.get('reviewed_at','')}")
                for lbl, val in [
                    ("Diagnosis",  case['doctor_diagnosis']),
                    ("Treatment",
                     case['doctor_prescription'] or "None given"),
                ]:
                    r1,r2 = st.columns([1,3])
                    r1.markdown(f"**{lbl}**")
                    r2.info(val)
                r1,r2 = st.columns([1,3])
                r1.markdown("**Next Step**")
                ref = case['doctor_referral']
                if "Emergency" in ref or "Urgent" in ref:
                    r2.error(f"URGENT: {ref}")
                elif "month" in ref:
                    r2.warning(f"Follow-up: {ref}")
                else:
                    r2.success(ref)
                if case['doctor_notes']:
                    r1,r2 = st.columns([1,3])
                    r1.markdown("**Notes**")
                    r2.info(case['doctor_notes'])
                # ── Pharmacy & Prescription ────────────────
                st.write("")
                st.markdown("**Find Nearby Pharmacies**")
                pharm_city = st.text_input(
                    "Enter your city",
                    placeholder="e.g. Bengaluru",
                    key=f"pharm_city_{case['case_id']}")
                if pharm_city:
                    purl = f"https://www.google.com/maps/search/pharmacy+near+{pharm_city.replace(' ', '+')}"
                    st.markdown(
                        f'<a href="{purl}" target="_blank">'
                        f'<button style="background:#0d9488;color:white;border:none;'
                        f'border-radius:8px;padding:9px 16px;width:100%;cursor:pointer;'
                        f'font-weight:600;font-family:\'Inter\',sans-serif;font-size:13px;">'
                        f'&#128205; Find Pharmacies near {pharm_city}</button></a>',
                        unsafe_allow_html=True)

                st.write("")
                st.markdown("**🛒 Order Medicines Online**")
                c1, c2, c3 = st.columns(3)
                c1.markdown(
                    '<a href="https://pharmeasy.in" target="_blank">'
                    '<button style="background:#0d9488;color:white;border:none;'
                    'border-radius:8px;padding:10px;width:100%;cursor:pointer;'
                    'font-weight:600;font-family:\'Inter\',sans-serif;font-size:13px;">'
                    'PharmEasy</button></a>',
                    unsafe_allow_html=True)
                c2.markdown(
                    '<a href="https://www.1mg.com" target="_blank">'
                    '<button style="background:#2563eb;color:white;border:none;'
                    'border-radius:8px;padding:10px;width:100%;cursor:pointer;'
                    'font-weight:600;font-family:\'Inter\',sans-serif;font-size:13px;">'
                    '1mg</button></a>',
                    unsafe_allow_html=True)
                c3.markdown(
                    '<a href="https://www.netmeds.com" target="_blank">'
                    '<button style="background:#0f766e;color:white;border:none;'
                    'border-radius:8px;padding:10px;width:100%;cursor:pointer;'
                    'font-weight:600;font-family:\'Inter\',sans-serif;font-size:13px;">'
                    'Netmeds</button></a>',
                    unsafe_allow_html=True)

                rx_pdf_key = f"rx_pdf_{case['case_id']}"
                if rx_pdf_key not in st.session_state:
                    if st.button("⬇️ Generate Prescription PDF", key=f"gen_rx_{case['case_id']}", use_container_width=True):
                        with st.spinner("Generating prescription..."):
                            from report_generator import generate_prescription_pdf
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                pdf_path = generate_prescription_pdf(
                                    patient_name=case['patient_name'],
                                    patient_age=case['patient_age'],
                                    patient_gender=case['patient_gender'],
                                    case_id=case['case_id'],
                                    doctor_diagnosis=case['doctor_diagnosis'],
                                    doctor_prescription=case['doctor_prescription'],
                                    doctor_referral=case['doctor_referral'],
                                    doctor_notes=case['doctor_notes'],
                                    risk_level=case['risk_level'],
                                    reviewed_at=case.get('reviewed_at', ''),
                                    output_path=tmp.name
                                )
                            with open(pdf_path, 'rb') as f:
                                st.session_state[rx_pdf_key] = f.read()
                        st.rerun()
                else:
                    st.download_button(
                        "⬇️ Download Prescription PDF",
                        data=st.session_state[rx_pdf_key],
                        file_name=f"prescription_{case['case_id']}.pdf",
                        mime="application/pdf",
                        key=f"rx_{case['case_id']}",
                        use_container_width=True)

            st.divider()
            render_chat(
                case['case_id'], 'patient',
                st.session_state['patient_user']['name']
            )

            # Download report
            st.write("")
            pdf_key = f"pdf_data_{case['case_id']}"
            if pdf_key not in st.session_state:
                if st.button("Download Report",
                             key=f"dl_{case['case_id']}",
                             use_container_width=True):
                    with st.spinner("Generating..."):
                        case_probs = np.array(
                            case.get('probs', [0]*8))
                        case_det   = case.get(
                            'detected_conditions', [])
                        msgs       = load_messages(
                            case['case_id'])
                        from database import get_patient_visits
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix='.pdf'
                        ) as tmp:
                            pdf_path = generate_report(
                                patient_name=case['patient_name'],
                                patient_age=case['patient_age'],
                                patient_gender=case['patient_gender'],
                                patient_email=case.get('patient_email',''),
                                symptoms=case['symptoms'],
                                quality_score=case['quality_score'],
                                quality_tips=[],
                                probs=case_probs,
                                detected_conditions=case_det,
                                risk_level=case['risk_level'],
                                risk_type=(
                                    'high' if 'High' in case['risk_level']
                                    or 'specialist' in case['risk_level']
                                    else 'moderate' if 'Moderate' in case['risk_level']
                                    else 'low'
                                ),
                                original_image_pil=Image.open(
                                    case['image_path'])
                                    if case.get('image_path') and
                                    os.path.exists(case['image_path'])
                                    else None,
                                heatmap_array=np.array(
                                    Image.open(case['heatmap_path']))
                                    if case.get('heatmap_path') and
                                    os.path.exists(case['heatmap_path'])
                                    else None,
                                doctor_name=case.get(
                                    'doctor_diagnosis','')[:30]
                                    if case.get('doctor_diagnosis')
                                    else None,
                                doctor_diagnosis=case.get(
                                    'doctor_diagnosis'),
                                doctor_prescription=case.get(
                                    'doctor_prescription'),
                                doctor_referral=case.get(
                                    'doctor_referral'),
                                doctor_notes=case.get('doctor_notes'),
                                reviewed_at=case.get('reviewed_at'),
                                chat_messages=msgs,
                                visit_history=get_patient_visits(
                                    case.get('patient_email','')),
                                output_path=tmp.name
                            )
                        with open(pdf_path, 'rb') as f:
                            st.session_state[pdf_key] = f.read()
                    st.rerun()
            else:
                st.download_button(
                    "Download Report",
                    data=st.session_state[pdf_key],
                    file_name=f"nayana_{case['patient_name'].replace(' ','_')}_{case['case_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"pdf_{case['case_id']}"
                )
                if st.button("Regenerate",
                             key=f"regen_{case['case_id']}"):
                    del st.session_state[pdf_key]
                    st.rerun()

# ══════════════════════════════════════════════════════════════
# LANDING
# ══════════════════════════════════════════════════════════════
if st.session_state['role'] is None:
    st.markdown("""
    <div class="nayana-hero">
        <div class="nayana-wordmark">na<span>ya</span>na</div>
        <div class="nayana-meaning">the eye &nbsp;&middot;&nbsp; AI-assisted tele-ophthalmology</div>
        <div class="nayana-tagline">
            Free AI eye screening — get results in under 3 minutes
            and connect with a specialist from anywhere.
        </div>
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-num">8</div>
                <div class="stat-lbl">conditions screened</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">6,392</div>
                <div class="stat-lbl">training cases</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">5</div>
                <div class="stat-lbl">languages</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">&lt;3 min</div>
                <div class="stat-lbl">to results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _,c1,c2,c3,_ = st.columns([0.5,1,1,1,0.5])
    with c1:
        st.markdown("""
        <div class="portal-card">
            <div class="portal-icon">&#128065;</div>
            <div class="portal-title">Patient</div>
            <div class="portal-sub">Screen your eyes and receive
            expert feedback from a doctor.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Screening", type="primary",
                     use_container_width=True,
                     key="go_patient"):
            st.session_state['role'] = 'patient'
            st.rerun()
    with c2:
        st.markdown("""
        <div class="portal-card doctor">
            <div class="portal-icon">&#129657;</div>
            <div class="portal-title">Doctor</div>
            <div class="portal-sub">Review AI-assisted cases and
            consult patients remotely.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Review Cases",
                     use_container_width=True,
                     key="go_doctor"):
            st.session_state['role'] = 'doctor'
            st.rerun()
    with c3:
        st.markdown("""
        <div class="portal-card admin">
            <div class="portal-icon">&#128737;</div>
            <div class="portal-title">Admin</div>
            <div class="portal-sub">Verify doctor registrations
            and manage the platform.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Admin Login",
                     use_container_width=True,
                     key="go_admin"):
            st.session_state['role'] = 'admin'
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PATIENT PORTAL
# ══════════════════════════════════════════════════════════════
elif st.session_state['role'] == 'patient':

    if not st.session_state['patient_logged_in']:
        _,col,_ = st.columns([1,1.6,1])
        with col:
            st.markdown(
                '<div class="page-title" '
                'style="text-align:center;">nayana</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="page-sub" '
                'style="text-align:center;">'
                'Sign in or create a free account</div>',
                unsafe_allow_html=True)
            tab1,tab2 = st.tabs(["Sign In","Create Account"])

            with tab1:
                st.write("")
                email = st.text_input(
                    "Email address", key="li_e",
                    placeholder="you@example.com")
                pw    = st.text_input(
                    "Password", type="password", key="li_p",
                    placeholder="Your password")
                st.write("")
                if st.button("Sign In", type="primary",
                             key="li_btn",
                             use_container_width=True):
                    if email and pw:
                        ok,user,msg = login_patient(email, pw)
                        if ok:
                            st.session_state[
                                'patient_logged_in'] = True
                            st.session_state[
                                'patient_user'] = user
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("Please fill in both fields")
                st.write("")
                if st.button("Back", key="back_p",
                             use_container_width=True):
                    st.session_state['role'] = None
                    st.rerun()

            with tab2:
                st.write("")
                c1,c2 = st.columns(2)
                rn  = c1.text_input("Your name",  key="rn")
                ra  = c2.number_input("Age",1,120,30, key="ra")
                rg  = st.selectbox("Gender",
                                    ["Male","Female","Other"],
                                    key="rg")
                re  = st.text_input("Email", key="re",
                                     placeholder="you@example.com")
                c1,c2 = st.columns(2)
                rp  = c1.text_input("Password",
                                     type="password", key="rp")
                rp2 = c2.text_input("Confirm password",
                                     type="password", key="rp2")
                if rp:
                    if len(rp) < 6:
                        st.error("Too short — min 6 characters")
                    elif len(rp) < 8:
                        st.warning("Weak — add more characters")
                    elif not any(c.isdigit() for c in rp):
                        st.warning("Medium — add a number")
                    elif not any(c.isupper() for c in rp):
                        st.warning("Medium — add an uppercase letter")
                    else:
                        st.success("Strong password!")
                st.write("")
                if st.button("Create Account", type="primary",
                             key="r_btn",
                             use_container_width=True):
                    if not all([rn,re,rp,rp2]):
                        st.error("Please fill in all fields")
                    elif rp != rp2:
                        st.error("Passwords don't match")
                    elif len(rp) < 6:
                        st.error("Min 6 characters")
                    else:
                        ok,msg = register_patient(rn,ra,rg,re,rp)
                        if ok:
                            st.success(
                                "Account created! Sign in now.")
                        else:
                            st.error(msg)
    else:
        user = st.session_state['patient_user']
        patient_navbar(user)
        # Urgent attention banner
        all_cases = load_cases()
        my_cases  = [c for c in all_cases if c.get('patient_email','') == user['email']]
        high_risk = [c for c in my_cases if 'High' in c['risk_level'] or 'specialist' in c['risk_level']]
        new_reviews = sum(1 for c in my_cases if c['status'] == 'Reviewed')
        if high_risk:
            st.error(f"URGENT: You have {len(high_risk)} high-risk screening(s) that need specialist attention. Go to Results to view.")
        elif new_reviews:
            st.success(f"Your doctor has reviewed {new_reviews} case(s). Check your Results for updates.")
        my_appts = [a for a in load_appointments() if a['patient_email'] == user['email']]
        confirmed_appts = [a for a in my_appts if a['status'] == 'Confirmed' and a.get('meet_link')]
        if confirmed_appts:
            for appt in confirmed_appts:
                st.success(f"Appointment confirmed — {appt['date']} at {appt['time_slot']} | [Join Google Meet]({appt['meet_link']})")

        # ── Screening ──────────────────────────────────────────
        if st.session_state['page'] == 'screening':
            step_bar(st.session_state.get('screening_step', 1))

            # STEP 1: Conversational chatbot screening flow
            if st.session_state['screening_step'] == 1:
                render_chatbot_screening(user)

            # STEP 2: Eye Photos
            elif st.session_state['screening_step'] == 2:
                st.markdown(
                    '<div class="page-title">'
                    'Take Your Eye Photos</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div class="page-sub">Upload a front eye '
                    'photo and/or a retinal scan</div>',
                    unsafe_allow_html=True)

                triage_result = st.session_state.get('triage') or {}
                triage_type   = triage_result.get('type', 'front')
                triage_reason = triage_result.get('reason', '')
                if triage_type == 'fundus':
                    st.warning(
                        "Based on your symptoms, we recommend "
                        f"a retinal scan. ({triage_reason})")
                else:
                    st.success(
                        "A front-eye photo may be enough. "
                        "You can also add a retinal scan below.")

                # Front eye
                st.markdown(
                    '<div class="page-title" '
                    'style="font-size:22px;margin-top:16px;">'
                    'Front Eye Photo</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div class="page-sub">Close-up of your '
                    'eye in good lighting</div>',
                    unsafe_allow_html=True)

                front_up = None
                up_tab, cam_tab = st.tabs([
                    "Upload Photo","Use Camera"])
                with up_tab:
                    front_up_file = st.file_uploader(
                        "Choose front eye photo",
                        type=["jpg","jpeg","png"],
                        key="front_eye_upload")
                    if front_up_file:
                        front_up = front_up_file
                with cam_tab:
                    if not st.session_state.get(
                        'show_front_cam', False
                    ):
                        st.info("Click below to activate "
                                "your camera")
                        if st.button("Activate Camera",
                                     key="activate_front_cam",
                                     type="primary"):
                            st.session_state[
                                'show_front_cam'] = True
                            st.rerun()
                    else:
                        front_up_cam = st.camera_input(
                            "Point your eye at the camera",
                            key="front_camera")
                        if front_up_cam:
                            raw_pil = Image.open(front_up_cam).convert('RGB')
                            cleaned_pil, found, annotated_pil = capture_and_detect_eye(raw_pil)
                            if not found:
                                st.warning("No eye detected — centre your eye and retake")
                            else:
                                c1, c2 = st.columns(2)
                                c1.image(annotated_pil, caption="Eye detected", use_container_width=True)
                                c2.image(cleaned_pil, caption="AI cleaned view", use_container_width=True)
                                st.success("Eye detected and cleaned — proceeding to analysis")
                                import io
                                buf = io.BytesIO()
                                cleaned_pil.save(buf, format='PNG')
                                buf.seek(0)
                                front_up = buf
                                st.session_state['front_pil'] = cleaned_pil
                        if st.button("Turn off camera",
                                     key="off_front_cam"):
                            st.session_state[
                                'show_front_cam'] = False
                            st.rerun()

                if front_up:
                    front_pil = Image.open(
                        front_up).convert('RGB')
                    st.session_state['front_pil'] = front_pil
                    fc1,fc2 = st.columns([1,1.6])
                    with fc1:
                        st.image(front_pil,
                                 caption="Your eye",
                                 use_container_width=True)
                    with fc2:
                        with st.spinner(
                            "Checking your eye..."
                        ):
                            fe_res = analyze_front_eye(front_pil)
                            recs, high_risk, needs_fundus = \
                                get_front_eye_recommendations(fe_res)
                        st.session_state['fe_results'] = fe_res
                        st.session_state['fe_recs']    = recs
                        st.markdown("**Quick findings:**")
                        for cond, score in sorted(
                            fe_res.items(),
                            key=lambda x: x[1], reverse=True
                        ):
                            level = ("High" if score > 0.6
                                     else "Mod" if score > 0.3
                                     else "Low")
                            level_icon = ("▲" if score > 0.6
                                          else "◆" if score > 0.3
                                          else "●")
                            bar_pct = int(score * 100)
                            st.markdown(f"""
                            <div style="display:flex;
                                justify-content:space-between;
                                align-items:center;
                                padding:9px 0;
                                border-bottom:1px solid rgba(59,130,246,0.1);">
                                <span style="font-size:13px;font-weight:600;flex:1;">{cond}</span>
                                <span style="font-size:12px;font-weight:700;
                                    opacity:0.7;margin-right:10px;">{level_icon} {level}</span>
                                <span style="font-size:13px;font-weight:800;
                                    font-family:'Space Mono',monospace;min-width:40px;text-align:right;">
                                    {bar_pct}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                    if recs:
                        for rec in recs:
                            st.warning(rec)
                    if needs_fundus:
                        st.error(
                            "Retinal scan recommended "
                            "based on findings.")

                st.write("")
                st.divider()

                # Fundus
                st.markdown(
                    '<div class="page-title" '
                    'style="font-size:22px;">'
                    'Retinal Scan (Fundus)</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div class="page-sub">For a deeper look '
                    '— upload if you have one</div>',
                    unsafe_allow_html=True)

                # New Dongle Selection Logic
                st.markdown('<div class="page-sub" style="font-weight:700; color:#3b82f6; margin-bottom:10px;">'
                            'Do you have our Nayana Fundus Dongle?</div>', unsafe_allow_html=True)
                
                dongle_opt = st.radio(
                    "Dongle Selection",
                    ["Yes, I have it", "No, manual upload"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="dongle_selection"
                )

                if dongle_opt == "Yes, I have it":
                    st.markdown("""
                    <div style="background:rgba(59,130,246,0.1); padding:15px; border-radius:10px; border-left:4px solid #3b82f6; margin-bottom:10px;">
                        <p style="font-size:14px; margin-bottom:5px;"><strong>How to use your Dongle:</strong></p>
                        <ol style="font-size:13px; padding-left:15px;">
                            <li>Attach the <b>Nayana Dongle</b> to your phone's primary camera lens.</li>
                            <li>Open your phone's <b>native camera app</b> (not this browser).</li>
                            <li>Position the dongle near the eye and capture a clear fundus image.</li>
                            <li>Ensure the files are saved to your device, then upload below.</li>
                        </ol>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:rgba(148,163,184,0.1); padding:15px; border-radius:10px; margin-bottom:10px;">
                        <p style="font-size:13px; color:#64748b;">Please upload a pre-captured retinal fundus scan from your device gallery for AI analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)

                uploaded = st.file_uploader(
                    "Upload fundus image",
                    type=["jpg","jpeg","png"],
                    key="fundus_upload",
                    label_visibility="collapsed"
                )

                if uploaded:
                    image_pil = Image.open(
                        uploaded).convert('RGB')
                    st.session_state['fundus_pil'] = image_pil
                    c1,c2 = st.columns([1,1.6])
                    with c1:
                        st.image(image_pil,
                                 caption="Fundus image",
                                 use_container_width=True)
                    with c2:
                        score, tips = check_quality(
                            np.array(image_pil))
                        qc = ("#2d9e6b" if score>=70
                              else "#f4a261" if score>=40
                              else "#e63946")
                        ql = ("Great!" if score>=70
                              else "Okay" if score>=40
                              else "Poor")
                        st.markdown(f"""
                        <div class="card">
                            <div class="section-label">
                                Photo quality</div>
                            <div class="quality-num"
                                 style="color:{qc};">
                                {score}%</div>
                            <div style="font-size:14px;">
                                {ql}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        for tip in tips:
                            st.warning(tip)
                    st.session_state['fundus_score'] = score
                    st.session_state['fundus_tips']  = tips

                st.write("")
                has_any = (
                    st.session_state.get('front_pil') is not None
                    or st.session_state.get('fundus_pil') is not None
                )
                if has_any:
                    if st.button("See My Results",
                                 type="primary",
                                 use_container_width=True):
                        st.session_state['screening_step'] = 3
                        st.rerun()
                else:
                    st.info("Upload at least one photo to continue.")

                st.write("")
                if st.button("Back to Symptoms",
                             use_container_width=True):
                    st.session_state['screening_step'] = 1
                    st.rerun()

            # STEP 3: Results
            elif st.session_state['screening_step'] == 3:
                pname      = st.session_state.get(
                    'pname', user['name'])
                page_      = st.session_state.get(
                    'page_', user['age'])
                pgender    = st.session_state.get(
                    'pgender', user['gender'])
                symp_final = st.session_state.get(
                    'symp_final', 'Not specified')

                # Bundle: enrich with voice raw text and triage reason
                voice_mem    = st.session_state.get('voice_memory')
                triage_data  = st.session_state.get('triage') or {}
                triage_reason = triage_data.get('reason', '')
                if voice_mem and voice_mem.get('english_text'):
                    raw = voice_mem['english_text']
                    if raw not in symp_final:
                        symp_final = f"{symp_final} | Voice: {raw}"
                if triage_reason:
                    symp_final = f"{symp_final} | Triage: {triage_reason}"
                front_pil  = st.session_state.get('front_pil')
                fundus_pil = st.session_state.get('fundus_pil')
                score      = st.session_state.get(
                    'fundus_score', 0)
                tips       = st.session_state.get(
                    'fundus_tips', [])
                fe_results = st.session_state.get(
                    'fe_results', {})
                fe_recs    = st.session_state.get('fe_recs', [])

                st.markdown(
                    '<div class="page-title">Your Results</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    '<div class="page-sub">'
                    'Here\'s what our AI found</div>',
                    unsafe_allow_html=True)

                # Front eye results
                if front_pil and fe_results:
                    st.markdown("### Front Eye Analysis")
                    fc1,fc2 = st.columns([1,2])
                    with fc1:
                        st.image(front_pil,
                                 caption="Your eye",
                                 use_container_width=True)
                    with fc2:
                        for cond, s in sorted(
                            fe_results.items(),
                            key=lambda x: x[1], reverse=True
                        ):
                            level = ("High" if s > 0.6
                                     else "Mod" if s > 0.3
                                     else "Low")
                            level_icon = ("▲" if s > 0.6
                                          else "◆" if s > 0.3
                                          else "●")
                            st.markdown(f"""
                            <div style="display:flex;
                                justify-content:space-between;
                                align-items:center;
                                padding:9px 0;
                                border-bottom:1px solid rgba(59,130,246,0.1);">
                                <span style="font-size:13px;font-weight:600;flex:1;">{cond}</span>
                                <span style="font-size:12px;font-weight:700;
                                    opacity:0.65;margin-right:10px;">{level_icon} {level}</span>
                                <span style="font-size:13px;font-weight:800;
                                    font-family:'Space Mono',monospace;min-width:40px;text-align:right;">
                                    {s*100:.0f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                    if fe_recs:
                        for rec in fe_recs:
                            st.warning(rec)
                    st.write("")

                # Fundus results
                probs     = None
                heatmap   = None
                risk_txt  = "No retinal scan provided"
                risk_type = "low"
                det_conds = []

                if fundus_pil:
                    st.markdown("### Retinal Scan Analysis")
                    with st.spinner("Analysing your scan..."):
                        probs   = predict(fundus_pil)
                        heatmap = get_heatmap(fundus_pil)
                    st.session_state['probs']   = probs
                    st.session_state['heatmap'] = heatmap

                    fig = green_chart(probs)
                    st.pyplot(fig)
                    plt.close()

                    risk_txt, risk_type = get_risk(probs)
                    risk_css = {
                        "high":     "risk-high",
                        "moderate": "risk-moderate",
                        "low":      "risk-low"
                    }[risk_type]
                    det_conds = [
                        (DISEASE_NAMES[i], probs[i])
                        for i in range(8) if probs[i] > 0.5
                    ]
                    card_cls = (
                        "highlight" if risk_type=='low'
                        else "warning" if risk_type=='moderate'
                        else "danger"
                    )
                    st.markdown(f"""
                    <div class="card {card_cls}">
                        <div style="display:flex;
                            align-items:center;
                            justify-content:space-between;
                            margin-bottom:8px;">
                            <div style="font-size:16px;
                                font-weight:800;">
                                Overall Result</div>
                            <span class="risk-pill {risk_css}">
                                {risk_type.title()} Risk</span>
                        </div>
                        <div style="font-size:15px;">
                            {risk_txt}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    for name,p in sorted(
                        det_conds, key=lambda x: x[1],
                        reverse=True
                    ):
                        if name == 'Normal':
                            st.success(
                                f"Normal — {p*100:.1f}%")
                        elif p > 0.7:
                            st.error(
                                f"Detected: {name} — "
                                f"{p*100:.1f}%")
                        else:
                            st.warning(
                                f"Possible: {name} — "
                                f"{p*100:.1f}%")

                    # Disease info cards
                    non_normal = [(n,p) for n,p in det_conds
                                  if n != 'Normal']
                    if non_normal:
                        st.divider()
                        st.markdown(
                            '<div class="page-title" '
                            'style="font-size:20px;">'
                            'What These Findings Mean</div>',
                            unsafe_allow_html=True)
                        for dname, dp in sorted(
                            non_normal,
                            key=lambda x: x[1], reverse=True
                        ):
                            info = DISEASE_INFO.get(dname)
                            if not info:
                                continue
                            with st.expander(
                                f"{dname} — What you need "
                                f"to know",
                                expanded=dp > 0.6
                            ):
                                st.markdown(f"""
<div style="padding:16px 20px;border-radius:12px;
background:rgba(99,102,241,0.06);
border:1px solid rgba(99,102,241,0.2);">
<b>What is it?</b><br>
<span style="color:#94a3b8;">{info['what']}</span><br><br>
<b>Symptoms to watch for:</b><br>
<span style="color:#94a3b8;">{info['symptoms']}</span><br><br>
<b>What to do:</b>
<span style="font-weight:700;color:{'#fca5a5' if info['serious'] else '#6ee7b7'};">
{info['urgency']}</span><br><br>
<i style="color:#64748b;">Tip: {info['tip']}</i>
</div>""", unsafe_allow_html=True)

                    if risk_type in ['high','moderate']:
                        st.divider()
                        maps_url = ("https://www.google.com/"
                                    "maps/search/"
                                    "ophthalmologist+near+me")
                        st.markdown(
                            f'<a href="{maps_url}" '
                            f'target="_blank">'
                            f'<button style="background:'
                            f'linear-gradient(135deg,'
                            f'#6366f1,#38bdf8);color:white;'
                            f'border:none;border-radius:12px;'
                            f'padding:11px 20px;width:100%;'
                            f'font-weight:700;cursor:pointer;'
                            f'font-size:14px;">'
                            f'Find Eye Clinics Near Me'
                            f'</button></a>',
                            unsafe_allow_html=True)
                        st.caption(
                            "Sankara Nethralaya · "
                            "Narayana Nethralaya · "
                            "LV Prasad Eye Institute · "
                            "Aravind Eye Hospital")
                        st.write("")

                    st.write("")
                    st.markdown("### Where the AI looked")
                    hc1,hc2 = st.columns(2)
                    hc1.image(fundus_pil.resize((224,224)),
                              caption="Your scan", width=220)
                    hc2.image(heatmap,
                              caption="AI focus", width=220)

                st.divider()
                st.markdown("### What would you like to do?")
                ac1,ac2 = st.columns(2)
                st.divider()
                st.markdown("### How would you like to connect with a doctor?")
                connect_choice = st.radio(
                    "Choose your preference",
                    ["Online — connect with a registered doctor", "Offline — find nearby hospitals"],
                    horizontal=True,
                    key="connect_choice"
                )
                st.write("")

                if connect_choice == "Offline — find nearby hospitals":
                    city = st.text_input("Enter your city", placeholder="e.g. Bengaluru, Chennai, Hyderabad", key="city_input")
                    if city:
                        maps_url = f"https://www.google.com/maps/search/eye+hospital+near+{city.replace(' ','+')}"
                        st.markdown(f'<a href="{maps_url}" target="_blank"><button style="background:#2d9e6b;color:white;border:none;border-radius:10px;padding:10px 20px;width:100%;font-weight:700;cursor:pointer;font-size:14px;">Find Eye Hospitals near {city}</button></a>', unsafe_allow_html=True)
                    else:
                        st.info("Enter your city to find nearby eye hospitals.")

                else:
                    doctors = get_all_doctors()
                    if not doctors:
                        st.warning("No doctors registered yet — check back soon.")
                    else:
                        st.markdown("**Registered Doctors on Nayana**")
                        for d in doctors:
                            initials = ''.join(p[0].upper() for p in d['name'].split()[:2])
                            st.markdown(f"""
                            <div style="display:flex;align-items:flex-start;gap:14px;
                                        background:var(--card-bg,rgba(30,37,53,1));
                                        border:1px solid rgba(59,130,246,0.15);
                                        border-left:3px solid #0d9488;
                                        border-radius:10px;padding:14px;margin-bottom:10px;">
                                <div class="doc-avatar">{initials}</div>
                                <div style="flex:1;">
                                  <div style="font-size:15px;font-weight:700;">Dr. {d['name']}</div>
                                  <div style="font-size:12px;margin-top:2px;opacity:0.7;">{d.get('specialization','Ophthalmologist')} &nbsp;&middot;&nbsp; {d.get('hospital','')}</div>
                                  <div style="font-size:11px;margin-top:4px;opacity:0.5;">{d.get('email','')}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    st.write("")
                    st.markdown("### Book an Appointment")
                    ac1,ac2 = st.columns(2)
                    with ac1:
                        doctors = get_all_doctors()
                        if not doctors:
                            st.warning("No doctors registered yet — check back soon.")
                        else:
                            doctor_options = {
                                f"Dr. {d['name']} — {d['specialization']} ({d['hospital']})": d
                                for d in doctors
                            }
                            selected_doc_label = st.selectbox(
                                "Choose a doctor",
                                list(doctor_options.keys()),
                                key="appt_doc"
                            )
                            selected_doc = doctor_options[selected_doc_label]
                            import datetime as dt
                            today = dt.date.today()
                            appt_date = st.date_input(
                                "Select date",
                                min_value=today,
                                max_value=today + dt.timedelta(days=30),
                                key="appt_date"
                            )

                            # ── Dynamic slot availability ──────
                            available_slots = get_available_slots(
                                selected_doc['email'], str(appt_date))
                            booked_slots = get_booked_slots(
                                selected_doc['email'], str(appt_date))

                            if not available_slots:
                                st.error(
                                    f"Dr. {selected_doc['name']} is fully booked "
                                    f"on {appt_date}. Please choose another date.")
                                time_slot = None
                            else:
                                # Show slot count info
                                total_slots = len(ALL_TIME_SLOTS)
                                taken = len(booked_slots)
                                if taken > 0:
                                    st.caption(
                                        f"{taken}/{total_slots} slots taken on "
                                        f"{appt_date} — "
                                        f"{len(available_slots)} available")
                                time_slot = st.selectbox(
                                    "Select time slot",
                                    available_slots,
                                    key="appt_time")

                            appt_notes = st.text_area(
                                "Notes for doctor (optional)",
                                placeholder="Mention any specific concerns...",
                                key="appt_notes")

                            if time_slot and st.button(
                                "Book Appointment & Send Case",
                                type="primary",
                                use_container_width=True,
                                key="book_appt"
                            ):
                                with st.spinner("Booking appointment..."):
                                    os.makedirs("cases_images", exist_ok=True)
                                    n = len(os.listdir("cases_images"))
                                    ip, hp = "", ""
                                    if fundus_pil and heatmap is not None:
                                        ip = f"cases_images/{pname.replace(' ','_')}_{n}_original.png"
                                        hp = f"cases_images/{pname.replace(' ','_')}_{n}_heatmap.png"
                                        fundus_pil.resize((300,300)).save(ip)
                                        Image.fromarray(heatmap).save(hp)
                                    if front_pil:
                                        fp = f"cases_images/{pname.replace(' ','_')}_{n}_front.png"
                                        front_pil.resize((300,300)).save(fp)
                                    fe_str = ""
                                    if fe_results:
                                        fe_str = " | Front-eye: " + " | ".join(
                                            f"{c}: {s*100:.0f}%" for c,s in sorted(
                                                fe_results.items(), key=lambda x: x[1], reverse=True))
                                    cid = save_case(
                                        patient_name=pname, patient_age=int(page_),
                                        patient_gender=pgender,
                                        symptoms=symp_final + fe_str,
                                        quality_score=score,
                                        probs=probs if probs is not None else np.zeros(8),
                                        detected_conditions=det_conds,
                                        risk_level=risk_txt,
                                        image_path=ip, heatmap_path=hp,
                                        patient_email=user['email'])
                                    appt_id, err = book_appointment(
                                        patient_email=user['email'],
                                        patient_name=pname,
                                        doctor_email=selected_doc['email'],
                                        doctor_name=selected_doc['name'],
                                        date=str(appt_date),
                                        time_slot=time_slot,
                                        case_id=cid,
                                        notes=appt_notes)
                                if err:
                                    st.error(err)
                                else:
                                    st.session_state['appt_success_details'] = {
                                        'doc_name': selected_doc['name'],
                                        'date': appt_date,
                                        'time_slot': time_slot,
                                        'appt_id': appt_id,
                                        'cid': cid
                                    }
                                    st.rerun()

                    with ac2:
                        if front_pil or (fundus_pil and probs is not None):
                            if st.button("Download Report", use_container_width=True, type="primary"):
                                with st.spinner("Generating report..."):
                                    fe_str = ""
                                    if fe_results:
                                        fe_str = "\n\nFront Eye Findings:\n" + "\n".join(
                                            f"  - {c}: {s*100:.0f}%" for c,s in sorted(fe_results.items(), key=lambda x: x[1], reverse=True))
                                        if fe_recs:
                                            fe_str += "\n\nFront Eye Recommendations:\n" + "\n".join(f"  - {r}" for r in fe_recs)
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                        pdf_path = generate_report(
                                            patient_name=pname, patient_age=int(page_),
                                            patient_gender=pgender, patient_email=user['email'],
                                            symptoms=symp_final + fe_str,
                                            quality_score=score, quality_tips=tips,
                                            probs=probs, detected_conditions=det_conds,
                                            risk_level=risk_txt, risk_type=risk_type,
                                            original_image_pil=fundus_pil,
                                            heatmap_array=heatmap,
                                            front_eye_image_pil=front_pil,
                                            front_eye_results=fe_results,
                                            front_eye_recommendations=fe_recs,
                                            triage_decision=st.session_state.get('triage', {}).get('type', 'front'),
                                            visit_history=get_patient_visits(user['email']),
                                            output_path=tmp.name)
                                with open(pdf_path,'rb') as f:
                                    st.download_button("Download PDF",
                                        data=f.read(),
                                        file_name=f"nayana_{pname.replace(' ','_')}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True)
                        else:
                            st.info("Complete a screening to download a report")

                st.write("")
                if st.button("Start a New Screening",
                             use_container_width=True):
                    for k in ['front_pil','fundus_pil',
                              'fe_results','fe_recs','probs',
                              'heatmap','fundus_score',
                              'fundus_tips']:
                        st.session_state[k] = None
                    st.session_state['show_front_cam']  = False
                    st.session_state['show_fundus_cam'] = False
                    st.session_state['screening_step']  = 1
                    st.rerun()

                st.write("")
                st.caption(
                    "Nayana is for screening only. "
                    "Always follow your doctor's advice.")

        # ── Results page ───────────────────────────────────────
        elif st.session_state['page'] == 'results':
            st.markdown(
                '<div class="page-title">My Results</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="page-sub">Your screening history '
                'and doctor responses</div>',
                unsafe_allow_html=True)
            all_cases = load_cases()
            my_cases  = [c for c in all_cases
                         if c.get('patient_email','') ==
                         user['email']]
            render_my_results(my_cases)

        # ── Health record page ─────────────────────────────────
        elif st.session_state['page'] == 'health_record':
            st.markdown(
                '<div class="page-title">'
                'My Health Record</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="page-sub">Your complete eye '
                'health file</div>',
                unsafe_allow_html=True)
            if st.session_state.get(
                'return_to_screening', False
            ):
                if st.button("Back to Screening",
                             type="primary",
                             key="back_to_screen"):
                    st.session_state[
                        'return_to_screening'] = False
                    st.session_state['page'] = 'screening'
                    st.rerun()
            render_patient_health_record(user)

        elif st.session_state['page'] == 'optical_scan':
            st.markdown('<div class="page-title">Eye Optical Health Scan</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub">Asha Unified Diagnostic System for Lens & Pupil Health.</div>', unsafe_allow_html=True)
            
            st.write("")
            
            # Info Cards
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div class="result-card" style="padding:20px;">
                    <h3 style="margin-top:0;color:#0d9488;">🔍 Lens Health Scan</h3>
                    <p style="font-size:14px;color:#64748b;">Analyze lens clarity, stability, and aging (yellowing). Uses median filtering and Hough circle detection.</p>
                    <div style="font-size:12px;background:rgba(13,148,136,0.1);padding:8px;border-radius:6px;">
                        <strong> </strong> 
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown("""
                <div class="result-card" style="padding:20px;">
                    <h3 style="margin-top:0;color:#3b82f6;">⚡ Pupil Reflex (PLR)</h3>
                    <p style="font-size:14px;color:#64748b;">Tests Pupillary Light Reflex by measuring constriction speed and percentage under light stimulation.</p>
                    <div style="font-size:12px;background:rgba(59,130,246,0.1);padding:8px;border-radius:6px;">
                        <strong> </strong> 
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            
            st.info("💡 **Neural Diagnostic:** This scanner uses high-precision **Neural Eye Tracking**. Align your eye in the central box until it turns green to perform a high-resolution Lens & Aging scan.")
            
            st.write("")
            if st.button("🚀 Start Diagnostic Scanner", type="primary", use_container_width=True):
                with st.spinner("Initializing camera system..."):
                    scanner_results = run_unified_scanner()
                    if scanner_results:
                        st.session_state['optical_results'] = scanner_results
                        st.success("Diagnostic scan completed successfully!")
                        st.rerun()
                    else:
                        st.warning("Scanner cancelled or closed without results.")

            if st.session_state.get('optical_results'):
                res = st.session_state['optical_results']
                st.divider()
                st.markdown('<div class="section-label">Scan Results & Clinical Advice</div>', unsafe_allow_html=True)
                
                # Clinical Logic
                status = "Optimal"
                advice = "Your lens health and pupillary reflex are within the youthful/healthy range. No immediate action required."
                interval = "Routine check in 6 months"
                clr = "#0d9488" # Teal
                
                # Check thresholds (Neural Clarity & Calibrated Aging)
                if res['clarity'] < 100:
                    status = "Critical"
                    advice = "Significant irregularities in lens sharpness detected. Avoid driving and consult an ophthalmologist immediately for a comprehensive exam."
                    interval = "Immediate Consultation Required"
                    clr = "#dc2626" # Red
                elif res['clarity'] < 160 or res['age_index'] > 1.25:
                    status = "Warning"
                    advice = "Moderate signs of lens maturity detected. Please re-scan in 48 hours for consistency. Consult an optometrist if symptoms persist."
                    interval = "Follow-up in 48 Hours"
                    clr = "#d97706" # Orange

                # Result Header
                st.markdown(f"""
                <div style="background:{clr}15; border:1px solid {clr}40; padding:20px; border-radius:12px; margin-bottom:20px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="margin:0; color:{clr};">Status: {status}</h2>
                        <span style="font-weight:700; color:{clr}; background:{clr}25; padding:4px 12px; border-radius:20px; font-size:14px;">{interval}</span>
                    </div>
                    <p style="margin:10px 0 0; font-size:15px; color:{'#94a3b8' if st.session_state['dark_mode'] else '#475569'};">{advice}</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([1, 1.2])
                with col1:
                    if res.get('heatmap_path') and os.path.exists(res['heatmap_path']):
                        st.image(res['heatmap_path'], caption="Optical Density Heatmap", use_container_width=True)
                    else:
                        st.caption("Heatmap unavailable")
                
                with col2:
                    st.markdown("**Metric Breakdown**")
                    
                    def metric_row(label, value, target, unit="", higher_is_better=False):
                        is_good = (value < target) if not higher_is_better else (value > target)
                        icon = "✅" if is_good else "⚠️"
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid rgba(148,163,184,0.1);">
                            <span style="font-size:14px;">{icon} {label}</span>
                            <span style="font-weight:700; font-size:14px;">{value:.1f}{unit}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    metric_row("Clarity (Sharpness)", res['clarity'], 160, "", True)
                    metric_row("Lens Stability", res['stability'], 8)
                    metric_row("Age Index (Calibrated)", res['age_index'], 1.25)

                st.write("")
                if st.button("Archive & Re-scan", use_container_width=True):
                    del st.session_state['optical_results']
                    st.rerun()

        # ── Medical History Form page ───────────────────────────
        elif st.session_state['page'] == 'medical_history_form':
            from medical_history import render_medical_history_form
            render_medical_history_form(user)

# ══════════════════════════════════════════════════════════════
# DOCTOR PORTAL
# ══════════════════════════════════════════════════════════════
elif st.session_state['role'] == 'doctor':

    if not st.session_state['doctor_logged_in']:
        _,col,_ = st.columns([1,1.6,1])
        with col:
            st.markdown(
                '<div class="page-title" '
                'style="text-align:center;">'
                'Doctor Portal</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="page-sub" '
                'style="text-align:center;">'
                'Sign in to review patient cases</div>',
                unsafe_allow_html=True)
            tab1,tab2 = st.tabs(["Sign In","Register"])

            with tab1:
                st.write("")
                de = st.text_input("Email", key="dli_e",
                                    placeholder="doctor@hospital.com")
                dp = st.text_input("Password",
                                    type="password", key="dli_p")
                st.write("")
                if st.button("Sign In", type="primary",
                             key="dli_btn",
                             use_container_width=True):
                    if de and dp:
                        ok,user,msg = login_doctor(de, dp)
                        if ok:
                            st.session_state[
                                'doctor_logged_in'] = True
                            st.session_state[
                                'doctor_user'] = user
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("Please fill in both fields")
                st.write("")
                if st.button("Back", key="back_d",
                             use_container_width=True):
                    st.session_state['role'] = None
                    st.rerun()

            with tab2:
                st.write("")
                c1,c2 = st.columns(2)
                dn   = c1.text_input("Full name",   key="dn")
                dsp  = c2.text_input(
                    "Specialization",
                    placeholder="Ophthalmologist", key="dsp")
                c1,c2 = st.columns(2)
                dh   = c1.text_input("Hospital",    key="dh")
                dl   = c2.text_input(
                    "License No.",
                    placeholder="e.g. MCI-12345 or KMC/12345/2020",
                    key="dl")
                dme  = st.text_input("Email",       key="dme")
                c1,c2 = st.columns(2)
                dpa  = c1.text_input("Password",
                                      type="password", key="dpa")
                dpa2 = c2.text_input("Confirm",
                                      type="password", key="dpa2")
                st.write("")
                st.markdown("**Upload Verification Document**")
                st.caption("Medical degree or state council registration — PDF or image")
                doc_file = st.file_uploader(
                    "Upload document",
                    type=["pdf","jpg","jpeg","png"],
                    key="doc_upload",
                    label_visibility="collapsed")
                st.write("")
                if st.button("Register", type="primary",
                             key="dr_btn",
                             use_container_width=True):
                    if not all([dn,dsp,dh,dl,dme,dpa]):
                        st.error("Please fill in all fields")
                    elif not doc_file:
                        st.error("Please upload your verification document")
                    elif dpa != dpa2:
                        st.error("Passwords don't match")
                    elif len(dpa) < 6:
                        st.error("Min 6 characters")
                    else:
                        os.makedirs(DOCS_DIR, exist_ok=True)
                        ext = doc_file.name.split('.')[-1]
                        doc_path = f"{DOCS_DIR}/{dme}_license.{ext}"
                        with open(doc_path, 'wb') as f:
                            f.write(doc_file.read())
                        ok,msg = register_doctor(
                            dn,dsp,dh,dl,dme,dpa,
                            doc_path=doc_path)
                        if ok:
                            st.success(
                                "Registration submitted! "
                                "Pending admin verification — "
                                "you'll be notified once approved.")
                        else:
                            st.error(msg)
    else:
        doc = st.session_state['doctor_user']
        doctor_navbar(doc)
        _dc, _da, _dm = get_doctor_notifications(doc['email'])
        if _dc > 0:
            st.error(f"URGENT: {_dc} case(s) pending your review.")
        if _da > 0:
            st.warning(f"{_da} new appointment(s) waiting for confirmation.")
        if _dm > 0:
            st.info(f"{_dm} unread message(s) from patients.")

        # ── Appointments page ──────────────────────────────────
        if st.session_state['doctor_page'] == 'appointments':
            st.markdown(
                '<div class="page-title">Appointments</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="page-sub">Upcoming and past '
                'appointments</div>',
                unsafe_allow_html=True)

            all_appts = [a for a in load_appointments()
                         if a['doctor_email'] == doc['email']]

            if not all_appts:
                st.info("No appointments booked yet.")
            else:
                pending_a   = [a for a in all_appts
                               if a['status']=='Pending']
                confirmed_a = [a for a in all_appts
                               if a['status']=='Confirmed']
                completed_a = [a for a in all_appts
                               if a['status']=='Completed']
                cancelled_a = [a for a in all_appts
                               if a['status']=='Cancelled']

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Pending",   len(pending_a))
                m2.metric("Confirmed", len(confirmed_a))
                m3.metric("Completed", len(completed_a))
                m4.metric("Cancelled", len(cancelled_a))
                st.write("")

                for appt in sorted(
                    all_appts,
                    key=lambda a: (a['date'], a['time_slot'])
                ):
                    status = appt['status']
                    with st.expander(
                        f"[{status}] "
                        f"{appt['appointment_id']} — "
                        f"{appt['patient_name']} — "
                        f"{appt['date']} {appt['time_slot']}"
                    ):
                        st.write(
                            f"**Patient:** "
                            f"{appt['patient_name']} "
                            f"({appt['patient_email']})")
                        st.write(
                            f"**Date:** {appt['date']} "
                            f"at {appt['time_slot']}")
                        st.write(
                            f"**Case ID:** {appt['case_id']}")
                        if appt['notes']:
                            st.write(f"**Notes:** {appt['notes']}")
                        if appt.get('meet_link') and status in ['Confirmed', 'Completed']:
                            st.markdown(f"""
                            <div style="background:rgba(45,158,107,0.1);border:1px solid rgba(45,158,107,0.3);
                                        border-radius:12px;padding:14px;margin:8px 0;">
                                <div style="font-size:12px;font-weight:700;color:#2d9e6b;margin-bottom:6px;">
                                    VIDEO CALL LINK
                                </div>
                                <a href="{appt['meet_link']}" target="_blank"
                                   style="color:#2d9e6b;font-size:14px;font-weight:600;">
                                    Join Google Meet
                                </a>
                                <div style="font-size:11px;color:#64748b;margin-top:4px;">
                                    {appt['date']} at {appt['time_slot']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.write("")
                        if status == 'Pending':
                            col1,col2 = st.columns(2)
                            if col1.button(
                                "Confirm",
                                key=f"conf_{appt['appointment_id']}",
                                use_container_width=True,
                                type="primary"
                            ):
                                update_appointment_status(
                                    appt['appointment_id'],
                                    'Confirmed')
                                st.rerun()
                            if col2.button(
                                "Cancel",
                                key=f"canc_{appt['appointment_id']}",
                                use_container_width=True
                            ):
                                update_appointment_status(
                                    appt['appointment_id'],
                                    'Cancelled')
                                st.rerun()
                        elif status == 'Confirmed':
                            if st.button(
                                "Mark as Completed",
                                key=f"comp_{appt['appointment_id']}",
                                use_container_width=True,
                                type="primary"
                            ):
                                update_appointment_status(
                                    appt['appointment_id'],
                                    'Completed')
                                st.rerun()

        # ── Messages page ──────────────────────────────────────
        elif st.session_state['doctor_page'] == 'messages':
            st.markdown(
                '<div class="page-title">Messages</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="page-sub">Patient conversations '
                '— only your assigned patients</div>',
                unsafe_allow_html=True)

            # Privacy: only show cases linked to this doctor
            my_appts   = [a for a in load_appointments()
                          if a['doctor_email'] == doc['email']]
            my_case_ids = [a['case_id'] for a in my_appts]
            all_cases  = [c for c in load_cases()
                          if c['case_id'] in my_case_ids]

            if not all_cases:
                st.info("No cases yet.")
            else:
                for case in all_cases:
                    msgs = load_messages(case['case_id'])
                    with st.expander(
                        f"{case['case_id']} — "
                        f"{case['patient_name']} "
                        f"({len(msgs)} messages)"
                    ):
                        render_chat(case['case_id'],
                                    'doctor', doc['name'])

        # ── Cases page ─────────────────────────────────────────
        elif st.session_state['doctor_page'] == 'cases':
            st.markdown(
                '<div class="page-title">Patient Cases</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="page-sub">Review AI-assisted '
                'screenings and send your diagnosis</div>',
                unsafe_allow_html=True)

            # Privacy: only show cases assigned to this doctor
            my_appts    = [a for a in load_appointments()
                           if a['doctor_email'] == doc['email']]
            my_case_ids = [a['case_id'] for a in my_appts]
            cases       = [c for c in load_cases()
                           if c['case_id'] in my_case_ids]

            if not cases:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">N</div>
                    <div class="empty-title">No cases yet</div>
                    <div class="empty-sub">Cases appear once
                    patients book appointments with you</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                total    = len(cases)
                pending  = sum(1 for c in cases
                               if c['status']=='Pending')
                reviewed = sum(1 for c in cases
                               if c['status']=='Reviewed')
                high     = sum(1 for c in cases
                               if 'High' in c['risk_level']
                               or 'specialist' in c['risk_level'])

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Total",    total)
                m2.metric("Pending",  pending)
                m3.metric("Reviewed", reviewed)
                m4.metric("Urgent",   high)
                st.write("")

                fc1,fc2 = st.columns([2,4])
                sf   = fc1.selectbox(
                    "Show", ["All","Pending","Reviewed"])
                srch = fc2.text_input(
                    "Search by name",
                    placeholder="Patient name...")

                filtered = cases
                if sf != "All":
                    filtered = [c for c in filtered
                                if c['status']==sf]
                if srch:
                    filtered = [c for c in filtered
                                if srch.lower() in
                                c['patient_name'].lower()]
                filtered = sorted(
                    filtered,
                    key=lambda c: (
                        0 if c['status']=='Pending' else 1,
                        0 if ('High' in c['risk_level']
                              or 'specialist' in c['risk_level'])
                        else 1
                    )
                )

                st.write(f"Showing {len(filtered)} cases")
                st.write("")

                for case in filtered:
                    risk    = case['risk_level']
                    status  = case['status']
                    is_high = ('High' in risk
                               or 'specialist' in risk)
                    is_pend = status == 'Pending'
                    risk_level_str = (
                        'High' if is_high
                        else 'Moderate' if 'Moderate' in risk
                        or 'follow' in risk.lower()
                        else 'Low'
                    )
                    risk_css = (
                        "risk-high" if is_high
                        else "risk-moderate"
                        if 'Moderate' in risk
                        or 'follow' in risk.lower()
                        else "risk-low"
                    )
                    stat_html = (
                        '<span class="status-pending">'
                        'Pending</span>'
                        if is_pend else
                        '<span class="status-reviewed">'
                        'Reviewed</span>'
                    )

                    with st.expander(
                        f"[{risk_level_str}] "
                        f"{case['case_id']} — "
                        f"{case['patient_name']} "
                        f"({case['patient_age']}y) — "
                        f"{status} — {case['timestamp']}",
                        expanded=is_high and is_pend
                    ):
                        # View History panel (moved to top)
                        with st.expander(
                            f"View Full History — "
                            f"{case['patient_name']}",
                            expanded=False
                        ):
                            render_doctor_patient_history(
                                case.get('patient_email', ''),
                                doc['name'],
                                doc['email']
                            )

                        c1,c2 = st.columns([1,1])
                        with c1:
                            st.markdown(f"""
                            <div class="doc-card">
                                <div class="section-label">
                                    Patient</div>
                                <div class="doc-name">
                                    {case['patient_name']}
                                    {stat_html}</div>
                                <div class="doc-meta">
                                    {case['patient_age']}y ·
                                    {case['patient_gender']} ·
                                    {case.get('patient_email',
                                              'N/A')}
                                </div>
                                <div style="margin-top:10px;
                                    font-size:13px;">
                                    <b>Symptoms:</b>
                                    {case['symptoms']}</div>
                                <div style="margin-top:6px;
                                    font-size:13px;
                                    color:#64748b;">
                                    Quality:
                                    {case['quality_score']}%
                                    · {case['timestamp']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.write("")
                            st.markdown("**AI Predictions**")
                            probs = case['probs']
                            for i,(name,p) in enumerate(
                                zip(DISEASE_NAMES,probs)
                            ):
                                if p > 0.3:
                                    level = (
                                        "High" if p>0.7
                                        else "Moderate" if p>0.5
                                        else "Low"
                                    )
                                    st.write(
                                        f"{level} — {name}: "
                                        f"{p*100:.1f}%")
                            st.markdown(
                                f'<div style="margin-top:12px;">'
                                f'<span class="risk-pill '
                                f'{risk_css}">'
                                f'{risk_level_str} Risk'
                                f'</span></div>',
                                unsafe_allow_html=True)

                        with c2:
                            if (case.get('image_path') and
                                    os.path.exists(
                                        case['image_path'])):
                                ic1,ic2 = st.columns(2)
                                ic1.image(
                                    Image.open(
                                        case['image_path']),
                                    caption="Retinal scan",
                                    width='stretch')
                                if (case.get('heatmap_path')
                                        and os.path.exists(
                                            case['heatmap_path']
                                        )):
                                    ic2.image(
                                        Image.open(
                                            case['heatmap_path']),
                                        caption="AI heatmap",
                                        width='stretch')

                            st.markdown("**Your Diagnosis**")
                            already_reviewed = (
                                status == "Reviewed" and
                                not st.session_state.get(
                                    f"edit_{case['case_id']}",
                                    False)
                            )

                            if already_reviewed:
                                st.success(
                                    f"Reviewed: "
                                    f"{case.get('reviewed_at','')}")
                                st.write(
                                    f"**Diagnosis:** "
                                    f"{case['doctor_diagnosis']}")
                                st.write(
                                    f"**Treatment:** "
                                    f"{case['doctor_prescription']}")
                                st.write(
                                    f"**Referral:** "
                                    f"{case['doctor_referral']}")
                                if case['doctor_notes']:
                                    st.write(
                                        f"**Notes:** "
                                        f"{case['doctor_notes']}")
                                st.divider()
                                render_chat(
                                    case['case_id'],
                                    'doctor', doc['name'])
                                if st.button(
                                    "Edit",
                                    key=f"edit_btn_"
                                        f"{case['case_id']}"
                                ):
                                    st.session_state[
                                        f"edit_"
                                        f"{case['case_id']}"
                                    ] = True
                                    st.rerun()
                            else:
                                diag = st.text_area(
                                    "Diagnosis",
                                    placeholder="e.g. Moderate "
                                    "Non-Proliferative DR",
                                    key=f"diag_"
                                        f"{case['case_id']}")
                                pres = st.text_area(
                                    "Treatment Plan",
                                    placeholder="e.g. Lucentis "
                                    "0.5mg, follow up 4 weeks",
                                    key=f"pres_"
                                        f"{case['case_id']}")
                                ref = st.selectbox(
                                    "Referral Decision",
                                    ["No referral needed",
                                     "Follow-up in 1 month",
                                     "Follow-up in 3 months",
                                     "Urgent — within 1 week",
                                     "Emergency — go immediately"],
                                    key=f"ref_"
                                        f"{case['case_id']}")
                                notes = st.text_area(
                                    "Additional Notes",
                                    placeholder="Anything else "
                                    "to mention",
                                    key=f"notes_"
                                        f"{case['case_id']}")
                                st.write("")
                                if st.button(
                                    "Submit Diagnosis",
                                    type="primary",
                                    key=f"sub_"
                                        f"{case['case_id']}",
                                    use_container_width=True
                                ):
                                    if diag:
                                        update_case(
                                            case['case_id'],
                                            diag,pres,ref,notes)
                                        st.success("Saved!")
                                        st.session_state[
                                            f"edit_"
                                            f"{case['case_id']}"
                                        ] = False
                                        st.rerun()
                                    else:
                                        st.error(
                                            "Please enter a "
                                            "diagnosis first")


# ══════════════════════════════════════════════════════════════
# ADMIN PORTAL
# ══════════════════════════════════════════════════════════════
elif st.session_state['role'] == 'admin':
    if not st.session_state['admin_logged_in']:
        _,col,_ = st.columns([1,1.6,1])
        with col:
            st.markdown(
                '<div class="page-title" style="text-align:center;">'
                'Admin Portal</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="page-sub" style="text-align:center;">'
                'Doctor verification dashboard</div>',
                unsafe_allow_html=True)
            st.write("")
            ae = st.text_input("Admin Email", key="adm_e",
                               placeholder="admin@nayana.com")
            ap = st.text_input("Password", type="password", key="adm_p")
            st.write("")
            if st.button("Sign In", type="primary",
                         use_container_width=True, key="adm_btn"):
                if ae and ap:
                    ok, msg = login_admin(ae, ap)
                    if ok:
                        st.session_state['admin_logged_in'] = True
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("Please fill in both fields")
            st.write("")
            if st.button("Back", use_container_width=True, key="adm_back"):
                st.session_state['role'] = None
                st.rerun()
    else:
        col1, col2 = st.columns([4,1])
        col1.markdown('<div class="page-title">Doctor Verification</div>', unsafe_allow_html=True)
        col1.markdown('<div class="page-sub">Review submitted credentials and approve or reject registrations.</div>', unsafe_allow_html=True)
        if col2.button("Sign Out", key="adm_so"):
            st.session_state['admin_logged_in'] = False
            st.session_state['role'] = None
            st.rerun()

        st.divider()

        pending = get_pending_doctors()

        if not pending:
            st.success("✓ All clear — no pending verifications.")
        else:
            st.warning(f"⏳ {len(pending)} doctor(s) awaiting verification")
            st.write("")
            for d in pending:
                with st.expander(
                    f"{d['name']} — {d['license_no']} — {d['email']}",
                    expanded=True
                ):
                    c1, c2 = st.columns([1,1])
                    with c1:
                        st.markdown("**Doctor Details**")
                        st.write(f"**Name:** {d['name']}")
                        st.write(f"**Specialization:** {d['specialization']}")
                        st.write(f"**Hospital:** {d['hospital']}")
                        st.write(f"**License No:** {d['license_no']}")
                        st.write(f"**Email:** {d['email']}")
                    with c2:
                        st.markdown("**Submitted Document**")
                        doc_path = d.get('doc_path', '')
                        if doc_path and os.path.exists(doc_path):
                            if doc_path.lower().endswith('.pdf'):
                                st.info(f"PDF uploaded: `{doc_path}`")
                                with open(doc_path, 'rb') as f:
                                    st.download_button(
                                        "Download Document",
                                        data=f.read(),
                                        file_name=os.path.basename(doc_path),
                                        mime="application/pdf",
                                        key=f"dl_doc_{d['email']}")
                            else:
                                st.image(doc_path,
                                         caption="Submitted document",
                                         use_container_width=True)
                        else:
                            st.warning("No document uploaded")

                    st.write("")
                    bc1, bc2 = st.columns(2)
                    if bc1.button(
                        "✅ Approve",
                        type="primary",
                        key=f"appr_{d['email']}",
                        use_container_width=True
                    ):
                        approve_doctor(d['email'])
                        st.success(f"Dr. {d['name']} approved — they can now log in.")
                        st.rerun()
                    if bc2.button(
                        "❌ Reject",
                        key=f"rej_{d['email']}",
                        use_container_width=True
                    ):
                        reject_doctor(d['email'])
                        st.warning(f"Dr. {d['name']} rejected.")
                        st.rerun()