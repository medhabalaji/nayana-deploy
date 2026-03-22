import streamlit as st
import cv2
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
from database import load_cases, save_case, update_case
from auth import (register_patient, login_patient,
                  register_doctor, login_doctor, get_all_doctors)
from styles import load_css
import json
from datetime import datetime
import tempfile
import os

# ── Appointments ───────────────────────────────────────────────
APPOINTMENTS_FILE = "appointments.json"

def load_appointments():
    if not os.path.exists(APPOINTMENTS_FILE):
        return []
    with open(APPOINTMENTS_FILE, 'r') as f:
        return json.load(f)

def save_appointments(appointments):
    with open(APPOINTMENTS_FILE, 'w') as f:
        json.dump(appointments, f, indent=2)

def book_appointment(patient_email, patient_name, doctor_email,
                     doctor_name, date, time_slot, case_id, notes):
    appointments = load_appointments()
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
    save_appointments(appointments)
    return appt_id

def update_appointment_status(appt_id, status):
    appointments = load_appointments()
    for a in appointments:
        if a['appointment_id'] == appt_id:
            a['status'] = status
            break
    save_appointments(appointments)

# ── Chat ───────────────────────────────────────────────────────
MESSAGES_FILE = "messages.json"

def load_all_messages():
    if not os.path.exists(MESSAGES_FILE):
        return {}
    with open(MESSAGES_FILE, "r") as f:
        return json.load(f)

def save_all_messages(data):
    with open(MESSAGES_FILE, "w") as f:
        json.dump(data, f, indent=2)

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
        "text": text,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
    })
    save_all_messages(data)

def render_chat(case_id, current_role, current_name):
    messages = load_messages(case_id)
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:11px;
                font-weight:700;letter-spacing:2px;
                text-transform:uppercase;color:#818cf8;
                margin-bottom:12px;">
        Messages
    </div>
    """, unsafe_allow_html=True)
    if not messages:
        st.caption("No messages yet — start the conversation below.")
    else:
        for msg in messages:
            is_me = msg['sender_role'] == current_role
            align = "flex-end" if is_me else "flex-start"
            bg    = "rgba(99,102,241,0.15)" if is_me else "rgba(255,255,255,0.06)"
            border= "rgba(99,102,241,0.4)" if is_me else "rgba(255,255,255,0.12)"
            name_color = "#a5b4fc" if is_me else "#94a3b8"
            text_color = "#e2e8f0"
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin-bottom:10px;">
                <div style="max-width:75%;background:{bg};
                            border:1px solid {border};
                            border-radius:16px;padding:12px 16px;">
                    <div style="font-size:11px;font-weight:700;
                                color:{name_color};margin-bottom:4px;">
                        {msg['sender_name']} ·
                        <span style="font-weight:400;color:#64748b;">
                            {msg['timestamp']}
                        </span>
                    </div>
                    <div style="font-size:14px;color:{text_color};line-height:1.5;">
                        {msg['text']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    key = f"chat_input_{case_id}_{current_role}"
    msg_text = st.text_input(
        "Type a message",
        placeholder="Write something...",
        key=key,
        label_visibility="collapsed"
    )
    if st.button("Send", key=f"send_{case_id}_{current_role}", type="primary"):
        if msg_text.strip():
            send_message(case_id, current_name, current_role, msg_text.strip())
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
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 6)
    m.load_state_dict(torch.load('eye_model_camera_best.pth',
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
    display_map = {
        'Cataracts': 'Cataract',
        'Conjunctivitis': 'Redness / Conjunctivitis',
        'Crossed_Eyes': 'Crossed Eyes',
        'Eyelid_Conditions': 'Eyelid Condition',
        'Normal': 'Normal',
        'Uveitis': 'Uveitis'
    }
    results = {}
    for cls, prob in zip(EYE_CLASSES, probs):
        display = display_map.get(cls, cls)
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
            elif condition == 'Eyelid Condition':
                recommendations.append("Eyelid condition detected — could be a stye or infection, consult a doctor")
            elif condition == 'Crossed Eyes':
                recommendations.append("Eye misalignment detected — consult an ophthalmologist")
    needs_fundus = any(results.get(c, 0) > 0.5 for c in ['Cataract', 'Uveitis'])
    return recommendations, high_risk, needs_fundus

from disease_info import DISEASE_INFO
from symptom_check import SYMPTOMS, triage
from patient_records import render_patient_health_record, render_doctor_patient_history

st.set_page_config(
    page_title="Nayana — AI Eye Screening",
    page_icon="N",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True

st.markdown(load_css(st.session_state['dark_mode']), unsafe_allow_html=True)

from constants import DISEASE_NAMES, DISEASE_COLORS

for key, val in {
    'role': None,
    'patient_logged_in': False,
    'patient_user': None,
    'doctor_logged_in': False,
    'doctor_user': None,
    'page': 'screening',
    'doctor_page': 'cases',
    'triage': None,
    'front_cam_open': False,
    'fundus_cam_open': False,
    'screening_step': 1,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers ────────────────────────────────────────────────────
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
        param1=50, param2=30, minRadius=30, maxRadius=200
    )
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
    enhanced = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

@st.cache_resource
def load_model():
    m = timm.create_model('efficientnet_b0', pretrained=False, num_classes=8)
    m.load_state_dict(torch.load('odir_model.pth', map_location='cpu'))
    m.eval()
    return m

def predict(image_pil):
    image_pil = preprocess_retinal(image_pil)
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        probs = torch.sigmoid(load_model()(tf(image_pil).unsqueeze(0)))[0]
    return probs.numpy()

def get_heatmap(image_pil):
    image_pil = preprocess_retinal(image_pil)
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tensor = tf(image_pil).unsqueeze(0)
    m      = load_model()
    cam    = GradCAM(model=m, target_layers=[m.blocks[-1][-1]])
    gcam   = cam(input_tensor=tensor)[0]
    rgb    = np.array(image_pil.resize((224,224))) / 255.0
    return show_cam_on_image(rgb.astype(np.float32), gcam, use_rgb=True)

def get_risk(probs):
    nn = [(DISEASE_NAMES[i], probs[i]) for i in range(1,8) if probs[i] > 0.3]
    if not nn:
        return "Looking good! No major concerns detected.", "low"
    elif max(p for _,p in nn) > 0.6:
        top = max(nn, key=lambda x: x[1])
        return f"Please see a specialist — {top[0]} detected.", "high"
    else:
        return "Some signs worth checking. Follow-up recommended.", "moderate"

def green_chart(probs):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    bars = ax.barh(DISEASE_NAMES, probs*100, color=DISEASE_COLORS, height=0.55, edgecolor='none')
    ax.set_xlabel("Confidence (%)", color='#74c69d', fontsize=9)
    ax.set_xlim(0, 108)
    ax.tick_params(colors='#2d6a4f', labelsize=9)
    for s in ax.spines.values():
        s.set_color('#b7e4c7')
    ax.axvline(50, color='#b7e4c7', lw=0.8, ls='--')
    for bar, p in zip(bars, probs):
        ax.text(p*100+1.5, bar.get_y()+bar.get_height()/2,
                f'{p*100:.1f}%', va='center', fontsize=8.5, color='#2d6a4f')
    plt.tight_layout(pad=0.5)
    return fig

model = load_model()

# ── Step bar ───────────────────────────────────────────────────
def step_bar(current_step):
    steps = ["Symptoms", "Eye Photos", "Results"]
    html  = '<div class="step-bar">'
    for i, label in enumerate(steps, 1):
        if i < current_step:
            dc, lc, di = "done",    "done",    "done"
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
    all_cases   = load_cases()
    my_cases    = [c for c in all_cases if c.get('patient_email','') == patient_email]
    new_reviews = sum(1 for c in my_cases if c['status'] == 'Reviewed')
    unread_msgs = sum(
        len([m for m in load_messages(c['case_id']) if m['sender_role'] == 'doctor'])
        for c in my_cases
    )
    my_appts     = [a for a in load_appointments() if a['patient_email'] == patient_email]
    appt_updates = sum(1 for a in my_appts if a['status'] in ['Confirmed', 'Cancelled'])
    return new_reviews, unread_msgs, appt_updates

def get_doctor_notifications(doctor_email):
    all_appts    = [a for a in load_appointments() if a['doctor_email'] == doctor_email]
    my_case_ids  = [a['case_id'] for a in all_appts]
    my_cases     = [c for c in load_cases() if c['case_id'] in my_case_ids]
    pending_cases = sum(1 for c in my_cases if c['status'] == 'Pending')
    pending_appts = sum(1 for a in all_appts if a['status'] == 'Pending')
    unread_msgs   = sum(
        len([m for m in load_messages(c['case_id']) if m['sender_role'] == 'patient'])
        for c in my_cases
    )
    return pending_cases, pending_appts, unread_msgs

def notif_label(label, count):
    return f"{label} ({count})" if count > 0 else label

# ── Navbars ────────────────────────────────────────────────────
def patient_navbar(user):
    new_reviews, unread_msgs, appt_updates = get_patient_notifications(user['email'])
    st.markdown(f"""
    <div class="topnav">
        <div class="topnav-brand">nayana</div>
        <div class="topnav-user">Hello, {user['name']}</div>
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns([2,1,1,1,1,0.6])
    if c2.button("Screening",
                 type=("primary" if st.session_state['page']=='screening' else "secondary"),
                 use_container_width=True, key="nav_scr"):
        st.session_state['page'] = 'screening'
        st.session_state['screening_step'] = 1
        st.rerun()
    if c3.button(notif_label("Results", new_reviews + unread_msgs),
                 type=("primary" if st.session_state['page']=='results' else "secondary"),
                 use_container_width=True, key="nav_res"):
        st.session_state['page'] = 'results'
        st.rerun()
    if c4.button("Health Record",
                 type=("primary" if st.session_state['page']=='health_record' else "secondary"),
                 use_container_width=True, key="nav_hr"):
        st.session_state['page'] = 'health_record'
        st.rerun()
    if c5.button("Sign Out", use_container_width=True, key="nav_so"):
        keys_to_clear = [k for k in st.session_state.keys() if k not in ['dark_mode']]
        for k in keys_to_clear:
            del st.session_state[k]
        st.rerun()
    dark_label = "Light" if st.session_state['dark_mode'] else "Dark"
    if c6.button(dark_label, use_container_width=True, key="nav_theme"):
        st.session_state['dark_mode'] = not st.session_state['dark_mode']
        st.rerun()
    st.write("")

def doctor_navbar(doc):
    pending_cases, pending_appts, unread_msgs = get_doctor_notifications(doc['email'])
    st.markdown(f"""
    <div class="topnav">
        <div class="topnav-brand">nayana <span style="font-size:12px;font-weight:600;color:#818cf8;margin-left:8px;letter-spacing:2px;">DOCTOR</span></div>
        <div class="topnav-user">Dr. {doc['name']}</div>
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns([2,1,1,1,1,0.6])
    if c2.button(notif_label("Cases", pending_cases),
                 type=("primary" if st.session_state['doctor_page']=='cases' else "secondary"),
                 use_container_width=True, key="dnav_cases"):
        st.session_state['doctor_page'] = 'cases'
        st.rerun()
    if c3.button(notif_label("Appointments", pending_appts),
                 type=("primary" if st.session_state['doctor_page']=='appointments' else "secondary"),
                 use_container_width=True, key="dnav_appt"):
        st.session_state['doctor_page'] = 'appointments'
        st.rerun()
    if c4.button(notif_label("Messages", unread_msgs),
                 type=("primary" if st.session_state['doctor_page']=='messages' else "secondary"),
                 use_container_width=True, key="dnav_msg"):
        st.session_state['doctor_page'] = 'messages'
        st.rerun()
    if c5.button("Sign Out", use_container_width=True, key="dnav_so"):
        keys_to_clear = [k for k in st.session_state.keys() if k not in ['dark_mode']]
        for k in keys_to_clear:
            del st.session_state[k]
        st.rerun()
    dark_label = "Light" if st.session_state['dark_mode'] else "Dark"
    if c6.button(dark_label, use_container_width=True, key="dnav_theme"):
        st.session_state['dark_mode'] = not st.session_state['dark_mode']
        st.rerun()
    st.write("")

# ── Results renderer ───────────────────────────────────────────
def render_my_results(my_cases):
    if not my_cases:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">N</div>
            <div class="empty-title">No screenings yet</div>
            <div class="empty-sub">Complete a screening to see your results here</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("New Screening", type="primary", key="empty_go_screen"):
            st.session_state['page'] = 'screening'
            st.session_state['screening_step'] = 1
            st.rerun()
        return

    total    = len(my_cases)
    reviewed = sum(1 for c in my_cases if c['status']=='Reviewed')
    pending  = total - reviewed
    high     = sum(1 for c in my_cases if 'High' in c['risk_level'] or 'specialist' in c['risk_level'])

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Scans", total)
    m2.metric("Reviewed",    reviewed)
    m3.metric("Pending",     pending)
    m4.metric("Urgent",      high)
    st.write("")

    if total > 1:
        st.markdown('<div class="section-label">Risk Trend Over Time</div>', unsafe_allow_html=True)
        risk_scores, timestamps = [], []
        for c in my_cases:
            r = c['risk_level'].lower()
            risk_scores.append(3 if ('high' in r or 'specialist' in r) else 2 if ('moderate' in r or 'follow' in r) else 1)
            timestamps.append(c['timestamp'][:6])
        fig, ax = plt.subplots(figsize=(8, 2.4))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        clrs = {1:'#2d9e6b', 2:'#f4a261', 3:'#e63946'}
        ax.plot(timestamps, risk_scores, color='#b7e4c7', linewidth=2, zorder=1)
        ax.scatter(timestamps, risk_scores, c=[clrs[s] for s in risk_scores], s=80, zorder=2)
        ax.set_yticks([1,2,3])
        ax.set_yticklabels(['Low','Moderate','High'], color='#2d6a4f', fontsize=9)
        ax.tick_params(axis='x', colors='#74c69d', labelsize=8)
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
        icon     = "[Reviewed]" if status == "Reviewed" else "[Pending]"
        ri       = "[High Risk]" if ("High" in risk or "specialist" in risk) else "[Moderate]" if ("Moderate" in risk or "follow" in risk.lower()) else "[Low Risk]"
        msgs     = load_messages(case['case_id'])
        doc_msgs = [m for m in msgs if m['sender_role'] == 'doctor']
        badge    = f" [{len(doc_msgs)} new message(s)]" if doc_msgs else ""

        with st.expander(
            f"{icon} {case['case_id']} — {case['timestamp']} — {ri} {status}{badge}",
            expanded=case['case_id']==st.session_state.get('last_case_id')
        ):
            c1,c2 = st.columns(2)
            with c1:
                if case.get('image_path') and os.path.exists(case['image_path']):
                    st.image(Image.open(case['image_path']), caption="Retinal scan", width='stretch')
                if case.get('heatmap_path') and os.path.exists(case['heatmap_path']):
                    st.image(Image.open(case['heatmap_path']), caption="AI heatmap", width='stretch')
            with c2:
                st.markdown("**What the AI found**")
                probs = case['probs']
                for i,(name,p) in enumerate(zip(DISEASE_NAMES,probs)):
                    if p > 0.3:
                        pc1,pc2 = st.columns([3,1])
                        pc1.progress(float(p), text=name)
                        if p>0.7: pc2.error(f"{p*100:.0f}%")
                        elif p>0.5: pc2.warning(f"{p*100:.0f}%")
                        else: pc2.info(f"{p*100:.0f}%")
                if "High" in risk or "specialist" in risk:
                    st.error(f"High Risk — {risk}")
                elif "Moderate" in risk or "follow" in risk.lower():
                    st.warning(f"Moderate — {risk}")
                else:
                    st.success(f"Low Risk — {risk}")

            st.divider()
            st.markdown("**Your doctor said:**")
            if status != "Reviewed":
                st.info("Awaiting review — check back soon")
            else:
                st.success(f"Reviewed: {case.get('reviewed_at','')}")
                for lbl, val in [
                    ("Diagnosis", case['doctor_diagnosis']),
                    ("Treatment", case['doctor_prescription'] or "None given"),
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
                    r2.success(f"{ref}")
                if case['doctor_notes']:
                    r1,r2 = st.columns([1,3])
                    r1.markdown("**Notes**")
                    r2.info(case['doctor_notes'])

            st.divider()
            render_chat(case['case_id'], 'patient',
                        st.session_state['patient_user']['name'])
            st.write("")
            pdf_key = f"pdf_data_{case['case_id']}"
            if pdf_key not in st.session_state:
                if st.button("Download Report", key=f"dl_{case['case_id']}", use_container_width=True):
                    with st.spinner("Generating..."):
                        case_probs = np.array(case.get('probs', [0]*8))
                        case_det   = case.get('detected_conditions', [])
                        msgs       = load_messages(case['case_id'])
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
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
                                risk_type=('high' if 'High' in case['risk_level'] or 'specialist' in case['risk_level'] else 'moderate' if 'Moderate' in case['risk_level'] else 'low'),
                                original_image_pil=Image.open(case['image_path']) if case.get('image_path') and os.path.exists(case['image_path']) else None,
                                heatmap_array=np.array(Image.open(case['heatmap_path'])) if case.get('heatmap_path') and os.path.exists(case['heatmap_path']) else None,
                                doctor_name=case.get('doctor_diagnosis','')[:30] if case.get('doctor_diagnosis') else None,
                                doctor_diagnosis=case.get('doctor_diagnosis'),
                                doctor_prescription=case.get('doctor_prescription'),
                                doctor_referral=case.get('doctor_referral'),
                                doctor_notes=case.get('doctor_notes'),
                                reviewed_at=case.get('reviewed_at'),
                                chat_messages=msgs,
                                visit_history=get_patient_visits(case.get('patient_email','')),
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
                if st.button("Regenerate", key=f"regen_{case['case_id']}"):
                    del st.session_state[pdf_key]
                    st.rerun()

# ══════════════════════════════════════════════════════════════
# LANDING
# ══════════════════════════════════════════════════════════════
if st.session_state['role'] is None:
    st.markdown("""
    <div class="nayana-hero">
        <div class="nayana-wordmark">nayana</div>
        <div class="nayana-meaning">the eye</div>
        <div class="nayana-tagline">
            Free AI eye screening — get results in 3 minutes
            and connect with a specialist from anywhere.
        </div>
        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-num">8</div>
                <div class="stat-lbl">conditions checked</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">6,392</div>
                <div class="stat-lbl">cases trained on</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">5</div>
                <div class="stat-lbl">languages</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">3 min</div>
                <div class="stat-lbl">to get results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _,c1,c2,_ = st.columns([1,1,1,1])
    with c1:
        st.markdown("""
        <div class="portal-card">
            <div class="portal-title">Patient</div>
            <div class="portal-sub">Screen my eyes and get expert feedback</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Screening", type="primary", use_container_width=True, key="go_patient"):
            st.session_state['role'] = 'patient'
            st.rerun()
    with c2:
        st.markdown("""
        <div class="portal-card">
            <div class="portal-title">Doctor</div>
            <div class="portal-sub">Review cases and help patients remotely</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Review Cases", use_container_width=True, key="go_doctor"):
            st.session_state['role'] = 'doctor'
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PATIENT PORTAL
# ══════════════════════════════════════════════════════════════
elif st.session_state['role'] == 'patient':

    if not st.session_state['patient_logged_in']:
        _,col,_ = st.columns([1,1.6,1])
        with col:
            st.markdown('<div class="page-title" style="text-align:center;">nayana</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub" style="text-align:center;">Sign in or create a free account</div>', unsafe_allow_html=True)
            tab1,tab2 = st.tabs(["Sign In","Create Account"])

            with tab1:
                st.write("")
                email = st.text_input("Email address", key="li_e", placeholder="you@example.com")
                pw    = st.text_input("Password", type="password", key="li_p", placeholder="Your password")
                st.write("")
                if st.button("Sign In", type="primary", key="li_btn", use_container_width=True):
                    if email and pw:
                        ok,user,msg = login_patient(email, pw)
                        if ok:
                            st.session_state['patient_logged_in'] = True
                            st.session_state['patient_user']      = user
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("Please fill in both fields")
                st.write("")
                if st.button("Back", key="back_p", use_container_width=True):
                    st.session_state['role'] = None
                    st.rerun()

            with tab2:
                st.write("")
                c1,c2 = st.columns(2)
                rn  = c1.text_input("Your name",  key="rn")
                ra  = c2.number_input("Age",1,120,30, key="ra")
                rg  = st.selectbox("Gender", ["Male","Female","Other"], key="rg")
                re  = st.text_input("Email", key="re", placeholder="you@example.com")
                c1,c2 = st.columns(2)
                rp  = c1.text_input("Password", type="password", key="rp")
                rp2 = c2.text_input("Confirm password", type="password", key="rp2")
                if rp:
                    if len(rp) < 6:
                        st.error("Too short — minimum 6 characters")
                    elif len(rp) < 8:
                        st.warning("Weak — add more characters")
                    elif not any(c.isdigit() for c in rp):
                        st.warning("Medium — add a number to strengthen")
                    elif not any(c.isupper() for c in rp):
                        st.warning("Medium — add an uppercase letter")
                    else:
                        st.success("Strong password!")
                st.write("")
                if st.button("Create Account", type="primary", key="r_btn", use_container_width=True):
                    if not all([rn,re,rp,rp2]):
                        st.error("Please fill in all fields")
                    elif rp != rp2:
                        st.error("Passwords don't match")
                    elif len(rp) < 6:
                        st.error("Min 6 characters")
                    else:
                        ok,msg = register_patient(rn,ra,rg,re,rp)
                        if ok:
                            st.success("Account created! Sign in now.")
                        else:
                            st.error(msg)
    else:
        user = st.session_state['patient_user']
        patient_navbar(user)

        if st.session_state['page'] == 'screening':
            step_bar(st.session_state.get('screening_step', 1))

            # ── STEP 1: Symptoms ───────────────────────────
            if st.session_state['screening_step'] == 1:
                st.markdown('<div class="page-title">How are your eyes feeling?</div>', unsafe_allow_html=True)
                st.markdown('<div class="page-sub">Answer a few quick questions so we know which test is right for you</div>', unsafe_allow_html=True)

                with st.expander("Your Details", expanded=True):
                    c1,c2,c3 = st.columns(3)
                    pname   = c1.text_input("Name", value=user['name'])
                    page_   = c2.number_input("Age",1,120,value=user['age'])
                    pgender = c3.selectbox("Gender",["Male","Female","Other"],
                        index=["Male","Female","Other"].index(user['gender']))
                    st.session_state['pname']   = pname
                    st.session_state['page_']   = page_
                    st.session_state['pgender'] = pgender

                st.write("")
                st.markdown('<div class="section-label">Describe your symptoms (optional)</div>', unsafe_allow_html=True)
                method = st.radio("How would you like to describe symptoms?", ["Type","Voice"], horizontal=True)

                if method == "Type":
                    COMMON_SYMPTOMS = [
                        "Blurred vision","Eye pain","Redness",
                        "Watering / discharge","Light sensitivity",
                        "Double vision","Floaters or dark spots",
                        "Headache","Itching","Swelling","Dryness",
                        "Night blindness","Halos around lights","Tunnel vision",
                    ]
                    picked = st.multiselect("Select all that apply", COMMON_SYMPTOMS, key="symp_pick")
                    symp_extra = st.text_input("Additional details (optional)", placeholder="e.g. pain started 3 days ago, worse at night...")
                    combined = ", ".join(picked)
                    if symp_extra:
                        combined = ", ".join(filter(None, [combined, symp_extra]))
                    if st.button("Confirm Symptoms", type="primary", key="symp_submit"):
                        st.session_state['symp_final'] = combined or "Not specified"
                        st.success(f"Noted: {st.session_state['symp_final']}")
                    detected = st.session_state.get('symp_final', combined or "Not specified")
                else:
                    lang = st.selectbox("Choose your language", ["Kannada","Hindi","Tamil","Telugu","English"])
                    if st.button("Start Recording", type="primary"):
                        with st.spinner(f"Listening in {lang}..."):
                            res = record_voice(lang)
                        if res["success"]:
                            st.success(f"Got it: {res['text']}")
                            if lang != "English":
                                st.caption(f"English: {res.get('english_text','')}")
                            st.info(f"Symptoms noted: {', '.join(res['symptoms'])}")
                            detected = ", ".join(res["symptoms"])
                            st.session_state['symptoms']   = detected
                            st.session_state['raw_speech'] = res['text']
                        else:
                            st.error(res["error"])
                            detected = "Not specified"
                    else:
                        detected = st.session_state.get('symptoms','Not specified')
                st.session_state['symp_final'] = detected

                st.write("")
                st.markdown('<div class="section-label">Tick anything that applies</div>', unsafe_allow_html=True)
                answers = {}
                cols = st.columns(2)
                for idx, question in enumerate(SYMPTOMS):
                    with cols[idx % 2]:
                        answers[question] = st.checkbox(question, key=f"q_{question}")
                st.session_state['triage'] = triage(answers)

                st.write("")
                if st.button("Continue to Eye Photos", type="primary", use_container_width=True):
                    from database import get_patient_record
                    record = get_patient_record(user['email'])
                    profile = record.get('profile', {}) if record else {}
                    is_incomplete = not profile.get('blood_group') or not profile.get('known_conditions')
                    if is_incomplete:
                        st.session_state['show_profile_prompt'] = True
                    else:
                        st.session_state['screening_step'] = 2
                    st.rerun()

                if st.session_state.get('show_profile_prompt', False):
                    st.divider()
                    st.markdown("### Complete your profile for better results")
                    st.caption("Your medical history helps the AI give more accurate recommendations.")
                    c1, c2 = st.columns(2)
                    if c1.button("Complete Profile Now", type="primary", use_container_width=True, key="go_profile"):
                        st.session_state['show_profile_prompt'] = False
                        st.session_state['return_to_screening'] = True
                        st.session_state['page'] = 'health_record'
                        st.rerun()
                    if c2.button("Skip for now", use_container_width=True, key="skip_profile"):
                        st.session_state['show_profile_prompt'] = False
                        st.session_state['screening_step'] = 2
                        st.rerun()

            # ── STEP 2: Eye Photos ─────────────────────────
            elif st.session_state['screening_step'] == 2:
                st.markdown('<div class="page-title">Take Your Eye Photos</div>', unsafe_allow_html=True)
                st.markdown('<div class="page-sub">Upload a front eye photo and/or a retinal scan</div>', unsafe_allow_html=True)

                triage_result = st.session_state.get('triage','front')
                if triage_result == 'fundus':
                    st.warning("Based on your symptoms, we recommend a retinal scan.")
                else:
                    st.success("A front-eye photo may be enough. You can also add a retinal scan below.")

                st.markdown('<div class="page-title" style="font-size:22px;margin-top:16px;">Front Eye Photo</div>', unsafe_allow_html=True)
                st.markdown('<div class="page-sub">Close-up of your eye in good lighting</div>', unsafe_allow_html=True)

                front_up = None
                up_tab, cam_tab = st.tabs(["Upload Photo", "Use Camera"])
                with up_tab:
                    front_up_file = st.file_uploader("Choose front eye photo", type=["jpg","jpeg","png"], key="front_eye_upload")
                    if front_up_file:
                        front_up = front_up_file
                with cam_tab:
                    if not st.session_state.get('show_front_cam', False):
                        st.info("Click below to activate your camera")
                        if st.button("Activate Camera", key="activate_front_cam", type="primary"):
                            st.session_state['show_front_cam'] = True
                            st.rerun()
                    else:
                        front_up_cam = st.camera_input("Point at your eye", key="front_camera")
                        if front_up_cam:
                            front_up = front_up_cam
                        if st.button("Turn off camera", key="off_front_cam"):
                            st.session_state['show_front_cam'] = False
                            st.rerun()

                if front_up:
                    front_pil = Image.open(front_up).convert('RGB')
                    st.session_state['front_pil'] = front_pil
                    fc1,fc2 = st.columns([1,1.6])
                    with fc1:
                        st.image(front_pil, caption="Your eye", use_container_width=True)
                    with fc2:
                        with st.spinner("Checking your eye..."):
                            fe_res = analyze_front_eye(front_pil)
                            recs, high_risk, needs_fundus = get_front_eye_recommendations(fe_res)
                        st.session_state['fe_results'] = fe_res
                        st.session_state['fe_recs']    = recs
                        st.markdown("**Quick findings:**")
                        for cond, score in sorted(fe_res.items(), key=lambda x: x[1], reverse=True):
                            col = ("#e63946" if score>0.6 else "#f4a261" if score>0.3 else "#2d9e6b")
                            st.markdown(f"""
                            <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(129,140,248,0.15);">
                                <span style="font-size:14px;font-weight:600;">{cond}</span>
                                <span style="font-size:15px;font-weight:800;color:{col};">{score*100:.0f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                    if recs:
                        for rec in recs:
                            st.warning(rec)
                    if needs_fundus:
                        st.error("Retinal scan recommended based on findings.")

                st.write("")
                st.divider()

                st.markdown('<div class="page-title" style="font-size:22px;">Retinal Scan (Fundus)</div>', unsafe_allow_html=True)
                st.markdown('<div class="page-sub">For a deeper look — upload if you have one</div>', unsafe_allow_html=True)

                uploaded = None
                up_tab2, cam_tab2 = st.tabs(["Upload Photo", "Use Camera"])
                with up_tab2:
                    uploaded_file = st.file_uploader("Upload fundus image", type=["jpg","jpeg","png"], key="fundus_upload", label_visibility="collapsed")
                    if uploaded_file:
                        uploaded = uploaded_file
                with cam_tab2:
                    if not st.session_state.get('show_fundus_cam', False):
                        st.info("Click below to activate your camera")
                        if st.button("Activate Camera", key="activate_fundus_cam", type="primary"):
                            st.session_state['show_fundus_cam'] = True
                            st.rerun()
                    else:
                        uploaded_cam = st.camera_input("Capture retinal image", key="fundus_camera")
                        if uploaded_cam:
                            uploaded = uploaded_cam
                        if st.button("Turn off camera", key="off_fundus_cam"):
                            st.session_state['show_fundus_cam'] = False
                            st.rerun()

                if uploaded:
                    image_pil = Image.open(uploaded).convert('RGB')
                    st.session_state['fundus_pil'] = image_pil
                    c1,c2 = st.columns([1,1.6])
                    with c1:
                        st.image(image_pil, caption="Fundus image", use_container_width=True)
                    with c2:
                        score, tips = check_quality(np.array(image_pil))
                        qc = ("#2d9e6b" if score>=70 else "#f4a261" if score>=40 else "#e63946")
                        ql = ("Great!" if score>=70 else "Okay" if score>=40 else "Poor")
                        st.markdown(f"""
                        <div class="card">
                            <div class="section-label">Photo quality</div>
                            <div class="quality-num" style="color:{qc};">{score}%</div>
                            <div style="font-size:14px;">{ql}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        for tip in tips:
                            st.warning(tip)
                    st.session_state['fundus_score'] = score
                    st.session_state['fundus_tips']  = tips

                st.write("")
                has_any = (st.session_state.get('front_pil') is not None
                           or st.session_state.get('fundus_pil') is not None)
                if has_any:
                    if st.button("See My Results", type="primary", use_container_width=True):
                        st.session_state['screening_step'] = 3
                        st.rerun()
                else:
                    st.info("Upload at least one photo to continue.")

                st.write("")
                if st.button("Back to Symptoms", use_container_width=True):
                    st.session_state['screening_step'] = 1
                    st.rerun()

            # ── STEP 3: Results ────────────────────────────
            elif st.session_state['screening_step'] == 3:
                pname      = st.session_state.get('pname', user['name'])
                page_      = st.session_state.get('page_', user['age'])
                pgender    = st.session_state.get('pgender', user['gender'])
                symp_final = st.session_state.get('symp_final','Not specified')
                front_pil  = st.session_state.get('front_pil')
                fundus_pil = st.session_state.get('fundus_pil')
                score      = st.session_state.get('fundus_score', 0)
                tips       = st.session_state.get('fundus_tips', [])
                fe_results = st.session_state.get('fe_results', {})
                fe_recs    = st.session_state.get('fe_recs', [])

                st.markdown('<div class="page-title">Your Results</div>', unsafe_allow_html=True)
                st.markdown('<div class="page-sub">Here\'s what our AI found</div>', unsafe_allow_html=True)

                if front_pil and fe_results:
                    st.markdown("### Front Eye Analysis")
                    fc1,fc2 = st.columns([1,2])
                    with fc1:
                        st.image(front_pil, caption="Your eye", use_container_width=True)
                    with fc2:
                        for cond, s in sorted(fe_results.items(), key=lambda x: x[1], reverse=True):
                            col = ("#e63946" if s>0.6 else "#f4a261" if s>0.3 else "#2d9e6b")
                            st.markdown(f"""
                            <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(129,140,248,0.15);">
                                <span style="font-size:14px;font-weight:600;">{cond}</span>
                                <span style="font-size:15px;font-weight:800;color:{col};">{s*100:.0f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                    if fe_recs:
                        for rec in fe_recs:
                            st.warning(rec)
                    st.write("")

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
                    risk_css  = {"high":"risk-high","moderate":"risk-moderate","low":"risk-low"}[risk_type]
                    det_conds = [(DISEASE_NAMES[i], probs[i]) for i in range(8) if probs[i] > 0.5]

                    card_cls = ("highlight" if risk_type=='low' else "warning" if risk_type=='moderate' else "danger")
                    st.markdown(f"""
                    <div class="card {card_cls}">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                            <div style="font-size:16px;font-weight:800;font-family:'Space Grotesk',sans-serif;">Overall Result</div>
                            <span class="risk-pill {risk_css}">{risk_type.title()} Risk</span>
                        </div>
                        <div style="font-size:15px;">{risk_txt}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    for name,p in sorted(det_conds, key=lambda x: x[1], reverse=True):
                        if name == 'Normal':
                            st.success(f"Normal — {p*100:.1f}%")
                        elif p > 0.7:
                            st.error(f"Detected: {name} — {p*100:.1f}%")
                        else:
                            st.warning(f"Possible: {name} — {p*100:.1f}%")

                    non_normal = [(n,p) for n,p in det_conds if n != 'Normal']
                    if non_normal:
                        st.divider()
                        st.markdown('<div class="page-title" style="font-size:20px;">What These Findings Mean</div>', unsafe_allow_html=True)
                        st.markdown('<div class="page-sub" style="margin-bottom:16px;">Plain-language explanations for patients</div>', unsafe_allow_html=True)
                        for dname, dp in sorted(non_normal, key=lambda x: x[1], reverse=True):
                            info = DISEASE_INFO.get(dname)
                            if not info:
                                continue
                            with st.expander(f"{dname} — What you need to know", expanded=dp > 0.6):
                                st.markdown(f"""
<div style="padding:16px 20px;border-radius:12px;background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);">
  <b>What is it?</b><br><span style="color:#94a3b8;">{info['what']}</span><br><br>
  <b>Symptoms to watch for:</b><br><span style="color:#94a3b8;">{info['symptoms']}</span><br><br>
  <b>What to do:</b> <span style="font-weight:700;color:{'#fca5a5' if info['serious'] else '#6ee7b7'};">{info['urgency']}</span><br><br>
  <i style="color:#64748b;">Tip: {info['tip']}</i>
</div>""", unsafe_allow_html=True)

                    if risk_type in ['high', 'moderate']:
                        st.divider()
                        maps_url = "https://www.google.com/maps/search/ophthalmologist+near+me"
                        st.markdown(f'<a href="{maps_url}" target="_blank"><button style="background:linear-gradient(135deg,#6366f1,#38bdf8);color:white;border:none;border-radius:12px;padding:11px 20px;width:100%;font-weight:700;cursor:pointer;font-size:14px;font-family:Space Grotesk,sans-serif;">Find Eye Clinics Near Me</button></a>', unsafe_allow_html=True)
                        st.caption("Sankara Nethralaya · Narayana Nethralaya · LV Prasad Eye Institute · Aravind Eye Hospital")
                        st.write("")

                    st.write("")
                    st.markdown("### Where the AI looked")
                    hc1,hc2 = st.columns(2)
                    hc1.image(fundus_pil.resize((224,224)), caption="Your scan", width=220)
                    hc2.image(heatmap, caption="AI focus", width=220)

                st.divider()
                st.markdown("### What would you like to do?")
                ac1,ac2 = st.columns(2)

                with ac1:
                    st.markdown("### Book an Appointment")
                    doctors = get_all_doctors()
                    if not doctors:
                        st.warning("No doctors registered yet — check back soon.")
                    else:
                        doctor_options = {
                            f"Dr. {d['name']} — {d['specialization']} ({d['hospital']})": d
                            for d in doctors
                        }
                        selected_doc_label = st.selectbox("Choose a doctor", list(doctor_options.keys()), key="appt_doc")
                        selected_doc = doctor_options[selected_doc_label]

                        import datetime as dt
                        today = dt.date.today()
                        appt_date = st.date_input("Select date",
                            min_value=today, max_value=today + dt.timedelta(days=30), key="appt_date")

                        time_slots = [
                            "09:00 AM","09:30 AM","10:00 AM","10:30 AM",
                            "11:00 AM","11:30 AM","12:00 PM","02:00 PM",
                            "02:30 PM","03:00 PM","03:30 PM","04:00 PM",
                            "04:30 PM","05:00 PM"
                        ]
                        time_slot  = st.selectbox("Select time slot", time_slots, key="appt_time")
                        appt_notes = st.text_area("Notes for doctor (optional)",
                            placeholder="Mention any specific concerns...", key="appt_notes")

                        if st.button("Book Appointment & Send Case", type="primary",
                                     use_container_width=True, key="book_appt"):
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
                                        f"{c}: {s*100:.0f}%" for c,s in sorted(fe_results.items(), key=lambda x: x[1], reverse=True))
                                cid = save_case(
                                    patient_name=pname, patient_age=int(page_),
                                    patient_gender=pgender, symptoms=symp_final + fe_str,
                                    quality_score=score,
                                    probs=probs if probs is not None else np.zeros(8),
                                    detected_conditions=det_conds, risk_level=risk_txt,
                                    image_path=ip, heatmap_path=hp, patient_email=user['email'])
                                appt_id = book_appointment(
                                    patient_email=user['email'], patient_name=pname,
                                    doctor_email=selected_doc['email'], doctor_name=selected_doc['name'],
                                    date=str(appt_date), time_slot=time_slot,
                                    case_id=cid, notes=appt_notes)
                            st.success(f"Appointment booked — ID: {appt_id} with Dr. {selected_doc['name']} on {appt_date} at {time_slot}")
                            st.session_state['last_case_id'] = cid

                with ac2:
                    if fundus_pil and probs is not None:
                        if st.button("Download Report", use_container_width=True):
                            with st.spinner("Building report..."):
                                fe_str = ""
                                if fe_results:
                                    fe_str = "\n\nFront Eye Findings:\n" + "\n".join(
                                        f"  - {c}: {s*100:.0f}%" for c,s in sorted(fe_results.items(), key=lambda x: x[1], reverse=True))
                                    if fe_recs:
                                        fe_str += "\n\nFront Eye Recommendations:\n" + "\n".join(f"  - {r}" for r in fe_recs)
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                    pdf_path = generate_report(
                                        patient_name=pname, patient_age=int(page_),
                                        patient_gender=pgender, symptoms=symp_final + fe_str,
                                        quality_score=score, quality_tips=tips,
                                        probs=probs, detected_conditions=det_conds,
                                        risk_level=risk_txt, original_image_pil=fundus_pil,
                                        heatmap_array=heatmap, output_path=tmp.name)
                            with open(pdf_path,'rb') as f:
                                st.download_button("Download PDF", data=f.read(),
                                    file_name=f"nayana_{pname.replace(' ','_')}.pdf",
                                    mime="application/pdf", use_container_width=True)
                    else:
                        st.info("Add a retinal scan to download a report")

                st.write("")
                if st.button("Start a New Screening", use_container_width=True):
                    for k in ['front_pil','fundus_pil','fe_results','fe_recs','probs','heatmap','fundus_score','fundus_tips']:
                        st.session_state[k] = None
                    st.session_state['front_cam_open']  = False
                    st.session_state['fundus_cam_open'] = False
                    st.session_state['screening_step']  = 1
                    st.rerun()

                st.write("")
                st.caption("Nayana is for screening only. Always follow your doctor's advice.")

        elif st.session_state['page'] == 'results':
            st.markdown('<div class="page-title">My Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub">Your screening history and doctor responses</div>', unsafe_allow_html=True)
            all_cases = load_cases()
            my_cases  = [c for c in all_cases if c.get('patient_email','') == user['email']]
            render_my_results(my_cases)

        elif st.session_state['page'] == 'health_record':
            st.markdown('<div class="page-title">My Health Record</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub">Your complete eye health history</div>', unsafe_allow_html=True)
            if st.session_state.get('return_to_screening', False):
                if st.button("Back to Screening", type="primary", key="back_to_screen"):
                    st.session_state['return_to_screening'] = False
                    st.session_state['page'] = 'screening'
                    st.rerun()
            render_patient_health_record(user)

# ══════════════════════════════════════════════════════════════
# DOCTOR PORTAL
# ══════════════════════════════════════════════════════════════
elif st.session_state['role'] == 'doctor':

    if not st.session_state['doctor_logged_in']:
        _,col,_ = st.columns([1,1.6,1])
        with col:
            st.markdown('<div class="page-title" style="text-align:center;">Doctor Portal</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub" style="text-align:center;">Sign in to review patient cases</div>', unsafe_allow_html=True)
            tab1,tab2 = st.tabs(["Sign In","Register"])

            with tab1:
                st.write("")
                de = st.text_input("Email", key="dli_e", placeholder="doctor@hospital.com")
                dp = st.text_input("Password", type="password", key="dli_p")
                st.write("")
                if st.button("Sign In", type="primary", key="dli_btn", use_container_width=True):
                    if de and dp:
                        ok,user,msg = login_doctor(de, dp)
                        if ok:
                            st.session_state['doctor_logged_in'] = True
                            st.session_state['doctor_user']      = user
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("Please fill in both fields")
                st.write("")
                if st.button("Back", key="back_d", use_container_width=True):
                    st.session_state['role'] = None
                    st.rerun()

            with tab2:
                st.write("")
                c1,c2 = st.columns(2)
                dn   = c1.text_input("Full name",   key="dn")
                dsp  = c2.text_input("Specialization", placeholder="Ophthalmologist", key="dsp")
                c1,c2 = st.columns(2)
                dh   = c1.text_input("Hospital",    key="dh")
                dl   = c2.text_input("License No.", key="dl")
                dme  = st.text_input("Email",       key="dme")
                c1,c2 = st.columns(2)
                dpa  = c1.text_input("Password", type="password", key="dpa")
                dpa2 = c2.text_input("Confirm",  type="password", key="dpa2")
                st.write("")
                if st.button("Register", type="primary", key="dr_btn", use_container_width=True):
                    if not all([dn,dsp,dh,dl,dme,dpa]):
                        st.error("Please fill in all fields")
                    elif dpa != dpa2:
                        st.error("Passwords don't match")
                    elif len(dpa) < 6:
                        st.error("Min 6 characters")
                    else:
                        ok,msg = register_doctor(dn,dsp,dh,dl,dme,dpa)
                        if ok:
                            st.success("Registered! Sign in now.")
                        else:
                            st.error(msg)
    else:
        doc = st.session_state['doctor_user']
        doctor_navbar(doc)

        if st.session_state['doctor_page'] == 'appointments':
            st.markdown('<div class="page-title">Appointments</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub">Upcoming and past appointments</div>', unsafe_allow_html=True)

            all_appts = [a for a in load_appointments() if a['doctor_email'] == doc['email']]

            if not all_appts:
                st.info("No appointments booked yet.")
            else:
                pending_a   = [a for a in all_appts if a['status'] == 'Pending']
                confirmed_a = [a for a in all_appts if a['status'] == 'Confirmed']
                completed_a = [a for a in all_appts if a['status'] == 'Completed']
                cancelled_a = [a for a in all_appts if a['status'] == 'Cancelled']

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Pending",   len(pending_a))
                m2.metric("Confirmed", len(confirmed_a))
                m3.metric("Completed", len(completed_a))
                m4.metric("Cancelled", len(cancelled_a))
                st.write("")

                for appt in sorted(all_appts, key=lambda a: (a['date'], a['time_slot'])):
                    status = appt['status']
                    label  = f"[{status}] {appt['appointment_id']} — {appt['patient_name']} — {appt['date']} {appt['time_slot']}"
                    with st.expander(label):
                        st.write(f"**Patient:** {appt['patient_name']} ({appt['patient_email']})")
                        st.write(f"**Date:** {appt['date']} at {appt['time_slot']}")
                        st.write(f"**Case ID:** {appt['case_id']}")
                        if appt['notes']:
                            st.write(f"**Notes:** {appt['notes']}")
                        st.write("")
                        if status == 'Pending':
                            col1,col2 = st.columns(2)
                            if col1.button("Confirm", key=f"conf_{appt['appointment_id']}", use_container_width=True, type="primary"):
                                update_appointment_status(appt['appointment_id'], 'Confirmed')
                                st.rerun()
                            if col2.button("Cancel", key=f"canc_{appt['appointment_id']}", use_container_width=True):
                                update_appointment_status(appt['appointment_id'], 'Cancelled')
                                st.rerun()
                        elif status == 'Confirmed':
                            if st.button("Mark as Completed", key=f"comp_{appt['appointment_id']}", use_container_width=True, type="primary"):
                                update_appointment_status(appt['appointment_id'], 'Completed')
                                st.rerun()

        elif st.session_state['doctor_page'] == 'messages':
            st.markdown('<div class="page-title">Messages</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub">All patient conversations</div>', unsafe_allow_html=True)
            all_cases = [c for c in load_cases() if any(
                a['doctor_email'] == doc['email'] and a['case_id'] == c['case_id']
                for a in load_appointments()
            )]
            if not all_cases:
                st.info("No cases yet.")
            else:
                for case in all_cases:
                    msgs = load_messages(case['case_id'])
                    with st.expander(f"{case['case_id']} — {case['patient_name']} ({len(msgs)} messages)"):
                        render_chat(case['case_id'], 'doctor', doc['name'])

        elif st.session_state['doctor_page'] == 'cases':
            st.markdown('<div class="page-title">Patient Cases</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub">Review AI-assisted screenings and send your diagnosis</div>', unsafe_allow_html=True)

            cases = load_cases()
            if not cases:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">N</div>
                    <div class="empty-title">No cases yet</div>
                    <div class="empty-sub">Cases appear once patients complete screenings</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                total    = len(cases)
                pending  = sum(1 for c in cases if c['status']=='Pending')
                reviewed = sum(1 for c in cases if c['status']=='Reviewed')
                high     = sum(1 for c in cases if 'High' in c['risk_level'] or 'specialist' in c['risk_level'])

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Total",    total)
                m2.metric("Pending",  pending)
                m3.metric("Reviewed", reviewed)
                m4.metric("Urgent",   high)
                st.write("")

                fc1,fc2 = st.columns([2,4])
                sf   = fc1.selectbox("Show", ["All","Pending","Reviewed"])
                srch = fc2.text_input("Search by name", placeholder="Patient name...")

                filtered = cases
                if sf != "All":
                    filtered = [c for c in filtered if c['status']==sf]
                if srch:
                    filtered = [c for c in filtered if srch.lower() in c['patient_name'].lower()]
                filtered = sorted(filtered, key=lambda c: (
                    0 if c['status']=='Pending' else 1,
                    0 if ('High' in c['risk_level'] or 'specialist' in c['risk_level']) else 1
                ))

                st.write(f"Showing {len(filtered)} cases")
                st.write("")

                for case in filtered:
                    risk    = case['risk_level']
                    status  = case['status']
                    is_high = ('High' in risk or 'specialist' in risk)
                    is_pend = status == 'Pending'
                    risk_level_str = 'High' if is_high else 'Moderate' if 'Moderate' in risk or 'follow' in risk.lower() else 'Low'
                    risk_css = ("risk-high" if is_high else "risk-moderate" if 'Moderate' in risk or 'follow' in risk.lower() else "risk-low")
                    stat_html = ('<span class="status-pending">Pending</span>' if is_pend else '<span class="status-reviewed">Reviewed</span>')

                    with st.expander(
                        f"[{risk_level_str}] {case['case_id']} — {case['patient_name']} ({case['patient_age']}y) — {status} — {case['timestamp']}",
                        expanded=is_high and is_pend
                    ):
                        c1,c2 = st.columns([1,1])
                        with c1:
                            st.markdown(f"""
                            <div class="doc-card">
                                <div class="section-label">Patient</div>
                                <div class="doc-name">{case['patient_name']} {stat_html}</div>
                                <div class="doc-meta">{case['patient_age']}y · {case['patient_gender']} · {case.get('patient_email','N/A')}</div>
                                <div style="margin-top:10px;font-size:13px;"><b>Symptoms:</b> {case['symptoms']}</div>
                                <div style="margin-top:6px;font-size:13px;color:#64748b;">Quality: {case['quality_score']}% · {case['timestamp']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.write("")
                            st.markdown("**AI Predictions**")
                            probs = case['probs']
                            for i,(name,p) in enumerate(zip(DISEASE_NAMES,probs)):
                                if p > 0.3:
                                    level = "High" if p>0.7 else "Moderate" if p>0.5 else "Low"
                                    st.write(f"{level} — {name}: {p*100:.1f}%")
                            st.markdown(f'<div style="margin-top:12px;"><span class="risk-pill {risk_css}">{risk_level_str} Risk</span></div>', unsafe_allow_html=True)

                        with c2:
                            if case.get('image_path') and os.path.exists(case['image_path']):
                                ic1,ic2 = st.columns(2)
                                ic1.image(Image.open(case['image_path']), caption="Retinal scan", width='stretch')
                                if case.get('heatmap_path') and os.path.exists(case['heatmap_path']):
                                    ic2.image(Image.open(case['heatmap_path']), caption="AI heatmap", width='stretch')

                            st.markdown("**Your Diagnosis**")
                            already_reviewed = (status == "Reviewed" and
                                not st.session_state.get(f"edit_{case['case_id']}", False))

                            if already_reviewed:
                                st.success(f"Reviewed: {case.get('reviewed_at','')}")
                                st.write(f"**Diagnosis:** {case['doctor_diagnosis']}")
                                st.write(f"**Treatment:** {case['doctor_prescription']}")
                                st.write(f"**Referral:** {case['doctor_referral']}")
                                if case['doctor_notes']:
                                    st.write(f"**Notes:** {case['doctor_notes']}")
                                st.divider()
                                render_chat(case['case_id'], 'doctor', doc['name'])
                                if st.button("Edit", key=f"edit_btn_{case['case_id']}"):
                                    st.session_state[f"edit_{case['case_id']}"] = True
                                    st.rerun()
                            else:
                                diag = st.text_area("Diagnosis",
                                    placeholder="e.g. Moderate Non-Proliferative DR",
                                    key=f"diag_{case['case_id']}")
                                pres = st.text_area("Treatment Plan",
                                    placeholder="e.g. Lucentis 0.5mg, follow up 4 weeks",
                                    key=f"pres_{case['case_id']}")
                                ref = st.selectbox("Referral Decision",
                                    ["No referral needed","Follow-up in 1 month","Follow-up in 3 months",
                                     "Urgent — visit within 1 week","Emergency — go immediately"],
                                    key=f"ref_{case['case_id']}")
                                notes = st.text_area("Additional Notes",
                                    placeholder="Anything else to mention",
                                    key=f"notes_{case['case_id']}")
                                st.write("")
                                if st.button("Submit Diagnosis", type="primary",
                                             key=f"sub_{case['case_id']}", use_container_width=True):
                                    if diag:
                                        update_case(case['case_id'], diag, pres, ref, notes)
                                        st.success("Saved!")
                                        st.session_state[f"edit_{case['case_id']}"] = False
                                        st.rerun()
                                    else:
                                        st.error("Please enter a diagnosis first")
                        st.divider()
        with st.expander(
            f"View Full History — {case['patient_name']}",
            expanded=False
        ):
            render_doctor_patient_history(
                case.get('patient_email', ''),
                doc['name']
            )
st.write("")
