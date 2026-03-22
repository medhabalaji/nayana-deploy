import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import io
from PIL import Image
from database import (
    get_patient_record, get_patient_visits,
    get_risk_trend, update_patient_profile,
    add_continuity_note, load_cases,
    load_appointments
)
from constants import DISEASE_NAMES, DISEASE_COLORS
import json
import tempfile

MESSAGES_FILE = "messages.json"

def load_messages(case_id):
    if not os.path.exists(MESSAGES_FILE):
        return []
    with open(MESSAGES_FILE, "r") as f:
        data = json.load(f)
    return data.get(case_id, [])

# ── Privacy enforcement ────────────────────────────────────────
def doctor_can_access_patient(doctor_email, patient_email):
    """
    Returns True only if the doctor has at least one appointment
    with this patient. Enforces strict access control.
    """
    appointments = load_appointments()
    return any(
        a['doctor_email'] == doctor_email and
        a['patient_email'] == patient_email
        for a in appointments
    )

def get_doctor_patients(doctor_email):
    """
    Returns list of unique patient emails that have booked
    with this doctor. Used to scope the doctor's view.
    """
    appointments = load_appointments()
    seen = set()
    patients = []
    for a in appointments:
        if (a['doctor_email'] == doctor_email and
                a['patient_email'] not in seen):
            seen.add(a['patient_email'])
            patients.append(a['patient_email'])
    return patients

# ── Chart helpers ──────────────────────────────────────────────
def generate_progression_summary(patient_email):
    visits = get_patient_visits(patient_email)
    if len(visits) < 2:
        return "Not enough visits to compare yet."
    last  = visits[-1]
    prev  = visits[-2]

    def risk_score(r):
        r = r.lower()
        if "high" in r or "specialist" in r: return 3
        elif "moderate" in r or "follow" in r: return 2
        return 1

    last_score = risk_score(last["risk_level"])
    prev_score = risk_score(prev["risk_level"])

    if last_score < prev_score:
        trend = "Your eye health has improved since your last visit."
    elif last_score > prev_score:
        trend = "Your eye health needs attention — risk has increased since your last visit."
    else:
        trend = "Your eye health is stable compared to your last visit."

    last_probs = last.get("probs", [])
    prev_probs = prev.get("probs", [])
    changes = []
    for i, name in enumerate(DISEASE_NAMES):
        if i == 0 or i >= len(last_probs) or i >= len(prev_probs):
            continue
        diff = last_probs[i] - prev_probs[i]
        if abs(diff) > 0.1:
            direction = "reduced" if diff < 0 else "increased"
            changes.append(
                f"{name} risk {direction} from "
                f"{prev_probs[i]*100:.0f}% to "
                f"{last_probs[i]*100:.0f}%."
            )
    if changes:
        trend += " " + " ".join(changes)
    else:
        trend += " No significant changes in individual conditions."
    return trend

def risk_trend_chart(patient_email, dark_mode=False):
    trend = get_risk_trend(patient_email)
    if len(trend) < 2:
        return None
    dates  = [t[0] for t in trend]
    scores = [t[1] for t in trend]
    bg = '#13131f' if dark_mode else '#ffffff'
    tc = '#94a3b8' if dark_mode else '#2d6a4f'
    fig, ax = plt.subplots(figsize=(8, 2.4))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    clrs = {1:'#2d9e6b', 2:'#f4a261', 3:'#e63946'}
    ax.plot(dates, scores, color='#b7e4c7', linewidth=2, zorder=1)
    ax.scatter(dates, scores,
               c=[clrs[s] for s in scores], s=80, zorder=2)
    ax.set_yticks([1,2,3])
    ax.set_yticklabels(['Low','Moderate','High'],
                       color=tc, fontsize=9)
    ax.tick_params(axis='x', colors=tc, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#b7e4c7')
    ax.set_ylim(0.5, 3.5)
    plt.tight_layout(pad=0.5)
    return fig

def disease_trend_chart(patient_email, dark_mode=False):
    visits = get_patient_visits(patient_email)
    if len(visits) < 2:
        return None
    dates = [v["timestamp"][:6] for v in visits]
    bg = '#13131f' if dark_mode else '#ffffff'
    tc = '#94a3b8' if dark_mode else '#2d6a4f'
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for i, (name, color) in enumerate(
        zip(DISEASE_NAMES[1:], DISEASE_COLORS[1:]), 1
    ):
        scores = []
        for v in visits:
            probs = v.get("probs", [])
            scores.append(
                float(probs[i]) * 100
                if len(probs) > i else 0
            )
        if max(scores) > 10:
            ax.plot(dates, scores, marker='o',
                    label=name, color=color,
                    linewidth=1.5, markersize=4)
    ax.set_ylabel("Confidence (%)", color=tc, fontsize=9)
    ax.tick_params(colors=tc, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#b7e4c7')
    ax.legend(fontsize=7, loc='upper right',
              facecolor=bg, labelcolor=tc)
    ax.set_ylim(0, 105)
    plt.tight_layout(pad=0.5)
    return fig

# ── Patient view ───────────────────────────────────────────────
def render_patient_health_record(user):
    """
    Renders the full patient health file.
    Only shows data belonging to the logged-in patient.
    """
    patient_email = user['email']
    record  = get_patient_record(patient_email)
    visits  = get_patient_visits(patient_email)
    dark    = st.session_state.get('dark_mode', True)

    if not record:
        st.info("No health record found yet. "
                "Complete a screening to start your file.")
        return

    profile = record["profile"]

    # ── Profile card ───────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.08);
                border:1px solid rgba(99,102,241,0.2);
                border-radius:16px;padding:20px 24px;
                margin-bottom:24px;">
        <div style="font-size:22px;font-weight:700;
                    margin-bottom:6px;">{profile['name']}</div>
        <div style="font-size:13px;color:#64748b;
                    margin-bottom:12px;">
            {profile['age']}y · {profile['gender']} ·
            ID: {profile['patient_id']} ·
            Member since {profile['joined']}
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div>
                <div style="font-size:11px;color:#64748b;
                            text-transform:uppercase;
                            letter-spacing:1px;">
                    Total Screenings
                </div>
                <div style="font-size:22px;font-weight:700;
                            color:#818cf8;">{len(visits)}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748b;
                            text-transform:uppercase;
                            letter-spacing:1px;">
                    Last Screening
                </div>
                <div style="font-size:14px;font-weight:600;">
                    {visits[-1]['timestamp'][:11]
                     if visits else 'None'}
                </div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748b;
                            text-transform:uppercase;
                            letter-spacing:1px;">
                    Blood Group
                </div>
                <div style="font-size:14px;font-weight:600;">
                    {profile.get('blood_group') or 'Not set'}
                </div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748b;
                            text-transform:uppercase;
                            letter-spacing:1px;">
                    Known Conditions
                </div>
                <div style="font-size:14px;font-weight:600;">
                    {', '.join(profile.get('known_conditions',[]))
                     or 'None recorded'}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Download full history PDF ──────────────────────────────
    if visits:
        if st.button("Download Full History as PDF",
                     use_container_width=True,
                     key="dl_full_history"):
            with st.spinner("Generating your complete health file..."):
                from report_generator import generate_full_history_pdf
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix='.pdf'
                ) as tmp:
                    pdf_path = generate_full_history_pdf(
                        patient_email=patient_email,
                        profile=profile,
                        visits=visits,
                        output_path=tmp.name,
                        include_doctor_notes=False
                    )
                with open(pdf_path, 'rb') as f:
                    st.download_button(
                        "Download PDF",
                        data=f.read(),
                        file_name=f"nayana_health_file_"
                                  f"{profile['name'].replace(' ','_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_full_history_btn"
                    )

    st.write("")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Visit History",
        "Prescriptions",
        "Trends",
        "My Profile"
    ])

    # ── TAB 1: Overview ────────────────────────────────────────
    with tab1:
        if len(visits) >= 2:
            summary = generate_progression_summary(patient_email)
            st.markdown(f"""
            <div style="background:rgba(45,158,107,0.1);
                        border:1px solid rgba(45,158,107,0.3);
                        border-radius:12px;
                        padding:16px 20px;margin-bottom:20px;">
                <div style="font-size:11px;font-weight:700;
                            letter-spacing:2px;
                            text-transform:uppercase;
                            color:#2d9e6b;margin-bottom:6px;">
                    Progression Summary
                </div>
                <div style="font-size:14px;line-height:1.6;">
                    {summary}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">'
                    'Risk Timeline</div>',
                    unsafe_allow_html=True)
        if visits:
            for v in reversed(visits):
                risk  = v['risk_level'].lower()
                if "high" in risk or "specialist" in risk:
                    color, label = "#e63946", "High Risk"
                elif "moderate" in risk or "follow" in risk:
                    color, label = "#f4a261", "Moderate"
                else:
                    color, label = "#2d9e6b", "Low Risk"
                det = v.get("detected_conditions", [])
                top_cond = ""
                if det:
                    top = max(det, key=lambda x: x[1])
                    top_cond = (f" — {top[0]} "
                                f"({top[1]*100:.0f}%)")
                st.markdown(f"""
                <div style="display:flex;align-items:center;
                            gap:12px;padding:10px 0;
                            border-bottom:1px solid
                            rgba(99,102,241,0.1);">
                    <div style="width:10px;height:10px;
                                border-radius:50%;
                                background:{color};
                                flex-shrink:0;"></div>
                    <div style="font-size:13px;color:#64748b;
                                min-width:120px;">
                        {v['timestamp'][:11]}
                    </div>
                    <div style="font-size:13px;font-weight:600;
                                color:{color};min-width:90px;">
                        {label}
                    </div>
                    <div style="font-size:13px;color:#94a3b8;">
                        {v['status']} {top_cond}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No visits yet.")

    # ── TAB 2: Visit History ───────────────────────────────────
    with tab2:
        if not visits:
            st.info("No visits recorded yet.")
        else:
            for v in reversed(visits):
                risk  = v['risk_level'].lower()
                badge = ("High Risk" if "high" in risk
                         or "specialist" in risk
                         else "Moderate" if "moderate" in risk
                         or "follow" in risk else "Low Risk")
                with st.expander(
                    f"{v['timestamp']} — {badge} "
                    f"— {v['status']}",
                    expanded=False
                ):
                    st.write(f"**Symptoms:** {v['symptoms']}")
                    st.write(
                        f"**Image Quality:** "
                        f"{v['quality_score']}%"
                    )
                    c1,c2 = st.columns(2)
                    with c1:
                        if (v.get('image_path') and
                                os.path.exists(v['image_path'])):
                            st.image(
                                Image.open(v['image_path']),
                                caption="Retinal scan",
                                use_container_width=True
                            )
                        if (v.get('heatmap_path') and
                                os.path.exists(v['heatmap_path'])):
                            st.image(
                                Image.open(v['heatmap_path']),
                                caption="AI heatmap",
                                use_container_width=True
                            )
                    with c2:
                        st.markdown("**AI Findings:**")
                        probs = v.get("probs", [])
                        for i,(name,p) in enumerate(
                            zip(DISEASE_NAMES, probs)
                        ):
                            if p > 0.3:
                                pc1,pc2 = st.columns([3,1])
                                pc1.progress(float(p), text=name)
                                if p>0.7:
                                    pc2.error(f"{p*100:.0f}%")
                                elif p>0.5:
                                    pc2.warning(f"{p*100:.0f}%")
                                else:
                                    pc2.info(f"{p*100:.0f}%")

                    if v['status'] == 'Reviewed':
                        st.divider()
                        st.markdown("**Doctor's Assessment:**")
                        st.write(
                            f"**Diagnosis:** "
                            f"{v['doctor_diagnosis']}"
                        )
                        st.write(
                            f"**Treatment:** "
                            f"{v['doctor_prescription'] or 'None'}"
                        )
                        st.write(
                            f"**Referral:** "
                            f"{v['doctor_referral']}"
                        )
                        if v['doctor_notes']:
                            st.write(
                                f"**Notes:** "
                                f"{v['doctor_notes']}"
                            )
                        st.write(
                            f"**Reviewed:** "
                            f"{v.get('reviewed_at','')}"
                        )
                    else:
                        st.info("Awaiting doctor review.")

                    # Download individual visit report
                    pdf_key = f"history_pdf_{v['case_id']}"
                    if pdf_key not in st.session_state:
                        if st.button(
                            "Download Report for this Visit",
                            key=f"dl_visit_{v['case_id']}",
                            use_container_width=True
                        ):
                            with st.spinner("Generating..."):
                                from report_generator import generate_report
                                case_probs = np.array(
                                    v.get('probs', [0]*8))
                                case_det = v.get(
                                    'detected_conditions', [])
                                msgs = load_messages(v['case_id'])
                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix='.pdf'
                                ) as tmp:
                                    pdf_path = generate_report(
                                        patient_name=v['patient_name'],
                                        patient_age=v['patient_age'],
                                        patient_gender=v['patient_gender'],
                                        patient_email=patient_email,
                                        symptoms=v['symptoms'],
                                        quality_score=v['quality_score'],
                                        quality_tips=[],
                                        probs=case_probs,
                                        detected_conditions=case_det,
                                        risk_level=v['risk_level'],
                                        risk_type=(
                                            'high' if 'High' in v['risk_level']
                                            or 'specialist' in v['risk_level']
                                            else 'moderate' if 'Moderate' in v['risk_level']
                                            else 'low'
                                        ),
                                        original_image_pil=Image.open(v['image_path'])
                                            if v.get('image_path') and
                                            os.path.exists(v['image_path'])
                                            else None,
                                        heatmap_array=np.array(
                                            Image.open(v['heatmap_path']))
                                            if v.get('heatmap_path') and
                                            os.path.exists(v['heatmap_path'])
                                            else None,
                                        doctor_name=v.get('doctor_diagnosis','')[:30]
                                            if v.get('doctor_diagnosis') else None,
                                        doctor_diagnosis=v.get('doctor_diagnosis'),
                                        doctor_prescription=v.get('doctor_prescription'),
                                        doctor_referral=v.get('doctor_referral'),
                                        doctor_notes=v.get('doctor_notes'),
                                        reviewed_at=v.get('reviewed_at'),
                                        chat_messages=msgs,
                                        visit_history=get_patient_visits(
                                            patient_email),
                                        output_path=tmp.name
                                    )
                                with open(pdf_path, 'rb') as f:
                                    st.session_state[pdf_key] = f.read()
                                st.rerun()
                    else:
                        st.download_button(
                            "Download Report",
                            data=st.session_state[pdf_key],
                            file_name=f"nayana_{v['case_id']}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"pdf_visit_{v['case_id']}"
                        )

    # ── TAB 3: Prescriptions ───────────────────────────────────
    with tab3:
        reviewed = [v for v in visits if v['status']=='Reviewed'
                    and v.get('doctor_prescription')]
        if not reviewed:
            st.info("No prescriptions yet.")
        else:
            st.markdown(
                '<div class="section-label">'
                'All Prescriptions</div>',
                unsafe_allow_html=True
            )
            for v in reversed(reviewed):
                st.markdown(f"""
                <div style="border-left:3px solid #818cf8;
                            padding:12px 16px;
                            margin-bottom:12px;
                            background:rgba(99,102,241,0.05);
                            border-radius:0 8px 8px 0;">
                    <div style="font-size:12px;color:#64748b;">
                        {v.get('reviewed_at','')[:11]}
                    </div>
                    <div style="font-size:15px;font-weight:600;
                                margin-top:4px;">
                        {v['doctor_prescription']}
                    </div>
                    <div style="font-size:13px;color:#94a3b8;
                                margin-top:4px;">
                        For: {v['doctor_diagnosis']}
                    </div>
                    <div style="font-size:12px;color:#64748b;
                                margin-top:2px;">
                        Referral: {v['doctor_referral']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 4: Trends ──────────────────────────────────────────
    with tab4:
        if len(visits) < 2:
            st.info("Need at least 2 visits to show trends.")
        else:
            st.markdown(
                '<div class="section-label">'
                'Risk Level Over Time</div>',
                unsafe_allow_html=True
            )
            fig1 = risk_trend_chart(patient_email, dark)
            if fig1:
                st.pyplot(fig1)
                plt.close()
            st.write("")
            st.markdown(
                '<div class="section-label">'
                'Disease Confidence Over Time</div>',
                unsafe_allow_html=True
            )
            fig2 = disease_trend_chart(patient_email, dark)
            if fig2:
                st.pyplot(fig2)
                plt.close()
            st.write("")
            st.markdown(
                '<div class="section-label">'
                'Visit Comparison</div>',
                unsafe_allow_html=True
            )
            rows = []
            for v in visits:
                det = v.get("detected_conditions", [])
                top = (max(det, key=lambda x: x[1])
                       if det else ("Normal", 0))
                risk = v['risk_level'].lower()
                level = ("High" if "high" in risk
                         or "specialist" in risk
                         else "Moderate" if "moderate" in risk
                         or "follow" in risk else "Low")
                rows.append({
                    "Date":        v['timestamp'][:11],
                    "Risk":        level,
                    "Top Finding": top[0],
                    "Confidence":  f"{top[1]*100:.0f}%",
                    "Status":      v['status']
                })
            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

    # ── TAB 5: My Profile ──────────────────────────────────────
    with tab5:
        st.markdown("**Update your medical profile**")
        st.caption(
            "This helps your doctor provide better care. "
            "Only your doctor can see this."
        )
        c1,c2 = st.columns(2)
        blood_opts = ["Not set","A+","A-","B+","B-",
                      "AB+","AB-","O+","O-"]
        blood_group = c1.selectbox(
            "Blood Group",
            blood_opts,
            index=blood_opts.index(
                profile.get("blood_group") or "Not set"
            )
        )
        known = c2.text_input(
            "Known Conditions",
            value=", ".join(profile.get("known_conditions",[])),
            placeholder="e.g. Diabetes, Hypertension"
        )
        family = st.text_input(
            "Family History",
            value=", ".join(profile.get("family_history",[])),
            placeholder="e.g. Glaucoma, Diabetic Retinopathy"
        )
        meds = st.text_input(
            "Current Medications",
            value=", ".join(
                profile.get("current_medications",[])),
            placeholder="e.g. Metformin, Atorvastatin"
        )
        allergies = st.text_input(
            "Allergies",
            value=", ".join(profile.get("allergies",[])),
            placeholder="e.g. Penicillin"
        )
        if st.button("Save Profile", type="primary"):
            update_patient_profile(patient_email, {
                "blood_group": (blood_group
                                if blood_group != "Not set"
                                else ""),
                "known_conditions": [
                    x.strip() for x in known.split(",")
                    if x.strip()
                ],
                "family_history": [
                    x.strip() for x in family.split(",")
                    if x.strip()
                ],
                "current_medications": [
                    x.strip() for x in meds.split(",")
                    if x.strip()
                ],
                "allergies": [
                    x.strip() for x in allergies.split(",")
                    if x.strip()
                ]
            })
            st.success("Profile saved!")


# ── Doctor view ────────────────────────────────────────────────
def render_doctor_patient_history(patient_email, doctor_name,
                                   doctor_email):
    """
    Renders the full patient file for a doctor.
    ONLY works if the doctor has an appointment with this patient.
    Doctor's private notes are shown here — patients never see these.
    """
    # ── Privacy check ──────────────────────────────────────────
    if not doctor_can_access_patient(doctor_email, patient_email):
        st.error(
            "Access denied. You can only view records of "
            "patients who have booked an appointment with you."
        )
        return

    record = get_patient_record(patient_email)
    visits = get_patient_visits(patient_email)
    dark   = st.session_state.get('dark_mode', True)

    if not record or not visits:
        st.info("No history found for this patient.")
        return

    profile = record["profile"]

    # ── Patient summary card ───────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.08);
                border:1px solid rgba(99,102,241,0.2);
                border-radius:12px;padding:16px 20px;
                margin-bottom:16px;">
        <div style="font-size:16px;font-weight:700;">
            {profile['name']} — Full Medical History
        </div>
        <div style="font-size:13px;color:#64748b;margin-top:4px;">
            {profile['age']}y · {profile['gender']} ·
            {len(visits)} visits ·
            Last seen {visits[-1]['timestamp'][:11]}
        </div>
        {f"<div style='margin-top:8px;font-size:12px;color:#94a3b8;'>Blood group: {profile['blood_group']}</div>" if profile.get('blood_group') else ""}
        {f"<div style='font-size:12px;color:#94a3b8;'>Known: {', '.join(profile['known_conditions'])}</div>" if profile.get('known_conditions') else ""}
        {f"<div style='font-size:12px;color:#94a3b8;'>Family history: {', '.join(profile['family_history'])}</div>" if profile.get('family_history') else ""}
        {f"<div style='font-size:12px;color:#94a3b8;'>Medications: {', '.join(profile['current_medications'])}</div>" if profile.get('current_medications') else ""}
        {f"<div style='font-size:12px;color:#f4a261;'>Allergies: {', '.join(profile['allergies'])}</div>" if profile.get('allergies') else ""}
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Visit Summary",
        "Prescriptions",
        "Trends",
        "Private Notes"
    ])

    # ── TAB 1: Visit summary ───────────────────────────────────
    with tab1:
        rows = []
        for v in visits:
            det = v.get("detected_conditions", [])
            top = (max(det, key=lambda x: x[1])
                   if det else ("Normal", 0))
            risk = v['risk_level'].lower()
            level = ("High" if "high" in risk
                     or "specialist" in risk
                     else "Moderate" if "moderate" in risk
                     or "follow" in risk else "Low")
            rows.append({
                "Date":        v['timestamp'][:11],
                "Risk":        level,
                "Top Finding": top[0],
                "Confidence":  f"{top[1]*100:.0f}%",
                "Diagnosis":   (v.get('doctor_diagnosis','Pending')[:30]
                                if v.get('doctor_diagnosis')
                                else 'Pending')
            })
        import pandas as pd
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

        st.write("")
        st.markdown("**Previous Diagnoses**")
        reviewed = [v for v in visits if v['status']=='Reviewed']
        if not reviewed:
            st.info("No reviewed cases yet.")
        for v in reversed(reviewed):
            st.markdown(f"""
            <div style="border-left:3px solid #818cf8;
                        padding:10px 16px;margin-bottom:10px;
                        background:rgba(99,102,241,0.05);
                        border-radius:0 8px 8px 0;">
                <div style="font-size:12px;color:#64748b;">
                    {v.get('reviewed_at','')}
                </div>
                <div style="font-size:14px;font-weight:600;
                            margin-top:2px;">
                    {v['doctor_diagnosis']}
                </div>
                <div style="font-size:13px;color:#94a3b8;">
                    Treatment: {v['doctor_prescription'] or 'None'}
                </div>
                <div style="font-size:13px;color:#94a3b8;">
                    Referral: {v['doctor_referral']}
                </div>
                {f"<div style='font-size:12px;color:#64748b;margin-top:4px;'>{v['doctor_notes']}</div>" if v['doctor_notes'] else ""}
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: Prescriptions ───────────────────────────────────
    with tab2:
        prescribed = [v for v in visits
                      if v['status']=='Reviewed'
                      and v.get('doctor_prescription')]
        if not prescribed:
            st.info("No prescriptions on record.")
        else:
            for v in reversed(prescribed):
                st.markdown(f"""
                <div style="border-left:3px solid #2d9e6b;
                            padding:12px 16px;
                            margin-bottom:12px;
                            background:rgba(45,158,107,0.05);
                            border-radius:0 8px 8px 0;">
                    <div style="font-size:12px;color:#64748b;">
                        {v.get('reviewed_at','')[:11]}
                    </div>
                    <div style="font-size:15px;font-weight:600;
                                margin-top:4px;">
                        {v['doctor_prescription']}
                    </div>
                    <div style="font-size:13px;color:#94a3b8;
                                margin-top:2px;">
                        Diagnosis: {v['doctor_diagnosis']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: Trends ──────────────────────────────────────────
    with tab3:
        if len(visits) < 2:
            st.info("Need at least 2 visits to show trends.")
        else:
            fig1 = risk_trend_chart(patient_email, dark)
            if fig1:
                st.pyplot(fig1)
                plt.close()
            fig2 = disease_trend_chart(patient_email, dark)
            if fig2:
                st.pyplot(fig2)
                plt.close()

    # ── TAB 4: Private Notes (doctor only) ─────────────────────
    with tab4:
        st.caption(
            "These notes are private and visible "
            "to doctors only. Patients cannot see this tab."
        )
        notes = record.get("continuity_notes", [])
        if notes:
            for n in reversed(notes):
                st.markdown(f"""
                <div style="border-left:3px solid #f4a261;
                            padding:10px 16px;
                            margin-bottom:10px;
                            background:rgba(244,162,97,0.06);
                            border-radius:0 8px 8px 0;">
                    <div style="font-size:12px;color:#64748b;">
                        {n['doctor_name']} · {n['date']}
                    </div>
                    <div style="font-size:14px;margin-top:4px;">
                        {n['note']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No notes yet.")

        st.write("")
        new_note = st.text_area(
            "Add a private note",
            placeholder="e.g. Patient has family history of DR. "
                        "Monitor annually even if results normal.",
            key=f"cont_note_{patient_email}_{id(patient_email)}"
        )
        if st.button("Save Note", type="primary",
                     key=f"save_note_{patient_email}_{id(patient_email)}"):
            if new_note.strip():
                add_continuity_note(
                    patient_email,if st.button("Download Full History as PDF",
             width='stretch',
             key="dl_full_history"):
    with st.spinner("Generating your complete health file..."):
        from report_generator import generate_report
        import zipfile
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zf:
            for v in visits:
                try:
                    case_probs = np.array(v.get('probs',[0]*8))
                    case_det   = v.get('detected_conditions',[])
                    msgs       = load_messages(v['case_id'])
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix='.pdf'
                    ) as tmp:
                        pdf_path = generate_report(
                            patient_name=v['patient_name'],
                            patient_age=v['patient_age'],
                            patient_gender=v['patient_gender'],
                            patient_email=patient_email,
                            symptoms=v['symptoms'],
                            quality_score=v['quality_score'],
                            quality_tips=[],
                            probs=case_probs,
                            detected_conditions=case_det,
                            risk_level=v['risk_level'],
                            risk_type=(
                                'high' if 'High' in v['risk_level']
                                or 'specialist' in v['risk_level']
                                else 'moderate' if 'Moderate' in v['risk_level']
                                else 'low'
                            ),
                            original_image_pil=Image.open(
                                v['image_path'])
                                if v.get('image_path') and
                                os.path.exists(v['image_path'])
                                else None,
                            heatmap_array=np.array(
                                Image.open(v['heatmap_path']))
                                if v.get('heatmap_path') and
                                os.path.exists(v['heatmap_path'])
                                else None,
                            doctor_diagnosis=v.get('doctor_diagnosis'),
                            doctor_prescription=v.get('doctor_prescription'),
                            doctor_referral=v.get('doctor_referral'),
                            doctor_notes=v.get('doctor_notes'),
                            reviewed_at=v.get('reviewed_at'),
                            chat_messages=msgs,
                            visit_history=visits,
                            output_path=tmp.name
                        )
                    zf.write(pdf_path,
                             f"nayana_{v['case_id']}.pdf")
                except Exception as e:
                    st.warning(
                        f"Could not generate report for "
                        f"{v['case_id']}: {e}")
        zip_buf.seek(0)
    st.download_button(
        "Download ZIP",
        data=zip_buf.read(),
        file_name=f"nayana_all_reports_"
                  f"{profile['name'].replace(' ','_')}.zip",
        mime="application/zip",
        width='stretch',
        key="dl_zip_btn"
    )
                    f"Dr. {doctor_name}",
                    new_note.strip()
                )
                st.success("Note saved!")
                st.rerun()
            else:
                st.warning("Please write a note first.")