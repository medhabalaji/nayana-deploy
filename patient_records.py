import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from PIL import Image
from database import (
    get_patient_record, get_patient_visits,
    get_risk_trend, update_patient_profile,
    add_continuity_note, load_cases
)
import json

MESSAGES_FILE = "messages.json"

def load_messages(case_id):
    if not os.path.exists(MESSAGES_FILE):
        return []
    with open(MESSAGES_FILE, "r") as f:
        data = json.load(f)
    return data.get(case_id, [])
DISEASE_NAMES = [
    'Normal', 'Diabetic Retinopathy', 'Glaucoma',
    'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other'
]
DISEASE_COLORS = [
    '#2d9e6b','#e63946','#f4a261','#457b9d',
    '#9b5de5','#f77f00','#00b4d8','#74c69d'
]

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
        trend = "Your eye health has worsened since your last visit. Please consult your doctor."
    else:
        trend = "Your eye health is stable compared to your last visit."

    # Compare disease confidence scores
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
                f"{name} risk {direction} from {prev_probs[i]*100:.0f}% to {last_probs[i]*100:.0f}%."
            )

    summary = trend
    if changes:
        summary += " " + " ".join(changes)
    else:
        summary += " No significant changes in individual conditions."

    return summary

def risk_trend_chart(patient_email):
    trend = get_risk_trend(patient_email)
    if len(trend) < 2:
        return None
    dates  = [t[0] for t in trend]
    scores = [t[1] for t in trend]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    clrs = {1:'#2d9e6b', 2:'#f4a261', 3:'#e63946'}
    ax.plot(dates, scores, color='#b7e4c7', linewidth=2, zorder=1)
    ax.scatter(dates, scores, c=[clrs[s] for s in scores], s=80, zorder=2)
    ax.set_yticks([1,2,3])
    ax.set_yticklabels(['Low','Moderate','High'], color='#2d6a4f', fontsize=9)
    ax.tick_params(axis='x', colors='#74c69d', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#b7e4c7')
    ax.set_ylim(0.5, 3.5)
    plt.tight_layout(pad=0.5)
    return fig

def disease_trend_chart(patient_email):
    visits = get_patient_visits(patient_email)
    if len(visits) < 2:
        return None
    dates = [v["timestamp"][:6] for v in visits]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    for i, (name, color) in enumerate(zip(DISEASE_NAMES[1:], DISEASE_COLORS[1:]), 1):
        scores = []
        for v in visits:
            probs = v.get("probs", [])
            scores.append(float(probs[i]) * 100 if len(probs) > i else 0)
        if max(scores) > 10:
            ax.plot(dates, scores, marker='o', label=name, color=color, linewidth=1.5, markersize=4)
    ax.set_ylabel("Confidence (%)", color='#2d6a4f', fontsize=9)
    ax.tick_params(colors='#2d6a4f', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#b7e4c7')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 105)
    plt.tight_layout(pad=0.5)
    return fig

def render_patient_health_record(user):
    patient_email = user['email']
    record  = get_patient_record(patient_email)
    visits  = get_patient_visits(patient_email)

    if not record:
        st.info("No health record found. Complete a screening to start your record.")
        return

    profile = record["profile"]

    # ── Profile card ───────────────────────────────────────────
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                border-radius:16px;padding:20px 24px;margin-bottom:24px;">
        <div style="font-size:20px;font-weight:800;font-family:'Space Grotesk',sans-serif;
                    color:#e2e8f0;margin-bottom:6px;">{profile['name']}</div>
        <div style="font-size:13px;color:#64748b;margin-bottom:12px;">
            {profile['age']}y · {profile['gender']} · ID: {profile['patient_id']} · Member since {profile['joined']}
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div><span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Total Screenings</span>
                 <div style="font-size:22px;font-weight:700;color:#818cf8;">{len(visits)}</div></div>
            <div><span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Last Screening</span>
                 <div style="font-size:14px;font-weight:600;color:#e2e8f0;">{visits[-1]['timestamp'][:11] if visits else 'None'}</div></div>
            <div><span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Blood Group</span>
                 <div style="font-size:14px;font-weight:600;color:#e2e8f0;">{profile.get('blood_group') or 'Not set'}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Download all reports button
    if st.button("Download All Reports as ZIP", use_container_width=True):
        import zipfile
        import tempfile
        from report_generator import generate_report
        with st.spinner("Generating all reports..."):
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w') as zf:
                for v in visits:
                    try:
                        case_probs = np.array(v.get('probs', [0]*8))
                        case_det   = v.get('detected_conditions', [])
                        msgs       = load_messages(v['case_id'])
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
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
                                original_image_pil=Image.open(v['image_path']) if v.get('image_path') and os.path.exists(v['image_path']) else None,
                                heatmap_array=np.array(Image.open(v['heatmap_path'])) if v.get('heatmap_path') and os.path.exists(v['heatmap_path']) else None,
                                doctor_diagnosis=v.get('doctor_diagnosis'),
                                doctor_prescription=v.get('doctor_prescription'),
                                doctor_referral=v.get('doctor_referral'),
                                doctor_notes=v.get('doctor_notes'),
                                reviewed_at=v.get('reviewed_at'),
                                chat_messages=msgs,
                                output_path=tmp.name
                            )
                        zf.write(pdf_path, f"nayana_{v['case_id']}.pdf")
                    except Exception as e:
                        st.warning(f"Could not generate report for {v['case_id']}: {e}")
            zip_buf.seek(0)
        st.download_button(
            "Download ZIP",
            data=zip_buf.read(),
            file_name=f"nayana_all_reports_{profile['name'].replace(' ','_')}.zip",
            mime="application/zip",
            use_container_width=True
        )
    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visit History", "Trends", "Profile"])

    # ── TAB 1: Overview ────────────────────────────────────────
    with tab1:
        if len(visits) >= 2:
            summary = generate_progression_summary(patient_email)
            st.markdown(f"""
            <div style="background:rgba(45,158,107,0.1);border:1px solid rgba(45,158,107,0.3);
                        border-radius:12px;padding:16px 20px;margin-bottom:20px;">
                <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                            color:#2d9e6b;margin-bottom:6px;">Progression Summary</div>
                <div style="font-size:14px;color:#e2e8f0;line-height:1.6;">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Risk Timeline</div>', unsafe_allow_html=True)
        if visits:
            for v in reversed(visits):
                risk  = v['risk_level']
                r_low = risk.lower()
                if "high" in r_low or "specialist" in r_low:
                    color, label = "#e63946", "High Risk"
                elif "moderate" in r_low or "follow" in r_low:
                    color, label = "#f4a261", "Moderate"
                else:
                    color, label = "#2d9e6b", "Low Risk"
                top_cond = ""
                det = v.get("detected_conditions", [])
                if det:
                    top = max(det, key=lambda x: x[1])
                    top_cond = f" — {top[0]} ({top[1]*100:.0f}%)"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:10px 0;
                            border-bottom:1px solid rgba(99,102,241,0.1);">
                    <div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                    <div style="font-size:13px;color:#64748b;min-width:100px;">{v['timestamp'][:11]}</div>
                    <div style="font-size:13px;font-weight:600;color:{color};min-width:80px;">{label}</div>
                    <div style="font-size:13px;color:#94a3b8;">{top_cond}</div>
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
                risk  = v['risk_level']
                r_low = risk.lower()
                if "high" in r_low or "specialist" in r_low:
                    badge = "High Risk"
                elif "moderate" in r_low or "follow" in r_low:
                    badge = "Moderate"
                else:
                    badge = "Low Risk"
                with st.expander(f"{v['timestamp']} — {badge} — {v['status']}"):
                    st.write(f"**Symptoms:** {v['symptoms']}")
                    st.write(f"**Image Quality:** {v['quality_score']}%")

                    c1,c2 = st.columns(2)
                    with c1:
                        if v.get('image_path') and os.path.exists(v['image_path']):
                            st.image(Image.open(v['image_path']), caption="Retinal scan", use_container_width=True)
                        if v.get('heatmap_path') and os.path.exists(v['heatmap_path']):
                            st.image(Image.open(v['heatmap_path']), caption="AI heatmap", use_container_width=True)
                    with c2:
                        st.markdown("**AI Findings:**")
                        probs = v.get("probs", [])
                        for i,(name,p) in enumerate(zip(DISEASE_NAMES, probs)):
                            if p > 0.3:
                                st.write(f"- {name}: {p*100:.1f}%")

                    if v['status'] == 'Reviewed':
                        st.divider()
                        st.markdown("**Doctor's Assessment:**")
                        st.write(f"**Diagnosis:** {v['doctor_diagnosis']}")
                        st.write(f"**Treatment:** {v['doctor_prescription'] or 'None'}")
                        st.write(f"**Referral:** {v['doctor_referral']}")
                        if v['doctor_notes']:
                            st.write(f"**Notes:** {v['doctor_notes']}")
                    else:
                        st.info("Awaiting doctor review.")

    # ── TAB 3: Trends ─────────────────────────────────────────
    with tab3:
        if len(visits) < 2:
            st.info("Need at least 2 visits to show trends.")
        else:
            st.markdown('<div class="section-label">Risk Level Over Time</div>', unsafe_allow_html=True)
            fig1 = risk_trend_chart(patient_email)
            if fig1:
                st.pyplot(fig1)
                plt.close()

            st.write("")
            st.markdown('<div class="section-label">Disease Confidence Over Time</div>', unsafe_allow_html=True)
            fig2 = disease_trend_chart(patient_email)
            if fig2:
                st.pyplot(fig2)
                plt.close()

            st.write("")
            st.markdown('<div class="section-label">Visit Comparison</div>', unsafe_allow_html=True)
            rows = []
            for v in visits:
                det   = v.get("detected_conditions", [])
                top   = max(det, key=lambda x: x[1]) if det else ("Normal", 0)
                risk  = v['risk_level']
                r_low = risk.lower()
                level = "High" if ("high" in r_low or "specialist" in r_low) else "Moderate" if ("moderate" in r_low or "follow" in r_low) else "Low"
                rows.append({
                    "Date":        v['timestamp'][:11],
                    "Risk":        level,
                    "Top Finding": top[0],
                    "Confidence":  f"{top[1]*100:.0f}%",
                    "Reviewed by": v['doctor_diagnosis'][:20] + "..." if v.get('doctor_diagnosis') else "Pending"
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── TAB 4: Profile ─────────────────────────────────────────
    with tab4:
        st.markdown("**Update your medical profile**")
        st.caption("This information helps your doctor provide better care.")

        c1,c2 = st.columns(2)
        blood_group = c1.selectbox("Blood Group",
            ["Not set","A+","A-","B+","B-","AB+","AB-","O+","O-"],
            index=["Not set","A+","A-","B+","B-","AB+","AB-","O+","O-"].index(
                profile.get("blood_group") or "Not set"))
        known = c2.text_input("Known Conditions",
            value=", ".join(profile.get("known_conditions", [])),
            placeholder="e.g. Diabetes, Hypertension")
        family = st.text_input("Family History",
            value=", ".join(profile.get("family_history", [])),
            placeholder="e.g. Glaucoma, Diabetic Retinopathy")
        meds = st.text_input("Current Medications",
            value=", ".join(profile.get("current_medications", [])),
            placeholder="e.g. Metformin, Atorvastatin")
        allergies = st.text_input("Allergies",
            value=", ".join(profile.get("allergies", [])),
            placeholder="e.g. Penicillin")

        if st.button("Save Profile", type="primary"):
            update_patient_profile(patient_email, {
                "blood_group":         blood_group if blood_group != "Not set" else "",
                "known_conditions":    [x.strip() for x in known.split(",") if x.strip()],
                "family_history":      [x.strip() for x in family.split(",") if x.strip()],
                "current_medications": [x.strip() for x in meds.split(",") if x.strip()],
                "allergies":           [x.strip() for x in allergies.split(",") if x.strip()]
            })
            st.success("Profile saved!")

def render_doctor_patient_history(patient_email, doctor_name):
    record = get_patient_record(patient_email)
    visits = get_patient_visits(patient_email)

    if not record or not visits:
        st.info("No history found for this patient.")
        return

    profile = record["profile"]

    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                border-radius:12px;padding:16px 20px;margin-bottom:16px;">
        <div style="font-size:16px;font-weight:700;color:#e2e8f0;">{profile['name']} — Full History</div>
        <div style="font-size:13px;color:#64748b;margin-top:4px;">
            {profile['age']}y · {profile['gender']} · {len(visits)} visits ·
            Last seen {visits[-1]['timestamp'][:11]}
        </div>
        {f"<div style='margin-top:8px;font-size:12px;color:#94a3b8;'>Known: {', '.join(profile['known_conditions'])}</div>" if profile.get('known_conditions') else ""}
        {f"<div style='font-size:12px;color:#94a3b8;'>Family history: {', '.join(profile['family_history'])}</div>" if profile.get('family_history') else ""}
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Visit Summary", "Trends", "Continuity Notes"])

    with tab1:
        rows = []
        for v in visits:
            det   = v.get("detected_conditions", [])
            top   = max(det, key=lambda x: x[1]) if det else ("Normal", 0)
            risk  = v['risk_level']
            r_low = risk.lower()
            level = "High" if ("high" in r_low or "specialist" in r_low) else "Moderate" if ("moderate" in r_low or "follow" in r_low) else "Low"
            rows.append({
                "Date":        v['timestamp'][:11],
                "Risk":        level,
                "Top Finding": top[0],
                "Confidence":  f"{top[1]*100:.0f}%",
                "Diagnosis":   v.get('doctor_diagnosis','Pending')[:30]
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.write("")
        st.markdown("**Previous Diagnoses**")
        reviewed = [v for v in visits if v['status'] == 'Reviewed']
        if not reviewed:
            st.info("No reviewed cases yet.")
        for v in reversed(reviewed):
            st.markdown(f"""
            <div style="border-left:3px solid #818cf8;padding:10px 16px;margin-bottom:10px;
                        background:rgba(99,102,241,0.05);border-radius:0 8px 8px 0;">
                <div style="font-size:12px;color:#64748b;">{v['reviewed_at']}</div>
                <div style="font-size:14px;font-weight:600;color:#e2e8f0;margin-top:2px;">{v['doctor_diagnosis']}</div>
                <div style="font-size:13px;color:#94a3b8;">Treatment: {v['doctor_prescription'] or 'None'}</div>
                <div style="font-size:13px;color:#94a3b8;">Referral: {v['doctor_referral']}</div>
                {f"<div style='font-size:12px;color:#64748b;margin-top:4px;'>{v['doctor_notes']}</div>" if v['doctor_notes'] else ""}
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        if len(visits) < 2:
            st.info("Need at least 2 visits to show trends.")
        else:
            from patient_records import risk_trend_chart, disease_trend_chart
            fig1 = risk_trend_chart(patient_email)
            if fig1:
                st.pyplot(fig1)
                plt.close()
            fig2 = disease_trend_chart(patient_email)
            if fig2:
                st.pyplot(fig2)
                plt.close()

    with tab3:
        notes = record.get("continuity_notes", [])
        if notes:
            for n in reversed(notes):
                st.markdown(f"""
                <div style="border-left:3px solid #f4a261;padding:10px 16px;margin-bottom:10px;
                            background:rgba(244,162,97,0.06);border-radius:0 8px 8px 0;">
                    <div style="font-size:12px;color:#64748b;">{n['doctor_name']} · {n['date']}</div>
                    <div style="font-size:14px;color:#e2e8f0;margin-top:4px;">{n['note']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No continuity notes yet.")

        st.write("")
        new_note = st.text_area("Add a continuity note",
            placeholder="e.g. Patient has family history of DR. Monitor annually.",
            key=f"cont_note_{patient_email}")
        if st.button("Save Note", key=f"save_note_{patient_email}", type="primary"):
            if new_note.strip():
                add_continuity_note(patient_email, f"Dr. {doctor_name}", new_note.strip())
                st.success("Note saved!")
                st.rerun()
            else:
                st.warning("Please write a note first.")