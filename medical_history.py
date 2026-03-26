import streamlit as st
import time
from database import update_patient_profile, get_patient_record, _register_visit

def render_medical_history_form(user):
    """
    Renders a dedicated, professional medical history intake form.
    Provides standard questions for the patient clinical profile.
    """
    patient_email = user['email']
    record = get_patient_record(patient_email)
    
    # Ensure record exists
    if not record:
        _register_visit(
            patient_email, 
            user.get('name',''), 
            user.get('age',30), 
            user.get('gender','Other'), 
            "PROFILE_INIT"
        )
        record = get_patient_record(patient_email)
    
    profile = record["profile"]

    st.markdown("""
        <div style="text-align:center;margin-bottom:30px;">
            <h2 style="color:#818cf8;margin-bottom:8px;">Clinical Profile Intake</h2>
            <p style="color:#64748b;font-size:14px;">Please complete your medical history to help our AI and doctors provide accurate guidance.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="background:rgba(255,255,255,0.02);padding:24px;border-radius:16px;border:1px solid rgba(129,140,248,0.1);">', unsafe_allow_html=True)
        
        # 1. General Health
        st.markdown("### 1. General Health")
        blood_opts = ["Not set","A+","A-","B+","B-","AB+","AB-","O+","O-"]
        blood_group = st.selectbox(
            "What is your blood group?",
            blood_opts,
            index=blood_opts.index(profile.get("blood_group") or "Not set")
        )
        
        st.divider()

        # 2. Personal History
        st.markdown("### 2. Personal Medical History")
        st.caption("Do you currently have or are you being treated for any of the following?")
        known_str = ", ".join(profile.get("known_conditions", []))
        known = st.text_input(
            "Known Conditions",
            value=known_str,
            placeholder="e.g. Diabetes, Hypertension, Thyroid issues"
        )

        st.divider()

        # 3. Family History
        st.markdown("### 3. Family Medical History")
        st.caption("Is there a history of eye diseases in your immediate family? (Parents/Siblings)")
        family_str = ", ".join(profile.get("family_history", []))
        family = st.text_input(
            "Eye Diseases in Family",
            value=family_str,
            placeholder="e.g. Glaucoma, Cataract, Diabetic Retinopathy"
        )

        st.divider()

        # 4. Medications & Allergies
        st.markdown("### 4. Medications & Allergies")
        meds_str = ", ".join(profile.get("current_medications", []))
        meds = st.text_area(
            "Are you currently taking any long-term medications?",
            value=meds_str,
            placeholder="e.g. Metformin, Insulin, Blood pressure meds",
            height=80
        )
        
        allergies_str = ", ".join(profile.get("allergies", []))
        allergies = st.text_input(
            "Do you have any known allergies?",
            value=allergies_str,
            placeholder="e.g. Penicillin, Sulfa drugs, Latex"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("Save & Return to Screening →", type="primary", use_container_width=True):
            update_patient_profile(patient_email, {
                "blood_group": blood_group if blood_group != "Not set" else "",
                "known_conditions": [x.strip() for x in known.split(",") if x.strip()],
                "family_history": [x.strip() for x in family.split(",") if x.strip()],
                "current_medications": [x.strip() for x in meds.split(",") if x.strip()],
                "allergies": [x.strip() for x in allergies.split(",") if x.strip()]
            })
            st.success("Profile updated successfully!")
            time.sleep(1)
            st.session_state['page'] = 'screening'
            st.session_state['screening_step'] = 2
            st.rerun()

    with c2:
        if st.button("Skip for now", use_container_width=True):
            st.session_state['page'] = 'screening'
            st.session_state['screening_step'] = 2
            st.rerun()
