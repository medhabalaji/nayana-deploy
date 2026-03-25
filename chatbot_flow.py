"""
chatbot_flow.py
Conversational screening flow for Nayana — replaces the old Step-1 form.

State machine (stored in st.session_state['chat_stage']):
  greeting → mode_select → typing / voice / questionnaire → symptom_confirm → done
"""
import streamlit as st
from datetime import datetime
from voice_input import SYMPTOM_KEYWORDS, extract_symptoms, record_voice, LANGUAGES
from symptom_check import SYMPTOMS, triage

# ── helpers ────────────────────────────────────────────────────

def _bot_bubble(text: str, avatar: str = "N"):
    """Render a left-aligned Nayana chat bubble."""
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:16px;">
      <div style="width:36px;height:36px;border-radius:50%;
                  background:linear-gradient(135deg,#818cf8,#38bdf8);
                  display:flex;align-items:center;justify-content:center;
                  font-weight:800;font-size:14px;color:#fff;flex-shrink:0;">
        {avatar}
      </div>
      <div style="background:rgba(129,140,248,0.12);border:1px solid rgba(129,140,248,0.3);
                  border-radius:0 18px 18px 18px;padding:14px 18px;max-width:80%;
                  font-size:15px;line-height:1.6;color:#e2e8f0;">
        {text}
      </div>
    </div>
    """, unsafe_allow_html=True)


def _user_bubble(text: str):
    """Render a right-aligned user reply bubble."""
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin-bottom:16px;">
      <div style="background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.3);
                  border-radius:18px 0 18px 18px;padding:14px 18px;max-width:80%;
                  font-size:15px;line-height:1.6;color:#e2e8f0;">
        {text}
      </div>
    </div>
    """, unsafe_allow_html=True)


def _step_indicator(label: str):
    st.markdown(f"""
    <div style="font-size:11px;font-weight:700;letter-spacing:2px;
                text-transform:uppercase;color:#818cf8;margin-bottom:20px;
                display:flex;align-items:center;gap:8px;">
      <span style="width:8px;height:8px;border-radius:50%;
                   background:#818cf8;display:inline-block;"></span>
      {label}
    </div>
    """, unsafe_allow_html=True)


def _divider():
    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(129,140,248,0.15);'
        'margin:20px 0;">',
        unsafe_allow_html=True)


def _symptom_tag(sym: str):
    return (f'<span style="display:inline-block;background:rgba(129,140,248,0.18);'
            f'border:1px solid rgba(129,140,248,0.35);border-radius:20px;'
            f'padding:4px 14px;font-size:13px;font-weight:600;color:#a5b4fc;'
            f'margin:3px;">{sym}</span>')


# ── keyword matching helpers ────────────────────────────────────

EXTENDED_KEYWORDS: dict[str, str] = {
    # ── Blurred Vision ──────────────────────────────────────────
    "blurred vision": "Blurred Vision", "blurry": "Blurred Vision",
    "blur": "Blurred Vision", "cannot see": "Blurred Vision",
    "cant see": "Blurred Vision", "can't see": "Blurred Vision",
    "cant read": "Blurred Vision", "can't read": "Blurred Vision",
    "poor vision": "Blurred Vision", "weak vision": "Blurred Vision",
    "hazy": "Blurred Vision", "foggy vision": "Blurred Vision",
    "unclear": "Blurred Vision", "fuzzy": "Blurred Vision",
    "indistinct": "Blurred Vision", "dim vision": "Blurred Vision",
    "misty": "Blurred Vision", "vision loss": "Blurred Vision",
    "losing sight": "Blurred Vision", "sight going": "Blurred Vision",
    "not clear": "Blurred Vision", "hard to see": "Blurred Vision",
    "trouble seeing": "Blurred Vision", "difficulty seeing": "Blurred Vision",
    "eyesight weak": "Blurred Vision", "deteriorating vision": "Blurred Vision",
    "vision deteriorating": "Blurred Vision",

    # ── Eye Pain ────────────────────────────────────────────────
    "pain": "Eye Pain", "ache": "Eye Pain", "hurts": "Eye Pain",
    "hurting": "Eye Pain", "sore": "Eye Pain", "burning": "Eye Pain",
    "stinging": "Eye Pain", "sting": "Eye Pain", "throb": "Eye Pain",
    "throbbing": "Eye Pain", "agony": "Eye Pain", "discomfort": "Eye Pain",
    "eye ache": "Eye Pain", "eye hurts": "Eye Pain", "painful eye": "Eye Pain",
    "eye is sore": "Eye Pain", "eye sore": "Eye Pain",
    "pressure in eye": "Eye Pain", "eye pressure": "Eye Pain",
    "stabbing pain": "Eye Pain", "sharp pain": "Eye Pain",
    "dull pain": "Eye Pain", "aching eye": "Eye Pain",
    "eye discomfort": "Eye Pain", "eye aching": "Eye Pain",
    "pain behind eye": "Eye Pain", "pain in eye": "Eye Pain",
    "eye throbs": "Eye Pain", "eye is aching": "Eye Pain",

    # ── Redness ─────────────────────────────────────────────────
    "redness": "Redness", "red": "Redness", "pink": "Redness",
    "irritation": "Redness", "irritated": "Redness",
    "bloodshot": "Redness", "blood shot": "Redness",
    "inflamed": "Redness", "inflammation": "Redness",
    "red eye": "Redness", "pink eye": "Redness",
    "eyes are red": "Redness", "eyes look red": "Redness",
    "eye redness": "Redness", "eye is red": "Redness",
    "conjunctivitis": "Redness", "conjunctiva": "Redness",
    "eye irritation": "Redness", "irritating": "Redness",

    # ── Watering / Discharge ────────────────────────────────────
    "watering": "Watering", "tears": "Watering", "discharge": "Watering",
    "wet": "Watering", "tearing": "Watering", "teary": "Watering",
    "runny eye": "Watering", "leaking": "Watering", "mucus": "Discharge",
    "pus": "Discharge", "sticky": "Discharge", "gooey": "Discharge",
    "crust": "Crusting", "crusting": "Crusting", "crusted": "Crusting",
    "eye crust": "Crusting", "eye discharge": "Discharge",
    "morning crust": "Crusting", "eyes stuck": "Crusting",
    "eyes glued": "Crusting", "matted": "Crusting",
    "excessive tearing": "Watering", "excessive tears": "Watering",

    # ── Light Sensitivity ───────────────────────────────────────
    "light sensitivity": "Light Sensitivity",
    "sensitive to light": "Light Sensitivity",
    "brightness": "Light Sensitivity", "bright light": "Light Sensitivity",
    "photophobia": "Light Sensitivity", "photosensitive": "Light Sensitivity",
    "light hurts": "Light Sensitivity", "light painful": "Light Sensitivity",
    "can't tolerate light": "Light Sensitivity",
    "cannot tolerate light": "Light Sensitivity",
    "squinting in light": "Light Sensitivity",
    "dislike bright light": "Light Sensitivity",
    "avoid light": "Light Sensitivity", "sunlight hurts": "Light Sensitivity",
    "sunlight pain": "Light Sensitivity",

    # ── Double Vision ───────────────────────────────────────────
    "double vision": "Double Vision", "double": "Double Vision",
    "two images": "Double Vision", "diplopia": "Double Vision",
    "seeing double": "Double Vision", "ghosting": "Double Vision",
    "ghost image": "Double Vision", "overlapping images": "Double Vision",
    "images overlap": "Double Vision",

    # ── Floaters ────────────────────────────────────────────────
    "floaters": "Floaters", "floater": "Floaters",
    "specks": "Floaters", "cobwebs": "Floaters",
    "strings": "Floaters", "threads in vision": "Floaters",
    "drifting spots": "Floaters", "moving spots": "Floaters",
    "black strings": "Floaters", "grey strings": "Floaters",
    "stuff floating": "Floaters",

    # ── Dark Spots ──────────────────────────────────────────────
    "spots": "Dark Spots", "dark spots": "Dark Spots",
    "black spots": "Dark Spots", "shadow": "Dark Spots",
    "shadows": "Dark Spots", "blind spot": "Dark Spots",
    "missing patch": "Dark Spots", "dark patch": "Dark Spots",
    "grey area": "Dark Spots", "gray area": "Dark Spots",

    # ── Headache ─────────────────────────────────────────────────
    "headache": "Headache", "head pain": "Headache",
    "migraine": "Headache", "head ache": "Headache",
    "temple pain": "Headache", "forehead pain": "Headache",
    "brow ache": "Headache", "eye headache": "Headache",
    "head hurts": "Headache", "pressure headache": "Headache",

    # ── Itching ──────────────────────────────────────────────────
    "itching": "Itching", "itchy": "Itching", "scratch": "Itching",
    "itch": "Itching", "itchiness": "Itching", "rubbing eye": "Itching",
    "want to rub": "Itching", "urge to scratch": "Itching",
    "allergic itch": "Itching", "eye itch": "Itching",

    # ── Swelling ─────────────────────────────────────────────────
    "swelling": "Swelling", "swollen": "Swelling", "puffiness": "Swelling",
    "puffy": "Swelling", "eyelid swelling": "Swelling",
    "lid swelling": "Swelling", "swollen eyelid": "Swelling",
    "swollen lid": "Swelling", "bump on eye": "Swelling",
    "lump on eyelid": "Swelling", "lump near eye": "Swelling",
    "stye": "Swelling", "sty": "Swelling",
    "chalazion": "Swelling", "periorbital swelling": "Swelling",

    # ── Dryness ──────────────────────────────────────────────────
    "dryness": "Dryness", "dry": "Dryness", "gritty": "Dryness",
    "sandy feeling": "Dryness", "sandy eyes": "Dryness",
    "scratchy": "Dryness", "rough feeling in eye": "Dryness",
    "like sand in eye": "Dryness", "eye feels dry": "Dryness",

    # ── Eye Fatigue ──────────────────────────────────────────────
    "tired": "Eye Fatigue", "fatigue": "Eye Fatigue", "strain": "Eye Fatigue",
    "tired eyes": "Eye Fatigue", "eye strain": "Eye Fatigue",
    "digital eye strain": "Eye Fatigue", "screen fatigue": "Eye Fatigue",
    "weary eyes": "Eye Fatigue", "heavy eyelids": "Eye Fatigue",
    "eyes heavy": "Eye Fatigue", "eyes feel heavy": "Eye Fatigue",

    # ── Night Blindness ──────────────────────────────────────────
    "night blindness": "Night Blindness", "night": "Night Blindness",
    "dark": "Night Blindness", "can't see in dark": "Night Blindness",
    "cannot see in dark": "Night Blindness",
    "trouble at night": "Night Blindness", "poor night vision": "Night Blindness",
    "driving at night": "Night Blindness", "difficulty dark": "Night Blindness",

    # ── Tunnel Vision ────────────────────────────────────────────
    "tunnel": "Tunnel Vision", "peripheral": "Tunnel Vision",
    "tunnel vision": "Tunnel Vision", "loss of side vision": "Tunnel Vision",
    "peripheral loss": "Tunnel Vision", "side vision loss": "Tunnel Vision",
    "limited field": "Tunnel Vision", "narrow vision": "Tunnel Vision",

    # ── Halos ────────────────────────────────────────────────────
    "halo": "Halos", "halos": "Halos", "glare": "Halos",
    "rings around lights": "Halos", "ring around light": "Halos",
    "star burst": "Halos", "starburst": "Halos",
    "light scatter": "Halos", "light spreading": "Halos",
    "glaring lights": "Halos",

    # ── Yellowing ────────────────────────────────────────────────
    "yellow": "Yellowing", "yellowing": "Yellowing",
    "yellowish": "Yellowing", "jaundice": "Yellowing",
    "yellow eye": "Yellowing", "white of eye is yellow": "Yellowing",
    "sclera yellow": "Yellowing",

    # ── Color Blindness ──────────────────────────────────────────
    "colour blindness": "Color Issues", "color blindness": "Color Issues",
    "colour blind": "Color Issues", "color blind": "Color Issues",
    "colors look wrong": "Color Issues", "colours wrong": "Color Issues",
    "faded colours": "Color Issues", "faded colors": "Color Issues",
    "washed out": "Color Issues",

    # ── Sudden / Severe Loss ─────────────────────────────────────
    "sudden blindness": "Sudden Vision Loss", "sudden loss": "Sudden Vision Loss",
    "suddenly can't see": "Sudden Vision Loss",
    "vision gone": "Sudden Vision Loss", "lost vision": "Sudden Vision Loss",
    "went blind": "Sudden Vision Loss", "blackout": "Sudden Vision Loss",
    "curtain over eye": "Sudden Vision Loss",
    "curtain falling": "Sudden Vision Loss",
    "veil over vision": "Sudden Vision Loss",

    # ── Pain on Movement ─────────────────────────────────────────
    "pain moving eye": "Pain On Movement",
    "pain when moving eye": "Pain On Movement",
    "hurts to move eye": "Pain On Movement",
    "eye movement painful": "Pain On Movement",
    "painful when look around": "Pain On Movement",

    # ── Squinting ────────────────────────────────────────────────
    "squinting": "Squinting", "squint": "Squinting",
    "cross eyed": "Squinting", "crossed eye": "Squinting",
    "eyes not aligned": "Squinting", "eye turn": "Squinting",
    "lazy eye": "Squinting", "amblyopia": "Squinting",
    "strabismus": "Squinting",

    # ── Generic ──────────────────────────────────────────────────
    "something in my eye": "Foreign Body Sensation",
    "foreign body": "Foreign Body Sensation",
    "feels like something": "Foreign Body Sensation",
    "grit in eye": "Foreign Body Sensation",
    "flashes": "Floaters", "flash of light": "Floaters",
    "lightning": "Floaters",
}


def _match_keywords(text: str) -> list[str]:
    t = text.lower()
    found = []
    for kw, sym in EXTENDED_KEYWORDS.items():
        if kw in t and sym not in found:
            found.append(sym)
    return found if found else []


def _closest_keywords(text: str, n: int = 5) -> list[str]:
    """Return up to n symptom display names that partially overlap with the text."""
    t_words = set(text.lower().split())
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for kw, sym in EXTENDED_KEYWORDS.items():
        kw_words = set(kw.split())
        overlap = len(t_words & kw_words)
        if overlap and sym not in seen:
            scored.append((overlap, sym))
            seen.add(sym)
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:n]]


# ── Questionnaire definition ────────────────────────────────────

QUESTIONNAIRE: list[tuple[str, str]] = [
    # (question, symptom_label_if_yes)
    ("Do you have redness in your eye?",                 "Redness"),
    ("Is your eye swollen or puffy?",                    "Swelling"),
    ("Do you notice yellowing of the white part?",       "Yellowing"),
    ("Does your vision appear cloudy or blurry?",        "Blurred Vision"),
    ("Is your eyelid drooping?",                         "Eyelid Drooping"),
    ("Do you have pain inside the eye?",                 "Eye Pain"),
    ("Do you see flashes of light?",                     "Floaters"),
    ("Do you see floaters or moving spots?",             "Floaters"),
    ("Do you have difficulty seeing at night?",          "Night Blindness"),
    ("Are your eyes itchy or irritated?",                "Itching"),
    ("Do your eyes water excessively or produce discharge?", "Watering"),
    ("Do you experience headaches near your eyes?",      "Headache"),
    ("Are your eyes dry or gritty?",                     "Dryness"),
    ("Do you feel eye strain after reading or screens?", "Eye Fatigue"),
    ("Do you see double?",                               "Double Vision"),
    ("Are you sensitive to bright lights?",              "Light Sensitivity"),
    ("Has your vision suddenly changed?",                "Sudden Vision Loss"),
    ("Do you see halos or rings around lights?",         "Halos"),
    ("Have you lost peripheral (side) vision?",          "Tunnel Vision"),
    ("Does your eye hurt when you move it?",             "Pain On Movement"),
]


# ── triage from collected symptoms ─────────────────────────────

_FUNDUS_TRIGGERS = {
    "Eye Pain", "Floaters", "Night Blindness",
    "Sudden Vision Loss", "Tunnel Vision", "Pain On Movement",
}

def _triage_from_symptoms(syms: list[str]) -> dict:
    for s in syms:
        if s in _FUNDUS_TRIGGERS:
            return {"type": "fundus", "reason": f"Flagged: {s}"}
    return {"type": "front", "reason": "No internal symptoms reported"}


# ── main render function ────────────────────────────────────────

def render_chatbot_screening(user: dict):
    """
    Drop-in replacement for the old Step 1 block in app.py.
    Sets:  symp_final, triage, pname, page_, pgender, screening_step (→ 2)
    """
    # ── pre-populate patient info silently ──────────────────────
    if 'pname'   not in st.session_state:
        st.session_state['pname']   = user.get('name',   '')
    if 'page_'   not in st.session_state:
        st.session_state['page_']   = user.get('age',    30)
    if 'pgender' not in st.session_state:
        st.session_state['pgender'] = user.get('gender', 'Other')

    # ── init FSM ────────────────────────────────────────────────
    if 'chat_stage'        not in st.session_state:
        st.session_state['chat_stage']        = 'greeting'
    if 'chat_symptoms'     not in st.session_state:
        st.session_state['chat_symptoms']     = []
    if 'quest_index'       not in st.session_state:
        st.session_state['quest_index']       = 0
    if 'chat_raw_text'     not in st.session_state:
        st.session_state['chat_raw_text']     = ''
    if 'chat_clarify_text' not in st.session_state:
        st.session_state['chat_clarify_text'] = ''

    stage = st.session_state['chat_stage']

    # ── chat window container ───────────────────────────────────
    st.markdown(
        '<div style="max-width:680px;margin:0 auto;">',
        unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # STAGE: greeting
    # ═══════════════════════════════════════════════════════════
    if stage == 'greeting':
        _step_indicator("Step 1 of 3 — Getting Started")
        _bot_bubble(
            f"Hi <strong>{user.get('name','there')}</strong> 👋 — I'm <strong>Nayana</strong>, "
            "your AI eye-screening assistant.<br><br>"
            "Would you like a <em>routine checkup</em>, or do you have "
            "specific <em>symptoms</em> you want to report?"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Routine Checkup", use_container_width=True,
                         type="primary", key="cb_routine"):
                st.session_state['chat_stage']    = 'routine_confirm'
                st.session_state['chat_symptoms'] = []
                st.rerun()
        with c2:
            if st.button("🔴 I Have Symptoms", use_container_width=True,
                         key="cb_symptoms"):
                st.session_state['chat_stage'] = 'mode_select'
                st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STAGE: routine_confirm
    # ═══════════════════════════════════════════════════════════
    elif stage == 'routine_confirm':
        _step_indicator("Step 1 of 3 — Routine Checkup")
        _user_bubble("Routine Checkup")
        _bot_bubble(
            "Great! We'll start with a quick front-eye photo.<br><br>"
            "Before we do — would you like to complete your health profile? "
            "It helps the AI give more accurate recommendations. "
            "Or we can skip it and go straight to the photo. 📷"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📋 Complete Profile", use_container_width=True,
                         key="cb_rp_profile"):
                st.session_state['symp_final'] = "None — routine checkup"
                st.session_state['triage']     = {"type": "front",
                                                   "reason": "Routine checkup"}
                st.session_state['return_to_screening'] = True
                st.session_state['page']                = 'health_record'
                st.rerun()
        with c2:
            if st.button("Yes, let's go! →", use_container_width=True,
                         type="primary", key="cb_rp_yes"):
                st.session_state['symp_final'] = "None — routine checkup"
                st.session_state['triage']     = {"type": "front",
                                                   "reason": "Routine checkup"}
                st.session_state['screening_step'] = 2
                st.rerun()
        with c3:
            if st.button("← Go back", use_container_width=True,
                         key="cb_rp_no"):
                st.session_state['chat_stage'] = 'greeting'
                st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STAGE: mode_select
    # ═══════════════════════════════════════════════════════════
    elif stage == 'mode_select':
        _step_indicator("Step 1 of 3 — Symptoms")
        _user_bubble("I have symptoms")
        # Warm, supportive message first
        _bot_bubble(
            "I'm here to help you. 💙 Let's walk through these symptoms "
            "step by step so we can guide you toward the right care.<br><br>"
            "How would you like to describe your symptoms?"
        )
        # Then the method options
        _bot_bubble(
            "<strong>⌨️ Type</strong> — free-text, I'll match your words<br>"
            "<strong>🎙️ Voice</strong> — speak in your language<br>"
            "<strong>📋 Questionnaire</strong> — I'll ask yes/no questions"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⌨️ Type", use_container_width=True,
                         type="primary", key="cb_mode_type"):
                st.session_state['chat_stage'] = 'typing'
                st.rerun()
        with c2:
            if st.button("🎙️ Voice", use_container_width=True,
                         key="cb_mode_voice"):
                st.session_state['chat_stage'] = 'voice'
                st.rerun()
        with c3:
            if st.button("📋 Questionnaire", use_container_width=True,
                         key="cb_mode_quest"):
                st.session_state['chat_stage'] = 'questionnaire'
                st.session_state['quest_index'] = 0
                st.session_state['chat_symptoms'] = []
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back", key="cb_mode_back"):
            st.session_state['chat_stage'] = 'greeting'
            st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STAGE: typing
    # ═══════════════════════════════════════════════════════════
    elif stage == 'typing':
        _step_indicator("Step 1 of 3 — Describe Symptoms")
        _user_bubble("I'd like to type my symptoms")

        # show confirmed symptoms so far
        existing = st.session_state.get('chat_symptoms', [])
        if existing:
            tags = " ".join(_symptom_tag(s) for s in existing)
            _bot_bubble(
                f"Got it so far: {tags}<br><br>"
                "Add more, or confirm below when you're done."
            )
        else:
            _bot_bubble(
                "Please describe what you're experiencing in your own words — "
                "e.g. <em>\"my eye is red and blurry\"</em>, "
                "<em>\"pain when I move my eye\"</em>. "
                "I'll pick out the key symptoms."
            )

        COMMON = [
            "Blurred Vision", "Eye Pain", "Redness",
            "Watering / Discharge", "Light Sensitivity", "Double Vision",
            "Floaters / Dark Spots", "Headache", "Itching",
            "Swelling", "Dryness", "Night Blindness",
            "Halos Around Lights", "Tunnel Vision",
        ]
        picked = st.multiselect(
            "Quick-pick common symptoms (optional)",
            COMMON,
            default=[s for s in existing if s in COMMON],
            key="cb_multi"
        )
        typed = st.text_input(
            "Or describe in your own words",
            placeholder="e.g. my right eye has been red and painful since yesterday",
            key="cb_typed"
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Analyse →", type="primary",
                         use_container_width=True, key="cb_type_analyse"):
                combined = list(set(picked))
                if typed.strip():
                    found = _match_keywords(typed)
                    if found:
                        combined = list(set(combined + found))
                        st.session_state['chat_symptoms'] = combined
                        st.session_state['chat_stage']    = 'symptom_confirm'
                    else:
                        # no match → clarify
                        st.session_state['chat_symptoms']     = combined
                        st.session_state['chat_clarify_text'] = typed.strip()
                        st.session_state['chat_stage']        = 'typing_clarify'
                elif combined:
                    st.session_state['chat_symptoms'] = combined
                    st.session_state['chat_stage']    = 'symptom_confirm'
                else:
                    st.warning("Please pick or type at least one symptom.")
                st.rerun()
        with c2:
            if st.button("← Back", use_container_width=True,
                         key="cb_type_back"):
                st.session_state['chat_stage'] = 'mode_select'
                st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STAGE: typing_clarify
    # ═══════════════════════════════════════════════════════════
    elif stage == 'typing_clarify':
        _step_indicator("Step 1 of 3 — Clarification")
        original = st.session_state.get('chat_clarify_text', '')
        _user_bubble(original)
        closest = _closest_keywords(original, 6)
        if closest:
            opts_html = " ".join(_symptom_tag(s) for s in closest)
            _bot_bubble(
                f"I didn't quite catch that — did you mean any of these?<br><br>"
                f"{opts_html}"
            )
            selected = []
            cols = st.columns(min(len(closest), 3))
            for i, sym in enumerate(closest):
                with cols[i % 3]:
                    if st.button(sym, key=f"cb_clar_{i}"):
                        st.session_state['chat_symptoms'].append(sym)
                        st.session_state['chat_stage'] = 'typing'
                        st.rerun()
        else:
            _bot_bubble(
                "I couldn't match that to a known symptom. "
                "No worries — I'll save your description as-is for the doctor to review."
            )
            # store raw text as the symptom anyway
            st.session_state['chat_symptoms'].append(original)
            if st.button("Continue →", type="primary", key="cb_clar_skip"):
                st.session_state['chat_stage'] = 'symptom_confirm'
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Keep typing instead", key="cb_clar_back"):
            st.session_state['chat_stage'] = 'typing'
            st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STAGE: voice
    # ═══════════════════════════════════════════════════════════
    elif stage == 'voice':
        _step_indicator("Step 1 of 3 — Voice Input")
        _user_bubble("I'd like to use voice")
        _bot_bubble(
            "Sure! Choose your language and tap <strong>Start Recording</strong>. "
            "Speak naturally — I'll listen for up to 15 seconds."
        )

        # Map each language to its name in the native script
        NATIVE_LABELS = {
            "Kannada": "ಕನ್ನಡ (Kannada)",
            "Hindi":   "हिन्दी (Hindi)",
            "Tamil":   "தமிழ் (Tamil)",
            "Telugu":  "తెలుగు (Telugu)",
            "English": "English",
        }
        lang_options = list(LANGUAGES.keys())
        lang = st.selectbox(
            "Choose your language / ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ",
            lang_options,
            format_func=lambda k: NATIVE_LABELS.get(k, k),
            key="cb_voice_lang"
        )

        if st.button("🎙️ Start Recording", type="primary",
                     key="cb_rec_start"):
            with st.spinner(f"Listening in {lang}…"):
                res = record_voice(lang)
            if res["success"]:
                found = extract_symptoms(res.get("english_text", res["text"]))
                st.session_state['chat_symptoms'] = list(set(
                    st.session_state.get('chat_symptoms', []) + found
                ))
                st.session_state['voice_memory']  = res
                st.session_state['raw_speech']    = res['text']
                _user_bubble(f'"{res["text"]}"')
                if lang != "English":
                    st.caption(f"Translated: {res.get('english_text','')}")
                tags = " ".join(_symptom_tag(s) for s in found) if found else "(none detected)"
                _bot_bubble(
                    f"Got it! Symptoms I detected: {tags}<br><br>"
                    "Confirm below or keep recording."
                )
            else:
                st.error(res["error"])

        existing = st.session_state.get('chat_symptoms', [])
        if existing:
            tags = " ".join(_symptom_tag(s) for s in existing)
            st.markdown(
                f'<div style="margin:12px 0;">Collected: {tags}</div>',
                unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm Symptoms →", type="primary",
                         use_container_width=True, key="cb_voice_confirm"):
                if existing:
                    st.session_state['chat_stage'] = 'symptom_confirm'
                    st.rerun()
                else:
                    st.warning("Please record your symptoms first.")
        with c2:
            if st.button("← Back", use_container_width=True,
                         key="cb_voice_back"):
                st.session_state['chat_stage'] = 'mode_select'
                st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STAGE: questionnaire
    # ═══════════════════════════════════════════════════════════
    elif stage == 'questionnaire':
        _step_indicator("Step 1 of 3 — Rapid Questionnaire")
        _user_bubble("Answer a quick questionnaire")
        _bot_bubble(
            "Please check any of the following that apply to you. "
            "You can select multiple options."
        )

        with st.container():
            st.markdown('<div style="background:rgba(255,255,255,0.03);padding:20px;border-radius:16px;border:1px solid rgba(129,140,248,0.2);">', unsafe_allow_html=True)
            
            # We'll show checkboxes for all items in QUESTIONNAIRE
            selected_symptoms = []
            cols = st.columns(2)
            for i, (question, symptom) in enumerate(QUESTIONNAIRE):
                with cols[i % 2]:
                    # Default state is based on if this symptom was already collected (e.g. via typing)
                    was_picked = symptom in st.session_state.get('chat_symptoms', [])
                    if st.checkbox(question, value=was_picked, key=f"q_check_{i}"):
                        selected_symptoms.append(symptom)
            
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm & Continue →", type="primary", use_container_width=True, key="cb_q_finish"):
                # Merge with existing symptoms and deduplicate
                st.session_state['chat_symptoms'] = list(set(
                    st.session_state.get('chat_symptoms', []) + selected_symptoms
                ))
                st.session_state['chat_stage'] = 'symptom_confirm'
                st.rerun()
        with c2:
            if st.button("← Back", use_container_width=True, key="cb_q_back"):
                st.session_state['chat_stage'] = 'mode_select'
                st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STAGE: symptom_confirm
    # ═══════════════════════════════════════════════════════════
    elif stage == 'symptom_confirm':
        _step_indicator("Step 1 of 3 — Confirm & Continue")
        syms = st.session_state.get('chat_symptoms', [])

        if syms:
            tags = " ".join(_symptom_tag(s) for s in syms)
            _bot_bubble(
                f"Here's what I've recorded for you:<br><br>"
                f"{tags}<br><br>"
                "Does this look right? You can go back and edit, "
                "or confirm and proceed to the eye photo."
            )
        else:
            _bot_bubble(
                "It looks like no specific symptoms were captured. "
                "You can go back to add some, or proceed with no symptoms."
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm & Continue →", type="primary",
                         use_container_width=True, key="cb_sc_confirm"):
                final = ", ".join(syms) if syms else "Not specified"
                st.session_state['symp_final'] = final
                st.session_state['triage']     = _triage_from_symptoms(syms)

                # also update profile check flag same as old code
                from database import get_patient_record
                record  = get_patient_record(user['email'])
                profile = (record.get('profile', {}) if record else {})
                if (not profile.get('blood_group') or
                        not profile.get('known_conditions')):
                    st.session_state['show_profile_prompt'] = True
                    st.session_state['chat_stage'] = 'profile_prompt'
                else:
                    st.session_state['screening_step'] = 2
                st.rerun()
        with c2:
            if st.button("← Edit Symptoms", use_container_width=True,
                         key="cb_sc_edit"):
                st.session_state['chat_stage']    = 'mode_select'
                st.session_state['chat_symptoms'] = []
                st.session_state['quest_index']   = 0
                st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STAGE: profile_prompt (mirrors old show_profile_prompt logic)
    # ═══════════════════════════════════════════════════════════
    elif stage == 'profile_prompt':
        _step_indicator("Step 1 of 3 — One More Thing")
        _bot_bubble(
            "Your medical profile is incomplete. Filling it in helps "
            "the AI give you more accurate recommendations.<br><br>"
            "Would you like to complete it now, or skip for this screening?"
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Complete Profile", type="primary",
                         use_container_width=True, key="cb_pp_yes"):
                st.session_state['show_profile_prompt'] = False
                st.session_state['return_to_screening'] = True
                st.session_state['page']                = 'health_record'
                st.rerun()
        with c2:
            if st.button("Skip for Now", use_container_width=True,
                         key="cb_pp_skip"):
                st.session_state['show_profile_prompt'] = False
                st.session_state['screening_step']      = 2
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
