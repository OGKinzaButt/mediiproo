"""
app.py
------
MediGuide AI -- Streamlit front end.

Flow:
  1. Gate screen: user pastes their own OpenAI API key (kept only in
     st.session_state for this session -- never written to disk).
  2. Sidebar: app info, disclaimer, model + language + cache settings.
  3. Form: patient details, symptoms, duration, severity, etc.
  4. On submit: build the LLMChain, run it, parse the JSON safely, stream
     a narrative explanation live, then render the results dashboard.
"""

import time
import streamlit as st

from src import config
from src.prompts import ASSESSMENT_PROMPT_TEMPLATE
from src.chains import build_llm, build_assessment_chain, stream_narrative
from src.cache_manager import set_cache_mode
from src.utils import safe_json_parse, validate_schema, format_symptom_list

st.set_page_config(
    page_title="MediGuide AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Shared CSS -- cream background + card styling used by the gate screen,
# plus the orange primary-button look used throughout the app.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg,#f2eee3 0%,#eee9dc 100%); }
#MainMenu, footer, header { visibility: hidden; }

/* The gate card is a real Streamlit bordered container so everything
   (icon, text, input, button) lives inside ONE actual DOM element --
   this is what makes the white rounded card render reliably. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#ffffff !important; border-radius:20px !important;
    border:none !important; box-shadow:0 20px 40px rgba(0,0,0,0.10);
    padding:0.4rem 0.6rem;
}

.gate-icon {
    width:52px; height:52px; border-radius:12px; background:#1f4f45;
    display:flex; align-items:center; justify-content:center; margin-bottom:16px;
}
.gate-eyebrow {
    color:#1f4f45; font-size:12px; font-weight:700; letter-spacing:1.5px;
    text-transform:uppercase; margin-bottom:6px;
}
.gate-title {
    font-family:Georgia, 'Times New Roman', serif; font-size:27px;
    font-weight:700; color:#1c1c1c; line-height:1.28; margin-bottom:14px;
}
.gate-desc { color:#555; font-size:14.5px; line-height:1.55; margin-bottom:18px; }
.gate-label { font-weight:600; font-size:13.5px; color:#222; margin-bottom:2px; }
.gate-link { color:#666; font-size:12.5px; margin:10px 0 4px 0; }
.gate-link a { color:#1f4f45; font-weight:600; }
.gate-lock { color:#8a8a8a; font-size:12px; margin-top:16px; line-height:1.5;
             border-top:1px dashed #e5e0d3; padding-top:14px; }

div.stButton > button {
    background:#c4681f !important; color:white !important; border:none !important;
    border-radius:10px !important; padding:0.6rem 1rem !important;
    font-weight:700 !important; width:100%;
}
div.stButton > button:hover { background:#a8571a !important; }

.urgency-badge {
    display:inline-block; padding:6px 14px; border-radius:8px;
    color:white; font-weight:700; font-size:14px; letter-spacing:0.5px;
}
</style>
""", unsafe_allow_html=True)

HEART_ICON = (
    '<div class="gate-icon"><svg width="22" height="22" viewBox="0 0 24 24" '
    'fill="none" stroke="white" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 '
    '5.5 0 0 0-7.8 7.8l1 1L12 21l7.8-7.8 1-1a5.5 5.5 0 0 0 0-7.8z"/></svg></div>'
)

if "api_key" not in st.session_state:
    st.session_state.api_key = config.ENV_OPENAI_API_KEY or ""
if "key_error" not in st.session_state:
    st.session_state.key_error = False
if "show_key" not in st.session_state:
    st.session_state.show_key = False

# ---------------------------------------------------------------------------
# GATE SCREEN -- shown until a key is present in session_state. Everything
# below lives inside ONE bordered container so the card renders as a
# single, correctly-nested block (not several stray <div> fragments).
# ---------------------------------------------------------------------------
if not st.session_state.api_key:
    st.write("")
    left, mid, right = st.columns([1, 1.3, 1])
    with mid:
        with st.container(border=True):
            st.markdown(HEART_ICON, unsafe_allow_html=True)
            st.markdown('<div class="gate-eyebrow">MEDIGUIDE AI &middot; SETUP</div>', unsafe_allow_html=True)
            st.markdown('<div class="gate-title">Connect your OpenAI key<br>to begin</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="gate-desc">MediGuide AI runs on your own OpenAI account. '
                'Drop in your API key below &mdash; it stays in this browser session only, '
                'and is never saved or sent anywhere else.</div>',
                unsafe_allow_html=True,
            )

            label_col, toggle_col = st.columns([3, 1])
            with label_col:
                st.markdown('<div class="gate-label">OpenAI API Key</div>', unsafe_allow_html=True)
            with toggle_col:
                st.checkbox("Show", value=st.session_state.show_key, key="show_key")

            key_input = st.text_input(
                "OpenAI API Key",
                placeholder="sk-...",
                type="default" if st.session_state.show_key else "password",
                label_visibility="collapsed",
                key="gate_key_input",
            )

            st.markdown(
                '<div class="gate-link">Don\'t have a key? Create one at '
                '<a href="https://platform.openai.com" target="_blank">platform.openai.com</a>.</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.key_error:
                st.error("Please enter your OpenAI API key.")

            if st.button("Start Assistant", key="start_assistant_btn"):
                if key_input and key_input.strip().startswith("sk-"):
                    st.session_state.api_key = key_input.strip()
                    st.session_state.key_error = False
                    st.rerun()
                else:
                    st.session_state.key_error = True
                    st.rerun()

            st.markdown(
                '<div class="gate-lock">🔒 Your key lives only in this session\'s memory '
                '&mdash; it disappears the moment you close or reload the tab.</div>',
                unsafe_allow_html=True,
            )
    st.stop()
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR -- app info, disclaimer, model + language + cache configuration.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🩺 {config.APP_NAME}")
    st.caption(config.APP_TAGLINE)
    st.warning(config.DISCLAIMER)

    st.markdown("---")
    st.markdown("**Model configuration**")
    model_name = st.selectbox("OpenAI model", config.AVAILABLE_MODELS, index=0)
    answer_language = st.selectbox("Answer language", config.LANGUAGES, index=0)

    st.markdown("---")
    st.markdown("**Caching**")
    cache_mode = st.selectbox("Cache mode", config.CACHE_MODES, index=1)
    st.caption(set_cache_mode(cache_mode))

    st.markdown("---")
    if st.button("Disconnect API key"):
        st.session_state.api_key = ""
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN AREA -- header + disclaimer + patient form.
# ---------------------------------------------------------------------------
st.title(f"🩺 {config.APP_NAME}")
st.caption(config.APP_TAGLINE)
st.info(config.DISCLAIMER)

with st.form("patient_form"):
    st.subheader("Patient information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input("Age", placeholder="e.g. 32")
    with col2:
        gender = st.selectbox("Gender", config.GENDERS)

    symptoms = st.multiselect("Symptoms", config.SYMPTOM_OPTIONS)
    extra_symptoms = st.text_input("Other symptoms (free text, optional)")

    col3, col4 = st.columns(2)
    with col3:
        duration = st.selectbox("Duration of symptoms", config.DURATION_OPTIONS)
    with col4:
        severity = st.slider("Severity (1 = mild, 10 = severe)", 1, 10, 3)

    conditions = st.text_area("Existing medical conditions", placeholder="e.g. asthma, diabetes...")
    medications = st.text_area("Current medications", placeholder="e.g. metformin 500mg...")
    notes = st.text_area("Additional notes", placeholder="Anything else worth mentioning...")

    submitted = st.form_submit_button("Get AI Guidance")

# ---------------------------------------------------------------------------
# SUBMIT HANDLING
# ---------------------------------------------------------------------------
if submitted:
    symptom_text = format_symptom_list(symptoms, extra_symptoms)

    # Testing scenario 5: empty symptoms -> warn, don't call the API.
    if not symptoms and not extra_symptoms.strip():
        st.warning("Please select or describe at least one symptom before requesting guidance.")
        st.stop()

    if not age.strip():
        st.warning("Please enter the patient's age.")
        st.stop()

    inputs = {
        "age": age,
        "gender": gender,
        "symptoms": symptom_text,
        "duration": duration,
        "severity": severity,
        "conditions": conditions or "None reported",
        "medications": medications or "None reported",
        "notes": notes or "None",
        "language": answer_language,
    }

    st.markdown("---")
    st.subheader("AI Guidance")

    # Build a non-streaming LLM + LLMChain for the structured JSON call.
    llm_json = build_llm(st.session_state.api_key, model_name, temperature=0.2, streaming=False)
    chain = build_assessment_chain(llm_json)

    start = time.time()
    with st.spinner("Analysing symptoms..."):
        try:
            raw_result = chain.invoke(inputs)
            raw_text = raw_result["text"] if isinstance(raw_result, dict) else str(raw_result)
        except Exception as exc:
            st.error(f"The AI request failed: {exc}")
            st.stop()
    elapsed = time.time() - start
    st.caption(f"Response time: {elapsed:.2f}s "
               f"(submit the same form again to see caching speed it up)")

    data = safe_json_parse(raw_text)

    if not validate_schema(data):
        st.error("The model did not return valid structured data. Showing raw output for debugging.")
        with st.expander("Raw model output"):
            st.code(raw_text)
        st.stop()

    # --- Streamed narrative ------------------------------------------------
    st.markdown("#### Plain-language summary")
    llm_stream = build_llm(st.session_state.api_key, model_name, temperature=0.4, streaming=True)
    st.write_stream(stream_narrative(llm_stream, inputs))

    # --- Dashboard -----------------------------------------------------------
    st.markdown("#### Results dashboard")

    urgency = str(data.get("urgency_level", "LOW")).upper().strip()
    color = config.URGENCY_COLORS.get(urgency, "#616161")

    m1, m2, m3 = st.columns(3)
    m1.metric("Urgency level", urgency)
    m2.metric("Reported severity", f"{severity}/10")
    m3.metric("Symptoms reported", len(symptoms) + (1 if extra_symptoms.strip() else 0))

    st.markdown(
        f'<span class="urgency-badge" style="background:{color};">URGENCY: {urgency}</span>',
        unsafe_allow_html=True,
    )

    if urgency == "EMERGENCY":
        st.error("🚨 This may be a medical emergency. Seek immediate emergency care or call your "
                  "local emergency number now.")
    elif urgency == "HIGH":
        st.warning("⚠️ This warrants prompt attention. Please contact a healthcare professional soon.")
    elif urgency == "MEDIUM":
        st.info("ℹ️ Consider seeing a healthcare professional if symptoms persist or worsen.")
    else:
        st.success("✅ Symptoms appear mild. Monitor at home and seek care if things change.")

    tab_summary, tab_conditions, tab_steps, tab_questions, tab_warnings = st.tabs(
        ["Summary", "Possible Conditions", "Next Steps", "Doctor Questions", "Warning Signs"]
    )

    with tab_summary:
        st.write(data.get("summary", "No summary provided."))

    with tab_conditions:
        conditions_list = data.get("possible_conditions", [])
        if conditions_list:
            for c in conditions_list:
                st.markdown(f"**{c.get('name', 'Unnamed')}** -- {c.get('reason', '')}")
        else:
            st.write("No specific conditions listed.")
        st.caption("These are educational possibilities only, not a diagnosis.")

    with tab_steps:
        for step in data.get("recommended_next_steps", []):
            st.markdown(f"- {step}")

    with tab_questions:
        for q in data.get("questions_for_doctor", []):
            st.markdown(f"- {q}")

    with tab_warnings:
        for w in data.get("warning_signs", []):
            st.markdown(f"- ⚠️ {w}")

    with st.expander("Raw JSON (debug)"):
        st.json(data)

    st.markdown("---")
    st.caption(config.DISCLAIMER)
