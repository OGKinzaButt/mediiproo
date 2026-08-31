"""
config.py
---------
Central place for app settings and the dropdown/multiselect options used
by the Streamlit form. Keeping these here means app.py stays clean and
anyone can tweak the choices without touching the UI code.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (if it exists) into the environment.
load_dotenv()

# The API key can come from .env (server-side deployments) OR be pasted
# by the user into the "Connect your OpenAI key" screen (kept in
# st.session_state only, never written to disk). app.py checks both
# places -- this constant is just the .env fallback.
ENV_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

APP_NAME = "MediGuide AI"
APP_TAGLINE = "AI-powered preliminary symptom guidance -- for education only."

DISCLAIMER = (
    "MediGuide AI is an educational prototype, not a licensed doctor, "
    "diagnostic tool, or emergency service. It never provides a confirmed "
    "diagnosis. Always consult a qualified healthcare professional, and "
    "call your local emergency number immediately if you are in a medical "
    "emergency."
)

AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"]

LANGUAGES = ["English", "Urdu", "Spanish", "French", "Arabic", "Hindi"]

GENDERS = ["Female", "Male", "Non-binary", "Prefer not to say"]

SYMPTOM_OPTIONS = [
    "Fever", "Cough", "Sore throat", "Runny nose", "Headache", "Fatigue",
    "Nausea", "Vomiting", "Diarrhea", "Abdominal pain", "Chest pain",
    "Shortness of breath", "Dizziness", "Rash", "Joint pain", "Muscle ache",
    "Back pain", "Loss of appetite", "Chills", "Sore eyes",
]

DURATION_OPTIONS = [
    "Less than 1 day", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks",
]

CACHE_MODES = [
    "None",
    "In-Memory (fast, per-session)",
    "SQLite (persists across restarts)",
]

URGENCY_COLORS = {
    "LOW": "#2e7d32",
    "MEDIUM": "#e08e00",
    "HIGH": "#d84315",
    "EMERGENCY": "#c62828",
}
