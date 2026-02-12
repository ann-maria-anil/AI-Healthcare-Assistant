import streamlit as st
import base64
import tempfile
import speech_recognition as sr
import requests

from st_audiorec import st_audiorec
from model_utils import load_dataset, build_tfidf, predict_disease
from genai_utils import configure_gemini, gemini_followup, gemini_medical_response


# =================================================
# Background Styling
# =================================================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .card {{
            background: white;
            padding: 22px;
            border-radius: 14px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .section-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 10px;
        }}

        .disease-card {{
            border-left: 6px solid #2563eb;
            background: #f8fafc;
            padding: 15px;
            font-weight: 600;
        }}

        div.stButton > button {{
            width: 100%;
            padding: 0.8em;
            font-size: 16px;
            font-weight: 600;
            border-radius: 10px;
            background-color: #f97316;
            color: white;
            border: none;
        }}

        div.stButton > button:hover {{
            background-color: #fb923c;
        }}

        .voice-box {{
            background: #f1f5f9;
            padding: 12px;
            border-left: 5px solid #2563eb;
            border-radius: 8px;
        }}
        </style>
    """, unsafe_allow_html=True)


# =================================================
# Speech to Text (Voice for Symptoms Only)
# =================================================
def speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name

    with sr.AudioFile(temp_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio, language="en-IN").lower()
    except:
        return ""


# =================================================
# Page Config
# =================================================
st.set_page_config("AI Healthcare Assistant", "🩺", layout="centered")
set_background("background.jpg")

st.title("🩺 AI Healthcare Assistant")
st.caption("Disease prediction using text or voice input")


# =================================================
# Session State Initialization
# =================================================
defaults = {
    "predicted": False,
    "interpretation_done": False,
    "voice_symptoms": "",
    "questions": "",
    "top_disease": "",
    "final_symptoms": "",
    "patient_context": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =================================================
# Load Dataset + Model
# =================================================
@st.cache_data
def load_all():
    df = load_dataset("correct_Symptom2Disease_Final_Dataset.xlsx")
    tfidf, tfidf_matrix = build_tfidf(df["cleaned_symptoms"])
    return df, tfidf, tfidf_matrix


df, tfidf, tfidf_matrix = load_all()


# =================================================
# Sidebar – Gemini API
# =================================================
with st.sidebar:
    st.header("🔐 Gemini API Configuration")

    configure_gemini()

    if not st.secrets.get("GEMINI_API_KEY", None):
        st.error("Gemini API key not configured in Secrets.")
    else:
        st.success("Gemini API key loaded securely.")

    st.subheader("🧠 System Architecture")
    st.info("""
🔹 Disease Prediction  
→ TF-IDF + Cosine Similarity (Local ML Model)

🔹 Clinical Explanation & Follow-up  
→ Google Gemini API (REST – v1)

Model Used: gemini-2.5-flash
""")

    


# =================================================
# STEP 1 – Symptom Input (Voice Enabled)
# =================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>1️⃣ Describe your symptoms</div>", unsafe_allow_html=True)

typed_symptoms = st.text_area("Type symptoms", height=120)

st.markdown("🎤 Or record voice input")
audio_bytes = st_audiorec()

if audio_bytes:
    st.session_state.voice_symptoms = speech_to_text(audio_bytes)

if st.session_state.voice_symptoms:
    st.markdown(
        f"<div class='voice-box'>Recognized: {st.session_state.voice_symptoms}</div>",
        unsafe_allow_html=True
    )

final_symptoms = (typed_symptoms + " " + st.session_state.voice_symptoms).strip()
st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# Predict Disease
# =================================================
if st.button("🔍 Predict Disease"):

    if final_symptoms == "":
        st.warning("Please enter or record symptoms.")
    else:
        result = predict_disease(final_symptoms, df, tfidf, tfidf_matrix)
        diseases = result["Disease"].tolist()

        st.session_state.predicted = True
        st.session_state.top_disease = diseases[0]
        st.session_state.final_symptoms = final_symptoms

        st.markdown("## 🧠 Predicted Diseases")
        for d in diseases:
            st.markdown(
                f"<div class='card disease-card'>{d}</div>",
                unsafe_allow_html=True
            )

        st.session_state.questions = gemini_followup(
            final_symptoms,
            st.session_state.top_disease
        )


# =================================================
# STEP 2 – Follow-up Questions (TEXT ONLY)
# =================================================
if st.session_state.predicted and not st.session_state.interpretation_done:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>2️⃣ Follow-up Questions</div>", unsafe_allow_html=True)

    st.markdown(st.session_state.questions)

    followup_text = st.text_area("Your answer")

    if st.button("Submit Follow-up"):
        if followup_text.strip() == "":
            st.warning("Please provide follow-up response.")
        else:
            st.session_state.patient_context = f"""
Symptoms:
{st.session_state.final_symptoms}

Predicted Disease:
{st.session_state.top_disease}

Patient Response:
{followup_text}
"""
            st.session_state.interpretation_done = True

    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# STEP 3 – Clinical Interpretation
# =================================================
if st.session_state.interpretation_done:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧠 Clinical Interpretation</div>", unsafe_allow_html=True)

    interpretation = gemini_medical_response(
        st.session_state.patient_context,
        st.session_state.top_disease,
        "Briefly interpret the symptoms clinically"
    )

    st.markdown(interpretation)
    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# STEP 4 – Menu Section
# =================================================
if st.session_state.interpretation_done:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>3️⃣ What would you like to know?</div>", unsafe_allow_html=True)

    option = st.selectbox(
        "Choose an option",
        [
            "Know more about the condition",
            "Recommended diagnostic tests",
            "Treatment and home remedies",
            "When to consult a doctor",
            "Summary view",
            "End conversation"
        ]
    )

    prompts = {
        "Know more about the condition": "Explain the disease in simple terms.",
        "Recommended diagnostic tests": "List appropriate diagnostic tests.",
        "Treatment and home remedies": "Explain treatment options and home care.",
        "When to consult a doctor": "Explain warning signs and red flags.",
        "Summary view": "Provide a concise medical summary."
    }

    if option == "End conversation":
        st.success("Conversation ended. Stay healthy.")
        st.stop()

    if st.button("Get Information"):
        response = gemini_medical_response(
            st.session_state.patient_context,
            st.session_state.top_disease,
            prompts[option]
        )
        st.markdown(response)

    st.markdown("</div>", unsafe_allow_html=True)


# =================================================
# Footer
# =================================================
st.caption("⚠️ For educational purposes only. Not a substitute for professional medical advice.")
