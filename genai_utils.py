import requests
import json
import time
API_KEY = None
BASE_URL ="https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent"

def configure_gemini(api_key: str):
    global API_KEY
    if api_key and api_key.strip():
        API_KEY = api_key.strip()
    else:
        API_KEY = None


# ==========================================
# Internal helper function (Endpoint call)
# ==========================================
def _call_gemini(prompt: str) -> str:
    if API_KEY is None:
        return None

    url = f"{BASE_URL}?key={API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 300
        }
    }

    try:
        start_time = time.time()

        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=10
        )

        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        if response.status_code != 200:
            print("Gemini API Error:", response.text)
            return None

        result = response.json()

        # Save response time in session state
        import streamlit as st
        st.session_state.api_response_time = response_time

        return result["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        print("Gemini Exception:", e)
        return None


# ==========================================
# Follow-up Questions
# ==========================================
def gemini_followup(symptoms: str, disease: str) -> str:
    prompt = f"""
You are a medical assistant.

Patient symptoms:
{symptoms}

Likely disease:
{disease}

Ask 2–3 short relevant follow-up questions.
Keep it concise and clinical.
"""

    response = _call_gemini(prompt)

    if response:
        return response.strip()

    # Safe fallback
    return (
        "1. When did your symptoms start?\n"
        "2. Are they worsening or improving?\n"
        "3. Do you have fever, severe pain, or breathing difficulty?"
    )


# ==========================================
# Medical Response
# ==========================================
def gemini_medical_response(context: str, disease: str, instruction: str) -> str:
    prompt = f"""
You are a medical assistant.

Context:
{context}

Primary suspected disease:
{disease}

Task:
{instruction}

Respond in clear, patient-friendly language.
"""

    response = _call_gemini(prompt)

    if response:
        return response.strip()

    return (
        f"The condition ({disease}) may require medical evaluation. "
        "If symptoms worsen or persist, consult a healthcare professional."
    )
