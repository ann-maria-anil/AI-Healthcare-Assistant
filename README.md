# AI-Powered Healthcare Diagnosis Assistant  
## MMMSMETA – Team Charlie  

---

### Project Overview  

The AI-Powered Healthcare Diagnosis Assistant is a web-based diagnostic support system that predicts diseases based on user-entered symptoms and enhances the output using Generative AI.

The system integrates Natural Language Processing (NLP), TF-IDF similarity ranking, and Google Gemini API to provide intelligent medical responses.

---

### System Architecture  

- Input Module (Voice / Text)  
- NLP Processing Layer  
- Machine Learning Layer  
- Generative AI Integration  
- Web Deployment Layer  

---

### Input Module  

Users can provide symptoms through:

- 🎤 Voice input (Speech-to-Text)  
- ⌨️ Manual text input  

The interface is built using Streamlit.

---

### NLP Processing  

The symptom text undergoes:

- Text cleaning  
- Tokenization  
- TF-IDF vectorization  

TF-IDF converts textual symptoms into numerical vectors for similarity comparison.

---

### Machine Learning Approach  

This project uses an information retrieval method:

1. Symptoms are converted into TF-IDF vectors  
2. Cosine similarity is computed  
3. Diseases are ranked by similarity score  
4. The top-ranked disease is selected  

This ensures fast and lightweight deployment.

---

### Generative AI Integration  

After prediction, Google Gemini API is used to:

- Generate clinical explanations  
- Suggest precautions  
- Ask follow-up questions  
- Provide patient-friendly summaries  

The API key is securely managed using Streamlit Secrets.

---

### Technical Stack  

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- SpeechRecognition  
- Google Gemini API
- NLTK

---

### Project Structure  

```
AI-Healthcare-Assistant/
│
├── disease_app.py
├── model_utils.py
├── genai_utils.py
├── correct_Symptom2Disease_Final_Dataset.xlsx
├── requirements.txt
└── README.md
```

---

### Live Deployment  
https://ai-healthcare-assistant-annmaria.streamlit.app/

---

### Disclaimer  

This project is for academic purposes i.e part of internship project only and not a substitute for professional medical advice.
