import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- CONFIGURATION ---
HF_REPO = "priyanshubahtt001/mental_health_app" 

st.set_page_config(page_title="Mental Health AI", page_icon="🧠")

@st.cache_resource
def load_model():
    # This setup is the most stable for Streamlit Cloud
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    # from_tf=True translates your Colab model so the Cloud can read it
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO, from_tf=True)
    return tokenizer, model

st.title("🧠 AI Mental Health Analyzer")
st.write("Enter text below to see the AI's detection of emotional state.")

try:
    with st.spinner("Loading AI Brain... (First run takes 2-3 minutes)"):
        tokenizer, model = load_model()
    
    text_input = st.text_area("How are you feeling?", height=150, placeholder="e.g., I feel very overwhelmed and tired lately...")
    
    if st.button("Analyze Status"):
        if not text_input.strip():
            st.warning("Please enter some text first!")
        else:
            # 1. Prepare text
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # 2. Run AI Prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
            
            # 3. Define the Classes (Matches your Combined Data order)
            class_names = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
            
            idx = np.argmax(probs)
            label = class_names[idx]
            conf = probs[idx] * 100
            
            # 4. Display results beautifully
            st.success(f"Detected State: **{label}**")
            st.write(f"Confidence: {conf:.2f}%")
            st.progress(int(conf))
            
            st.write("---")
            st.write("Detailed Probabilities:")
            st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(class_names))})

except Exception as e:
    st.error(f"System Error: {e}")
    st.info("Try refreshing the page or checking the 'Manage app' logs.")
