import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- CONFIGURATION ---
HF_REPO = "priyanshubahtt001/mental_health_app" 
# ---------------------

st.set_page_config(page_title="Mental Health AI", page_icon="🧠")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    # Adding from_tf=True tells the code to read your TensorFlow files
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO, from_tf=True)
    return tokenizer, model

st.title("🧠 AI Mental Health Analyzer")
st.write("Type a sentence below to analyze the emotional state.")

try:
    with st.spinner("Initializing AI... (The first time takes about 1 minute)"):
        tokenizer, model = load_model()
    
    text_input = st.text_area("How are you feeling?", height=150, placeholder="Type your thoughts here...")
    
    if st.button("Analyze Text"):
        if not text_input.strip():
            st.warning("Please type something first!")
        else:
            # 1. Preprocess
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            # 2. Predict (Disable gradients to save memory)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
            
            # 3. Get Result
            class_names = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
            
            predicted_index = np.argmax(probabilities)
            predicted_class = class_names[predicted_index]
            confidence = probabilities[predicted_index] * 100
            
            # 4. Display Result
            st.success(f"Detection Result: **{predicted_class}**")
            st.progress(int(confidence))
            st.write(f"Confidence Score: {confidence:.2f}%")
            
            st.write("---")
            st.write("Detailed Breakdown:")
            # Create a dictionary for the chart
            chart_data = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            st.bar_chart(chart_data)

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Try refreshing the page or checking the 'Manage app' logs.")

