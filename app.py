import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np

# --- CONFIGURATION ---
HF_REPO = "priyanshubahtt001/mental_health_app" 
# ---------------------

st.set_page_config(page_title="Mental Health AI", page_icon="🧠")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    model = TFAutoModelForSequenceClassification.from_pretrained(HF_REPO)
    return tokenizer, model

st.title("🧠 AI Mental Health Analyzer")
st.write("Type a sentence below to analyze the emotional state.")

try:
    with st.spinner("Downloading your model... (First run takes about 1 minute)"):
        tokenizer, model = load_model()
    
    text_input = st.text_area("How are you feeling?", height=150)
    
    if st.button("Analyze Text"):
        if not text_input.strip():
            st.warning("Please type something first!")
        else:
            # 1. Preprocess
            inputs = tokenizer(text_input, return_tensors="tf", truncation=True, padding=True, max_length=128)
            
            # 2. Predict
            outputs = model(inputs)
            probabilities = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
            
            # 3. Get Result (Ensure this matches your training data order!)
            class_names = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
            
            predicted_index = np.argmax(probabilities)
            predicted_class = class_names[predicted_index]
            confidence = probabilities[predicted_index] * 100
            
            # 4. Display
            st.success(f"Prediction: **{predicted_class}**")
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")
            
            st.write("---")
            st.write("Detailed Probabilities:")
            st.bar_chart({class_names[i]: probabilities[i] for i in range(len(class_names))})

except Exception as e:
    st.error(f"Error loading model: {e}")