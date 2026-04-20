import os

# --- 1. FORCE SYSTEM ARCHITECTURE ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_KERAS"] = "1"

import streamlit as st
import numpy as np

# --- 2. THE BULLETPROOF IMPORT BLOCK ---
try:
    import tensorflow as tf
    import tf_keras
    import torch
    # We import these specifically so 'transformers' finds them
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoConfig
    backend_status = "✅ System initialized with TensorFlow backend."
except ImportError as e:
    # Fallback: Sometimes 'transformers' needs a second to see tf-keras
    try:
        from transformers import AutoTokenizer, AutoConfig
        from transformers import TFAutoModelForSequenceClassification
        backend_status = "✅ System initialized (Fallback mode)."
    except:
        backend_status = f"❌ Backend Error: {e}"

# --- CONFIGURATION ---
HF_REPO = "priyanshubahtt001/mental_health_app" 

st.set_page_config(page_title="Mental Health AI", page_icon="🧠")

@st.cache_resource
def load_model():
    # Fix: AutoConfig is now properly imported above
    config = AutoConfig.from_pretrained(HF_REPO)
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    
    # Load with cross-framework flags
    model = TFAutoModelForSequenceClassification.from_pretrained(
        HF_REPO,
        config=config,
        from_pt=True,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

st.title("🧠 AI Mental Health Analyzer")
st.info(backend_status)

try:
    with st.spinner("Loading DistilBERT weights... (This takes 30-60 seconds)"):
        tokenizer, model = load_model()
    
    text_input = st.text_area("How are you feeling?", placeholder="Type your thoughts here...", height=150)
    
    if st.button("Analyze"):
        if not text_input.strip():
            st.warning("Please enter text first.")
        else:
            # Preprocess
            inputs = tokenizer(text_input, return_tensors="tf", truncation=True, padding=True, max_length=256)
            
            # Predict
            outputs = model(inputs)
            probabilities = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
            
            # Labels
            class_names = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
            predicted_index = np.argmax(probabilities)
            
            # Output
            st.markdown(f"### Result: **{class_names[predicted_index]}**")
            st.progress(float(probabilities[predicted_index]))
            
            st.divider()
            chart_data = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            st.bar_chart(chart_data)

except Exception as e:
    st.error(f"Prediction Error: {e}")
