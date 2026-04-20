import os

# --- 1. DEFENSIVE INITIALIZATION (MUST BE AT THE TOP) ---
# This forces the backend detection for Python 3.13 compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_KERAS"] = "1"

import streamlit as st
import numpy as np

# Prevent the 'torchvision' scan crash and force backend discovery
try:
    import tensorflow as tf
    import tf_keras
    import torch
    # Import torchvision if available to satisfy the transformers scanner
    try: import torchvision 
    except ImportError: pass
    
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoConfig
except ImportError as e:
    st.error(f"Backend Initialization Error: {e}. Please ensure 'tf-keras' is installed.")

# --- CONFIGURATION ---
HF_REPO = "priyanshubahtt001/mental_health_app" 

st.set_page_config(page_title="Mental Health AI", page_icon="🧠")

@st.cache_resource
def load_model():
    # Force loading via Config to handshake with Safetensors
    config = AutoConfig.from_pretrained(HF_REPO)
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    
    # Load model with specific flags for stability
    model = TFAutoModelForSequenceClassification.from_pretrained(
        HF_REPO,
        config=config,
        from_pt=True,          # Bridge from PyTorch training
        use_safetensors=True,  # Modern security/speed format
        low_cpu_mem_usage=True # Crucial for Python 3.13/Cloud limits
    )
    return tokenizer, model

st.title("🧠 AI Mental Health Analyzer")
st.write("Analyze the emotional state of text using a fine-tuned DistilBERT model.")

try:
    with st.spinner("Initializing AI Model... This may take a minute on first run."):
        tokenizer, model = load_model()
    
    text_input = st.text_area("How are you feeling today?", placeholder="e.g., I've been feeling quite overwhelmed lately...", height=150)
    
    if st.button("Run Analysis"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # 1. Preprocess
            inputs = tokenizer(text_input, return_tensors="tf", truncation=True, padding=True, max_length=256)
            
            # 2. Predict
            outputs = model(inputs)
            # Use logits to calculate probabilities
            probabilities = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
            
            # 3. Mapping (Matches LabelEncoder order from training)
            class_names = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
            predicted_index = np.argmax(probabilities)
            predicted_class = class_names[predicted_index]
            confidence = probabilities[predicted_index] * 100
            
            # 4. Results Display
            st.markdown(f"### Prediction: **{predicted_class}**")
            st.progress(float(probabilities[predicted_index]))
            st.write(f"Confidence Level: **{confidence:.2f}%**")
            
            st.divider()
            st.write("#### Emotional Breakdown:")
            # Create a clean chart for the presentation
            chart_data = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            st.bar_chart(chart_data)

except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.info("Check your requirements.txt for 'tf-keras' and 'torchvision'.")
