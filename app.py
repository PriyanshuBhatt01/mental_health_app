import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# --- CONFIGURATION ---
HF_REPO = "priyanshubahtt001/mental_health_app" 

st.set_page_config(page_title="Mental Health AI", page_icon="🧠")

@st.cache_resource
def load_model():
    # We use the standard AutoModel (PyTorch) which is way more stable
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    config = AutoConfig.from_pretrained(HF_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO,
        config=config,
        use_safetensors=True # This loads your model.safetensors perfectly
    )
    return tokenizer, model

st.title("🧠 AI Mental Health Analyzer")
st.write("Analyze the emotional state of text using a fine-tuned DistilBERT model.")

try:
    with st.spinner("Loading AI Brain..."):
        tokenizer, model = load_model()
        model.eval() # Set to evaluation mode
    
    text_input = st.text_area("How are you feeling?", placeholder="Type here...", height=150)
    
    if st.button("Analyze"):
        if not text_input.strip():
            st.warning("Please enter text.")
        else:
            # 1. Preprocess (PyTorch style)
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
            
            # 2. Predict (No gradient calculation needed for inference)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
            
            # 3. Mapping
            class_names = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
            predicted_index = np.argmax(probabilities)
            
            # 4. Results
            st.markdown(f"### Prediction: **{class_names[predicted_index]}**")
            st.progress(float(probabilities[predicted_index]))
            st.write(f"Confidence: **{probabilities[predicted_index]*100:.2f}%**")
            
            st.divider()
            chart_data = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            st.bar_chart(chart_data)

except Exception as e:
    st.error(f"System Error: {e}")
