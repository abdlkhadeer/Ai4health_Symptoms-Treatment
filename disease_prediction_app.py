import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from rapidfuzz import process

# Theming: Add a custom theme in a config.toml file
st.set_page_config(page_title="Disease Prediction Chatbot", page_icon="🩺", layout="wide")

# Caching functions for performance
@st.cache_resource
def load_model():
    return joblib.load("disease_prediction_model.pkl")

@st.cache_data
def load_metadata():
    with open("disease_metadata.json", "r") as file:
        return json.load(file)

@st.cache_data
def load_symptoms_dict():
    return joblib.load("symptoms_dict.pkl")

# Load resources
model = load_model()
mlb = joblib.load("mlb.pkl")
symptoms_dict = load_symptoms_dict()
disease_metadata = load_metadata()

# List of all symptoms in the trained dataset (columns of encoded symptoms)
symptom_columns = mlb.classes_

# Function to preprocess symptoms
def preprocess_input_symptoms(input_symptoms):
    standardized_symptoms = [symptom.strip().replace(" ", "_") for symptom in input_symptoms]
    input_vector = [1 if symptom in standardized_symptoms else 0 for symptom in symptom_columns]
    return np.array(input_vector).reshape(1, -1)

# Function to clean user input (conversational phrases)
def clean_user_input(input_text):
    input_text = input_text.lower()
    conversational_mapping = {
        "i am feeling": "",
        "like": "",
        "i have": "",
        "i am experiencing": "",
        "i am having": "",
        "it's": "",
        "i feel": "",
    }
    for word, replacement in conversational_mapping.items():
        input_text = input_text.replace(word, replacement)
    return input_text.strip()

# Streamlit Layout
st.title("🩺 Disease Prediction Chatbot")
st.markdown("""
### 🤖 Welcome to Your Health Assistant!
Tell me your symptoms, and I’ll predict the disease while providing you with helpful information such as treatments, causes, and prevention tips.
""")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "About", "Contact", "Developer", "Blog"])

# Home Page
if menu == "Home":
    st.header("🚨 Enter Your Symptoms")
    st.markdown("Chat with me to find out possible health issues.")

    # Chatbot UI
    st.markdown("### 🤖 Chatbot")
    bot_response = "Hello! Please describe your symptoms below (e.g., 'I have a headache and fever')."
    st.info(bot_response)

    # User Input
    user_input = st.text_input("👤 You:", placeholder="Type your symptoms here...")

    if st.button("🔎 Predict Disease"):
        if user_input:
            # Clean and split user input
            cleaned_input = clean_user_input(user_input)
            input_symptoms = [symptom.strip() for symptom in cleaned_input.split(",")]

            # Preprocess symptoms into binary vector
            input_vector = preprocess_input_symptoms(input_symptoms)

            try:
                # Make prediction
                prediction = model.predict(input_vector)
                predicted_disease = prediction[0]

                # Display prediction result
                st.markdown(f"""
                ## 🩺 Predicted Disease:  
                <div style="color:#2F8C2F; font-size:28px; font-weight:bold;">{predicted_disease}</div>
                """, unsafe_allow_html=True)

                # Fetch and Display disease metadata
                if predicted_disease in disease_metadata:
                    disease_info = disease_metadata[predicted_disease]

                    st.markdown("---")  # Divider
                    st.markdown("### 📋 Disease Information")
                    st.markdown(f"""
                    <div style="background-color:#f8f9fa; padding:15px; border-radius:8px;">
                        <p style="font-size:16px;"><strong>Description:</strong> {disease_info.get('Description', 'N/A')}</p>
                        <p style="font-size:16px;"><strong>Symptoms:</strong> {disease_info.get('Symptoms', 'N/A')}</p>
                        <p style="font-size:16px;"><strong>Causes:</strong> {disease_info.get('Causes', 'N/A')}</p>
                        <p style="font-size:16px;"><strong>Prevention:</strong> {disease_info.get('Prevention', 'N/A')}</p>
                        <p style="font-size:16px;"><strong>Treatment:</strong> {disease_info.get('Treatment', 'N/A')}</p>
                        <p style="font-size:16px;"><strong>Illness Type:</strong> {disease_info.get('Illness_Type', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning("⚠️ No additional information available for this disease.")

                # Follow-Up Options
                st.markdown("### 🔄 What would you like to do next?")
                follow_up = st.radio("", ["🔍 Ask more about this disease", "🆕 Enter new symptoms", "🚪 Exit"])

                if follow_up == "🔍 Ask more about this disease":
                    st.write(f"🤖 **Bot:** What would you like to know more about **{predicted_disease}**?")
                elif follow_up == "🆕 Enter new symptoms":
                    st.experimental_rerun()
                else:
                    st.write("🤖 **Bot:** Take care! Feel free to return anytime. 👋")

            except Exception as e:
                st.error(f"❌ An error occurred while predicting: {e}")
        else:
            st.warning("⚠️ Please enter your symptoms to proceed.")

# About Page
elif menu == "About":
    st.header("ℹ️ About This App")
    st.markdown("""
    This AI-based chatbot assists users in predicting potential diseases based on their symptoms.  
    It also provides additional insights like causes, treatments, and preventive measures.
    
    **Technologies Used:**  
    - Streamlit for the interactive user interface.  
    - Machine Learning model for disease prediction.  
    - Preprocessed health data for metadata insights.
    """)

# Contact Page
elif menu == "Contact":
    st.header("📞 Contact Us")
    st.markdown("""
    - **Email:** support@vitalenex.com  
    - **Phone:** +123456789  
    - **Website:** [Visit Us](https://www.vitalenex.com)
    """)
# Developer Page
elif menu == "Developer":
    st.header("👩‍💻 Developer")
    st.markdown("""
        This app was built using Streamlit and a trained machine learning model for healthcare predictions. 
        It uses a combination of preprocessing techniques and metadata to provide accurate results.
    """)

# Blog Page
elif menu == "Blog":
    st.header("📝 Blog")
    st.markdown("""
        Check out the latest health and tech articles. 
        Stay informed on innovative healthcare solutions and AI advancements.
    """)

