import os
import cv2
import numpy as np
import streamlit as st
import google.generativeai as genai
import base64
from datetime import datetime

# Page configuration (read from config.toml for Streamlit Cloud)
st.set_page_config(
    page_title="Advanced Dental X-Ray Analysis System",
    page_icon="🦷",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0083B8;
        color: white;
    }
    .disclaimer {
        background-color: #FFF3CD;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main title with professional styling
st.title("🦷 Advanced Dental X-Ray Analysis System")
st.markdown("---")

# Initialize session state
if 'patient_history' not in st.session_state:
    st.session_state['patient_history'] = {}

if 'api_key_provided' not in st.session_state:
    st.session_state['api_key_provided'] = False

# Configure API Key (Streamlit secrets or environment variables)
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not api_key:
    st.error("⚠️ Please set the API key in Streamlit Secrets or environment variables.")
    st.stop()

# Configure Gemini API
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")
except Exception as e:
    st.error(f"Error configuring Gemini API: {str(e)}")
    st.stop()

# Sidebar: Patient Information
with st.sidebar:
    st.header("Patient Information")

    # Patient history form
    with st.form("patient_history"):
        st.session_state.patient_history.update({
            'name': st.text_input("Patient Name", "John Doe"),
            'age': st.number_input("Age", 30, 120),
            'gender': st.selectbox("Gender", ["Male", "Female", "Other"]),
            'medical_history': st.multiselect("Medical History",
                                              ["Diabetes", "Hypertension", "Heart Disease", "None"], ["None"]),
            'dental_complaints': st.text_area("Current Dental Complaints", "Tooth pain"),
            'previous_treatments': st.text_area("Previous Dental Treatments", "None"),
            'smoking': st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"]),
            'last_dental_visit': st.date_input("Last Dental Visit", datetime.now())
        })
        submit_button = st.form_submit_button("Save Patient Information")

# Main Content Area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("X-Ray Upload & Processing")
    uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Read file bytes
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if img is None:
                st.error("Error: Unable to read the image. Please try another file.")
                st.stop()

            # Display original image
            st.image(img, caption="Original X-ray", use_column_width=True)

            # Image processing options
            st.subheader("Image Enhancement Options")
            denoise_strength = st.slider("Denoising Strength", 1, 20, 10)
            contrast_limit = st.slider("Contrast Enhancement", 1.0, 5.0, 2.0)

            try:
                # Process image
                img_processed = cv2.fastNlMeansDenoising(img, None, denoise_strength, 7, 21)
                clahe = cv2.createCLAHE(clipLimit=contrast_limit, tileGridSize=(8, 8))
                img_processed = clahe.apply(img_processed)

                # Display enhanced image
                st.image(img_processed, caption="Enhanced X-ray", use_column_width=True)

                # Encode image
                _, img_encoded = cv2.imencode('.png', img_processed)
                base64_image = base64.b64encode(img_encoded).decode('utf-8')

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.stop()

        except Exception as e:
            st.error(f"Error reading image: {str(e)}")
            st.stop()

with col2:
    if uploaded_file and 'name' in st.session_state.patient_history and st.session_state.patient_history['name']:
        st.header("Analysis & Results")

        analysis_type = st.multiselect("Select Analysis Focus Areas",
                                       ["Cavity Detection", "Bone Density", "Root Canal Assessment",
                                        "Periodontal Status", "Wisdom Teeth", "Overall Assessment"])

        if st.button("Generate Analysis") and model:
            with st.spinner("Analyzing X-ray..."):
                try:
                    prompt = f"""
                    Please analyze this dental X-ray image with the following context:

                    Patient Information:
                    - Name: {st.session_state.patient_history['name']}
                    - Age: {st.session_state.patient_history['age']}
                    - Gender: {st.session_state.patient_history['gender']}
                    - Medical History: {', '.join(st.session_state.patient_history['medical_history'])}
                    - Current Complaints: {st.session_state.patient_history['dental_complaints']}
                    - Previous Treatments: {st.session_state.patient_history['previous_treatments']}
                    - Smoking Status: {st.session_state.patient_history['smoking']}

                    Focus Areas: {', '.join(analysis_type)}

                    Provide:
                    1. Identified abnormalities
                    2. Potential diagnosis
                    3. Recommendations
                    """

                    response = model.generate_content(prompt)
                    st.markdown("### Analysis Results")
                    st.write(response.text)

                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")

                st.markdown("### Disclaimer")
                st.warning("This analysis is for informational purposes only. Consult a professional.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2024 Advanced Dental X-Ray Analysis System</p>", unsafe_allow_html=True)
