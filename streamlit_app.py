import os
import cv2
import numpy as np
import streamlit as st
from google.generativeai import configure, GenerativeModel
import base64
from datetime import datetime

# Page configuration
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
    st.session_state.patient_history = {}

# Sidebar for API configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Your API Key", type="password")
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("⚠️ Please provide an API key.")
            st.stop()

    st.markdown("---")
    st.header("Patient Information")

    # Patient history form
    with st.form("patient_history"):
        st.session_state.patient_history.update({
            'name': st.text_input("Patient Name"),
            'age': st.number_input("Age", 1, 120),
            'gender': st.selectbox("Gender", ["Male", "Female", "Other"]),
            'medical_history': st.multiselect("Medical History",
                                              ["Diabetes", "Hypertension", "Heart Disease", "None"]),
            'dental_complaints': st.text_area("Current Dental Complaints"),
            'previous_treatments': st.text_area("Previous Dental Treatments"),
            'smoking': st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"]),
            'last_dental_visit': st.date_input("Last Dental Visit")
        })
        submit_button = st.form_submit_button("Save Patient Information")

# Configure Gemini Pro
configure(api_key=api_key)
model = GenerativeModel("gemini-pro")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("X-Ray Upload & Processing")
    uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display original image
        st.image(uploaded_file, caption="Original X-ray", use_column_width=True)

        # Image processing options
        st.subheader("Image Enhancement Options")
        denoise_strength = st.slider("Denoising Strength", 1, 20, 10)
        contrast_limit = st.slider("Contrast Enhancement", 1.0, 5.0, 2.0)

        # Process image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # Enhanced preprocessing
        img = cv2.fastNlMeansDenoising(img, None, denoise_strength, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=contrast_limit, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Display enhanced image
        st.image(img, caption="Enhanced X-ray", use_column_width=True)

        # Encode image
        _, img_encoded = cv2.imencode('.png', img)
        base64_image = base64.b64encode(img_encoded).decode('utf-8')

with col2:
    if uploaded_file and 'name' in st.session_state.patient_history and st.session_state.patient_history['name']:
        st.header("Analysis & Results")

        # Analysis options
        analysis_type = st.multiselect("Select Analysis Focus Areas",
                                       ["Cavity Detection", "Bone Density", "Root Canal Assessment",
                                        "Periodontal Status", "Wisdom Teeth", "Overall Assessment"])

        if st.button("Generate Analysis"):
            with st.spinner("Analyzing X-ray..."):
                # Construct detailed prompt
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

                Please provide a detailed analysis including:
                1. Identified abnormalities or concerns
                2. Potential diagnosis considerations
                3. Recommended additional examinations if needed
                4. Treatment suggestions
                5. Risk factors based on patient history

                Format the response in a clear, structured manner.
                """

                response = model.generate_content(prompt)

                # Display results
                st.markdown("### Analysis Results")
                st.write(response.text)

                # Generate report
                st.markdown("### Report Generation")
                if st.button("Generate PDF Report"):
                    st.info("Report generation functionality can be implemented here")

                # Display disclaimer
                st.markdown("""
                <div class="disclaimer">
                    <h4>⚠️ Medical Disclaimer</h4>
                    <p>This analysis is generated by AI and is for informational purposes only. 
                    It should not be considered as a definitive diagnosis. Please consult with a 
                    qualified dental professional for accurate diagnosis and treatment planning.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Please upload an X-ray image and complete patient information to proceed with analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Advanced Dental X-Ray Analysis System v2.0</p>
        <p>Developed by:</p>
        <p>Supervised and reviewed by:</p>
        <p>Participants:</p>
        <p>UI Designer:</p>
    </div>
    """,
    unsafe_allow_html=True
)
