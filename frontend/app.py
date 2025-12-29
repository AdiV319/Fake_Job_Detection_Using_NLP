import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Fake Job Post Detection System")
st.write("Paste a job description below to check whether it is **Real or Fake**.")

# ---------------- SESSION STATE ----------------
if "job_text" not in st.session_state:
    st.session_state.job_text = ""

# ---------------- TEXT AREA ----------------
st.text_area(
    "Enter Job Description",
    height=220,
    key="job_text",
    placeholder="Paste the job description here..."
)

# ---------------- CALLBACKS ----------------
def clear_text():
    st.session_state.job_text = ""

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Predict"):
        if st.session_state.job_text.strip() == "":
            st.warning("Please enter a job description.")
        else:
            with st.spinner("Analyzing job post..."):
                response = requests.post(
                    API_URL,
                    json={"job_description": st.session_state.job_text}
                )

            if response.status_code == 200:
                result = response.json()

                if result["prediction"] == "Fake":
                    st.error(f"🚨 Fake Job Detected ({result['confidence']})")
                else:
                    st.success(f"✅ Real Job Post ({result['confidence']})")

                st.info(f"⏱ Processing Time: {result['processing_time_ms']} ms")
            else:
                st.error("Prediction failed. Please try again.")

with col2:
    st.button("🧹 Clear", on_click=clear_text)
