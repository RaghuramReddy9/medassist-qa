import streamlit as st
import requests

st.set_page_config(page_title="MedAssist AI", layout="centered")

st.title("🩺 MedAssist - AI Medical Assistant")
st.subheader("Ask any medical question")

# user prompt input
prompt = st.text_input("Enter your question:")

if st.button("Generate Response") and prompt:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/generate",
            json={"prompt": prompt}
        )
        if response.status_code == 200:
            result = response.json()["response"]
            st.success(result)
        else:
            st.error(f"Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection failed: {e}")