import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# ----------------- CONFIG -----------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Gemini setup
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ----------------- GEMINI STREAM -----------------
def get_answer_gemini_stream(question: str) -> str:
    """Stream response from Gemini model."""
    messages = [
        {"role": "system", "content": "You are a helpful agriculture chatbot for farmers in India."},
        {"role": "user", "content": question}
    ]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    response = gemini_model.generate_content(prompt, stream=True)

    answer = ""
    answer_box = st.empty()
    for chunk in response:
        if chunk.text:
            answer += chunk.text
            answer_box.markdown(answer)  # progressively update
    return answer.strip()

# ----------------- STREAMLIT APP -----------------
def main():
    st.set_page_config(page_title="🌾 FarmBot", page_icon="🤖", layout="centered")
    st.title("🌾 The FarmBot")
    st.markdown("Ask your **farming & agriculture queries** and get instant AI-powered answers!")

    # --- User Input ---
    user_input = st.text_input("💬 Type your question here:")

    if user_input:
        st.subheader("💡 Answer:")
        with st.spinner("Thinking... 🤔"):
            get_answer_gemini_stream(user_input)

# ----------------- RUN APP -----------------
if __name__ == "__main__":
    main()
