import os
import requests
from gtts import gTTS
import string
import random
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# Load environment variables
load_dotenv()
hugging_face_api_key = os.getenv("HUGGINGFACE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# --- Gemini Text Answer ---
def get_answer_gemini(question: str) -> str:
    messages = [
        {"role": "system", "content": "I want you to act like a helpful agriculture chatbot and help farmers with their query"},
        {"role": "user", "content": "Give a Brief of Agriculture Seasons in India"},
        {"role": "system", "content": """In India, the agricultural season consists of three major seasons: 
        the Kharif (monsoon), the Rabi (winter), and the Zaid (summer) seasons. Each season has its own crops and practices."""},
        {"role": "user", "content": question}
    ]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# --- Text-to-Speech ---
def text_to_audio(text: str, filename: str):
    tts = gTTS(text)
    path = f"audio/{filename}.mp3"
    os.makedirs("audio", exist_ok=True)
    tts.save(path)
    return path

# --- Audio-to-Text (using HuggingFace API) ---
def process_audio(filepath: str) -> str:
    API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
    headers = {"Authorization": f"Bearer {hugging_face_api_key}"}
    with open(filepath, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    result = response.json()
    return result.get("text", "")

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="🌾 FarmBot", page_icon="🤖", layout="centered")
    st.title("🌾 The FarmBot")
    st.markdown("Ask your farming and agriculture queries. Get **text + audio** responses!")

    # Text input
    user_input = st.text_input("Type your question here:", "")

    if user_input:
        with st.spinner("Thinking... 🤔"):
            answer = get_answer_gemini(user_input)

            # Save response as audio
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            audio_path = text_to_audio(answer, res)

        st.subheader("💡 Your Answer:")
        st.write(answer)

        st.audio(audio_path, format="audio/mp3")

    # Audio upload
    st.markdown("---")
    st.subheader("🎤 Ask with Voice")
    audio_file = st.file_uploader("Upload your question (WAV/MP3)", type=["wav", "mp3"])

    if audio_file is not None:
        with st.spinner("Transcribing audio... 🎧"):
            temp_path = f"temp_{audio_file.name}"
            with open(temp_path, "wb") as f:
                f.write(audio_file.read())

            transcript = process_audio(temp_path)
            st.write(f"🗣 You said: **{transcript}**")

            if transcript:
                answer = get_answer_gemini(transcript)

                res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                audio_path = text_to_audio(answer, res)

                st.subheader("💡 Your Answer:")
                st.write(answer)
                st.audio(audio_path, format="audio/mp3")

if __name__ == "__main__":
    main()
