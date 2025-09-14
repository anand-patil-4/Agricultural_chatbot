import os
import random
import string
import asyncio
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts

# ----------------- CONFIG -----------------
load_dotenv()
hugging_face_api_key = os.getenv("HUGGINGFACE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Gemini setup
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ----------------- GEMINI STREAM -----------------
def get_answer_gemini_stream(question: str) -> str:
    """Stream response from Gemini model"""
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
            answer_box.markdown(answer)  # progressively update UI
    return answer.strip()

# ----------------- EDGE TTS -----------------
async def text_to_audio_async(text: str, filename: str):
    """Convert text to speech using Edge TTS"""
    path = f"audio/{filename}.mp3"
    os.makedirs("audio", exist_ok=True)
    communicate = edge_tts.Communicate(text, voice="en-IN-NeerjaNeural")  # Indian English female voice
    await communicate.save(path)
    return path

def text_to_audio(text: str, filename: str):
    return asyncio.run(text_to_audio_async(text, filename))

# ----------------- STREAMLIT APP -----------------
def main():
    st.set_page_config(page_title="🌾 FarmBot", page_icon="🤖", layout="centered")
    st.title("🌾 The FarmBot")
    st.markdown("Ask your **farming & agriculture queries**. Get instant text + audio answers!")

    # --- User Input ---
    user_input = st.text_input("💬 Type your question here:")

    if user_input:
        st.subheader("💡 Answer:")
        with st.spinner("Thinking... 🤔"):
            # Stream Gemini Answer
            answer = get_answer_gemini_stream(user_input)

        # Generate audio AFTER text is shown
        with st.spinner("Generating audio... 🔊"):
            res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            audio_path = text_to_audio(answer, res)
            st.audio(audio_path, format="audio/mp3")

    # --- Upload Audio Question ---
    st.markdown("---")
    st.subheader("🎤 Ask with Voice (Optional)")

    audio_file = st.file_uploader("Upload your question (WAV/MP3)", type=["wav", "mp3"])

    if audio_file:
        temp_path = f"temp_{audio_file.name}"
        with open(temp_path, "wb") as f:
            f.write(audio_file.read())

        # ⚡ Option 1: Whisper (if available)
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(temp_path)
            transcript = result["text"]
        except Exception:
            # ⚡ Option 2: HuggingFace API (fallback)
            import requests
            API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
            headers = {"Authorization": f"Bearer {hugging_face_api_key}"}
            with open(temp_path, "rb") as f:
                data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)
            transcript = response.json().get("text", "")

        if transcript:
            st.write(f"🗣 You said: **{transcript}**")

            st.subheader("💡 Answer:")
            with st.spinner("Thinking... 🤔"):
                answer = get_answer_gemini_stream(transcript)

            with st.spinner("Generating audio... 🔊"):
                res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                audio_path = text_to_audio(answer, res)
                st.audio(audio_path, format="audio/mp3")

# ----------------- RUN APP -----------------
if __name__ == "__main__":
    main()
