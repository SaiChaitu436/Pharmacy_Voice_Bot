import streamlit as st
import assemblyai as aai
import os
import base64
import tempfile
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import requests
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import time

# ✅ Load environment variables
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("aai.settings.api_key")
ELEVENLABS_API_KEY = os.getenv("elevenlabs_api_key")
SAMBANOVA_API_URL = os.getenv("SAMBANOVA_API_URL")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
VOICE_ID = os.getenv("elevenlabs_voice_id")
MODEL_ID = os.getenv("elevenlabs_model_id")
COLLECTION_NAME = "drug_data"

# ✅ Configure AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber()

# ✅ Configure ElevenLabs Client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ✅ Initialize HuggingFace embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Qdrant client
client = QdrantClient(url=os.getenv("qdrant_client_url"), api_key=os.getenv("qdrant_client_key"))

# ✅ Function to retrieve context from Qdrant
def retrieve_context(user_input):
    query_embedding = hf_embeddings.embed_query(user_input)
    results = client.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=5)
    return "\n".join([
        f"Brand: {hit.payload['brand_name']}, Manufacturer: {hit.payload['manufacturer']}, Dosage: {hit.payload['dosage_form']}"
        for hit in results
    ])

# ✅ Function to query SambaNova API
def query_sambanova(prompt):
    headers = {"Authorization": f"Bearer {SAMBANOVA_API_KEY}"}
    payload = {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }
    response = requests.post(SAMBANOVA_API_URL, json=payload, headers=headers)
    return response.json().get("choices")[0].get("message").get("content", "Error fetching response")

# ✅ Autoplay audio function
def play_audio_autoplay(audio_stream):
    audio_data = b"".join(audio_stream)
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# ✅ UI Styling
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎤", layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎤 AI Voice Assistant")

greeting = "Thank you for calling Sails Pharma Assistant Saambaa. How can I assist you?"
st.write(f"💬 {greeting}")

if "greeting_played" not in st.session_state:
    st.session_state.greeting_played = False

if not st.session_state.greeting_played:
    audio_stream = elevenlabs_client.text_to_speech.convert(text=greeting, voice_id=VOICE_ID, model_id=MODEL_ID)
    play_audio_autoplay(audio_stream)
    st.session_state.greeting_played = True

# ✅ Recording Function with Countdown
def record_audio(duration=5, samplerate=44100):
    try:
        st.session_state.recording = True
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)

        countdown_placeholder = st.empty()
        for i in range(duration, 0, -1):
            if not st.session_state.recording:
                break  # Stop recording if interrupted
            countdown_placeholder.write(f"🎤 Recording... {i} sec remaining")
            time.sleep(1)

        countdown_placeholder.empty()
        sd.wait()
        st.session_state.recording = False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            write(temp_wav.name, samplerate, audio_data)
            return temp_wav.name
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

# ✅ Recording Button
if st.button("🎙️ Start Recording"):
    audio_path = record_audio()
    if audio_path:
        st.success("✅ Recording complete! Processing audio...")
        transcript = transcriber.transcribe(audio_path).text
        st.write(f"📝 **You said:** {transcript}")

        context = retrieve_context(transcript)
        prompt = f"Answer based on context:\nContext: {context}\nQuestion: {transcript}."
        ai_response = query_sambanova(prompt)
        
        st.write(f"🤖 **AI Response:** {ai_response}")

        with st.spinner("Generating audio response..."):
            audio_stream = elevenlabs_client.text_to_speech.convert(text=ai_response, voice_id=VOICE_ID, model_id=MODEL_ID)
            play_audio_autoplay(audio_stream)

        os.unlink(audio_path)
    else:
        st.error("❌ Audio recording failed. Please try again.")

# ✅ File Upload Option
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    st.success("✅ Audio uploaded successfully! Processing...")
    transcript = transcriber.transcribe(temp_path).text
    st.write(f"📝 **You said:** {transcript}")

    context = retrieve_context(transcript)
    prompt = f"Answer based on context:\nContext: {context}\nQuestion: {transcript}."
    ai_response = query_sambanova(prompt)
    st.write(f"🤖 **AI Response:** {ai_response}")

    with st.spinner("Generating audio response..."):
        audio_stream = elevenlabs_client.text_to_speech.convert(text=ai_response, voice_id=VOICE_ID, model_id=MODEL_ID)
        play_audio_autoplay(audio_stream)

    os.unlink(temp_path)
