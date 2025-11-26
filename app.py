import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import google.generativeai as genai
import requests
import base64
import os

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')

# ==========================================
# 1. API KEYS
# ==========================================
# 🔑 PASTE YOUR KEYS HERE
GOOGLE_API_KEY = "AIzaSyAH-c6ISU1OCnPuzqZQNFBaYmm_iOurzhI"
DEEPGRAM_API_KEY = "e62de15e4f02b05d198520e347120975fa1fb7d8"

# Simple check to warn you if keys are missing
if GOOGLE_API_KEY.startswith("PASTE") or DEEPGRAM_API_KEY.startswith("PASTE"):
    st.error("🚨 Stop! You need to paste your API Keys in lines 19 & 20 of the code.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 2. BEAUTIFUL UI STYLING
# ==========================================
st.set_page_config(page_title="Serenity AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    /* MAIN BACKGROUND */
    .stApp {
        background: url("https://images.unsplash.com/photo-1518173946687-a4c88928d999?q=80&w=2070&auto=format&fit=crop") no-repeat center center fixed;
        background-size: cover;
    }
    
    /* GLASS SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* TITLE STYLING */
    h1 {
        color: #2c3e50;
        text-shadow: 0px 2px 4px rgba(255,255,255, 0.8);
        font-weight: 800;
        background-color: rgba(255,255,255,0.4);
        padding: 10px 20px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
        display: inline-block;
    }
    
    /* CHAT BUBBLES */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin-bottom: 15px;
    }
    
    /* User Message (Green Tint) */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background: rgba(220, 255, 220, 0.75);
        border-right: 5px solid #4CAF50;
    }
    
    /* Input Box */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.5);
        color: #333;
    }

    /* Sidebar Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .glass-card h4 { margin: 0; font-size: 14px; color: #555; }
    .glass-card p { margin: 5px 0 0 0; font-weight: bold; color: #222; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD BRAIN (Safe Mode)
# ==========================================
@st.cache_resource
def load_brain():
    # Verify files exist before trying to load
    required_files = ['counselor_model_full.pkl', 'counselor_vectorizer_full.pkl', 'counselor_encoder_full.pkl']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        st.error(f"⚠️ System Error: Missing Brain Files: {missing}")
        st.info("Please ensure the .pkl files are in the same folder as this script.")
        return None, None, None

    try:
        with open('counselor_model_full.pkl', 'rb') as f: model = pickle.load(f)
        with open('counselor_vectorizer_full.pkl', 'rb') as f: vectorizer = pickle.load(f)
        with open('counselor_encoder_full.pkl', 'rb') as f: encoder = pickle.load(f)
        return model, vectorizer, encoder
    except Exception as e:
        st.error(f"⚠️ Corrupt File Error: {e}")
        return None, None, None

model_clf, vectorizer, label_encoder = load_brain()
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

# ==========================================
# 4. INTELLIGENCE (Gemini 2.5 Flash)
# ==========================================
def generate_text_response(label, user_text):
    if label == 'Suicidal':
        return ("⚠️ CRISIS DETECTED: You are not alone. Please reach out for help immediately. "
                "Call 988 or 911. I am an AI and cannot provide emergency care.")

    prompts = {
        'Anxiety': "You are a warm, soothing counselor. Speak slowly. Validate the anxiety first, then offer a breathing technique.",
        'Depression': "You are a gentle, supportive friend. Use short, soft sentences. Validate their feelings. Do not give 'advice', just listen.",
        'Stress': "You are a helpful, practical coach. Suggest one tiny, easy step they can take right now.",
        'Bi-Polar': "You are a steady, consistent support. Encourage keeping a small routine. Be very calm.",
        'Normal': "You are a friendly, conversational companion. Keep it light and engaging."
    }
    
    system_instruction = prompts.get(label, "Be a kind, supportive listener.")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(f"System: {system_instruction}\nUser: {user_text}")
        return response.text
    except Exception as e:
        return f"I'm having trouble connecting to the cloud. (Error: {e})"

# ==========================================
# 5. VOICE (Deepgram Aura)
# ==========================================
def generate_audio_deepgram(text):
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    try:
        response = requests.post(url, headers=headers, json=payload)
        output_file = "reply.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
        return output_file
    except:
        return None

def autoplay_audio(file_path: str):
    if not file_path: return
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# ==========================================
# 6. MAIN LAYOUT
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712038.png", width=80)
    st.title("Serenity")
    st.markdown("### System Status")
    
    # Custom HTML Cards
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #4CAF50;">
        <h4>🧠 Brain</h4>
        <p>Gemini 2.5 Flash</p>
    </div>
    <div class="glass-card" style="border-left: 5px solid #2196F3;">
        <h4>🗣️ Voice</h4>
        <p>Deepgram Aura</p>
    </div>
    <div class="glass-card" style="border-left: 5px solid #FFC107;">
        <h4>🛡️ Classifier</h4>
        <p>Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("A Safe Space for Students.")

# Title
st.title("🌿 Serenity AI Counsellor")
st.markdown("#### *How are you feeling right now?*")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Type here..."):
    # 1. User
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Logic
    if model_clf:
        # Diagnosis
        cleaned_text = preprocess_text(prompt)
        vec_input = vectorizer.transform([cleaned_text])
        pred_index = model_clf.predict(vec_input)[0]
        predicted_label = label_encoder.inverse_transform([pred_index])[0]
        
        # Generation
        with st.spinner("Thinking..."):
            reply_text = generate_text_response(predicted_label, prompt)
            audio_file = generate_audio_deepgram(reply_text)
            
        # 3. Response
        with st.chat_message("assistant"):
            st.markdown(reply_text)
            if predicted_label != "Normal":
                st.caption(f"🛡️ *Sensing: {predicted_label}*")
            autoplay_audio(audio_file)
            
        st.session_state.messages.append({"role": "assistant", "content": reply_text})
    else:
        st.error("Cannot process message: Brain is not loaded.")