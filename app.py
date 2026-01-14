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

# NEW: Import our Knowledge Base
from knowledge_base import clinical_protocols, few_shot_examples

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# ==========================================
# 1. API KEYS
# ==========================================
GOOGLE_API_KEY = "// Insert your Google Gemini API key here //"
DEEPGRAM_API_KEY = "// Insert your Deepgram API key here //"

genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 2. UI STYLING
# ==========================================
st.set_page_config(page_title="Serenity AI", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1518173946687-a4c88928d999?q=80&w=2070&auto=format&fit=crop") no-repeat center center fixed;
        background-size: cover;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.3);
    }
    h1 {
        color: #2c3e50;
        text-shadow: 0px 2px 4px rgba(255,255,255, 0.8);
        background-color: rgba(255,255,255,0.4);
        padding: 10px 20px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
        display: inline-block;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin-bottom: 15px;
    }
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background: rgba(220, 255, 220, 0.75);
        border-right: 5px solid #4CAF50;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.5);
    }
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
# 3. LOAD BRAIN
# ==========================================
@st.cache_resource
def load_brain():
    try:
        with open('counselor_model_full.pkl', 'rb') as f: model = pickle.load(f)
        with open('counselor_vectorizer_full.pkl', 'rb') as f: vectorizer = pickle.load(f)
        with open('counselor_encoder_full.pkl', 'rb') as f: encoder = pickle.load(f)
        return model, vectorizer, encoder
    except:
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
# 4. INTELLIGENCE (RAG + Prompt Engineering)
# ==========================================
def generate_rag_response(label, user_text):
    # Safety Gate
    if label == 'Suicidal':
        return ("⚠️ CRISIS DETECTED: You are not alone. Please reach out for help immediately. "
                "Call 988 or 911. I am an AI and cannot provide emergency care.")

    # 1. RETRIEVE context from Knowledge Base
    protocol_context = clinical_protocols.get(label, "Use general supportive listening.")
    example_context = few_shot_examples.get(label, "")

    # 2. CONSTRUCT THE SUPER PROMPT - MODIFIED FOR LONGER RESPONSES
    system_prompt = f"""
    ROLE: You are an expert Clinical Psychologist specializing in CBT and DBT with 20+ years of experience.
    
    CURRENT PATIENT STATUS: The patient is experiencing {label}.
    
    CLINICAL PROTOCOL TO USE:
    {protocol_context}
    
    FEW-SHOT TRAINING EXAMPLE:
    {example_context}
    
    INSTRUCTIONS:
    1. START with warm validation of the user's emotion (2-3 sentences acknowledging their feelings)
    2. EXPLAIN the psychological mechanism behind what they're experiencing (3-4 sentences)
    3. INTRODUCE the specific therapeutic technique from the protocol with its rationale (2-3 sentences)
    4. GUIDE the user through the technique step-by-step with detailed instructions (5-7 sentences)
    5. PROVIDE a concrete example or scenario to illustrate the technique (2-3 sentences)
    6. OFFER additional coping strategies or tips related to their situation (3-4 sentences)
    7. END with encouragement and an open invitation to continue the conversation (1-2 sentences)
    
    TONE: Warm, professional, empathetic, and conversational
    TARGET LENGTH: 15-20 sentences (approximately 250-350 words)
    
    Do NOT rush through the response. Take time to fully explain concepts and provide comprehensive guidance.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Configure generation for longer outputs
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,  # Increased from default
        }
        
        response = model.generate_content(
            f"System Context: {system_prompt}\n\nUser Input: {user_text}",
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        return f"I'm having trouble connecting. (Error: {e})"

# ==========================================
# 5. VOICE
# ==========================================
def generate_audio_deepgram(text):
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, json={"text": text})
        with open("reply.mp3", "wb") as f: f.write(response.content)
        return "reply.mp3"
    except: return None

def autoplay_audio(file_path):
    if not file_path: return
    with open(file_path, "rb") as f: data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">', unsafe_allow_html=True)

# ==========================================
# 6. MAIN LAYOUT
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712038.png", width=80)
    st.title("Serenity")
    
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #4CAF50;">
        <h4>🧠 Brain</h4>
        <p>Gemini 2.5 Flash</p>
    </div>
    <div class="glass-card" style="border-left: 5px solid #9C27B0;">
        <h4>📚 RAG System</h4>
        <p>Active (CBT Protocols)</p>
    </div>
    <div class="glass-card" style="border-left: 5px solid #FFC107;">
        <h4>🛡️ Classifier</h4>
        <p>Logistic Regression</p>
    </div>
    <div class="glass-card" style="border-left: 5px solid #2196F3;">
        <h4>📝 Response Mode</h4>
        <p>Comprehensive (250-350 words)</p>
    </div>
    """, unsafe_allow_html=True)

    # LIVE DEBUGGING (Shows the teacher the RAG is working)
    st.markdown("---")
    st.markdown("### 🔍 Live RAG Context")
    if "last_rag_context" in st.session_state:
        with st.expander("See Injected Protocol"):
            st.info(st.session_state.last_rag_context)

st.title("🌿 Serenity AI Counsellor")
st.markdown("#### *How are you feeling right now?*")

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Type here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if model_clf:
        cleaned_text = preprocess_text(prompt)
        vec_input = vectorizer.transform([cleaned_text])
        pred_index = model_clf.predict(vec_input)[0]
        predicted_label = label_encoder.inverse_transform([pred_index])[0]
        
        # Save context for debug sidebar
        st.session_state.last_rag_context = clinical_protocols.get(predicted_label, "General Support")
        
        with st.spinner("Retrieving Clinical Protocol & Generating Comprehensive Response..."):
            reply_text = generate_rag_response(predicted_label, prompt)
            audio_file = generate_audio_deepgram(reply_text)
            
        with st.chat_message("assistant"):
            st.markdown(reply_text)
            st.caption(f"🛡️ Diagnosis: {predicted_label} | 📚 Protocol: {predicted_label} CBT | 📊 Response Length: ~{len(reply_text.split())} words")
            autoplay_audio(audio_file)
            
        st.session_state.messages.append({"role": "assistant", "content": reply_text})