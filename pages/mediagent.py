import os
import uuid
import time
import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
from dotenv import load_dotenv
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# LangChain + Groq
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# ---------------------------
# Train a simple ML classifier
# ---------------------------
train_texts = [
    # Medical
    "What are the symptoms of diabetes?",
    "Suggest a diet plan for weight loss",
    "Yoga poses for back pain relief",
    "Treatment for high blood pressure",
    "Best exercises for fitness",
    "Headache and dizziness when I skip meals",
    "I feel weak during gym workouts, what should I do?",
    "Healthy recipes for better heart health",
    "Tips to reduce stress and improve sleep",
    "Can dehydration cause fatigue?",
    
    # Non-medical
    "How to cook pasta?",
    "Tell me about the latest iPhone",
    "Who won the football match?",
    "How to fix my laptop?",
    "Movie recommendations please",
    "Best tourist places in Europe",
    "Explain quantum computing basics",
    "History of World War II",
    "Weather forecast for tomorrow",
    "How does blockchain work?"
]
train_labels = ["medical"] * 10 + ["non-medical"] * 10

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
clf = MultinomialNB().fit(X_train, train_labels)

def is_medical_query_ml(text: str) -> bool:
    X_test = vectorizer.transform([text])
    proba = clf.predict_proba(X_test)[0]
    prediction = clf.classes_[proba.argmax()]
    confidence = proba.max()

    st.write(f"🔍 DEBUG → Prediction: {prediction}, Confidence: {confidence:.2f}")

    # Allow only strong medical predictions
    if prediction == "medical" and confidence >= 0.6:
        return True
    
    return False


# ---------------------------
# Load API key
# ---------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set it in your .env file.")
    st.stop()

GROQ_API_KEY = GROQ_API_KEY.strip()
st.sidebar.info(f"✅ Loaded Groq API key (starts with: {GROQ_API_KEY[:8]}...)")

# ---------------------------
# Initialize Groq LLM
# ---------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    groq_api_key=GROQ_API_KEY
)

# ---------------------------
# Wikipedia Tool (new style)
# ---------------------------
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wiki]

system_prompt = (
    "You are MediBot, a multilingual medical assistant. "
    "Always answer in a structured format with clear headings. "
    "Focus only on medical and health-oriented queries."
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    system_message=system_prompt
)

# ---------------------------
# Token & Session Tracker
# ---------------------------
DAILY_LIMIT = 100000  # free tier token limit
RESET_INTERVAL = 60 * 60 * 24  # 24 hours

if "token_usage" not in st.session_state:
    st.session_state.token_usage = 0
if "reset_time" not in st.session_state:
    st.session_state.reset_time = time.time() + RESET_INTERVAL
if "queries" not in st.session_state:
    st.session_state.queries = 0
if "response_times" not in st.session_state:
    st.session_state.response_times = []
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

def update_token_usage(text: str):
    """Estimate token usage (~4 chars per token)."""
    est_tokens = max(1, len(text) // 4)
    st.session_state.token_usage += est_tokens
    return est_tokens

def time_until_reset():
    remaining = max(0, int(st.session_state.reset_time - time.time()))
    mins, secs = divmod(remaining, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def show_usage_banner():
    usage = st.session_state.token_usage
    reset_in = time_until_reset()
    if usage >= DAILY_LIMIT:
        st.error(f"⚠ You’ve hit the Free plan limit for MediAgent. Limit resets in {reset_in}.")
    elif usage >= 0.9 * DAILY_LIMIT:
        st.warning(f"⚠ You are close to your daily limit ({usage}/{DAILY_LIMIT}). Resets in {reset_in}.")
    elif usage >= 0.7 * DAILY_LIMIT:
        st.info(f"ℹ You have used {usage}/{DAILY_LIMIT} tokens today. Resets in {reset_in}.")

# ---------------------------
# Language Settings
# ---------------------------
# ---------------------------
# Language Settings (Expanded)
# ---------------------------

LANGUAGES = {
    # Existing Indian languages
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "kn": "Kannada",
    
    # Additional Indian languages
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "or": "Odia",
    "ur": "Urdu",
    
    # Global languages
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean"
}

# gTTS supports most of these languages by ISO code
GTTS_LANG_MAP = {
    k: k for k in LANGUAGES.keys()
}

# Speech recognition codes for Google API (some may not be fully supported)
SPEECH_LANG_CODES = {
    "en": "en-US",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "kn": "kn-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "ml": "ml-IN",
    "or": "or-IN",
    "ur": "ur-IN",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "it": "it-IT",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "ar": "ar-SA",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "ja": "ja-JP",
    "ko": "ko-KR"
}


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="MediBot", page_icon="🩺", layout="centered")
st.title("🩺 MediAgent – Smarter Healthcare, Closer to You")

# Show usage banner at the top
show_usage_banner()

target_lang = st.selectbox("Choose your language", list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
voice_output = st.checkbox("Enable voice response (TTS)")

# --- Voice Input ---
st.subheader("🎤 Voice Input (Speak in your language)")

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

if st.button("Record Voice", key="record_voice_btn"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Please speak now...")
        audio = recognizer.listen(source, timeout=5)
        st.info("Transcribing...")
        try:
            lang_code = SPEECH_LANG_CODES.get(target_lang, "en-US")
            st.session_state.voice_text = recognizer.recognize_google(audio, language=lang_code)
            st.success(f"Recognized: {st.session_state.voice_text}")
        except Exception as e:
            st.error(f"Voice recognition failed: {str(e)}")

# --- Text Input ---
user_input = st.text_area(
    "Ask for Medical Queries (in your language):",
    value=st.session_state.voice_text
)

if st.button("Get Advice", key="get_advice_btn"):
    if st.session_state.token_usage >= DAILY_LIMIT:
        st.error("❌ Daily token limit reached. Please try again after reset.")
    elif user_input:
        try:
            start_time = time.time()
            
            # Step 1: Translate to English
            to_english = GoogleTranslator(source='auto', target='en').translate(user_input)

            # Step 2: ML Classify Query
            if not is_medical_query_ml(to_english):
                st.warning("❌ This query is not related to medical topics. Please Try Again☺.")
                st.stop()

            # Step 3: Get Advice from Agent
            advice_en = agent.run(to_english)

            # Response time
            elapsed = time.time() - start_time
            st.session_state.response_times.append(elapsed)
            st.session_state.queries += 1

            # Update usage
            update_token_usage(to_english + advice_en)

            # Step 4: Translate Back
            translated_advice = GoogleTranslator(source='en', target=target_lang).translate(advice_en)

            # Step 5: Display
            st.markdown("## 🩺 MediBot Advice")
            st.markdown(translated_advice)

            # Optional Voice
            if voice_output:
                gtts_lang = GTTS_LANG_MAP.get(target_lang, "en")
                filename = f"output_{uuid.uuid4().hex}.mp3"
                tts = gTTS(translated_advice, lang=gtts_lang)
                tts.save(filename)
                st.audio(filename, format="audio/mp3")
                os.remove(filename)

        except Exception as e:
            st.error(f"⚠ Error: {str(e)}")
    else:
        st.warning("Please enter your query to get advice.")

# ---------------------------
# Sidebar - Research Stats
# ---------------------------
st.sidebar.markdown("## 📊 Research Metrics")
st.sidebar.write(f"*Tokens used today:* {st.session_state.token_usage:,} / {DAILY_LIMIT:,}")
st.sidebar.write(f"*Total Queries:* {st.session_state.queries}")

if st.session_state.response_times:
    avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
    st.sidebar.write(f"*Avg Response Time:* {avg_time:.2f} sec")
    st.sidebar.write(f"*Last Response Time:* {st.session_state.response_times[-1]:.2f} sec")

elapsed_session = time.time() - st.session_state.start_time
st.sidebar.write(f"*Session Duration:* {elapsed_session/60:.1f} min")

st.sidebar.write("*Model:* LLaMA-3.3-70B (Groq)")
st.sidebar.write("*Creativity levels:* 0.5")
st.sidebar.write(f"⏳ *Resets in:* {time_until_reset()}")

st.sidebar.progress(min(st.session_state.token_usage / DAILY_LIMIT, 1.0))
st.set_page_config(page_title="MediBot", page_icon="🩺", layout="centered")

