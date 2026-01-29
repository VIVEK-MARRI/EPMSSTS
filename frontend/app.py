"""
Beautiful MVP UI for EPMSSTS - Emotion-Preserving Multilingual Speech-to-Speech Translation
"""
import io
import requests
import streamlit as st

BACKEND_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="EPMSSTS - AI Speech Translation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        font-size: 2.5em;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.1em;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    .card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #6366f1;
    }
    .card.success {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    .section-title {
        font-size: 1.8em;
        font-weight: 700;
        color: #333;
        margin: 30px 0 20px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid #6366f1;
    }
    .emotion-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 600;
        margin: 5px;
    }
    .emotion-happy { background: #fef08a; color: #854d0e; }
    .emotion-sad { background: #e0e7ff; color: #3730a3; }
    .emotion-angry { background: #fee2e2; color: #7f1d1d; }
    .emotion-neutral { background: #dbeafe; color: #082f49; }
    .emotion-fearful { background: #fce7f3; color: #831843; }
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    .divider {
        margin: 30px 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, transparent);
    }
    </style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header"""
    st.markdown("""
        <div class="main-header">
            <h1>🌍 EPMSSTS</h1>
            <p>Emotion-Preserving Multilingual Speech-to-Speech Translation</p>
            <p style="font-size: 0.95em; margin-top: 15px; opacity: 0.9;">
                Translate speech across languages while preserving emotion and dialect
            </p>
        </div>
    """, unsafe_allow_html=True)


def post_request(endpoint, files, data):
    """Helper function to make POST requests to backend"""
    url = f"{BACKEND_BASE_URL}{endpoint}"
    resp = requests.post(url, files=files, data=data, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Backend error ({resp.status_code}): {resp.text}")
    return resp.json()


def render_emotion_badge(emotion: str, confidence: float = None):
    """Render emotion badge with styling"""
    emotion_lower = emotion.lower()
    badge_class = f"emotion-{emotion_lower}"
    conf_text = f" ({confidence:.1%})" if confidence else ""
    return f'<span class="emotion-badge {badge_class}">{emotion}{conf_text}</span>'


# Main app
render_header()

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None

# Create tabs
tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Record Audio"])

with tab1:
    st.markdown('<h2 class="section-title">📁 Upload & Translate</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.write("**Select your audio file:**")
        audio_file = st.file_uploader(
            "Choose audio file", 
            type=["wav", "mp3", "ogg", "m4a"],
            label_visibility="collapsed"
        )
        if audio_file:
            st.audio(audio_file)
    
    with col2:
        st.write("**Settings:**")
        target_lang = st.radio(
            "Target Language",
            ["English", "Hindi", "Telugu"],
            label_visibility="collapsed"
        )
        target_emotion = st.radio(
            "Target Emotion",
            ["Neutral", "Happy", "Sad", "Angry", "Fearful"],
            label_visibility="collapsed"
        )
    
    if audio_file and st.button("🚀 Process & Translate", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                audio_bytes = audio_file.read()
                files = {"file": (audio_file.name, io.BytesIO(audio_bytes), audio_file.type)}
                
                lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
                emotion_map = {
                    "Neutral": "neutral", "Happy": "happy", "Sad": "sad",
                    "Angry": "angry", "Fearful": "fearful"
                }
                
                # Step 1: Transcribe
                st.info("📝 Transcribing...")
                stt_result = post_request(
                    "/stt/transcribe",
                    files,
                    {}
                )
                
                # Step 2: Analyze emotion
                st.info("😊 Analyzing emotion...")
                emotion_result = post_request(
                    "/emotion/analyze",
                    files,
                    {}
                )
                
                # Step 3: Detect dialect
                st.info("🗣️ Detecting dialect...")
                dialect_result = post_request(
                    "/dialect/detect",
                    files,
                    {}
                )
                
                # Step 4: Full pipeline
                st.info("🌐 Translating...")
                pipeline_result = post_request(
                    "/process/speech-to-speech",
                    files,
                    {
                        "target_lang": lang_map[target_lang],
                        "target_emotion": emotion_map[target_emotion]
                    }
                )
                
                st.session_state.results = {
                    "stt": stt_result,
                    "emotion": emotion_result,
                    "dialect": dialect_result,
                    "pipeline": pipeline_result,
                    "target_lang": target_lang,
                    "target_emotion": target_emotion
                }
                st.success("✅ Done!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with tab2:
    st.markdown('<h2 class="section-title">🎤 Live Recording</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.write("**Record your message:**")
        audio_data = st.audio_input("Record", label_visibility="collapsed")
        if audio_data:
            st.audio(audio_data)
    
    with col2:
        st.write("**Settings:**")
        live_target_lang = st.radio(
            "Target Language",
            ["English", "Hindi", "Telugu"],
            label_visibility="collapsed",
            key="live_lang"
        )
        live_target_emotion = st.radio(
            "Target Emotion",
            ["Neutral", "Happy", "Sad", "Angry", "Fearful"],
            label_visibility="collapsed",
            key="live_emotion"
        )
    
    if audio_data and st.button("🚀 Process Recording", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                files = {"file": ("recording.wav", audio_data, "audio/wav")}
                
                lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
                emotion_map = {
                    "Neutral": "neutral", "Happy": "happy", "Sad": "sad",
                    "Angry": "angry", "Fearful": "fearful"
                }
                
                st.info("📝 Transcribing...")
                stt_result = post_request(
                    "/stt/transcribe",
                    files,
                    {}
                )
                
                st.info("😊 Analyzing emotion...")
                emotion_result = post_request(
                    "/emotion/analyze",
                    files,
                    {}
                )
                
                st.info("🗣️ Detecting dialect...")
                dialect_result = post_request(
                    "/dialect/detect",
                    files,
                    {}
                )
                
                st.info("🌐 Translating...")
                pipeline_result = post_request(
                    "/process/speech-to-speech",
                    files,
                    {
                        "target_lang": lang_map[live_target_lang],
                        "target_emotion": emotion_map[live_target_emotion]
                    }
                )
                
                st.session_state.results = {
                    "stt": stt_result,
                    "emotion": emotion_result,
                    "dialect": dialect_result,
                    "pipeline": pipeline_result,
                    "target_lang": live_target_lang,
                    "target_emotion": live_target_emotion
                }
                st.success("✅ Done!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.results:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">📊 Results</h2>', unsafe_allow_html=True)
    
    res = st.session_state.results
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎙️ Original Audio")
        
        if "stt" in res:
            st.write(f"**Transcript:** {res['stt'].get('text', 'N/A')}")
        
        if "emotion" in res:
            emotion_data = res["emotion"]
            if isinstance(emotion_data, dict) and "emotion" in emotion_data:
                emotion = emotion_data.get("emotion", "Unknown")
                confidence = emotion_data.get("confidence", 0)
                st.markdown(f"**Emotion:** {render_emotion_badge(emotion, confidence)}", unsafe_allow_html=True)
        
        if "dialect" in res:
            dialect_data = res["dialect"]
            if isinstance(dialect_data, dict) and "dialect" in dialect_data:
                st.write(f"**Dialect:** 🗣️ {dialect_data.get('dialect', 'N/A')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card success">', unsafe_allow_html=True)
        st.markdown("### 🌍 Translated Output")
        
        st.write(f"**Language:** {res.get('target_lang', 'N/A')}")
        st.markdown(f"**Emotion:** {render_emotion_badge(res.get('target_emotion', 'N/A').title())}", unsafe_allow_html=True)
        
        if "pipeline" in res:
            pipeline = res["pipeline"]
            if "translated_text" in pipeline:
                st.write(f"**Translation:** {pipeline.get('translated_text', 'N/A')}")
            
            if "output_audio_url" in pipeline:
                st.write("**Audio Output:**")
                st.audio(f"{BACKEND_BASE_URL}{pipeline['output_audio_url']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📋 Full Results"):
        st.json(res)


