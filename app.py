import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import edge_tts
import asyncio
import easyocr
from PIL import Image
import numpy as np
from langdetect import detect
import os
import base64
import folium
from streamlit_folium import st_folium
from folium.features import DivIcon
import torch
import docx
import fitz  # PyMuPDF
import gc

# Configuration de la page
st.set_page_config(page_title="Universal Bridge AI", layout="wide")

# --- 1. DONNÉES ET CONFIGURATION ---
MAP_DATA = {
    "Français": {"coords": [46.2276, 2.2137], "iso": "fr", "flag": "🇫🇷"},
    "Anglais": {"coords": [37.0902, -95.7129], "iso": "en", "flag": "🇺🇸"},
    "Turc": {"coords": [38.9637, 35.2433], "iso": "tr", "flag": "🇹🇷"},
    "Espagnol": {"coords": [40.4637, -3.7492], "iso": "es", "flag": "🇪🇸"},
    "Chinois": {"coords": [35.8617, 104.1954], "iso": "zh", "flag": "🇨🇳"},
    "Coréen": {"coords": [35.9078, 127.7669], "iso": "ko", "flag": "🇰🇷"},
}

LANG_CODES = {
    "Français": "fra_Latn", "Anglais": "eng_Latn", "Turc": "tur_Latn",
    "Espagnol": "spa_Latn", "Chinois": "zho_Hans", "Coréen": "kor_Hang"
}

DETECTION_MAP = {
    'fr': "Français", 'en': "Anglais", 'tr': "Turc",
    'es': "Espagnol", 'zh': "Chinois", 'ko': "Coréen"
}

VOICE_MAPPING = {
    "Français": {"Homme": "fr-FR-HenriNeural", "Femme": "fr-FR-DeniseNeural"},
    "Anglais": {"Homme": "en-US-GuyNeural", "Femme": "en-US-JennyNeural"},
    "Turc": {"Homme": "tr-TR-AhmetNeural", "Femme": "tr-TR-EmelNeural"},
    "Espagnol": {"Homme": "es-ES-AlvaroNeural", "Femme": "es-ES-ElviraNeural"},
    "Chinois": {"Homme": "zh-CN-YunxiNeural", "Femme": "zh-CN-XiaoxiaoNeural"},
    "Coréen": {"Homme": "ko-KR-InJoonNeural", "Femme": "ko-KR-SunHiNeural"},
}

# --- 2. CHARGEMENT DES MODÈLES (OPTIMISÉ RAM) ---
@st.cache_resource
def load_essentials():
    # Traduction (NLLB-200) en demi-précision (float16)
    nllb_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(nllb_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model, torch_dtype=torch.float16)
    
    # OCR
    reader = easyocr.Reader(['fr', 'en', 'es', 'tr', 'ch_sim', 'ko'])
    
    # Chatbot (Blenderbot) en demi-précision
    chat_model_name = "facebook/blenderbot-400M-distill"
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
    chat_model = AutoModelForCausalLM.from_pretrained(chat_model_name, torch_dtype=torch.float16)
    
    return tokenizer, model, reader, chat_tokenizer, chat_model

# Lancement du chargement
with st.spinner("Chargement des cerveaux de l'IA..."):
    tokenizer, model, reader, chat_tokenizer, chat_model = load_essentials()
    gc.collect() # Nettoyage immédiat de la RAM après chargement

# --- 3. LOGIQUE FONCTIONNELLE ---
async def generate_audio(text, voice, filename):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

# --- 4. INTERFACE ---
st.title("🌐 Universal Bridge AI")
st.info("Traduction, OCR et Chatbot Intelligent")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Entrée")
    uploaded_file = st.file_uploader("Document ou Image", type=["pdf", "docx", "png", "jpg"])
    
    file_text = ""
    if uploaded_file:
        if uploaded_file.type in ["image/png", "image/jpeg"]:
            image = Image.open(uploaded_file)
            file_text = " ".join(reader.readtext(np.array(image), detail=0))
        elif uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            file_text = " ".join([page.get_text() for page in doc])
    
    input_text = st.text_area("Texte :", value=file_text, height=150)
    target_lang = st.selectbox("Vers :", list(LANG_CODES.keys()))
    voice_type = st.radio("Voix :", ["Femme", "Homme"], horizontal=True)

    with st.expander("💬 Chatbot d'aide"):
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        
        if prompt := st.chat_input("Posez une question..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            inputs = chat_tokenizer([prompt], return_tensors="pt")
            res_ids = chat_model.generate(**inputs, max_new_tokens=50)
            answer = chat_tokenizer.decode(res_ids[0], skip_special_tokens=True)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()

with col2:
    st.subheader("📤 Résultat")
    if st.button("🚀 TRADUIRE"):
        if input_text.strip():
            with st.spinner("L'IA réfléchit..."):
                # Traduction
                inputs = tokenizer(input_text, return_tensors="pt")
                translated_tokens = model.generate(
                    **inputs, 
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODES[target_lang])
                )
                translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                st.success(translation)
                
                # Audio
                v_name = VOICE_MAPPING[target_lang][voice_type]
                asyncio.run(generate_audio(translation, v_name, "output.mp3"))
                st.audio("output.mp3")
                
                # Carte
                m = folium.Map(location=MAP_DATA[target_lang]["coords"], zoom_start=4)
                folium.Marker(MAP_DATA[target_lang]["coords"], popup=target_lang).add_to(m)
                st_folium(m, width=500, height=250)
                
                gc.collect() # Libère la RAM après l'effort
