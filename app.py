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

# --- 1. FONCTIONS ET COORDONNÉES ---
MAP_DATA = {
    "Français": {"coords": [46.2276, 2.2137], "iso": "fr", "img": "france.png", "flag": "🇫🇷"},
    "Anglais": {"coords": [37.0902, -95.7129], "iso": "en", "img": "royaume-uni.png", "flag": "🇺🇸"},
    "Turc": {"coords": [38.9637, 35.2433], "iso": "tr", "img": "dinde.png", "flag": "🇹🇷"},
    "Espagnol": {"coords": [40.4637, -3.7492], "iso": "es", "img": "drapeau.png", "flag": "🇪🇸"},
    "Chinois": {"coords": [35.8617, 104.1954], "iso": "zh", "img": "chine.png", "flag": "🇨🇳"},
    "Coréen": {"coords": [35.9078, 127.7669], "iso": "ko", "img": "coree-du-sud.png", "flag": "🇰🇷"},
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

# --- 2. CHARGEMENT OPTIMISÉ DES MODÈLES (POUR ÉVITER LE CRASH RAM) ---
@st.cache_resource
def load_essentials():
    # Traduction NLLB-200 (Version Distilled)
    nllb_model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
    # On force le chargement en float16 pour économiser 50% de RAM
    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name, torch_dtype=torch.float16)
    
    # OCR
    reader = easyocr.Reader(['fr', 'en', 'es', 'tr', 'ch_sim', 'ko'])
    
    # Chatbot Blenderbot
    chat_model_name = "facebook/blenderbot-400M-distill"
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
    chat_model = AutoModelForCausalLM.from_pretrained(chat_model_name, torch_dtype=torch.float16)
    
    return tokenizer, model, reader, chat_tokenizer, chat_model

# Exécution du chargement
with st.spinner("Initialisation des systèmes IA (Veuillez patienter)..."):
    tokenizer, model, reader, chat_tokenizer, chat_model = load_essentials()
    gc.collect() # Nettoyage de la RAM après chargement

# --- 3. FONCTIONS UTILITAIRES ---
async def generate_audio(text, voice, filename):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([p.text for p in doc.paragraphs])

# --- 4. INTERFACE UTILISATEUR ---
st.title("🌐 Universal Bridge AI")
st.markdown("---")

# Gestion de l'historique et du chat dans session_state
if "history" not in st.session_state: st.session_state.history = []
if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
if "detected_info" not in st.session_state: st.session_state.detected_info = None

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Entrée")
    
    # Option d'importation de fichier
    uploaded_file = st.file_uploader("Importer un document (PDF, DOCX, Image)", type=["pdf", "docx", "png", "jpg", "jpeg"])
    file_text = ""
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            file_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_text = extract_text_from_docx(uploaded_file)
        else:
            image = Image.open(uploaded_file)
            file_text = " ".join(reader.readtext(np.array(image), detail=0))
    
    input_text = st.text_area("Texte à traduire :", value=file_text, height=150)
    
    target_lang = st.selectbox("Traduire vers :", list(LANG_CODES.keys()))
    voice_type = st.radio("Voix :", ["Femme", "Homme"], horizontal=True)

    # Section Chatbot (Optionnelle pour économiser la RAM en cas de bug)
    with st.expander("💬 Chatbot d'assistance"):
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if prompt := st.chat_input("Posez une question..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            inputs = chat_tokenizer([prompt], return_tensors="pt")
            res_tokens = chat_model.generate(**inputs, max_new_tokens=100)
            response = chat_tokenizer.decode(res_tokens[0], skip_special_tokens=True)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

    if st.button("🔍 Détecter la langue"):
        if input_text.strip():
            iso = detect(input_text).split('-')[0]
            st.session_state.detected_info = DETECTION_MAP.get(iso, "Inconnue")
            st.info(f"Langue détectée : {st.session_state.detected_info}")

with col2:
    st.subheader("📤 Résultat")
    if st.button("🚀 TRADUIRE"):
        if input_text.strip():
            with st.spinner("Traduction en cours..."):
                target_code = LANG_CODES[target_lang]
                inputs = tokenizer(input_text, return_tensors="pt")
                translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code))
                translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                st.success(translation)
                
                # Audio
                voice = VOICE_MAPPING[target_lang][voice_type]
                asyncio.run(generate_audio(translation, voice, "output.mp3"))
                st.audio("output.mp3")
                
                # Map
                m = folium.Map(location=MAP_DATA[target_lang]["coords"], zoom_start=4)
                folium.Marker(MAP_DATA[target_lang]["coords"], popup=target_lang).add_to(m)
                st_folium(m, width=700, height=300)
                
                gc.collect() # Libère la mémoire après chaque traduction

# Sidebar Historique
with st.sidebar:
    st.title("📜 Historique")
    if st.button("Effacer l'historique"): st.session_state.history = []
