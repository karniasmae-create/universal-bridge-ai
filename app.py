import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import edge_tts
import asyncio
import easyocr
from PIL import Image
import numpy as np
from langdetect import detect
import folium
from streamlit_folium import st_folium
import torch
import docx
import fitz  # PyMuPDF
import gc

# Config
st.set_page_config(page_title="Universal Bridge AI", layout="wide")

# Données de la carte
MAP_DATA = {
    "Français": {"coords": [46.2276, 2.2137], "iso": "fr", "flag": "🇫🇷"},
    "Anglais": {"coords": [37.0902, -95.7129], "iso": "en", "flag": "🇺🇸"},
    "Turc": {"coords": [38.9637, 35.2433], "iso": "tr", "flag": "🇹🇷"},
    "Espagnol": {"coords": [40.4637, -3.7492], "iso": "es", "flag": "🇪🇸"},
    "Chinois": {"coords": [35.8617, 104.1954], "iso": "zh", "flag": "🇨🇳"},
    "Coréen": {"coords": [35.9078, 127.7669], "iso": "ko", "flag": "🇰🇷"},
}

LANG_CODES = {"Français": "fra_Latn", "Anglais": "eng_Latn", "Turc": "tur_Latn", "Espagnol": "spa_Latn", "Chinois": "zho_Hans", "Coréen": "kor_Hang"}
VOICE_MAPPING = {
    "Français": {"Homme": "fr-FR-HenriNeural", "Femme": "fr-FR-DeniseNeural"},
    "Anglais": {"Homme": "en-US-GuyNeural", "Femme": "en-US-JennyNeural"},
}

@st.cache_resource
def load_essentials():
    try:
        # 1. Traduction (NLLB)
        nllb_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(nllb_name, use_fast=False) # Désactive Fast si erreur
        model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name, torch_dtype=torch.float16)
        
        # 2. OCR
        reader = easyocr.Reader(['fr', 'en'])
        
        # 3. Chatbot
        chat_name = "facebook/blenderbot-400M-distill"
        chat_tokenizer = AutoTokenizer.from_pretrained(chat_name)
        chat_model = AutoModelForCausalLM.from_pretrained(chat_name, torch_dtype=torch.float16)
        
        return tokenizer, model, reader, chat_tokenizer, chat_model
    except Exception as e:
        st.error(f"Erreur fatale lors du chargement : {e}")
        return None, None, None, None, None

# Interface
st.title("🌐 Universal Bridge AI")

tokenizer, model, reader, chat_tokenizer, chat_model = load_essentials()

if tokenizer is None:
    st.warning("⚠️ L'application n'a pas pu charger les modèles IA. Vérifiez que 'sentencepiece' est bien dans requirements.txt.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Entrée")
    input_text = st.text_area("Texte à traduire :", height=150)
    target_lang = st.selectbox("Langue cible :", list(LANG_CODES.keys()))

with col2:
    st.subheader("📤 Résultat")
    if st.button("🚀 TRADUIRE"):
        if input_text:
            with st.spinner("Traduction..."):
                inputs = tokenizer(input_text, return_tensors="pt")
                translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODES[target_lang]))
                result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                st.success(result)
                
                # Carte
                m = folium.Map(location=MAP_DATA[target_lang]["coords"], zoom_start=4)
                folium.Marker(MAP_DATA[target_lang]["coords"]).add_to(m)
                st_folium(m, width=500, height=250)
                
                gc.collect()
