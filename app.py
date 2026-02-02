import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
import easyocr
import asyncio
import edge_tts
from PIL import Image
import numpy as np
import folium
from streamlit_folium import st_folium
import gc

# Configuration ultra-légère
st.set_page_config(page_title="Universal Bridge AI", layout="centered")

# --- CONFIGURATION ---
LANG_CODES = {"Français": "fra_Latn", "Anglais": "eng_Latn", "Turc": "tur_Latn", "Espagnol": "spa_Latn", "Chinois": "zho_Hans", "Coréen": "kor_Hang"}
MAP_DATA = {"Français": [46.2, 2.2], "Anglais": [37.0, -95.0], "Turc": [38.9, 35.2], "Espagnol": [40.4, -3.7], "Chinois": [35.8, 104.1], "Coréen": [35.9, 127.7]}

# --- FONCTIONS DE CHARGEMENT À LA DEMANDE ---
def get_translator():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return tokenizer, model

def get_ocr():
    return easyocr.Reader(['fr', 'en', 'es'])

# --- INTERFACE ---
st.title("🌐 Universal Bridge AI")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Entrée")
    input_text = st.text_area("Texte à traduire :", height=150)
    target_lang = st.selectbox("Vers :", list(LANG_CODES.keys()))
    
    # Bouton de traduction
    if st.button("🚀 TRADUIRE"):
        if input_text.strip():
            with st.spinner("IA en cours de chargement..."):
                # On charge, on utilise, et on libère la RAM
                tokenizer, model = get_translator()
                inputs = tokenizer(input_text, return_tensors="pt")
                translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODES[target_lang]))
                translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                st.session_state.result = translation
                # Nettoyage manuel de la mémoire
                del tokenizer
                del model
                gc.collect()

with col2:
    st.subheader("📤 Résultat")
    if "result" in st.session_state:
        st.success(st.session_state.result)
        
        # Affichage de la carte simple
        m = folium.Map(location=MAP_DATA[target_lang], zoom_start=3)
        folium.Marker(MAP_DATA[target_lang]).add_to(m)
        st_folium(m, height=250)

# --- SIDEBAR POUR L'OCR (SÉPARÉ POUR ÉVITER LES CRASHES) ---
with st.sidebar:
    st.header("🖼️ Option OCR")
    img_file = st.file_uploader("Extraire texte d'une image", type=['png', 'jpg'])
    if img_file and st.button("Lire l'image"):
        reader = get_ocr()
        image = Image.open(img_file)
        result = reader.readtext(np.array(image), detail=0)
        st.write("Texte trouvé :", " ".join(result))
        del reader
        gc.collect()
