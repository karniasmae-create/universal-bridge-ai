import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

# --- 1. FONCTIONS ET COORDONNÉES ---
MAP_DATA = {
    "Français": {"coords": [46.2276, 2.2137], "iso": "fr", "img": "france.png", "flag": "🇫🇷"},
    "Anglais": {"coords": [37.0902, -95.7129], "iso": "en", "img": "royaume-uni.png", "flag": "🇺🇸"},
    "Turc": {"coords": [38.9637, 35.2433], "iso": "tr", "img": "dinde.png", "flag": "🇹🇷"},
    "Espagnol": {"coords": [40.4637, -3.7492], "iso": "es", "img": "drapeau.png", "flag": "🇪🇸"},
    "Chinois": {"coords": [35.8617, 104.1954], "iso": "zh", "img": "chine.png", "flag": "🇨🇳"},
    "Coréen": {"coords": [35.9078, 127.7669], "iso": "ko", "img": "coree-du-sud.png", "flag": "🇰🇷"}
}

DETECTION_MAP = {v["iso"]: {"coords": v["coords"], "name": k, "img": v["img"], "flag": v["flag"]} for k, v in MAP_DATA.items()}

def get_base64(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

def set_background(png_file):
    bin_str = get_base64(png_file)
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
        }}
        [data-testid="stVerticalBlock"] > div {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

# --- 2. CONFIGURATION ET CHARGEMENT DES MODÈLES ---
st.set_page_config(page_title="Universal Bridge AI", layout="wide", page_icon="🌍")
set_background("background.jpg")

# Initialisation des états de session
if 'chat_messages' not in st.session_state: st.session_state.chat_messages = []
if 'history' not in st.session_state: st.session_state.history = []
if 'detected_info' not in st.session_state: st.session_state.detected_info = None

@st.cache_resource
def load_essentials():
    # Modèle Traduction NLLB
    nllb_model_name = "facebook/nllb-200-distilled-600M"
    n_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
    n_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
    
    # OCR
    ocr_reader = easyocr.Reader(['fr', 'en', 'tr', 'es']) 
    
    # Chatbot Multilingue (Blenderbot)
    chat_model_name = "facebook/blenderbot-400M-distill"
    c_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
    c_model = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name)
    
    return n_tokenizer, n_model, ocr_reader, c_tokenizer, c_model

tokenizer, model, reader, chat_tokenizer, chat_model = load_essentials()

LANG_CODES = {"Français": "fra_Latn", "Anglais": "eng_Latn", "Turc": "tur_Latn", "Espagnol": "spa_Latn", "Chinois": "zho_Hans", "Coréen": "kor_Hang"}
VOICE_MAPPING = {
    "Français": {"Féminine": "fr-FR-DeniseNeural", "Masculine": "fr-FR-HenriNeural"},
    "Anglais": {"Féminine": "en-US-AriaNeural", "Masculine": "en-US-GuyNeural"},
    "Turc": {"Féminine": "tr-TR-EmelNeural", "Masculine": "tr-TR-AhmetNeural"},
    "Espagnol": {"Féminine": "es-ES-ElviraNeural", "Masculine": "es-ES-AlvaroNeural"},
    "Chinois": {"Féminine": "zh-CN-XiaoxiaoNeural", "Masculine": "zh-CN-YunxiNeural"},
    "Coréen": {"Féminine": "ko-KR-SunHiNeural", "Masculine": "ko-KR-InJoonNeural"}
}

async def generate_audio(text, voice_name, filename):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(filename)

# --- 3. BARRE LATÉRALE (Paramètres & Carte) ---
with st.sidebar:
    st.title("⚙️ Paramètres")
    lang_options = [f"{MAP_DATA[l]['flag']} {l}" for l in MAP_DATA.keys()]
    selected_lang_full = st.selectbox("🎯 Traduire vers", lang_options)
    target_lang = selected_lang_full.split(" ")[1]
    voice_type = st.radio("🗣️ Voix", ["Féminine", "Masculine"])
    
    if st.button("🗑️ Effacer le Chat"):
        st.session_state.chat_messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("📍 Localisation")
    
    m = folium.Map(location=[20, 0], zoom_start=1, tiles="CartoDB positron")
    
    # Marqueur Cible
    target_coords = MAP_DATA[target_lang]["coords"]
    folium.Marker(target_coords, popup=f"Cible: {target_lang}", icon=folium.Icon(color="blue")).add_to(m)

    # Marqueur Source avec sécurité Image/Emoji
    if st.session_state.detected_info:
        det = st.session_state.detected_info
        img_64 = get_base64(det["img"])
        if img_64:
            icon_html = f'''<div style="border: 2px solid #7C3AED; border-radius: 5px; background:white; width: 40px; height: 25px;">
                                <img src="data:image/png;base64,{img_64}" style="width: 100%; height: 100%; object-fit: cover;">
                            </div>'''
        else:
            icon_html = f'''<div style="font-size: 24px; text-align: center;">{det['flag']}</div>'''
            
        folium.Marker(
            location=det["coords"],
            icon=DivIcon(icon_size=(40, 25), icon_anchor=(20, 12), html=icon_html),
            popup=f"Origine: {det['name']}"
        ).add_to(m)
        m.location = det["coords"]
        
    st_folium(m, height=250, width=250, key="sidebar_map")

# --- 4. INTERFACE PRINCIPALE ---
st.write("# 🌐 Universal Bridge AI")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📥 Saisie & Chat")
    tabs = st.tabs(["✍️ Texte", "🖼️ OCR", "📄 Fichier", "🤖 Chatbot"])
    
    # Variable globale de saisie pour la traduction
    input_text = ""

    with tabs[0]:
        input_text = st.text_area("Saisissez votre texte :", height=150, key="main_text_input")

    with tabs[1]:
        img_file = st.file_uploader("Importer une image", type=['png', 'jpg', 'jpeg'])
        if img_file:
            img = Image.open(img_file)
            st.image(img, width=300)
            with st.spinner("Extraction OCR..."):
                results = reader.readtext(np.array(img))
                input_text = " ".join([res[1] for res in results])
                st.text_area("Texte extrait :", value=input_text, height=100)

    with tabs[2]:
        doc_file = st.file_uploader("Importer un document", type=['txt', 'docx', 'pdf'])
        if doc_file:
            file_ext = doc_file.name.split('.')[-1].lower()
            if file_ext == 'txt': input_text = doc_file.read().decode("utf-8")
            elif file_ext == 'docx':
                doc_obj = docx.Document(doc_file)
                input_text = "\n".join([p.text for p in doc_obj.paragraphs])
            elif file_ext == 'pdf':
                with fitz.open(stream=doc_file.read(), filetype="pdf") as pdf_doc:
                    input_text = "".join([p.get_text() for p in pdf_doc])
            st.success("Document chargé avec succès !")

    with tabs[3]:
        st.write("### 🤖 Assistant IA (Blenderbot)")
        chat_container = st.container(height=350)
        with chat_container:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Discutez avec l'IA..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Génération réponse Chat
            inputs = chat_tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                res_tokens = chat_model.generate(**inputs, max_length=100)
            response = chat_tokenizer.decode(res_tokens[0], skip_special_tokens=True)
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

    if st.button("🔍 Détecter la langue source"):
        if input_text.strip():
            iso_code = detect(input_text).split('-')[0]
            if iso_code in DETECTION_MAP:
                st.session_state.detected_info = DETECTION_MAP[iso_code]
                st.success(f"Langue : {st.session_state.detected_info['name']}")
                st.rerun()

with col2:
    st.subheader("📤 Résultat")
    if st.button("🚀 TRADUIRE"):
        if input_text.strip():
            target_code = LANG_CODES[target_lang]
            inputs = tokenizer(input_text, return_tensors="pt")
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code))
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            st.success(translation)
            
            # Synthèse Vocale
            voice = VOICE_MAPPING[target_lang][voice_type]
            asyncio.run(generate_audio(translation, voice, "output.mp3"))
            st.audio("output.mp3")
            
            # Historique
            st.session_state.history.append({"src": input_text[:30], "res": translation, "lang": target_lang})

# --- 5. HISTORIQUE ---
with st.expander("📜 Historique des traductions"):
    for item in reversed(st.session_state.history):
        st.write(f"**{item['lang']}**: {item['res']}")