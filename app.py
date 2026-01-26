import streamlit as st
import asyncio
import base64
import os
import numpy as np
import folium
import docx
import fitz  # PyMuPDF
import easyocr
import edge_tts
from PIL import Image
from langdetect import detect
from streamlit_folium import st_folium
from folium.features import DivIcon
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from streamlit_mic_recorder import mic_recorder

# --- 1. CONFIGURATION (IMPÉRATIVEMENT EN PREMIER) ---
st.set_page_config(page_title="Universal Bridge AI", layout="wide", page_icon="🌍")

# --- 2. DONNÉES DE RÉFÉRENCE ---
MAP_DATA = {
    "Français": {"coords": [46.2276, 2.2137], "flag": "🇫🇷", "iso": "fr", "nllb": "fra_Latn", "icon": "france.png"},
    "Anglais": {"coords": [37.0902, -95.7129], "flag": "🇺🇸", "iso": "en", "nllb": "eng_Latn", "icon": "royaume-uni.png"},
    "Turc": {"coords": [38.9637, 35.2433], "flag": "🇹🇷", "iso": "tr", "nllb": "tur_Latn", "icon": "dinde.png"},
    "Espagnol": {"coords": [40.4637, -3.7492], "flag": "🇪🇸", "iso": "es", "nllb": "spa_Latn", "icon": "drapeau.png"},
    "Chinois": {"coords": [35.8617, 104.1954], "flag": "🇨🇳", "iso": "zh", "nllb": "zho_Hans", "icon": "chine.png"},
    "Coréen": {"coords": [35.9078, 127.7669], "flag": "🇰🇷", "iso": "ko", "nllb": "kor_Hang", "icon": "coree-du-sud.png"}
}

DETECTION_MAP = {v["iso"]: {"coords": v["coords"], "flag": v["flag"], "name": k, "icon": v["icon"]} for k, v in MAP_DATA.items()}

VOICE_MAPPING = {
    "Français": {"Féminine": "fr-FR-DeniseNeural", "Masculine": "fr-FR-HenriNeural"},
    "Anglais": {"Féminine": "en-US-AriaNeural", "Masculine": "en-US-GuyNeural"},
    "Turc": {"Féminine": "tr-TR-EmelNeural", "Masculine": "tr-TR-AhmetNeural"},
    "Espagnol": {"Féminine": "es-ES-ElviraNeural", "Masculine": "es-ES-AlvaroNeural"},
    "Chinois": {"Féminine": "zh-CN-XiaoxiaoNeural", "Masculine": "zh-CN-YunxiNeural"},
    "Coréen": {"Féminine": "ko-KR-SunHiNeural", "Masculine": "ko-KR-InJoonNeural"}
}

# --- 3. FONCTIONS TECHNIQUES ---
def get_base64_img(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def apply_design(bg_file):
    bg_base = get_base64_img(bg_file)
    if bg_base:
        st.markdown(f"""
        <style>
        .stApp {{ background-image: url("data:image/png;base64,{bg_base}"); background-size: cover; background-attachment: fixed; }}
        [data-testid="stVerticalBlock"] > div {{ background-color: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        .stButton>button {{ background: linear-gradient(45deg, #1E3A8A, #7C3AED); color: white; border-radius: 25px; border: none; width: 100%; transition: 0.3s; }}
        .stButton>button:hover {{ transform: scale(1.02); box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4); }}
        </style>
        """, unsafe_allow_html=True)

@st.cache_resource
def load_ai_engine():
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    mod = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    ocr = easyocr.Reader(['fr', 'en', 'es', 'tr'])
    return tok, mod, ocr

async def generate_voice(text, voice_name, file_name):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(file_name)

# --- 4. INITIALISATION ---
apply_design("background.jpg")
tokenizer, nllb_model, ocr_reader = load_ai_engine()

if 'history' not in st.session_state: st.session_state.history = []
if 'detected_info' not in st.session_state: st.session_state.detected_info = None

# --- 5. BARRE LATÉRALE ---
with st.sidebar:
    st.title("⚙️ Paramètres")
    target_lang = st.selectbox("🎯 Vers quelle langue ?", list(MAP_DATA.keys()))
    voice_choice = st.radio("🗣️ Genre de la voix", ["Féminine", "Masculine"])
    
    st.divider()
    st.subheader("📍 Suivi Géographique")
    
    m = folium.Map(location=[20, 0], zoom_start=1, tiles="CartoDB positron")
    folium.Marker(MAP_DATA[target_lang]["coords"], popup=f"Cible: {target_lang}", icon=folium.Icon(color="blue")).add_to(m)
    
    if st.session_state.detected_info:
        info = st.session_state.detected_info
        img_b64 = get_base64_img(info["icon"])
        if img_b64:
            html = f'<img src="data:image/png;base64,{img_b64}" style="width:40px;height:40px;border-radius:50%;border:2px solid #7C3AED;box-shadow:0 0 8px rgba(0,0,0,0.4);">'
            folium.Marker(location=info["coords"], icon=DivIcon(html=html)).add_to(m)
            m.location = info["coords"]
            m.zoom_start = 3

    st_folium(m, height=250, width=250, key="sidebar_map")
    st.info("🔗 Pour partager l'app : déploie-la sur Streamlit Cloud via GitHub.")

# --- 6. ZONE PRINCIPALE ---
st.title("🌐 Universal Bridge AI")
st.caption("Système intelligent : Texte | Image | Document | Voix")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("📥 Saisie")
    tabs = st.tabs(["✍️ Texte", "🎙️ Vocal", "🖼️ OCR", "📄 Fichier"])
    final_text = ""

    with tabs[0]:
        final_text = st.text_area("Tapez ici :", height=150, placeholder="Ex: Bonjour, comment allez-vous ?", key="text_in")

    with tabs[1]:
        st.write("🎤 Cliquez pour enregistrer :")
        audio_rec = mic_recorder(start_prompt="Démarrer", stop_prompt="Arrêter", key='recorder')
        if audio_rec: st.info("Audio reçu. Transcription en attente (API requise).")

    with tabs[2]:
        img_up = st.file_uploader("Téléverser image", type=['png', 'jpg', 'jpeg'])
        if img_up:
            with st.spinner("Lecture de l'image..."):
                res_ocr = ocr_reader.readtext(np.array(Image.open(img_up)))
                final_text = " ".join([r[1] for r in res_ocr])
                st.info(f"Texte extrait : {final_text}")

    with tabs[3]:
        doc_up = st.file_uploader("Document (PDF/DOCX/TXT)", type=['txt', 'docx', 'pdf'])
        if doc_up:
            ext = doc_up.name.split('.')[-1].lower()
            if ext == 'txt': final_text = doc_up.read().decode()
            elif ext == 'docx': final_text = "\n".join([p.text for p in docx.Document(doc_up).paragraphs])
            elif ext == 'pdf':
                with fitz.open(stream=doc_up.read(), filetype="pdf") as pdf:
                    final_text = "".join([page.get_text() for page in pdf])

with col_right:
    st.subheader("📤 Résultat")
    if st.button("🚀 TRADUIRE ET LIRE"):
        if final_text.strip():
            try:
                # 1. Détection Auto
                lang_iso = detect(final_text)
                st.session_state.detected_info = DETECTION_MAP.get(lang_iso, None)
                
                # 2. Traduction
                nllb_code = MAP_DATA[target_lang]["nllb"]
                inputs = tokenizer(final_text, return_tensors="pt")
                translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_code))
                result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
                st.success(f"**Traduction ({target_lang}) :**\n\n{result}")

                # 3. Lecture Bilingue
                st.divider()
                au1, au2 = st.columns(2)
                with au1:
                    st.caption("Lecture Originale")
                    # Détermine la voix source
                    src_name = st.session_state.detected_info['name'] if st.session_state.detected_info else "Anglais"
                    v_orig = VOICE_MAPPING.get(src_name, VOICE_MAPPING["Anglais"])[voice_choice]
                    asyncio.run(generate_voice(final_text, v_orig, "src.mp3"))
                    st.audio("src.mp3")
                with au2:
                    st.caption(f"Lecture {target_lang}")
                    v_dest = VOICE_MAPPING[target_lang][voice_choice]
                    asyncio.run(generate_voice(result, v_dest, "dest.mp3"))
                    st.audio("dest.mp3")
                
                st.session_state.history.append({"src": final_text[:30], "res": result, "l": target_lang})
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

# --- 7. HISTORIQUE ---
st.divider()
with st.expander("📜 Voir l'historique"):
    for h in reversed(st.session_state.history):
        st.write(f"**Vers {h['l']}** : {h['res']} *(Source: {h['src']}...)*")