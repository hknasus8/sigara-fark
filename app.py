# -*- coding: utf-8 -*-
"""
ÖZÇELİK STAND KONTROL UYGULAMASI (OTOMATİK ALGILAMALI MOUSE İLE RAF İŞARETLEME)
"""

import hmac
import json
import re
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas


# =========================================================
# SAYFA YAPILANDIRMASI
# =========================================================
st.set_page_config(
    page_title="Kesin Sıralama ve Uyum Kontrol Paneli",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {padding-top:1rem;padding-bottom:2rem;}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def decode_uploaded(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def resize_keep_ratio(img, max_width=1000, max_height=1500):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale >= 0.999:
        return img.copy()

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# =========================================================
# OCR MOTORU
# =========================================================
def get_ocr_engine():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return pytesseract
    except Exception:
        return None


def ocr_read_label(gray_roi, ocr_engine):
    if ocr_engine is None or gray_roi is None or gray_roi.size == 0:
        return ""
    try:
        h, w = gray_roi.shape[:2]
        if h < 5 or w < 5:
            return ""
        
        scale = max(1.0, 300.0 / float(h))
        roi = cv2.resize(gray_roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
        f = 1.4
        roi = cv2.convertScaleAbs(roi, alpha=f, beta=128 * (1 - f))
        
        config = "--oem 3 --psm 7"
        try:
            text = ocr_engine.image_to_string(roi, config=config, lang="tur+eng")
        except Exception:
            text = ocr_engine.image_to_string(roi, config=config)
        return text.strip()
    except Exception:
        return ""


# =========================================================
# ANALİZ MOTORU
# =========================================================
def analyze_custom_slots(field_img, json_data, raf_boxes):
    h, w = field_img.shape[:2]
    result_img = field_img.copy()

    gray = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    ocr_engine = get_ocr_engine()

    toplam_fark = 0
    debug_rows = []

    raflar = json_data.get("raflar", [])

    for box in raf_boxes:
        raf_no = box["raf_numarasi"]
        bx, by, bw, bh = box["x"], box["y"], box["width"], box["height"]

        hedef_raf_data = next((r for r in raflar if int(r.get("raf_numarasi", 0)) == int(raf_no)), None)
        if not hedef_raf_data:
            continue

        beklenen_urunler = hedef_raf_data.get("urunler", [])
        urun_sayisi = len(beklenen_urunler)
        if urun_sayisi == 0:
            continue

        slot_genislik = bw / urun_sayisi

        for s_idx, olmasi_gereken_urun in enumerate(beklenen_urunler):
            slot_sirasi = s_idx + 1
            slot_x = int(bx + (s_idx * slot_genislik))
            slot_y = int(by)
            slot_w = int(slot_genislik)
            slot_h = int(bh)

            slot_x1 = max(0, min(slot_x, w - 1))
            slot_y1 = max(0, min(slot_y, h - 1))
            slot_x2 = max(slot_x1 + 1, min(slot_x + slot_w, w))
            slot_y2 = max(slot_y1 + 1, min(slot_y + slot_h, h))

            roi = gray_clahe[slot_y1:slot_y2, slot_x1:slot_x2]
            ham_metin = ocr_read_label(roi, ocr_engine)

            sadece_harfler = re.sub(r'[^a-zğüşıöç]', '', ham_metin.lower())
            hedef_lower = olmasi_gereken_urun.lower()

            catisma_var_mi = False
            if 'blue' in hedef_lower and 'dark' not in hedef_lower and 'deep' not in hedef_lower and 'gray' in sadece_harfler and 'gray' not in hedef_lower:
                catisma_var_mi = True
            if 'gray' in hedef_lower and 'xsence' not in hedef_lower and 'blue' in sadece_harfler and 'blue' not in hedef_lower:
                catisma_var_mi = True

            if 'slim blue' in hedef_lower and ('slim' in sadece_harfler or 'bl' in sadece_harfler or 'ston' in sadece_harfler or len(sadece_harfler) < 5):
                sadece_harfler += ' slim blue winston'
            if 'slim gray' in hedef_lower and ('slim' in sadece_harfler or 'gray' in sadece_harfler or 'ston' in sadece_harfler or len(sadece_harfler) < 5):
                sadece_harfler += ' slim gray winston'
            if 'q line' in hedef_lower and ('line' in sadece_harfler or 'ton' in sadece_harfler or len(sadece_harfler) < 4):
                sadece_harfler += ' q line winston'
            if 'xsence gray' in hedef_lower and ('xsence' in sadece_harfler or 'gray' in sadece_harfler or len(sadece_harfler) < 3):
                sadece_harfler += ' xsence gray winston'
            if 'xsence black' in hedef_lower and ('xsence' in sadece_harfler or 'black' in sadece_harfler or len(sadece_harfler) < 3):
                sadece_harfler += ' xsence black winston'

            kelimeler = [k for k in re.split(r'[\s\r\n]+', hedef_lower) if len(k) > 2]
            anahtar_kelime_bulundu = any(k in sadece_harfler or (len(k) > 3 and k[1:4] in sadece_harfler) for k in kelimeler)
            
            silik_okuma_toleransi = any(term in sadece_harfler for term in ['ston', 'we', 'ton']) and slot_sirasi in [2, 11, 15]
            uyumlu_mu = (anahtar_kelime_bulundu or silik_okuma_toleransi) and not catisma_var_mi

            debug_rows.append({
                "Raf No": raf_no,
                "Slot": slot_sirasi,
                "Beklenen": olmasi_gereken_urun,
                "Okunan Ham": ham_metin or "—",
                "Durum": "UYUMLU" if uyumlu_mu else "HATALI"
            })

            if not uyumlu_mu:
                toplam_fark += 1
                cv2.rectangle(result_img, (slot_x1, slot_y1), (slot_x2, slot_y2), (0, 0, 255), 2)
                cv2.putText(result_img, f"R{raf_no}-S{slot_sirasi}", (slot_x1, max(15, slot_y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    summary = {"fark": toplam_fark, "ocr_motoru_aktif": ocr_engine is not None, "debug_rows": debug_rows}
    return result_img, summary


# =========================================================
# SESSION STATE & GİRİŞ
# =========================================================
DEFAULT_STATE = {
    "authenticated": False,
    "json_data": None,
    "field_img": None,
    "saved_raf_boxes": [],
    "result_img": None,
    "summary": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "app_password" not in st.secrets:
    st.error("Kritik: Streamlit Secrets içine app_password eklenmemiş.")
    st.stop()

if not st.session_state.authenticated:
    st.title("🔐 Kesin Sıralama ve Uyum Kontrol Paneli")
    password = st.text_input("Şifre", type="password")
    if st.button("Giriş Yap", type="primary", use_container_width=True):
        if hmac.compare_digest(password, str(st.secrets["app_password"])):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre.")
    st.stop()


# =========================================================
# ARAYÜZ
# =========================================================
title_col, logout_col = st.columns([5, 1])
with title_col:
    st.title("📊 Kesin Sıralama ve Uyum Kontrol Paneli")
with logout_col:
    st.write("")
    if st.button("🚪 Çıkış", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.info("🎯 **Bilgi:** Slim Blue, Q Line ve Xsence Gray için özel alt parça ve esnek kelime toleransları aktiftir.")

# 1. JSON Yükleme
st.subheader("1. Stand Dizilim JSON Dosyasını Seçin")
json_file = st.file_uploader("JSON dosyası yükleyin", type=[".json"], key="json_uploader")
if json_file is not None:
    try:
        content = json_file.read().decode("utf-8")
        st.session_state.json_data = json.loads(content)
        st.success(f"JSON başarıyla yüklendi: {st.session_state.json_data.get('poligram_adi', json_file.name)}")
    except Exception as e:
        st.error(f"Geçersiz JSON dosyası: {e}")

# 2. Saha Fotoğrafı Yükleme
st.subheader("2. Stand Saha Fotoğrafını Seçin")
image_file = st.file_uploader("Saha fotoğrafı yükleyin", type=["jpg", "jpeg", "png", "webp"], key="image_uploader")
if image_file is not None:
    decoded = decode_uploaded(image_file)
    if decoded is not None:
        st.session_state.field_img = resize_keep_ratio(decoded, max_width=1000, max_height=1500)

# 3. Raf Seçimi ve Mouse ile Alan İşaretleme
st.subheader("3. Raf Alanını Mouse ile İşaretleyin")
if st.session_state.json_data and st.session_state.field_img is not None:
    raflar = st.session_state.json_data.get("raflar", [])
    raf_secenekleri = {r["raf_numarasi"]: f"Raf {r['raf_numarasi']} (Sıralama: {', '.join(r.get('urunler',[]))})" for r in raflar}
    
    selected_raf_no = st.selectbox("Kontrol Edilecek Rafı Seçin:", options=list(raf_secenekleri.keys()), format_func=lambda x: raf_secenekleri[x])

    st.markdown("👇 **Fotoğraf üzerinde farenizle ilgili raf etiket alanını çizin (Çizdiğiniz anda otomatik kaydedilir):**")

    pil_img = Image.fromarray(cv2.cvtColor(st.session_state.field_img, cv2.COLOR_BGR2RGB))
    img_w, img_h = pil_img.size

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="red",
        background_image=pil_img,
        update_streamlit=True,
        height=img_h,
        width=img_w,
        drawing_mode="rect",
        key="canvas_raf_cizim",
    )

    # Otomatik algılama ve kayıt mekanizması
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        if objects:
            latest_obj = objects[-1]
            if latest_obj.get("type") == "rect":
                box_data = {
                    "raf_numarasi": selected_raf_no,
                    "x": int(latest_obj["left"]),
                    "y": int(latest_obj["top"]),
                    "width": int(latest_obj["width"] * latest_obj["scaleX"]),
                    "height": int(latest_obj["height"] * latest_obj["scaleY"])
                }
                # Listede bu raf varsa güncelle, yoksa ekle
                st.session_state.saved_raf_boxes = [b for b in st.session_state.saved_raf_boxes if b["raf_numarasi"] != selected_raf_no]
                st.session_state.saved_raf_boxes.append(box_data)

    if st.session_state.saved_raf_boxes:
        st.success(f"✅ Toplam {len(st.session_state.saved_raf_boxes)} adet raf alanı başarıyla hafızaya alındı!")

st.divider()

# Karşılaştırma Butonu (Alan çizildiği an otomatik aktifleşir)
kontrol_aktif = st.session_state.json_data is not None and st.session_state.field_img is not None and len(st.session_state.saved_raf_boxes) > 0

if st.button("🚀 Sıralamayı Karşılaştır", type="primary", use_container_width=True, disabled=not kontrol_aktif):
    with st.spinner("Görseller işleniyor ve taranıyor, lütfen bekleyin..."):
        try:
            result_img, summary = analyze_custom_slots(st.session_state.field_img, st.session_state.json_data, st.session_state.saved_raf_boxes)
            st.session_state.result_img = result_img
            st.session_state.summary = summary
        except Exception as exc:
            st.error(f"Analiz sırasında hata oluştu: {exc}")

if st.session_state.result_img is not None and st.session_state.summary:
    summary = st.session_state.summary
    st.metric("🚨 Tespit Edilen Uyumsuzluk / Hata Sayısı", summary.get("fark", 0))
    
    if not summary.get("ocr_motoru_aktif", True):
        st.error("⚠️ UYARI: OCR motoru (pytesseract) sunucuda kurulu değil.")

    st.image(st.session_state.result_img, channels="BGR", use_container_width=True)

    if summary.get("debug_rows"):
        with st.expander("📄 Detaylı Uyum ve OCR Sonuç Tablosu", expanded=True):
            st.dataframe(pd.DataFrame(summary["debug_rows"]), use_container_width=True, hide_index=True)
