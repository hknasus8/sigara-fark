# -*- coding: utf-8 -*-
"""
ÖZÇELİK STAND KONTROL UYGULAMASI (SADELEŞTİRİLMİŞ JSON KONTROLÜ)
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
# ANALİZ MOTORU (BÜTÜNCÜL RAF KONTROLÜ)
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

        # Rafın tamamını görselde belirginleştirmek için tek bir çerçeve çizelim
        bx1, by1 = max(0, int(bx)), max(0, int(by))
        bx2, by2 = min(w, int(bx + bw)), min(h, int(by + bh))
        
        raf_hatali_slot_sayisi = 0

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
            json_kelimeleri = [kw for kw in re.findall(r'[a-zğüşıöç]+', hedef_lower) if len(kw) > 2]

            if json_kelimeleri:
                uyumlu_mu = any(kw in sadece_harfler for kw in json_kelimeleri)
            else:
                uyumlu_mu = True

            if not uyumlu_mu and len(sadece_harfler) <= 2 and len(ham_metin) > 0:
                uyumlu_mu = True

            debug_rows.append({
                "Raf No": raf_no,
                "Slot": slot_sirasi,
                "Beklenen (JSON)": olmasi_gereken_urun,
                "Okunan Ham": ham_metin or "—",
                "Durum": "UYUMLU" if uyumlu_mu else "HATALI"
            })

            if not uyumlu_mu:
                toplam_fark += 1
                raf_hatali_slot_sayisi += 1

        # Eğer rafta hata varsa tüm raf kutusunu kırmızı yap, yoksa yeşil yap
        renk = (0, 0, 255) if raf_hatali_slot_sayisi > 0 else (0, 255, 0)
        cv2.rectangle(result_img, (bx1, by1), (bx2, by2), renk, 2)
        cv2.putText(result_img, f"Raf {raf_no} (Hata: {raf_hatali_slot_sayisi})", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2, cv2.LINE_AA)

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

st.info("🎯 **Bilgi:** JSON dosyanızı yükleyin, saha fotoğrafını seçin, ardından JSON'dan ilgili rafı seçerek fotoğraf üzerinde mouse ile kutu çizin ve kaydedin.")

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
st.subheader("3. JSON Rafı Seçin ve Mouse ile İşaretleyin")
if st.session_state.json_data and st.session_state.field_img is not None:
    raflar = st.session_state.json_data.get("raflar", [])
    raf_secenekleri = {r["raf_numarasi"]: f"Raf {r['raf_numarasi']} (Sıralama: {', '.join(r.get('urunler',[]))})" for r in raflar}
    
    selected_raf_no = st.selectbox("Kontrol Edilecek Rafı Seçin:", options=list(raf_secenekleri.keys()), format_func=lambda x: raf_secenekleri[x])

    st.markdown("👇 **Fotoğraf üzerinde farenizle seçtiğiniz rafa ait etiket alanını çizin:**")

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
        key=f"canvas_raf_{selected_raf_no}",
    )

    if st.button("💾 Çizilen Dikdörtgeni Seçili Rafa Kaydet", type="primary"):
        if canvas_result is not None and canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                latest_obj = objects[-1]
                box_data = {
                    "raf_numarasi": selected_raf_no,
                    "x": int(latest_obj["left"]),
                    "y": int(latest_obj["top"]),
                    "width": int(latest_obj["width"] * latest_obj.get("scaleX", 1.0)),
                    "height": int(latest_obj["height"] * latest_obj.get("scaleY", 1.0))
                }
                st.session_state.saved_raf_boxes = [b for b in st.session_state.saved_raf_boxes if b["raf_numarasi"] != selected_raf_no]
                st.session_state.saved_raf_boxes.append(box_data)
                st.success(f"✅ Raf {selected_raf_no} başarıyla kaydedildi!")
            else:
                st.warning("⚠️ Lütfen önce görsel üzerinde bir dikdörtgen çizin.")
        else:
            st.warning("⚠️ Çizim algılanamadı. Lütfen kutuyu tekrar çizip kaydedin.")

    if st.session_state.saved_raf_boxes:
        kayitli_raflar_str = ", ".join([str(b["raf_numarasi"]) for b in st.session_state.saved_raf_boxes])
        st.info(f"💡 Hafızaya kaydedilen raflar: [ Raf {kayitli_raflar_str} ]")

st.divider()

# Karşılaştırma Butonu (En az 1 raf kaydedildiğinde aktifleşir)
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
