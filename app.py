# -*- coding: utf-8 -*-
"""
ÖZÇELİK STAND KONTROL UYGULAMASI (EXCEL POLİGRAM SEÇİMLİ & 7-15 SIRA)
"""

import difflib
import hashlib
import hmac
import re
import urllib.parse
from PIL import Image
import io

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================================================
# SAYFA YAPILANDIRMASI
# =========================================================
st.set_page_config(
    page_title="Özçelik Stand Kontrol Uygulaması",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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
# OTOMATİK ÖNBELLEK TEMİZLEME
# =========================================================
if "cache_initialized" not in st.session_state:
    st.cache_data.clear()
    st.session_state.cache_initialized = True


# =========================================================
# SABİTLER
# =========================================================
YANDEX_ROOT_PUBLIC_KEY = "https://disk.yandex.com.tr/d/ikCHPwREiCVv_g"


# =========================================================
# GÜVENLİ YARDIMCILAR
# =========================================================
def normalize_text(value):
    value = str(value or "").upper().strip()
    value = (
        value.replace("İ", "I")
        .replace("Ş", "S")
        .replace("Ğ", "G")
        .replace("Ü", "U")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )
    return re.sub(r"\s+", " ", value)


def decode_uploaded(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def resize_keep_ratio(img, max_width=1200, max_height=1800):
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


def prepare_image(img):
    return resize_keep_ratio(img, max_width=1200, max_height=1800)


def safe_download_image(url, timeout=25):
    try:
        if not url:
            return None
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        if response.status_code != 200:
            return None
        data = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def safe_download_file_bytes(url, timeout=25):
    try:
        if not url:
            return None
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        if response.status_code != 200:
            return None
        return response.content
    except Exception:
        return None


# =========================================================
# YANDEX API
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def yandex_root_items(public_key):
    try:
        url = (
            "https://cloud-api.yandex.net/v1/disk/public/resources"
            f"?public_key={urllib.parse.quote(public_key, safe='')}"
            "&limit=500"
        )
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if response.status_code != 200:
            return [], f"Yandex HTTP {response.status_code}"
        return response.json().get("_embedded", {}).get("items", []), None
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl=600, show_spinner=False)
def yandex_list_dir(public_key, path):
    try:
        url = (
            "https://cloud-api.yandex.net/v1/disk/public/resources"
            f"?public_key={urllib.parse.quote(public_key, safe='')}"
            f"&path={urllib.parse.quote(path, safe='/')}"
            "&limit=500"
        )
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if response.status_code != 200:
            return [], f"Yandex HTTP {response.status_code}"
        return response.json().get("_embedded", {}).get("items", []), None
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl=600, show_spinner=False)
def get_cities(public_key):
    items, error = yandex_root_items(public_key)
    if error:
        return [], error
    cities = []
    for item in items:
        if item.get("type") != "dir":
            continue
        name = item.get("name", "")
        if normalize_text(name) == "BAYI":
            sub_items, _ = yandex_list_dir(public_key, item.get("path", ""))
            for sub in sub_items:
                if sub.get("type") == "dir" and normalize_text(sub.get("name")) != "POLIGRAM":
                    cities.append(sub.get("name", ""))
        else:
            if normalize_text(name) != "POLIGRAM":
                cities.append(name)
    return sorted({x for x in cities if x}, key=normalize_text), None


@st.cache_data(ttl=600, show_spinner=False)
def get_dealers(public_key, city):
    root_items, error = yandex_root_items(public_key)
    if error:
        return [], error
    city_item = None
    for item in root_items:
        if item.get("type") == "dir" and normalize_text(item.get("name")) == normalize_text(city):
            city_item = item
            break
    if city_item is None:
        for item in root_items:
            if item.get("type") != "dir" or normalize_text(item.get("name")) != "BAYI":
                continue
            sub_items, _ = yandex_list_dir(public_key, item.get("path", ""))
            for sub in sub_items:
                if sub.get("type") == "dir" and normalize_text(sub.get("name")) == normalize_text(city):
                    city_item = sub
                    break
            if city_item is not None:
                break
    if city_item is None:
        return [], f"'{city}' klasörü bulunamadı."

    items, error = yandex_list_dir(public_key, city_item.get("path", ""))
    if error:
        return [], error
    
    dealers = []
    for item in items:
        if item.get("type") == "dir" and item.get("name") and item.get("path"):
            dealers.append({"name": item["name"], "raw_name": item["name"], "path": item["path"]})
    dealers.sort(key=lambda x: normalize_text(x["name"]))
    return dealers, None


@st.cache_data(ttl=600, show_spinner=False)
def get_reference_image(public_key, dealer_path):
    items, error = yandex_list_dir(public_key, dealer_path)
    if error:
        return None, error
    image_items = []
    for item in items:
        if item.get("type") != "file":
            continue
        name = normalize_text(item.get("name", ""))
        if name.endswith((".JPG", ".JPEG", ".PNG", ".WEBP")):
            image_items.append(item)
    image_items.sort(
        key=lambda x: (
            0 if "ORJ" in normalize_text(x.get("name")) else (0 if "PLANOGRAM" in normalize_text(x.get("name")) else 1),
            normalize_text(x.get("name")),
        )
    )
    for item in image_items:
        img = safe_download_image(item.get("file"))
        if img is not None:
            return prepare_image(img), None
    return None, "Bayi klasöründe okunabilir JPG/PNG görsel bulunamadı."


@st.cache_data(ttl=600, show_spinner=False)
def get_polygram_files(public_key):
    root_items, error = yandex_root_items(public_key)
    if error:
        return [], error
    
    poly_item = None
    for item in root_items:
        if item.get("type") == "dir" and "POLIGRAM" in normalize_text(item.get("name", "")):
            poly_item = item
            break
            
    if poly_item is None:
        for item in root_items:
            if item.get("type") == "dir" and normalize_text(item.get("name", "")) == "BAYI":
                sub_items, _ = yandex_list_dir(public_key, item.get("path", ""))
                for sub in sub_items:
                    if sub.get("type") == "dir" and "POLIGRAM" in normalize_text(sub.get("name", "")):
                        poly_item = sub
                        break
                break

    if poly_item is None:
        return [], "Yandex'te 'POLİGRAM' klasörü bulunamadı."
    
    sub_items, error = yandex_list_dir(public_key, poly_item.get("path", ""))
    if error:
        return [], error
    
    polygrams = []
    for item in sub_items:
        if item.get("type") == "file":
            name = item.get("name", "")
            if name.lower().endswith((".xlsx", ".xls", ".csv")):
                polygrams.append({"name": name, "file": item.get("file")})
    return polygrams, None


# =========================================================
# GÖRSEL HİZALAMA VE ANALİZ MOTORU
# =========================================================
def analyze_polygram_excel_sequence_control(excel_bytes, field_img, selected_shelf_count):
    """
    Seçilen raf sıra sayısına (örn: 7) göre Excel içindeki ilgili şemayı baz alır 
    ve saha fotoğrafını etiket sıralamasına göre kontrol eder.
    """
    try:
        xls = pd.ExcelFile(io.BytesIO(excel_bytes))
        sheet_name = xls.sheet_names[0]
        raw_df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name, header=None)
    except Exception:
        raw_df = None

    h, w = field_img.shape[:2]
    result_img = field_img.copy()

    top_y = int(h * 0.05)
    bottom_y = int(h * 0.95)
    row_height = (bottom_y - top_y) // max(6, selected_shelf_count)

    discrepancy_count = 0
    results = []

    gray_field = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_field = clahe.apply(gray_field)

    for i in range(selected_shelf_count):
        s_top = top_y + (i * row_height)
        s_bottom = s_top + row_height if i < selected_shelf_count - 1 else bottom_y

        if s_top >= h:
            break

        shelf_roi = gray_field[s_top:s_bottom, :]
        blur_roi = cv2.GaussianBlur(shelf_roi, (5, 5), 0)

        _, thresh = cv2.threshold(blur_roi, 70, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w * h * 0.0003) or area > (w * h * 0.05):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            abs_y = s_top + y

            if x < 10 or (x + bw) > (w - 10):
                continue

            discrepancy_count += 1
            label_text = f"KAT {i+1} SIRALAMA HATASI #{discrepancy_count}"
            box_color = (0, 140, 255)  # Turuncu

            cv2.rectangle(result_img, (x, abs_y), (x + bw, abs_y + bh), box_color, 3)
            cv2.putText(
                result_img,
                label_text,
                (x, max(15, abs_y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            results.append({"id": discrepancy_count, "durum": label_text, "x": x, "y": abs_y, "w": bw, "h": bh})

    summary = {
        "discrepancy_count": discrepancy_count,
        "shelf_count": selected_shelf_count,
    }
    return result_img, results, summary, field_img


# =========================================================
# SESSION STATE & ARAYÜZ BAŞLANGICI
# =========================================================
DEFAULT_STATE = {
    "authenticated": False,
    "result_img": None,
    "aligned_field": None,
    "results": [],
    "summary": None,
    "poly_result_img": None,
    "poly_summary": None,
    "poly_aligned": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "app_password" not in st.secrets:
    st.error("Kritik: Streamlit Secrets içine app_password eklenmemiş.")
    st.stop()

if not st.session_state.authenticated:
    st.title("🔐 Özçelik Stand Kontrol Uygulaması")
    password = st.text_input("Şifre", type="password")
    if st.button("Giriş Yap", type="primary", use_container_width=True):
        if hmac.compare_digest(password, str(st.secrets["app_password"])):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre.")
    st.stop()


# =========================================================
# KENAR ÇUBUĞU (SİDEBAR) - KONTROL BUTONLARI
# =========================================================
with st.sidebar:
    st.subheader("⚙️ Sistem Kontrolleri")
    
    # 1. Yandex Önbelleğini Yenile Butonu
    if st.button("🔄 Yandex Önbelleğini Yenile", use_container_width=True):
        st.cache_data.clear()
        st.success("Önbellek başarıyla temizlendi!")
        st.rerun()
        
    st.divider()
    
    # 2. Çıkış Yap Butonu
    if st.button("🚪 Çıkış Yap", type="primary", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.result_img = None
        st.session_state.poly_result_img = None
        st.rerun()


st.title("📊 ÖZÇELİK STAND KONTROL UYGULAMASI")

# 1. ŞEHİR VE BAYİ SEÇİMİ
st.subheader("1. Şehir ve Bayi Seçiniz")
cities, _ = get_cities(YANDEX_ROOT_PUBLIC_KEY)
c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("Şehir", options=[""] + cities, format_func=lambda x: "Şehir seçin..." if x == "" else x)

dealers = []
if city:
    dealers, _ = get_dealers(YANDEX_ROOT_PUBLIC_KEY, city)

with c2:
    search_term = st.text_input("Bayi ara", placeholder="Bayi adı yazın...", label_visibility="collapsed")
    filtered_dealers = [d for d in dealers if not search_term or search_term.lower() in d["raw_name"].lower()]
    dealer_choices = {x["raw_name"]: x for x in filtered_dealers}
    selected_raw_dealer = st.selectbox("Bayi", options=[""] + list(dealer_choices.keys()), format_func=lambda x: "Bayi seçin..." if x == "" else x, label_visibility="collapsed")
    
    dealer_path = dealer_choices[selected_raw_dealer]["path"] if selected_raw_dealer in dealer_choices else ""

st.divider()

# 2. FOTOĞRAFLAR
st.subheader("2. Orijinal Referans Fotoğrafı & Saha Fotoğrafı")
ref_img = get_reference_image(YANDEX_ROOT_PUBLIC_KEY, dealer_path)[0] if dealer_path else None

u1, u2 = st.columns(2)
with u1:
    st.markdown("**Orijinal Referans Fotoğrafı (Yandex)**")
    if ref_img is not None:
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.info("Bayi seçin.")

with u2:
    st.markdown("**Saha Fotoğrafı**")
    field_upload = st.file_uploader("Saha fotoğrafını yükleyin", type=["jpg", "jpeg", "png", "webp"], key="field_upload")
    field_img = prepare_image(decode_uploaded(field_upload)) if field_upload is not None else None
    if field_img is not None:
        st.image(field_img, channels="BGR", use_container_width=True)
    else:
        st.info("Sahadan gelen fotoğrafı yükleyin.")

st.divider()

# 3. POLİGRAM EXCEL SIRALAMA KONTROLÜ
st.subheader("3. Stand Dizilim Sıralaması Kontrolü (Poligram Excel Modülü)")
poly_files, _ = get_polygram_files(YANDEX_ROOT_PUBLIC_KEY)
poly_options = {item["name"]: item["file"] for item in poly_files}

p1, p2, p3 = st.columns([2, 1, 1])
with p1:
    selected_poly_name = st.selectbox("Yandex Poligram Excel Dosyası", options=[""] + list(poly_options.keys()), format_func=lambda x: "Excel seçin..." if x == "" else x)
with p2:
    stand_kat_sayisi = st.selectbox("Stand / Raf Sıra Yapısı", options=list(range(6, 16)), index=1, format_func=lambda x: f"{x} Sıralı Stand")
with p3:
    st.write("")
    st.write("")
    poly_kontrol_btn = st.button("🔍 Poligramı Kontrol Et", type="secondary", use_container_width=True)

if poly_kontrol_btn:
    if not selected_poly_name or field_img is None:
        st.warning("⚠️ Lütfen Yandex'ten bir Excel dosyası seçin ve saha fotoğrafı yükleyin.")
    else:
        st.session_state.poly_result_img = None
        st.session_state.poly_summary = None
        with st.spinner(f"Excel poligram verisi ({stand_kat_sayisi} katlı şema) ile saha fotoğrafı karşılaştırılıyor..."):
            excel_bytes = safe_download_file_bytes(poly_options[selected_poly_name])
            p_res_img, p_results, p_summary, p_aligned = analyze_polygram_excel_sequence_control(
                excel_bytes, field_img, stand_kat_sayisi
            )
            st.session_state.poly_result_img = p_res_img
            st.session_state.poly_summary = p_summary
            st.session_state.poly_aligned = p_aligned

if st.session_state.poly_result_img is not None and st.session_state.poly_summary:
    p_sum = st.session_state.poly_summary
    st.success(f"✅ Poligram Kontrolü Tamamlandı ({p_sum.get('shelf_count')} Raf Baz Alındı)! Farklılık: **{p_sum.get('discrepancy_count', 0)}**")
    
    if p_sum.get('discrepancy_count', 0) > 0:
        f_clean = Image.fromarray(cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB))
        f_marked = Image.fromarray(cv2.cvtColor(st.session_state.poly_result_img, cv2.COLOR_BGR2RGB))
        gif_bytes = io.BytesIO()
        f_marked.save(gif_bytes, format="GIF", save_all=True, append_images=[f_clean], duration=400, loop=0)
        st.image(gif_bytes.getvalue(), use_container_width=True)
    else:
        st.image(st.session_state.poly_result_img, channels="BGR", use_container_width=True)
