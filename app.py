# -*- coding: utf-8 -*-
"""
ÖZÇELİK STAND KONTROL UYGULAMASI (ETİKET & UYUM ENTEGRELI)
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
import requests
import streamlit as st


# =========================================================
# SAYFA YAPILANDIRMASI
# =========================================================
st.set_page_config(
    page_title="Özçelik Stand Kontrol Uygulaması",
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
# OTOMATİK ÖNBELLEK TEMİZLEME (UYGULAMA BAŞLANGICI)
# =========================================================
if "cache_initialized" not in st.session_state:
    st.cache_data.clear()
    st.session_state.cache_initialized = True


# =========================================================
# SABİTLER
# =========================================================
YANDEX_ROOT_PUBLIC_KEY = "https://disk.yandex.com.tr/d/ikCHPwREiCVv_g"
RAF_SAYISI = 6  


# =========================================================
# GÜVENLİ YARDIMCILAR
# =========================================================
def safe_float(value, default=0.0):
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


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
                if sub.get("type") == "dir":
                    cities.append(sub.get("name", ""))
        else:
            cities.append(name)
    cities = sorted({x for x in cities if x}, key=normalize_text)
    return cities, None


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
            raw_name = item["name"]
            dealers.append({"name": raw_name, "raw_name": raw_name, "path": item["path"]})
            
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


# =========================================================
# GÖRSEL HİZALAMA VE POG / İHLAL & ETİKET ANALİZ MOTORU
# =========================================================
def gray_normalize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def align_images_feature(reference, target):
    h, w = reference.shape[:2]
    if target.shape[:2] != (w, h):
        target = cv2.resize(target, (w, h), interpolation=cv2.INTER_AREA)
    
    ref_gray = gray_normalize(reference)
    tar_gray = gray_normalize(target)

    orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.15, nlevels=8, edgeThreshold=10, fastThreshold=7)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(tar_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return target, False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des2, des1, k=2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.80 * n.distance:
            good.append(m)

    if len(good) < 10:
        return target, False

    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if matrix is None:
        return target, False

    aligned = cv2.warpPerspective(target, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned, True


def analyze_planogram_grid_free(reference, field, roi_top_ratio=0.05, roi_bottom_ratio=0.82):
    h, w = reference.shape[:2]
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_AREA)

    aligned, aligned_ok = align_images_feature(reference, field)
    result_img = aligned.copy()
    
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    tar_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_gray = clahe.apply(ref_gray)
    tar_gray = clahe.apply(tar_gray)

    top_y = int(h * roi_top_ratio)
    bottom_y = int(h * roi_bottom_ratio)
    
    total_estimated_shelves = 9  
    shelf_height = (bottom_y - top_y) // 6  

    results = []
    fark_sayisi = 0
    kontrol_edilmeyen_rakip_raf = 0
    missing_label_count = 0 
    product_label_mismatch_count = 0  

    for i in range(total_estimated_shelves):
        s_top = top_y + (i * shelf_height)
        s_bottom = s_top + shelf_height if i < total_estimated_shelves - 1 else bottom_y

        if s_top >= h:
            break

        if i < 6:
            ref_roi = ref_gray[s_top:s_bottom, :]
            tar_roi = tar_gray[s_top:s_bottom, :]

            ref_roi_blur = cv2.GaussianBlur(ref_roi, (5, 5), 0)
            tar_roi_blur = cv2.GaussianBlur(tar_roi, (5, 5), 0)

            diff = cv2.absdiff(ref_roi_blur, tar_roi_blur)
            _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (w * h * 0.0002) or area > (w * h * 0.08):
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                abs_y = s_top + y

                if x < 10 or (x + bw) > (w - 10):
                    continue
                
                roi_target_piece = tar_roi[y:y+bh, x:x+bw]
                mean_brightness = np.mean(roi_target_piece) if roi_target_piece.size > 0 else 128
                ref_piece = ref_roi[y:y+bh, x:x+bw] if (y+bh <= ref_roi.shape[0] and x+bw <= ref_roi.shape[1]) else None
                
                if mean_brightness > 190: 
                    missing_label_count += 1
                    fark_sayisi += 1
                    etiket_turu = f"EKSİK ETİKET #{missing_label_count}"
                    box_color = (0, 165, 255) # Turuncu
                elif ref_piece is not None and np.mean(np.abs(ref_piece.astype(np.float32) - roi_target_piece.astype(np.float32))) > 40:
                    product_label_mismatch_count += 1
                    fark_sayisi += 1
                    etiket_turu = f"FARKLILIK #{product_label_mismatch_count}"
                    box_color = (255, 0, 255) # Mor/Pembe
                else:
                    continue

                box_thickness = 3

                cv2.rectangle(result_img, (x, abs_y), (x + bw, abs_y + bh), box_color, box_thickness)
                cv2.putText(
                    result_img,
                    etiket_turu,
                    (x, max(15, abs_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                results.append({"id": fark_sayisi, "durum": etiket_turu, "x": x, "y": abs_y, "w": bw, "h": bh, "alan": area})

        else:
            tar_roi = tar_gray[s_top:s_bottom, :]
            tar_roi_blur = cv2.GaussianBlur(tar_roi, (5, 5), 0)
            
            _, thresh = cv2.threshold(tar_roi_blur, 100, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (w * h * 0.001):
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                abs_y = s_top + y
                
                if x < 10 or (x + bw) > (w - 10) or bw < 20 or bh < 20:
                    continue

                kontrol_edilmeyen_rakip_raf += 1
                etiket_turu = f"KONTROL EDİLMEYEN RAF #{kontrol_edilmeyen_rakip_raf}"
                
                cv2.line(result_img, (x, abs_y), (x + bw, abs_y + bh), (255, 255, 255), 3)
                cv2.line(result_img, (x, abs_y + bh), (x + bw, abs_y), (255, 255, 255), 3)
                cv2.rectangle(result_img, (x, abs_y), (x + bw, abs_y + bh), (255, 255, 255), 2)
                
                cv2.putText(
                    result_img,
                    etiket_turu,
                    (x, max(15, abs_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                
                results.append({"id": f"X_{kontrol_edilmeyen_rakip_raf}", "durum": etiket_turu, "x": x, "y": abs_y, "w": bw, "h": bh, "alan": area})

    summary = {
        "fark": fark_sayisi,
        "etiket_eksigi": missing_label_count,
        "urun_etiket_uyumsuzluk": product_label_mismatch_count,
        "kontrol_edilmeyen_rakip_raf": kontrol_edilmeyen_rakip_raf,
        "hizalama_ok": aligned_ok,
        "hizalama": "Hibrit Motor",
    }

    return result_img, results, summary, aligned


def build_report(dealer, results, summary):
    from datetime import datetime
    lines = [
        "=== ÖZÇELİK STAND DENETİM RAPORU ===",
        f"Bayi: {dealer}",
        "Tarih: " + datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "",
        "Eksik Etiket Sayısı: " + str(summary.get('etiket_eksigi', 0)),
        "Farklılık Sayısı: " + str(summary.get('urun_etiket_uyumsuzluk', 0)),
        "Kontrol Edilmeyen Rakip Raf Sayısı: " + str(summary.get('kontrol_edilmeyen_rakip_raf', 0)),
        "",
        "--- DETAYLI İHLAL / EKSİK KAYITLARI ---"
    ]
    for item in results:
        lines.append(f"ID #{item.get('id')} ({item.get('durum')}) | Konum: X={item.get('x')}, Y={item.get('y')} | Boyut: {item.get('w')}x{item.get('h')}")
    return "\n".join(lines)


# =========================================================
# SESSION STATE & GİRİŞ
# =========================================================
DEFAULT_STATE = {
    "authenticated": False,
    "result_img": None,
    "aligned_field": None,
    "results": [],
    "summary": None,
    "report": "",
    "analysis_key": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "app_password" not in st.secrets:
    st.error("Kritik: Streamlit Secrets içine app_password eklenmemiş.")
    st.stop()

if not st.session_state.authenticated:
    st.title("🔐 Özçelik Stand Kontrol Uygulaması")
    try:
        st.image(Image.open("logo.jpg"), width=250)
    except Exception:
        pass
    password = st.text_input("Şifre", type="password")
    if st.button("Giriş Yap", type="primary", use_container_width=True):
        if hmac.compare_digest(password, str(st.secrets["app_password"])):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre.")
    st.stop()


# =========================================================
# ARAYÜZ (SOL MENÜ TAMAMEN KAPATILDI)
# =========================================================
def clear_yandex_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state.result_img = None
    st.session_state.aligned_field = None
    st.session_state.results = []
    st.session_state.summary = None
    st.session_state.report = ""


title_col, refresh_col, logout_col = st.columns([4, 1, 1])
with title_col:
    st.title("📊 ÖZÇELİK STAND KONTROL UYGULAMASI")
with refresh_col:
    st.write("")
    if st.button("🔄 Yenile", use_container_width=True, key="main_refresh_btn"):
        clear_yandex_cache()
        st.rerun()
with logout_col:
    st.write("")
    if st.button("🚪 Çıkış", use_container_width=True, key="main_logout_btn"):
        st.session_state.authenticated = False
        st.session_state.result_img = None
        st.session_state.aligned_field = None
        st.session_state.results = []
        st.session_state.summary = None
        st.rerun()

st.subheader("1. Şehir ve Bayi Seçiniz")
cities, city_error = get_cities(YANDEX_ROOT_PUBLIC_KEY)
if city_error:
    st.warning("Yandex şehir listesi alınamadı: " + str(city_error))

c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("Şehir", options=[""] + cities, format_func=lambda x: "Şehir seçin..." if x == "" else x)

dealers = []
dealer_error = None
if city:
    dealers, dealer_error = get_dealers(YANDEX_ROOT_PUBLIC_KEY, city)

with c2:
    st.markdown("**Bayi Arama ve Seçim**")
    search_term = st.text_input("Bayi ara", placeholder="Jandarma, HTC vb. yazın...", key="dealer_search_box", label_visibility="collapsed")
    
    def tr_lower(text):
        return str(text).replace("İ", "i").replace("I", "ı").lower()

    filtered_dealers = []
    search_cleaned = tr_lower(search_term).strip()
    search_words = [w for w in search_cleaned.split() if w]
    for dealer in dealers:
        dealer_name_lower = tr_lower(dealer["raw_name"])
        if not search_words or all(word in dealer_name_lower for word in search_words):
            filtered_dealers.append(dealer)

    dealer_choices = {x["raw_name"]: x for x in filtered_dealers}
    dealer_raw_names = list(dealer_choices.keys())
    selected_raw_dealer = st.selectbox("Bayi", options=[""] + dealer_raw_names, format_func=lambda x: "Arama sonucu eşleşen bayiyi seçin..." if x == "" else x, label_visibility="collapsed")
    
    dealer_name = ""
    dealer_path = ""
    if selected_raw_dealer in dealer_choices:
        dealer_name = dealer_choices[selected_raw_dealer]["raw_name"]
        dealer_path = dealer_choices[selected_raw_dealer]["path"]

st.divider()
st.subheader("2. Orijinal Referans Fotoğrafı")

ref_img = None
if dealer_path:
    with st.spinner("Sistemdeki orijinal referans fotoğrafı bulunuyor..."):
        ref_img, ref_error = get_reference_image(YANDEX_ROOT_PUBLIC_KEY, dealer_path)

u1, u2 = st.columns(2)
with u1:
    st.markdown("**Orijinal Referans Fotoğrafı**")
    if ref_img is None:
        ref_upload = st.file_uploader("İsterseniz elle yükleyin", type=["jpg", "jpeg", "png", "webp"], key="ref_upload")
        if ref_upload is not None:
            ref_img = prepare_image(decode_uploaded(ref_upload))
    if ref_img is not None:
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.info("Şehir/bayi seçin veya referans görsel yükleyin.")

with u2:
    st.markdown("**Saha Fotoğrafı**")
    field_upload = st.file_uploader("Saha fotoğrafını yükleyin", type=["jpg", "jpeg", "png", "webp"], key="field_upload")
    field_img = prepare_image(decode_uploaded(field_upload)) if field_upload is not None else None
    if field_img is not None:
        st.image(field_img, channels="BGR", use_container_width=True)
    else:
        st.info("Sahadan gelen fotoğrafı yükleyin.")

st.divider()

ready = ref_img is not None and field_img is not None

if st.button("🚀 KONTROLÜ BAŞLAT", type="primary", use_container_width=True, disabled=not ready):
    st.session_state.result_img = None
    st.session_state.aligned_field = None
    st.session_state.results = []
    st.session_state.summary = None
    st.session_state.report = ""

    with st.spinner("Etiket ve planogram uyum analizi çalıştırılıyor..."):
        try:
            result_img, results, summary, aligned_field = analyze_planogram_grid_free(
                ref_img, field_img
            )

            st.session_state.result_img = result_img
            st.session_state.aligned_field = aligned_field
            st.session_state.results = results
            st.session_state.summary = summary
            st.session_state.report = build_report(dealer_name or "Manuel", results, summary)
        except Exception as exc:
            st.error("Analiz motorunda hata oluştu: " + str(exc))

if st.session_state.result_img is not None and st.session_state.summary:
    summary = st.session_state.summary
    
    m1, m2, m3 = st.columns(3)
    m1.metric("🟧 Eksik Etiket", summary.get("etiket_eksigi", 0))
    m2.metric("🟪 Farklılıklar", summary.get("urun_etiket_uyumsuzluk", 0))
    m3.metric("⬜ Kontrol Edilmeyen Rakip Raf", summary.get("kontrol_edilmeyen_rakip_raf", 0))

    has_issues = (summary.get("etiket_eksigi", 0) > 0) or (summary.get("urun_etiket_uyumsuzluk", 0) > 0)

    if has_issues and st.session_state.get("aligned_field") is not None:
        frame_clean = Image.fromarray(cv2.cvtColor(st.session_state.aligned_field, cv2.COLOR_BGR2RGB))
        frame_marked = Image.fromarray(cv2.cvtColor(st.session_state.result_img, cv2.COLOR_BGR2RGB))
        
        gif_io = io.BytesIO()
        frame_marked.save(
            gif_io,
            format="GIF",
            save_all=True,
            append_images=[frame_clean],
            duration=500,
            loop=0
        )
        st.image(gif_io.getvalue(), use_container_width=True)
    else:
        st.image(st.session_state.result_img, channels="BGR", use_container_width=True)

    d1, d2 = st.columns(2)
    ok, encoded = cv2.imencode(".jpg", st.session_state.result_img)
    if ok:
        with d1:
            st.download_button("📥 Denetim Görselini İndir", data=encoded.tobytes(), file_name="stand_kontrol_sonuc.jpg", mime="image/jpeg", use_container_width=True)
    if st.session_state.report:
        with d2:
            st.download_button("📄 Detaylı Raporu İndir", data=st.session_state.report.encode("utf-8"), file_name="stand_kontrol_rapor.txt", mime="text/plain", use_container_width=True)
else:
    st.info("Saha fotoğraflarını yükleyin, ardından kontrolü başlatın.")
