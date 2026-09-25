# -*- coding: utf-8 -*-
"""
ÖZÇELİK STAND KONTROL UYGULAMASI (ERTEKİN POLİGRAM VE HATA EŞLEŞTİRME)
"""

import difflib
import hashlib
import hmac
import os
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
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout, allow_redirects=True)
        if response.status_code != 200:
            return None
        data = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# =========================================================
# YANDEX API VE POLİGRAM EXCEL YÖNETİMİ
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
def get_yandex_poligram_models(public_key):
    root_items, error = yandex_root_items(public_key)
    if error:
        return [], error
    
    poligram_folder_path = None
    poligram_files = []

    for item in root_items:
        name_norm = normalize_text(item.get("name"))
        if "POLIGRAM" in name_norm:
            if item.get("type") == "dir":
                poligram_folder_path = item.get("path")
                break
            elif item.get("type") == "file" and item.get("name", "").lower().endswith((".xlsx", ".xls")):
                poligram_files.append({"name": item.get("name"), "file_url": item.get("file"), "path": item.get("path")})

    if not poligram_folder_path and not poligram_files:
        for item in root_items:
            if item.get("type") == "dir":
                sub_items, _ = yandex_list_dir(public_key, item.get("path", ""))
                for sub in sub_items:
                    sub_name_norm = normalize_text(sub.get("name"))
                    if "POLIGRAM" in sub_name_norm:
                        if sub.get("type") == "dir":
                            poligram_folder_path = sub.get("path")
                            break
                        elif sub.get("type") == "file" and sub.get("name", "").lower().endswith((".xlsx", ".xls")):
                            poligram_files.append({"name": sub.get("name"), "file_url": sub.get("file"), "path": sub.get("path")})
                if poligram_folder_path:
                    break

    if poligram_folder_path:
        items, error = yandex_list_dir(public_key, poligram_folder_path)
        if not error:
            for item in items:
                if item.get("type") == "file":
                    name = item.get("name", "")
                    if name.lower().endswith((".xlsx", ".xls")):
                        poligram_files.append({"name": name, "file_url": item.get("file"), "path": item.get("path")})

    if not poligram_files:
        return [], "Yandex Disk üzerinde 'POLİGRAM' içeren klasör veya Excel dosyası bulunamadı."

    return poligram_files, None


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


def load_excel_from_url(public_key, file_item):
    try:
        file_path = file_item.get("path")
        content = None
        if file_path:
            api_url = (
                "https://cloud-api.yandex.net/v1/disk/public/resources/download"
                f"?public_key={urllib.parse.quote(public_key, safe='')}"
                f"&path={urllib.parse.quote(file_path, safe='/')}"
            )
            res = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if res.status_code == 200:
                href = res.json().get("href")
                if href:
                    file_res = requests.get(href, timeout=20, allow_redirects=True)
                    if file_res.status_code == 200:
                        content = file_res.content
        
        if not content:
            download_url = file_item.get("file_url")
            if download_url:
                response = requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20, allow_redirects=True)
                if response.status_code == 200:
                    content = response.content

        if content:
            xls = pd.ExcelFile(io.BytesIO(content), engine='openpyxl')
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if not df.empty:
                    return df
    except Exception as e:
        print("Excel okuma hatası:", e)
    
    return pd.DataFrame()


# =========================================================
# OCR MOTORU (RAF ETİKETİ / SİGARA İSMİ OKUMA)
# =========================================================
def get_ocr_engine():
    """Tesseract OCR motorunu (varsa) yükler. Kurulu değilse None döner."""
    try:
        import pytesseract
        # Sistemde tesseract binary'si gerçekten çalışıyor mu diye hızlı kontrol
        pytesseract.get_tesseract_version()
        return pytesseract
    except Exception:
        return None


def ocr_read_label(gray_roi, ocr_engine):
    """Tek bir raf gözündeki (kırpılmış) etiket bölgesinden metin okur."""
    if ocr_engine is None or gray_roi is None or gray_roi.size == 0:
        return ""
    try:
        h, w = gray_roi.shape[:2]
        if h < 5 or w < 5:
            return ""
        # OCR doğruluğunu artırmak için küçük kırpımları büyüt
        scale = max(1.0, 220.0 / float(h))
        roi = cv2.resize(gray_roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Tek satır/tek blok metin okuma modu, Türkçe + İngilizce
        config = "--oem 3 --psm 7"
        try:
            text = ocr_engine.image_to_string(thresh, config=config, lang="tur+eng")
        except Exception:
            text = ocr_engine.image_to_string(thresh, config=config)
        return text.strip()
    except Exception:
        return ""


def text_match_ratio(detected, expected):
    """Okunan metin ile Excel'deki beklenen ürün adını normalize edip bulanık eşleştirir."""
    det_n = normalize_text(detected)
    exp_n = normalize_text(expected)
    if not det_n or not exp_n:
        return 0.0
    if exp_n in det_n or det_n in exp_n:
        return 1.0
    return difflib.SequenceMatcher(None, det_n, exp_n).ratio()


# =========================================================
# GÖRSEL HİZALAMA VE ANALİZ MOTORLARI
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


def analyze_poligram_model(field_img, poligram_df, match_threshold=0.55, label_band=(0.55, 0.98)):
    """
    Poligram modeli (Excel) ile saha fotoğrafını karşılaştırır.
    - Excel: satır = raf, sütun 1..N = o raftaki slotların beklenen sigara/ürün ismi.
    - Fotoğraf aynı (raf x slot) grid'ine bölünür.
    - Her gözün etiket bandı (rafın alt kısmı, fiyat/isim etiketinin olduğu yer) OCR ile okunur.
    - Okunan metin, Excel'deki beklenen isimle bulanık (fuzzy) karşılaştırılır.
    - Eşleşme oranı eşik değerin altındaysa (ya da okunamadıysa) o slot kırmızı çerçeveyle işaretlenir.
    """
    h, w = field_img.shape[:2]
    result_img = field_img.copy()

    gray = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    ocr_engine = get_ocr_engine()

    if poligram_df is not None and not poligram_df.empty:
        num_rows = len(poligram_df)
        num_cols = poligram_df.shape[1] - 1
    else:
        num_rows = 7
        num_cols = 11

    shelf_h = h // max(num_rows, 1)
    col_w = w // max(num_cols, 1)

    fark_sayisi = 0
    okunamayan_sayisi = 0
    results = []

    for r_idx in range(num_rows):
        s_top = r_idx * shelf_h
        s_bottom = (r_idx + 1) * shelf_h if r_idx < num_rows - 1 else h

        if poligram_df is not None and not poligram_df.empty and r_idx < len(poligram_df):
            row_data = poligram_df.iloc[r_idx]
        else:
            row_data = None

        for c_idx in range(num_cols):
            expected_raw = row_data.iloc[c_idx + 1] if row_data is not None and (c_idx + 1) < len(row_data) else None
            expected_product = "" if expected_raw is None or pd.isna(expected_raw) else str(expected_raw).strip()

            # Poligram modelinde o slot boş bırakılmışsa (ürün beklenmiyorsa) kontrol dışı bırak
            if not expected_product:
                continue

            c_left = c_idx * col_w
            c_right = (c_idx + 1) * col_w if c_idx < num_cols - 1 else w

            # Etiket bandı: rafın alt kısmı (fiyat/isim etiketinin tipik olarak bulunduğu yer)
            label_top = s_top + int(shelf_h * label_band[0])
            label_bottom = s_top + int(shelf_h * label_band[1])
            roi = gray_clahe[label_top:label_bottom, c_left:c_right]

            detected_text = ocr_read_label(roi, ocr_engine)
            similarity = text_match_ratio(detected_text, expected_product)
            is_mismatch = similarity < match_threshold

            if not detected_text:
                okunamayan_sayisi += 1

            if is_mismatch:
                fark_sayisi += 1
                box_x1 = c_left + int(col_w * 0.05)
                box_y1 = s_top + int(shelf_h * 0.10)
                box_x2 = c_right - int(col_w * 0.05)
                box_y2 = s_bottom - int(shelf_h * 0.05)

                cv2.rectangle(result_img, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 2)
                etiket = "OKUNAMADI" if not detected_text else "UYUMSUZ"
                cv2.putText(
                    result_img,
                    f"{etiket} (Raf {r_idx+1})",
                    (box_x1, max(15, box_y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                results.append({
                    "id": fark_sayisi,
                    "durum": (
                        f"POLİGRAM UYUMSUZLUĞU: Raf {r_idx+1}, Slot {c_idx+1} | "
                        f"Beklenen: {expected_product} | Okunan: {detected_text or '—'} "
                        f"(Benzerlik: {similarity:.2f})"
                    ),
                    "x": box_x1, "y": box_y1, "w": box_x2 - box_x1, "h": box_y2 - box_y1
                })

    ocr_uyarisi = ""
    if ocr_engine is None:
        ocr_uyarisi = " | UYARI: OCR motoru (pytesseract/tesseract) sunucuda kurulu değil, hiçbir etiket okunamadı."

    summary = {
        "fark": fark_sayisi,
        "etiket_eksigi": okunamayan_sayisi,
        "urun_etiket_uyumsuzluk": fark_sayisi,
        "kontrol_edilmeyen_rakip_raf": 0,
        "hizalama_ok": ocr_engine is not None,
        "hizalama": f"Poligram OCR Eşleştirmesi ({num_rows} Raf, {num_cols} Kolon){ocr_uyarisi}",
    }
    return result_img, results, summary, field_img


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
                    box_color = (0, 0, 255)
                elif ref_piece is not None and np.mean(np.abs(ref_piece.astype(np.float32) - roi_target_piece.astype(np.float32))) > 40:
                    product_label_mismatch_count += 1
                    fark_sayisi += 1
                    etiket_turu = f"FARKLILIK #{product_label_mismatch_count}"
                    box_color = (0, 0, 255)
                else:
                    continue

                cv2.rectangle(result_img, (x, abs_y), (x + bw, abs_y + bh), box_color, 3)
                cv2.putText(
                    result_img,
                    etiket_turu,
                    (x, max(15, abs_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
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
                
                cv2.rectangle(result_img, (x, abs_y), (x + bw, abs_y + bh), (0, 0, 255), 2)
                cv2.putText(
                    result_img,
                    etiket_turu,
                    (x, max(15, abs_y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
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
        "hizalama": "Standart Hibrit Motor",
    }
    return result_img, results, summary, aligned


def build_report(dealer, results, summary):
    from datetime import datetime
    lines = [
        "=== ÖZÇELİK STAND DENETİM RAPORU ===",
        f"Bayi / Model: {dealer}",
        "Tarih: " + datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "",
        "Toplam Fark / Hata Sayısı: " + str(summary.get('fark', 0)),
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
# ARAYÜZ
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

st.subheader("0. Kontrol Modu Seçimi")
kontrol_modu = st.radio(
    "Kontrol Yöntemini Seçin",
    options=["Standart Referans Kontrolü", "POLİGRAM (Excel Modeli ile Kontrol)"],
    horizontal=True
)

st.divider()

city, dealer_name, dealer_path = "", "", ""
selected_poligram_item = None
field_img = None

if kontrol_modu == "Standart Referans Kontrolü":
    st.subheader("1. Şehir ve Bayi Seçiniz")
    cities, city_error = get_cities(YANDEX_ROOT_PUBLIC_KEY)
    if city_error:
        st.warning("Yandex şehir listesi alınamadı: " + str(city_error))

    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("Şehir", options=[""] + cities, format_func=lambda x: "Şehir seçin..." if x == "" else x)

    dealers = []
    if city:
        dealers, _ = get_dealers(YANDEX_ROOT_PUBLIC_KEY, city)

    with c2:
        st.markdown("**Bayi Arama ve Seçim**")
        search_term = st.text_input("Bayi ara", placeholder="Bayi adı yazın...", key="dealer_search_box", label_visibility="collapsed")
        
        filtered_dealers = []
        for dealer in dealers:
            if not search_term or normalize_text(search_term) in normalize_text(dealer["raw_name"]):
                filtered_dealers.append(dealer)

        dealer_choices = {x["raw_name"]: x for x in filtered_dealers}
        selected_raw_dealer = st.selectbox("Bayi", options=[""] + list(dealer_choices.keys()), format_func=lambda x: "Bayi seçin..." if x == "" else x, label_visibility="collapsed")
        
        if selected_raw_dealer in dealer_choices:
            dealer_name = dealer_choices[selected_raw_dealer]["raw_name"]
            dealer_path = dealer_choices[selected_raw_dealer]["path"]

    st.divider()
    st.subheader("2. Fotoğraf Yükleme ve Kontrol")

    ref_img = None
    if dealer_path:
        with st.spinner("Orijinal referans fotoğrafı yükleniyor..."):
            ref_img, _ = get_reference_image(YANDEX_ROOT_PUBLIC_KEY, dealer_path)

    u1, u2 = st.columns(2)
    with u1:
        st.markdown("**Orijinal Referans Fotoğrafı**")
        if ref_img is not None:
            st.image(ref_img, channels="BGR", use_container_width=True)
        else:
            st.info("Şehir/bayi seçin.")

    with u2:
        st.markdown("**Bayi Saha Fotoğrafı**")
        field_upload = st.file_uploader("Kontrol edilecek bayi fotoğrafını yükleyin", type=["jpg", "jpeg", "png", "webp"], key="field_upload_std")
        field_img = prepare_image(decode_uploaded(field_upload)) if field_upload is not None else None
        if field_img is not None:
            st.image(field_img, channels="BGR", use_container_width=True)
        else:
            st.info("Sahadan gelen fotoğrafı yükleyin.")

else:
    st.subheader("1. POLİGRAM Modeli ve Saha Fotoğrafı Seçimi")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Yandex Disk POLİGRAM Modelleri**")
        yandex_poligrams, pol_error = get_yandex_poligram_models(YANDEX_ROOT_PUBLIC_KEY)
        if pol_error:
            st.warning(str(pol_error))

        poligram_dict = {item["name"]: item for item in yandex_poligrams}
        selected_poligram_name = st.selectbox(
            "Poligram Excel modeli seçin",
            options=[""] + list(poligram_dict.keys()),
            format_func=lambda x: "Poligram Excel modeli seçin..." if x == "" else x,
            label_visibility="collapsed"
        )
        if selected_poligram_name in poligram_dict:
            selected_poligram_item = poligram_dict[selected_poligram_name]

        pol_df = None
        if selected_poligram_item:
            with st.spinner("Excel modeli yükleniyor..."):
                pol_df = load_excel_from_url(YANDEX_ROOT_PUBLIC_KEY, selected_poligram_item)
            if pol_df is not None:
                st.success(f"Model Yüklendi: {selected_poligram_item['name']}")
            else:
                st.warning("Seçilen Excel dosyası okunamadı.")
        else:
            st.info("Yandex Disk'ten bir Poligram modeli seçin.")

    with col_p2:
        st.markdown("**Bayi Saha Fotoğrafı**")
        field_upload_poly = st.file_uploader("Kontrol edilecek bayi fotoğrafını yükleyin", type=["jpg", "jpeg", "png", "webp"], key="field_upload_poly")
        field_img = prepare_image(decode_uploaded(field_upload_poly)) if field_upload_poly is not None else None
        if field_img is not None:
            st.image(field_img, channels="BGR", use_container_width=True)
        else:
            st.info("Sahadan gelen fotoğrafı yükleyin.")

st.divider()

if kontrol_modu == "Standart Referans Kontrolü":
    ready = ref_img is not None and field_img is not None
else:
    ready = selected_poligram_item is not None and field_img is not None and 'pol_df' in locals() and pol_df is not None

if st.button("🚀 KONTROLÜ BAŞLAT", type="primary", use_container_width=True, disabled=not ready):
    st.session_state.result_img = None
    st.session_state.aligned_field = None
    st.session_state.results = []
    st.session_state.summary = None
    st.session_state.report = ""

    with st.spinner("Poligram ve ürün uygunluk analizi gerçekleştiriliyor..."):
        try:
            if kontrol_modu == "Standart Referans Kontrolü":
                result_img, results, summary, aligned_field = analyze_planogram_grid_free(ref_img, field_img)
            else:
                pol_df = load_excel_from_url(YANDEX_ROOT_PUBLIC_KEY, selected_poligram_item)
                result_img, results, summary, aligned_field = analyze_poligram_model(field_img, pol_df)

            st.session_state.result_img = result_img
            st.session_state.aligned_field = aligned_field
            st.session_state.results = results
            st.session_state.summary = summary
            st.session_state.report = build_report(dealer_name or selected_poligram_item["name"], results, summary)
        except Exception as exc:
            st.error("Analiz sırasında hata oluştu: " + str(exc))

if st.session_state.result_img is not None and st.session_state.summary:
    summary = st.session_state.summary
    
    st.metric("🚨 Tespit Edilen Toplam Fark / Uyumsuzluk", summary.get("fark", 0))
    st.image(st.session_state.result_img, channels="BGR", use_container_width=True)

    d1, d2 = st.columns(2)
    ok, encoded = cv2.imencode(".jpg", st.session_state.result_img)
    if ok:
        with d1:
            st.download_button("📥 Kontrol Görselini İndir", data=encoded.tobytes(), file_name="poligram_kontrol_sonuc.jpg", mime="image/jpeg", use_container_width=True)
    if st.session_state.report:
        with d2:
            st.download_button("📄 Detaylı Raporu İndir", data=st.session_state.report.encode("utf-8"), file_name="poligram_kontrol_rapor.txt", mime="text/plain", use_container_width=True)
else:
    st.info("Gerekli seçimleri yapıp saha fotoğrafını yükledikten sonra kontrolü başlatabilirsiniz.")
