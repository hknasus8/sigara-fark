# -*- coding: utf-8 -*-
"""
ÖZÇELİK STAND KONTROL UYGULAMASI
Raf Bazlı Analiz + Kalın Kırmızı ("FARKLI GÖRSEL") + Nokta Atışı Yeşil (Doğru İşaretlenen Alan)
"""

import difflib
import hashlib
import hmac
import re
import urllib.parse
from PIL import Image

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
# HASSAS RAF BAZLI HİZALAMA VE ANALİZ 
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


def _count_wide_segments(binary_row, min_w=15, max_w=90):
    """Bir satırda genişliği min_w..max_w arasında olan ayrı 'parlak' bölüm sayısı."""
    cnt = 0
    run = 0
    for v in binary_row:
        if v:
            run += 1
        else:
            if min_w <= run <= max_w:
                cnt += 1
            run = 0
    if min_w <= run <= max_w:
        cnt += 1
    return cnt


def find_label_bands(gray, top_y, bottom_y):
    """
    Etiket şeritlerini (fiyat/isim etiketi sıraları) SABİT bir yükseklik
    varsaymadan, İÇERİĞE bakarak bulur: bir etiket şeridi, yan yana çok
    sayıda orta genişlikte parlak dikdörtgenden (etiketlerden) oluşan,
    nispeten KISA (15-42px) bir bant olarak görünür. Ürün fotoğrafı
    bantları ise çok daha yüksektir (>45px) ve bu şekilde elenir.
    """
    roi = gray[top_y:bottom_y, :]
    if roi.size == 0:
        return []

    _, bright = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
    bright01 = (bright > 0).astype(np.uint8)

    seg_counts = np.array(
        [_count_wide_segments(bright01[y]) for y in range(bright01.shape[0])],
        dtype=np.float32,
    )
    if seg_counts.size == 0:
        return []

    smooth = np.convolve(seg_counts, np.ones(5) / 5.0, mode="same")
    is_label_row = smooth >= 6

    bands = []
    in_band = False
    start = 0
    for y, v in enumerate(is_label_row):
        if v and not in_band:
            start = y
            in_band = True
        elif not v and in_band:
            bands.append((start, y))
            in_band = False
    if in_band:
        bands.append((start, len(is_label_row)))

    return [(b[0] + top_y, b[1] + top_y) for b in bands if 14 <= (b[1] - b[0]) <= 42]


def segment_label_cells(gray, band_top, band_bot):
    """Bir etiket şeridi bandı içindeki ayrı etiket hücrelerini (x,y,w,h) döndürür."""
    band = gray[band_top:band_bot, :]
    if band.size == 0:
        return []

    _, mask = cv2.threshold(band, 150, 255, cv2.THRESH_BINARY)
    mask01 = (mask > 0).astype(np.float32)
    col_frac = mask01.mean(axis=0)

    is_label_col = col_frac > 0.4
    x_ranges = []
    in_cell = False
    start = 0
    for x, v in enumerate(is_label_col):
        if v and not in_cell:
            start = x
            in_cell = True
        elif not v and in_cell:
            x_ranges.append((start, x))
            in_cell = False
    if in_cell:
        x_ranges.append((start, len(is_label_col)))

    cells = []
    for x1, x2 in x_ranges:
        bw = x2 - x1
        if bw < 15 or bw > 100:
            continue
        row_frac = mask01[:, x1:x2].mean(axis=1)
        rows = np.where(row_frac > 0.75)[0]
        if rows.size == 0:
            continue
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        bh = y2 - y1
        if bh < 6:
            continue
        cells.append((x1, band_top + y1, bw, bh))
    return cells


def detect_empty_labels(ref_gray, tar_gray, result_img, top_y, bottom_y, w):
    """
    Referans görüntüdeki gerçek etiket şeritlerini (band + hücre bazında)
    otomatik bulur; her hücreyi sahadaki (hizalanmış) karşılığıyla
    karşılaştırır. Referansta yazı/metin var (yüksek piksel varyansı) ama
    sahada aynı hücre parlak/beyaz kalıp İÇİ BOŞ (düşük varyans) ise,
    o hücreyi kalın YEŞİL çerçeve ile "EKSİK ETİKET" olarak işaretler.
    """
    label_bands = find_label_bands(ref_gray, top_y, bottom_y)

    eksik_etiket_sayisi = 0
    eksik_etiket_results = []

    for band_top, band_bot in label_bands:
        cells = segment_label_cells(ref_gray, band_top, band_bot)
        for (x, y, bw, bh) in cells:
            ref_cell = ref_gray[y:y + bh, x:x + bw]
            tar_cell = tar_gray[y:y + bh, x:x + bw]
            if ref_cell.size == 0 or tar_cell.size == 0:
                continue

            ref_std = float(np.std(ref_cell))
            tar_std = float(np.std(tar_cell))
            tar_mean = float(np.mean(tar_cell))

            # Referans hücrede belirgin metin/desen var (yüksek varyans),
            # sahadaki aynı hücre hâlâ parlak (etiket kağıdı yerinde duruyor)
            # ama üzerinde neredeyse hiç yazı yok (varyans referansın çok altında).
            if ref_std > 20 and tar_mean > 140 and tar_std < 35 and tar_std < ref_std * 0.72:
                eksik_etiket_sayisi += 1

                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
                cv2.putText(
                    result_img,
                    f"EKSIK ETIKET #{eksik_etiket_sayisi}",
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                eksik_etiket_results.append(
                    {"id": eksik_etiket_sayisi, "durum": "EKSIK ETIKET", "x": x, "y": y, "w": bw, "h": bh, "alan": bw * bh}
                )

    return eksik_etiket_sayisi, eksik_etiket_results


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
    shelf_height = (bottom_y - top_y) // 6

    results = []
    fark_sayisi = 0
    paket_eksigi_sayisi = 0
    farkli_gorsel_sayisi = 0
    eksik_etiket_sayisi = 0

    for i in range(6):
        s_top = top_y + (i * shelf_height)
        s_bottom = s_top + shelf_height if i < 5 else bottom_y

        ref_roi = ref_gray[s_top:s_bottom, :]
        tar_roi = tar_gray[s_top:s_bottom, :]

        ref_roi_blur = cv2.GaussianBlur(ref_roi, (5, 5), 0)
        tar_roi_blur = cv2.GaussianBlur(tar_roi, (5, 5), 0)

        diff = cv2.absdiff(ref_roi_blur, tar_roi_blur)
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w * h * 0.0012) or area > (w * h * 0.08):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            abs_y = s_top + y

            if x < 10 or (x + bw) > (w - 10):
                continue

            check_label_y1 = min(s_bottom - 5, abs_y + bh - int(bh * 0.2))
            check_label_y2 = min(s_bottom, abs_y + bh + int(bh * 0.3))
            label_strip_region = tar_gray[check_label_y1:check_label_y2, max(0, x-5):min(w, x+bw+5)]
            
            has_label = False
            if label_strip_region.size > 0:
                mean_brightness = np.mean(label_strip_region)
                if mean_brightness > 75: 
                    has_label = True

            fark_sayisi += 1
            aspect_ratio = float(bw) / max(1, bh)
            
            if 0.2 < aspect_ratio < 2.0:
                if has_label:
                    farkli_gorsel_sayisi += 1
                    etiket_turu = f"FARKLI GÖRSEL #{farkli_gorsel_sayisi}"
                    box_color = (0, 0, 255) # Kırmızı
                    box_thickness = 4
                else:
                    paket_eksigi_sayisi += 1
                    etiket_turu = f"EKSİK #{paket_eksigi_sayisi}"
                    box_color = (0, 0, 255)
                    box_thickness = 2
            else:
                etiket_turu = f"FARK #{fark_sayisi}"
                box_color = (0, 0, 255)
                box_thickness = 2

            cv2.rectangle(result_img, (x, abs_y), (x + bw, abs_y + bh), box_color, box_thickness)
            cv2.putText(
                result_img,
                etiket_turu,
                (x, max(15, abs_y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                box_color,
                1,
                cv2.LINE_AA,
            )

            results.append({"id": fark_sayisi, "durum": etiket_turu, "x": x, "y": abs_y, "w": bw, "h": bh, "alan": area})

    # =====================================================
    # GENEL TARAMA: TÜM ROI İÇİNDEKİ GERÇEK ETİKET ŞERİTLERİNİ BUL VE KONTROL ET
    # (Raf yükseklikleri eşit olmadığından sabit bant varsayımı yerine
    #  etiketler içeriklerine göre otomatik tespit edilir)
    # =====================================================
    eksik_etiket_sayisi, eksik_etiket_results = detect_empty_labels(
        ref_gray, tar_gray, result_img, top_y, bottom_y, w
    )
    results.extend(eksik_etiket_results)
    fark_sayisi += eksik_etiket_sayisi

    summary = {
        "fark": fark_sayisi,
        "paket_eksigi": paket_eksigi_sayisi,
        "farkli_gorsel": farkli_gorsel_sayisi,
        "eksik_etiket": eksik_etiket_sayisi,
        "supheli": 0,
        "uyumlu": 0,
        "hizalama_ok": aligned_ok,
        "hizalama": "Raf Bazlı Hibrit",
        "inliers": 0,
    }

    return result_img, results, summary, aligned


def build_report(dealer, results, summary):
    from datetime import datetime
    lines = [
        "=== ÖZÇELİK STAND KONTROL RAPORU (İLK 6 RAF) ===",
        f"Bayi: {dealer}",
        "Tarih: " + datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "",
        "Eksik Paket Sayısı: " + str(summary.get('paket_eksigi', 0)),
        "Farklı Görsel Sayısı (Kalın Kırmızı): " + str(summary.get('farkli_gorsel', 0)),
        "Eksik Etiket Sayısı (Kalın Yeşil): " + str(summary.get('eksik_etiket', 0)),
        "Toplam Tespit Edilen Fark: " + str(summary['fark']),
        "",
        "--- FARK BÖLGELERİ ---"
    ]
    for item in results:
        lines.append(f"Fark ID #{item.get('id')} ({item.get('durum')}) | Konum: X={item.get('x')}, Y={item.get('y')} | Boyut: {item.get('w')}x{item.get('h')}")
    return "\n".join(lines)


# =========================================================
# SESSION STATE & GİRİŞ
# =========================================================
DEFAULT_STATE = {
    "authenticated": False,
    "result_img": None,
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
# SIDEBAR VE ARAYÜZ
# =========================================================
with st.sidebar:
    st.header("⚙️ Denetim Ayarları")
    if st.button("🔄 Yandex Önbelleğini Yenile", use_container_width=True):
        try:
            yandex_root_items.clear()
            yandex_list_dir.clear()
            get_cities.clear()
            get_dealers.clear()
            get_reference_image.clear()
        except Exception:
            st.cache_data.clear()
        st.session_state.result_img = None
        st.session_state.results = []
        st.session_state.summary = None
        st.session_state.report = ""
        st.rerun()

    if st.button("🚪 Çıkış Yap", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.result_img = None
        st.session_state.results = []
        st.session_state.summary = None
        st.rerun()


def clear_yandex_cache():
    try:
        yandex_root_items.clear()
        yandex_list_dir.clear()
        get_cities.clear()
        get_dealers.clear()
        get_reference_image.clear()
    except Exception:
        st.cache_data.clear()
    st.session_state.result_img = None
    st.session_state.results = []
    st.session_state.summary = None
    st.session_state.report = ""


title_col, refresh_col = st.columns([5, 1])
with title_col:
    st.title("📊 ÖZÇELİK STAND KONTROL UYGULAMASI (İLK 6 RAF)")
with refresh_col:
    st.write("")
    if st.button("🔄 Yenile", use_container_width=True, key="main_refresh_btn"):
        clear_yandex_cache()
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
st.subheader("2. Orijinal Referans Fotoğraf ve Saha Fotoğrafı")

ref_img = None
if dealer_path:
    with st.spinner("Sistemdeki orijinal fotoğraf bulunuyor..."):
        ref_img, ref_error = get_reference_image(YANDEX_ROOT_PUBLIC_KEY, dealer_path)

u1, u2 = st.columns(2)
with u1:
    st.markdown("**Orijinal Referans Fotoğraf**")
    if ref_img is None:
        ref_upload = st.file_uploader("İsterseniz elle yükleyin", type=["jpg", "jpeg", "png", "webp"], key="ref_upload")
        if ref_upload is not None:
            ref_img = prepare_image(decode_uploaded(ref_upload))
    if ref_img is not None:
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.info("Şehir/bayi seçin veya görsel yükleyin.")

with u2:
    st.markdown("**Saha'dan Gelen Fotoğraf**")
    field_upload = st.file_uploader("Saha fotoğrafını yükleyin", type=["jpg", "jpeg", "png", "webp"], key="field_upload")
    field_img = prepare_image(decode_uploaded(field_upload)) if field_upload is not None else None
    if field_img is not None:
        st.image(field_img, channels="BGR", use_container_width=True)
    else:
        st.info("Sahadan gelen fotoğrafı yükleyin.")

st.divider()

ready = ref_img is not None and field_img is not None

if st.button("🚀 KONTROLE BAŞLA", type="primary", use_container_width=True, disabled=not ready):
    st.session_state.result_img = None
    st.session_state.results = []
    st.session_state.summary = None
    st.session_state.report = ""

    with st.spinner("İlk 6 raf analiz ediliyor (Ürün altı etiketler ve görseller taranıyor)..."):
        try:
            result_img, results, summary, aligned_field = analyze_planogram_grid_free(
                ref_img, field_img
            )

            st.session_state.result_img = result_img
            st.session_state.results = results
            st.session_state.summary = summary
            st.session_state.report = build_report(dealer_name or "Manuel", results, summary)
        except Exception as exc:
            st.error("Analiz sırasında hata oluştu: " + str(exc))

if st.session_state.result_img is not None and st.session_state.summary:
    summary = st.session_state.summary
    m1, m2, m3 = st.columns(3)
    m1.metric("Farklı Görsel (Kırmızı)", summary.get("farkli_gorsel", 0))
    m2.metric("Eksik Etiket (Yeşil)", summary.get("eksik_etiket", 0))
    m3.metric("Paket Eksiği", summary.get("paket_eksigi", 0))

    st.image(st.session_state.result_img, channels="BGR", use_container_width=True)

    d1, d2 = st.columns(2)
    ok, encoded = cv2.imencode(".jpg", st.session_state.result_img)
    if ok:
        with d1:
            st.download_button("📥 İşaretli Görseli İndir", data=encoded.tobytes(), file_name="denetim_sonuc.jpg", mime="image/jpeg", use_container_width=True)
    if st.session_state.report:
        with d2:
            st.download_button("📄 Raporu İndir", data=st.session_state.report.encode("utf-8"), file_name="rapor.txt", mime="text/plain", use_container_width=True)
else:
    st.info("Analiz için referans ve saha fotoğrafını yükleyin, ardından 'KONTROLE BAŞLA' düğmesine basın.")
