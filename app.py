# -*- coding: utf-8 -*-
"""
ÖZÇELİK STAND KONTROL UYGULAMASI
Gelişmiş Etiket ve Paket Sayımı Sürümü (İlk 6 Raf Modülü)

Kurulum:
    pip install streamlit opencv-python-headless numpy requests pillow

Streamlit Cloud Secrets:
    app_password = "SIFRENIZ"
"""

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


# =========================================================
# GÖRSEL OKUMA VE ÖN İŞLEME
# =========================================================
def decode_uploaded(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        data = np.frombuffer(
            uploaded_file.getvalue(),
            dtype=np.uint8,
        )
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def resize_keep_ratio(
    img,
    max_width=1200,
    max_height=1800,
):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale = min(
        max_width / float(w),
        max_height / float(h),
        1.0,
    )
    if scale >= 0.999:
        return img.copy()

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def prepare_image(img):
    return resize_keep_ratio(
        img,
        max_width=1200,
        max_height=1800,
    )


def detect_shelf_top(img, search_ratio=0.45, extra_margin=0.035):
    try:
        h, w = img.shape[:2]
        search_h = max(10, int(h * search_ratio))
        gray = cv2.cvtColor(
            img[:search_h, :], cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        row_mean = gray.mean(axis=1)
        row_std = gray.std(axis=1)

        k = max(3, search_h // 60)
        kernel = np.ones(k, dtype=np.float32) / k
        row_mean_s = np.convolve(row_mean, kernel, mode="same")
        row_std_s = np.convolve(row_std, kernel, mode="same")

        bright_uniform = (row_mean_s > 150) & (row_std_s < 25)
        idx = np.where(bright_uniform)[0]
        if len(idx) == 0:
            return 0.0

        start = int(idx[0])
        end = start
        for i in idx:
            if i - end <= 3:
                end = i
            else:
                break

        shelf_start = min(search_h - 1, end + int(h * extra_margin))
        ratio = safe_float(shelf_start / h, 0.0)
        return max(0.0, min(0.4, ratio))
    except Exception:
        return 0.0


def safe_download_image(url, timeout=25):
    try:
        if not url:
            return None
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout,
        )
        if response.status_code != 200:
            return None
        data = np.frombuffer(
            response.content,
            dtype=np.uint8,
        )
        return cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )
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
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        if response.status_code != 200:
            return [], f"Yandex HTTP {response.status_code}"
        return (
            response.json()
            .get("_embedded", {})
            .get("items", []),
            None,
        )
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
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        if response.status_code != 200:
            return [], f"Yandex HTTP {response.status_code}"
        return (
            response.json()
            .get("_embedded", {})
            .get("items", []),
            None,
        )
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
            sub_items, _ = yandex_list_dir(
                public_key,
                item.get("path", ""),
            )
            for sub in sub_items:
                if sub.get("type") == "dir":
                    cities.append(
                        sub.get("name", "")
                    )
        else:
            cities.append(name)
    cities = sorted(
        {x for x in cities if x},
        key=normalize_text,
    )
    return cities, None


@st.cache_data(ttl=600, show_spinner=False)
def get_dealers(public_key, city):
    root_items, error = yandex_root_items(public_key)
    if error:
        return [], error
    city_item = None
    for item in root_items:
        if (
            item.get("type") == "dir"
            and normalize_text(item.get("name"))
            == normalize_text(city)
        ):
            city_item = item
            break
    if city_item is None:
        for item in root_items:
            if item.get("type") != "dir":
                continue
            if (
                normalize_text(item.get("name"))
                != "BAYI"
            ):
                continue
            sub_items, _ = yandex_list_dir(
                public_key,
                item.get("path", ""),
            )
            for sub in sub_items:
                if (
                    sub.get("type") == "dir"
                    and normalize_text(
                        sub.get("name")
                    )
                    == normalize_text(city)
                ):
                    city_item = sub
                    break
            if city_item is not None:
                break
    if city_item is None:
        return [], f"'{city}' klasörü bulunamadı."

    items, error = yandex_list_dir(
        public_key,
        city_item.get("path", ""),
    )
    if error:
        return [], error
    
    dealers = []
    prefix_to_remove = normalize_text(city) + " -"
    
    for item in items:
        if (
            item.get("type") == "dir"
            and item.get("name")
            and item.get("path")
        ):
            raw_name = item["name"]
            cleaned_name = raw_name
            
            upper_raw = normalize_text(raw_name)
            if upper_raw.startswith(prefix_to_remove):
                cleaned_name = raw_name[len(prefix_to_remove):].strip()
            elif " - " in raw_name:
                parts = raw_name.split(" - ", 1)
                if normalize_text(parts[0]) == normalize_text(city):
                    cleaned_name = parts[1].strip()

            dealers.append(
                {
                    "name": cleaned_name,
                    "raw_name": raw_name,
                    "path": item["path"],
                }
            )
            
    dealers.sort(
        key=lambda x: normalize_text(x["name"])
    )
    return dealers, None


@st.cache_data(ttl=600, show_spinner=False)
def get_reference_image(public_key, dealer_path):
    items, error = yandex_list_dir(
        public_key,
        dealer_path,
    )
    if error:
        return None, error
    image_items = []
    for item in items:
        if item.get("type") != "file":
            continue
        name = normalize_text(
            item.get("name", "")
        )
        if name.endswith(
            (".JPG", ".JPEG", ".PNG", ".WEBP")
        ):
            image_items.append(item)
    image_items.sort(
        key=lambda x: (
            0
            if "ORJ" in normalize_text(x.get("name"))
            else (
                0
                if "PLANOGRAM"
                in normalize_text(x.get("name"))
                else 1
            ),
            normalize_text(x.get("name")),
        )
    )
    for item in image_items:
        img = safe_download_image(
            item.get("file")
        )
        if img is not None:
            return prepare_image(img), None
    return (
        None,
        "Bayi klasöründe okunabilir JPG/PNG görsel bulunamadı.",
    )


# =========================================================
# HİZALAMA (ORB + RANSAC & ECC)
# =========================================================
def gray_normalize(img):
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY,
    )
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    return clahe.apply(gray)


def orb_align(reference, target):
    h, w = reference.shape[:2]
    if target.shape[:2] != (h, w):
        target = cv2.resize(
            target,
            (w, h),
            interpolation=cv2.INTER_AREA,
        )
    ref_gray = gray_normalize(reference)
    tar_gray = gray_normalize(target)

    orb = cv2.ORB_create(
        nfeatures=7000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=8,
    )
    kp1, des1 = orb.detectAndCompute(
        ref_gray,
        None,
    )
    kp2, des2 = orb.detectAndCompute(
        tar_gray,
        None,
    )

    if (
        des1 is None
        or des2 is None
        or len(kp1) < 15
        or len(kp2) < 15
    ):
        return target, False, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )
    pairs = matcher.knnMatch(
        des2,
        des1,
        k=2,
    )
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.76 * n.distance:
            good.append(m)

    if len(good) < 15:
        return target, False, len(good)

    src = np.float32(
        [kp2[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    dst = np.float32(
        [kp1[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        5.0,
    )
    if matrix is None or mask is None:
        return target, False, 0

    inliers = int(mask.sum())
    ratio = inliers / max(1, len(good))
    if inliers < 12 or ratio < 0.22:
        return target, False, inliers

    aligned = cv2.warpPerspective(
        target,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, True, inliers


def ecc_align(reference, target):
    h, w = reference.shape[:2]
    if target.shape[:2] != (w, h):
        target = cv2.resize(
            target,
            (w, h),
            interpolation=cv2.INTER_AREA,
        )
    ref_gray = (
        cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)
        / 255.0
    )
    tar_gray = (
        cv2.cvtColor(
            target,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)
        / 255.0
    )
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )
    try:
        cv2.findTransformECC(
            ref_gray,
            tar_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            3,
        )
        aligned = cv2.warpAffine(
            target,
            warp,
            (w, h),
            flags=(
                cv2.INTER_LINEAR
                + cv2.WARP_INVERSE_MAP
            ),
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, True
    except Exception:
        return target, False


def align_images(reference, target):
    aligned, ok, inliers = orb_align(
        reference,
        target,
    )
    if ok:
        return (
            aligned,
            True,
            inliers,
            "ORB/RANSAC",
        )
    aligned, ok = ecc_align(
        reference,
        target,
    )
    if ok:
        return aligned, True, 0, "ECC"
    return (
        target,
        False,
        0,
        "Ölçek eşitleme",
    )


# =========================================================
# İLK 6 RAF MODÜLÜ İÇİN KONTUR ANALİZİ
# =========================================================
def analyze_planogram_grid_free(
    reference,
    field,
    roi_top_ratio=0.0,
    roi_bottom_ratio=0.65,
    edge_margin_ratio=0.025,
    illumination_normalize=True,
):
    h, w = reference.shape[:2]

    field = cv2.resize(
        field,
        (w, h),
        interpolation=cv2.INTER_AREA,
    )
    aligned, aligned_ok, inliers, method = (
        align_images(reference, field)
    )

    roi_top = max(0, min(h - 1, int(h * roi_top_ratio)))
    roi_bottom = max(roi_top + 1, min(h, int(h * roi_bottom_ratio)))

    margin_x = int(w * edge_margin_ratio)
    margin_y = int(h * edge_margin_ratio)

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    tar_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    if illumination_normalize:
        roi_ref = ref_gray[roi_top:roi_bottom, margin_x:w - margin_x].astype(np.float32)
        roi_tar = tar_gray[roi_top:roi_bottom, margin_x:w - margin_x].astype(np.float32)
        ref_mean, ref_std = roi_ref.mean(), roi_ref.std() + 1e-6
        tar_mean, tar_std = roi_tar.mean(), roi_tar.std() + 1e-6
        tar_gray = np.clip(
            (tar_gray.astype(np.float32) - tar_mean) * (ref_std / tar_std) + ref_mean,
            0,
            255,
        ).astype(np.uint8)

    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    tar_gray = cv2.GaussianBlur(tar_gray, (5, 5), 0)

    diff = cv2.absdiff(ref_gray, tar_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    thresh[roi_bottom:, :] = 0
    thresh[:roi_top, :] = 0
    if margin_x > 0:
        thresh[:, :margin_x] = 0
        thresh[:, w - margin_x:] = 0
    if margin_y > 0:
        thresh[:margin_y, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    result_img = aligned.copy()
    cv2.line(result_img, (0, roi_bottom), (w, roi_bottom), (255, 180, 0), 2)
    if roi_top > 0:
        cv2.line(result_img, (0, roi_top), (w, roi_top), (255, 180, 0), 2)
    results = []
    fark_sayisi = 0
    paket_eksigi_sayisi = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h * 0.0002) or area > (w * h * 0.15):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if y > roi_bottom or y < roi_top:
            continue

        fark_sayisi += 1

        aspect_ratio = float(bw) / max(1, bh)
        if 0.3 < aspect_ratio < 1.8 and area < (w * h * 0.02):
            paket_eksigi_sayisi += 1
            etiket_turu = f"PAKET #{paket_eksigi_sayisi}"
        else:
            etiket_turu = f"FARK #{fark_sayisi}"

        cv2.rectangle(
            result_img,
            (x, y),
            (x + bw, y + bh),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            result_img,
            etiket_turu,
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        results.append(
            {
                "id": fark_sayisi,
                "durum": "FARK",
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "alan": area,
            }
        )

    summary = {
        "fark": fark_sayisi,
        "paket_eksigi": paket_eksigi_sayisi,
        "supheli": 0,
        "uyumlu": 0,
        "hizalama_ok": aligned_ok,
        "hizalama": method,
        "inliers": inliers,
    }

    return result_img, results, summary


# =========================================================
# RAPOR OLUŞTURUCU
# =========================================================
def build_report(
    dealer,
    results,
    summary,
):
    from datetime import datetime

    lines = [
        "=== ÖZÇELİK STAND KONTROL RAPORU (İLK 6 RAF) ===",
        f"Bayi: {dealer}",
        (
            "Tarih: "
            + datetime.now().strftime(
                "%d.%m.%Y %H:%M:%S"
            )
        ),
        "",
        "Eksik Sigara Paketi Sayısı (İlk 6 Raf): " + str(summary.get('paket_eksigi', 0)),
        "Toplam Tespit Edilen Etiket/Fark: " + str(summary['fark']),
        "",
        "--- FARK BÖLGELERİ ---",
    ]

    for item in results:
        lines.append(
            f"Fark #{item.get('id')} | "
            f"Konum: X={item.get('x')}, Y={item.get('y')} | "
            f"Boyut: {item.get('w')}x{item.get('h')}"
        )

    return "\n".join(lines)


# =========================================================
# SESSION STATE
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


# =========================================================
# GİRİŞ EKRANI
# =========================================================
if "app_password" not in st.secrets:
    st.error(
        "Kritik: Streamlit Secrets içine "
        "app_password eklenmemiş."
    )
    st.stop()

if not st.session_state.authenticated:
    st.title(
        "🔐 Özçelik Stand Kontrol Uygulaması"
    )
    
    try:
        logo_img = Image.open("logo.jpg")
        st.image(logo_img, width=250)
    except Exception:
        pass

    st.caption("Güvenli giriş")

    password = st.text_input(
        "Şifre",
        type="password",
    )

    if st.button(
        "Giriş Yap",
        type="primary",
        use_container_width=True,
    ):
        if hmac.compare_digest(
            password,
            str(st.secrets["app_password"]),
        ):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre.")

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Denetim Ayarları")

    if st.button(
        "🔄 Yandex Önbelleğini Yenile",
        use_container_width=True,
        help=(
            "Yandex Disk'ten çekilen şehir, bayi ve "
            "referans fotoğraf listelerini yeniden yükler."
        ),
    ):
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
        st.toast(
            "Yandex önbelleği temizlendi, veriler yeniden yükleniyor...",
            icon="🔄",
        )
        st.rerun()

    if st.button(
        "🚪 Çıkış Yap",
        use_container_width=True,
    ):
        st.session_state.authenticated = False
        st.session_state.result_img = None
        st.session_state.results = []
        st.session_state.summary = None
        st.rerun()


# =========================================================
# ANA EKRAN & LOKASYON SEÇİMİ
# =========================================================
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
    st.title(
        "📊 ÖZÇELİK STAND KONTROL UYGULAMASI (İLK 6 RAF)"
    )
with refresh_col:
    st.write("")
    if st.button(
        "🔄 Yandex Önbelleğini Yenile",
        use_container_width=True,
        key="main_refresh_btn",
        help=(
            "Yandex Disk'ten çekilen şehir, bayi ve "
            "referans fotoğraf listelerini yeniden yükler."
        ),
    ):
        clear_yandex_cache()
        st.toast(
            "Yandex önbelleği temizlendi, veriler yeniden yükleniyor...",
            icon="🔄",
        )
        st.rerun()

st.subheader("1. Şehir ve Bayi Seçiniz")

cities, city_error = get_cities(
    YANDEX_ROOT_PUBLIC_KEY
)

if city_error:
    st.warning(
        "Yandex şehir listesi alınamadı: "
        + str(city_error)
    )

c1, c2 = st.columns(2)

with c1:
    city = st.selectbox(
        "Şehir",
        options=[""] + cities,
        format_func=lambda x: (
            "Şehir seçin..."
            if x == ""
            else x
        ),
    )

dealers = []
dealer_error = None
if city:
    dealers, dealer_error = get_dealers(
        YANDEX_ROOT_PUBLIC_KEY,
        city,
    )
    if dealer_error:
        st.warning(
            "Yandex bayi listesi alınamadı: "
            + str(dealer_error)
        )

with c2:
    dealer_choices = {x["raw_name"]: x for x in dealers}
    dealer_raw_names = list(dealer_choices.keys())
    
    selected_raw_dealer = st.selectbox(
        "Bayi",
        options=[""] + dealer_raw_names,
        format_func=lambda x: (
            "Bayi seçin veya yazın..."
            if x == ""
            else dealer_choices[x]["name"] if x in dealer_choices else x
        ),
    )
    
    dealer_name = ""
    dealer_path = ""
    if selected_raw_dealer in dealer_choices:
        dealer_name = dealer_choices[selected_raw_dealer]["name"]
        dealer_path = dealer_choices[selected_raw_dealer]["path"]


# =========================================================
# GÖRSEL YÜKLEME VE GÖRÜNTÜLEME
# =========================================================
st.divider()
st.subheader("2. Orjinal Referans Fotoğraf")

ref_img = None
if dealer_path:
    with st.spinner(
        "Sistemdeki orijinal fotoğraf bulunuyor..."
    ):
        ref_img, ref_error = (
            get_reference_image(
                YANDEX_ROOT_PUBLIC_KEY,
                dealer_path,
            )
        )
    if ref_img is None and ref_error:
        st.warning(
            "Yandex referans görseli alınamadı: "
            + str(ref_error)
        )

u1, u2 = st.columns(2)

with u1:
    st.markdown(
        "**Orijinal Referans Fotoğraf**"
    )
    if ref_img is None:
        ref_upload = st.file_uploader(
            "İsterseniz elle yükleyin",
            type=["jpg", "jpeg", "png", "webp"],
            key="ref_upload",
        )
        if ref_upload is not None:
            ref_img = prepare_image(
                decode_uploaded(ref_upload)
            )

    if ref_img is not None:
        st.image(
            ref_img,
            channels="BGR",
            use_container_width=True,
        )
    else:
        st.info("Şehir/bayi seçin veya görsel yükleyin.")

with u2:
    st.markdown("**Saha'dan Gelen Fotoğraf**")
    field_upload = st.file_uploader(
        "Saha fotoğrafını yükleyin",
        type=["jpg", "jpeg", "png", "webp"],
        key="field_upload",
    )
    field_img = (
        prepare_image(
            decode_uploaded(field_upload)
        )
        if field_upload is not None
        else None
    )

    if field_img is not None:
        st.image(
            field_img,
            channels="BGR",
            use_container_width=True,
        )
    else:
        st.info("Sahadan gelen fotoğrafı yükleyin.")


# =========================================================
# ANALİZ ÇALIŞTIRMA
# =========================================================
st.divider()

auto_top_pct = 0
if ref_img is not None and field_img is not None:
    try:
        auto_top_pct = int(
            round(
                max(
                    detect_shelf_top(ref_img),
                    detect_shelf_top(field_img),
                )
                * 100
            )
        )
    except Exception:
        auto_top_pct = 0

roi_widget_key = "roi_slider_" + (
    dealer_path if dealer_path else "manuel"
)

with st.expander("⚙️ Gelişmiş Analiz Ayarları", expanded=False):
    if auto_top_pct > 0:
        st.caption(
            f"📐 Raf üstünde alakasız eşyalar otomatik tespit edildi "
            f"— üst sınır otomatik olarak %{auto_top_pct} önerildi. "
            "Gerekirse aşağıdan elle değiştirebilirsiniz."
        )
    else:
        st.caption(
            "Fotoğraf rafın tam sınırına kırpılmamışsa (raf üstünde "
            "çakmak, süs eşyası vb. görünüyorsa) üst sınırı artırarak "
            "bu alanı analiz dışı bırakabilirsiniz."
        )
    roi_range = st.slider(
        "Analiz Edilecek Raf Bölgesi (görüntü yüksekliğinin %'si)",
        min_value=0,
        max_value=100,
        value=(auto_top_pct, 65),
        step=1,
        key=roi_widget_key,
        help=(
            "Üst sınır fotoğraftan otomatik tahmin edilir. "
            "Raf üstünde alakasız eşyalar varsa alt sınırı "
            "düşürmeden üst sınırı artırın."
        ),
    )
    illumination_normalize = st.checkbox(
        "Işık/Parlaklık Farkını Otomatik Dengele",
        value=True,
        help=(
            "İki fotoğraf farklı ışıkta/flaşla çekildiyse, genel "
            "parlaklık farkının sahte fark olarak işaretlenmesini önler."
        ),
    )

roi_top_ratio = roi_range[0] / 100.0
roi_bottom_ratio = roi_range[1] / 100.0

ready = (
    ref_img is not None
    and field_img is not None
    and roi_bottom_ratio > roi_top_ratio
)

if ref_img is not None and field_img is not None and roi_bottom_ratio <= roi_top_ratio:
    st.warning(
        "Raf bölgesi ayarı geçersiz: üst sınır, alt sınırdan "
        "küçük olmalı."
    )

if st.button(
    "🚀 KONTROLE BAŞLA",
    type="primary",
    use_container_width=True,
    disabled=not ready,
):
    st.session_state.result_img = None
    st.session_state.results = []
    st.session_state.summary = None
    st.session_state.report = ""

    with st.spinner("İlk 6 raf için analiz yapılıyor..."):
        try:
            result_img, results, summary = (
                analyze_planogram_grid_free(
                    ref_img,
                    field_img,
                    roi_top_ratio=roi_top_ratio,
                    roi_bottom_ratio=roi_bottom_ratio,
                    illumination_normalize=illumination_normalize,
                )
            )
            report = build_report(
                dealer_name or "Manuel",
                results,
                summary,
            )

            st.session_state.result_img = (
                result_img
            )
            st.session_state.results = results
            st.session_state.summary = summary
            st.session_state.report = report
        except Exception as exc:
            st.session_state.result_img = None
            st.session_state.results = []
            st.session_state.summary = None
            st.session_state.report = ""
            st.error(
                "Analiz sırasında bir hata oluştu. "
                "Lütfen fotoğrafların bozuk olmadığından emin olup "
                "tekrar deneyin.\n\nTeknik detay: " + str(exc)
            )


# =========================================================
# SONUÇ EKRANI
# =========================================================
if (
    st.session_state.result_img is not None
    and st.session_state.summary
):
    summary = st.session_state.summary

    st.image(
        st.session_state.result_img,
        channels="BGR",
        use_container_width=True,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        st.session_state.result_img,
    )
    if ok:
        st.download_button(
            "📥 İşaretli Denetim Görselini İndir",
            data=encoded.tobytes(),
            file_name=(
                f"{(dealer_name or 'planogram').replace(' ', '_')}"
                "_ilk6raf_denetim.jpg"
            ),
            mime="image/jpeg",
            use_container_width=True,
        )

else:
    st.info(
        "Analiz için referans ve saha fotoğrafını yükleyin, "
        "ardından 'KONTROLE BAŞLA' düğmesine basın."
    )

st.markdown(
    """
<br>
<div style="text-align:center;color:#777;">
Developed by Hakan
</div>
""",
    unsafe_allow_html=True,
)
