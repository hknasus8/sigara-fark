# -*- coding: utf-8 -*-
"""
ÖZÇELİK STAND KONTROL UYGULAMASI
Gelişmiş Etiket ve Paket Sayımı Sürümü (İlk 6 Raf Modülü)
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

# OCR (etiket / ürün adı okuma) modülü opsiyoneldir.
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False


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

RAF_SAYISI = 6  # İlk 6 raf modülü
OCR_LANG_TRY_ORDER = ("tur+eng", "eng")
TAG_MIN_AREA_RATIO = 0.0012
TAG_SEARCH_BAND_RATIO = 0.22
TAG_MIN_MEAN_BRIGHTNESS = 165
TAG_MAX_STD_BRIGHTNESS = 75
NAME_STRIP_HEIGHT_RATIO = 0.18
NAME_STRIP_GAP_RATIO = 0.05
DEFAULT_LABEL_SIM_THRESHOLD = 0.55


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


def detect_real_shelf_boundaries(img, roi_top_px, roi_bottom_px, num_shelves=6):
    """
    Fotoğraftaki gerçek raf çizgilerini (yatay kenar ve koyu bant geçişlerini)
    piksel yoğunluğu analiziyle otomatik tespit eder.
    """
    try:
        h, w = img.shape[:2]
        if roi_bottom_px <= roi_top_px:
            return [round(100 * i / num_shelves) for i in range(1, num_shelves)]

        roi = img[roi_top_px:roi_bottom_px, int(w*0.1):int(w*0.9)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Dikey yönde yatay çizgileri bulmak için gradyan ve ortalama analizi
        row_mean = np.mean(gray, axis=1)
        grad_y = np.gradient(row_mean)
        
        total_h = roi_bottom_px - roi_top_px
        ideal_step = total_h / num_shelves
        
        detected_bounds = []
        for i in range(1, num_shelves):
            expected_y = int(i * ideal_step)
            # Beklenen konumun etrafında (+/- %25 marj) en güçlü yatay çizgiyi ara
            search_window = int(ideal_step * 0.25)
            s_start = max(0, expected_y - search_window)
            s_end = min(total_h, expected_y + search_window)
            
            if s_end > s_start:
                segment = np.abs(grad_y[s_start:s_end])
                if len(segment) > 0 and np.max(segment) > 1.0:
                    best_offset = np.argmax(segment)
                    best_y = s_start + best_offset
                else:
                    best_y = expected_y
                
                pct = int(round(100 * best_y / total_h))
                detected_bounds.append(max(1, min(99, pct)))
            else:
                detected_bounds.append(int(round(100 * i / num_shelves)))
                
        # Sıralı ve çakışmasız olduğundan emin ol
        corrected = []
        last = 0
        for b in sorted(detected_bounds):
            safe_b = max(b, last + 2)
            corrected.append(safe_b)
            last = safe_b
        return corrected
    except Exception:
        return [round(100 * i / num_shelves) for i in range(1, num_shelves)]


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
    for item in items:
        if (
            item.get("type") == "dir"
            and item.get("name")
            and item.get("path")
        ):
            raw_name = item["name"]
            dealers.append(
                {
                    "name": raw_name,
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
    if target.shape[:2] != (w, h):
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

    return result_img, results, summary, aligned


# =========================================================
# ETİKET / ÜRÜN ADI KONTROL MODÜLÜ (OCR)
# =========================================================
def normalize_ocr_text(value):
    value = normalize_text(value)
    value = re.sub(r"[^A-Z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def text_similarity(a, b):
    a = normalize_ocr_text(a)
    b = normalize_ocr_text(b)
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()

    words_a = set(a.split())
    words_b = set(b.split())
    if words_a and words_b:
        overlap = len(words_a & words_b) / max(
            1, min(len(words_a), len(words_b))
        )
    else:
        overlap = 0.0

    if a in b or b in a:
        overlap = max(overlap, 0.85)

    return max(ratio, overlap)


def ocr_text_from_crop(crop_bgr, lang="eng"):
    if not OCR_AVAILABLE or crop_bgr is None:
        return ""
    if crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return ""
    try:
        scale = max(1, int(120 / max(1, crop_bgr.shape[0])))
        big = cv2.resize(
            crop_bgr,
            None,
            fx=scale + 2,
            fy=scale + 2,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 40, 40)
        _, th = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if np.mean(th) < 127:
            th = cv2.bitwise_not(th)
        txt = pytesseract.image_to_string(
            th, lang=lang, config="--psm 6"
        )
        return txt
    except Exception:
        return ""


def ocr_with_fallback(crop_bgr):
    for lang in OCR_LANG_TRY_ORDER:
        try:
            txt = ocr_text_from_crop(crop_bgr, lang=lang)
            if txt.strip():
                return txt
        except Exception:
            continue
    return ""


def detect_tag_boxes(band_bgr):
    h, w = band_bgr.shape[:2]
    search_top = int(h * (1.0 - TAG_SEARCH_BAND_RATIO))
    sub = band_bgr[search_top:h, :]
    if sub.size == 0:
        return []

    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = w * (h * TAG_SEARCH_BAND_RATIO) * TAG_MIN_AREA_RATIO
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area:
            continue
        if bw < w * 0.015 or bw > w * 0.22:
            continue
        if bh < sub.shape[0] * 0.25:
            continue

        patch = gray[y : y + bh, x : x + bw]
        if patch.size == 0:
            continue
        if patch.mean() < TAG_MIN_MEAN_BRIGHTNESS:
            continue
        if patch.std() > TAG_MAX_STD_BRIGHTNESS:
            continue

        boxes.append((x, y + search_top, bw, bh))

    boxes.sort(key=lambda b: b[0])
    return boxes


def extract_name_strip_above(band_bgr, tag_box):
    h, w = band_bgr.shape[:2]
    x, y, bw, bh = tag_box
    strip_h = max(6, int(h * NAME_STRIP_HEIGHT_RATIO))
    gap = max(1, int(h * NAME_STRIP_GAP_RATIO))
    y2 = max(0, y - gap)
    y1 = max(0, y2 - strip_h)
    pad = int(bw * 0.15)
    x1 = max(0, x - pad)
    x2 = min(w, x + bw + pad)
    if y2 <= y1 or x2 <= x1:
        return None, (x1, y1, x2 - x1, y2 - y1)
    return band_bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def analyze_band_labels(band_bgr, sim_threshold=DEFAULT_LABEL_SIM_THRESHOLD):
    results = []
    if not OCR_AVAILABLE:
        return results

    tag_boxes = detect_tag_boxes(band_bgr)
    if not tag_boxes:
        return results

    widths = [b[2] for b in tag_boxes]
    median_w = float(np.median(widths)) if widths else 0.0

    gaps = []
    for i in range(len(tag_boxes) - 1):
        x1_end = tag_boxes[i][0] + tag_boxes[i][2]
        x2_start = tag_boxes[i + 1][0]
        gap_w = x2_start - x1_end
        if median_w > 0 and gap_w > median_w * 1.3:
            gaps.append(
                (
                    x1_end,
                    tag_boxes[i][1],
                    gap_w,
                    tag_boxes[i][3],
                )
            )

    for box in tag_boxes:
        x, y, bw, bh = box
        tag_crop = band_bgr[y : y + bh, x : x + bw]
        tag_text = normalize_ocr_text(ocr_with_fallback(tag_crop))

        name_crop, name_box = extract_name_strip_above(band_bgr, box)
        name_text = normalize_ocr_text(ocr_with_fallback(name_crop))

        if not tag_text:
            durum = "ETIKET_EKSIK"
            sim = 0.0
        elif not name_text:
            durum = "SUPHELI"
            sim = 0.0
        else:
            sim = text_similarity(tag_text, name_text)
            durum = "UYUMLU" if sim >= sim_threshold else "UYUMSUZ"

        results.append(
            {
                "durum": durum,
                "tag_box": box,
                "name_box": name_box,
                "tag_text": tag_text,
                "name_text": name_text,
                "benzerlik": round(sim, 2),
            }
        )

    for gx, gy, gw, gh in gaps:
        results.append(
            {
                "durum": "ETIKET_EKSIK",
                "tag_box": (gx, gy, gw, gh),
                "name_box": None,
                "tag_text": "",
                "name_text": "",
                "benzerlik": 0.0,
            }
        )

    return results


def draw_label_results(result_img, band_results, x_offset, y_offset):
    for item in band_results:
        durum = item["durum"]
        tx, ty, tw, th = item["tag_box"]
        tag_pt1 = (x_offset + tx, y_offset + ty)
        tag_pt2 = (x_offset + tx + tw, y_offset + ty + th)

        if durum == "UYUMSUZ":
            color = (0, 0, 255)
            label = "ETIKET UYUSMUYOR"
        elif durum == "ETIKET_EKSIK":
            color = (0, 140, 255)
            label = "ETİKET EKSİK"
        else:
            continue

        cv2.rectangle(result_img, tag_pt1, tag_pt2, color, 2)
        cv2.putText(
            result_img,
            label,
            (tag_pt1[0], max(12, tag_pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

        name_box = item.get("name_box")
        if name_box is not None and durum == "UYUMSUZ":
            nx, ny, nw, nh = name_box
            name_pt1 = (x_offset + nx, y_offset + ny)
            name_pt2 = (x_offset + nx + nw, y_offset + ny + nh)
            cv2.rectangle(result_img, name_pt1, name_pt2, color, 2)

            arrow_x = x_offset + nx + nw // 2
            arrow_top = name_pt2[1]
            arrow_bottom = tag_pt1[1]
            if arrow_bottom > arrow_top:
                cv2.arrowedLine(
                    result_img,
                    (arrow_x, arrow_top),
                    (arrow_x, arrow_bottom),
                    color,
                    2,
                    tipLength=0.35,
                )
                cv2.arrowedLine(
                    result_img,
                    (arrow_x, arrow_bottom),
                    (arrow_x, arrow_top),
                    color,
                    2,
                    tipLength=0.35,
                )


def split_bands(roi_top_px, roi_bottom_px, boundaries_ratio):
    total = roi_bottom_px - roi_top_px
    cuts = [roi_top_px]
    for r in boundaries_ratio:
        cuts.append(roi_top_px + int(total * r))
    cuts.append(roi_bottom_px)

    bands = []
    for i in range(len(cuts) - 1):
        top = max(roi_top_px, min(roi_bottom_px, cuts[i]))
        bottom = max(roi_top_px, min(roi_bottom_px, cuts[i + 1]))
        if bottom > top:
            bands.append((top, bottom))
    return bands


def analyze_all_bands_labels(
    field_aligned_img,
    bands,
    side_margin_ratio=0.01,
    sim_threshold=DEFAULT_LABEL_SIM_THRESHOLD,
):
    h, w = field_aligned_img.shape[:2]
    margin_x = int(w * side_margin_ratio)

    all_results = []
    for band_top, band_bottom in bands:
        band_crop = field_aligned_img[
            band_top:band_bottom, margin_x : w - margin_x
        ]
        band_results = analyze_band_labels(
            band_crop, sim_threshold=sim_threshold
        )
        for item in band_results:
            item["band_top"] = band_top
            item["band_bottom"] = band_bottom
        all_results.append(
            {
                "band_top": band_top,
                "band_bottom": band_bottom,
                "items": band_results,
                "x_offset": margin_x,
            }
        )
    return all_results


# =========================================================
# RAPOR OLUŞTURUCU
# =========================================================
def build_report(
    dealer,
    results,
    summary,
    label_bands=None,
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
    ]

    if label_bands:
        toplam_uyumsuz = summary.get("etiket_uyumsuz", 0)
        toplam_eksik = summary.get("etiket_eksik", 0)
        lines += [
            "",
            "--- ETİKET / ÜRÜN ADI KONTROLÜ (OCR) ---",
            f"Etiket-Ürün Adı Uyuşmayan Sayısı: {toplam_uyumsuz}",
            f"Eksik Etiket Sayısı: {toplam_eksik}",
        ]
        for band_idx, band in enumerate(label_bands, start=1):
            for item in band["items"]:
                if item["durum"] == "UYUMSUZ":
                    lines.append(
                        f"Raf {band_idx} | UYUMSUZ | "
                        f"Etiket: '{item['tag_text']}' <> "
                        f"Paket: '{item['name_text']}' "
                        f"(benzerlik={item['benzerlik']})"
                    )
                elif item["durum"] == "ETIKET_EKSIK":
                    lines.append(
                        f"Raf {band_idx} | ETİKET EKSİK | "
                        f"Konum X={item['tag_box'][0]}"
                    )

    lines += [
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
    st.markdown("**Bayi Arama ve Seçim**")
    search_term = st.text_input(
        "Bayi ara",
        placeholder="Jandarma, HTC vb. yazın...",
        key="dealer_search_box",
        label_visibility="collapsed",
    )
    
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
    
    selected_raw_dealer = st.selectbox(
        "Bayi",
        options=[""] + dealer_raw_names,
        format_func=lambda x: (
            "Arama sonucu eşleşen bayiyi seçin..."
            if x == ""
            else x
        ),
        label_visibility="collapsed",
    )
    
    dealer_name = ""
    dealer_path = ""
    if selected_raw_dealer in dealer_choices:
        dealer_name = dealer_choices[selected_raw_dealer]["raw_name"]
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
            "Fotoğraf rafın tam sınırına kırpılmamışsa üst sınırı artırarak "
            "bu alanı analiz dışı bırakabilirsiniz."
        )
    roi_range = st.slider(
        "Analiz Edilecek Raf Bölgesi (görüntü yüksekliğinin %'si)",
        min_value=0,
        max_value=100,
        value=(auto_top_pct, 65),
        step=1,
        key=roi_widget_key,
    )
    illumination_normalize = st.checkbox(
        "Işık/Parlaklık Farkını Otomatik Dengele",
        value=True,
    )

    st.divider()
    if OCR_AVAILABLE:
        label_check_enabled = st.checkbox(
            "🏷️ Etiket / Ürün Adı Kontrolünü Etkinleştir (OCR)",
            value=True,
        )
        label_sim_threshold = st.slider(
            "Etiket Eşleşme Hassasiyeti",
            min_value=0.30,
            max_value=0.90,
            value=DEFAULT_LABEL_SIM_THRESHOLD,
            step=0.05,
            disabled=not label_check_enabled,
        )
    else:
        label_check_enabled = False
        label_sim_threshold = DEFAULT_LABEL_SIM_THRESHOLD

roi_top_ratio = roi_range[0] / 100.0
roi_bottom_ratio = roi_range[1] / 100.0

# ---------------------------------------------------------
# 6 RAF SINIRLARINI KALİBRE ET (GERÇEK GÖRSEL PİKSEL ANALİZİ)
# ---------------------------------------------------------
band_widget_key = "band_boundaries_" + (
    dealer_path if dealer_path else "manuel"
)

if label_check_enabled:
    with st.expander(
        "📐 İlk 6 Raf Sınırlarını Kalibre Et (Gerçek Raf Otomatik Tespiti)", expanded=True
    ):
        st.caption(
            "Uygulama, fotoğraftaki gerçek raf hatlarını analiz ederek sınırları otomatik hizalamıştır. "
            "İsterseniz kaydırma çubuklarıyla rafların yerini hassas biçimde ince ayarlayabilirsiniz."
        )
        
        session_key_bounds = f"stored_bounds_{band_widget_key}"
        
        # Eğer henüz hesaplanmadıysa veya görseller değiştiyse gerçek piksel analiziyle bul
        if session_key_bounds not in st.session_state or st.session_state.get("last_dealer") != dealer_path:
            st.session_state["last_dealer"] = dealer_path
            if ref_img is not None:
                h_ref, w_ref = ref_img.shape[:2]
                r_top_px = int(h_ref * roi_top_ratio)
                r_bot_px = int(h_ref * roi_bottom_ratio)
                st.session_state[session_key_bounds] = detect_real_shelf_boundaries(ref_img, r_top_px, r_bot_px, RAF_SAYISI)
            else:
                st.session_state[session_key_bounds] = [round(100 * i / RAF_SAYISI) for i in range(1, RAF_SAYISI)]

        cols = st.columns(RAF_SAYISI - 1)
        current_bounds = st.session_state[session_key_bounds]
        new_bounds = []
        
        for i, col in enumerate(cols):
            with col:
                min_limit = current_bounds[i-1] + 1 if i > 0 else 1
                max_limit = current_bounds[i+1] - 1 if i < len(current_bounds) - 1 else 99
                
                val_default = max(min_limit, min(max_limit, current_bounds[i]))
                
                val = st.slider(
                    f"Raf Sınırı {i + 1}",
                    min_value=min_limit,
                    max_value=max_limit,
                    value=val_default,
                    step=1,
                    key=f"{band_widget_key}_{i}",
                )
                new_bounds.append(val)
        
        corrected_bounds = []
        last_val = 0
        for b in new_bounds:
            safe_b = max(b, last_val + 1)
            corrected_bounds.append(safe_b)
            last_val = safe_b
            
        st.session_state[session_key_bounds] = corrected_bounds
        band_bounds_pct = corrected_bounds
else:
    band_bounds_pct = [round(100 * i / RAF_SAYISI) for i in range(1, RAF_SAYISI)]

band_boundaries_ratio = []
roi_span_pct = max(1, (roi_range[1] - roi_range[0]))
for pct in band_bounds_pct:
    rel = (pct - roi_range[0]) / roi_span_pct
    band_boundaries_ratio.append(max(0.0, min(1.0, rel)))
band_boundaries_ratio = sorted(set(band_boundaries_ratio))

ready = (
    ref_img is not None
    and field_img is not None
    and roi_bottom_ratio > roi_top_ratio
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

    with st.spinner("İlk 6 raf için gerçek oranlarla analiz yapılıyor..."):
        try:
            result_img, results, summary, aligned_field = (
                analyze_planogram_grid_free(
                    ref_img,
                    field_img,
                    roi_top_ratio=roi_top_ratio,
                    roi_bottom_ratio=roi_bottom_ratio,
                    illumination_normalize=illumination_normalize,
                )
            )

            h_aligned = aligned_field.shape[0]
            roi_top_px = int(h_aligned * roi_top_ratio)
            roi_bottom_px = int(h_aligned * roi_bottom_ratio)

            bands = split_bands(
                roi_top_px, roi_bottom_px, band_boundaries_ratio
            )

            for band_top, band_bottom in bands:
                cv2.rectangle(
                    result_img,
                    (2, band_top),
                    (result_img.shape[1] - 2, band_bottom),
                    (0, 0, 255),
                    2,
                )

            label_bands = []
            etiket_uyumsuz = 0
            etiket_eksik = 0
            if label_check_enabled and OCR_AVAILABLE:
                with st.spinner(
                    "Etiket / ürün adı OCR kontrolü yapılıyor..."
                ):
                    label_bands = analyze_all_bands_labels(
                        aligned_field,
                        bands,
                        sim_threshold=label_sim_threshold,
                    )
                    for band in label_bands:
                        draw_label_results(
                            result_img,
                            band["items"],
                            band["x_offset"],
                            band["band_top"],
                        )
                        for item in band["items"]:
                            if item["durum"] == "UYUMSUZ":
                                etiket_uyumsuz += 1
                            elif item["durum"] == "ETIKET_EKSIK":
                                etiket_eksik += 1

            summary["etiket_uyumsuz"] = etiket_uyumsuz
            summary["etiket_eksik"] = etiket_eksik

            report = build_report(
                dealer_name or "Manuel",
                results,
                summary,
                label_bands=label_bands,
            )

            st.session_state.result_img = result_img
            st.session_state.results = results
            st.session_state.summary = summary
            st.session_state.report = report
        except Exception as exc:
            st.session_state.result_img = None
            st.session_state.results = []
            st.session_state.summary = None
            st.session_state.report = ""
            st.error(
                "Analiz sırasında bir hata oluştu.\n\nDetay: " + str(exc)
            )


# =========================================================
# SONUÇ EKRANI
# =========================================================
if (
    st.session_state.result_img is not None
    and st.session_state.summary
):
    summary = st.session_state.summary

    m1, m2, m3 = st.columns(3)
    m1.metric("Eksik Paket / Fark", summary.get("paket_eksigi", 0))
    m2.metric(
        "Etiket-Ürün Uyuşmazlığı",
        summary.get("etiket_uyumsuz", 0),
    )
    m3.metric("Eksik Etiket", summary.get("etiket_eksik", 0))

    st.image(
        st.session_state.result_img,
        channels="BGR",
        use_container_width=True,
    )

    d1, d2 = st.columns(2)

    ok, encoded = cv2.imencode(
        ".jpg",
        st.session_state.result_img,
    )
    if ok:
        with d1:
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

    if st.session_state.report:
        with d2:
            st.download_button(
                "📄 Metin Raporunu İndir",
                data=st.session_state.report.encode("utf-8"),
                file_name=(
                    f"{(dealer_name or 'planogram').replace(' ', '_')}"
                    "_ilk6raf_rapor.txt"
                ),
                mime="text/plain",
                use_container_width=True,
            )

    with st.expander("📄 Metin Raporu", expanded=False):
        st.text(st.session_state.report)

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
