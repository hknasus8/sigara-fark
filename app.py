# -*- coding: utf-8 -*-
"""
SIGARA STANDI PLANOGRAM DENETİM SİSTEMİ
Robust & Dynamic Streamlit Sürümü
"""

import re
import urllib.parse

import cv2
import numpy as np
import requests
import streamlit as st


# =========================================================
# SAYFA YAPILANDIRMASI
# =========================================================
st.set_page_config(
    page_title="Sigara Standı Planogram Denetim Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
.block-container {padding-top:1rem;padding-bottom:2rem;}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SABİTLER
# =========================================================
YANDEX_ROOT_PUBLIC_KEY = "https://disk.yandex.com.tr/d/ikCHPwREiCVv_g"

DEFAULT_ROWS = 7
DEFAULT_COLS = 11

DIFF_SCORE_THRESHOLD = 0.36
SUSPICIOUS_SCORE_THRESHOLD = 0.27
EMPTY_THRESHOLD = 0.62


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def safe_float(value, default=0.0):
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def clamp01(value):
    return max(0.0, min(1.0, safe_float(value)))


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


def make_result(**kwargs):
    item = {
        "raf": 0,
        "slot": 0,
        "score": 0.0,
        "ssim": 0.0,
        "renk": 0.0,
        "kenar": 0.0,
        "orb": 0.0,
        "pixel": 0.0,
        "bosluk": 0.0,
        "durum": "ŞÜPHELİ",
        "different": False,
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
    }
    item.update(kwargs)
    for key in ("score", "ssim", "renk", "kenar", "orb", "pixel", "bosluk"):
        item[key] = clamp01(item.get(key, 0.0))
    return item


# =========================================================
# GÖRSEL İŞLEMLERİ
# =========================================================
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
        response = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout
        )
        if response.status_code != 200:
            return None
        data = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# =========================================================
# YANDEX API FONKSİYONLARI
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def yandex_root_items(public_key):
    try:
        url = (
            "https://cloud-api.yandex.net/v1/disk/public/resources"
            f"?public_key={urllib.parse.quote(public_key, safe='')}&limit=500"
        )
        response = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20
        )
        if response.status_code != 200:
            return [], f"Yandex HTTP {response.status_code}"
        return (
            response.json().get("_embedded", {}).get("items", []),
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
            f"&path={urllib.parse.quote(path, safe='/')}&limit=500"
        )
        response = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20
        )
        if response.status_code != 200:
            return [], f"Yandex HTTP {response.status_code}"
        return (
            response.json().get("_embedded", {}).get("items", []),
            None,
        )
    except Exception as exc:
        return [], str(exc)


# =========================================================
# GÖRSEL HİZALAMA
# =========================================================
def gray_normalize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def orb_align(reference, target):
    if reference is None or target is None:
        return target, False, 0
    h, w = reference.shape[:2]
    target = cv2.resize(target, (w, h), interpolation=cv2.INTER_AREA)

    ref_gray = gray_normalize(reference)
    tar_gray = gray_normalize(target)

    orb = cv2.ORB_create(
        nfeatures=7000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=8,
    )
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(tar_gray, None)

    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        return target, False, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des2, des1, k=2)

    good = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.76 * n.distance:
                good.append(m)

    if len(good) < 15:
        return target, False, len(good)

    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if matrix is None or mask is None:
        return target, False, 0

    inliers = int(mask.sum())
    ratio = inliers / max(1, len(good))
    if inliers < 12 or ratio < 0.22:
        return target, False, inliers

    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    if not np.all(np.isfinite(warped_corners)):
        return target, False, inliers

    area = cv2.contourArea(warped_corners.astype(np.float32))
    if area < w * h * 0.35 or area > w * h * 2.8:
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
    target = cv2.resize(target, (w, h), interpolation=cv2.INTER_AREA)

    ref_gray = (
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    )
    tar_gray = (
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    )

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )

    try:
        cv2.findTransformECC(
            ref_gray, tar_gray, warp, cv2.MOTION_AFFINE, criteria, None, 3
        )
        aligned = cv2.warpAffine(
            target,
            warp,
            (w, h),
            flags=(cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP),
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, True
    except Exception:
        return target, False


def align_images(reference, target):
    aligned, ok, inliers = orb_align(reference, target)
    if ok:
        return aligned, True, inliers, "ORB/RANSAC"

    aligned, ok = ecc_align(reference, target)
    if ok:
        return aligned, True, 0, "ECC"

    return target, False, 0, "Ölçek eşitleme"


# =========================================================
# DİNAMİK SLOT GEOMETRİSİ
# =========================================================
def grid_boxes(h, w, rows, cols):
    boxes = []
    for row in range(rows):
        y1 = int(round(row * h / rows))
        y2 = int(round((row + 1) * h / rows))
        for col in range(cols):
            x1 = int(round(col * w / cols))
            x2 = int(round((col + 1) * w / cols))
            boxes.append((row + 1, col + 1, x1, y1, x2, y2))
    return boxes


def crop_slot(img, box, x_pad=0.04, y_pad=0.10):
    _, _, x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    px = max(2, int(width * x_pad))
    py = max(2, int(height * y_pad))

    return img[y1 + py : y2 - py, x1 + px : x2 - px]


# =========================================================
# SLOT METRİKLERİ VE ANALİZİ
# =========================================================
def resize_gray(slot, size=(180, 150)):
    if slot is None or slot.size == 0:
        return None
    gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def structural_similarity(a, b):
    a = resize_gray(a)
    b = resize_gray(b)
    if a is None or b is None:
        return 0.0

    af, bf = a.astype(np.float32), b.astype(np.float32)
    mu_a = cv2.GaussianBlur(af, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(bf, (11, 11), 1.5)

    sigma_a = cv2.GaussianBlur(af * af, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(bf * bf, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(af * bf, (11, 11), 1.5) - mu_a * mu_b

    c1, c2 = 6.5025, 58.5225
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (
        sigma_a + sigma_b + c2
    )

    score = np.mean(numerator / (denominator + 1e-8))
    return clamp01((score + 1.0) / 2.0)


def color_similarity(a, b):
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0

    a = cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (64, 64), interpolation=cv2.INTER_AREA)

    hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)

    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [24, 16], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [24, 16], [0, 180, 0, 256])

    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return clamp01((corr + 1.0) / 2.0)


def edge_similarity(a, b):
    ga, gb = resize_gray(a), resize_gray(b)
    if ga is None or gb is None:
        return 0.0

    ea = cv2.Canny(ga, 50, 140)
    eb = cv2.Canny(gb, 50, 140)

    intersection = np.logical_and(ea > 0, eb > 0).sum()
    union = np.logical_or(ea > 0, eb > 0).sum()
    if union == 0:
        return 1.0
    return clamp01(intersection / union)


def orb_similarity(a, b):
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    ga, gb = resize_gray(a, (220, 180)), resize_gray(b, (220, 180))

    orb = cv2.ORB_create(nfeatures=600, fastThreshold=10)
    k1, d1 = orb.detectAndCompute(ga, None)
    k2, d2 = orb.detectAndCompute(gb, None)

    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(d1, d2, k=2)

    good, total = 0, 0
    for pair in pairs:
        if len(pair) == 2:
            total += 1
            m, n = pair
            if m.distance < 0.78 * n.distance:
                good += 1

    return clamp01(
        good / max(8.0, min(len(k1), len(k2)) * 0.30)
    )


def occupancy(slot):
    if slot is None or slot.size == 0:
        return 0.0
    gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (120, 160), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(gray, 45, 130)
    edge_density = float(np.mean(edges > 0))
    mean_value = float(np.mean(gray)) / 255.0
    dark = 1.0 - mean_value

    return clamp01(0.55 * edge_density / 0.22 + 0.45 * dark)


def compare_slot(ref_slot, current_slot):
    if (
        ref_slot is None
        or current_slot is None
        or ref_slot.size == 0
        or current_slot.size == 0
    ):
        return make_result(score=1.0, durum="ŞÜPHELİ", different=True)

    ssim = structural_similarity(ref_slot, current_slot)
    renk = color_similarity(ref_slot, current_slot)
    kenar = edge_similarity(ref_slot, current_slot)
    orb = orb_similarity(ref_slot, current_slot)

    a = cv2.resize(ref_slot, (160, 180), interpolation=cv2.INTER_AREA)
    b = cv2.resize(current_slot, (160, 180), interpolation=cv2.INTER_AREA)

    ga = cv2.normalize(
        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )
    gb = cv2.normalize(
        cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    pixel_diff = clamp01(
        float(np.mean(cv2.absdiff(ga, gb))) / 70.0
    )
    structural_diff = 1.0 - ssim
    color_diff = 1.0 - renk
    edge_diff = 1.0 - kenar
    orb_diff = 1.0 - orb

    occ_a = occupancy(ref_slot)
    occ_b = occupancy(current_slot)
    empty_diff = clamp01(abs(occ_a - occ_b) / 0.45)

    score = clamp01(
        structural_diff * 0.34
        + color_diff * 0.18
        + edge_diff * 0.20
        + orb_diff * 0.10
        + pixel_diff * 0.10
        + empty_diff * 0.08
    )

    strong_change = (
        structural_diff > 0.43
        or color_diff > 0.40
        or edge_diff > 0.48
    )

    if score >= DIFF_SCORE_THRESHOLD and strong_change:
        status = "FARK"
        different = True
    elif score >= SUSPICIOUS_SCORE_THRESHOLD or empty_diff > EMPTY_THRESHOLD:
        status = "ŞÜPHELİ"
        different = True
    else:
        status = "UYUMLU"
        different = False

    return make_result(
        score=score,
        ssim=ssim,
        renk=renk,
        kenar=kenar,
        orb=orb,
        pixel=pixel_diff,
        bosluk=empty_diff,
        durum=status,
        different=different,
    )


# =========================================================
# ANA PLANOGRAM ANALİZİ (DİNAMİK)
# =========================================================
def analyze_planogram(reference, field, rows, cols):
    h, w = reference.shape[:2]
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_AREA)

    aligned, aligned_ok, inliers, method = align_images(reference, field)

    results = []
    result_img = aligned.copy()
    boxes = grid_boxes(h, w, rows, cols)

    for box in boxes:
        row, col, x1, y1, x2, y2 = box
        ref_slot = crop_slot(reference, box)
        current_slot = crop_slot(aligned, box)

        metrics = compare_slot(ref_slot, current_slot)
        metrics.update(
            {
                "raf": row,
                "slot": col,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
        metrics = make_result(**metrics)
        results.append(metrics)

        if metrics["durum"] == "FARK":
            color, thickness = (0, 0, 255), 4
        elif metrics["durum"] == "ŞÜPHELİ":
            color, thickness = (0, 165, 255), 3
        else:
            color, thickness = (0, 180, 0), 2

        cv2.rectangle(
            result_img,
            (x1 + 2, y1 + 2),
            (x2 - 2, y2 - 2),
            color,
            thickness,
        )
        label = f"R{row}/S{col} %{metrics['score'] * 100:.0f}"
        cv2.putText(
            result_img,
            label,
            (x1 + 6, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    fark = sum(1 for x in results if x["durum"] == "FARK")
    supheli = sum(1 for x in results if x["durum"] == "ŞÜPHELİ")
    uyumlu = sum(1 for x in results if x["durum"] == "UYUMLU")

    header_height = max(72, int(h * 0.055))
    cv2.rectangle(result_img, (0, 0), (w, header_height), (18, 18, 18), -1)

    header1 = f"FARK: {fark} | SUPHELI: {supheli} | UYUMLU: {uyumlu}"
    header2 = f"Hizalama: {method} | Inlier: {inliers}"

    cv2.putText(
        result_img,
        header1,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result_img,
        header2,
        (14, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return (
        result_img,
        results,
        {
            "fark": fark,
            "supheli": supheli,
            "uyumlu": uyumlu,
            "hizalama_ok": aligned_ok,
            "hizalama": method,
            "inliers": inliers,
        },
    )


# =========================================================
# RAPOR OLUŞTURUCU (DİNAMİK KAPASİTE)
# =========================================================
def build_report(dealer, results, summary, capacity):
    from datetime import datetime

    lines = [
        "=== SİGARA STANDI PLANOGRAM DENETİM RAPORU ===",
        f"Bayi: {dealer}",
        f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
        "",
        f"Toplam slot / kapasite: {capacity}",
        f"FARK: {summary['fark']}",
        f"ŞÜPHELİ: {summary['supheli']}",
        f"UYUMLU: {summary['uyumlu']}",
        f"Hizalama: {summary['hizalama']} / inlier={summary['inliers']}",
        "",
        "--- SLOT DETAYI ---",
    ]

    for item in results:
        lines.append(
            f"R{item.get('raf', 0)}/S{item.get('slot', 0)} | "
            f"Durum={item.get('durum', 'ŞÜPHELİ')} | "
            f"Skor=%{safe_float(item.get('score')) * 100:.1f}"
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
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# GİRİŞ KONTROLÜ
# =========================================================
try:
    app_password = st.secrets["app_password"]
except Exception:
    app_password = None

if not st.session_state.authenticated:
    st.title("🔐 Kurumsal Planogram Denetim Sistemi")
    st.caption("Güvenli giriş")

    if not app_password:
        st.error("⚠️ Streamlit Secrets içinde app_password tanımlanmamış.")

    password = st.text_input("Şifre", type="password")
    if st.button("Giriş Yap", type="primary", use_container_width=True):
        if app_password and password == str(app_password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre.")
    st.stop()


# =========================================================
# SIDEBAR (DİNAMİK KONTROLLER)
# =========================================================
with st.sidebar:
    st.header("⚙️ Denetim Ayarları")

    rows = st.number_input(
        "Raf sayısı", min_value=1, max_value=20, value=DEFAULT_ROWS, step=1
    )
    cols = st.number_input(
        "Slot / kolon sayısı", min_value=1, max_value=30, value=DEFAULT_COLS, step=1
    )

    capacity = rows * cols

    st.caption(
        f"Dinamik Geometri: {rows} Raf × {cols} Kolon = {capacity} Toplam Slot. "
        "Slot sayısı gerçek SKU/stok adedi değildir."
    )

    if st.button("🚪 Çıkış Yap", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# =========================================================
# ANA ARAYÜZ (DİNAMİK VE TABLOSUZ)
# =========================================================
st.title("📊 SİGARA STANDI PLANOGRAM DENETİM SİSTEMİ")
st.caption("Referans planogram ↔ saha fotoğrafı karşılaştırma motoru")

st.markdown("### 1. Görseller")
col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader("Dijital Planogram / Referans", type=["jpg", "jpeg", "png", "webp"])

with col2:
    field_file = st.file_uploader("Saha Fotoğrafı", type=["jpg", "jpeg", "png", "webp"])

if ref_file and field_file:
    ref_img = decode_uploaded(ref_file)
    field_img = decode_uploaded(field_file)

    if ref_img is not None and field_img is not None:
        reference_img = prepare_image(ref_img)
        field_target = prepare_image(field_img)

        if st.button("🔍 Analizi Başlat", type="primary", use_container_width=True):
            with st.spinner("Görsel işleniyor ve eşleştiriliyor..."):
                res_img, results, summary = analyze_planogram(
                    reference_img, field_target, rows=rows, cols=cols
                )
                st.session_state.result_img = res_img
                st.session_state.results = results
                st.session_state.summary = summary
                st.session_state.report = build_report("Manuel Bayi", results, summary, capacity)

        if st.session_state.result_img is not None:
            st.markdown("### 3. Analiz Sonucu")
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("🔴 FARK", st.session_state.summary["fark"])
            sc2.metric("🟠 ŞÜPHELİ", st.session_state.summary["supheli"])
            sc3.metric("🟢 UYUMLU", st.session_state.summary["uyumlu"])

            total_slots = capacity
            matched_slots = st.session_state.summary["uyumlu"]
            compliance_rate = (matched_slots / max(1, total_slots)) * 100
            st.metric("📊 Uyum Oranı", f"%{compliance_rate:.1f}")

            st.image(st.session_state.result_img, channels="BGR", use_container_width=True)

            st.markdown("### 6. Rapor")
            st.text_area("Rapor Çıktısı", st.session_state.report, height=200)

st.markdown("---")
st.caption("Developed by Hakan")
