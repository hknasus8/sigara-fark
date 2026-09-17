# -*- coding: utf-8 -*-
"""
SIGARA STANDI PLANOGRAM DENETİM SİSTEMİ
Robust Streamlit sürümü

Kurulum:
    pip install streamlit opencv-python-headless numpy requests pillow

Streamlit Cloud Secrets:
    app_password = "SIFRENIZ"

Yandex klasör yapısı için mevcut public key korunmuştur.
"""

import hashlib
import re
import urllib.parse

import cv2
import numpy as np
import requests
import streamlit as st


# =========================================================
# SAYFA
# =========================================================
st.set_page_config(
    page_title="Sigara Standı Denetim Sistemi",
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

# Denetim yalnızca ilk N rafı kapsar (talep: sadece ilk 6 raf kontrol edilecek).
ANALYZE_ROW_LIMIT = 6

# Fark karar eşikleri.
DIFF_SCORE_THRESHOLD = 0.36
SUSPICIOUS_SCORE_THRESHOLD = 0.27
EMPTY_THRESHOLD = 0.62

BRANDS = [
    "CAMEL",
    "WINSTON",
    "LD",
    "MONTE CARLO",
    "PARLIAMENT",
    "MARLBORO",
    "MURATTI",
    "LARK",
    "CHESTERFIELD",
    "KENT",
    "ROTHMANS",
    "DAVIDOFF",
]


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
    """
    Analiz sonucundaki bütün alanları garanti eder.
    Böylece item['renk'], item['ssim'] vb. KeyError oluşturmaz.
    """
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

    for key in (
        "score",
        "ssim",
        "renk",
        "kenar",
        "orb",
        "pixel",
        "bosluk",
    ):
        item[key] = clamp01(item.get(key, 0.0))

    return item


# =========================================================
# GÖRSEL OKUMA
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
# YANDEX
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

            if normalize_text(item.get("name")) != "BAYI":
                continue

            sub_items, _ = yandex_list_dir(
                public_key,
                item.get("path", ""),
            )

            for sub in sub_items:
                if (
                    sub.get("type") == "dir"
                    and normalize_text(sub.get("name"))
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
            dealers.append(
                {
                    "name": item["name"],
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
# HİZALAMA
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
    """
    ORB + RANSAC.
    Saha fotoğrafındaki perspektif ve konum farkını azaltır.
    """
    if reference is None or target is None:
        return target, False, 0

    h, w = reference.shape[:2]

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

    if des1 is None or des2 is None:
        return target, False, 0

    if len(kp1) < 15 or len(kp2) < 15:
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

    corners = np.float32(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ]
    ).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(
        corners,
        matrix,
    ).reshape(-1, 2)

    if not np.all(
        np.isfinite(warped_corners)
    ):
        return target, False, inliers

    area = cv2.contourArea(
        warped_corners.astype(np.float32)
    )

    if (
        area < w * h * 0.35
        or area > w * h * 2.8
    ):
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
    """
    ORB başarısız olduğunda affine ECC hizalama.
    """
    h, w = reference.shape[:2]

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

    warp = np.eye(
        2,
        3,
        dtype=np.float32,
    )

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
        return (
            aligned,
            True,
            0,
            "ECC",
        )

    return (
        target,
        False,
        0,
        "Ölçek eşitleme",
    )


# =========================================================
# 7 x 11 SLOT GEOMETRİSİ
# =========================================================
def grid_boxes(
    h,
    w,
    rows=DEFAULT_ROWS,
    cols=DEFAULT_COLS,
):
    boxes = []

    for row in range(rows):
        y1 = int(
            round(row * h / rows)
        )
        y2 = int(
            round((row + 1) * h / rows)
        )

        for col in range(cols):
            x1 = int(
                round(col * w / cols)
            )
            x2 = int(
                round((col + 1) * w / cols)
            )

            boxes.append(
                (
                    row + 1,
                    col + 1,
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

    return boxes


def crop_slot(
    img,
    box,
    x_pad=0.04,
    y_pad=0.10,
):
    _, _, x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    px = max(
        2,
        int(width * x_pad),
    )
    py = max(
        2,
        int(height * y_pad),
    )

    a = img[
        y1 + py:y2 - py,
        x1 + px:x2 - px,
    ]

    return a


# =========================================================
# SLOT METRİKLERİ
# =========================================================
def resize_gray(
    slot,
    size=(180, 150),
):
    if (
        slot is None
        or slot.size == 0
    ):
        return None

    gray = cv2.cvtColor(
        slot,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        size,
        interpolation=cv2.INTER_AREA,
    )

    return cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )


def structural_similarity(a, b):
    a = resize_gray(a)
    b = resize_gray(b)

    if a is None or b is None:
        return 0.0

    af = a.astype(np.float32)
    bf = b.astype(np.float32)

    mu_a = cv2.GaussianBlur(
        af,
        (11, 11),
        1.5,
    )
    mu_b = cv2.GaussianBlur(
        bf,
        (11, 11),
        1.5,
    )

    sigma_a = (
        cv2.GaussianBlur(
            af * af,
            (11, 11),
            1.5,
        )
        - mu_a * mu_a
    )

    sigma_b = (
        cv2.GaussianBlur(
            bf * bf,
            (11, 11),
            1.5,
        )
        - mu_b * mu_b
    )

    sigma_ab = (
        cv2.GaussianBlur(
            af * bf,
            (11, 11),
            1.5,
        )
        - mu_a * mu_b
    )

    c1 = 6.5025
    c2 = 58.5225

    numerator = (
        (2 * mu_a * mu_b + c1)
        * (2 * sigma_ab + c2)
    )

    denominator = (
        (mu_a * mu_a + mu_b * mu_b + c1)
        * (sigma_a + sigma_b + c2)
    )

    score = np.mean(
        numerator
        / (denominator + 1e-8)
    )

    return clamp01(
        (score + 1.0) / 2.0
    )


def color_similarity(a, b):
    if (
        a is None
        or b is None
        or a.size == 0
        or b.size == 0
    ):
        return 0.0

    a = cv2.resize(
        a,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )
    b = cv2.resize(
        b,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )

    hsv_a = cv2.cvtColor(
        a,
        cv2.COLOR_BGR2HSV,
    )
    hsv_b = cv2.cvtColor(
        b,
        cv2.COLOR_BGR2HSV,
    )

    hist_a = cv2.calcHist(
        [hsv_a],
        [0, 1],
        None,
        [24, 16],
        [0, 180, 0, 256],
    )

    hist_b = cv2.calcHist(
        [hsv_b],
        [0, 1],
        None,
        [24, 16],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        hist_a,
        hist_a,
    )
    cv2.normalize(
        hist_b,
        hist_b,
    )

    corr = cv2.compareHist(
        hist_a,
        hist_b,
        cv2.HISTCMP_CORREL,
    )

    return clamp01(
        (corr + 1.0) / 2.0
    )


def edge_similarity(a, b):
    ga = resize_gray(a)
    gb = resize_gray(b)

    if ga is None or gb is None:
        return 0.0

    ea = cv2.Canny(
        ga,
        50,
        140,
    )
    eb = cv2.Canny(
        gb,
        50,
        140,
    )

    intersection = np.logical_and(
        ea > 0,
        eb > 0,
    ).sum()

    union = np.logical_or(
        ea > 0,
        eb > 0,
    ).sum()

    if union == 0:
        return 1.0

    return clamp01(
        intersection / union
    )


def orb_similarity(a, b):
    if (
        a is None
        or b is None
        or a.size == 0
        or b.size == 0
    ):
        return 0.0

    ga = resize_gray(
        a,
        (220, 180),
    )
    gb = resize_gray(
        b,
        (220, 180),
    )

    orb = cv2.ORB_create(
        nfeatures=600,
        fastThreshold=10,
    )

    k1, d1 = orb.detectAndCompute(
        ga,
        None,
    )
    k2, d2 = orb.detectAndCompute(
        gb,
        None,
    )

    if (
        d1 is None
        or d2 is None
        or len(k1) < 4
        or len(k2) < 4
    ):
        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    pairs = matcher.knnMatch(
        d1,
        d2,
        k=2,
    )

    good = 0
    total = 0

    for pair in pairs:
        if len(pair) != 2:
            continue

        total += 1

        m, n = pair

        if m.distance < 0.78 * n.distance:
            good += 1

    return clamp01(
        good
        / max(
            8.0,
            min(
                len(k1),
                len(k2),
            )
            * 0.30,
        )
    )


def occupancy(slot):
    """
    Slot içindeki ürün/boşluk yapısını yaklaşık ölçer.
    Gerçek stok adedi değildir.
    """
    if (
        slot is None
        or slot.size == 0
    ):
        return 0.0

    gray = cv2.cvtColor(
        slot,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        (120, 160),
        interpolation=cv2.INTER_AREA,
    )

    edges = cv2.Canny(
        gray,
        45,
        130,
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    mean_value = (
        float(np.mean(gray))
        / 255.0
    )

    dark = 1.0 - mean_value

    return clamp01(
        0.55 * edge_density / 0.22
        + 0.45 * dark
    )


def compare_slot(
    ref_slot,
    current_slot,
):
    if (
        ref_slot is None
        or current_slot is None
        or ref_slot.size == 0
        or current_slot.size == 0
    ):
        return make_result(
            score=1.0,
            durum="ŞÜPHELİ",
            different=True,
        )

    ssim = structural_similarity(
        ref_slot,
        current_slot,
    )

    renk = color_similarity(
        ref_slot,
        current_slot,
    )

    kenar = edge_similarity(
        ref_slot,
        current_slot,
    )

    orb = orb_similarity(
        ref_slot,
        current_slot,
    )

    # Normalize edilmiş ham piksel farkı.
    a = cv2.resize(
        ref_slot,
        (160, 180),
        interpolation=cv2.INTER_AREA,
    )
    b = cv2.resize(
        current_slot,
        (160, 180),
        interpolation=cv2.INTER_AREA,
    )

    ga = cv2.cvtColor(
        a,
        cv2.COLOR_BGR2GRAY,
    )
    gb = cv2.cvtColor(
        b,
        cv2.COLOR_BGR2GRAY,
    )

    ga = cv2.normalize(
        ga,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )
    gb = cv2.normalize(
        gb,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    pixel_diff = clamp01(
        float(
            np.mean(
                cv2.absdiff(
                    ga,
                    gb,
                )
            )
        )
        / 70.0
    )

    structural_diff = 1.0 - ssim
    color_diff = 1.0 - renk
    edge_diff = 1.0 - kenar
    orb_diff = 1.0 - orb

    occ_a = occupancy(ref_slot)
    occ_b = occupancy(current_slot)

    empty_diff = clamp01(
        abs(occ_a - occ_b) / 0.45
    )

    # Birleşik fark skoru.
    score = (
        structural_diff * 0.34
        + color_diff * 0.18
        + edge_diff * 0.20
        + orb_diff * 0.10
        + pixel_diff * 0.10
        + empty_diff * 0.08
    )

    score = clamp01(score)

    strong_change = (
        structural_diff > 0.43
        or color_diff > 0.40
        or edge_diff > 0.48
    )

    if (
        score >= DIFF_SCORE_THRESHOLD
        and strong_change
    ):
        status = "FARK"
        different = True

    elif (
        score >= SUSPICIOUS_SCORE_THRESHOLD
        or empty_diff > EMPTY_THRESHOLD
    ):
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
# ANA PLANOGRAM ANALİZİ
# =========================================================
def analyze_planogram(
    reference,
    field,
    rows=DEFAULT_ROWS,
    cols=DEFAULT_COLS,
    max_check_rows=ANALYZE_ROW_LIMIT,
):
    h, w = reference.shape[:2]

    field = cv2.resize(
        field,
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    aligned, aligned_ok, inliers, method = (
        align_images(
            reference,
            field,
        )
    )

    results = []
    result_img = aligned.copy()

    boxes = grid_boxes(
        h,
        w,
        rows,
        cols,
    )

    for box in boxes:
        row, col, x1, y1, x2, y2 = box

        if max_check_rows and row > max_check_rows:
            continue

        ref_slot = crop_slot(
            reference,
            box,
        )

        current_slot = crop_slot(
            aligned,
            box,
        )

        metrics = compare_slot(
            ref_slot,
            current_slot,
        )

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

        metrics = make_result(
            **metrics
        )

        results.append(metrics)

        # GÜNCELLENDİ: Hem "FARK" hem de "ŞÜPHELİ" (tüm tespit edilen değişimler) KIRMIZI çerçeve içine alınır.
        if metrics["durum"] in ["FARK", "ŞÜPHELİ"]:
            color = (0, 0, 255)  # Kırmızı
            thickness = 3

            cv2.rectangle(
                result_img,
                (x1 + 2, y1 + 2),
                (x2 - 2, y2 - 2),
                color,
                thickness,
            )

            label = (
                f"R{row}/S{col} "
                f"%{metrics['score'] * 100:.0f}"
            )

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

    fark = sum(
        1
        for x in results
        if x["durum"] == "FARK"
    )

    supheli = sum(
        1
        for x in results
        if x["durum"] == "ŞÜPHELİ"
    )

    uyumlu = sum(
        1
        for x in results
        if x["durum"] == "UYUMLU"
    )

    header_height = max(
        72,
        int(h * 0.055),
    )

    cv2.rectangle(
        result_img,
        (0, 0),
        (w, header_height),
        (18, 18, 18),
        -1,
    )

    header1 = (
        f"FARK: {fark} | "
        f"SUPHELI: {supheli} | "
        f"UYUMLU: {uyumlu}"
    )

    header2 = (
        f"Hizalama: {method} | "
        f"Inlier: {inliers} | "
        f"Kontrol edilen raf: ilk {max_check_rows}"
    )

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
            "kontrol_edilen_raf": max_check_rows,
            "hizalama_ok": aligned_ok,
            "hizalama": method,
            "inliers": inliers,
        },
    )


# =========================================================
# RAPOR
# =========================================================
def build_report(
    dealer,
    results,
    summary,
    capacity,
):
    from datetime import datetime

    lines = [
        "=== SİGARA STANDI PLANOGRAM DENETİM RAPORU ===",
        f"Bayi: {dealer}",
        (
            "Tarih: "
            + datetime.now().strftime(
                "%d.%m.%Y %H:%M:%S"
            )
        ),
        "",
        f"Kontrol edilen raf: ilk "
        f"{summary.get('kontrol_edilen_raf', ANALYZE_ROW_LIMIT)}",
        f"Toplam slot: {len(results)}",
        f"FARK: {summary['fark']}",
        f"ŞÜPHELİ: {summary['supheli']}",
        f"UYUMLU: {summary['uyumlu']}",
        f"Tanımlı kapasite: {capacity}",
        (
            f"Hizalama: {summary['hizalama']} "
            f"/ inlier={summary['inliers']}"
        ),
        "",
        "--- SLOT DETAYI ---",
    ]

    for item in results:
        lines.append(
            f"R{item.get('raf', 0)}/"
            f"S{item.get('slot', 0)} | "
            f"Durum={item.get('durum', 'ŞÜPHELİ')} | "
            f"Skor=%{safe_float(item.get('score')) * 100:.1f} | "
            f"Yapı=%{safe_float(item.get('ssim')) * 100:.1f} | "
            f"Renk=%{safe_float(item.get('renk')) * 100:.1f} | "
            f"Kenar=%{safe_float(item.get('kenar')) * 100:.1f} | "
            f"ORB=%{safe_float(item.get('orb')) * 100:.1f}"
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
# GİRİŞ
# =========================================================
if "app_password" not in st.secrets:
    st.error(
        "Kritik: Streamlit Secrets içine "
        "app_password eklenmemiş.\n\n"
        "Örnek:\n"
        'app_password = "SIFRENIZ"'
    )
    st.stop()

if not st.session_state.authenticated:
    st.title(
        "🔐 Kurumsal Planogram Denetim Sistemi"
    )
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
        if password == str(
            st.secrets["app_password"]
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

    rows = st.number_input(
        "Raf sayısı",
        min_value=1,
        max_value=20,
        value=DEFAULT_ROWS,
        step=1,
    )

    cols = st.number_input(
        "Slot / kolon sayısı",
        min_value=1,
        max_value=30,
        value=DEFAULT_COLS,
        step=1,
    )

    capacity = st.number_input(
        "Toplam ürün/slot kapasitesi",
        min_value=1,
        max_value=1000,
        value=77,
        step=1,
    )

    st.caption(
        "Varsayılan geometri: 7 raf × 11 slot = 77. "
        "Slot sayısı gerçek SKU/stok adedi değildir. "
        f"Denetim, girilen raf sayısı ne olursa olsun yalnızca "
        f"ilk {ANALYZE_ROW_LIMIT} rafı kapsar."
    )

    if st.button(
        "🔄 Önbelleği Temizle",
        use_container_width=True,
    ):
        yandex_root_items.clear()
        yandex_list_dir.clear()
        get_cities.clear()
        get_dealers.clear()
        get_reference_image.clear()

        st.session_state.result_img = None
        st.session_state.results = []
        st.session_state.summary = None
        st.session_state.report = ""

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
# BAŞLIK
# =========================================================
st.title(
    "📊 SİGARA STANDI DENETİM SİSTEMİ"
)


# =========================================================
# LOKASYON / BAYİ
# =========================================================
st.subheader(
    "1. Şehir Seçiniz"
)

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

with c2:
    dealer_name = st.selectbox(
        "Bayi",
        options=[""]
        + [
            x["name"]
            for x in dealers
        ],
        format_func=lambda x: (
            "Bayi seçin..."
            if x == ""
            else x
        ),
    )

dealer_path = ""

if dealer_name:
    for dealer in dealers:
        if dealer["name"] == dealer_name:
            dealer_path = dealer["path"]
            break

if dealer_error:
    st.warning(
        "Bayi listesi: "
        + str(dealer_error)
    )


# =========================================================
# GÖRSELLER
# =========================================================
st.divider()
st.subheader("2. Görseller")

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

    if ref_error:
        st.warning(ref_error)

u1, u2 = st.columns(2)

with u1:
    st.markdown(
        "**Orijinal Referans Fotoğrafı Getirildi**"
        if ref_img is not None
        else "**Orijinal Referans Fotoğrafı**"
    )

    if ref_img is None:
        ref_upload = st.file_uploader(
            "İsterseniz referans fotoğrafını elle yükleyin",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            key="ref_upload",
        )

        if ref_upload is not None:
            ref_img = prepare_image(
                decode_uploaded(
                    ref_upload
                )
            )

    if ref_img is not None:
        st.image(
            ref_img,
            channels="BGR",
            use_container_width=True,
        )
    else:
        st.info(
            "Şehir + bayi seçin veya "
            "referans görseli yükleyin."
        )

with u2:
    st.markdown(
        "**Saha'dan Gelen Fotoğraf**"
    )

    field_upload = st.file_uploader(
        "Saha fotoğrafını yükleyin",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="field_upload",
    )

    field_img = (
        prepare_image(
            decode_uploaded(
                field_upload
            )
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
        st.info(
            "Sahadan gelen fotoğrafı yükleyin."
        )


# =========================================================
# ANALİZ
# =========================================================
st.divider()

ready = (
    ref_img is not None
    and field_img is not None
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

    with st.spinner(
        "Fotoğraflar hizalanıyor ve "
        f"{int(rows)}×{int(cols)} slot analiz ediliyor..."
    ):
        result_img, results, summary = (
            analyze_planogram(
                ref_img,
                field_img,
                rows=int(rows),
                cols=int(cols),
            )
        )

        report = build_report(
            dealer_name or "Manuel",
            results,
            summary,
            int(capacity),
        )

        st.session_state.result_img = result_img
        st.session_state.results = results
        st.session_state.summary = summary
        st.session_state.report = report
        st.session_state.analysis_key = (
            hashlib.md5(
                result_img.tobytes()
            ).hexdigest()
        )


# =========================================================
# SONUÇ
# =========================================================
if (
    st.session_state.result_img is not None
    and st.session_state.summary
):
    summary = st.session_state.summary
    results = st.session_state.results

    total = max(
        1,
        len(results),
    )

    fark = int(
        summary.get("fark", 0)
    )
    supheli = int(
        summary.get("supheli", 0)
    )
    uyumlu = int(
        summary.get("uyumlu", 0)
    )

    uyum_orani = (
        100.0 * uyumlu / total
    )

    st.subheader(
        "3. Analiz Sonucu "
        f"(ilk {summary.get('kontrol_edilen_raf', ANALYZE_ROW_LIMIT)} raf)"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "🔴 FARK",
        fark,
    )

    m2.metric(
        "🟠 ŞÜPHELİ",
        supheli,
    )

    m3.metric(
        "🟢 UYUMLU",
        uyumlu,
    )

    m4.metric(
        "📊 Uyum Oranı",
        f"%{uyum_orani:.1f}",
    )

    if summary.get(
        "hizalama_ok"
    ):
        st.success(
            "Fotoğraf hizalaması: "
            f"{summary.get('hizalama')} "
            f"(inlier={summary.get('inliers')})"
        )
    else:
        st.warning(
            "Tam geometrik hizalama "
            "doğrulanamadı. Analiz ölçek "
            "eşitleme üzerinden yapıldı."
        )

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
                "_denetim.jpg"
            ),
            mime="image/jpeg",
            use_container_width=True,
        )

else:
    st.info(
        "Analiz için referans planogram ve "
        "saha fotoğrafını yükleyin. Ardından "
        "'KONTROLE BAŞLA' "
        "düğmesine basın."
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
