import streamlit as st
import cv2
import numpy as np
import requests
import urllib.parse
import math


# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Sigara Standı Planogram Denetim Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}

[data-testid="stHeader"] {
    visibility:hidden;
    display:none!important;
}

[data-testid="stToolbar"] {
    visibility:hidden;
    display:none!important;
}

.block-container {
    padding-top:1rem;
    padding-bottom:2rem;
}

div.stButton > button {
    min-height:46px;
    font-weight:700;
}

[data-testid="stMetricValue"] {
    font-size:28px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# GENEL AYARLAR
# ============================================================

YANDEX_ROOT_PUBLIC_KEY = (
    "https://disk.yandex.com.tr/d/ikCHPwREiCVv_g"
)

RAF_SAYISI = 7
SLOT_SAYISI = 11
TOPLAM_SLOT = RAF_SAYISI * SLOT_SAYISI


# ============================================================
# GÖRÜNTÜ BOYUTLANDIRMA
# ============================================================

def resmi_boyutlandir(img, max_genislik=1600):

    if img is None:
        return None

    h, w = img.shape[:2]

    if w <= max_genislik:
        return img

    oran = max_genislik / float(w)

    yeni_h = int(h * oran)

    return cv2.resize(
        img,
        (max_genislik, yeni_h),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# KONTRAST / NORMALİZASYON
# ============================================================

def gri_normalize(img, size=(180, 180)):

    if img is None or img.size == 0:
        return None

    img = cv2.resize(
        img,
        size,
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return gray


# ============================================================
# SSIM
# ============================================================

def ssim_hesapla(img1, img2):

    a = gri_normalize(img1)
    b = gri_normalize(img2)

    if a is None or b is None:
        return 0.0

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mu1 = cv2.GaussianBlur(
        a,
        (11, 11),
        1.5
    )

    mu2 = cv2.GaussianBlur(
        b,
        (11, 11),
        1.5
    )

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        cv2.GaussianBlur(
            a * a,
            (11, 11),
            1.5
        )
        - mu1_sq
    )

    sigma2_sq = (
        cv2.GaussianBlur(
            b * b,
            (11, 11),
            1.5
        )
        - mu2_sq
    )

    sigma12 = (
        cv2.GaussianBlur(
            a * b,
            (11, 11),
            1.5
        )
        - mu1_mu2
    )

    c1 = 6.5025
    c2 = 58.5225

    pay = (
        (2 * mu1_mu2 + c1)
        *
        (2 * sigma12 + c2)
    )

    payda = (
        (mu1_sq + mu2_sq + c1)
        *
        (sigma1_sq + sigma2_sq + c2)
    )

    sonuc = pay / (payda + 1e-8)

    return float(
        np.clip(
            np.mean(sonuc),
            0,
            1
        )
    )


# ============================================================
# RENK BENZERLİĞİ
# ============================================================

def renk_benzerligi(img1, img2):

    if (
        img1 is None
        or img2 is None
        or img1.size == 0
        or img2.size == 0
    ):
        return 0.0

    img1 = cv2.resize(
        img1,
        (180, 180)
    )

    img2 = cv2.resize(
        img2,
        (180, 180)
    )

    hsv1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2HSV
    )

    hsv2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2HSV
    )

    hist1 = cv2.calcHist(
        [hsv1],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    hist2 = cv2.calcHist(
        [hsv2],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        hist1,
        hist1
    )

    cv2.normalize(
        hist2,
        hist2
    )

    score = cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_CORREL
    )

    return float(
        np.clip(
            (score + 1) / 2,
            0,
            1
        )
    )


# ============================================================
# KENAR BENZERLİĞİ
# ============================================================

def kenar_benzerligi(img1, img2):

    a = gri_normalize(
        img1,
        (160, 160)
    )

    b = gri_normalize(
        img2,
        (160, 160)
    )

    if a is None or b is None:
        return 0.0

    edge1 = cv2.Canny(
        a,
        50,
        150
    )

    edge2 = cv2.Canny(
        b,
        50,
        150
    )

    # Hafif kalınlaştırma
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    edge1 = cv2.dilate(
        edge1,
        kernel,
        iterations=1
    )

    edge2 = cv2.dilate(
        edge2,
        kernel,
        iterations=1
    )

    ortak = np.logical_and(
        edge1 > 0,
        edge2 > 0
    ).sum()

    toplam = np.logical_or(
        edge1 > 0,
        edge2 > 0
    ).sum()

    if toplam == 0:
        return 0.0

    return float(
        ortak / toplam
    )


# ============================================================
# ORB
# ============================================================

def orb_benzerligi(img1, img2):

    if (
        img1 is None
        or img2 is None
        or img1.size == 0
        or img2.size == 0
    ):
        return 0.0

    a = cv2.resize(
        img1,
        (280, 280),
        interpolation=cv2.INTER_AREA
    )

    b = cv2.resize(
        img2,
        (280, 280),
        interpolation=cv2.INTER_AREA
    )

    gray1 = cv2.cvtColor(
        a,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        b,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8
    )

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:
        return 0.0

    if len(des1) < 5 or len(des2) < 5:
        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    try:

        matches = matcher.knnMatch(
            des1,
            des2,
            k=2
        )

    except Exception:
        return 0.0

    iyi = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.76 * n.distance:
            iyi.append(m)

    return float(
        np.clip(
            len(iyi) / 30.0,
            0,
            1
        )
    )


# ============================================================
# GÖRÜNTÜ DOLULUK ANALİZİ
# ============================================================

def doluluk(img):

    if img is None or img.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (120, 120),
        interpolation=cv2.INTER_AREA
    )

    edge = cv2.Canny(
        gray,
        50,
        150
    )

    return float(
        np.mean(edge > 0)
    )


# ============================================================
# FOTOĞRAF PERSPEKTİF HİZALAMA
# ============================================================

def perspektif_hizala(
    referans,
    saha
):

    if referans is None or saha is None:
        return saha, False, 0

    ref_small = referans.copy()
    saha_small = saha.copy()

    maksimum = 1200

    if ref_small.shape[1] > maksimum:

        oran = (
            maksimum
            /
            ref_small.shape[1]
        )

        ref_small = cv2.resize(
            ref_small,
            (
                maksimum,
                int(
                    ref_small.shape[0]
                    * oran
                )
            ),
            interpolation=cv2.INTER_AREA
        )

    if saha_small.shape[1] > maksimum:

        oran = (
            maksimum
            /
            saha_small.shape[1]
        )

        saha_small = cv2.resize(
            saha_small,
            (
                maksimum,
                int(
                    saha_small.shape[0]
                    * oran
                )
            ),
            interpolation=cv2.INTER_AREA
        )

    gray_ref = cv2.cvtColor(
        ref_small,
        cv2.COLOR_BGR2GRAY
    )

    gray_saha = cv2.cvtColor(
        saha_small,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=5000
    )

    kp1, des1 = orb.detectAndCompute(
        gray_ref,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray_saha,
        None
    )

    if des1 is None or des2 is None:
        return saha, False, 0

    if len(kp1) < 15 or len(kp2) < 15:
        return saha, False, 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    try:

        matches = matcher.knnMatch(
            des2,
            des1,
            k=2
        )

    except Exception:
        return saha, False, 0

    good = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.72 * n.distance:
            good.append(m)

    if len(good) < 12:
        return saha, False, len(good)

    src = np.float32([
        kp2[m.queryIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    dst = np.float32([
        kp1[m.trainIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        return saha, False, 0

    inliers = int(
        mask.sum()
    ) if mask is not None else 0

    if inliers < 8:
        return saha, False, inliers

    h, w = referans.shape[:2]

    hizali = cv2.warpPerspective(
        saha,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return (
        hizali,
        True,
        inliers
    )


# ============================================================
# RAF BÖLGELERİ
# ============================================================
#
# ÖNEMLİ:
# Eski sistem 0-100% görüntüyü 7 eşit parçaya bölüyordu.
# Bu stand fotoğrafında bu yanlıştı.
#
# Burada ürün raflarının gerçek yaklaşık konumları
# kullanılıyor.
#
# Oranlar gerektiğinde sidebar'dan değiştirilebilir.
# ============================================================

DEFAULT_RAF_SINIRLARI = [
    0.115,
    0.245,
    0.365,
    0.485,
    0.605,
    0.725,
    0.845,
    0.965
]


def raf_slotlarini_olustur(
    img,
    raf_sinirlari,
    kolon_sayisi=11
):

    h, w = img.shape[:2]

    x_sol = int(
        w * 0.055
    )

    x_sag = int(
        w * 0.945
    )

    slotlar = []

    for raf in range(7):

        y1 = int(
            h * raf_sinirlari[raf]
        )

        y2 = int(
            h * raf_sinirlari[raf + 1]
        )

        slot_genisligi = (
            x_sag - x_sol
        ) / float(kolon_sayisi)

        for kolon in range(
            kolon_sayisi
        ):

            x1 = int(
                x_sol
                +
                kolon * slot_genisligi
            )

            x2 = int(
                x_sol
                +
                (kolon + 1)
                * slot_genisligi
            )

            # Plastik ayırıcıları azalt
            kenar_x = int(
                (x2 - x1)
                * 0.06
            )

            # Üst/alt kenarları azalt
            kenar_y = int(
                (y2 - y1)
                * 0.05
            )

            slotlar.append({
                "raf": raf + 1,
                "slot": kolon + 1,
                "x1": x1 + kenar_x,
                "y1": y1 + kenar_y,
                "x2": x2 - kenar_x,
                "y2": y2 - kenar_y
            })

    return slotlar


# ============================================================
# ÜRÜN BÖLGESİ ÇIKARMA
# ============================================================

def urun_bolgesi_al(
    img,
    x1,
    y1,
    x2,
    y2
):

    slot = img[
        y1:y2,
        x1:x2
    ]

    if slot is None or slot.size == 0:
        return None

    h, w = slot.shape[:2]

    # Üstteki çok küçük raf boşluğunu at
    ust = int(
        h * 0.04
    )

    # Alt taraftaki fiyat etiketi / beyaz etiket
    # ve raf kısmını büyük ölçüde dışarıda bırak
    alt = int(
        h * 0.82
    )

    if alt <= ust:
        return slot

    return slot[
        ust:alt,
        int(w * 0.04):
        int(w * 0.96)
    ]


# ============================================================
# KÜÇÜK KAYMALARA TOLERANS
# ============================================================

def en_iyi_karsilastirma(
    ref_img,
    saha_img
):

    if (
        ref_img is None
        or saha_img is None
        or ref_img.size == 0
        or saha_img.size == 0
    ):
        return {
            "score": 0,
            "ssim": 0,
            "renk": 0,
            "kenar": 0,
            "orb": 0
        }

    h, w = saha_img.shape[:2]

    # Ürün birkaç piksel sağ/sol/yukarı/aşağı
    # kaymışsa farklı crop deniyoruz.
    kaymalar = [
        (0, 0),
        (-0.04, 0),
        (0.04, 0),
        (0, -0.04),
        (0, 0.04)
    ]

    en_iyi = None

    for dx, dy in kaymalar:

        x1 = max(
            0,
            int(w * max(0, dx))
        )

        x2 = min(
            w,
            int(w * (1 + min(0, dx)))
        )

        y1 = max(
            0,
            int(h * max(0, dy))
        )

        y2 = min(
            h,
            int(h * (1 + min(0, dy)))
        )

        current = saha_img[
            y1:y2,
            x1:x2
        ]

        if current.size == 0:
            continue

        current = cv2.resize(
            current,
            (
                ref_img.shape[1],
                ref_img.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

        ssim = ssim_hesapla(
            ref_img,
            current
        )

        renk = renk_benzerligi(
            ref_img,
            current
        )

        kenar = kenar_benzerligi(
            ref_img,
            current
        )

        orb = orb_benzerligi(
            ref_img,
            current
        )

        # Ağırlıklar
        score = (
            ssim * 0.45
            +
            renk * 0.20
            +
            kenar * 0.15
            +
            orb * 0.20
        )

        veri = {
            "score": float(score),
            "ssim": float(ssim),
            "renk": float(renk),
            "kenar": float(kenar),
            "orb": float(orb)
        }

        if (
            en_iyi is None
            or veri["score"]
            >
            en_iyi["score"]
        ):
            en_iyi = veri

    if en_iyi is None:

        return {
            "score": 0,
            "ssim": 0,
            "renk": 0,
            "kenar": 0,
            "orb": 0
        }

    return en_iyi


# ============================================================
# SLOT ANALİZİ
# ============================================================

def slot_analiz_et(
    ref_img,
    saha_img,
    koordinat
):

    x1 = koordinat["x1"]
    y1 = koordinat["y1"]
    x2 = koordinat["x2"]
    y2 = koordinat["y2"]

    ref_urun = urun_bolgesi_al(
        ref_img,
        x1,
        y1,
        x2,
        y2
    )

    saha_urun = urun_bolgesi_al(
        saha_img,
        x1,
        y1,
        x2,
        y2
    )

    if (
        ref_urun is None
        or saha_urun is None
        or ref_urun.size == 0
        or saha_urun.size == 0
    ):

        return {
            **koordinat,
            "score": 0,
            "ssim": 0,
            "renk": 0,
            "kenar": 0,
            "orb": 0,
            "ref_doluluk": 0,
            "saha_doluluk": 0
        }

    sonuc = en_iyi_karsilastirma(
        ref_urun,
        saha_urun
    )

    ref_doluluk = doluluk(
        ref_urun
    )

    saha_doluluk = doluluk(
        saha_urun
    )

    return {
        **koordinat,
        **sonuc,
        "ref_doluluk": ref_doluluk,
        "saha_doluluk": saha_doluluk
    }


# ============================================================
# BOŞ SLOT TESPİTİ
# ============================================================

def bos_slot_mu(
    ref_slot,
    saha_slot
):

    if (
        ref_slot is None
        or saha_slot is None
        or ref_slot.size == 0
        or saha_slot.size == 0
    ):
        return False

    ref_d = doluluk(
        ref_slot
    )

    saha_d = doluluk(
        saha_slot
    )

    # Sahadaki yapı referansın çok altındaysa
    # boş olma ihtimali yüksek.
    if (
        saha_d < 0.42 * max(
            ref_d,
            0.01
        )
    ):
        return True

    return False


# ============================================================
# YANDEX ŞEHİRLER
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def yandex_sehirleri_getir(
    public_key
):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    sehirler = []

    try:

        root_url = (
            "https://cloud-api.yandex.net/v1/disk/"
            "public/resources"
            f"?public_key={public_key}"
            "&limit=500"
        )

        resp = requests.get(
            root_url,
            headers=headers,
            timeout=20
        )

        if resp.status_code != 200:
            return [], (
                f"Kök dizin okunamadı "
                f"(HTTP {resp.status_code})"
            )

        items = (
            resp.json()
            .get("_embedded", {})
            .get("items", [])
        )

        for item in items:

            if item.get("type") != "dir":
                continue

            name = item.get(
                "name",
                ""
            )

            if name.upper() in [
                "BAYİ",
                "BAYI"
            ]:

                bayi_path = item.get(
                    "path"
                )

                sub_url = (
                    "https://cloud-api.yandex.net/v1/disk/"
                    "public/resources"
                    f"?public_key={public_key}"
                    f"&path={urllib.parse.quote(bayi_path, safe='/')}"
                    "&limit=500"
                )

                sub_resp = requests.get(
                    sub_url,
                    headers=headers,
                    timeout=20
                )

                if sub_resp.status_code == 200:

                    sub_items = (
                        sub_resp.json()
                        .get("_embedded", {})
                        .get("items", [])
                    )

                    for sub in sub_items:

                        if sub.get("type") == "dir":

                            if sub.get("name"):
                                sehirler.append(
                                    sub["name"]
                                )

            else:

                if name:
                    sehirler.append(
                        name
                    )

        return sorted(
            list(set(sehirler))
        ), None

    except Exception as e:

        return [], str(e)


# ============================================================
# YANDEX BAYİLER
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def yandex_sehir_bayilerini_getir(
    public_key,
    sehir_adi
):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:

        root_url = (
            "https://cloud-api.yandex.net/v1/disk/"
            "public/resources"
            f"?public_key={public_key}"
            "&limit=500"
        )

        resp = requests.get(
            root_url,
            headers=headers,
            timeout=20
        )

        if resp.status_code != 200:
            return [], "Kök dizin okunamadı."

        root_items = (
            resp.json()
            .get("_embedded", {})
            .get("items", [])
        )

        sehir_item = None

        for item in root_items:

            if (
                item.get("type") == "dir"
                and item.get("name", "").upper()
                == sehir_adi.upper()
            ):

                sehir_item = item
                break

        if sehir_item is None:

            for item in root_items:

                if (
                    item.get("type") == "dir"
                    and item.get("name", "").upper()
                    in ["BAYİ", "BAYI"]
                ):

                    bayi_path = item.get(
                        "path"
                    )

                    sub_url = (
                        "https://cloud-api.yandex.net/v1/disk/"
                        "public/resources"
                        f"?public_key={public_key}"
                        f"&path={urllib.parse.quote(bayi_path, safe='/')}"
                        "&limit=500"
                    )

                    sub_resp = requests.get(
                        sub_url,
                        headers=headers,
                        timeout=20
                    )

                    if sub_resp.status_code != 200:
                        continue

                    sub_items = (
                        sub_resp.json()
                        .get("_embedded", {})
                        .get("items", [])
                    )

                    for sub in sub_items:

                        if (
                            sub.get("type") == "dir"
                            and sub.get("name", "").upper()
                            == sehir_adi.upper()
                        ):

                            sehir_item = sub
                            break

                    if sehir_item:
                        break

        if sehir_item is None:

            return [], (
                f"'{sehir_adi}' klasörü bulunamadı."
            )

        sehir_path = sehir_item.get(
            "path"
        )

        url = (
            "https://cloud-api.yandex.net/v1/disk/"
            "public/resources"
            f"?public_key={public_key}"
            f"&path={urllib.parse.quote(sehir_path, safe='/')}"
            "&limit=500"
        )

        city_resp = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        if city_resp.status_code != 200:
            return [], (
                "Şehir içeriği okunamadı."
            )

        items = (
            city_resp.json()
            .get("_embedded", {})
            .get("items", [])
        )

        bayiler = []

        for item in items:

            if item.get("type") != "dir":
                continue

            name = item.get("name")
            path = item.get("path")

            if name and path:

                bayiler.append({
                    "name": name,
                    "path": path
                })

        return sorted(
            bayiler,
            key=lambda x: x["name"]
        ), None

    except Exception as e:

        return [], str(e)


# ============================================================
# YANDEX BAYİ GÖRSELİ
# ============================================================

@st.cache_data(
    ttl=600,
    show_spinner=False
)
def yandex_bayi_gorseli_getir(
    public_key,
    bayi_path
):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:

        url = (
            "https://cloud-api.yandex.net/v1/disk/"
            "public/resources"
            f"?public_key={public_key}"
            f"&path={urllib.parse.quote(bayi_path, safe='/')}"
            "&limit=500"
        )

        resp = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        if resp.status_code != 200:
            return None, (
                "Bayi klasörü okunamadı."
            )

        items = (
            resp.json()
            .get("_embedded", {})
            .get("items", [])
        )

        # Önce görsel bul
        for item in items:

            if item.get("type") != "file":
                continue

            name = item.get(
                "name",
                ""
            ).lower()

            if name.endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp"
                )
            ):

                file_url = item.get(
                    "file"
                )

                if not file_url:
                    continue

                img_resp = requests.get(
                    file_url,
                    headers=headers,
                    timeout=30
                )

                if img_resp.status_code != 200:
                    continue

                data = np.frombuffer(
                    img_resp.content,
                    dtype=np.uint8
                )

                img = cv2.imdecode(
                    data,
                    cv2.IMREAD_COLOR
                )

                if img is not None:

                    return (
                        resmi_boyutlandir(
                            img,
                            1600
                        ),
                        None
                    )

        return None, (
            "Bayi klasöründe görsel bulunamadı."
        )

    except Exception as e:

        return None, str(e)


# ============================================================
# ŞİFRE
# ============================================================

if "app_password" not in st.secrets:

    st.error(
        "⚠️ 'app_password' Streamlit secrets "
        "içinde tanımlı değil."
    )

    st.stop()

APP_PASSWORD = st.secrets[
    "app_password"
]

if "authenticated" not in st.session_state:

    st.session_state.authenticated = False


if not st.session_state.authenticated:

    st.title(
        "🔐 Kurumsal Planogram Denetim Sistemi"
    )

    st.caption(
        "Developed by Hakan"
    )

    password = st.text_input(
        "Şifre",
        type="password"
    )

    if st.button(
        "Giriş Yap",
        type="primary",
        use_container_width=True
    ):

        if password == APP_PASSWORD:

            st.session_state.authenticated = True

            st.rerun()

        else:

            st.error(
                "❌ Hatalı şifre!"
            )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Analiz Ayarları"
)

st.sidebar.info(
    """
    Yeni motor:

    ✓ Perspektif düzeltme
    ✓ Raf bazlı analiz
    ✓ Slot bazlı analiz
    ✓ Renk karşılaştırması
    ✓ Yapısal karşılaştırma
    ✓ Kenar analizi
    ✓ ORB
    ✓ Küçük konum kaymalarına tolerans
    ✓ Boş slot kontrolü
    """
)

fark_esigi = st.sidebar.slider(
    "🔴 Fark Eşiği",
    0.25,
    0.70,
    0.38,
    0.01
)

supheli_esigi = st.sidebar.slider(
    "🟠 Şüpheli Eşiği",
    0.40,
    0.85,
    0.55,
    0.01
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📐 Raf Konum Ayarı"
)

st.sidebar.caption(
    "Stand fotoğrafı değişirse raf sınırlarını "
    "buradan küçük miktarda ayarlayabilirsiniz."
)

raf_sinirlari = []

for i, varsayilan in enumerate(
    DEFAULT_RAF_SINIRLARI
):

    deger = st.sidebar.slider(
        f"R{i+1} sınırı",
        0.00,
        1.00,
        float(varsayilan),
        0.005,
        key=f"raf_sinir_{i}"
    )

    raf_sinirlari.append(
        deger
    )


# ============================================================
# BAŞLIK
# ============================================================

col_baslik, col_cikis = st.columns(
    [5, 1]
)

with col_baslik:

    st.title(
        "SİGARA STANDI PLANOGRAM DENETİM SİSTEMİ"
    )

    st.caption(
        "Advanced Multi-Metric Planogram Engine"
    )

with col_cikis:

    if st.button(
        "🚪 Çıkış Yap"
    ):

        st.session_state.authenticated = False

        st.rerun()


st.markdown("---")


# ============================================================
# ŞEHİR / BAYİ
# ============================================================

st.subheader(
    "1. Lokasyon ve Bayi Seçimi"
)

sehirler, sehir_hata = (
    yandex_sehirleri_getir(
        YANDEX_ROOT_PUBLIC_KEY
    )
)

if sehir_hata:
    st.warning(
        sehir_hata
    )

col1, col2 = st.columns(2)

with col1:

    secilen_sehir = st.selectbox(
        "Şehir",
        sehirler,
        index=None,
        placeholder="Şehir seçin..."
    )


bayiler = []

if secilen_sehir:

    bayiler, bayi_hata = (
        yandex_sehir_bayilerini_getir(
            YANDEX_ROOT_PUBLIC_KEY,
            secilen_sehir
        )
    )

    if bayi_hata:
        st.warning(
            bayi_hata
        )


with col2:

    if bayiler:

        secilen_bayi = st.selectbox(
            "Bayi",
            [
                x["name"]
                for x in bayiler
            ],
            index=None,
            placeholder="Bayi seçin..."
        )

    else:

        secilen_bayi = None

        st.selectbox(
            "Bayi",
            ["Önce şehir seçin"],
            disabled=True
        )


# ============================================================
# BAYİ PATH
# ============================================================

secilen_path = None

if secilen_bayi:

    for bayi in bayiler:

        if bayi["name"] == secilen_bayi:

            secilen_path = bayi["path"]

            break


# ============================================================
# ÖNBELLEK
# ============================================================

if st.button(
    "🔄 Yandex Önbelleğini Yenile"
):

    yandex_sehirleri_getir.clear()

    yandex_sehir_bayilerini_getir.clear()

    yandex_bayi_gorseli_getir.clear()

    st.toast(
        "Önbellek temizlendi.",
        icon="🔄"
    )

    st.rerun()


# ============================================================
# REFERANS GÖRÜNTÜ
# ============================================================

ref_img = None

if secilen_path:

    with st.spinner(
        "Referans planogram yükleniyor..."
    ):

        ref_img, hata = (
            yandex_bayi_gorseli_getir(
                YANDEX_ROOT_PUBLIC_KEY,
                secilen_path
            )
        )

    if hata:
        st.warning(
            hata
        )


# ============================================================
# FOTOĞRAF ALANI
# ============================================================

st.markdown("---")

col_ref, col_saha = st.columns(2)

with col_ref:

    st.subheader(
        "2. Dijital Planogram"
    )

    if ref_img is not None:

        st.success(
            "✅ Referans hazır"
        )

        st.image(
            ref_img,
            channels="BGR",
            use_container_width=True
        )

    else:

        st.info(
            "Şehir ve bayi seçin."
        )


with col_saha:

    st.subheader(
        "3. Saha Fotoğrafı"
    )

    curr_file = None
    curr_img = None

    if secilen_path:

        curr_file = st.file_uploader(
            "Saha fotoğrafını yükleyin",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            key="saha_fotografi"
        )

        if curr_file:

            bytes_data = np.asarray(
                bytearray(
                    curr_file.getvalue()
                ),
                dtype=np.uint8
            )

            raw = cv2.imdecode(
                bytes_data,
                cv2.IMREAD_COLOR
            )

            curr_img = resmi_boyutlandir(
                raw,
                1600
            )

            if curr_img is not None:

                st.success(
                    "✅ Saha fotoğrafı hazır"
                )

                st.image(
                    curr_img,
                    channels="BGR",
                    use_container_width=True
                )


# ============================================================
# KAPASİTE
# ============================================================

st.markdown("---")

st.subheader(
    "4. Stand Kapasitesi"
)

ideal_slot = st.number_input(
    "Standda bulunması gereken toplam slot",
    min_value=1,
    max_value=200,
    value=77,
    step=1
)


# ============================================================
# ANALİZ BUTONU
# ============================================================

st.markdown("---")

hazir = (
    ref_img is not None
    and curr_img is not None
    and secilen_bayi is not None
)

if hazir:

    if st.button(
        "🚀 PLANOGRAM FARKLARINI BUL",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Gelişmiş planogram analiz motoru çalışıyor..."
        ):

            # ------------------------------------------------
            # RAF SINIRLARI KONTROL
            # ------------------------------------------------

            sirali = all(
                raf_sinirlari[i]
                <
                raf_sinirlari[i + 1]
                for i in range(
                    len(raf_sinirlari) - 1
                )
            )

            if not sirali:

                st.error(
                    "❌ Raf sınırları doğru sırada değil."
                )

                st.stop()

            # ------------------------------------------------
            # PERSPEKTİF
            # ------------------------------------------------

            hizali_img, hizalama_ok, inliers = (
                perspektif_hizala(
                    ref_img,
                    curr_img
                )
            )

            # Boyut eşitle
            h, w = ref_img.shape[:2]

            hizali_img = cv2.resize(
                hizali_img,
                (w, h),
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------
            # SLOT KOORDİNATLARI
            # ------------------------------------------------

            slotlar = raf_slotlarini_olustur(
                ref_img,
                raf_sinirlari,
                SLOT_SAYISI
            )

            sonuc_img = hizali_img.copy()

            farklar = []
            supheliler = []
            uyumlular = []
            boslar = []

            # ------------------------------------------------
            # SLOT ANALİZİ
            # ------------------------------------------------

            ilerleme = st.progress(
                0,
                text="Slotlar analiz ediliyor..."
            )

            for index, slot in enumerate(
                slotlar
            ):

                analiz = slot_analiz_et(
                    ref_img,
                    hizali_img,
                    slot
                )

                score = analiz["score"]

                x1 = slot["x1"]
                y1 = slot["y1"]
                x2 = slot["x2"]
                y2 = slot["y2"]

                # --------------------------------------------
                # BOŞLUK KONTROLÜ
                # --------------------------------------------

                ref_full = ref_img[
                    y1:y2,
                    x1:x2
                ]

                saha_full = hizali_img[
                    y1:y2,
                    x1:x2
                ]

                bos_mu = bos_slot_mu(
                    ref_full,
                    saha_full
                )

                # --------------------------------------------
                # SINIFLANDIRMA
                # --------------------------------------------

                if bos_mu:

                    durum = "BOS"

                    boslar.append(
                        analiz
                    )

                    renk = (
                        255,
                        255,
                        255
                    )

                    kalinlik = 3

                elif score < fark_esigi:

                    durum = "FARKLI"

                    farklar.append(
                        analiz
                    )

                    renk = (
                        0,
                        0,
                        255
                    )

                    kalinlik = 4

                elif score < supheli_esigi:

                    durum = "SUPHELI"

                    supheliler.append(
                        analiz
                    )

                    renk = (
                        0,
                        165,
                        255
                    )

                    kalinlik = 4

                else:

                    durum = "UYUMLU"

                    uyumlular.append(
                        analiz
                    )

                    renk = (
                        0,
                        190,
                        0
                    )

                    kalinlik = 2

                # --------------------------------------------
                # KUTU
                # --------------------------------------------

                cv2.rectangle(
                    sonuc_img,
                    (x1, y1),
                    (x2, y2),
                    renk,
                    kalinlik
                )

                # --------------------------------------------
                # ETİKET
                # --------------------------------------------

                if durum == "FARKLI":
                    prefix = "FARK"

                elif durum == "SUPHELI":
                    prefix = "?"

                elif durum == "BOS":
                    prefix = "BOS"

                else:
                    prefix = "OK"

                label = (
                    f"{prefix} "
                    f"R{slot['raf']}/S{slot['slot']}"
                )

                text_y = max(
                    18,
                    y1 - 5
                )

                cv2.putText(
                    sonuc_img,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    renk,
                    2,
                    cv2.LINE_AA
                )

                ilerleme.progress(
                    (index + 1)
                    /
                    len(slotlar),
                    text=(
                        f"Slot {index + 1}/"
                        f"{len(slotlar)} analiz ediliyor..."
                    )
                )

            ilerleme.empty()

            # ------------------------------------------------
            # UYUM ORANI
            # ------------------------------------------------
            #
            # Şüpheli ve boş alanlar otomatik olarak
            # "uyumlu" sayılmaz.
            # ------------------------------------------------

            toplam = (
                len(farklar)
                +
                len(supheliler)
                +
                len(uyumlular)
                +
                len(boslar)
            )

            uyumlu_adet = len(
                uyumlular
            )

            if toplam > 0:

                uyum_yuzdesi = (
                    uyumlu_adet
                    /
                    toplam
                ) * 100

            else:

                uyum_yuzdesi = 0

            # ------------------------------------------------
            # ÜST BİLGİ PANELİ
            # ------------------------------------------------

            panel_h = 125

            cv2.rectangle(
                sonuc_img,
                (0, 0),
                (w, panel_h),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                sonuc_img,
                f"Bayi: {secilen_bayi}",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                sonuc_img,
                (
                    f"FARK: {len(farklar)}   "
                    f"SUPHELI: {len(supheliler)}   "
                    f"BOS: {len(boslar)}   "
                    f"UYUMLU: {len(uyumlular)}"
                ),
                (18, 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                sonuc_img,
                f"UYUM ORANI: %{uyum_yuzdesi:.1f}",
                (18, 103),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # SESSION
            # ------------------------------------------------

            st.session_state.result_img = (
                sonuc_img
            )

            st.session_state.farklar = (
                farklar
            )

            st.session_state.supheliler = (
                supheliler
            )

            st.session_state.uyumlular = (
                uyumlular
            )

            st.session_state.boslar = (
                boslar
            )

            st.session_state.uyum_yuzdesi = (
                uyum_yuzdesi
            )

            st.session_state.hizalama_ok = (
                hizalama_ok
            )

            st.session_state.inliers = (
                inliers
            )

            st.session_state.analiz_yapildi = True


# ============================================================
# SONUÇ
# ============================================================

if st.session_state.get(
    "analiz_yapildi",
    False
):

    st.markdown("---")

    st.subheader(
        "📊 DENETİM SONUCU"
    )

    farklar = st.session_state.get(
        "farklar",
        []
    )

    supheliler = st.session_state.get(
        "supheliler",
        []
    )

    uyumlular = st.session_state.get(
        "uyumlular",
        []
    )

    boslar = st.session_state.get(
        "boslar",
        []
    )

    toplam = (
        len(farklar)
        +
        len(supheliler)
        +
        len(uyumlular)
        +
        len(boslar)
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Toplam Slot",
        toplam
    )

    c2.metric(
        "🔴 Farklı",
        len(farklar)
    )

    c3.metric(
        "🟠 Şüpheli",
        len(supheliler)
    )

    c4.metric(
        "⚪ Boş",
        len(boslar)
    )

    c5.metric(
        "🟢 Uyumlu",
        len(uyumlular)
    )

    st.metric(
        "📈 Planogram Uyum Oranı",
        (
            f"%"
            f"{st.session_state.get('uyum_yuzdesi', 0):.1f}"
        )
    )

    # --------------------------------------------------------
    # HİZALAMA
    # --------------------------------------------------------

    if st.session_state.get(
        "hizalama_ok",
        False
    ):

        st.success(
            "✅ Saha fotoğrafı referans planograma "
            "perspektif olarak hizalandı. "
            f"Geçerli eşleşme: "
            f"{st.session_state.get('inliers', 0)}"
        )

    else:

        st.warning(
            "⚠️ Perspektif otomatik olarak tam "
            "hizalanamadı. Slot analizi yine yapıldı."
        )

    # --------------------------------------------------------
    # SONUÇ GÖRSELİ
    # --------------------------------------------------------

    st.subheader(
        "🔍 Slot Bazlı Analiz"
    )

    st.caption(
        "🟢 Uyumlu   |   "
        "🟠 Şüpheli   |   "
        "🔴 Farklı   |   "
        "⚪ Boş"
    )

    gorsel_genislik = st.slider(
        "Sonuç görseli",
        500,
        1800,
        1000,
        100
    )

    st.image(
        st.session_state.result_img,
        channels="BGR",
        width=gorsel_genislik
    )

    # --------------------------------------------------------
    # FARKLAR
    # --------------------------------------------------------

    if farklar:

        st.markdown("---")

        st.subheader(
            "🔴 Tespit Edilen Farklı Slotlar"
        )

        for item in farklar:

            st.error(
                (
                    f"R{item['raf']}/S{item['slot']}  —  "
                    f"Benzerlik: "
                    f"%{item['score'] * 100:.1f}"
                )
            )

            st.caption(
                (
                    f"Yapı: %{item['ssim'] * 100:.1f} | "
                    f"Renk: %{item['renk'] * 100:.1f} | "
                    f"Kenar: %{item['kenar'] * 100:.1f} | "
                    f"ORB: %{item['orb'] * 100:.1f}"
                )
            )

    else:

        st.success(
            "🔴 Fark eşiğinin altında slot bulunmadı."
        )

    # --------------------------------------------------------
    # ŞÜPHELİLER
    # --------------------------------------------------------

    if supheliler:

        st.markdown("---")

        st.subheader(
            "🟠 Şüpheli Slotlar"
        )

        for item in supheliler:

            st.warning(
                (
                    f"R{item['raf']}/S{item['slot']}  —  "
                    f"Benzerlik: "
                    f"%{item['score'] * 100:.1f}"
                )
            )

    # --------------------------------------------------------
    # BOŞ SLOT
    # --------------------------------------------------------

    if boslar:

        st.markdown("---")

        st.subheader(
            "⚪ Boş Tespit Edilen Slotlar"
        )

        bos_text = ", ".join(
            [
                f"R{x['raf']}/S{x['slot']}"
                for x in boslar
            ]
        )

        st.info(
            bos_text
        )

    # --------------------------------------------------------
    # RAPOR
    # --------------------------------------------------------

    st.markdown("---")

    ok, jpg = cv2.imencode(
        ".jpg",
        st.session_state.result_img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if ok:

        dosya_adi = (
            secilen_bayi
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        st.download_button(
            "📥 Planogram Denetim Raporunu İndir",
            data=jpg.tobytes(),
            file_name=(
                f"{dosya_adi}_"
                "planogram_raporu.jpg"
            ),
            mime="image/jpeg",
            use_container_width=True
        )


# ============================================================
# HAZIR DEĞİL
# ============================================================

else:

    st.info(
        "ℹ️ Analiz için şehir, bayi ve saha fotoğrafı seçin."
    )


# ============================================================
# ALT BİLGİ
# ============================================================

st.markdown(
    """
    <br>
    <hr>
    <p style="text-align:center;color:gray;">
    Advanced Planogram Detection Engine —
    Developed by Hakan
    </p>
    """,
    unsafe_allow_html=True
)
