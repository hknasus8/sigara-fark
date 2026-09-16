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
    page_title="Sigara Standı Kurumsal Planogram Denetim Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}
.stAppToolbar {visibility:hidden;display:none!important;}
[data-testid="stHeader"] {visibility:hidden;display:none!important;}
[data-testid="stToolbar"] {visibility:hidden;display:none!important;}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

div.stButton > button {
    min-height: 46px;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
}
</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def resmi_boyutlandir(img, max_genislik=1600):
    if img is None:
        return None

    h, w = img.shape[:2]

    if w > max_genislik:
        oran = max_genislik / float(w)
        yeni_h = int(h * oran)

        return cv2.resize(
            img,
            (max_genislik, yeni_h),
            interpolation=cv2.INTER_AREA
        )

    return img


def normalize_gray(img, size=(160, 160)):
    """
    Görüntüyü karşılaştırma için normalize eder.
    """
    if img is None or img.size == 0:
        return None

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kontrast normalizasyonu
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return gray


def structural_similarity(img1, img2):
    """
    Basit SSIM benzeri yapısal benzerlik.
    Harici skimage gerektirmez.
    """

    a = normalize_gray(img1)
    b = normalize_gray(img2)

    if a is None or b is None:
        return 0.0

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    C1 = 6.5025
    C2 = 58.5225

    ssim = (
        ((2 * mu_ab + C1) * (2 * sigma_ab + C2))
        /
        ((mu_a2 + mu_b2 + C1) *
         (sigma_a2 + sigma_b2 + C2))
    )

    return float(np.clip(np.mean(ssim), 0, 1))


def histogram_similarity(img1, img2):
    """
    HSV renk yapısını karşılaştırır.
    """

    if img1 is None or img2 is None:
        return 0.0

    img1 = cv2.resize(img1, (180, 180))
    img2 = cv2.resize(img2, (180, 180))

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

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

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    score = cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_CORREL
    )

    return float(np.clip((score + 1) / 2, 0, 1))


def edge_similarity(img1, img2):
    """
    Ürün kutularındaki yazı / çizgi / şekil yapısını karşılaştırır.
    """

    a = normalize_gray(img1)
    b = normalize_gray(img2)

    if a is None or b is None:
        return 0.0

    edge_a = cv2.Canny(a, 60, 150)
    edge_b = cv2.Canny(b, 60, 150)

    intersection = np.logical_and(
        edge_a > 0,
        edge_b > 0
    ).sum()

    union = np.logical_or(
        edge_a > 0,
        edge_b > 0
    ).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def orb_similarity(img1, img2):
    """
    Ürün üzerindeki lokal özellikleri karşılaştırır.
    """

    if img1 is None or img2 is None:
        return 0.0

    gray1 = cv2.cvtColor(
        cv2.resize(img1, (300, 300)),
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        cv2.resize(img2, (300, 300)),
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=800,
        scaleFactor=1.2,
        nlevels=8
    )

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0.0

    if len(des1) < 5 or len(des2) < 5:
        return 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    try:
        matches = matcher.knnMatch(
            des1,
            des2,
            k=2
        )
    except Exception:
        return 0.0

    good = []

    for pair in matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 25 iyi eşleşme = yaklaşık tam benzerlik
    score = min(len(good) / 25.0, 1.0)

    return float(score)


# ============================================================
# PERSPEKTİF HİZALAMA
# ============================================================

def perspektif_hizala(reference, current):
    """
    Saha fotoğrafını referans fotoğrafa mümkün olduğunca hizalar.

    ORB + RANSAC Homography.
    """

    if reference is None or current is None:
        return current, False, 0

    ref_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    cur_gray = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY
    )

    # Çok büyük görüntülerde hız
    max_w = 1200

    def resize_for_match(img):
        h, w = img.shape[:2]

        if w <= max_w:
            return img, 1.0

        scale = max_w / w

        return cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        ), scale

    ref_small, ref_scale = resize_for_match(reference)
    cur_small, cur_scale = resize_for_match(current)

    ref_gray = cv2.cvtColor(
        ref_small,
        cv2.COLOR_BGR2GRAY
    )

    cur_gray = cv2.cvtColor(
        cur_small,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=4000,
        scaleFactor=1.2,
        nlevels=8
    )

    kp1, des1 = orb.detectAndCompute(
        ref_gray,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        cur_gray,
        None
    )

    if des1 is None or des2 is None:
        return current, False, 0

    if len(kp1) < 10 or len(kp2) < 10:
        return current, False, 0

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
        return current, False, 0

    good = []

    for pair in matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.72 * n.distance:
            good.append(m)

    if len(good) < 12:
        return current, False, len(good)

    src_pts = np.float32([
        kp2[m.queryIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp1[m.trainIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        return current, False, 0

    inliers = int(mask.sum()) if mask is not None else 0

    if inliers < 8:
        return current, False, inliers

    h, w = reference.shape[:2]

    aligned = cv2.warpPerspective(
        current,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return aligned, True, inliers


# ============================================================
# SLOT ANALİZİ
# ============================================================

def slot_doluluk(img):
    """
    Slotun gerçekten ürün içerip içermediğini yaklaşık olarak belirler.
    """

    if img is None or img.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (120, 120)
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_density = np.mean(edges > 0)

    # Ürünlerde genellikle daha fazla yapı vardır.
    return float(edge_density)


def slot_karsilastir(ref_slot, curr_slot):
    """
    Bir slotu 4 farklı yöntemle karşılaştırır.
    """

    ssim = structural_similarity(
        ref_slot,
        curr_slot
    )

    hist = histogram_similarity(
        ref_slot,
        curr_slot
    )

    edge = edge_similarity(
        ref_slot,
        curr_slot
    )

    orb = orb_similarity(
        ref_slot,
        curr_slot
    )

    # Ağırlıklı skor
    toplam = (
        ssim * 0.40 +
        hist * 0.20 +
        edge * 0.15 +
        orb * 0.25
    )

    ref_doluluk = slot_doluluk(ref_slot)
    curr_doluluk = slot_doluluk(curr_slot)

    doluluk_farki = abs(
        ref_doluluk - curr_doluluk
    )

    # Büyük doluluk farkı varsa ceza
    if doluluk_farki > 0.045:
        toplam -= min(
            0.25,
            doluluk_farki * 2.0
        )

    toplam = float(
        np.clip(toplam, 0, 1)
    )

    # Durum
    if toplam >= 0.70:
        durum = "UYUMLU"

    elif toplam >= 0.52:
        durum = "ŞÜPHELİ"

    else:
        durum = "FARKLI"

    return {
        "score": toplam,
        "ssim": ssim,
        "hist": hist,
        "edge": edge,
        "orb": orb,
        "durum": durum
    }


# ============================================================
# STAND SLOT KOORDİNATLARI
# ============================================================

def slot_koordinatlari(img):
    """
    Standın gerçek ürün bölgesini kullanır.

    Üst boşluk ve alttaki koli/stok alanı
    analiz dışıdır.
    """

    h, w = img.shape[:2]

    # Stand içindeki ürün alanı.
    #
    # Üst boşluk yaklaşık %10
    # Alt stok alanı yaklaşık %89'dan sonra
    #
    # Bu oranlar kullanıcının gönderdiği
    # stand fotoğrafındaki yapıya göre ayarlanmıştır.

    x_sol = int(w * 0.075)
    x_sag = int(w * 0.935)

    y_ust = int(h * 0.115)
    y_alt = int(h * 0.875)

    raf_sayisi = 7
    kolon_sayisi = 11

    raf_h = (y_alt - y_ust) / raf_sayisi
    slot_w = (x_sag - x_sol) / kolon_sayisi

    koordinatlar = []

    for r in range(raf_sayisi):

        y1 = int(y_ust + r * raf_h)
        y2 = int(y_ust + (r + 1) * raf_h)

        for c in range(kolon_sayisi):

            x1 = int(x_sol + c * slot_w)
            x2 = int(x_sol + (c + 1) * slot_w)

            # Slot kenarlarını biraz kırp.
            # Şeffaf plastik ayırıcıların etkisini azaltır.
            dx = int((x2 - x1) * 0.08)
            dy = int((y2 - y1) * 0.12)

            sx1 = x1 + dx
            sx2 = x2 - dx
            sy1 = y1 + dy
            sy2 = y2 - dy

            koordinatlar.append(
                {
                    "raf": r + 1,
                    "slot": c + 1,
                    "x1": sx1,
                    "y1": sy1,
                    "x2": sx2,
                    "y2": sy2
                }
            )

    return koordinatlar


# ============================================================
# YANDEX
# ============================================================

YANDEX_ROOT_PUBLIC_KEY = (
    "https://disk.yandex.com.tr/d/ikCHPwREiCVv_g"
)


@st.cache_data(ttl=600, show_spinner=False)
def yandex_sehirleri_getir(public_key):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    sehirler = []

    try:

        root_url = (
            "https://cloud-api.yandex.net/v1/disk/"
            f"public/resources?public_key={public_key}&limit=200"
        )

        resp = requests.get(
            root_url,
            headers=headers,
            timeout=15
        )

        if resp.status_code != 200:
            return [], (
                f"Kök Dizin Okunamadı "
                f"(HTTP {resp.status_code})"
            )

        root_items = (
            resp.json()
            .get("_embedded", {})
            .get("items", [])
        )

        for item in root_items:

            if item.get("type") != "dir":
                continue

            name = item.get("name")

            if not name:
                continue

            if name.upper() in ["BAYİ", "BAYI"]:

                bayi_path = item.get("path")

                sub_url = (
                    "https://cloud-api.yandex.net/v1/disk/"
                    f"public/resources?public_key={public_key}"
                    f"&path={urllib.parse.quote(bayi_path, safe='/')}"
                    "&limit=500"
                )

                sub_resp = requests.get(
                    sub_url,
                    headers=headers,
                    timeout=15
                )

                if sub_resp.status_code == 200:

                    sub_items = (
                        sub_resp.json()
                        .get("_embedded", {})
                        .get("items", [])
                    )

                    for sub_item in sub_items:

                        if sub_item.get("type") == "dir":

                            sub_name = sub_item.get("name")

                            if sub_name:
                                sehirler.append(
                                    sub_name
                                )

            else:
                sehirler.append(name)

        return sorted(
            list(set(sehirler))
        ), None

    except Exception as e:
        return [], str(e)


@st.cache_data(ttl=600, show_spinner=False)
def yandex_sehir_bayilerini_getir(
    public_key,
    sehir_adi
):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    bayiler = []

    try:

        root_url = (
            "https://cloud-api.yandex.net/v1/disk/"
            f"public/resources?public_key={public_key}&limit=200"
        )

        resp = requests.get(
            root_url,
            headers=headers,
            timeout=15
        )

        if resp.status_code != 200:
            return [], "Kök Dizin Okunamadı"

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

                    bayi_path = item.get("path")

                    sub_url = (
                        "https://cloud-api.yandex.net/v1/disk/"
                        f"public/resources?public_key={public_key}"
                        f"&path={urllib.parse.quote(bayi_path, safe='/')}"
                        "&limit=500"
                    )

                    sub_resp = requests.get(
                        sub_url,
                        headers=headers,
                        timeout=15
                    )

                    if sub_resp.status_code == 200:

                        sub_items = (
                            sub_resp.json()
                            .get("_embedded", {})
                            .get("items", [])
                        )

                        for sub_item in sub_items:

                            if (
                                sub_item.get("type") == "dir"
                                and sub_item.get("name", "").upper()
                                == sehir_adi.upper()
                            ):
                                sehir_item = sub_item
                                break

                    if sehir_item:
                        break

        if not sehir_item:
            return [], (
                f"'{sehir_adi}' klasörü bulunamadı."
            )

        sehir_path = sehir_item.get("path")

        bayi_url = (
            "https://cloud-api.yandex.net/v1/disk/"
            f"public/resources?public_key={public_key}"
            f"&path={urllib.parse.quote(sehir_path, safe='/')}"
            "&limit=500"
        )

        bayi_resp = requests.get(
            bayi_url,
            headers=headers,
            timeout=15
        )

        if bayi_resp.status_code != 200:
            return [], "Şehir içeriği okunamadı"

        bayi_items = (
            bayi_resp.json()
            .get("_embedded", {})
            .get("items", [])
        )

        for item in bayi_items:

            if item.get("type") == "dir":

                name = item.get("name")
                path = item.get("path")

                if name and path:

                    bayiler.append(
                        {
                            "name": name,
                            "path": path
                        }
                    )

        return sorted(
            bayiler,
            key=lambda x: x["name"]
        ), None

    except Exception as e:
        return [], str(e)


@st.cache_data(ttl=600, show_spinner=False)
def yandex_bayi_gorseli_getir(
    public_key,
    bayi_path
):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:

        encoded_path = urllib.parse.quote(
            bayi_path,
            safe="/"
        )

        api_url = (
            "https://cloud-api.yandex.net/v1/disk/"
            f"public/resources?public_key={public_key}"
            f"&path={encoded_path}&limit=200"
        )

        resp = requests.get(
            api_url,
            headers=headers,
            timeout=15
        )

        if resp.status_code != 200:
            return None, "Klasör içeriği okunamadı."

        embedded = resp.json().get(
            "_embedded"
        )

        if not embedded:
            return None, "Seçilen bayi klasörü boş."

        items = embedded.get(
            "items",
            []
        )

        for item in items:

            if item.get("type") != "file":
                continue

            name = item.get(
                "name",
                ""
            ).lower()

            if name.endswith(
                (".jpg", ".jpeg", ".png")
            ):

                download_url = item.get("file")

                if download_url:

                    img_resp = requests.get(
                        download_url,
                        headers=headers,
                        timeout=20
                    )

                    if img_resp.status_code == 200:

                        data = np.asarray(
                            bytearray(img_resp.content),
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
            "İçerikte uygun görsel bulunamadı."
        )

    except Exception as e:

        return None, f"Hata: {e}"


# ============================================================
# GİRİŞ
# ============================================================

if "app_password" not in st.secrets:

    st.error(
        "⚠️ Kritik Güvenlik Uyarısı: "
        "'app_password' Streamlit secrets içinde tanımlı değil!"
    )

    st.stop()


app_pass = st.secrets["app_password"]


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


if not st.session_state.authenticated:

    st.title(
        "🔐 Kurumsal Planogram Denetim Sistemi"
    )

    st.markdown(
        """
        <p style='color:gray;font-size:14px;'>
        Developed by Hakan
        </p>
        """,
        unsafe_allow_html=True
    )

    sifre_input = st.text_input(
        "Şifre",
        type="password"
    )

    if st.button(
        "Giriş Yap",
        type="primary"
    ):

        if sifre_input == app_pass:

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

benzerlik_esigi = st.sidebar.slider(
    "Fark Algılama Eşiği",
    0.35,
    0.80,
    0.52,
    0.01
)

supheli_esigi = st.sidebar.slider(
    "Şüpheli Bölge Eşiği",
    0.45,
    0.75,
    0.70,
    0.01
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    📌 Sistem artık:

    • Perspektif hizalama
    • SSIM
    • Renk analizi
    • Kenar analizi
    • ORB özellik eşleme
    • Slot doluluk kontrolü

    kullanmaktadır.
    """
)


# ============================================================
# BAŞLIK
# ============================================================

col1, col2 = st.columns(
    [5, 1]
)

with col1:

    st.title(
        "SİGARA STANDI PLANOGRAM DENETİM SİSTEMİ"
    )

    st.caption(
        "Multi-Metric Planogram Comparison Engine"
    )


with col2:

    if st.button(
        "🚪 Çıkış Yap"
    ):

        st.session_state.authenticated = False

        st.rerun()


st.markdown("---")


# ============================================================
# LOKASYON
# ============================================================

st.subheader(
    "1. Lokasyon ve Bayi Seçimi"
)

dinamik_sehirler, sehir_hata = (
    yandex_sehirleri_getir(
        YANDEX_ROOT_PUBLIC_KEY
    )
)

if sehir_hata:
    st.warning(sehir_hata)


col_s1, col_s2 = st.columns(2)


with col_s1:

    secilen_sehir_adi = st.selectbox(
        "Şehir Seçin",
        options=(
            dinamik_sehirler
            if dinamik_sehirler
            else ["Şehir Bulunamadı"]
        ),
        index=None,
        placeholder="Lütfen bir şehir seçin..."
    )


bayiler_listesi = []
bayi_hata = None


if secilen_sehir_adi:

    bayiler_listesi, bayi_hata = (
        yandex_sehir_bayilerini_getir(
            YANDEX_ROOT_PUBLIC_KEY,
            secilen_sehir_adi
        )
    )


with col_s2:

    if secilen_sehir_adi:

        if bayiler_listesi:

            secilen_bayi_adi = st.selectbox(
                "Bayi Seçin",
                options=[
                    b["name"]
                    for b in bayiler_listesi
                ],
                index=None,
                placeholder="Lütfen bir bayi seçin..."
            )

        else:

            secilen_bayi_adi = st.selectbox(
                "Bayi Seçin",
                ["Bayi Bulunamadı"],
                index=None
            )

    else:

        secilen_bayi_adi = st.selectbox(
            "Bayi Seçin",
            ["Önce Şehir Seçmelisiniz"],
            index=None,
            disabled=True
        )


# ============================================================
# BAYİ PATH
# ============================================================

secilen_bayi_path = ""


if (
    secilen_sehir_adi
    and secilen_bayi_adi
    and bayiler_listesi
):

    for bayi in bayiler_listesi:

        if bayi["name"] == secilen_bayi_adi:

            secilen_bayi_path = bayi["path"]

            break


if st.button(
    "🔄 Önbelleği Yenile"
):

    yandex_sehirleri_getir.clear()
    yandex_sehir_bayilerini_getir.clear()
    yandex_bayi_gorseli_getir.clear()

    st.toast(
        "Önbellek temizlendi!",
        icon="🔄"
    )

    st.rerun()


# ============================================================
# REFERANS
# ============================================================

ref_img = None
hata_mesaji = None


if secilen_bayi_path:

    with st.spinner(
        "Referans planogram yükleniyor..."
    ):

        ref_img, hata_mesaji = (
            yandex_bayi_gorseli_getir(
                YANDEX_ROOT_PUBLIC_KEY,
                secilen_bayi_path
            )
        )


if hata_mesaji:
    st.warning(hata_mesaji)


# ============================================================
# FOTOĞRAFLAR
# ============================================================

col_up1, col_up2 = st.columns(2)


with col_up1:

    st.subheader(
        "2. Dijital Planogram"
    )

    if ref_img is not None:

        st.success(
            "✅ Referans planogram hazır"
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


with col_up2:

    st.subheader(
        "3. Saha Fotoğrafı"
    )

    curr_file = None
    curr_img = None

    if secilen_bayi_path:

        curr_file = st.file_uploader(
            "Saha fotoğrafını yükleyin",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="curr"
        )

        if curr_file is not None:

            curr_bytes = np.asarray(
                bytearray(
                    curr_file.getvalue()
                ),
                dtype=np.uint8
            )

            raw_curr_img = cv2.imdecode(
                curr_bytes,
                cv2.IMREAD_COLOR
            )

            curr_img = resmi_boyutlandir(
                raw_curr_img,
                1600
            )

            if curr_img is not None:

                st.success(
                    "✅ Saha fotoğrafı işlendi"
                )

                st.image(
                    curr_img,
                    channels="BGR",
                    use_container_width=True
                )


# ============================================================
# ANALİZ
# ============================================================

st.markdown("---")

st.subheader(
    "4. Planogram Analizi"
)


if (
    secilen_bayi_path
    and ref_img is not None
    and curr_img is not None
):

    analiz_baslat = st.button(
        "🚀 PLANOGRAM FARKLARINI BUL",
        type="primary",
        use_container_width=True
    )

    if analiz_baslat:

        with st.spinner(
            "Fotoğraflar hizalanıyor ve 77 slot analiz ediliyor..."
        ):

            # ------------------------------------------------
            # PERSPEKTİF HİZALAMA
            # ------------------------------------------------

            aligned_img, hizalandi, inlier = (
                perspektif_hizala(
                    ref_img,
                    curr_img
                )
            )

            # Her durumda referans boyutuna getir
            curr_resized = cv2.resize(
                aligned_img,
                (
                    ref_img.shape[1],
                    ref_img.shape[0]
                ),
                interpolation=cv2.INTER_AREA
            )

            result_img = curr_resized.copy()

            # ------------------------------------------------
            # SLOT LİSTESİ
            # ------------------------------------------------

            slots = slot_koordinatlari(
                ref_img
            )

            farklar = []
            supheliler = []
            uyumlu = []

            # ------------------------------------------------
            # SLOT ANALİZİ
            # ------------------------------------------------

            for slot_info in slots:

                x1 = slot_info["x1"]
                y1 = slot_info["y1"]
                x2 = slot_info["x2"]
                y2 = slot_info["y2"]

                ref_slot = ref_img[
                    y1:y2,
                    x1:x2
                ]

                curr_slot = curr_resized[
                    y1:y2,
                    x1:x2
                ]

                if (
                    ref_slot.size == 0
                    or curr_slot.size == 0
                ):
                    continue

                analiz = slot_karsilastir(
                    ref_slot,
                    curr_slot
                )

                score = analiz["score"]

                slot_info = {
                    **slot_info,
                    **analiz
                }

                # --------------------------------------------
                # KARAR
                # --------------------------------------------

                if score < benzerlik_esigi:

                    farklar.append(
                        slot_info
                    )

                    cv2.rectangle(
                        result_img,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        5
                    )

                    cv2.putText(
                        result_img,
                        f"FARK R{slot_info['raf']}/S{slot_info['slot']}",
                        (
                            x1,
                            max(25, y1 - 8)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2
                    )

                elif score < supheli_esigi:

                    supheliler.append(
                        slot_info
                    )

                    cv2.rectangle(
                        result_img,
                        (x1, y1),
                        (x2, y2),
                        (0, 165, 255),
                        4
                    )

                    cv2.putText(
                        result_img,
                        f"S? R{slot_info['raf']}/S{slot_info['slot']}",
                        (
                            x1,
                            max(25, y1 - 8)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 165, 255),
                        2
                    )

                else:

                    uyumlu.append(
                        slot_info
                    )

                    cv2.rectangle(
                        result_img,
                        (x1, y1),
                        (x2, y2),
                        (0, 180, 0),
                        2
                    )

            # ------------------------------------------------
            # SONUÇ
            # ------------------------------------------------

            toplam_slot = len(slots)

            fark_sayisi = len(farklar)

            supheli_sayisi = len(
                supheliler
            )

            uyumlu_sayisi = len(
                uyumlu
            )

            if toplam_slot > 0:

                uyum_orani = (
                    uyumlu_sayisi
                    /
                    toplam_slot
                ) * 100

            else:

                uyum_orani = 0

            # ------------------------------------------------
            # ÜST BİLGİ
            # ------------------------------------------------

            cv2.rectangle(
                result_img,
                (0, 0),
                (
                    result_img.shape[1],
                    115
                ),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                result_img,
                f"Bayi: {secilen_bayi_adi}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2
            )

            cv2.putText(
                result_img,
                (
                    f"FARK: {fark_sayisi} | "
                    f"SÜPHELİ: {supheli_sayisi} | "
                    f"UYUMLU: {uyumlu_sayisi}"
                ),
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

            cv2.putText(
                result_img,
                f"UYUM: %{uyum_orani:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # SESSION
            # ------------------------------------------------

            st.session_state.result_img = result_img

            st.session_state.farklar = farklar

            st.session_state.supheliler = (
                supheliler
            )

            st.session_state.uyumlu = uyumlu

            st.session_state.uyum_orani = (
                uyum_orani
            )

            st.session_state.hizalandi = (
                hizalandi
            )

            st.session_state.inlier = (
                inlier
            )

            st.session_state.analiz_yapildi = True


# ============================================================
# SONUÇLAR
# ============================================================

if (
    st.session_state.get(
        "analiz_yapildi",
        False
    )
):

    st.markdown("---")

    st.subheader(
        "📊 Denetim Sonucu"
    )

    # --------------------------------------------------------
    # METRİKLER
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Toplam Slot",
        len(
            st.session_state.get(
                "uyumlu",
                []
            )
        )
        +
        len(
            st.session_state.get(
                "farklar",
                []
            )
        )
        +
        len(
            st.session_state.get(
                "supheliler",
                []
            )
        )
    )

    c2.metric(
        "🔴 Farklı",
        len(
            st.session_state.get(
                "farklar",
                []
            )
        )
    )

    c3.metric(
        "🟠 Şüpheli",
        len(
            st.session_state.get(
                "supheliler",
                []
            )
        )
    )

    c4.metric(
        "🟢 Uyumlu",
        len(
            st.session_state.get(
                "uyumlu",
                []
            )
        )
    )

    st.metric(
        "📈 Planogram Uyum Oranı",
        f"%{st.session_state.get('uyum_orani', 0):.1f}"
    )

    # --------------------------------------------------------
    # HİZALAMA DURUMU
    # --------------------------------------------------------

    if st.session_state.get(
        "hizalandi",
        False
    ):

        st.success(
            "✅ Fotoğraflar perspektif olarak hizalandı."
            f" ({st.session_state.get('inlier', 0)} "
            "geçerli özellik eşleşmesi)"
        )

    else:

        st.warning(
            "⚠️ Fotoğraflar otomatik perspektif "
            "olarak tam hizalanamadı. "
            "Slot analizi yine gerçekleştirildi."
        )

    # --------------------------------------------------------
    # GÖRSEL
    # --------------------------------------------------------

    st.subheader(
        "🔍 Fark Analiz Görseli"
    )

    sonuc_gorsel_genisligi = st.slider(
        "Görsel Boyutu",
        500,
        2000,
        1100,
        100
    )

    st.image(
        st.session_state.result_img,
        channels="BGR",
        width=sonuc_gorsel_genisligi
    )

    # --------------------------------------------------------
    # FARK TABLOSU
    # --------------------------------------------------------

    farklar = st.session_state.get(
        "farklar",
        []
    )

    if farklar:

        st.markdown("---")

        st.subheader(
            "🔴 Tespit Edilen Farklar"
        )

        for item in farklar:

            st.error(
                f"Raf {item['raf']} / "
                f"Slot {item['slot']}  →  "
                f"Benzerlik %{item['score'] * 100:.1f}"
            )

            st.caption(
                f"SSIM: %{item['ssim'] * 100:.1f} | "
                f"Renk: %{item['hist'] * 100:.1f} | "
                f"Kenar: %{item['edge'] * 100:.1f} | "
                f"ORB: %{item['orb'] * 100:.1f}"
            )

    else:

        st.success(
            "🔴 Belirlenen eşik altında fark bulunamadı."
        )

    # --------------------------------------------------------
    # ŞÜPHELİLER
    # --------------------------------------------------------

    supheliler = st.session_state.get(
        "supheliler",
        []
    )

    if supheliler:

        st.markdown("---")

        st.subheader(
            "🟠 Kontrol Edilmesi Gereken Şüpheli Slotlar"
        )

        for item in supheliler:

            st.warning(
                f"Raf {item['raf']} / "
                f"Slot {item['slot']} → "
                f"Benzerlik %{item['score'] * 100:.1f}"
            )

    # --------------------------------------------------------
    # RAPOR İNDİR
    # --------------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".jpg",
        st.session_state.result_img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if success:

        st.download_button(
            label="📥 Kurumsal Denetim Raporunu İndir",
            data=encoded_image.tobytes(),
            file_name=(
                f"{secilen_bayi_adi.replace(' ', '_')}"
                "_planogram_rapor.jpg"
            ),
            mime="image/jpeg",
            use_container_width=True
        )


# ============================================================
# ALT BİLGİ
# ============================================================

st.markdown(
    """
    <br>
    <p style='text-align:center;color:gray;'>
    Kurumsal Planogram Denetim Sistemi —
    Developed by Hakan
    </p>
    """,
    unsafe_allow_html=True
)
