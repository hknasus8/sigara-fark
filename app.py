import streamlit as st
import cv2
import numpy as np
import os
import requests
import urllib.parse

st.set_page_config(
    page_title="Sigara Standı Akıllı Denetim Sistemi",
    page_icon="🚬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.components.v1.html(
    """
    <script>
        const doc = window.parent.document;
        doc.documentElement.lang = 'tr';
        doc.documentElement.setAttribute('translate', 'no');
    </script>
    """,
    height=0,
    width=0
)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppToolbar {visibility: hidden; display: none !important;}
    [data-testid="stHeader"] {visibility: hidden; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden; display: none !important;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

def resmi_boyutlandir(img, max_genislik=1000):
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_genislik:
        oran = max_genislik / float(w)
        yeni_yukseklik = int(h * oran)
        return cv2.resize(img, (max_genislik, yeni_yukseklik), interpolation=cv2.INTER_AREA)
    return img

if "app_password" not in st.secrets:
    st.error("⚠️ Kritik Güvenlik Uyarısı: 'app_password' Streamlit secrets içinde tanımlı değil!")
    st.stop()

app_pass = st.secrets["app_password"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    logo_yolu = "logo.jpg"
    if os.path.exists(logo_yolu):
        st.image(logo_yolu, width=180)
    else:
        try:
            st.image("https://raw.githubusercontent.com/hknasus8/sigara-fark/main/logo.jpg", width=180)
        except Exception:
            pass

    st.title("🔐 Sigara Standı Akıllı Denetim Sistemi - Giriş")
    st.markdown("<p style='color: gray; font-size: 14px; margin-top: -15px;'>Developed by Hakan</p>", unsafe_allow_html=True)
    
    sifre_input = st.text_input("Şifre", type="password")
    if st.button("Giriş Yap", type="primary"):
        if sifre_input == app_pass:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre!")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Uygulama Ayarları")
min_area_val = st.sidebar.slider("Minimum Eksik Boyutu (Hassasiyet)", 50, 2000, 150, step=25)
fark_esigi = st.sidebar.slider("Piksel Fark Eşiği (Yoğunluk)", 20, 100, 40, step=5)

YANDEX_ROOT_PUBLIC_KEY = "https://disk.yandex.com.tr/d/JXJNYBDAk6fePw"

@st.cache_data(ttl=3600, show_spinner=False)
def yandex_sehirleri_getir(public_key):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        # Kök dizin için path=/ parametresi açıkça eklendi
        api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path=%2F&limit=200"
        resp = requests.get(api_url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return [], f"HTTP {resp.status_code} - {resp.text[:100]}"
        data = resp.json().get("_embedded", {})
        items = data.get("items", [])
        sehirler = []
        for item in items:
            if item.get("type") == "dir":
                name = item.get("name")
                if name:
                    sehirler.append({"name": name, "path": f"/{name}"})
        return sorted(sehirler, key=lambda x: x["name"]), None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=1800, show_spinner=False)
def yandex_bayileri_getir(public_key, sehir_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        encoded_path = urllib.parse.quote(sehir_path, safe='/')
        api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={encoded_path}&limit=500"
        resp = requests.get(api_url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return [], f"HTTP {resp.status_code} - {resp.text[:100]}"
        data = resp.json().get("_embedded", {})
        items = data.get("items", [])
        bayiler = []
        for item in items:
            if item.get("type") == "dir":
                name = item.get("name")
                if name:
                    bayiler.append({"name": name, "path": f"{sehir_path}/{name}"})
        return sorted(bayiler, key=lambda x: x["name"]), None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=600, show_spinner=False)
def yandex_bayi_gorseli_getir(public_key, bayi_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        encoded_path = urllib.parse.quote(bayi_path, safe='/')
        api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={encoded_path}&limit=200"
        resp = requests.get(api_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None, f"Klasör içeriği okunamadı (HTTP {resp.status_code})."

        final_embedded = resp.json().get("_embedded")
        if not final_embedded:
            return None, "Seçilen bayi klasörü boş."
            
        sub_items = final_embedded.get("items", [])
        gorsel_download_url = None
        for sub_item in sub_items:
            if sub_item.get("type") == "file":
                file_name = sub_item.get("name", "").lower()
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    gorsel_download_url = sub_item.get("file")
                    break

        if gorsel_download_url:
            img_resp = requests.get(gorsel_download_url, headers=headers, timeout=15)
            if img_resp.status_code == 200:
                image_bytes = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    return resmi_boyutlandir(img), None

        return None, "Klasör bulundu ancak içinde .jpg/.png görsel yok."
    except Exception as e:
        return None, f"Bağlantı hatası: {e}"

col_baslik, col_cikis = st.columns([5, 1])
with col_baslik:
    st.title("SİGARA STANDI AKILLI DENETİM SİSTEMİ")
    st.markdown("<p style='color: gray; font-size: 14px; margin-top: -15px;'>Developed by Hakan</p>", unsafe_allow_html=True)

with col_cikis:
    st.write("") 
    if st.button("🚪 Çıkış Yap", type="secondary"):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("---")
st.subheader("1. Lokasyon ve Bayi Seçimi")

sehirler_listesi, err = yandex_sehirleri_getir(YANDEX_ROOT_PUBLIC_KEY)
if err or not sehirler_listesi:
    st.error(f"Yandex Disk şehirler yüklenemedi: {err}")
    st.stop()

col_s1, col_s2 = st.columns(2)

with col_s1:
    secilen_sehir_adi = st.selectbox("Şehir Seçin", [s["name"] for s in sehirler_listesi])
    secilen_sehir_path = next(s["path"] for s in sehirler_listesi if s["name"] == secilen_sehir_adi)

bayiler_listesi, b_err = yandex_bayileri_getir(YANDEX_ROOT_PUBLIC_KEY, secilen_sehir_path)

with col_s2:
    if b_err:
        st.error(f"Bayiler yüklenemedi: {b_err}")
        bayiler_listesi = []
    
    secilen_bayi_adi = st.selectbox("Bayi Seçin", [b["name"] for b in bayiler_listesi] if bayiler_listesi else ["Bayi Bulunamadı"])
    secilen_bayi_path = next((b["path"] for b in bayiler_listesi if b["name"] == secilen_bayi_adi), "")

st.markdown(f"**Seçilen Konum:** `{secilen_sehir_adi} / {secilen_bayi_adi}`")

if st.button("🔄 Önbelleği Yenile"):
    yandex_sehirleri_getir.clear()
    yandex_bayileri_getir.clear()
    yandex_bayi_gorseli_getir.clear()
    st.toast("Önbellek temizlendi!", icon="🔄")
    st.rerun()

st.markdown("---")

ref_img = None
hata_mesaji = None
if secilen_bayi_path:
    with st.spinner(f"'{secilen_bayi_adi}' için Yandex Disk'ten görsel yükleniyor..."):
        ref_img, hata_mesaji = yandex_bayi_gorseli_getir(YANDEX_ROOT_PUBLIC_KEY, secilen_bayi_path)

col_up1, col_up2 = st.columns(2)
with col_up1:
    st.subheader("2. Referans (İdeal) Görsel")
    if ref_img is not None:
        st.success(f"✅ '{secilen_bayi_adi}' Yandex'ten Yüklendi")
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.warning(f"⚠️ Görsel yüklenemedi. Nedeni: {hata_mesaji}")

with col_up2:
    st.subheader("3. Sahadan Gelen Görsel")
    curr_file = st.file_uploader("Fotoğraf yükleyin", type=["jpg", "jpeg", "png"], key="curr")
    curr_img = None
    if curr_file is not None:
        curr_bytes = np.asarray(bytearray(curr_file.read()), dtype=np.uint8)
        raw_curr_img = cv2.imdecode(curr_bytes, cv2.IMREAD_COLOR)
        curr_img = resmi_boyutlandir(raw_curr_img)
        st.success("✅ Fotoğraf yüklendi")
        st.image(curr_img, channels="BGR", use_container_width=True)

st.markdown("---")
st.subheader("4. Stand Kapasite Ayarı")
ideal_urun_sayisi = st.number_input("Standda Bulunması Gereken Toplam Ürün (Slot) Sayısı", min_value=1, value=50, step=1)
st.markdown("---")

if "result_img" not in st.session_state:
    st.session_state.result_img = None
if "eksik_sayisi" not in st.session_state:
    st.session_state.eksik_sayisi = 0
if "raf_yuzdesi" not in st.session_state:
    st.session_state.raf_yuzdesi = 100.0
if "analiz_yapildi" not in st.session_state:
    st.session_state.analiz_yapildi = False

if ref_img is not None and curr_file is not None and curr_img is not None:
    if st.button("Hassas Farkı Analiz Et", type="primary"):
        with st.spinner("Gelişmiş hibrit matris ve piksel analizi yapılıyor..."):
            if ref_img.shape[:2] != curr_img.shape[:2]:
                curr_img = cv2.resize(curr_img, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_AREA)

            gray_ref = cv2.GaussianBlur(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            gray_curr = cv2.GaussianBlur(cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)

            diff = cv2.absdiff(gray_ref, gray_curr)
            _, thresh = cv2.threshold(diff, fark_esigi, 255, cv2.THRESH_BINARY)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            img_h, img_w = curr_img.shape[:2]
            
            for c in contours:
                if cv2.contourArea(c) > min_area_val:
                    x, y, w, h = cv2.boundingRect(c)
                    if (img_w * 0.01 < x < img_w * 0.99) and (img_h * 0.02 < y < img_h * 0.98):
                        boxes.append([x, y, x + w, y + h])

            def non_max_suppression(boxes, overlapThresh=0.15):
                if not boxes: return []
                boxes = np.array(boxes)
                pick = []
                x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                idxs = np.argsort(y2)
                while len(idxs) > 0:
                    last = len(idxs) - 1
                    i = idxs[last]
                    pick.append(i)
                    xx1 = np.maximum(x1[i], x1[idxs[:last]])
                    yy1 = np.maximum(y1[i], y1[idxs[:last]])
                    xx2 = np.minimum(x2[i], x2[idxs[:last]])
                    yy2 = np.minimum(y2[i], y2[idxs[:last]])
                    w = np.maximum(0, xx2 - xx1 + 1)
                    h = np.maximum(0, yy2 - yy1 + 1)
                    overlap = (w * h) / area[idxs[:last]]
                    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
                return boxes[pick].astype("int")

            filtered_boxes = non_max_suppression(boxes)
            result_img = curr_img.copy()
            eksik_sayisi = len(filtered_boxes)
            hesaplanan_yuzde = max(0.0, 100.0 - ((eksik_sayisi / max(1, ideal_urun_sayisi)) * 100.0))

            for idx, (startX, startY, endX, endY) in enumerate(filtered_boxes, 1):
                cv2.rectangle(result_img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(result_img, f"#{idx}", (startX + 3, startY + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.rectangle(result_img, (0, 0), (img_w, 100), (0, 0, 0), -1)
            cv2.putText(result_img, f"Bayi: {secilen_bayi_adi}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_img, f"Eksik Alan: {eksik_sayisi} | Raf Uygunluk: %{hesaplanan_yuzde:.1f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if eksik_sayisi > 0 else (0, 255, 0), 2)

            st.session_state.result_img = result_img
            st.session_state.eksik_sayisi = eksik_sayisi
            st.session_state.raf_yuzdesi = hesaplanan_yuzde
            st.session_state.analiz_yapildi = True

    if st.session_state.analiz_yapildi and st.session_state.result_img is not None:
        st.subheader("Tespit Edilen Eksikler ve Detaylı Rapor")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("📊 Hesaplanan Raf Doğruluk Oranı", f"%{st.session_state.raf_yuzdesi:.1f}")
        col_m2.metric("⚠️ Tespit Edilen Eksik/Boşluk Alan", f"{st.session_state.eksik_sayisi} Adet")
        
        sonuc_gorsel_genisligi = st.slider("🔍 Sonuç Görseli Boyutunu Ayarla (Piksel)", 300, 2000, 800, step=100)
        st.image(st.session_state.result_img, channels="BGR", width=sonuc_gorsel_genisligi)

        success, encoded_image = cv2.imencode(".jpg", st.session_state.result_img)
        if success:
            st.download_button(
                label="📥 Sonuç Fotoğrafını İndir",
                data=encoded_image.tobytes(),
                file_name=f"{secilen_bayi_adi.replace(' ', '_')}_analiz_sonucu.jpg",
                mime="image/jpeg"
            )
else:
    st.info("ℹ️ Analiz yapabilmek için lütfen yukarıdan şehir ve bayinizi seçin, ardından sağdaki alandan **Sahadan Gelen Fotoğrafı** yükleyin.")

st.markdown("<br><p style='text-align: center; color: gray;'>Developed by Hakan</p>", unsafe_allow_html=True)
