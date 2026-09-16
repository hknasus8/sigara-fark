import streamlit as st
import cv2
import numpy as np
import requests
import urllib.parse
import easyocr
from difflib import SequenceMatcher

st.set_page_config(
    page_title="Sigara Standı Planogram Denetim Sistemi",
    page_icon="🚬",
    layout="wide",
    initial_sidebar_state="expanded"
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

@st.cache_resource
def ocr_okuyucu_yukle():
    return easyocr.Reader(['tr', 'en'], gpu=False)

with st.spinner("AI Planogram motoru hazırlanıyor..."):
    reader = ocr_okuyucu_yukle()

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
    st.title("🔐 Planogram & Raf Denetim Sistemi - Giriş")
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
st.sidebar.header("Planogram Eşleşme Ayarları")
kolon_sayisi = st.sidebar.slider("Her Raftaki Slot / Ürün Sayısı", 8, 16, 12, step=1)
benzerlik_farki_esigi = st.sidebar.slider("Planogram Uyumsuzluk Hassasiyeti", 0.1, 0.8, 0.35, step=0.05)

YANDEX_ROOT_PUBLIC_KEY = "https://disk.yandex.com.tr/d/ikCHPwREiCVv_g"

@st.cache_data(ttl=600, show_spinner=False)
def yandex_sehirleri_getir(public_key):
    headers = {"User-Agent": "Mozilla/5.0"}
    sehirler = []
    try:
        root_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&limit=200"
        resp = requests.get(root_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return [], f"Kök Dizin Okunamadı (HTTP {resp.status_code})"

        root_items = resp.json().get("_embedded", {}).get("items", [])
        for item in root_items:
            if item.get("type") == "dir":
                b_name = item.get("name")
                if b_name.upper() in ["BAYİ", "BAYI"]:
                    bayi_path = item.get("path")
                    sub_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={urllib.parse.quote(bayi_path, safe='/')}&limit=200"
                    sub_resp = requests.get(sub_url, headers=headers, timeout=15)
                    if sub_resp.status_code == 200:
                        for sub_item in sub_resp.json().get("_embedded", {}).get("items", []):
                            if sub_item.get("type") == "dir":
                                sehirler.append(sub_item.get("name"))
                else:
                    sehirler.append(b_name)
        return sorted(list(set(sehirler))), None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=600, show_spinner=False)
def yandex_sehir_bayilerini_getir(public_key, sehir_adi):
    headers = {"User-Agent": "Mozilla/5.0"}
    bayiler = []
    try:
        root_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&limit=200"
        resp = requests.get(root_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return [], "Kök Dizin Okunamadı"

        root_items = resp.json().get("_embedded", {}).get("items", [])
        sehir_item_found = None

        for item in root_items:
            if item.get("type") == "dir" and item.get("name", "").upper() == sehir_adi.upper():
                sehir_item_found = item
                break

        if not sehir_item_found:
            for item in root_items:
                if item.get("type") == "dir" and item.get("name", "").upper() in ["BAYİ", "BAYI"]:
                    bayi_path = item.get("path")
                    sub_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={urllib.parse.quote(bayi_path, safe='/')}&limit=200"
                    sub_resp = requests.get(sub_url, headers=headers, timeout=15)
                    if sub_resp.status_code == 200:
                        for sub_item in sub_resp.json().get("_embedded", {}).get("items", []):
                            if sub_item.get("type") == "dir" and sub_item.get("name", "").upper() == sehir_adi.upper():
                                sehir_item_found = sub_item
                                break
                    if sehir_item_found:
                        break

        if not sehir_item_found:
            return [], f"'{sehir_adi}' klasörü bulunamadı."

        sehir_path = sehir_item_found.get("path")
        bayi_list_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={urllib.parse.quote(sehir_path, safe='/')}&limit=500"
        bayi_resp = requests.get(bayi_list_url, headers=headers, timeout=15)
        if bayi_resp.status_code != 200:
            return [], "Şehir içeriği okunamadı"

        bayi_items = bayi_resp.json().get("_embedded", {}).get("items", [])
        for b_item in bayi_items:
            if b_item.get("type") == "dir":
                b_name = b_item.get("name")
                b_path = b_item.get("path")
                if b_name and b_path:
                    bayiler.append({"name": b_name, "path": b_path})

        return sorted(bayiler, key=lambda x: x["name"]), None
    except Exception as e:
        return [], str(e)

@st.cache_data(ttl=600, show_spinner=False)
def yandex_bayi_gorseli_getir(public_key, bayi_path):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        encoded_path = urllib.parse.quote(bayi_path, safe='/')
        api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={public_key}&path={encoded_path}&limit=200"
        resp = requests.get(api_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None, "Klasör içeriği okunamadı."

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

        return None, "İçerikte uygun görsel bulunamadı."
    except Exception as e:
        return None, f"Hata: {e}"

col_baslik, col_cikis = st.columns([5, 1])
with col_baslik:
    st.title("SİGARA STANDI PLANOGRAM DENETİM SİSTEMİ")
    st.markdown("<p style='color: gray; font-size: 14px; margin-top: -15px;'>Planogram Compliance & Product Mismatch Engine</p>", unsafe_allow_html=True)

with col_cikis:
    st.write("")
    if st.button("🚪 Çıkış Yap", type="secondary"):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("---")
st.subheader("1. Lokasyon ve Bayi Seçimi")

dinamik_sehirler, sehir_hata = yandex_sehirleri_getir(YANDEX_ROOT_PUBLIC_KEY)
col_s1, col_s2 = st.columns(2)

with col_s1:
    secilen_sehir_adi = st.selectbox(
        "Şehir Seçin",
        options=dinamik_sehirler if dinamik_sehirler else ["Şehir Bulunamadı"],
        index=None,
        placeholder="Lütfen bir şehir seçin..."
    )

bayiler_listesi = []
bayi_hata = None
if secilen_sehir_adi:
    bayiler_listesi, bayi_hata = yandex_sehir_bayilerini_getir(YANDEX_ROOT_PUBLIC_KEY, secilen_sehir_adi)

with col_s2:
    if secilen_sehir_adi:
        if bayiler_listesi:
            secilen_bayi_adi = st.selectbox(
                "Bayi Seçin",
                options=[b["name"] for b in bayiler_listesi],
                index=None,
                placeholder="Lütfen bir bayi seçin..."
            )
        else:
            secilen_bayi_adi = st.selectbox("Bayi Seçin", ["Bayi Bulunamadı"], index=None)
    else:
        secilen_bayi_adi = st.selectbox("Bayi Seçin", ["Önce Şehir Seçmelisiniz"], index=None, disabled=True)

secilen_bayi_path = ""
if secilen_sehir_adi and secilen_bayi_adi and bayiler_listesi and secilen_bayi_adi != "Bayi Bulunamadı":
    for b in bayiler_listesi:
        if b["name"] == secilen_bayi_adi:
            secilen_bayi_path = b["path"]
            break

if st.button("🔄 Önbelleği Yenile"):
    yandex_sehirleri_getir.clear()
    yandex_sehir_bayilerini_getir.clear()
    yandex_bayi_gorseli_getir.clear()
    st.toast("Önbellek temizlendi!", icon="🔄")
    st.rerun()

st.markdown("---")

ref_img = None
hata_mesaji = None
if secilen_bayi_path:
    with st.spinner(f"'{secilen_bayi_adi}' için Planogram Referansı Yükleniyor..."):
        ref_img, hata_mesaji = yandex_bayi_gorseli_getir(YANDEX_ROOT_PUBLIC_KEY, secilen_bayi_path)

col_up1, col_up2 = st.columns(2)
with col_up1:
    st.subheader("2. Dijital Planogram (Referans Şablon)")
    if secilen_bayi_path and ref_img is not None:
        st.success(f"✅ Planogram Şablonu Hazır")
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.info("ℹ️ Şehir ve bayi seçin.")

with col_up2:
    st.subheader("3. Saha Fotoğrafı")
    if secilen_bayi_path:
        curr_file = st.file_uploader("Saha fotoğrafını yükleyin", type=["jpg", "jpeg", "png"], key="curr")
        curr_img = None
        if curr_file is not None:
            curr_bytes = np.asarray(bytearray(curr_file.read()), dtype=np.uint8)
            raw_curr_img = cv2.imdecode(curr_bytes, cv2.IMREAD_COLOR)
            curr_img = resmi_boyutlandir(raw_curr_img)
            st.success("✅ Saha fotoğrafı işlendi")
            st.image(curr_img, channels="BGR", use_container_width=True)
    else:
        st.file_uploader("Saha fotoğrafını yükleyin", type=["jpg", "jpeg", "png"], key="curr_disabled", disabled=True)

st.markdown("---")
st.subheader("4. Stand Kapasite Ayarı")
ideal_urun_sayisi = st.number_input("Standda Bulunması Gereken Toplam Slot Sayısı", min_value=1, value=60, step=1)
st.markdown("---")

if "result_img" not in st.session_state:
    st.session_state.result_img = None
if "uyumsuz_sayisi" not in st.session_state:
    st.session_state.uyumsuz_sayisi = 0
if "raf_yuzdesi" not in st.session_state:
    st.session_state.raf_yuzdesi = 100.0
if "aksiyon_maddeleri" not in st.session_state:
    st.session_state.aksiyon_maddeleri = []
if "analiz_yapildi" not in st.session_state:
    st.session_state.analiz_yapildi = False

if secilen_sehir_adi and secilen_bayi_adi and ref_img is not None and 'curr_file' in locals() and curr_file is not None and 'curr_img' in locals() and curr_img is not None:
    if st.button("🚀 Planogram Uyumsuzluklarını Denetle", type="primary"):
        with st.spinner("Referans planogram ile saha fotoğrafı slot bazlı karşılaştırılıyor..."):

            img_h, img_w = ref_img.shape[:2]
            curr_resized = cv2.resize(curr_img, (img_w, img_h), interpolation=cv2.INTER_AREA)
            result_img = curr_resized.copy()

            uyumsuz_slotlar = []
            aksiyon_maddeleri = []

            raf_yuksekligi = img_h / 6.0
            slot_genisligi = img_w / float(kolon_sayisi)

            # PLANOGRAM & SLOT BAZLI YAPISAL KARŞILAŞTIRMA (Yer Değişikliği / Uyumsuzluk Tespiti)
            for raf_idx in range(6):
                y_baslangic = int(raf_idx * raf_yuksekligi)
                y_bitis = int((raf_idx + 1) * raf_yuksekligi)

                for col_idx in range(kolon_sayisi):
                    x_baslangic = int(col_idx * slot_genisligi)
                    x_bitis = int((col_idx + 1) * slot_genisligi)

                    # Referans slot (İdeal Planogram)
                    ref_slot = ref_img[y_baslangic + int(raf_yuksekligi*0.1): y_bitis - int(raf_yuksekligi*0.1), 
                                       x_baslangic + 5: x_bitis - 5]
                    
                    # Saha slotu (Gerçek Durum)
                    curr_slot = curr_resized[y_baslangic + int(raf_yuksekligi*0.1): y_bitis - int(raf_yuksekligi*0.1), 
                                             x_baslangic + 5: x_bitis - 5]

                    if ref_slot.size == 0 or curr_slot.size == 0:
                        continue

                    # Gri tonlamaya çevirip yapısal farka (Structural Difference / Mean Absolute Difference) bakıyoruz
                    ref_gray = cv2.cvtColor(ref_slot, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(curr_slot, cv2.COLOR_BGR2GRAY)

                    # Normalize edilmiş mutlak fark
                    fark = np.mean(cv2.absdiff(ref_gray, curr_gray)) / 255.0

                    # Eğer fark belirlenen eşikten büyükse, bu slotta planogram dışı ürün var veya yer değişmiş demektir
                    if fark > benzerlik_farki_esigi:
                        uyumsuz_slotlar.append([x_baslangic, y_baslangic, x_bitis, y_bitis])
                        aksiyon_maddeleri.append(f"{raf_idx + 1}. Raf, {col_idx + 1}. Slot noktasında planogram dışı ürün/yerleşim uyumsuzluğu tespit edildi.")

            uyumsuz_sayisi = len(uyumsuz_slotlar)
            for idx, (startX, startY, endX, endY) in enumerate(uyumsuz_slotlar, 1):
                cv2.rectangle(result_img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(result_img, f"Uyumsuz #{idx}", (startX + 2, startY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 2)

            kirmizi_x_baslangic_y = int(raf_yuksekligi * 6)
            if kirmizi_x_baslangic_y < img_h:
                overlay = result_img.copy()
                cv2.rectangle(overlay, (0, kirmizi_x_baslangic_y), (img_w, img_h), (0, 0, 50), -1)
                cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0, result_img)

                for y_pos in range(kirmizi_x_baslangic_y + int(raf_yuksekligi/2), img_h, int(raf_yuksekligi)):
                    center_x = int(img_w / 2)
                    cv2.putText(result_img, "X - KONTROL EDILMEDI", (center_x - 150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.line(result_img, (center_x - 180, y_pos - 20), (center_x - 160, y_pos + 10), (0, 0, 255), 3)

            cv2.rectangle(result_img, (0, 0), (img_w, 100), (0, 0, 0), -1)
            cv2.putText(result_img, f"Bayi: {secilen_bayi_adi}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_img, f"Planogram Uyumsuzluk: {uyumsuz_sayisi} Slot", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            hesaplanan_yuzde = max(0.0, 100.0 - ((uyumsuz_sayisi / max(1, ideal_urun_sayisi)) * 100.0))

            st.session_state.result_img = result_img
            st.session_state.uyumsuz_sayisi = uyumsuz_sayisi
            st.session_state.raf_yuzdesi = hesaplanan_yuzde
            st.session_state.aksiyon_maddeleri = aksiyon_maddeleri
            st.session_state.analiz_yapildi = True

    if st.session_state.analiz_yapildi and st.session_state.result_img is not None:
        st.subheader("📋 Planogram Denetim Raporu ve Saha Talimatları")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("📊 Planogram Uyum Oranı", f"%{st.session_state.raf_yuzdesi:.1f}")
        col_m2.metric("⚠️ Uyumsuz/Yer Değişen Slot", f"{st.session_state.uyumsuz_sayisi} Adet")

        if st.session_state.aksiyon_maddeleri:
            st.markdown("> 🔔 **Saha Personeline Gönderilecek Anlık Düzeltme Talimatları:**")
            for aksiyon in st.session_state.aksiyon_maddeleri:
                st.markdown(f"- ⚠️ {aksiyon}")
        else:
            st.success("✅ Tebrikler! Saha fotoğrafı referans planogram şablonu ile birebir uyumlu.")

        sonuc_gorsel_genisligi = st.slider("🔍 Denetim Görseli Boyutunu Ayarla", 300, 2000, 800, step=100)
        st.image(st.session_state.result_img, channels="BGR", width=sonuc_gorsel_genisligi)

        success, encoded_image = cv2.imencode(".jpg", st.session_state.result_img)
        if success:
            st.download_button(
                label="📥 Planogram Raporunu İndir",
                data=encoded_image.tobytes(),
                file_name=f"{secilen_bayi_adi.replace(' ', '_')}_planogram_rapor.jpg",
                mime="image/jpeg"
            )
else:
    st.info("ℹ️ Planogram analizi için şehir, bayi seçin ve sahadan gelen fotoğrafı yükleyin.")

st.markdown("<br><p style='text-align: center; color: gray;'>Developed by Hakan</p>", unsafe_allow_html=True)
