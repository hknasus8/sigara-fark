import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import requests
import difflib

# Sayfa yapılandırması
st.set_page_config(
    page_title="Sigara Standı Akıllı Denetim Sistemi",
    page_icon="🚬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tarayıcının Google Translate açmasını engellemek için bileşen enjeksiyonu
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

# Sağ üstteki Share, GitHub ve diğer araç çubuğu elemanlarını gizleyen CSS
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

# Görsel boyutlandırma optimizasyon fonksiyonu
def resmi_boyutlandir(img, max_genislik=1000):
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_genislik:
        oran = max_genislik / float(w)
        yeni_yukseklik = int(h * oran)
        return cv2.resize(img, (max_genislik, yeni_yukseklik), interpolation=cv2.INTER_AREA)
    return img

# --- ŞİFRE KONTROLÜ VE EKRAN MANTIĞI ---
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
    st.markdown("Devam etmek için lütfen giriş şifresini girin.")
    
    sifre_input = st.text_input("Şifre", type="password")
    if st.button("Giriş Yap", type="primary"):
        # Güvenli şifre kontrolü (st.secrets kullanımı)
        app_pass = st.secrets.get("app_password", "qwert123")
        if sifre_input == app_pass:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre! Lütfen tekrar deneyin.")
    st.stop()
# ---------------------

st.sidebar.markdown("---")
st.sidebar.header("Uygulama Ayarları")
min_area_val = st.sidebar.slider("Minimum Eksik Boyutu (Hassasiyet)", 50, 2000, 200, step=50)

# Yandex Disk 'BAYİ' Klasörünün Public Linki
YANDEX_ROOT_PUBLIC_KEY = "https://disk.yandex.com.tr/d/JXJNYBDAk6fePw"

# Excel dosyasından bayileri okuma (openpyxl gereksinimi kontrolü)
excel_dosya_adi = "bayiler.xlsx"
bayi_listesi = []

if os.path.exists(excel_dosya_adi):
    try:
        df_bayiler = pd.read_excel(excel_dosya_adi, sheet_name="DATA")
        if "UNVAN" in df_bayiler.columns:
            bayi_listesi = df_bayiler["UNVAN"].dropna().astype(str).tolist()
        else:
            bayi_listesi = df_bayiler.iloc[:, 0].dropna().astype(str).tolist()
    except Exception as e:
        st.error(f"Excel okunurken hata oluştu (openpyxl kurulu olduğundan emin olun): {e}")

if not bayi_listesi:
    bayi_listesi = ["Excel dosyasından unvanlar okunamadı"]

# ANA EKRAN - BAŞLIK VE SAĞ ÜST ÇIKIŞ BUTONU
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

st.subheader("1. Denetlenecek Bayiyi Seçin")
secilen_bayi = st.selectbox("Bayi Seçimi", bayi_listesi, label_visibility="collapsed")
st.markdown(f"**Seçilen Bayi:** `{secilen_bayi}`")
st.markdown("---")

# Yandex Disk'ten difflib ile en yakın eşleşmeyi bulan ve timeout korumalı fonksiyon
@st.cache_data(ttl=600, show_spinner=False)
def yandex_bayi_gorseli_getir_cached(public_key, bayi_adi):
    try:
        api_url = f"https://cloud-api.yandex.net:443/v1/disk/public/resources?public_key={public_key}&limit=2000"
        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return None, f"Yandex API Hatası: HTTP {resp.status_code}"
        
        data = resp.json()
        items = data.get("_embedded", {}).get("items", [])
        
        hedef_aranan = bayi_adi.strip().lower()
        en_iyi_eslesme_path = None
        en_yuksek_benzerlik = 0.0
        
        # Kesin substring yerine difflib ile en yakın klasör adını bulma
        for item in items:
            if item.get("type") == "dir":
                item_adi = item.get("name", "").strip().lower()
                oran = difflib.SequenceMatcher(None, hedef_aranan, item_adi).ratio()
                if oran > en_yuksek_benzerlik and oran > 0.4:  # %40 ve üzeri benzerlik eşiği
                    en_yuksek_benzerlik = oran
                    en_iyi_eslesme_path = item.get("path")
        
        if not en_iyi_eslesme_path:
            return None, "Yandex'te eşleşen klasör bulunamadı."
            
        sub_api_url = f"https://cloud-api.yandex.net:443/v1/disk/public/resources?public_key={public_key}&path={en_iyi_eslesme_path}&limit=100"
        sub_resp = requests.get(sub_api_url, timeout=10)
        if sub_resp.status_code != 200:
            return None, "Klasör içeriği okunamadı."
            
        sub_items = sub_resp.json().get("_embedded", {}).get("items", [])
        
        gorsel_download_url = None
        for sub_item in sub_items:
            if sub_item.get("type") == "file":
                file_name = sub_item.get("name", "").lower()
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    gorsel_download_url = sub_item.get("file")
                    break
                    
        if gorsel_download_url:
            img_resp = requests.get(gorsel_download_url, timeout=15)
            if img_resp.status_code == 200:
                image_bytes = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                return resmi_boyutlandir(img), None
                
        return None, "Klasör içerisinde uygun görsel (jpg/png) bulunamadı."
    except requests.exceptions.Timeout:
        return None, "Yandex sunucusuna bağlanırken zaman aşımı (timeout) oluştu."
    except Exception as e:
        return None, f"Yandex bağlantı hatası: {e}"

# Referans görseli Yandex Disk'ten çekme
ref_img = None
hata_mesaji = None
with st.spinner(f"'{secilen_bayi}' için Yandex Disk'te arama yapılıyor..."):
    ref_img, hata_mesaji = yandex_bayi_gorseli_getir_cached(YANDEX_ROOT_PUBLIC_KEY, secilen_bayi)

# Görseller: Referans Görsel (Yandex) ve Sahadan Gelen (Manuel Yükleme)
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.subheader("2. Referans (İdeal) Görsel")
    if ref_img is not None:
        st.success(f"✅ '{secilen_bayi}' Yandex'ten Yüklendi")
        st.image(ref_img, channels="BGR", use_container_width=True)
    else:
        st.warning(f"⚠️ '{secilen_bayi}' için görsel yüklenemedi. Nedeni: {hata_mesaji}")

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

# Kullanıcının standdaki toplam ürün sayısını elle girebileceği alan
st.subheader("4. Stand Kapasite Ayarı")
ideal_urun_sayisi = st.number_input(
    "Standda Bulunması Gereken Toplam Ürün (Slot) Sayısı",
    min_value=1,
    value=50,
    step=1,
    help="Bu sayı, tespit edilen eksiklere göre raf uygunluk yüzdesinin hesaplanmasında kullanılacaktır."
)

st.markdown("---")

# Session state tanımlamaları
if "result_img" not in st.session_state:
    st.session_state.result_img = None
if "eksik_sayisi" not in st.session_state:
    st.session_state.eksik_sayisi = 0
if "raf_yuzdesi" not in st.session_state:
    st.session_state.raf_yuzdesi = 100.0
if "analiz_yapildi" not in st.session_state:
    st.session_state.analiz_yapildi = False

if ref_img is not None and curr_file is not None and curr_img is not None:
    if st.button("Farkı Analiz Et ve Eksikleri Bul", type="primary"):
        with st.spinner("Gelişmiş açı ve eksik analizi yapılıyor..."):
            
            if ref_img.shape[:2] != curr_img.shape[:2]:
                curr_img = cv2.resize(curr_img, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_AREA)

            gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

            gray_ref = cv2.GaussianBlur(gray_ref, (11, 11), 0)
            gray_curr = cv2.GaussianBlur(gray_curr, (11, 11), 0)

            diff = cv2.absdiff(gray_ref, gray_curr)
            _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

            kernel = np.ones((7, 7), np.uint8)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            img_h, img_w = curr_img.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area_val:
                    x, y, w, h = cv2.boundingRect(contour)
                    if (img_w * 0.02 < x < img_w * 0.98) and (img_h * 0.05 < y < img_h * 0.98):
                        boxes.append([x, y, x + w, y + h])

            def non_max_suppression(boxes, overlapThresh=0.2):
                if len(boxes) == 0:
                    return []
                boxes = np.array(boxes)
                pick = []
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]
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
            
            # Sıfıra bölünme riski korumasıyla yüzdelik hesaplama
            guvenli_toplam_slot = max(1, ideal_urun_sayisi)
            hesaplanan_yuzde = max(0.0, 100.0 - ((eksik_sayisi / guvenli_toplam_slot) * 100.0))

            for idx, (startX, startY, endX, endY) in enumerate(filtered_boxes, 1):
                cv2.rectangle(result_img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(result_img, f"#{idx}", (startX + 3, startY + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.rectangle(result_img, (0, 0), (img_w, 100), (0, 0, 0), -1)
            cv2.putText(result_img, f"Bayi: {secilen_bayi}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_img, f"Eksik Alan: {eksik_sayisi} | Raf Uygunluk: %{hesaplanan_yuzde:.1f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if eksik_sayisi > 0 else (0, 255, 0), 2, cv2.LINE_AA)

            st.session_state.result_img = result_img
            st.session_state.eksik_sayisi = eksik_sayisi
            st.session_state.raf_yuzdesi = hesaplanan_yuzde
            st.session_state.analiz_yapildi = True

    if st.session_state.analiz_yapildi and st.session_state.result_img is not None:
        st.subheader("Tespit Edilen Eksikler ve Raf Doğruluk Raporu")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="📊 Hesaplanan Raf Doğruluk Oranı", value=f"%{st.session_state.raf_yuzdesi:.1f}")
        with col_m2:
            st.metric(label="⚠️ Tespit Edilen Eksik/Boşluk Alan", value=f"{st.session_state.eksik_sayisi} Adet")
        
        sonuc_gorsel_genisligi = st.slider("🔍 Sonuç Görseli Boyutunu Ayarla (Piksel)", 300, 2000, 800, step=100, key="dinamik_boyut")
        
        st.image(st.session_state.result_img, channels="BGR", width=sonuc_gorsel_genisligi)

        success, encoded_image = cv2.imencode(".jpg", st.session_state.result_img)
        if success:
            st.download_button(
                label="📥 Sonuç Fotoğrafını İndir",
                data=encoded_image.tobytes(),
                file_name=f"{secilen_bayi.replace(' ', '_')}_analiz_sonucu.jpg",
                mime="image/jpeg"
            )

        if st.session_state.eksik_sayisi > 0:
            st.error(f"{secilen_bayi} denetimi tamamlandı: Toplam {st.session_state.eksik_sayisi} eksik alan bulundu. Girilen kapasiteye göre raf uygunluk seviyesi %{st.session_state.raf_yuzdesi:.1f}.")
        else:
            st.success(f"{secilen_bayi} denetimi tamamlandı: Raf düzeni kusursuz (%100).")
else:
    st.info("ℹ️ Analiz yapabilmek için lütfen yukarıdan bayinizi seçin ve sağdaki alandan **Sahadan Gelen Fotoğrafı** yükleyin.")

st.markdown("<br><p style='text-align: center; color: gray;'>Developed by Hakan</p>", unsafe_allow_html=True)
